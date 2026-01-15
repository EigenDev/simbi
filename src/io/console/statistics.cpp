#include "io/console/statistics.hpp"

#if GPU_ENABLED
#include "build_config.hpp"
simbi::real gpu_theoretical_bw = 1.0;
#endif

#include <cstdint>
#include <string>
#include <thread>

namespace simbi {
    namespace statistics {

        cpu_info_t cpu_info_t::gather()
        {
            cpu_info_t info{};

            // set thread count
            info.num_threads = std::thread::hardware_concurrency();

#if defined(PLATFORM_WINDOWS)
            // windows implementation for cpu model and frequency
            HKEY hKey;
            if (RegOpenKeyExA(
                    HKEY_LOCAL_MACHINE,
                    "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                    0,
                    KEY_READ,
                    &hKey
                ) == ERROR_SUCCESS) {
                char  value[1024];
                DWORD value_size = sizeof(value);

                // get cpu model name
                if (RegQueryValueExA(
                        hKey,
                        "ProcessorNameString",
                        NULL,
                        NULL,
                        (LPBYTE) value,
                        &value_size
                    ) == ERROR_SUCCESS) {
                    info.model_name = value;
                }

                // get cpu frequency
                DWORD mhz;
                DWORD data_size = sizeof(mhz);
                if (RegQueryValueExA(hKey, "~MHz", NULL, NULL, (LPBYTE) &mhz, &data_size) ==
                    ERROR_SUCCESS) {
                    info.frequency_mhz = static_cast<double>(mhz);
                }

                RegCloseKey(hKey);
            }

            // determine physical core count
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            info.num_cores = sysInfo.dwNumberOfProcessors;

            // try to get cache information
            DWORD buffer_size = 0;
            GetLogicalProcessorInformation(0, &buffer_size);
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                    buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                );
                if (GetLogicalProcessorInformation(&buffer[0], &buffer_size)) {
                    for (const auto& i : buffer) {
                        if (i.Relationship == RelationCache) {
                            CACHE_DESCRIPTOR Cache = i.Cache;
                            if (Cache.Level == 1) {
                                info.l1_cache_size = Cache.Size;
                            }
                            else if (Cache.Level == 2) {
                                info.l2_cache_size = Cache.Size;
                            }
                            else if (Cache.Level == 3) {
                                info.l3_cache_size = Cache.Size;
                            }
                        }
                    }
                }
            }

#elif defined(PLATFORM_MACOS)
            // macos implementation
            char   buffer[1024];
            size_t size = sizeof(buffer);

            // get cpu model name
            if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, NULL, 0) == 0) {
                info.model_name = buffer;
            }

            // get cpu frequency
            uint64_t freq = 0;
            size          = sizeof(freq);
            if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) == 0) {
                info.frequency_mhz = static_cast<double>(freq) / 1000000.0;
            }

            // get physical core count
            std::int64_t core_count = 0;
            size                    = sizeof(core_count);
            if (sysctlbyname("hw.physicalcpu", &core_count, &size, NULL, 0) == 0) {
                info.num_cores = core_count;
            }

            // get cache sizes
            uint64_t cache_size = 0;
            size                = sizeof(cache_size);
            if (sysctlbyname("hw.l1dcachesize", &cache_size, &size, NULL, 0) == 0) {
                info.l1_cache_size = cache_size;
            }

            size = sizeof(cache_size);
            if (sysctlbyname("hw.l2cachesize", &cache_size, &size, NULL, 0) == 0) {
                info.l2_cache_size = cache_size;
            }

            size = sizeof(cache_size);
            if (sysctlbyname("hw.l3cachesize", &cache_size, &size, NULL, 0) == 0) {
                info.l3_cache_size = cache_size;
            }

#elif defined(PLATFORM_LINUX)
            // linux implementation

            // read cpu model from /proc/cpuinfo
            std::ifstream cpuinfo("/proc/cpuinfo");
            std::string   line;
            std::int64_t  core_count = 0;
            std::string   model_name;
            double        cpu_freq = 0.0;

            while (std::getline(cpuinfo, line)) {
                // get cpu model
                if (line.find("model name") != std::string::npos && model_name.empty()) {
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        model_name = line.substr(pos + 2);
                    }
                }

                // get cpu frequency
                if (line.find("cpu MHz") != std::string::npos && cpu_freq == 0.0) {
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        try {
                            cpu_freq = std::stod(line.substr(pos + 2));
                        }
                        catch (...) {
                            cpu_freq = 0.0;
                        }
                    }
                }

                // count unique physical cores (not hyperthreaded ones)
                if (line.find("physical id") != std::string::npos) {
                    std::int64_t physical_id = 0;
                    size_t       pos         = line.find(':');
                    if (pos != std::string::npos) {
                        try {
                            physical_id = std::stoi(line.substr(pos + 2));
                            core_count  = std::max(core_count, physical_id + 1);
                        }
                        catch (...) {
                        }
                    }
                }
            }

            info.model_name    = model_name;
            info.frequency_mhz = cpu_freq;
            info.num_cores =
                core_count > 0 ? core_count : info.num_threads / 2; // fallback estimate

            // try to get cache information from sysfs
            auto read_cache_size = [](std::int64_t level) -> size_t {
                std::string path =
                    "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(level) + "/size";
                std::ifstream cache_file(path);
                if (!cache_file) {
                    return 0;
                }

                std::string size_str;
                cache_file >> size_str;

                size_t size       = 0;
                size_t multiplier = 1;

                // parse sizes like "32K" or "1M"
                if (!size_str.empty() && (size_str.back() == 'K' || size_str.back() == 'k')) {
                    multiplier = 1024;
                    size_str.pop_back();
                }
                else if (!size_str.empty() && (size_str.back() == 'M' || size_str.back() == 'm')) {
                    multiplier = 1024 * 1024;
                    size_str.pop_back();
                }

                try {
                    size = std::stoull(size_str) * multiplier;
                }
                catch (...) {
                    size = 0;
                }

                return size;
            };

            // try to read l1, l2, and l3 cache sizes
            info.l1_cache_size = read_cache_size(0); // l1 data cache
            info.l2_cache_size = read_cache_size(2); // l2 cache
            info.l3_cache_size = read_cache_size(3); // l3 cache
#endif

            return info;
        }
    } // namespace statistics

    statistics::os_info_t statistics::os_info_t::gather()
    {
        statistics::os_info_t info{};

#if defined(PLATFORM_WINDOWS)
        info.name = "Windows";
        OSVERSIONINFOEXA osvi;
        ZeroMemory(&osvi, sizeof(OSVERSIONINFOEXA));
        osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXA);

        // note: getversiona is deprecated, but simple for demonstration
        if (GetVersionExA(reinterpret_cast<OSVERSIONINFOA*>(&osvi))) {
            info.version =
                std::to_string(osvi.dwMajorVersion) + "." + std::to_string(osvi.dwMinorVersion);
        }

#elif defined(PLATFORM_MACOS)
        info.name = "macOS";
        char   str[256];
        size_t size = sizeof(str);
        if (sysctlbyname("kern.osrelease", str, &size, NULL, 0) == 0) {
            info.version = str;
        }

#elif defined(PLATFORM_LINUX)
        info.name = "Linux";
        // try to get distribution info from /etc/os-release
        std::ifstream os_release("/etc/os-release");
        std::string   line;
        while (std::getline(os_release, line)) {
            if (line.find("NAME=") == 0) {
                std::string name = line.substr(5);
                // remove quotes if present
                if (name.front() == '"' && name.back() == '"') {
                    name = name.substr(1, name.size() - 2);
                }
                info.name = name;
            }
            else if (line.find("VERSION=") == 0) {
                std::string version = line.substr(8);
                // remove quotes if present
                if (version.front() == '"' && version.back() == '"') {
                    version = version.substr(1, version.size() - 2);
                }
                info.version = version;
            }
        }
#endif

        return info;
    }

} // namespace simbi
