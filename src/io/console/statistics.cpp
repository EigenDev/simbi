#include "io/console/statistics.hpp"

#include "io/display/renderer.hpp"
#include "io/display/terminal.hpp"

#if GPU_ENABLED
#include "xpu/vendors/cuda/device_queries.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if GPU_ENABLED
#include "build_config.hpp"
simbi::real gpu_theoretical_bw = 1.0;
#endif

namespace simbi {
    namespace statistics {

        // structure to hold cpu information
        struct CPUInfo
        {
            std::string  model_name;
            std::int64_t num_cores;
            std::int64_t num_threads;
            double       frequency_mhz;
            size_t       l1_cache_size;
            size_t       l2_cache_size;
            size_t       l3_cache_size;

            // get current cpu information
            static CPUInfo gather()
            {
                CPUInfo info{};

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
                    std::string path = "/sys/devices/system/cpu/cpu0/cache/index" +
                                       std::to_string(level) + "/size";
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
                    else if (!size_str.empty() &&
                             (size_str.back() == 'M' || size_str.back() == 'm')) {
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
        };

        // structure to hold os information
        struct OSInfo
        {
            std::string name;
            std::string version;

            static OSInfo gather()
            {
                OSInfo info{};

#if defined(PLATFORM_WINDOWS)
                info.name = "Windows";
                OSVERSIONINFOEXA osvi;
                ZeroMemory(&osvi, sizeof(OSVERSIONINFOEXA));
                osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXA);

                // note: getversiona is deprecated, but simple for demonstration
                if (GetVersionExA(reinterpret_cast<OSVERSIONINFOA*>(&osvi))) {
                    info.version = std::to_string(osvi.dwMajorVersion) + "." +
                                   std::to_string(osvi.dwMinorVersion);
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
        };

        // display system information using PrettyTable
        void display_system_info()
        {
#if GPU_ENABLED
            using namespace xpu::vendors::cuda;
#endif
            using namespace display;

            const int   width = terminal_t::width();
            renderer_t  renderer;
            box_chars_t box =
                terminal_t::supports_unicode() ? box_chars_t::modern() : box_chars_t::simple();

            // gather system info
            CPUInfo     cpu_info  = CPUInfo::gather();
            OSInfo      os_info   = OSInfo::gather();
            MemoryStats mem_stats = MemoryStats::current();

            // build single unified table
            std::vector<std::string>              headers = {"Category", "Property", "Value"};
            std::vector<std::vector<std::string>> rows;

            // cpu info
            rows.push_back({"CPU", "Model", cpu_info.model_name});
            rows.push_back({"", "Cores", std::to_string(cpu_info.num_cores)});
            rows.push_back({"", "Threads", std::to_string(cpu_info.num_threads)});

            if (cpu_info.frequency_mhz > 0) {
                std::ostringstream freq_str;
                freq_str
                    << (cpu_info.frequency_mhz >= 1000
                            ? std::to_string(static_cast<int>(cpu_info.frequency_mhz / 1000)) +
                                  " GHz"
                            : std::to_string(static_cast<int>(cpu_info.frequency_mhz)) + " MHz");
                rows.push_back({"", "Frequency", freq_str.str()});
            }

            if (cpu_info.l3_cache_size > 0) {
                rows.push_back({"", "L3 Cache", format_bytes(cpu_info.l3_cache_size)});
            }

            // os info
            std::string os_version = os_info.name;
            if (!os_info.version.empty()) {
                os_version += " " + os_info.version;
            }
            rows.push_back({"System", "OS", os_version});

            // memory info
            std::ostringstream ram_usage;
            ram_usage << std::fixed << std::setprecision(1) << mem_stats.percent_used << "%";
            rows.push_back(
                {"Memory",
                 "System RAM",
                 format_bytes(mem_stats.total_physical) + " (" +
                     format_bytes(mem_stats.used_physical) + " used, " + ram_usage.str() + ")"}
            );

            rows.push_back({"", "Process", format_bytes(mem_stats.process_physical)});

            if (mem_stats.total_virtual > 0) {
                double swap_percent =
                    (static_cast<double>(mem_stats.used_virtual) / mem_stats.total_virtual) * 100.0;
                std::ostringstream swap_str;
                swap_str << std::fixed << std::setprecision(1) << swap_percent << "%";
                rows.push_back(
                    {"",
                     "Swap",
                     format_bytes(mem_stats.total_virtual) + " (" +
                         format_bytes(mem_stats.used_virtual) + " used, " + swap_str.str() + ")"}
                );
            }

#if GPU_ENABLED
            auto dev_count = get_device_count();
            if (dev_count > 0) {
                auto props = get_properties(0);
                rows.push_back({"GPU", "Device", props.name});
                rows.push_back(
                    {"",
                     "Compute",
                     std::to_string(props.compute_capability_major) + "." +
                         std::to_string(props.compute_capability_minor)}
                );
                rows.push_back({"", "Memory", format_bytes(props.total_memory)});

                int mem_clock_rate = 0;
                cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, 0);
                std::ostringstream bandwidth;
                bandwidth << std::fixed << std::setprecision(1)
                          << (2.0 * mem_clock_rate * (props.memory_bus_width_bits / 8) / 1.0e6)
                          << " GB/s";
                rows.push_back({"", "Bandwidth", bandwidth.str()});
            }
#endif

            // render single table
            std::cout << color::title() << "\n";
            renderer.render_title(std::cout, "SYSTEM INFORMATION", width);
            std::cout << color::reset();

            renderer.calculate_layout(headers, rows[0], width);
            renderer.render_row(std::cout, headers, true);
            renderer.render_separator(std::cout);
            for (const auto& row : rows) {
                renderer.render_row(std::cout, row, false);
            }
            renderer.render_border_bottom(std::cout);

            // separator before simulation
            std::cout << "\n" << color::border();
            for (int ii = 0; ii < width; ++ii) {
                std::cout << box.horizontal;
            }
            // breathing room before dynamic table outpout
            std::cout << std::string(40, '\n');
            std::cout << color::reset() << "\n\n";
        }

    } // namespace statistics
} // namespace simbi
