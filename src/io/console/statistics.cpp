#include "io/console/statistics.hpp"
#include "config.hpp"
#include "io/tabulate/table.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/sysctl.h>
#include <thread>
#include <vector>
// we'll need to check if GPU code is enabled
#if GPU_ENABLED
#include "adapter/device_adapter_api.hpp"
real gpu_theoretical_bw = 1.0;
#endif

namespace simbi {
    namespace statistics {

        // structure to hold cpu information
        struct CPUInfo {
            std::string model_name;
            std::int64_t num_cores;
            std::int64_t num_threads;
            double frequency_mhz;
            size_t l1_cache_size;
            size_t l2_cache_size;
            size_t l3_cache_size;

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
                    char value[1024];
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
                    if (RegQueryValueExA(
                            hKey,
                            "~MHz",
                            NULL,
                            NULL,
                            (LPBYTE) &mhz,
                            &data_size
                        ) == ERROR_SUCCESS) {
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
                        buffer_size /
                        sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                    );
                    if (GetLogicalProcessorInformation(
                            &buffer[0],
                            &buffer_size
                        )) {
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
                char buffer[1024];
                size_t size = sizeof(buffer);

                // get cpu model name
                if (sysctlbyname(
                        "machdep.cpu.brand_string",
                        &buffer,
                        &size,
                        NULL,
                        0
                    ) == 0) {
                    info.model_name = buffer;
                }

                // get cpu frequency
                uint64_t freq = 0;
                size          = sizeof(freq);
                if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) ==
                    0) {
                    info.frequency_mhz = static_cast<double>(freq) / 1000000.0;
                }

                // get physical core count
                std::int64_t core_count = 0;
                size                    = sizeof(core_count);
                if (sysctlbyname(
                        "hw.physicalcpu",
                        &core_count,
                        &size,
                        NULL,
                        0
                    ) == 0) {
                    info.num_cores = core_count;
                }

                // get cache sizes
                uint64_t cache_size = 0;
                size                = sizeof(cache_size);
                if (sysctlbyname(
                        "hw.l1dcachesize",
                        &cache_size,
                        &size,
                        NULL,
                        0
                    ) == 0) {
                    info.l1_cache_size = cache_size;
                }

                size = sizeof(cache_size);
                if (sysctlbyname(
                        "hw.l2cachesize",
                        &cache_size,
                        &size,
                        NULL,
                        0
                    ) == 0) {
                    info.l2_cache_size = cache_size;
                }

                size = sizeof(cache_size);
                if (sysctlbyname(
                        "hw.l3cachesize",
                        &cache_size,
                        &size,
                        NULL,
                        0
                    ) == 0) {
                    info.l3_cache_size = cache_size;
                }

#elif defined(PLATFORM_LINUX)
                // linux implementation

                // read cpu model from /proc/cpuinfo
                std::ifstream cpuinfo("/proc/cpuinfo");
                std::string line;
                std::int64_t core_count = 0;
                std::string model_name;
                double cpu_freq = 0.0;

                while (std::getline(cpuinfo, line)) {
                    // get cpu model
                    if (line.find("model name") != std::string::npos &&
                        model_name.empty()) {
                        size_t pos = line.find(':');
                        if (pos != std::string::npos) {
                            model_name = line.substr(pos + 2);
                        }
                    }

                    // get cpu frequency
                    if (line.find("cpu MHz") != std::string::npos &&
                        cpu_freq == 0.0) {
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
                        size_t pos               = line.find(':');
                        if (pos != std::string::npos) {
                            try {
                                physical_id = std::stoi(line.substr(pos + 2));
                                core_count =
                                    std::max(core_count, physical_id + 1);
                            }
                            catch (...) {
                            }
                        }
                    }
                }

                info.model_name    = model_name;
                info.frequency_mhz = cpu_freq;
                info.num_cores     = core_count > 0 ? core_count
                                                    : info.num_threads /
                                                      2;   // fallback estimate

                // try to get cache information from sysfs
                auto read_cache_size = [](std::int64_t level) -> size_t {
                    std::string path =
                        "/sys/devices/system/cpu/cpu0/cache/index" +
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
                    if (!size_str.empty() &&
                        (size_str.back() == 'K' || size_str.back() == 'k')) {
                        multiplier = 1024;
                        size_str.pop_back();
                    }
                    else if (!size_str.empty() && (size_str.back() == 'M' ||
                                                   size_str.back() == 'm')) {
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
                info.l1_cache_size = read_cache_size(0);   // l1 data cache
                info.l2_cache_size = read_cache_size(2);   // l2 cache
                info.l3_cache_size = read_cache_size(3);   // l3 cache
#endif

                return info;
            }
        };

        // structure to hold os information
        struct OSInfo {
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
                char str[256];
                size_t size = sizeof(str);
                if (sysctlbyname("kern.osrelease", str, &size, NULL, 0) == 0) {
                    info.version = str;
                }

#elif defined(PLATFORM_LINUX)
                info.name = "Linux";
                // try to get distribution info from /etc/os-release
                std::ifstream os_release("/etc/os-release");
                std::string line;
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
            std::cout << std::string(104, '=') << "\n";

            // initialize common settings
            constexpr std::int64_t SYSTEM_INFO_TABLE_WIDTH = 104;

            // CPU information table
            {
                auto cpu_table = io::TableFactory::create_system_info_table();
                cpu_table.set_title("CPU & System Information");
                cpu_table.center_table(true);

                // set table header
                std::vector<std::string> header = {"Property", "Value"};
                cpu_table.set_header(header);
                cpu_table.set_column_alignment(0, io::Alignment::Left);
                cpu_table.set_column_alignment(1, io::Alignment::Left);

                // gather CPU info
                CPUInfo cpu_info = CPUInfo::gather();
                OSInfo os_info   = OSInfo::gather();

                // add CPU information
                cpu_table.add_row({"CPU Model", cpu_info.model_name});
                cpu_table.add_row(
                    {"Physical Cores", std::to_string(cpu_info.num_cores)}
                );
                cpu_table.add_row(
                    {"Logical Threads", std::to_string(cpu_info.num_threads)}
                );

                // add CPU frequency if available
                if (cpu_info.frequency_mhz > 0) {
                    std::ostringstream freq_str;
                    freq_str << std::fixed << std::setprecision(2);
                    if (cpu_info.frequency_mhz >= 1000) {
                        freq_str << (cpu_info.frequency_mhz / 1000) << " GHz";
                    }
                    else {
                        freq_str << cpu_info.frequency_mhz << " MHz";
                    }
                    cpu_table.add_row({"CPU Frequency", freq_str.str()});
                }

                // add cache information if available
                if (cpu_info.l1_cache_size > 0) {
                    cpu_table.add_row(
                        {"L1 Cache", format_bytes(cpu_info.l1_cache_size)}
                    );
                }
                if (cpu_info.l2_cache_size > 0) {
                    cpu_table.add_row(
                        {"L2 Cache", format_bytes(cpu_info.l2_cache_size)}
                    );
                }
                if (cpu_info.l3_cache_size > 0) {
                    cpu_table.add_row(
                        {"L3 Cache", format_bytes(cpu_info.l3_cache_size)}
                    );
                }

                // add OS information
                std::string os_version = os_info.name;
                if (!os_info.version.empty()) {
                    os_version += " " + os_info.version;
                }
                cpu_table.add_row({"Operating System", os_version});

                cpu_table.set_minimum_width(SYSTEM_INFO_TABLE_WIDTH);

                // prstd::int64_t the table
                cpu_table.print();
            }

            std::cout << std::endl;

            // Memory information table
            {

                auto memory_table =
                    io::TableFactory::create_system_info_table();
                memory_table.set_title("Memory Information");
                memory_table.center_table(true);

                // set table header
                std::vector<std::string> header =
                    {"Memory Type", "Total", "Used", "Available", "Usage"};
                memory_table.set_header(header);

                // align columns
                for (std::int64_t i = 0; i < 5; i++) {
                    memory_table.set_column_alignment(
                        i,
                        i == 0 ? io::Alignment::Left : io::Alignment::Right
                    );
                }

                // gather memory stats
                MemoryStats mem_stats = MemoryStats::current();

                // format memory usage percentage
                std::ostringstream usage_str;
                usage_str << std::fixed << std::setprecision(1)
                          << mem_stats.percent_used << "%";

                // add system RAM information
                memory_table.add_row(
                    {"System RAM",
                     format_bytes(mem_stats.total_physical),
                     format_bytes(mem_stats.used_physical),
                     format_bytes(mem_stats.available_physical),
                     usage_str.str()}
                );

                // add virtual memory/swap information if available
                if (mem_stats.total_virtual > 0) {
                    double swap_percent = 0.0;
                    if (mem_stats.total_virtual > 0) {
                        swap_percent =
                            (static_cast<double>(mem_stats.used_virtual) /
                             mem_stats.total_virtual) *
                            100.0;
                    }

                    std::ostringstream swap_usage_str;
                    swap_usage_str << std::fixed << std::setprecision(1)
                                   << swap_percent << "%";

                    memory_table.add_row(
                        {"Virtual Memory/Swap",
                         format_bytes(mem_stats.total_virtual),
                         format_bytes(mem_stats.used_virtual),
                         format_bytes(mem_stats.available_virtual),
                         swap_usage_str.str()}
                    );
                }

                // add process memory information
                memory_table.add_row(
                    {"Process Memory",
                     "N/A",
                     format_bytes(mem_stats.process_physical),
                     "N/A",
                     "N/A"}
                );

                memory_table.set_minimum_width(SYSTEM_INFO_TABLE_WIDTH);

                // prstd::int64_t the memory table
                memory_table.print();
            }

#if GPU_ENABLED
            std::cout << std::endl;

            // GPU information table
            {

                auto gpu_table = io::TableFactory::create_system_info_table();
                gpu_table.set_title("GPU Information");

                std::int64_t dev_count;
                gpu::api::get_device_count(&dev_count);

                if (dev_count == 0) {
                    // set table header for no GPU case
                    std::vector<std::string> header = {"Status"};
                    gpu_table.set_header(header);
                    gpu_table.add_row({"No CUDA-capable device detected"});
                }
                else {
                    // set table header for GPU info
                    std::vector<std::string> header = {"Property", "Value"};
                    gpu_table.set_header(header);
                    gpu_table.set_column_alignment(0, io::Alignment::Left);
                    gpu_table.set_column_alignment(1, io::Alignment::Left);

                    // we'll show info for the first GPU (can be extended to
                    // multiple)
                    adapter::device_properties_t<> props;
                    gpu::api::get_device_properties(&props, 0);

                    // add GPU details
                    gpu_table.add_row({"Device Name", props.name});
                    gpu_table.add_row(
                        {"Compute Capability",
                         std::to_string(props.major) + "." +
                             std::to_string(props.minor)}
                    );
                    gpu_table.add_row(
                        {"Global Memory", format_bytes(props.totalGlobalMem)}
                    );

                    std::ostringstream mem_clock;
                    mem_clock << std::fixed << std::setprecision(1)
                              << (props.memoryClockRate / 1000.0) << " MHz";
                    gpu_table.add_row({"Memory Clock", mem_clock.str()});

                    gpu_table.add_row(
                        {"Memory Bus Width",
                         std::to_string(props.memoryBusWidth) + " bits"}
                    );

                    std::ostringstream bandwidth;
                    bandwidth << std::fixed << std::setprecision(1)
                              << (2.0 * props.memoryClockRate *
                                  (props.memoryBusWidth / 8) / 1.0e6)
                              << " GB/s";
                    gpu_table.add_row({"Peak Bandwidth", bandwidth.str()});

                    gpu_table.add_row(
                        {"Shared Memory/Block",
                         format_bytes(props.sharedMemPerBlock)}
                    );
                    gpu_table.add_row(
                        {"Warp Size", std::to_string(props.warpSize)}
                    );
                    gpu_table.add_row(
                        {"Max Threads/Block",
                         std::to_string(props.maxThreadsPerBlock)}
                    );

                    std::ostringstream block_dim;
                    block_dim << "[" << props.maxThreadsDim[0] << ", "
                              << props.maxThreadsDim[1] << ", "
                              << props.maxThreadsDim[2] << "]";
                    gpu_table.add_row(
                        {"Max Block Dimensions", block_dim.str()}
                    );

                    std::ostringstream grid_dim;
                    grid_dim << "[" << props.maxGridSize[0] << ", "
                             << props.maxGridSize[1] << ", "
                             << props.maxGridSize[2] << "]";
                    gpu_table.add_row({"Max Grid Dimensions", grid_dim.str()});
                }

                gpu_table.set_minimum_width(SYSTEM_INFO_TABLE_WIDTH);

                // prstd::int64_t the GPU table
                gpu_table.print();
            }
#endif

            // add space to scroll the screen up before simulation starts
            const auto vspace = global::on_sm ? 42 : 40;
            std::cout << std::string(vspace, '\n');
        }

    }   // namespace statistics
}   // namespace simbi
