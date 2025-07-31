#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// platform detection
#if defined(_WIN32) || defined(_WIN64)
#define PLATFORM_WINDOWS
#include <pdh.h>
#include <psapi.h>
#include <windows.h>
#pragma comment(lib, "pdh.lib")
#elif defined(__APPLE__) || defined(__MACH__)
#define PLATFORM_MACOS
#include <mach/host_info.h>
#include <mach/kern_return.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_time.h>
#include <mach/mach_types.h>
#include <mach/message.h>
#include <mach/task.h>
#include <mach/task_info.h>
#include <mach/vm_page_size.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__) || defined(__linux) || defined(linux)
#define PLATFORM_LINUX
#include <fstream>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace simbi {
    namespace statistics {

        // formatter for memory sizes (bytes -> KB, MB, GB, etc.)
        inline std::string
        format_bytes(size_t bytes, std::int64_t precision = 2)
        {
            constexpr double kb = 1024.0;
            constexpr double mb = kb * kb;
            constexpr double gb = mb * kb;
            constexpr double tb = gb * kb;

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision);

            if (bytes >= static_cast<size_t>(tb)) {
                oss << bytes / tb << " TB";
            }
            else if (bytes >= static_cast<size_t>(gb)) {
                oss << bytes / gb << " GB";
            }
            else if (bytes >= static_cast<size_t>(mb)) {
                oss << bytes / mb << " MB";
            }
            else if (bytes >= static_cast<size_t>(kb)) {
                oss << bytes / kb << " KB";
            }
            else {
                oss << bytes << " bytes";
            }

            return oss.str();
        }

        // structure to hold system memory information
        struct MemoryStats {
            size_t total_physical;       // total physical memory
            size_t available_physical;   // available physical memory
            size_t used_physical;        // used physical memory
            double percent_used;         // percentage of physical memory used

            size_t total_virtual;       // total virtual memory (swap)
            size_t available_virtual;   // available virtual memory
            size_t used_virtual;        // used virtual memory

            size_t
                process_physical;     // physical memory used by current process
            size_t process_virtual;   // virtual memory used by current process

            // get current memory statistics
            static MemoryStats current()
            {
                MemoryStats stats{};

#if defined(PLATFORM_WINDOWS)
                // windows implementation
                MEMORYSTATUSEX mem_info;
                mem_info.dwLength = sizeof(MEMORYSTATUSEX);
                GlobalMemoryStatusEx(&mem_info);

                stats.total_physical     = mem_info.ullTotalPhys;
                stats.available_physical = mem_info.ullAvailPhys;
                stats.used_physical =
                    stats.total_physical - stats.available_physical;
                stats.percent_used = mem_info.dwMemoryLoad;

                stats.total_virtual     = mem_info.ullTotalPageFile;
                stats.available_virtual = mem_info.ullAvailPageFile;
                stats.used_virtual =
                    stats.total_virtual - stats.available_virtual;

                // process memory info
                PROCESS_MEMORY_COUNTERS_EX pmc;
                if (GetProcessMemoryInfo(
                        GetCurrentProcess(),
                        (PROCESS_MEMORY_COUNTERS*) &pmc,
                        sizeof(pmc)
                    )) {
                    stats.process_physical = pmc.WorkingSetSize;
                    stats.process_virtual  = pmc.PrivateUsage;
                }

#elif defined(PLATFORM_MACOS)
                // macos implementation
                // system memory
                int64_t total_mem = 0;
                size_t len        = sizeof(total_mem);
                if (sysctlbyname("hw.memsize", &total_mem, &len, NULL, 0) ==
                    0) {
                    stats.total_physical = total_mem;
                }

                vm_statistics64_data_t vm_stats;
                mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
                if (host_statistics64(
                        mach_host_self(),
                        HOST_VM_INFO64,
                        (host_info64_t) &vm_stats,
                        &count
                    ) == KERN_SUCCESS) {
                    stats.available_physical =
                        vm_stats.free_count * vm_page_size;
                    stats.used_physical =
                        stats.total_physical - stats.available_physical;
                    stats.percent_used =
                        (static_cast<double>(stats.used_physical) /
                         stats.total_physical) *
                        100.0;
                }

                // virtual memory (swap)
                xsw_usage swap_usage;
                len = sizeof(swap_usage);
                if (sysctlbyname("vm.swapusage", &swap_usage, &len, NULL, 0) ==
                    0) {
                    stats.total_virtual     = swap_usage.xsu_total;
                    stats.used_virtual      = swap_usage.xsu_used;
                    stats.available_virtual = swap_usage.xsu_avail;
                }

                // process memory
                task_basic_info_data_t task_info_data;
                mach_msg_type_number_t info_count = TASK_BASIC_INFO_COUNT;
                if (task_info(
                        mach_task_self(),
                        TASK_BASIC_INFO,
                        (task_info_t) &task_info_data,
                        &info_count
                    ) == KERN_SUCCESS) {
                    stats.process_physical = task_info_data.resident_size;
                    stats.process_virtual  = task_info_data.virtual_size;
                }

#elif defined(PLATFORM_LINUX)
                // linux implementation
                struct sysinfo sys_info;
                if (sysinfo(&sys_info) == 0) {
                    stats.total_physical =
                        sys_info.totalram * sys_info.mem_unit;
                    stats.available_physical =
                        sys_info.freeram * sys_info.mem_unit;
                    stats.used_physical =
                        stats.total_physical - stats.available_physical;
                    stats.percent_used =
                        (static_cast<double>(stats.used_physical) /
                         stats.total_physical) *
                        100.0;

                    stats.total_virtual =
                        sys_info.totalswap * sys_info.mem_unit;
                    stats.available_virtual =
                        sys_info.freeswap * sys_info.mem_unit;
                    stats.used_virtual =
                        stats.total_virtual - stats.available_virtual;
                }

                // process memory
                std::ifstream status_file("/proc/self/status");
                std::string line;
                while (std::getline(status_file, line)) {
                    if (line.find("VmRSS:") != std::string::npos) {
                        stats.process_physical =
                            std::stoull(
                                line.substr(line.find_first_of("0123456789"))
                            ) *
                            1024;
                    }
                    else if (line.find("VmSize:") != std::string::npos) {
                        stats.process_virtual =
                            std::stoull(
                                line.substr(line.find_first_of("0123456789"))
                            ) *
                            1024;
                    }
                }
#endif

                return stats;
            }
        };

        // display system information using PrettyTable
        void display_system_info();

    }   // namespace statistics
}   // namespace simbi

#endif   // STATISTICS_HPP
