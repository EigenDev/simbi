#include "common/helpers.hpp"
#include "H5Cpp.h"
#include <thread>

//==================================
//              GPU HELPERS
//==================================
real gpu_theoretical_bw = 1.0;
using namespace H5;

namespace simbi {
    namespace helpers {
        // Flag that detects whether program was terminated by external forces
        sig_bool killsig_received = false;

        InterruptException::InterruptException(int s) : status(s) {}

        const char* InterruptException::what() const noexcept
        {
            return "Simulation interrupted. Saving last checkpoint...";
        }

        void catch_signals()
        {
            const static auto signal_handler = [](int sig) {
                killsig_received = true;
            };
            std::signal(SIGTERM, signal_handler);
            std::signal(SIGINT, signal_handler);
            if (killsig_received) {
                killsig_received = false;
                throw helpers::InterruptException(1);
            }
        }

        SimulationFailureException::SimulationFailureException() = default;

        const char* SimulationFailureException::what() const noexcept
        {
            // crashed in bold red!
            return "\033[1;31mSimulation Crashed\033[0m";
        }

        //====================================================================================================
        //                                  WRITE DATA TO FILE
        //====================================================================================================
        std::string
        create_step_str(const real current_time, const int max_order_of_mag)
        {
            if (current_time == 0) {
                return "000_000";
            }
            const int current_time_int = std::round(1e3 * current_time);
            const int num_zeros =
                max_order_of_mag - static_cast<int>(std::log10(current_time));
            std::string time_string =
                std::string(num_zeros, '0') + std::to_string(current_time_int);
            separate<3, '_'>(time_string);
            return time_string;
        }

        void anyDisplayProps()
        {
// Adapted from:
// https://stackoverflow.com/questions/5689028/how-to-get-card-specs-programmatically-in-cuda
#if GPU_CODE
            const int kb = 1024;
            const int mb = kb * kb;
            int devCount;
            gpu::api::getDeviceCount(&devCount);
            std::cout << std::string(80, '=') << "\n";
            std::cout << "GPU Device(s): " << std::endl << std::endl;

            for (int i = 0; i < devCount; ++i) {
                devProp_t props;
                gpu::api::getDeviceProperties(&props, i);
                std::cout << "  Device number:   " << i << std::endl;
                std::cout << "  Device name:     " << props.name << ": "
                          << props.major << "." << props.minor << std::endl;
                std::cout << "  Global memory:   " << props.totalGlobalMem / mb
                          << "mb" << std::endl;
                std::cout << "  Shared memory:   "
                          << props.sharedMemPerBlock / kb << "kb" << std::endl;
                std::cout << "  Constant memory: " << props.totalConstMem / kb
                          << "kb" << std::endl;
                std::cout << "  Block registers: " << props.regsPerBlock
                          << std::endl
                          << std::endl;

                std::cout << "  Warp size:         " << props.warpSize
                          << std::endl;
                std::cout << "  Threads per block: " << props.maxThreadsPerBlock
                          << std::endl;
                std::cout << "  Max block dimensions: [ "
                          << props.maxThreadsDim[0] << ", "
                          << props.maxThreadsDim[1] << ", "
                          << props.maxThreadsDim[2] << " ]" << std::endl;
                std::cout << "  Max grid dimensions:  [ "
                          << props.maxGridSize[0] << ", "
                          << props.maxGridSize[1] << ", "
                          << props.maxGridSize[2] << " ]" << std::endl;
                std::cout << "  Memory Clock Rate (KHz): "
                          << props.memoryClockRate << std::endl;
                std::cout << "  Memory Bus Width (bits): "
                          << props.memoryBusWidth << std::endl;
                std::cout << "  Peak Memory Bandwidth (GB/s): "
                          << 2.0 * props.memoryClockRate *
                                 (props.memoryBusWidth / 8) / 1.0e6
                          << std::endl;
                std::cout << std::endl;
                gpu_theoretical_bw = 2.0 * props.memoryClockRate *
                                     (props.memoryBusWidth / 8) / 1.0e6;
            }
#else
            const auto processor_count = std::thread::hardware_concurrency();
            std::cout << std::string(80, '=') << "\n";
            std::cout << "CPU Compute Thread(s): " << processor_count
                      << std::endl;
#endif
        }
    }   // namespace helpers
}   // namespace simbi
