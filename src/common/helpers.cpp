#include "common/helpers.hpp"
#include "H5Cpp.h"
#include <iomanip>
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

        std::string format_real(real value)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << value;
            std::string str = oss.str();

            // Replace decimal point with underscore
            std::replace(str.begin(), str.end(), '.', '_');

            // Pad with zeros if necessary
            if (str.find('_') == std::string::npos) {
                str += "_000";
            }
            else {
                while (str.substr(str.find('_') + 1).length() < 3) {
                    str += "0";
                }
            }

            // Ensure the string is at least in the hundreds place
            if (str[0] == '-') {
                while (str.find('_') < 4) {
                    str.insert(1, "0");
                }
            }
            else {
                while (str.find('_') < 3) {
                    str.insert(0, "0");
                }
            }

            // Insert underscores for thousands, millions, etc.
            int insert_position = str.find('_') - 3;
            while (insert_position > 0) {
                str.insert(insert_position, "_");
                insert_position -= 3;
            }

            return str;
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
                      << std::endl
                      << std::endl
                      << std::endl
                      << std::endl
                      << std::endl
                      << std::endl;

#endif
        }

        void display_message(const std::string& full_filename)
        {
            // Save cursor position
            std::cout << "\033[s";

            // Move to a dedicated line (e.g., the bottom of the screen)
            std::cout << "\033[999B";   // Move cursor to bottom of the screen
            std::cout << "\033[3A";     // Move cursor up three lines]"
            // std::cout << "\033[2K";   // Clear the line

            // Display the message
            std::cout << "[Writing File: " << full_filename << "]" << std::endl;

            // Clear the message
            // std::cout << "\033[2K";   // Clear the line

            // Restore cursor position
            std::cout << "\033[u";
        }
    }   // namespace helpers
}   // namespace simbi
