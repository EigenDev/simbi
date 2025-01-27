#include "util/tools/helpers.hpp"
#include "io/exceptions.hpp"
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

        std::string getColorCode(Color color)
        {
            switch (color) {
                case Color::RED: return "\033[31m";
                case Color::GREEN: return "\033[32m";
                case Color::YELLOW: return "\033[33m";
                case Color::BLUE: return "\033[34m";
                case Color::MAGENTA: return "\033[35m";
                case Color::CYAN: return "\033[36m";
                case Color::WHITE: return "\033[37m";
                case Color::LIGHT_BLUE: return "\033[0;94m";
                case Color::LIGHT_CYAN: return "\033[0;96m";
                case Color::LIGHT_GREEN: return "\033[0;92m";
                case Color::LIGHT_GREY: return "\033[0;37m";
                case Color::LIGHT_MAGENTA: return "\033[0;95m";
                case Color::LIGHT_RED: return "\033[0;91m";
                case Color::LIGHT_YELLOW: return "\033[0;93m";
                case Color::BLACK: return "\033[0;30m";
                case Color::DARK_GREY: return "\033[0;90m";
                case Color::BOLD: return "\033[1m";
                default: return "\033[0m";
            }
        }

        void catch_signals()
        {
            const static auto signal_handler = [](int sig) {
                killsig_received = true;
            };
            std::signal(SIGTERM, signal_handler);
            std::signal(SIGINT, signal_handler);
            std::signal(SIGABRT, signal_handler);
            std::signal(SIGSEGV, signal_handler);
            std::signal(SIGQUIT, signal_handler);
            if (killsig_received) {
                killsig_received = false;
                throw exception::InterruptException(1);
            }
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
                ;
                gpu_theoretical_bw = 2.0 * props.memoryClockRate *
                                     (props.memoryBusWidth / 8) / 1.0e6;
            }
#else
            const auto processor_count = std::thread::hardware_concurrency();
            std::cout << std::string(80, '=') << "\n";
            std::cout << "CPU Compute Thread(s): " << processor_count
                      << std::endl;

#endif
            // Because I need all of the previous text off of the screen before
            // the simulation starts and the table gets generated, we insert
            // about 40 new to effectively scroll the screen up
            std::cout << std::string(40, '\n');
        }

        /**
         * @brief calculate the bracketing function described in Kastaun,
         * Kalinani, & Colfi (2021)
         *
         * @param mu minimization variable
         * @param beesq rescaled magnetic field squared
         * @param r vector of rescaled momentum
         * @param beedr inner product between rescaled magnetic field & momentum
         * @return Eq. (49)
         */
        DEV real kkc_fmu49(
            const real mu,
            const real beesq,
            const real beedrsq,
            const real r
        )
        {
            // the minimum enthalpy is unity for non-relativistic flows
            constexpr real hlim = 1.0;

            // Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;

            return mu * std::sqrt(hlim * hlim + rbar_sq) - 1.0;
        }

        /**
         * @brief Returns the master function described in Kastaun, Kalinani, &
         * Colfi (2021)
         *
         * @param mu minimization variable
         * @param r vector of rescaled momentum
         * @param rparr parallel component of rescaled momentum vector
         * @param beesq rescaled magnetic field squared
         * @param beedr inner product between rescaled magnetic field & momentum
         * @param qterm rescaled gas energy density
         * @param dterm mass density
         * @param gamma adiabatic index
         * @return Eq. (44)
         */
        DEV real kkc_fmu44(
            const real mu,
            const real r,
            const real rparr,
            const real rperp,
            const real beesq,
            const real beedrsq,
            const real qterm,
            const real dterm,
            const real gamma
        )
        {
            // Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;

            // Equation (39)
            const real qbar =
                qterm - 0.5 * (beesq + mu * mu * x * x * beesq * rperp * rperp);

            // Equation (32) inverted and squared
            const real vsq  = mu * mu * rbar_sq;
            const real gbsq = vsq / std::abs(1.0 - vsq);
            const real g    = std::sqrt(1.0 + gbsq);

            // Equation (41)
            const real rhohat = dterm / g;

            // Equation (42)
            const real epshat = g * (qbar - mu * rbar_sq) + gbsq / (1.0 + g);

            // Equation (43)
            const real phat = (gamma - 1.0) * rhohat * epshat;
            const real ahat = phat / (rhohat * (1.0 + epshat));

            // Equation (46) - (48)
            const real vhatA = (1.0 + ahat) * (1.0 + epshat) / g;
            const real vhatB = (1.0 + ahat) * (1.0 + qbar - mu * rbar_sq);
            const real vhat  = my_max(vhatA, vhatB);

            // Equation (45)
            const real muhat = 1.0 / (vhat + rbar_sq * mu);

            return mu - muhat;
        }

        /**
         * @brief calculate relativistic f & df/dp from Mignone and Bodo (2005)
         * @param gamma adiabatic index
         * @param tau energy density minus rest mass energy
         * @param d lab frame density
         * @param S lab frame momentum density
         * @param p pressure
         */
        DEV std::tuple<real, real>
        newton_fg(real gamma, real tau, real d, real s, real p)
        {
            const auto et  = tau + d + p;
            const auto v2  = s * s / (et * et);
            const auto w   = 1.0 / std::sqrt(1.0 - v2);
            const auto rho = d / w;
            const auto eps =
                (tau + (1.0 - w) * d + (1.0 - w * w) * p) / (d * w);
            const auto c2 = (gamma - 1) * gamma * eps / (1.0 + gamma * eps);
            return std::make_tuple(
                (gamma - 1.0) * rho * eps - p,
                c2 * v2 - 1.0
            );
        }
    }   // namespace helpers
}   // namespace simbi
