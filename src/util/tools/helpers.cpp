#include "util/tools/helpers.hpp"
#include "io/exceptions.hpp"
#include <csignal>
#include <iomanip>
#include <thread>

//==================================
//              GPU HELPERS
//==================================
using namespace H5;

namespace simbi {
    namespace helpers {
        // Flag that detects whether program was terminated by external forces
        std::atomic<bool> killsig_received = false;

        std::string get_color_code(Color color)
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
            const static auto signal_handler = [](int) {
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
            const real rsbsq,
            const real r
        )
        {
            // the minimum enthalpy is unity for non-relativistic flows
            constexpr real hlim = 1.0;

            // Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = r * r * x * x + mu * x * (1.0 + x) * rsbsq;

            return mu * std::sqrt(hlim * hlim + rbar_sq) - 1.0;
        }

        /**
         * @brief Returns the master function described in Kastaun, Kalinani, &
         * Colfi (2021)
         *
         * @param mu minimization variable
         * @param r vector of rescaled momentum
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
            const real rperp_sq,
            const real beesq,
            const real rdbsq,
            const real qterm,
            const real dterm,
            const real gamma
        )
        {
            constexpr real h0 = 1.0;
            // Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = r * r * x * x + mu * x * (1.0 + x) * rdbsq;

            // Equation (39)
            const real qbar =
                qterm - 0.5 * (beesq + mu * mu * x * x * beesq * rperp_sq);

            // Equation (32) & (33)
            const real z_upper = r / h0;
            const real v_limit = z_upper / std::sqrt(1.0 + z_upper * z_upper);
            const real vsq     = my_min(mu * mu * rbar_sq, v_limit * v_limit);
            const real gbsq    = vsq / (1.0 - vsq);

            const real g = std::sqrt(1.0 + gbsq);

            // Equation (41)
            const real rhohat = dterm / g;

            // Equation (42)
            const real eps = g * (qbar - mu * rbar_sq) + gbsq / (1.0 + g);

            // zero-temperature limit for gamma-law EoS
            constexpr auto pfloor = 1.0e-3;
            const real eps_min    = pfloor / (rhohat * (gamma - 1.0));
            const real epshat     = my_max(eps, eps_min);

            // Equation (43)
            const real phat = (gamma - 1.0) * rhohat * epshat;
            const real ahat = phat / (rhohat * (1.0 + epshat));

            // Equation (46) - (48)
            const real nu_hatA = (1.0 + ahat) * (1.0 + epshat) / g;
            const real nu_hatB = (1.0 + ahat) * (1.0 + qbar - mu * rbar_sq);
            const real nu_hat =
                (eps < eps_min) ? nu_hatA : my_max(nu_hatA, nu_hatB);

            // Equation (45)
            const real muhat = 1.0 / (nu_hat + rbar_sq * mu);

            return mu - muhat;
        }

        DEV real find_mu_plus(const real beesq, const real rsbsq, const real r)
        {
            constexpr real h0 = 1.0;

            // If r < h0, we can use 1/h0 as upper bound
            if (r < h0) {
                return 1.0 / h0;   // Since h0 = 1.0, this is just 1.0
            }

            // otherwise, we need to find \mu^\plus where fa(\mu^\plus) = 0
            // here, we'll start with a reasonably large upper bound
            real mu_lower = 0.0;
            real mu_upper = 2.0;   // Start with a guess

            // ensure upper bound gives positive fa
            real f_upper = kkc_fmu49(mu_upper, beesq, rsbsq, r);
            while (f_upper < 0.0) {
                mu_upper *= 2.0;
                f_upper = kkc_fmu49(mu_upper, beesq, rsbsq, r);
            }

            // now we find the root using bisection
            const int max_iter = 50;
            size_type iter     = 0;
            real f_mid, mu_mid = 1.0;
            real f_lower = kkc_fmu49(mu_lower, beesq, rsbsq, r);

            while (iter < max_iter && (mu_upper - mu_lower) > global::epsilon) {
                mu_mid = 0.5 * (mu_lower + mu_upper);
                f_mid  = kkc_fmu49(mu_mid, beesq, rsbsq, r);

                if (std::abs(f_mid) < global::epsilon) {
                    // found a good enough root (!)
                    break;
                }

                if (f_mid * f_lower < 0.0) {
                    mu_upper = mu_mid;
                    f_upper  = f_mid;
                }
                else {
                    mu_lower = mu_mid;
                    f_lower  = f_mid;
                }

                iter++;
            }

            // we add a small safety margin to ensure the root is contained
            return mu_mid * 1.000001;
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

        std::string error_code_to_string(ErrorCode code)
        {
            // handle bit-field error codes
            if (code == ErrorCode::NONE) {
                return "No error";
            }

            std::string result;
            // Check each error code and append the corresponding message
            // to the result string. For each error code after the first one,
            // we add 'or' to the result string.
            if (has_error(code, ErrorCode::NEGATIVE_PRESSURE)) {
                result += "negative pressure or ";
            }
            if (has_error(code, ErrorCode::NON_FINITE_PRESSURE)) {
                result += "non-finite pressure or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_DENSITY)) {
                result += "negative density or ";
            }
            if (has_error(code, ErrorCode::SUPERLUMINAL_VELOCITY)) {
                result += "superluminal velocity or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_ENERGY)) {
                result += "negative energy or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_ENTROPY)) {
                result += "negative entropy or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_MASS)) {
                result += "negative mass or ";
            }
            if (has_error(code, ErrorCode::NON_FINITE_ROOT)) {
                result += "non-finite root or ";
            }
            if (has_error(code, ErrorCode::MAX_ITER)) {
                result += "maximum iterations reached or ";
            }
            if (has_error(code, ErrorCode::UNDEFINED)) {
                result += "undefined error or ";
            }
            if (result.empty()) {
                return "Unknown error";
            }
            result.pop_back();   // Remove the trailing space
            result.pop_back();   // Remove the trailing r
            result.pop_back();   // Remove the trailing o
            result.pop_back();   // Remove the trailing space
            return result;
        }
    }   // namespace helpers
}   // namespace simbi
