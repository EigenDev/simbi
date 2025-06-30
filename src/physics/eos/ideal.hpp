#ifndef SIMBI_EOS_IDEAL_HPP
#define SIMBI_EOS_IDEAL_HPP

#include "config.hpp"
#include "core/utility/enums.hpp"
#include <cmath>   // for std::sqrt
#include <iostream>

namespace simbi::eos {
    template <Regime R>
    struct ideal_gas_eos_t {
        double gamma;

        DEV auto sound_speed(double rho, double pressure) const
        {
            return std::sqrt(
                gamma * pressure / (rho * enthalpy(rho, pressure))
            );
        }

        DEV auto enthalpy(double rho, double pressure) const
        {
            if constexpr (!(R == Regime::SRHD && R == Regime::RMHD)) {
                return 1.0;
            }
            return 1.0 + gamma * pressure / (rho * (gamma - 1.0));
        }

        DEV auto internal_energy(double rho, double pressure) const
        {
            return pressure / (rho * (gamma - 1.0));
        }
    };
}   // namespace simbi::eos

#endif
