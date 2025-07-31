#ifndef EOS_IDEAL_HPP
#define EOS_IDEAL_HPP

#include "config.hpp"
#include "utility/enums.hpp"

#include <cmath>   // for std::sqrt

namespace simbi::eos {
    template <Regime R>
    struct ideal_gas_eos_t {
        real gamma;

        DEV auto sound_speed(real rho, real pressure) const
        {
            return std::sqrt(
                gamma * pressure / (rho * enthalpy(rho, pressure))
            );
        }

        DEV auto enthalpy(real rho, real pressure) const
        {
            if constexpr (!(R == Regime::SRHD || R == Regime::RMHD)) {
                return 1.0;
            }
            return 1.0 + gamma * pressure / (rho * (gamma - 1.0));
        }

        DEV auto internal_energy(real rho, real pressure) const
        {
            return pressure / (rho * (gamma - 1.0));
        }
    };
}   // namespace simbi::eos

#endif
