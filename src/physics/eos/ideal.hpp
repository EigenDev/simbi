// =============================================================================
// ideal.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "decorators.hpp"
#include "utility/enums.hpp"

#include <cmath> // for std::sqrt

namespace simbi::eos {
    template <regime_t R>
    struct ideal_gas_eos_t
    {
        real gamma;

        DEV real sound_speed(real rho, real pressure) const
        {
            return std::sqrt(gamma * pressure / (rho * enthalpy(rho, pressure)));
        }

        DEV real enthalpy(real rho, real pressure) const
        {
            if constexpr (!(R == regime_t::SRHD || R == regime_t::RMHD)) {
                return 1.0;
            }
            return 1.0 + gamma * pressure / (rho * (gamma - 1.0));
        }

        DEV real specific_internal_energy(real rho, real pressure) const
        {
            return pressure / (rho * (gamma - 1.0));
        }
    };
} // namespace simbi::eos


