#ifndef ISOTHERMAL_HPP
#define ISOTHERMAL_HPP

#include "config.hpp"

#include <cmath>   // for std::sqrt

namespace simbi::eos {
    struct isothermal_gas_eos_t {
        // this is a nominal param. gamma = 1 in isothermal gas
        real gamma;

        DEV auto sound_speed(real rho, real pressure) const
        {
            return std::sqrt(pressure / rho);
        }

        DEV auto enthalpy(real /*rho*/, real /*pressure*/) const
        {
            return 1.0;
            // if constexpr (R == Regime::NEWTONIAN || R == Regime::MHD) {
            //     return 1.0;
            // }
            // 1 + cs^2
            // return 1.0 + pressure / rho;
        }
    };
}   // namespace simbi::eos

#endif
