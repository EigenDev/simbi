#ifndef ISOTHERMAL_HPP
#define ISOTHERMAL_HPP

#include "config.hpp"
#include <cmath>   // for std::sqrt

namespace simbi::eos {
    struct isothermal_gas_eos_t {
        double cs_squared;

        DEV auto sound_speed(double /*rho*/, double /*pressure*/) const
        {
            return std::sqrt(cs_squared);
        }

        DEV auto enthalpy(double /*rho*/, double /*pressure*/) const
        {
            return 1.0 + cs_squared;
        }
    };
}   // namespace simbi::eos

#endif
