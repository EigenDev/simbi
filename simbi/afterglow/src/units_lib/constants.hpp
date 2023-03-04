/*
A header file where physical constants will live (in CGS)
*/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include "units.hpp"

namespace constants
{
    constexpr auto c_light  = 2.99792458e10  * units::cm / units::s;
    constexpr auto h_planck = 6.6260755e-27  * units::erg * units::s;
    constexpr auto h_bar    = 1.05457266e-27 * units::erg * units::s;
    constexpr auto newtonG  = 6.67259e-8     * units::cm3 / (units::gram * units::s * units::s);
    constexpr auto e_charge = 4.8032068e-10  * units::statC;
    constexpr auto m_e      = 9.109389e-28   * units::gram;
    constexpr auto m_p      = 1.6726231e-24  * units::gram;
    constexpr auto m_n      = 1.6749286e-24  * units::gram;
    constexpr auto m_H      = 1.6733e-24     * units::gram;
    constexpr auto amu      = 1.6605402e-24  * units::gram;
    constexpr auto nA       = 6.0221367e23;
    constexpr auto kB       = 1.380658e-16   * units::erg / units::kelvin;
    constexpr auto eV2erg   = 1.6021772e-12  * units::erg;
    constexpr auto aRad     = 7.5646e-15     * units::erg / units::cm3 / (units::kelvin * units::kelvin * units::kelvin * units::kelvin);
    constexpr auto sigmaB   = 5.67051e-5     * units::erg / units::cm2 / (units::kelvin * units::kelvin * units::kelvin * units::kelvin);
    constexpr auto alpha    = 7.29735308e-3;
    constexpr auto rydB     = 2.1798741e-11  * units::erg;
    constexpr auto sigmaT   = 6.6524e-25     * units::cm2;
} // namespace constants

#endif