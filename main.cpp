#include <iostream>
#include "units.hpp"
#include "constants.hpp"

using std::pow;
using std::sqrt;

int main()
{
    const auto sunInGrams  = 1.989e33 * units::gram;
    const auto sunInKiloG  = sunInGrams.to(units::kg);
    const auto sunInMsun   = sunInKiloG.to(units::mSun);
    
    std::cout << "Sun's mass in grams:"     << sunInGrams << "\n";
    std::cout << "Sun's mass in kilograms:" << sunInKiloG << "\n";
    std::cout << "Sun's mass in SolarMass:" << sunInMsun << "\n";
    std::cout << std::string(80, '=') << "\n";

    const auto someMass  = 1.0 * units::gram;
    const auto dt        = 0.1 * units::s;
    const auto ell       = 1.0 * units::cm;

    const auto v = ell / dt; // velocity in cgs
    const auto a = v   / dt; // acceleration in cgs
    const auto a2 = a * a;   // square of the acceleration 

    const auto vmks = v.to(units::km / units::s);             // velocity in mks
    const auto amks = a.to(units::km / units::s / units::s);  // acceleration in mks
    std::cout << std::string(80, '=') << "\n";
    std::cout << "velocity in cgs: " << v  << "\n";
    std::cout << "acceleration: " << a  << "\n";
    std::cout << "velocity in kms:" << vmks << "\n";
    std::cout << "acceleration in kms:" << amks << "\n";
    std::cout << "Sqrt of squared acceleration using sqrt: " << units::math::sqrt(a2) << "\n";
    std::cout << "Sqrt of squared acceleration using pow: "  <<  units::math::pow<std::ratio<1,2>>(a2) << "\n";
    
    // spectral flux test 
    const auto cgsFlux = 1e-26 * units::erg / units::s / units::cm2 / units::hz;
    const auto spec    = cgsFlux.to(units::mjy);
    std::cout << "flux in mJy: " << spec << "\n";
    // Constants testing
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Physical constants" << "\n";
    std::cout << "speed of light: "               <<  constants::c_light  << "\n";
    std::cout << "Planck constant: "              <<  constants::h_planck << "\n";
    std::cout << "Planck constant over 2Pi: "     <<  constants::h_bar    << "\n";
    std::cout << "Newton's G: "                   <<  constants::newtonG  << "\n";
    std::cout << "Elementary charge: "            <<  constants::e_charge << "\n";
    std::cout << "Mass of electron: "             <<  constants::m_e      << "\n";
    std::cout << "Mass of proton: "               <<  constants::m_p      << "\n";
    std::cout << "Mass of neutron: "              <<  constants::m_n      << "\n";
    std::cout << "Mass of hydrogen: "             <<  constants::m_H      << "\n";
    std::cout << "Atomic mass unit: "             <<  constants::amu      << "\n";
    std::cout << "Avagadro's number: "            <<  constants::nA       << "\n";
    std::cout << "Boltzmann constant: "           <<  constants::kB       << "\n";
    std::cout << "eV to ergs conversion: "        <<  constants::eV2erg   << "\n";
    std::cout << "Radiation density constant a: " <<  constants::aRad     << "\n";
    std::cout << "Stefan-Boltzmann constant: "    <<  constants::sigmaB   << "\n";
    std::cout << "Fine structure constant: "      <<  constants::alpha    << "\n";
    std::cout << "The Rydberg constant: "         <<  constants::rydB     << "\n";
    return 0;
}