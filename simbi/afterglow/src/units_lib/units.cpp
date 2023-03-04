/**
    UNITS is a library built to maintain some dimensional-type safety to 
    physcs calculation
    @file units.cpp
    @author Marcus DuPont
    @version 0.1 05/10/22
*/
#include "units.hpp"

namespace units
{
    std::map<Mass_t, std::string> mass_dict = {
        {Mass_t::Gram,"g"},
        {Mass_t::Kilogram, "kg"},
        {Mass_t::SolarMass, "M_sun"},
    };

    std::map<Length_t, std::string> length_dict = {
        {Length_t::Centimeter,"cm"},
        {Length_t::Kilometer, "km"},
        {Length_t::Lightyear, "ly"},
        {Length_t::Parsec, "pc"},
    };

    std::map<Time_t, std::string> time_dict = {
        {Time_t::Second,"s"},
        {Time_t::Hour, "hr"},
        {Time_t::Day, "day"},
        {Time_t::Year, "yr"},
    };

    std::map<Charge_t, std::string> charge_dict = {
        {Charge_t::Coulomb,"C"},
        {Charge_t::StatCoulomb, "statC"},
    };

    std::map<Temperature_t, std::string> temp_dict = {
        {Temperature_t::Kelvin,"K"},
        {Temperature_t::Celcius, "Cel"},
        {Temperature_t::Fahrenheit, "Fah"},
    };

    std::map<Irradiance_t, std::string> rad_dict = {
        {Irradiance_t::ErgCM2P2,"erg per cm2 per s2"},
    };

    std::map<Angle_t, std::string> angle_dict = {
        {Angle_t::Radian,"rad"},
        {Angle_t::Degree, "deg"},
    };
} // namespace units
