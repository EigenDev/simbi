/**
    A units library that will help in
    performing type-safe physics operations
    while taking advantage of the
    speedups of c++
    @file units.hpp
    @author Marcus DuPont
    @version 0.1 05/10/22
    Adapted from:Scott Meyers, Ph.D.
*/
#ifndef UNITS_HPP
#define UNITS_HPP

#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <ratio>
#include <string>

#if __cplusplus >= 202002L && !defined(__clang__)
using string_literal = std::string;
#else
using string_literal = const char*;
#endif
/*
Template class for units library.
The format is:
m = mass, l = length, t = time, q = charge,
temp = temperature, intensity = intesoty, angle = angle
*/
namespace units {
    template <typename T>
    constexpr int sgn(T val);

    //=======================================
    // Mass type specializations
    //=======================================
    enum class Mass_t {
        Gram,
        Kilogram,
        SolarMass,
    };

    template <typename P = double>
    class Kilogram;

    template <typename P = double>
    class SolarMass;

    template <typename P = double>
    class Gram;

    template <typename P>
    std::ostream& operator<<(std::ostream& os, const Gram<P>& mass);

    template <typename P>
    std::ostream& operator<<(std::ostream& os, const SolarMass<P>& mass);

    template <typename P>
    std::ostream& operator<<(std::ostream& os, const Kilogram<P>& mass);

    //=======================================
    // Time type specializations
    //=======================================
    template <typename P = double>
    class Year;

    template <typename P = double>
    class Day;

    template <typename P = double>
    class Hour;

    template <typename P = double>
    class Second;

    template <typename P>
    std::ostream& operator<<(std::ostream& os, const Second<P>& t);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Year<P>& t);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Day<P>& t);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Hour<P>& t);

    enum class Time_t {
        Second,
        Hour,
        Day,
        Year,
    };
    //=======================================
    // Length type specializations
    //=======================================
    template <typename P = double>
    class Parsec;

    template <typename P = double>
    class Lightyear;

    template <typename P = double>
    class Kilometer;

    template <typename P = double>
    class Meter;

    template <typename P = double>
    class Centimeter;

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Centimeter<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Meter<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Kilometer<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Lightyear<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Parsec<P>& ell);

    enum class Length_t {
        Centimeter,
        Meter,
        Kilometer,
        Parsec,
        Lightyear,
    };

    //=======================================
    // Charge type specializations
    //=======================================
    template <typename P = double>
    class Coulomb;

    template <typename P = double>
    class StatCoulomb;

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const StatCoulomb<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Coulomb<P>& ell);

    enum class Charge_t {
        StatCoulomb,
        Coulomb,
    };
    //=======================================
    // Temperature type specializations
    //=======================================
    template <typename P = double>
    class Celcius;

    template <typename P = double>
    class Fahrenheit;

    template <typename P = double>
    class ElectronVolt;

    template <typename P = double>
    class Kelvin;

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Kelvin<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Fahrenheit<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Celcius<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const ElectronVolt<P>& ell);

    enum class Temperature_t {
        Kelvin,
        Celcius,
        Fahrenheit,
    };
    //=======================================
    // Irradiance type specializations
    //=======================================
    // Erg per square meter per second
    template <typename P = double>
    class ErgCM2P2;

    enum class Irradiance_t {
        ErgCM2P2,
    };
    //=======================================
    // Angle type specializations
    //=======================================
    template <typename P = double>
    class Degree;

    template <typename P = double>
    class Radian;

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Radian<P>& ell);

    template <typename P>
    std::ostream& operator<<(std::ostream& out, const Degree<P>& ell);

    enum class Angle_t {
        Degree,
        Radian,
    };

    //=============================================================================================
    extern std::map<Mass_t, std::string> mass_dict;
    extern std::map<Length_t, std::string> length_dict;
    extern std::map<Time_t, std::string> time_dict;
    extern std::map<Charge_t, std::string> charge_dict;
    extern std::map<Temperature_t, std::string> temp_dict;
    extern std::map<Irradiance_t, std::string> rad_dict;
    extern std::map<Angle_t, std::string> angle_dict;

    /*
    Strong typed quantities class that allows conversion from
    one set of units to another. Useful for dimensional analyis,
    so in a way, we have some ``dimensional safety'' when
    performing physics calculation
    */
    template <
        typename P,                                   // Precision
        typename m         = int,                     // power of mass
        typename l         = int,                     // power of length
        typename t         = int,                     // power of time
        typename q         = int,                     // power of charge
        typename temp      = int,                     // power of temperature
        typename intensity = int,                     // power of irradiance
        typename angle     = int,                     // power of angle
        Mass_t M           = Mass_t::Gram,            // Mass unit type
        Length_t L         = Length_t::Centimeter,    // Length unit type
        Time_t T           = Time_t::Second,          // Time unit type
        Charge_t Q         = Charge_t::StatCoulomb,   // Charge unit type
        Temperature_t K    = Temperature_t::Kelvin,   // Temperature unit type
        Irradiance_t I =
            Irradiance_t::ErgCM2P2,    // Luminous Intensity unit type
        Angle_t A = Angle_t::Radian>   // Angle unit type
    struct quantity;
    //=============================================================================================
    template <typename T>
    constexpr string_literal rat2str();

    //=============================================================================================
    template <
        typename P,                                 // Precision
        typename m,                                 // power of mass
        typename l,                                 // power of length
        typename t,                                 // power of time
        typename q,                                 // power of charge
        typename temp,                              // power of temperature
        typename intensity,                         // power of irradiance
        typename angle,                             // power of angle
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    std::ostream& operator<<(
        std::ostream& out,
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& ell
    );

    // "Non-assignment operators are best implemented as non-members"
    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator<(
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator>(
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator+(
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator-(
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator*(
        double lhs,
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator*(
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        double rhs
    );

    template <
        typename P,
        typename m,
        typename l,
        typename t,
        typename q,
        typename temp,
        typename intensity,
        typename angle,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator/(
        const quantity<
            P,
            m,
            l,
            t,
            q,
            temp,
            intensity,
            angle,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        double rhs
    );

    template <
        typename P,
        typename m1,
        typename l1,
        typename t1,
        typename q1,
        typename temp1,
        typename intensity1,
        typename angle1,
        typename m2,
        typename l2,
        typename t2,
        typename q2,
        typename temp2,
        typename intensity2,
        typename angle2,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator*(
        const quantity<
            P,
            m1,
            l1,
            t1,
            q1,
            temp1,
            intensity1,
            angle1,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m2,
            l2,
            t2,
            q2,
            temp2,
            intensity2,
            angle2,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m1,
        typename l1,
        typename t1,
        typename q1,
        typename temp1,
        typename intensity1,
        typename angle1,
        typename m2,
        typename l2,
        typename t2,
        typename q2,
        typename temp2,
        typename intensity2,
        typename angle2,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    const auto operator-(
        const quantity<
            P,
            m1,
            l1,
            t1,
            q1,
            temp1,
            intensity1,
            angle1,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m2,
            l2,
            t2,
            q2,
            temp2,
            intensity2,
            angle2,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m1,
        typename l1,
        typename t1,
        typename q1,
        typename temp1,
        typename intensity1,
        typename angle1,
        typename m2,
        typename l2,
        typename t2,
        typename q2,
        typename temp2,
        typename intensity2,
        typename angle2,
        Mass_t M        = Mass_t::Gram,             // Mass unit type
        Length_t L      = Length_t::Centimeter,     // Length unit type
        Time_t T        = Time_t::Second,           // Time unit type
        Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
        Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
        Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity unit
                                                    // type
        Angle_t A = Angle_t::Radian>                // Angle unit type
    constexpr auto operator/(
        const quantity<
            P,
            m1,
            l1,
            t1,
            q1,
            temp1,
            intensity1,
            angle1,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& lhs,
        const quantity<
            P,
            m2,
            l2,
            t2,
            q2,
            temp2,
            intensity2,
            angle2,
            M,
            L,
            T,
            Q,
            K,
            I,
            A>& rhs
    );

    template <
        typename P,
        typename m1,
        typename l1,
        typename t1,
        typename q1,
        typename temp1,
        typename intensity1,
        typename angle1,
        typename m2,
        typename l2,
        typename t2,
        typename q2,
        typename temp2,
        typename intensity2,
        typename angle2,
        Mass_t M1,
        Mass_t M2,
        Length_t L1,
        Length_t L2,
        Time_t T1,
        Time_t T2,
        Charge_t Q1,
        Charge_t Q2,
        Temperature_t K1,
        Temperature_t K2,
        Irradiance_t I1,
        Irradiance_t I2,
        Angle_t A1,
        Angle_t A2>
    constexpr auto operator/(
        const quantity<
            P,
            m1,
            l1,
            t1,
            q1,
            temp1,
            intensity1,
            angle1,
            M1,
            L1,
            T1,
            Q1,
            K1,
            I1,
            A1>& lhs,
        const quantity<
            P,
            m2,
            l2,
            t2,
            q2,
            temp2,
            intensity2,
            angle2,
            M2,
            L2,
            T2,
            Q2,
            K2,
            I2,
            A2>& rhs
    );

    // Special case for dimensionless type
    template <typename P>
    struct quantity<
        P,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>;

    namespace math {
        template <
            typename power,
            typename P,
            typename m,
            typename l,
            typename t,
            typename q,
            typename temp,
            typename intensity,
            typename angle,
            Mass_t M        = Mass_t::Gram,             // Mass unit type
            Length_t L      = Length_t::Centimeter,     // Length unit type
            Time_t T        = Time_t::Second,           // Time unit type
            Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
            Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
            Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity
                                                        // unit type
            Angle_t A = Angle_t::Radian>                // Angle unit type
        constexpr auto pow(const quantity<
                           P,
                           m,
                           l,
                           t,
                           q,
                           temp,
                           intensity,
                           angle,
                           M,
                           L,
                           T,
                           Q,
                           K,
                           I,
                           A>& quant);

        template <
            typename P,
            typename m,
            typename l,
            typename t,
            typename q,
            typename temp,
            typename intensity,
            typename angle,
            Mass_t M        = Mass_t::Gram,             // Mass unit type
            Length_t L      = Length_t::Centimeter,     // Length unit type
            Time_t T        = Time_t::Second,           // Time unit type
            Charge_t Q      = Charge_t::StatCoulomb,    // Charge unit type
            Temperature_t K = Temperature_t::Kelvin,    // Temperature unit type
            Irradiance_t I  = Irradiance_t::ErgCM2P2,   // Luminous Intensity
                                                        // unit type
            Angle_t A = Angle_t::Radian>                // Angle unit type
        constexpr auto sqrt(const quantity<
                            P,
                            m,
                            l,
                            t,
                            q,
                            temp,
                            intensity,
                            angle,
                            M,
                            L,
                            T,
                            Q,
                            K,
                            I,
                            A>& val);
    }   // namespace math

}   // namespace units

#include "units.tpp"
#endif