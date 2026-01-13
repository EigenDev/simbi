// =============================================================================
// units.hpp
//
// compile-time dimensional analysis for physical calculations.
// uses c++20 template parameter deduction for clean syntax.
//
// dimensions tracked: [M^m L^l T^t Q^q K^k]
//   M = mass (grams)
//   L = length (centimeters)
//   T = time (seconds)
//   Q = charge (statcoulombs)
//   K = temperature (kelvin)
//
// usage:
//   mass_t m = 1.0;           // 1 gram
//   length_t r = 1e10;        // 1e10 cm
//   velocity_t v = r / (1.0 * second);  // cm/s, type-checked
//   energy_t e = m * v * v;   // erg, dimensions verified at compile time
//
// operations:
//   - same dimensions: +, -, <, >
//   - multiplication/division: automatic dimension tracking
//   - scalar multiply/divide: preserves dimensions
//
// =============================================================================

#ifndef SIMBI_AFTERGLOW_UNITS_HPP
#define SIMBI_AFTERGLOW_UNITS_HPP

#include <cmath>
#include <ostream>

namespace simbi::afterglow {

    // core quantity type: value with compile-time dimensions
    template <int M, int L, int T, int Q, int K>
    struct quantity_t
    {
        double value;

        constexpr quantity_t(double v = 0.0) : value(v) {}

        // arithmetic for same dimensions
        constexpr quantity_t operator+(quantity_t rhs) const
        {
            return {value + rhs.value};
        }
        constexpr quantity_t operator-(quantity_t rhs) const
        {
            return {value - rhs.value};
        }
        constexpr quantity_t operator-() const
        {
            return {-value};
        }

        constexpr quantity_t& operator+=(quantity_t rhs)
        {
            value += rhs.value;
            return *this;
        }
        constexpr quantity_t& operator-=(quantity_t rhs)
        {
            value -= rhs.value;
            return *this;
        }

        // scalar operations
        constexpr quantity_t operator*(double s) const
        {
            return {value * s};
        }
        constexpr quantity_t operator/(double s) const
        {
            return {value / s};
        }
        constexpr quantity_t& operator*=(double s)
        {
            value *= s;
            return *this;
        }
        constexpr quantity_t& operator/=(double s)
        {
            value /= s;
            return *this;
        }

        friend constexpr quantity_t operator*(double s, quantity_t q)
        {
            return {s * q.value};
        }
        friend constexpr auto operator/(double s, quantity_t q)
        {
            return quantity_t<-M, -L, -T, -Q, -K>{s / q.value};
        }

        // comparisons
        constexpr bool operator<(quantity_t rhs) const
        {
            return value < rhs.value;
        }
        constexpr bool operator>(quantity_t rhs) const
        {
            return value > rhs.value;
        }
        constexpr bool operator<=(quantity_t rhs) const
        {
            return value <= rhs.value;
        }
        constexpr bool operator>=(quantity_t rhs) const
        {
            return value >= rhs.value;
        }
        constexpr bool operator==(quantity_t rhs) const
        {
            return value == rhs.value;
        }
        constexpr bool operator!=(quantity_t rhs) const
        {
            return value != rhs.value;
        }

        // multiplication: add dimensions
        template <int M2, int L2, int T2, int Q2, int K2>
        constexpr auto operator*(quantity_t<M2, L2, T2, Q2, K2> rhs) const
        {
            return quantity_t<M + M2, L + L2, T + T2, Q + Q2, K + K2>{value * rhs.value};
        }

        // division: subtract dimensions
        template <int M2, int L2, int T2, int Q2, int K2>
        constexpr auto operator/(quantity_t<M2, L2, T2, Q2, K2> rhs) const
        {
            return quantity_t<M - M2, L - L2, T - T2, Q - Q2, K - K2>{value / rhs.value};
        }
    };

    // dimensionless specialization: implicit conversion to/from double
    template <>
    struct quantity_t<0, 0, 0, 0, 0>
    {
        double value;

        constexpr quantity_t(double v = 0.0) : value(v) {}
        constexpr operator double() const
        {
            return value;
        }

        constexpr quantity_t operator+(quantity_t rhs) const
        {
            return {value + rhs.value};
        }
        constexpr quantity_t operator-(quantity_t rhs) const
        {
            return {value - rhs.value};
        }
        constexpr quantity_t operator-() const
        {
            return {-value};
        }
        constexpr quantity_t operator*(double s) const
        {
            return {value * s};
        }
        constexpr quantity_t operator/(double s) const
        {
            return {value / s};
        }

        friend constexpr quantity_t operator*(double s, quantity_t q)
        {
            return {s * q.value};
        }
        friend constexpr quantity_t operator/(double s, quantity_t q)
        {
            return {s / q.value};
        }

        constexpr bool operator<(quantity_t rhs) const
        {
            return value < rhs.value;
        }
        constexpr bool operator>(quantity_t rhs) const
        {
            return value > rhs.value;
        }
    };

    // =============================================================================
    // fundamental dimensions (cgs units)
    // =============================================================================

    using mass_t        = quantity_t<1, 0, 0, 0, 0>; // gram
    using length_t      = quantity_t<0, 1, 0, 0, 0>; // centimeter
    using time_t        = quantity_t<0, 0, 1, 0, 0>; // second
    using charge_t      = quantity_t<0, 0, 0, 1, 0>; // statcoulomb
    using temperature_t = quantity_t<0, 0, 0, 0, 1>; // kelvin

    // derived dimensions
    using dimensionless_t  = quantity_t<0, 0, 0, 0, 0>;
    using velocity_t       = quantity_t<0, 1, -1, 0, 0>;  // cm/s
    using acceleration_t   = quantity_t<0, 1, -2, 0, 0>;  // cm/s^2
    using energy_t         = quantity_t<1, 2, -2, 0, 0>;  // erg
    using power_t          = quantity_t<1, 2, -3, 0, 0>;  // erg/s
    using force_t          = quantity_t<1, 1, -2, 0, 0>;  // dyne
    using area_t           = quantity_t<0, 2, 0, 0, 0>;   // cm^2
    using volume_t         = quantity_t<0, 3, 0, 0, 0>;   // cm^3
    using frequency_t      = quantity_t<0, 0, -1, 0, 0>;  // Hz
    using mass_density_t   = quantity_t<1, -3, 0, 0, 0>;  // g/cm^3
    using energy_density_t = quantity_t<1, -1, -2, 0, 0>; // erg/cm^3
    using number_density_t = quantity_t<0, -3, 0, 0, 0>;  // cm^-3

    // electromagnetic
    // note: magnetic field in cgs gaussian units has fractional dimensions [g^1/2 cm^-1/2 s^-1]
    // which cannot be represented with integer template parameters. use double for B in gauss.
    // B^2 / 8π has dimensions of energy density and can be properly typed.
    using magnetic_field_squared_t = quantity_t<1, -1, -2, 0, 0>; // gauss^2 (energy density)
    using electric_field_t         = quantity_t<1, -1, -1, 0, 0>; // statV/cm

    // radiative (note: "per frequency" is implicit in spectral quantities)
    using spectral_flux_t       = quantity_t<1, 0, -2, 0, 0>;  // erg/cm^2/s (flux density)
    using spectral_power_t      = quantity_t<1, 2, -2, 0, 0>;  // erg (energy per frequency)
    using emissivity_t          = quantity_t<1, -1, -2, 0, 0>; // erg/cm/s
    using spectral_emissivity_t = quantity_t<1, -1, -2, 0, 0>; // erg/cm/s (same as emissivity)

    // =============================================================================
    // common unit literals (cgs)
    // =============================================================================

    namespace literals {
        // fundamental
        constexpr mass_t operator""_g(long double v)
        {
            return {static_cast<double>(v)};
        }
        constexpr mass_t operator""_g(unsigned long long v)
        {
            return {static_cast<double>(v)};
        }

        constexpr length_t operator""_cm(long double v)
        {
            return {static_cast<double>(v)};
        }
        constexpr length_t operator""_cm(unsigned long long v)
        {
            return {static_cast<double>(v)};
        }

        constexpr time_t operator""_s(long double v)
        {
            return {static_cast<double>(v)};
        }
        constexpr time_t operator""_s(unsigned long long v)
        {
            return {static_cast<double>(v)};
        }

        constexpr temperature_t operator""_K(long double v)
        {
            return {static_cast<double>(v)};
        }
        constexpr temperature_t operator""_K(unsigned long long v)
        {
            return {static_cast<double>(v)};
        }

        // derived
        constexpr energy_t operator""_erg(long double v)
        {
            return {static_cast<double>(v)};
        }
        constexpr energy_t operator""_erg(unsigned long long v)
        {
            return {static_cast<double>(v)};
        }

        constexpr frequency_t operator""_Hz(long double v)
        {
            return {static_cast<double>(v)};
        }
        constexpr frequency_t operator""_Hz(unsigned long long v)
        {
            return {static_cast<double>(v)};
        }
    } // namespace literals

    // =============================================================================
    // unit conversions (define base units, derive others)
    // =============================================================================

    // mass
    constexpr double kg_to_g    = 1e3;
    constexpr double msun_to_g  = 1.98841e33;
    constexpr mass_t kilogram   = mass_t{kg_to_g};
    constexpr mass_t solar_mass = mass_t{msun_to_g};

    // length
    constexpr double   m_to_cm   = 1e2;
    constexpr double   km_to_cm  = 1e5;
    constexpr double   pc_to_cm  = 3.0857e18;
    constexpr double   ly_to_cm  = 9.4607e17;
    constexpr length_t meter     = length_t{m_to_cm};
    constexpr length_t kilometer = length_t{km_to_cm};
    constexpr length_t parsec    = length_t{pc_to_cm};
    constexpr length_t lightyear = length_t{ly_to_cm};

    // time
    constexpr double min_to_s = 60.0;
    constexpr double hr_to_s  = 3600.0;
    constexpr double day_to_s = 86400.0;
    constexpr double yr_to_s  = 31557600.0;
    constexpr time_t minute   = time_t{min_to_s};
    constexpr time_t hour     = time_t{hr_to_s};
    constexpr time_t day      = time_t{day_to_s};
    constexpr time_t year     = time_t{yr_to_s};

    // radiative
    constexpr double          jy_to_cgs   = 1e-23; // jansky to erg/cm^2/s/Hz
    constexpr double          mjy_to_cgs  = 1e-26; // millijansky
    constexpr spectral_flux_t jansky      = spectral_flux_t{jy_to_cgs};
    constexpr spectral_flux_t millijansky = spectral_flux_t{mjy_to_cgs};

    // =============================================================================
    // unit constants (simple multipliers for clean code)
    // =============================================================================

    namespace units {
        // base units
        constexpr mass_t        gram{1.0};   // gram
        constexpr length_t      cm{1.0};     // centimeter
        constexpr time_t        s{1.0};      // second
        constexpr charge_t      statC{1.0};  // statcoulomb
        constexpr temperature_t kelvin{1.0}; // kelvin

        // derived units
        constexpr energy_t         erg{1.0};         // erg
        constexpr frequency_t      hz{1.0};          // hertz
        constexpr area_t           cm2{1.0};         // cm^2
        constexpr volume_t         cm3{1.0};         // cm^3
        constexpr velocity_t       cm_per_s{1.0};    // cm/s
        constexpr mass_density_t   g_per_cm3{1.0};   // g/cm^3
        constexpr energy_density_t erg_per_cm3{1.0}; // erg/cm^3
        constexpr power_t          erg_per_s{1.0};   // erg/s
        constexpr spectral_flux_t  mjy{mjy_to_cgs};  // millijansky
        constexpr spectral_flux_t  jy{jy_to_cgs};    // jansky
    } // namespace units

    // =============================================================================
    // math functions preserving dimensions
    // =============================================================================

    // sqrt: halve all dimensions
    template <int M, int L, int T, int Q, int K>
    constexpr auto sqrt(quantity_t<M, L, T, Q, K> q)
    {
        static_assert(
            M % 2 == 0 && L % 2 == 0 && T % 2 == 0 && Q % 2 == 0 && K % 2 == 0,
            "sqrt requires even dimension powers"
        );
        return quantity_t<M / 2, L / 2, T / 2, Q / 2, K / 2>{std::sqrt(q.value)};
    }

    // pow<N>: multiply dimensions by N
    template <int N, int M, int L, int T, int Q, int K>
    constexpr auto pow(quantity_t<M, L, T, Q, K> q)
    {
        return quantity_t<M * N, L * N, T * N, Q * N, K * N>{std::pow(q.value, N)};
    }

    // abs: preserve dimensions
    template <int M, int L, int T, int Q, int K>
    constexpr auto abs(quantity_t<M, L, T, Q, K> q)
    {
        return quantity_t<M, L, T, Q, K>{std::abs(q.value)};
    }

    // =============================================================================
    // output streaming
    // =============================================================================

    template <int M, int L, int T, int Q, int K>
    std::ostream& operator<<(std::ostream& os, quantity_t<M, L, T, Q, K> q)
    {
        os << q.value << " [";
        bool first = true;

        auto append_dim = [&](const char* name, int power) {
            if (power != 0) {
                if (!first) {
                    os << " ";
                }
                os << name;
                if (power != 1) {
                    os << "^" << power;
                }
                first = false;
            }
        };

        append_dim("g", M);
        append_dim("cm", L);
        append_dim("s", T);
        append_dim("statC", Q);
        append_dim("K", K);

        if (first) {
            os << "1"; // dimensionless
        }
        os << "]";
        return os;
    }

} // namespace simbi::afterglow

#endif // SIMBI_AFTERGLOW_UNITS_HPP
