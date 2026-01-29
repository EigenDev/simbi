// =============================================================================
// units.hpp
//
// compile-time dimensional analysis for physical calculations.
// uses c++11 std::ratio for fractional dimension powers.
//
// dimensions tracked: [M^m L^l T^t Q^q K^k]
//   M = mass (grams)
//   L = length (centimeters)
//   T = time (seconds)
//   Q = charge (statcoulombs)
//   K = temperature (kelvin)
//
// fractional dimensions supported via std::ratio:
//   magnetic field (gauss): g^{1/2} cm^{-1/2} s^{-1}
//   charge (statcoulomb): g^{1/2} cm^{3/2} s^{-1}
//
// usage:
//   mass_t m = 1.0;           // 1 gram
//   length_t r = 1e10;        // 1e10 cm
//   velocity_t v = r / (1.0 * second);  // cm/s, type-checked
//   energy_t e = m * v * v;   // erg, dimensions verified at compile time
//   magnetic_field_t b = 100.0; // 100 gauss, with fractional dimensions
//
// operations:
//   - same dimensions: +, -, <, >
//   - multiplication/division: automatic dimension tracking
//   - scalar multiply/divide: preserves dimensions
//
// =============================================================================
#pragma once

#include <cmath>
#include <ostream>
#include <ratio>

namespace simbi::afterglow {

    // core quantity type: value with compile-time fractional dimensions
    template <
        typename M = std::ratio<0>,
        typename L = std::ratio<0>,
        typename T = std::ratio<0>,
        typename Q = std::ratio<0>,
        typename K = std::ratio<0>>
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
            using M_neg = std::ratio<-M::num, M::den>;
            using L_neg = std::ratio<-L::num, L::den>;
            using T_neg = std::ratio<-T::num, T::den>;
            using Q_neg = std::ratio<-Q::num, Q::den>;
            using K_neg = std::ratio<-K::num, K::den>;
            return quantity_t<M_neg, L_neg, T_neg, Q_neg, K_neg>{s / q.value};
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
        template <typename M2, typename L2, typename T2, typename Q2, typename K2>
        constexpr auto operator*(quantity_t<M2, L2, T2, Q2, K2> rhs) const
        {
            using M_result = std::ratio_add<M, M2>;
            using L_result = std::ratio_add<L, L2>;
            using T_result = std::ratio_add<T, T2>;
            using Q_result = std::ratio_add<Q, Q2>;
            using K_result = std::ratio_add<K, K2>;
            return quantity_t<M_result, L_result, T_result, Q_result, K_result>{value * rhs.value};
        }

        // division: subtract dimensions
        template <typename M2, typename L2, typename T2, typename Q2, typename K2>
        constexpr auto operator/(quantity_t<M2, L2, T2, Q2, K2> rhs) const
        {
            using M_result = std::ratio_subtract<M, M2>;
            using L_result = std::ratio_subtract<L, L2>;
            using T_result = std::ratio_subtract<T, T2>;
            using Q_result = std::ratio_subtract<Q, Q2>;
            using K_result = std::ratio_subtract<K, K2>;
            return quantity_t<M_result, L_result, T_result, Q_result, K_result>{value / rhs.value};
        }
    };

    // dimensionless specialization: implicit conversion to/from double
    template <>
    struct quantity_t<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>
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

    // =========================================================================
    // base physical dimensions (fundamental units)
    // =========================================================================

    using mass_t = quantity_t<
        std::ratio<1>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>; // gram
    using length_t = quantity_t<
        std::ratio<0>,
        std::ratio<1>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>; // centimeter
    using time_t = quantity_t<
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<1>,
        std::ratio<0>,
        std::ratio<0>>; // second
    using charge_t = quantity_t<
        std::ratio<1, 2>,
        std::ratio<3, 2>,
        std::ratio<-1>,
        std::ratio<0>,
        std::ratio<0>>; // statcoulomb (esu)
    using temperature_t = quantity_t<
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<1>>; // kelvin

    // =========================================================================
    // derived dimensions (integer powers)
    // =========================================================================

    using dimensionless_t =
        quantity_t<std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using velocity_t = quantity_t<
        std::ratio<0>,
        std::ratio<1>,
        std::ratio<-1>,
        std::ratio<0>,
        std::ratio<0>>; // cm/s
    using acceleration_t = quantity_t<
        std::ratio<0>,
        std::ratio<1>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // cm/s^2
    using energy_t = quantity_t<
        std::ratio<1>,
        std::ratio<2>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // erg
    using power_t = quantity_t<
        std::ratio<1>,
        std::ratio<2>,
        std::ratio<-3>,
        std::ratio<0>,
        std::ratio<0>>; // erg/s
    using force_t = quantity_t<
        std::ratio<1>,
        std::ratio<1>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // dyne
    using area_t = quantity_t<
        std::ratio<0>,
        std::ratio<2>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>; // cm^2
    using volume_t = quantity_t<
        std::ratio<0>,
        std::ratio<3>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>; // cm^3
    using frequency_t = quantity_t<
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<-1>,
        std::ratio<0>,
        std::ratio<0>>; // Hz
    using mass_density_t = quantity_t<
        std::ratio<1>,
        std::ratio<-3>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>; // g/cm^3
    using energy_density_t = quantity_t<
        std::ratio<1>,
        std::ratio<-1>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // erg/cm^3
    using number_density_t = quantity_t<
        std::ratio<0>,
        std::ratio<-3>,
        std::ratio<0>,
        std::ratio<0>,
        std::ratio<0>>; // cm^-3

    // =========================================================================
    // electromagnetic (fractional dimensions in cgs gaussian units)
    // =========================================================================

    // magnetic field: g^{1/2} cm^{-1/2} s^{-1} (gauss)
    using magnetic_field_t = quantity_t<
        std::ratio<1, 2>,
        std::ratio<-1, 2>,
        std::ratio<-1>,
        std::ratio<0>,
        std::ratio<0>>;

    // electric field: g^{1/2} cm^{-1/2} s^{-1} (statV/cm, same dimensions as B in gaussian units)
    using electric_field_t = quantity_t<
        std::ratio<1, 2>,
        std::ratio<-1, 2>,
        std::ratio<-1>,
        std::ratio<0>,
        std::ratio<0>>;

    // B^2 has dimensions of energy density (useful for u_B = B^2 / 8\pi)
    using magnetic_energy_density_t =
        quantity_t<std::ratio<1>, std::ratio<-1>, std::ratio<-2>, std::ratio<0>, std::ratio<0>>;

    // =========================================================================
    // radiative quantities (spectral = "per frequency" implicit)
    // =========================================================================

    using spectral_flux_t = quantity_t<
        std::ratio<1>,
        std::ratio<0>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // erg/cm^2/s (flux density)
    using spectral_power_t = quantity_t<
        std::ratio<1>,
        std::ratio<2>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // erg (energy per frequency)
    using emissivity_t = quantity_t<
        std::ratio<1>,
        std::ratio<-1>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // erg/cm/s
    using spectral_emissivity_t = quantity_t<
        std::ratio<1>,
        std::ratio<-1>,
        std::ratio<-2>,
        std::ratio<0>,
        std::ratio<0>>; // erg/cm/s (same as emissivity)

    // =========================================================================
    // user-defined literals
    // =========================================================================

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

    // =========================================================================
    // conversion factors and derived units
    // =========================================================================

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

    // =========================================================================
    // base units namespace (for explicit construction)
    // =========================================================================

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
        constexpr magnetic_field_t gauss{1.0};       // gauss
    } // namespace units

    // =========================================================================
    // mathematical operations
    // =========================================================================

    // sqrt: halve all dimension powers
    template <typename M, typename L, typename T, typename Q, typename K>
    constexpr auto sqrt(quantity_t<M, L, T, Q, K> q)
    {
        using M_half = std::ratio_divide<M, std::ratio<2>>;
        using L_half = std::ratio_divide<L, std::ratio<2>>;
        using T_half = std::ratio_divide<T, std::ratio<2>>;
        using Q_half = std::ratio_divide<Q, std::ratio<2>>;
        using K_half = std::ratio_divide<K, std::ratio<2>>;
        return quantity_t<M_half, L_half, T_half, Q_half, K_half>{std::sqrt(q.value)};
    }

    // pow<N, D>: multiply dimensions by N/D (rational power)
    template <
        std::intmax_t N,
        std::intmax_t D = 1,
        typename M,
        typename L,
        typename T,
        typename Q,
        typename K>
    constexpr auto pow(quantity_t<M, L, T, Q, K> q)
    {
        using power_ratio = std::ratio<N, D>;
        using M_pow       = std::ratio_multiply<M, power_ratio>;
        using L_pow       = std::ratio_multiply<L, power_ratio>;
        using T_pow       = std::ratio_multiply<T, power_ratio>;
        using Q_pow       = std::ratio_multiply<Q, power_ratio>;
        using K_pow       = std::ratio_multiply<K, power_ratio>;
        return quantity_t<M_pow, L_pow, T_pow, Q_pow, K_pow>{
            std::pow(q.value, static_cast<double>(N) / D)
        };
    }

    // abs: preserve dimensions
    template <typename M, typename L, typename T, typename Q, typename K>
    constexpr auto abs(quantity_t<M, L, T, Q, K> q)
    {
        return quantity_t<M, L, T, Q, K>{std::abs(q.value)};
    }

    // =========================================================================
    // output stream (for debugging)
    // =========================================================================

    namespace detail {
        // helper to format rational exponents
        template <typename R>
        void print_dimension(std::ostream& os, const char* name, bool& first)
        {
            if (R::num != 0) {
                if (!first) {
                    os << " ";
                }
                os << name;
                if (R::den == 1 && R::num != 1) {
                    os << "^" << R::num;
                }
                else if (R::den != 1) {
                    os << "^(" << R::num << "/" << R::den << ")";
                }
                first = false;
            }
        }
    } // namespace detail

    template <typename M, typename L, typename T, typename Q, typename K>
    std::ostream& operator<<(std::ostream& os, quantity_t<M, L, T, Q, K> q)
    {
        os << q.value << " [";
        bool first = true;

        detail::print_dimension<M>(os, "g", first);
        detail::print_dimension<L>(os, "cm", first);
        detail::print_dimension<T>(os, "s", first);
        detail::print_dimension<Q>(os, "statC", first);
        detail::print_dimension<K>(os, "K", first);

        if (first) {
            os << "1"; // dimensionless
        }
        os << "]";
        return os;
    }

} // namespace simbi::afterglow
