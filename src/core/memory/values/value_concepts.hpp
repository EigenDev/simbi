#ifndef VALUE_CONCEPTS_HPP
#define VALUE_CONCEPTS_HPP

#include "config.hpp"
#include "core/containers/vector.hpp"
#include "core/utility/enums.hpp"
#include <concepts>

namespace simbi::concepts {
    // concept defining a state variable - structural approach
    template <typename T>
    concept is_hydro_primitive_c = requires(T t) {
        { t.rho } -> std::convertible_to<real>;
        { t.vel } -> std::convertible_to<spatial_vector_t<real, T::dimensions>>;
        { t.pre } -> std::convertible_to<real>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_hydro_conserved_c = requires(T t) {
        { t.den } -> std::convertible_to<real>;
        { t.mom } -> std::convertible_to<spatial_vector_t<real, T::dimensions>>;
        { t.nrg } -> std::convertible_to<real>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_mhd_primitive_c = requires(T t) {
        { t.rho } -> std::convertible_to<real>;
        { t.vel } -> std::convertible_to<spatial_vector_t<real, T::dimensions>>;
        { t.pre } -> std::convertible_to<real>;
        {
            t.mag
        } -> std::convertible_to<magnetic_vector_t<real, T::dimensions>>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_mhd_conserved_c = requires(T t) {
        { t.den } -> std::convertible_to<real>;
        { t.mom } -> std::convertible_to<spatial_vector_t<real, T::dimensions>>;
        { t.nrg } -> std::convertible_to<real>;
        {
            t.mag
        } -> std::convertible_to<magnetic_vector_t<real, T::dimensions>>;
        { t.chi } -> std::convertible_to<real>;
    };

    // concept defining a state variable - type approach
    template <typename T>
    concept is_any_state_variable_c =
        is_hydro_primitive_c<T> || is_hydro_conserved_c<T> ||
        is_mhd_primitive_c<T> || is_mhd_conserved_c<T>;

    template <typename T>
    concept is_relativistic_c = requires {
        { T::regime } -> std::convertible_to<Regime>;
        requires T::regime == Regime::SRHD || T::regime == Regime::RMHD;
    };

    template <typename T>
    concept is_relativistic_primitive_c =
        (is_hydro_primitive_c<T> || is_mhd_primitive_c<T>) &&
        is_relativistic_c<T>;

    template <typename T>
    concept is_relativistic_conserved_c =
        (is_hydro_conserved_c<T> || is_mhd_conserved_c<T>) &&
        is_relativistic_c<T>;

    template <typename T>
    concept is_srhd_c = is_relativistic_c<T> && T::regime == Regime::SRHD;

    template <typename T>
    concept is_rmhd_c = is_relativistic_c<T> && T::regime == Regime::RMHD;

}   // namespace simbi::concepts
#endif   // VALUE_CONCEPTS_HPP
