#pragma once

#include "compat.hpp"
#include "utility/enums.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace simbi {
    template <typename T, std::uint64_t Rank>
    struct vector_t;
} // namespace simbi

namespace simbi::concepts {
    // =============================================================================
    // Core Concepts
    // ============================================================================
    template <typename T>
    concept Arithmetic = std::integral<T> || std::floating_point<T>;

    template <typename F, std::uint64_t Rank>
    concept ArrayFunction = requires(F f, std::array<size_t, Rank> point) {
        { f(point) };
    };

    template <std::uint64_t Rank>
    concept valid_dimension = (Rank >= 1 && Rank <= 3);

    // concept defining a state variable - structural approach
    template <typename T>
    concept is_hydro_primitive_c = requires(T t) {
        { t.rho } -> std::convertible_to<real>;
        { t.vel } -> std::convertible_to<vector_t<real, std::remove_reference_t<T>::rank>>;
        { t.pre } -> std::convertible_to<real>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_hydro_conserved_c = requires(T t) {
        { t.den } -> std::convertible_to<real>;
        { t.mom } -> std::convertible_to<vector_t<real, std::remove_reference_t<T>::rank>>;
        { t.nrg } -> std::convertible_to<real>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_mhd_primitive_c = requires(T t) {
        { t.rho } -> std::convertible_to<real>;
        { t.vel } -> std::convertible_to<vector_t<real, std::remove_reference_t<T>::rank>>;
        { t.pre } -> std::convertible_to<real>;
        { t.mag } -> std::convertible_to<vector_t<real, std::remove_reference_t<T>::rank>>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_mhd_conserved_c = requires(T t) {
        { t.den } -> std::convertible_to<real>;
        { t.mom } -> std::convertible_to<vector_t<real, std::remove_reference_t<T>::rank>>;
        { t.nrg } -> std::convertible_to<real>;
        { t.mag } -> std::convertible_to<vector_t<real, std::remove_reference_t<T>::rank>>;
        { t.chi } -> std::convertible_to<real>;
    };

    // concept defining a state variable - type approach
    template <typename T>
    concept is_any_state_variable_c = is_hydro_primitive_c<T> || is_hydro_conserved_c<T> ||
                                      is_mhd_primitive_c<T> || is_mhd_conserved_c<T>;

    template <typename T>
    concept is_relativistic_c = requires {
        { T::regime } -> std::convertible_to<regime_t>;
        requires T::regime == regime_t::SRHD || T::regime == regime_t::RMHD;
    };

    template <typename T>
    concept is_relativistic_primitive_c =
        (is_hydro_primitive_c<T> || is_mhd_primitive_c<T>) && is_relativistic_c<T>;

    template <typename T>
    concept is_relativistic_conserved_c =
        (is_hydro_conserved_c<T> || is_mhd_conserved_c<T>) && is_relativistic_c<T>;

    template <typename T>
    concept is_srhd_c = is_relativistic_c<T> && T::regime == regime_t::SRHD;

    template <typename T>
    concept is_rmhd_c = is_relativistic_c<T> && T::regime == regime_t::RMHD;

    template <typename T>
    concept is_newtonian_c = requires {
        { T::regime } -> std::convertible_to<regime_t>;
        requires T::regime == regime_t::NEWTONIAN;
    };

    template <typename T>
    concept is_mhd_c = requires {
        { T::regime } -> std::convertible_to<regime_t>;
        requires T::regime == regime_t::MHD || T::regime == regime_t::RMHD;
    };

    template <typename T>
    concept vector_like_c = requires(T vec, size_t i) {
        { vec[i] } -> std::convertible_to<typename T::value_type>;
        { vec.size() } -> std::convertible_to<size_t>;
        { T::rank } -> std::convertible_to<size_t>;
    };

    template <typename T>
    concept field_like_c = requires {
        { T::handle_type };
    };

    // =============================================================================
    // computation protocol concepts
    // =============================================================================

    // helper concepts for detection (prefixed to avoid conflicts with
    // traits.hpp)
    template <typename T>
    concept has_computable_value_type = requires { typename T::value_type; };

    template <typename T>
    concept has_computable_argument_type = requires { typename T::argument_type; };

    template <typename T>
    concept has_computable_rank = requires {
        { T::rank } -> std::convertible_to<std::uint64_t>;
    };

    template <typename T>
    concept already_computable =
        has_computable_value_type<T> && has_computable_argument_type<T> && has_computable_rank<T>;

    // core computable: any callable with explicit type metadata
    template <typename T>
    concept computable = requires {
        typename T::value_type;
        typename T::argument_type;
        { T::rank } -> std::convertible_to<std::uint64_t>;
    } && requires(const T& comp, typename T::argument_type arg) {
        { comp(arg) } -> std::convertible_to<typename T::value_type>;
    };

} // namespace simbi::concepts
