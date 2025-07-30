#ifndef SIMBI_CONCEPTS_HPP
#define SIMBI_CONCEPTS_HPP

#include "config.hpp"
#include "utility/enums.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace simbi {
    template <typename T, std::uint64_t Dims>
    struct vector_t;
}   // namespace simbi

namespace simbi::concepts {
    // =============================================================================
    // Core Concepts
    // =============================================================================
    namespace sim_type {
        template <Regime R>
        concept MHD = R == Regime::RMHD;

        template <Regime R>
        concept Relativistic = R == Regime::SRHD || R == Regime::RMHD;

        template <Regime R>
        concept Newtonian = R == Regime::NEWTONIAN;
    }   // namespace sim_type

    template <typename T>
    concept Arithmetic = std::integral<T> || std::floating_point<T>;

    template <typename F, std::uint64_t Dims>
    concept ArrayFunction = requires(F f, std::array<size_t, Dims> point) {
        { f(point) };
    };

    template <std::uint64_t Dims>
    concept valid_dimension = (Dims >= 1 && Dims <= 3);

    // concept for iterable types that support begin/end
    template <typename T>
    concept Iterable = requires(T t) {
        { std::begin(t) } -> std::input_iterator;
        { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
    };

    // concept for containers that support indexing and size
    template <typename T>
    concept Indexable = requires(T t, std::uint64_t i) {
        { t[i] } -> std::convertible_to<typename T::value_type>;
        { t.size() } -> std::convertible_to<std::uint64_t>;
    };

    // concept for containers that support both iterating and indexing
    template <typename T>
    concept Container = Iterable<T> && Indexable<T>;

    // concept defining a state variable - structural approach
    template <typename T>
    concept is_hydro_primitive_c = requires(T t) {
        { t.rho } -> std::convertible_to<real>;
        {
            t.vel
        } -> std::convertible_to<
            vector_t<real, std::remove_reference_t<T>::dimensions>>;
        { t.pre } -> std::convertible_to<real>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_hydro_conserved_c = requires(T t) {
        { t.den } -> std::convertible_to<real>;
        {
            t.mom
        } -> std::convertible_to<
            vector_t<real, std::remove_reference_t<T>::dimensions>>;
        { t.nrg } -> std::convertible_to<real>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_mhd_primitive_c = requires(T t) {
        { t.rho } -> std::convertible_to<real>;
        {
            t.vel
        } -> std::convertible_to<
            vector_t<real, std::remove_reference_t<T>::dimensions>>;
        { t.pre } -> std::convertible_to<real>;
        {
            t.mag
        } -> std::convertible_to<
            vector_t<real, std::remove_reference_t<T>::dimensions>>;
        { t.chi } -> std::convertible_to<real>;
    };

    template <typename T>
    concept is_mhd_conserved_c = requires(T t) {
        { t.den } -> std::convertible_to<real>;
        {
            t.mom
        } -> std::convertible_to<
            vector_t<real, std::remove_reference_t<T>::dimensions>>;
        { t.nrg } -> std::convertible_to<real>;
        {
            t.mag
        } -> std::convertible_to<
            vector_t<real, std::remove_reference_t<T>::dimensions>>;
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

    template <typename T>
    concept is_newtonian_c = requires {
        { T::regime } -> std::convertible_to<Regime>;
        requires T::regime == Regime::NEWTONIAN;
    };

    template <typename T>
    concept is_mhd_c = requires {
        { T::regime } -> std::convertible_to<Regime>;
        requires T::regime == Regime::MHD || T::regime == Regime::RMHD;
    };

    template <typename T>
    concept VectorLike = requires(T vec, size_t i) {
        { vec[i] } -> std::convertible_to<typename T::value_type>;
        { vec.size() } -> std::convertible_to<size_t>;
        { T::dimensions } -> std::convertible_to<size_t>;
    };

    // base lazy range concept
    template <typename T>
    concept LazyRange = requires(T t, std::uint64_t i) {
        { t[i] } -> std::convertible_to<typename T::value_type>;
        { t.size() } -> std::convertible_to<std::uint64_t>;
        typename T::value_type;
    };

    // core expression concept - defines what makes something an expression
    template <typename T>
    concept Expression = requires(const T& t) {
        { t.size() } -> std::convertible_to<std::uint64_t>;
        { t.realize() };   // all expressions must be materializable
    };

}   // namespace simbi::concepts

#endif
