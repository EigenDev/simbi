
/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            functional.hpp
 *  * @brief           functional programming primitives for iterable types
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-03-01
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  */

#ifndef FUNCTIONAL_PROGRAMMING_HPP
#define FUNCTIONAL_PROGRAMMING_HPP

#include "build_options.hpp"
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace simbi::fp {

    // concepts to constrain function inputs

    // concept for iterable types that support begin/end
    template <typename T>
    concept Iterable = requires(T t) {
        { std::begin(t) } -> std::input_iterator;
        { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
    };

    // concept for containers that support indexing and size
    template <typename T>
    concept Indexable = requires(T t, size_type i) {
        { t[i] } -> std::convertible_to<typename T::value_type>;
        { t.size() } -> std::convertible_to<size_type>;
    };

    // concept for containers that support both iterating and indexing
    template <typename T>
    concept Container = Iterable<T> && Indexable<T>;

    // -------------------------------------------------------------
    // core functional operations
    // -------------------------------------------------------------

    // invoke: a cpu/gpu compatible implementation that works like
    // std::invoke
    // template <typename F, typename... Args>
    // DUAL constexpr auto invoke(F&& f, Args&&... args)
    // {
    //     if constexpr (std::is_member_function_pointer_v<
    //                       std::remove_cvref_t<F>>) {
    //         return (std::forward<F>(f)->*(std::forward<Args>(args)...));
    //     }
    //     else {
    //         return std::forward<F>(f)(std::forward<Args>(args)...);
    //     }
    // }

    // map: transform each element with a function, returning a new container of
    // the same type
    template <Container C, typename F>
    DUAL constexpr auto map(const C& container, F&& f)
    {
        using result_t = std::invoke_result_t<F, typename C::value_type>;
        C result{};

        for (size_type i = 0; i < container.size(); ++i) {
            result[i] = std::invoke(std::forward<F>(f), container[i]);
        }

        return result;
    }

    // reduce: combine elements using a binary function, with optional initial
    // value
    template <Container C, typename F>
    DUAL constexpr auto
    reduce(const C& container, F&& f, typename C::value_type init = {})
    {
        using value_t  = typename C::value_type;
        value_t result = init;

        for (size_type i = 0; i < container.size(); ++i) {
            result = std::invoke(std::forward<F>(f), result, container[i]);
        }

        return result;
    }

    // fold: like reduce but with explicit initial value of possibly different
    // type
    template <Container C, typename F, typename U>
    DUAL constexpr auto fold(const C& container, F&& f, U init)
    {
        U result = init;

        for (size_type i = 0; i < container.size(); ++i) {
            result = std::invoke(std::forward<F>(f), result, container[i]);
        }

        return result;
    }

    // zip: combine two containers element-wise with a binary function
    template <Container C1, Container C2, typename F>
    DUAL constexpr auto zip(const C1& container1, const C2& container2, F&& f)
    {
        using T1       = typename C1::value_type;
        using T2       = typename C2::value_type;
        using result_t = std::invoke_result_t<F, T1, T2>;

        // use the type of the first container as the result container type
        // but with the result_t as its value type
        using result_container_t = std::remove_cvref_t<C1>;

        result_container_t result{};
        const size_type min_size =
            std::min(container1.size(), container2.size());

        for (size_type i = 0; i < min_size; ++i) {
            result[i] =
                std::invoke(std::forward<F>(f), container1[i], container2[i]);
        }

        return result;
    }

    // -------------------------------------------------------------
    // higher-order functions
    // -------------------------------------------------------------

    // compose: create a function that applies g after f
    template <typename F, typename G>
    DUAL constexpr auto compose(F&& f, G&& g)
    {
        return [f = std::forward<F>(f),
                g = std::forward<G>(g)]<typename... Args>(Args&&... args) {
            return g(f(std::forward<Args>(args)...));
        };
    }

    // curry: transform a function that takes multiple arguments into a sequence
    // of functions that each take a single argument
    template <typename F>
    DUAL constexpr auto curry(F&& f)
    {
        return [f = std::forward<F>(f)]<typename T>(T&& t) {
            return [f = std::forward<F>(f),
                    t = std::forward<T>(t)]<typename... Args>(Args&&... args) {
                return f(t, std::forward<Args>(args)...);
            };
        };
    }

    // -------------------------------------------------------------
    // common functional operations
    // -------------------------------------------------------------

    // sum: add all elements in a container
    template <Container C>
    DUAL constexpr auto sum(const C& container)
    {
        return reduce(container, std::plus<>{});
    }

    // product: multiply all elements in a container
    template <Container C>
    DUAL constexpr auto product(const C& container)
    {
        using value_t = typename C::value_type;
        return reduce(container, std::multiplies<>{}, value_t{1});
    }

    // any: check if any element satisfies a predicate
    template <Container C, typename F>
    DUAL constexpr bool any_of(const C& container, F&& pred)
    {
        for (size_type i = 0; i < container.size(); ++i) {
            if (std::invoke(std::forward<F>(pred), container[i])) {
                return true;
            }
        }
        return false;
    }

    // all: check if all elements satisfy a predicate
    template <Container C, typename F>
    DUAL constexpr bool all_of(const C& container, F&& pred)
    {
        for (size_type i = 0; i < container.size(); ++i) {
            if (!std::invoke(std::forward<F>(pred), container[i])) {
                return false;
            }
        }
        return true;
    }

    // filter: create a new container with elements that satisfy a predicate
    template <Container C, typename F>
    DUAL auto filter(const C& container, F&& pred)
    {
        C result{};
        size_type result_idx = 0;

        for (size_type i = 0; i < container.size(); ++i) {
            if (std::invoke(std::forward<F>(pred), container[i])) {
                result[result_idx++] = container[i];
            }
        }

        // Note: For fixed-size containers like Vector, this assumes
        // we're only using the first result_idx elements
        return result;
    }

    // pipeline operator for functional composition
    template <typename T, typename F>
    DUAL constexpr auto operator|(T&& value, F&& f)
        -> decltype(std::invoke(std::forward<F>(f), std::forward<T>(value)))
    {
        return std::invoke(std::forward<F>(f), std::forward<T>(value));
    }
}   // namespace simbi::fp

#endif
