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

#include "compute/math/lazy_expr.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "data/containers/ndarray.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace simbi::helpers {
    template <std::size_t N, typename F>
    void compile_time_for(F&& func);

    template <std::int64_t Start, std::int64_t End, typename Func>
    void compile_time_for(Func&& f);
}   // namespace simbi::helpers

namespace simbi::fp {
    using namespace simbi::nd;
    using namespace simbi::concepts;
    using namespace simbi::expr;

    template <typename Container, typename NewType>
    struct rebind {
        static_assert(
            sizeof(Container) == 0,
            "rebind not specialized for this type"
        );
    };

    // specialize for your specific containers
    template <typename T, std::uint64_t Dims, typename NewType>
    struct rebind<ndarray_t<T, Dims>, NewType> {
        using type = ndarray_t<NewType, Dims>;
    };

    template <typename T, typename NewType>
    struct rebind<std::vector<T>, NewType> {
        using type = std::vector<NewType>;
    };

    template <typename T, size_t N, typename NewType>
    struct rebind<std::array<T, N>, NewType> {
        using type = std::array<NewType, N>;
    };

    template <VectorLike V, typename NewType>
    struct rebind<V, NewType> {
        using type = vector_t<NewType, V::dimensions>;
    };

    template <typename T>
    struct rebind<integer_range_t<T>, T> {
        using type = integer_range_t<T>;
    };

    // helper alias
    template <typename Container, typename NewType>
    using rebind_t = typename rebind<Container, NewType>::type;

    // -------------------------------------------------------------
    // core functional operations
    // -------------------------------------------------------------

    // map: transform each element with a function, returning a new
    // container of the same type
    template <Container C, typename F>
    DUAL constexpr auto map(const C& container, F&& f)
    {
        using T                  = typename C::value_type;
        using result_t           = std::invoke_result_t<F, T>;
        using result_container_t = rebind_t<C, result_t>;
        result_container_t result{};

        for (std::uint64_t ii = 0; ii < container.size(); ++ii) {
            result[ii] = std::invoke(std::forward<F>(f), container[ii]);
        }

        return result;
    }

    // reduce: combine elements using a binary function, with optional
    // initial value
    template <Container C, typename F>
    DUAL constexpr auto
    reduce(const C& container, F&& f, typename C::value_type init = {})
    {
        using value_t  = typename C::value_type;
        value_t result = init;

        for (std::uint64_t ii = 0; ii < container.size(); ++ii) {
            result = std::invoke(std::forward<F>(f), result, container[ii]);
        }

        return result;
    }

    // fold: like reduce but with explicit initial value of possibly
    // different type
    template <Container C, typename F, typename U>
    DUAL constexpr auto fold(const C& container, F&& f, U init)
    {
        U result = init;

        for (std::uint64_t ii = 0; ii < container.size(); ++ii) {
            result = std::invoke(std::forward<F>(f), result, container[ii]);
        }

        return result;
    }

    // zip: combine two containers element-wise with a binary function
    template <Container C1, Container C2, typename F>
    DUAL constexpr auto zip(const C1& container1, const C2& container2, F&& f)
    {
        using T1                 = typename C1::value_type;
        using T2                 = typename C2::value_type;
        using result_t           = std::invoke_result_t<F, T1, T2>;
        using result_container_t = rebind_t<C1, result_t>;

        result_container_t result{};
        const std::uint64_t min_size =
            std::min(container1.size(), container2.size());

        for (std::uint64_t ii = 0; ii < min_size; ++ii) {
            result[ii] =
                std::invoke(std::forward<F>(f), container1[ii], container2[ii]);
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

    // curry: transform a function that takes multiple arguments into a
    // sequence of functions that each take a single argument
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
        for (std::uint64_t i = 0; i < container.size(); ++i) {
            if (std::invoke(std::forward<F>(pred), container[i])) {
                return true;
            }
        }
        return false;
    }

    // general any_of predicate combinator
    template <typename T, typename... Predicates>
    DUAL constexpr bool any_of(const T& value, Predicates&&... preds)
    {
        return (... || (std::invoke(std::forward<Predicates>(preds), value)));
    }

    // all: check if all elements satisfy a predicate
    template <Container C, typename F>
    DUAL constexpr bool all_of(const C& container, F&& pred)
    {
        for (std::uint64_t i = 0; i < container.size(); ++i) {
            if (!std::invoke(std::forward<F>(pred), container[i])) {
                return false;
            }
        }
        return true;
    }

    // all: general predicate combinator
    template <typename T, typename... Predicates>
    DUAL constexpr bool all_of(const T& value, Predicates&&... preds)
    {
        return (... && (std::invoke(std::forward<Predicates>(preds), value)));
    }

    // filter: create a new container with elements that satisfy a predicate
    template <Container C, typename F>
    DUAL auto filter(const C& container, F&& pred)
    {
        C result{};
        std::uint64_t result_idx = 0;

        for (std::uint64_t i = 0; i < container.size(); ++i) {
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

    // create integer ranges
    constexpr auto range(std::uint64_t end)
    {
        return integer_range_t<std::uint64_t>{end};
    }

    constexpr auto range(std::uint64_t start, std::uint64_t end)
    {
        return integer_range_t<std::uint64_t>{start, end};
    }

    constexpr auto
    range(std::uint64_t start, std::uint64_t end, std::uint64_t step)
    {
        return integer_range_t<std::uint64_t>{start, end, step};
    }

    // infinite sequences with custom generators
    template <typename Generator>
    constexpr auto generate(Generator&& gen)
    {
        return generator_range_t<Generator>{std::forward<Generator>(gen)};
    }

    template <std::uint64_t Current, std::uint64_t End, typename F>
    constexpr auto build_expression_impl(F&& f)
    {
        if constexpr (Current >= End) {
            static_assert(Current < End, "Empty range not supported");
        }
        else if constexpr (Current == End - 1) {
            // last element - just return f(Current)
            return f(std::integral_constant<std::uint64_t, Current>{});
        }
        else {
            // recursive case - f(Current) + rest
            return f(std::integral_constant<std::uint64_t, Current>{}) +
                   build_expression_impl<Current + 1, End>(std::forward<F>(f));
        }
    }

    template <std::uint64_t N, typename F>
    constexpr auto build_expression_ct(F&& f)
    {
        return build_expression_impl<0, N>(std::forward<F>(f));
    }

    // template <std::uint64_t N, typename F>
    // constexpr auto accumulate_expression_ct(F&& f)
    // {
    //     // start with the first expression to determine the result type
    //     auto result = f(std::integral_constant<std::uint64_t, 0>{});

    //     helpers::compile_time_for<1, N>([&](auto idx_constant) {
    //         result = result + f(idx_constant);
    //     });

    //     return result;
    // }

    // take first N from any range
    // template <LazyRange R>
    // constexpr auto take(const R& range, std::uint64_t n)
    // {
    //     return take_range_t<R>{range, n};
    // }

}   // namespace simbi::fp

#endif
