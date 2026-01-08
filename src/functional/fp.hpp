#ifndef FP_TOOLKIT_HPP
#define FP_TOOLKIT_HPP

#include "compat.hpp"

#include <cstdint>
#include <tuple>

namespace simbi::grid {
    // domain: pure topology defined by integer vectors
    // represents the half-open interval [start, end)
    template <std::uint64_t Rank>
    struct domain_t;
} // namespace simbi::grid

namespace simbi::fp {

    template <typename T, std::uint64_t Rank>
    struct vector_t;

    // =========================================================================
    // lambda lifting
    //
    // provides a wrapper to make __device__ lambdas more robust for nvcc's
    // type deduction by giving them a "cleaner" type.
    // =========================================================================

    template <typename L>
    struct lifted_lambda_t
    {
        mutable L lambda;

        DUAL lifted_lambda_t(L l) : lambda(std::move(l)) {}

        template <typename... Args>
        constexpr DUAL auto operator()(Args&&... args) const
        {
            return lambda(std::forward<Args>(args)...);
        }
    };

    template <typename L>
    constexpr DUAL auto lift(L lambda)
    {
        return lifted_lambda_t<L>{lambda};
    }

    // =========================================================================
    // common device functors
    // =========================================================================

    // pair creation
    struct make_pair_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A a, B b) const
        {
            return std::make_pair(a, b);
        }
    };

    // scalar multiplication
    template <typename S>
    struct scalar_multiply_t
    {
        S scalar;

        template <typename T>
        constexpr DUAL auto operator()(T x) const
        {
            return x * scalar;
        }
    };

    // scalar addition
    template <typename S>
    struct scalar_add_t
    {
        S scalar;

        template <typename T>
        constexpr DUAL auto operator()(T x) const
        {
            return x + scalar;
        }
    };

    // zero initializer
    struct zero_t
    {
        template <typename T>
        constexpr DUAL T operator()(const T&) const
        {
            return T{};
        }
    };

    // factory functions
    inline constexpr make_pair_t make_pair_func{};
    inline constexpr zero_t      zero_func{};

    template <typename S>
    constexpr auto scalar_multiply(S scalar)
    {
        return scalar_multiply_t<S>{scalar};
    }

    template <typename S>
    constexpr auto scalar_add(S scalar)
    {
        return scalar_add_t<S>{scalar};
    }

    // =========================================================================
    // core composition
    // =========================================================================

    template <typename F, typename G>
    struct compose_t
    {
        F f;
        G g;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const
        {
            return f(g(std::forward<Arg>(arg)));
        }
    };

    template <typename F, typename G>
    DUAL compose_t<F, G> compose(F f, G g)
    {
        return {f, g};
    }

    // variadic compose: base case
    template <typename F>
    constexpr DUAL auto compose(F f)
    {
        return f;
    }

    // variadic compose: recursive case
    template <typename F, typename G, typename... Rest>
    constexpr DUAL auto compose(F f, G g, Rest... rest)
    {
        if constexpr (sizeof...(Rest) == 0) {
            return compose_t<F, G>{f, g};
        }
        else {
            return compose_t<F, decltype(compose(g, rest...))>{f, compose(g, rest...)};
        }
    }

    // =========================================================================
    // partial application
    // =========================================================================

    template <typename F, typename... BoundArgs>
    struct partial_t
    {
        F                        f;
        std::tuple<BoundArgs...> bound_args;

        template <typename... FreeArgs>
        constexpr DUAL decltype(auto) operator()(FreeArgs&&... free_args) const
        {
            return std::apply(
                [&](auto&&... bound) {
                    return f(
                        std::forward<decltype(bound)>(bound)...,
                        std::forward<FreeArgs>(free_args)...
                    );
                },
                bound_args
            );
        }
    };

    template <typename F, typename... BoundArgs>
    constexpr DUAL auto partial(F f, BoundArgs... args)
    {
        return partial_t<F, BoundArgs...>{f, std::make_tuple(args...)};
    }

    // =========================================================================
    // conditional selection
    // =========================================================================

    template <typename Pred, typename A, typename B>
    struct select_t
    {
        Pred pred;
        A    a;
        B    b;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const
        {
            return pred(arg) ? a(arg) : b(arg);
        }
    };

    template <typename Pred, typename A, typename B>
    constexpr DUAL auto select(Pred pred, A a, B b)
    {
        return select_t<Pred, A, B>{pred, a, b};
    }

    // =========================================================================
    // zip operations
    // =========================================================================

    template <typename F, typename G, typename BinaryOp>
    struct zip_t
    {
        F        f;
        G        g;
        BinaryOp op;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const
        {
            return op(f(arg), g(arg));
        }
    };

    template <typename F, typename G, typename BinaryOp>
    constexpr DUAL auto zip(F f, G g, BinaryOp op)
    {
        return zip_t<F, G, BinaryOp>{f, g, op};
    }

    // =========================================================================
    // basic operators
    // =========================================================================

    struct add_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) + std::forward<B>(b);
        }
    };

    struct subtract_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) - std::forward<B>(b);
        }
    };

    struct multiply_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) * std::forward<B>(b);
        }
    };

    struct divide_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) / std::forward<B>(b);
        }
    };

    struct min_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return (a < b) ? std::forward<A>(a) : std::forward<B>(b);
        }
    };

    struct max_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return (a > b) ? std::forward<A>(a) : std::forward<B>(b);
        }
    };

    struct abs_op_t
    {
        template <typename T>
        constexpr DUAL auto operator()(T&& x) const
        {
            return (x < T{}) ? -std::forward<T>(x) : std::forward<T>(x);
        }
    };

    struct negate_op_t
    {
        template <typename T>
        constexpr DUAL auto operator()(T&& x) const
        {
            return -std::forward<T>(x);
        }
    };

    struct average_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return (std::forward<A>(a) + std::forward<B>(b)) * 0.5;
        }
    };

    inline constexpr add_op_t      add_op{};
    inline constexpr subtract_op_t subtract_op{};
    inline constexpr multiply_op_t multiply_op{};
    inline constexpr divide_op_t   divide_op{};
    inline constexpr min_op_t      min_op{};
    inline constexpr max_op_t      max_op{};
    inline constexpr abs_op_t      abs_op{};
    inline constexpr negate_op_t   negate_op{};
    inline constexpr average_op_t  average_op{};

    // =========================================================================
    // coordinate transforms
    // =========================================================================

    template <std::uint64_t Rank>
    struct offset_transform_t
    {
        vector_t<std::int64_t, Rank> offset;

        template <typename Coord>
        constexpr DUAL auto operator()(Coord&& coord) const
        {
            return std::forward<Coord>(coord) + offset;
        }
    };

    template <std::uint64_t Rank>
    constexpr auto offset_transform(vector_t<std::int64_t, Rank> offset)
    {
        return offset_transform_t<Rank>{offset};
    }

    // =========================================================================
    // domain predicates
    // =========================================================================

    template <std::uint64_t Rank>
    struct domain_predicate_t
    {
        grid::domain_t<Rank> domain;

        template <typename Coord>
        constexpr DUAL bool operator()(Coord&& coord) const
        {
            return domain.contains(std::forward<Coord>(coord));
        }
    };

    template <std::uint64_t Rank>
    constexpr auto domain_predicate(grid::domain_t<Rank> domain)
    {
        return domain_predicate_t<Rank>{domain};
    }

    template <std::uint64_t Rank>
    struct contains_op_t
    {
        grid::domain_t<Rank> domain;

        template <typename Coord>
        constexpr DUAL bool operator()(Coord&& coord) const
        {
            return domain.contains(std::forward<Coord>(coord));
        }
    };

    template <std::uint64_t Rank>
    constexpr auto contains_op(grid::domain_t<Rank> domain)
    {
        return contains_op_t<Rank>{domain};
    }

    // =========================================================================
    // utilities
    // =========================================================================

    struct identity_t
    {
        template <typename T>
        constexpr DUAL auto operator()(T&& x) const
        {
            return std::forward<T>(x);
        }
    };

    template <typename T>
    struct constant_t
    {
        T value;

        template <typename U>
        constexpr DUAL const T& operator()(U&&) const
        {
            return value;
        }
    };

    inline constexpr identity_t identity{};

    template <typename T>
    constexpr auto constant(T value)
    {
        return constant_t<T>{std::move(value)};
    }

    // =========================================================================
    // folds and reductions (simplified)
    // =========================================================================

    template <typename Range, typename BinaryOp>
    constexpr DUAL auto fold(Range&& range, BinaryOp op)
    {
        auto it = std::begin(range);
        if (it == std::end(range)) {
            return typename std::iterator_traits<decltype(it)>::value_type{};
        }

        auto result = *it;
        ++it;
        for (; it != std::end(range); ++it) {
            result = op(result, *it);
        }
        return result;
    }

    template <typename Range, typename T, typename BinaryOp>
    constexpr DUAL auto reduce(Range&& range, T init, BinaryOp op)
    {
        auto result = init;
        for (const auto& item : range) {
            result = op(result, item);
        }
        return result;
    }

} // namespace simbi::fp

#endif // FP_TOOLKIT_HPP
