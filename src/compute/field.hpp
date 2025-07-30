#ifndef FIELD_HPP
#define FIELD_HPP

#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "domain/domain.hpp"
#include "execution/executor.hpp"
#include "memory/accessor.hpp"

#include <cstdint>
#include <functional>
#include <type_traits>

namespace simbi {
    using namespace mem;
    template <std::uint64_t Dims, typename Func>
    struct compute_field_t;

    // deduction guides for compute_field_t
    // to make my life easier...
    template <std::uint64_t Dims, typename Func>
    compute_field_t(Func&&, domain_t<Dims>)
        -> compute_field_t<Dims, std::decay_t<Func>>;

    template <typename T, std::uint64_t Dims>
    compute_field_t(accessor_t<T, Dims>, domain_t<Dims>)
        -> compute_field_t<Dims, accessor_t<T, Dims>>;

    template <typename T, std::uint64_t Dims>
    using field_t = compute_field_t<Dims, accessor_t<T, Dims>>;

    namespace detail {
        // type trait to check if F is accessor_t
        template <typename F>
        struct is_accessor : std::false_type {
        };

        template <typename T, std::uint64_t Dims>
        struct is_accessor<accessor_t<T, Dims>> : std::true_type {
        };

        template <typename F>
        inline constexpr bool is_accessor_v = is_accessor<F>::value;

        // helper to get value type from function
        template <typename F, std::uint64_t Dims>
        using field_value_t = std::invoke_result_t<F, coordinate_t<Dims>>;

        template <typename F, std::uint64_t Dims>
        struct returns_reference {
            template <typename G>
            static auto test(
                int
            ) -> std::is_reference<std::invoke_result_t<G, coordinate_t<Dims>>>;

            template <typename>
            static std::false_type test(...);

            static constexpr bool value = decltype(test<F>(0))::value;
        };

        template <typename F, std::uint64_t Dims>
        inline constexpr bool returns_reference_v =
            returns_reference<F, Dims>::value;

    }   // namespace detail

    // unified field abstraction - everything is a compute_field_t
    template <std::uint64_t Dims, typename Func>
    struct compute_field_t {
        using value_type = detail::field_value_t<Func, Dims>;
        static constexpr std::uint64_t dimensions = Dims;

        Func function;
        domain_t<Dims> domain_;

        // basic queries
        constexpr auto domain() const { return domain_; }
        auto data() const
            requires detail::is_accessor_v<Func>
        {
            return function.data();
        }

        // assignment materialization - triggers lazy evaluation
        template <typename OtherFunc>
        auto operator=(const compute_field_t<Dims, OtherFunc>& source)
        {
            if constexpr (detail::is_accessor_v<Func>) {
                // direct memory storage
                function.commit(source, async::default_executor());
                if (domain_.empty()) {
                    domain_ = source.domain_;
                }
            }
            else if constexpr (detail::returns_reference_v<Func, Dims>) {
                // storage view - can assign directly
                async::default_executor()
                    .for_each(
                        domain_,
                        [this, source](auto coord) {
                            function(coord) = source.function(coord);
                        }
                    )
                    .wait();
            }
            else {
                // pure comp (including computation views) - just copy
                function = source.function;
                domain_  = source.domain_;
            }

            return *this;
        }

        auto clone() const
            requires detail::is_accessor_v<Func>
        {
            // clone accessor
            auto new_accessor = function.clone();
            return compute_field_t<Dims, decltype(new_accessor)>{
              std::move(new_accessor),
              domain_
            };
        }

        // mathematical function evaluation
        value_type operator()(const coordinate_t<Dims>& coord) const
        {
            return function(coord);
        }

        constexpr value_type operator[](coordinate_t<Dims> coord) const
        {
            return function(coord);
        }
        constexpr value_type operator[](std::int64_t idx) const
        {
            return function(domain_.linear_to_coord(idx));
        }

        // slicing - creates lazy view with coordinate transform
        constexpr auto operator[](domain_t<Dims> subdomain)
        {
            if constexpr (detail::is_accessor_v<Func>) {
                auto view_function =
                    [=, this](coordinate_t<Dims> local_coord) -> auto& {
                    return function[subdomain.start + local_coord];
                };
                auto local_domain = make_domain(subdomain.shape());
                return compute_field_t<Dims, decltype(view_function)>{
                  std::move(view_function),
                  local_domain
                };
            }
            else {
                auto view_function =
                    [=, func = function](coordinate_t<Dims> local_coord) {
                        return func(subdomain.start + local_coord);
                    };
                auto local_domain = make_domain(subdomain.shape());
                return compute_field_t<Dims, decltype(view_function)>{
                  std::move(view_function),
                  local_domain
                };
            }
        }

        constexpr auto operator[](domain_t<Dims> subdomain) const
        {
            if constexpr (detail::is_accessor_v<Func>) {
                auto view_function =
                    [=, this](coordinate_t<Dims> local_coord) -> const auto& {
                    return function[subdomain.start + local_coord];
                };
                auto local_domain = make_domain(subdomain.shape());
                return compute_field_t<Dims, decltype(view_function)>{
                  std::move(view_function),
                  local_domain
                };
            }
            else {
                auto view_function =
                    [=, func = function](coordinate_t<Dims> local_coord) {
                        return func(subdomain.start + local_coord);
                    };
                auto local_domain = make_domain(subdomain.shape());
                return compute_field_t<Dims, decltype(view_function)>{
                  std::move(view_function),
                  local_domain
                };
            }
        }

        // map: apply unary / binary operation to every element
        template <typename SomeOp>
        auto map(SomeOp op) const
        {
            if constexpr (std::is_invocable_v<SomeOp, value_type>) {
                auto mapped_function = [my_func = function,
                                        op](coordinate_t<Dims> coord) {
                    return op(my_func(coord));
                };
                return compute_field_t<Dims, decltype(mapped_function)>{
                  std::move(mapped_function),
                  domain_
                };
            }
            else if constexpr (std::is_invocable_v<
                                   SomeOp,
                                   coordinate_t<Dims>,
                                   value_type>) {
                auto mapped_function = [my_func = function,
                                        op](coordinate_t<Dims> coord) {
                    return op(coord, my_func(coord));
                };
                return compute_field_t<Dims, decltype(mapped_function)>{
                  std::move(mapped_function),
                  domain_
                };
            }
            else {
                static_assert(
                    false,
                    "map function must accept either (value) or (coord, value)"
                );
            }
        }

        // zip: combine with another field using binary operation
        template <typename OtherFunc, typename BinaryOp>
        auto
        zip(const compute_field_t<Dims, OtherFunc>& other, BinaryOp op) const
        {
            using namespace domain_algebra;
            auto combined_domain = intersection(domain_, other.domain_);

            auto zipped_func = [f = function, g = other.function, op](
                                   auto coord
                               ) { return op(f(coord), g(coord)); };

            return compute_field_t<Dims, decltype(zipped_func)>{
              std::move(zipped_func),
              combined_domain
            };
        }

        // at: restrict to subdomain (lens)
        auto at(const domain_t<Dims>& subdomain) const
        {
            using namespace domain_algebra;
            auto restricted_domain = intersection(domain_, subdomain);
            return compute_field_t{function, restricted_domain};
        }

        // insert: overlay another field
        template <typename OtherFunc>
        auto insert(const compute_field_t<Dims, OtherFunc> overlay) const
        {
            using namespace domain_algebra;
            auto union_domain  = domain_union(domain_, overlay.domain_);
            auto combined_func = [=, func = function](auto coord) {
                if (overlay.domain_.contains(coord)) {
                    return overlay.function(coord);   // use overlay value
                }
                return func(coord);   // use base value
            };
            return compute_field_t<Dims, decltype(combined_func)>{
              std::move(combined_func),
              union_domain
            };
        }

        template <typename Executor = async::default_executor_t>
        auto
        commit(const Executor& executor = async::default_executor_t{}) const
        {
            if constexpr (detail::is_accessor_v<Func>) {
                return *this;
            }
            else {
                auto result = accessor_t<value_type, Dims>{domain_};
                result.commit(
                    *this,
                    executor
                );   // materialize lazy computation
                return compute_field_t<Dims, accessor_t<value_type, Dims>>{
                  std::move(result),
                  domain_
                };
            }
        }
    };

    // arithmetic operators using functional combinators
    template <std::uint64_t Dims, typename FuncA, typename FuncB>
    auto operator+(
        const compute_field_t<Dims, FuncA>& a,
        const compute_field_t<Dims, FuncB>& b
    )
    {
        return a.zip(b, std::plus{});
    }

    template <std::uint64_t Dims, typename FuncA, typename FuncB>
    auto operator-(
        const compute_field_t<Dims, FuncA>& a,
        const compute_field_t<Dims, FuncB>& b
    )
    {
        return a.zip(b, std::minus{});
    }

    template <std::uint64_t Dims, typename FuncA, typename FuncB>
    auto operator*(
        const compute_field_t<Dims, FuncA>& a,
        const compute_field_t<Dims, FuncB>& b
    )
    {
        return a.zip(b, std::multiplies{});
    }

    template <std::uint64_t Dims, typename FuncA, typename FuncB>
    auto operator/(
        const compute_field_t<Dims, FuncA>& a,
        const compute_field_t<Dims, FuncB>& b
    )
    {
        return a.zip(b, std::divides{});
    }

    // scalar operations using map
    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator*(compute_field_t<Dims, Func> field, Scalar scalar)
    {
        return field.map([=](auto value) { return value * scalar; });
    }

    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator*(Scalar scalar, compute_field_t<Dims, Func> field)
    {
        return field * scalar;
    }

    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator/(compute_field_t<Dims, Func> field, Scalar scalar)
    {
        return field.map([=](auto value) { return value / scalar; });
    }

    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator+(compute_field_t<Dims, Func> field, Scalar scalar)
    {
        return field.map([=](auto value) { return value + scalar; });
    }

    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator+(Scalar scalar, compute_field_t<Dims, Func> field)
    {
        return field + scalar;   // commutative
    }

    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator-(compute_field_t<Dims, Func> field, Scalar scalar)
    {
        return field.map([=](auto value) { return value - scalar; });
    }

    template <std::uint64_t Dims, typename Func, typename Scalar>
    auto operator-(Scalar scalar, compute_field_t<Dims, Func> field)
    {
        return field.map([=](auto value) { return scalar - value; });
    }

    // factory functions for creating fields
    template <std::uint64_t Dims, typename F>
    auto field(const domain_t<Dims>& domain, F&& fn)
    {
        return compute_field_t{std::forward<F>(fn), domain};
    }

    template <typename T, std::uint64_t Dims>
    auto from_numpy_field(T* numpy_data, const iarray<Dims>& shape)
    {
        auto accessor = accessor_t<T, Dims>::from_numpy(numpy_data, shape);
        auto domain   = make_domain(shape);
        return compute_field_t{std::move(accessor), domain};
    }

    // coordinate field factory (useful for testing and initialization...maybe)
    template <std::uint64_t Dims>
    auto identity(const domain_t<Dims>& domain)
    {
        auto coord_func = [](auto coord) { return coord; };
        return compute_field_t{coord_func, domain};
    }

}   // namespace simbi

#endif   // FIELD_HPP
