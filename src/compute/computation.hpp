#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "domain/domain.hpp"
#include "functional/fp.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace simbi {

    // forward declarations
    template <std::uint64_t Dims, typename F>
    struct computation_t;

    namespace detail {
        // extract return type from callable
        template <typename F, std::uint64_t Dims>
        using computation_value_t = std::invoke_result_t<F, coordinate_t<Dims>>;

        // check if computation returns reference (indicates it wraps storage)
        template <typename F, std::uint64_t Dims>
        struct returns_reference {
            template <typename C>
            static auto test(int)
                -> std::is_reference<decltype(std::declval<C>()(
                    std::declval<coordinate_t<Dims>>()
                ))>;

            template <typename>
            static std::false_type test(...);

            static constexpr bool value = decltype(test<F>(0))::value;
        };

        template <typename F, std::uint64_t Dims>
        inline constexpr bool returns_reference_v =
            returns_reference<F, Dims>::value;
    }   // namespace detail

    // pure lazy computation graph - no memory, no device knowledge
    // immutable, composable, device-agnostic
    template <std::uint64_t Dims, typename F>
    struct computation_t {
        using value_type = detail::computation_value_t<F, Dims>;
        static constexpr std::uint64_t dimensions = Dims;

        F func;
        domain_t<Dims> domain_;

        // construction
        constexpr computation_t(F f, domain_t<Dims> d)
            : func(std::move(f)), domain_(d)
        {
        }

        // domain query
        constexpr const domain_t<Dims>& domain() const { return domain_; }

        // evaluation - this is what makes it a computation
        constexpr DUAL decltype(auto)
        operator()(const coordinate_t<Dims>& coord) const
        {
            return func(coord);
        }

        constexpr DUAL decltype(auto)
        operator[](const coordinate_t<Dims>& coord) const
        {
            return func(coord);
        }

        // map: transform values
        // (f ∘ g)(x) = f(g(x))
        template <typename UnaryOp>
        auto map(UnaryOp op) const
        {
            auto new_func = fp::compose(op, func);
            return computation_t<Dims, decltype(new_func)>{
              std::move(new_func),
              domain_
            };
        }

        // enum_map: transform with coordinates
        // f(coord, value)
        template <typename BinaryOp>
        auto enum_map(BinaryOp op) const
        {
            auto coord_func = fp::identity;
            auto value_func = func;
            auto enum_func  = fp::zip(coord_func, value_func, op);
            return computation_t<Dims, decltype(enum_func)>{
              std::move(enum_func),
              domain_
            };
        }

        // coord_map: transform coordinates before evaluation
        // g(f(coord))
        template <typename UnaryOp>
        auto coord_map(UnaryOp op) const
        {
            auto coord_func  = fp::identity;
            auto mapped_func = fp::compose(op, coord_func);
            return computation_t<Dims, decltype(mapped_func)>{
              std::move(mapped_func),
              domain_
            };
        }

        // zip: combine two computations element-wise
        // binary_op(f(coord), g(coord))
        template <typename G, typename BinaryOp>
        auto zip(const computation_t<Dims, G>& other, BinaryOp op) const
        {
            using namespace domain_algebra;
            auto combined_domain = intersection(domain_, other.domain_);
            auto zipped_func     = fp::zip(func, other.func, op);

            return computation_t<Dims, decltype(zipped_func)>{
              std::move(zipped_func),
              combined_domain
            };
        }

        // slice: restrict to subdomain with coordinate offset
        // evaluates func(coord + offset)
        auto operator[](const domain_t<Dims>& subdomain) const
        {
            auto offset_func =
                fp::compose(func, fp::partial(fp::add_op, subdomain.start));
            auto local_domain = make_domain(subdomain.shape());

            return computation_t<Dims, decltype(offset_func)>{
              std::move(offset_func),
              local_domain
            };
        }

        // at: restrict domain without coordinate transformation
        auto at(const domain_t<Dims>& subdomain) const
        {
            using namespace domain_algebra;
            auto restricted_domain = intersection(domain_, subdomain);
            return computation_t{func, restricted_domain};
        }

        // insert: overlay another computation conditionally
        // if coord in overlay.domain then overlay(coord) else this(coord)
        template <typename G>
        auto insert(const computation_t<Dims, G>& overlay) const
        {
            using namespace domain_algebra;
            auto union_domain      = union_of(domain_, overlay.domain_);
            auto overlay_predicate = fp::contains_op(overlay.domain_);
            auto insert_func =
                fp::select(overlay_predicate, overlay.func, func);

            return computation_t<Dims, decltype(insert_func)>{
              std::move(insert_func),
              union_domain
            };
        }

        // select: conditional computation
        // pred(coord) ? true_comp(coord) : false_comp(coord)
        template <typename Pred, typename TrueF, typename FalseF>
        static auto select(
            Pred pred,
            const computation_t<Dims, TrueF>& true_comp,
            const computation_t<Dims, FalseF>& false_comp
        )
        {
            using namespace domain_algebra;
            auto combined_domain =
                union_of(true_comp.domain_, false_comp.domain_);
            auto select_func =
                fp::select(pred, true_comp.func, false_comp.func);

            return computation_t<Dims, decltype(select_func)>{
              std::move(select_func),
              combined_domain
            };
        }
    };

    // deduction guide
    template <std::uint64_t Dims, typename F>
    computation_t(F, domain_t<Dims>) -> computation_t<Dims, F>;

    // arithmetic operators via zip
    template <std::uint64_t Dims, typename F, typename G>
    auto
    operator+(const computation_t<Dims, F>& a, const computation_t<Dims, G>& b)
    {
        return a.zip(b, fp::add_op);
    }

    template <std::uint64_t Dims, typename F, typename G>
    auto
    operator-(const computation_t<Dims, F>& a, const computation_t<Dims, G>& b)
    {
        return a.zip(b, fp::subtract_op);
    }

    template <std::uint64_t Dims, typename F, typename G>
    auto
    operator*(const computation_t<Dims, F>& a, const computation_t<Dims, G>& b)
    {
        return a.zip(b, fp::multiply_op);
    }

    template <std::uint64_t Dims, typename F, typename G>
    auto
    operator/(const computation_t<Dims, F>& a, const computation_t<Dims, G>& b)
    {
        return a.zip(b, fp::divide_op);
    }

    // scalar arithmetic
    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator*(const computation_t<Dims, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::multiply_op);

        return computation_t<Dims, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator*(Scalar scalar, const computation_t<Dims, F>& comp)
    {
        return comp * scalar;
    }

    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator/(const computation_t<Dims, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::divide_op);

        return computation_t<Dims, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator+(const computation_t<Dims, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::add_op);

        return computation_t<Dims, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator+(Scalar scalar, const computation_t<Dims, F>& comp)
    {
        return comp + scalar;
    }

    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator-(const computation_t<Dims, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::subtract_op);

        return computation_t<Dims, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Dims, typename F, typename Scalar>
    auto operator-(Scalar scalar, const computation_t<Dims, F>& comp)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(scalar_func, comp.func, fp::subtract_op);

        return computation_t<Dims, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    // factory function for creating computations
    template <std::uint64_t Dims, typename F>
    auto computation(const domain_t<Dims>& domain, F&& func)
    {
        return computation_t<Dims, std::decay_t<F>>{
          std::forward<F>(func),
          domain
        };
    }

    // identity computation: returns coordinates
    template <std::uint64_t Dims>
    auto identity(const domain_t<Dims>& domain)
    {
        return computation_t{fp::identity, domain};
    }

    // constant computation: returns same value everywhere
    template <std::uint64_t Dims, typename T>
    auto constant(const domain_t<Dims>& domain, T value)
    {
        return computation_t{fp::constant(value), domain};
    }

}   // namespace simbi

#endif   // COMPUTATION_HPP
