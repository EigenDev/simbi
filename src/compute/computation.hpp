#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP

#include "base/concepts.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"
#include "grid/algebra.hpp"
#include "grid/domain.hpp"

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace simbi::compute {

    // forward declarations
    template <std::uint64_t Rank, concepts::computable F>
        requires(F::rank == Rank)
    struct computation_t;

    namespace detail {
        // type trait to detect computation nodes
        template <typename T>
        struct is_computation : std::false_type {
        };

        template <std::uint64_t Rank, typename F>
        struct is_computation<computation_t<Rank, F>> : std::true_type {
        };
    }   // namespace detail

    // -------------------------------------------------------------------------
    // bound computation
    // holds a lazy expression and the resource required to execute it
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank, typename Computation, typename Executor>
    struct bound_computation_t {
        Computation comp;
        Executor& exec;
    };

    namespace detail {
        // extract return type from callable
        template <typename F, std::uint64_t Rank>
        using computation_value_t = std::invoke_result_t<F, coordinate_t<Rank>>;

        // check if computation returns reference (indicates it wraps storage)
        template <typename F, std::uint64_t Rank>
        struct returns_reference {
            template <typename C>
            static auto test(int)
                -> std::is_reference<decltype(std::declval<C>()(
                    std::declval<coordinate_t<Rank>>()
                ))>;

            template <typename>
            static std::false_type test(...);

            static constexpr bool value = decltype(test<F>(0))::value;
        };

        template <typename F, std::uint64_t Rank>
        inline constexpr bool returns_reference_v =
            returns_reference<F, Rank>::value;
    }   // namespace detail

    // pure lazy computation graph - no memory, no device knowledge
    // immutable, composable, device-agnostic
    template <std::uint64_t Rank, concepts::computable F>
        requires(F::rank == Rank)
    struct computation_t {
        using value_type                    = typename F::value_type;
        using argument_type                 = typename F::argument_type;
        static constexpr std::uint64_t rank = Rank;

        F func;
        grid::domain_t<Rank> domain_;

        // construction
        constexpr computation_t(F f, grid::domain_t<Rank> d)
            : func(std::move(f)), domain_(d)
        {
        }

        template <typename Executor>
        auto with(Executor& exec) const
        {
            return bound_computation_t<Rank, decltype(*this), Executor>{
              *this,
              exec
            };
        }

        // domain query
        constexpr const grid::domain_t<Rank>& domain() const { return domain_; }

        // evaluation - this is what makes it a computation
        constexpr DUAL value_type operator()(argument_type coord) const
        {
            return func(coord);
        }

        constexpr DUAL value_type operator[](argument_type coord) const
        {
            return func(coord);
        }

        // map: transform values
        // (f ∘ g)(x) = f(g(x))
        template <typename UnaryOp>
        auto map(UnaryOp op) const
        {
            auto new_func = fp::compose(op, func);
            return computation_t<Rank, decltype(new_func)>{
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
            return computation_t<Rank, decltype(enum_func)>{
              std::move(enum_func),
              domain_
            };
        }

        // space_map: transform coordinates before evaluation
        // g(f(coord))
        template <typename UnaryOp>
        auto space_map(UnaryOp op) const
        {
            auto coord_func  = fp::identity;
            auto mapped_func = fp::compose(op, coord_func);
            return computation_t<Rank, decltype(mapped_func)>{
              std::move(mapped_func),
              domain_
            };
        }

        // ---------------------------------------------------------------------
        // remap
        // pre-process the coordinate: f(x) -> f( op(x) )
        // used for: Boundary Conditions (Reflect, Periodic), Stencils
        // ---------------------------------------------------------------------
        template <typename UnaryOp>
        auto remap(UnaryOp op) const
        {
            // compose: field( map(coord) )``
            auto mapped_func = fp::compose(func, std::move(op));

            return computation_t<Rank, decltype(mapped_func)>{
              std::move(mapped_func),
              domain_
            };
        }

        // zip: combine two computations element-wise
        // binary_op(f(coord), g(coord))
        template <concepts::computable G, typename BinaryOp>
            requires(G::rank == Rank)
        auto zip(const computation_t<Rank, G>& other, BinaryOp op) const
        {
            using namespace grid::domain_algebra;
            auto combined_domain = intersection(domain_, other.domain_);
            auto zipped_func     = fp::zip(func, other.func, op);

            return computation_t<Rank, decltype(zipped_func)>{
              std::move(zipped_func),
              combined_domain
            };
        }

        // slice: restrict to subdomain with coordinate offset
        // evaluates func(coord + offset)
        auto operator[](const grid::domain_t<Rank>& subdomain) const
        {
            auto offset_func =
                fp::compose(func, fp::partial(fp::add_op, subdomain.start));
            auto local_domain = make_domain(subdomain.shape());

            return computation_t<Rank, decltype(offset_func)>{
              std::move(offset_func),
              local_domain
            };
        }

        // at: restrict domain without coordinate transformation
        auto at(const grid::domain_t<Rank>& subdomain) const
        {
            using namespace grid::domain_algebra;
            auto restricted_domain = intersection(domain_, subdomain);
            return computation_t{func, restricted_domain};
        }

        // insert: overlay another computation conditionally
        // if coord in overlay.domain then overlay(coord) else this(coord)
        template <typename G>
        auto insert(const computation_t<Rank, G>& overlay) const
        {
            using namespace grid::domain_algebra;
            auto union_domain      = union_of(domain_, overlay.domain_);
            auto overlay_predicate = fp::contains_op(overlay.domain_);
            auto insert_func =
                fp::select(overlay_predicate, overlay.func, func);

            return computation_t<Rank, decltype(insert_func)>{
              std::move(insert_func),
              union_domain
            };
        }

        // select: conditional computation
        // pred(coord) ? true_comp(coord) : false_comp(coord)
        template <typename Pred, typename TrueF, typename FalseF>
        static auto select(
            Pred pred,
            const computation_t<Rank, TrueF>& true_comp,
            const computation_t<Rank, FalseF>& false_comp
        )
        {
            using namespace grid::domain_algebra;
            auto combined_domain =
                union_of(true_comp.domain_, false_comp.domain_);
            auto select_func =
                fp::select(pred, true_comp.func, false_comp.func);

            return computation_t<Rank, decltype(select_func)>{
              std::move(select_func),
              combined_domain
            };
        }
    };

    // deduction guide
    template <std::uint64_t Rank, typename F>
    computation_t(F, grid::domain_t<Rank>) -> computation_t<Rank, F>;

    // arithmetic operators via zip
    template <std::uint64_t Rank, typename F, typename G>
    auto
    operator+(const computation_t<Rank, F>& a, const computation_t<Rank, G>& b)
    {
        return a.zip(b, fp::add_op);
    }

    template <std::uint64_t Rank, typename F, typename G>
    auto
    operator-(const computation_t<Rank, F>& a, const computation_t<Rank, G>& b)
    {
        return a.zip(b, fp::subtract_op);
    }

    template <std::uint64_t Rank, typename F, typename G>
    auto
    operator*(const computation_t<Rank, F>& a, const computation_t<Rank, G>& b)
    {
        return a.zip(b, fp::multiply_op);
    }

    template <std::uint64_t Rank, typename F, typename G>
    auto
    operator/(const computation_t<Rank, F>& a, const computation_t<Rank, G>& b)
    {
        return a.zip(b, fp::divide_op);
    }

    // scalar arithmetic
    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator*(const computation_t<Rank, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::multiply_op);

        return computation_t<Rank, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator*(Scalar scalar, const computation_t<Rank, F>& comp)
    {
        return comp * scalar;
    }

    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator/(const computation_t<Rank, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::divide_op);

        return computation_t<Rank, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator+(const computation_t<Rank, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::add_op);

        return computation_t<Rank, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator+(Scalar scalar, const computation_t<Rank, F>& comp)
    {
        return comp + scalar;
    }

    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator-(const computation_t<Rank, F>& comp, Scalar scalar)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(comp.func, scalar_func, fp::subtract_op);

        return computation_t<Rank, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    template <std::uint64_t Rank, typename F, typename Scalar>
    auto operator-(Scalar scalar, const computation_t<Rank, F>& comp)
    {
        auto scalar_func = fp::constant(scalar);
        auto result_func = fp::zip(scalar_func, comp.func, fp::subtract_op);

        return computation_t<Rank, decltype(result_func)>{
          std::move(result_func),
          comp.domain_
        };
    }

    // factory function for creating computations
    template <std::uint64_t Rank, typename F>
    auto computation(const grid::domain_t<Rank>& domain, F&& func)
    {
        return computation_t<Rank, std::decay_t<F>>{
          std::forward<F>(func),
          domain
        };
    }

    // identity computation: returns coordinates
    template <std::uint64_t Rank>
    auto identity(const grid::domain_t<Rank>& domain)
    {
        // adapter functor satisfying simbi::concepts::computable
        struct identity_functor_t {
            using value_type    = coordinate_t<Rank>;
            using argument_type = coordinate_t<Rank>;
            enum {
                rank = Rank
            };

            DUAL value_type operator()(argument_type coord) const
            {
                return coord;
            }
        };

        return computation_t<Rank, identity_functor_t>{
          identity_functor_t{},
          domain
        };
    }

    // constant computation: returns same value everywhere
    template <std::uint64_t Rank, typename T>
    auto constant(const grid::domain_t<Rank>& domain, T value)
    {
        // adapter functor satisfying simbi::concepts::computable
        struct constant_functor_t {
            using value_type    = T;
            using argument_type = coordinate_t<Rank>;
            enum {
                rank = Rank
            };

            T v;

            constant_functor_t() = default;
            explicit constant_functor_t(T vv) : v(std::move(vv)) {}

            DUAL value_type operator()(argument_type) const { return v; }
        };

        return computation_t<Rank, constant_functor_t>{
          constant_functor_t{std::move(value)},
          domain
        };
    }

    // -------------------------------------------------------------------------
    // view lifting factory
    // converts a view (or field) into an identity computation
    // u -> f(x) = u(x)
    // -------------------------------------------------------------------------

    // trait to check for domain() member function
    template <typename T>
    concept has_domain_method = requires(const T& t) {
        { t.domain() } -> std::same_as<const grid::domain_t<T::rank>&>;
    };

    // overload for objects that know their own domain (e.g. field_t,
    // field_view_t)
    template <typename View>
        requires has_domain_method<View>
    auto computation(const View& v)
    {
        // construct computation with view's domain and the view itself as the
        // functor
        return computation(v.domain(), v);
    }

    template <typename T>
    inline constexpr bool is_computation_v =
        detail::is_computation<std::decay_t<T>>::value;

}   // namespace simbi::compute

#endif   // COMPUTATION_HPP
