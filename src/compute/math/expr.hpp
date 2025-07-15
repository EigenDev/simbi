#ifndef SIMBI_EXPRESSIONS_HPP
#define SIMBI_EXPRESSIONS_HPP

#include "compute/functional/fp.hpp"
#include "compute/math/domain.hpp"
#include "config.hpp"
#include "core/base/stencil_view.hpp"
#include "core/utility/enums.hpp"
#include "data/containers/vector.hpp"
#include "physics/hydro/solvers/hllc.hpp"
#include "physics/hydro/solvers/hlld.hpp"
#include "physics/hydro/solvers/hlle.hpp"
#include "system/mesh/mesh_ops.hpp"
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace simbi {
    // forward declarations
    template <std::uint64_t Dims>
    struct domain_t;

    template <typename T, std::uint64_t Dims>
    struct field_t;
}   // namespace simbi

namespace simbi::expr {
    using namespace base::stencils;
    using namespace simbi::set_ops;

    template <typename T>
    concept expression_operation =
        requires { typename T::is_expression_operation; };

    // =================================================================
    // core expression interface - crtp base
    // =================================================================

    template <typename Derived>
    struct expression_t {
        constexpr const Derived& derived() const
        {
            return static_cast<const Derived&>(*this);
        }

        // core evaluation interface
        template <typename Coord>
        constexpr auto operator()(Coord coord) const
        {
            return derived().eval(coord);
        }

        // domain access
        constexpr auto domain() const { return derived().domain(); }

        // mathematical composition - returns new expression types
        template <typename Other>
        constexpr auto operator+(Other&& other) &&
        {
            return make_addition(
                std::move(derived()),
                std::forward<Other>(other)
            );
        }

        template <typename Other>
        constexpr auto operator-(Other&& other) &&
        {
            return make_subtraction(
                std::move(derived()),
                std::forward<Other>(other)
            );
        }

        template <typename Scalar>
        constexpr auto operator*(Scalar scalar) &&
            requires std::is_arithmetic_v<Scalar>
        {
            return make_scaling(std::move(derived()), scalar);
        }
    };

    // =================================================================
    // domain operation expression
    // =================================================================

    template <typename Domain, typename Operation>
    struct domain_operation_expr_t
        : expression_t<domain_operation_expr_t<Domain, Operation>> {
        Domain domain_;
        Operation operation_;

        constexpr domain_operation_expr_t(Domain domain, Operation op)
            : domain_(std::move(domain)), operation_(std::move(op))
        {
        }

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            return operation_(coord);
        }

        constexpr auto domain() const { return domain_; }
    };

    // =================================================================
    // fused expression
    // =================================================================

    template <typename Domain, typename... Operations>
    struct fused_expr_t : expression_t<fused_expr_t<Domain, Operations...>> {
        Domain domain_;
        std::tuple<Operations...> operations_;

        constexpr fused_expr_t(Domain domain, Operations... ops)
            : domain_(std::move(domain)), operations_(std::move(ops)...)
        {
        }

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            return std::apply(
                [coord](const auto&... ops) {
                    // fold expression - single evaluation pass
                    return (ops(coord) + ...);
                },
                operations_
            );
        }

        constexpr auto domain() const { return domain_; }
    };

    // =================================================================
    // field expression - data access
    // =================================================================

    template <typename Field>
    struct field_expr_t : expression_t<field_expr_t<Field>> {
        const Field& field_;

        constexpr explicit field_expr_t(const Field& field) : field_(field) {}

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            return field_[coord];
        }

        constexpr auto domain() const { return field_.domain(); }
    };

    // =================================================================
    // map expression - coordinate-based transformations
    // =================================================================

    template <typename Domain, typename MapFunc>
    struct map_expr_t : expression_t<map_expr_t<Domain, MapFunc>> {
        using expression_operation = void;
        Domain domain_;
        MapFunc func_;

        constexpr map_expr_t(Domain domain, MapFunc func)
            : domain_(std::move(domain)), func_(std::move(func))
        {
        }

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            if constexpr (std::is_invocable_v<MapFunc, Coord>) {
                // function only wants coordinate
                return func_(coord);
            }
            else {
                // this shouldn't happen for pure map operations
                static_assert(
                    std::is_invocable_v<MapFunc, Coord>,
                    "map function must be callable with coordinate"
                );
            }
        }

        constexpr auto domain() const { return domain_; }
    };

    // domain expression
    template <std::uint64_t Dims>
    struct domain_expr_t : expression_t<domain_expr_t<Dims>> {
        domain_t<Dims> domain_;

        explicit domain_expr_t(domain_t<Dims> dom) : domain_(dom) {}

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            return coord;   // pass through coordinates
        }

        constexpr auto domain() const { return domain_; }

        // pipeline operator for domain | map(...) syntax
        template <typename MapFunc>
        constexpr auto operator|(MapFunc&& func) const
        {
            return map_expr_t<domain_t<Dims>, std::decay_t<MapFunc>>(
                domain_,
                std::forward<MapFunc>(func)
            );
        }
    };

    // =================================================================
    // binary expressions - general composition
    // =================================================================

    template <typename Left, typename Right, typename BinaryOp>
    struct binary_expr_t : expression_t<binary_expr_t<Left, Right, BinaryOp>> {
        Left left_;
        Right right_;
        BinaryOp op_;

        constexpr binary_expr_t(Left left, Right right, BinaryOp op)
            : left_(std::move(left)),
              right_(std::move(right)),
              op_(std::move(op))
        {
        }

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            return op_(left_.eval(coord), right_.eval(coord));
        }

        constexpr auto domain() const { return left_.domain(); }
    };

    // =================================================================
    // unary expressions
    // =================================================================

    template <typename Input, typename UnaryOp>
    struct unary_expr_t : expression_t<unary_expr_t<Input, UnaryOp>> {
        Input input_;
        UnaryOp op_;

        constexpr unary_expr_t(Input input, UnaryOp op)
            : input_(std::move(input)), op_(std::move(op))
        {
        }

        template <typename Coord>
        constexpr auto eval(Coord coord) const
        {
            return op_(input_.eval(coord));
        }

        constexpr auto domain() const { return input_.domain(); }
    };

    // =================================================================
    // type traits for fusion detection
    // =================================================================

    template <typename T>
    struct is_domain_operation : std::false_type {
    };

    template <typename Domain, typename Operation>
    struct is_domain_operation<domain_operation_expr_t<Domain, Operation>>
        : std::true_type {
    };

    template <typename T>
    constexpr bool is_domain_operation_v = is_domain_operation<T>::value;

    template <typename Left, typename Right>
    constexpr bool can_fuse_v = is_domain_operation_v<std::decay_t<Left>> &&
                                is_domain_operation_v<std::decay_t<Right>>;

    // =================================================================
    // smart constructors with automatic fusion
    // =================================================================
    struct add_op_t {
        template <typename L, typename R>
        constexpr auto operator()(L&& l, R&& r) const
        {
            return std::forward<L>(l) + std::forward<R>(r);
        }
    };

    // addition - detects fusion opportunities automatically
    template <typename Left, typename Right>
    constexpr auto make_addition(Left&& left, Right&& right)
    {
        if constexpr (can_fuse_v<Left, Right>) {
            // simple case: both are domain operations
            auto domain = left.domain();
            return fused_expr_t<
                decltype(domain),
                decltype(left.operation_),
                decltype(right.operation_)>(
                std::move(domain),
                std::move(left.operation_),
                std::move(right.operation_)
            );
        }
        else {
            // everything else becomes a binary expression
            // fusion will happen recursively at evaluation time
            return binary_expr_t<
                std::decay_t<Left>,
                std::decay_t<Right>,
                add_op_t>(
                std::forward<Left>(left),
                std::forward<Right>(right),
                add_op_t{}
            );
        }
    }

    struct sub_op_t {
        template <typename L, typename R>
        constexpr auto operator()(L&& l, R&& r) const
        {
            return std::forward<L>(l) - std::forward<R>(r);
        }
    };

    template <typename Left, typename Right>
    constexpr auto make_subtraction(Left&& left, Right&& right)
    {
        return binary_expr_t<std::decay_t<Left>, std::decay_t<Right>, sub_op_t>(
            std::forward<Left>(left),
            std::forward<Right>(right),
            sub_op_t{}
        );
    }

    template <typename Scalar>
    struct scale_op_t {
        Scalar factor_;
        template <typename T>
        constexpr auto operator()(T&& value) const
        {
            return std::forward<T>(value) * factor_;
        }
    };

    template <typename Input, typename Scalar>
    constexpr auto make_scaling(Input&& input, Scalar scalar)
    {
        return unary_expr_t<std::decay_t<Input>, scale_op_t<Scalar>>(
            std::forward<Input>(input),
            scale_op_t{scalar}
        );
    }

    // =================================================================
    // cfd operation definitions - your existing physics
    // =================================================================

    // flux divergence operation
    template <typename HydroState>
    struct flux_divergence_op_t {
        const HydroState& state_;
        real dt_;

        template <typename Coord>
        auto operator()(Coord coord) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            constexpr auto dims = HydroState::dimensions;

            conserved_t divergence{};
            const auto dv = mesh::volume(coord, state_.mesh);

            // compute divergence using pre-computed fluxes
            for (std::uint64_t dim = 0; dim < dims; ++dim) {
                auto offset     = unit_vectors::logical_offset<dims>(dim);
                auto coord_plus = coord + offset;

                // flux values at left and right faces
                auto fd = active_staggered_domain(state_.mesh.domain, dim);
                auto fl = state_.flux[dim][fd][coord];
                auto fr = state_.flux[dim][fd][coord_plus];

                // geometric face areas
                auto al = mesh::face_area(coord, dim, Dir::W, state_.mesh);
                auto ar = mesh::face_area(coord, dim, Dir::E, state_.mesh);

                // add contribution to divergence
                divergence = divergence + (fr * ar - fl * al) / dv;
            }

            return divergence * (-dt_);
        }
    };

    // gravity source operation
    template <typename HydroState>
    struct gravity_sources_op_t {
        const HydroState& state_;
        real dt_;

        template <typename Coord>
        auto operator()(Coord coord) const
        {
            if (!state_.sources.gravity_source.enabled) {
                return typename HydroState::conserved_t{};
            }

            const auto position     = mesh::centroid(coord, state_.mesh);
            const auto conservative = state_.cons[coord];

            return state_.sources.gravity_source
                .apply(position, conservative, state_.metadata.time, dt_);
        }
    };

    // hydro source operation
    template <typename HydroState>
    struct hydro_sources_op_t {
        const HydroState& state_;
        real dt_;

        template <typename Coord>
        auto operator()(Coord coord) const
        {
            if (!state_.sources.hydro_source.enabled) {
                return typename HydroState::conserved_t{};
            }

            const auto position  = mesh::centroid(coord, state_.mesh);
            const auto primitive = state_.prim[coord];

            return state_.sources.hydro_source.apply(
                position,
                primitive,
                state_.metadata.time,
                state_.metadata.gamma
            );
        }
    };

    // geometric source operation
    template <typename HydroState>
    struct geometric_sources_op_t {
        const HydroState& state_;
        real dt_;

        template <typename Coord>
        auto operator()(Coord coord) const
        {
            // geometric sources only exist for non-cartesian geometries
            if constexpr (HydroState::geometry_t == Geometry::CARTESIAN) {
                return typename HydroState::conserved_t{};
            }
            else {
                const auto p = state_.prim[state_.mesh.domain];
                return mesh::geometric_source_terms(
                           p[coord],
                           coord,
                           state_.mesh,
                           state_.metadata.gamma
                       ) *
                       dt_;
            }
        }
    };

    // flux computation operation
    template <typename HydroState, typename prim_field>
    struct compute_fluxes_op_t {
        const HydroState& state_;
        prim_field prims_;
        std::uint64_t dir_;
        static constexpr auto rec_t    = HydroState::reconstruct_t;
        static constexpr auto solver_t = HydroState::solver_t;

        template <typename Coord>
        auto operator()(Coord coord) const
        {
            // get riemann solver based on compile-time parameters
            constexpr auto solver = get_riemann_solver();
            constexpr auto dims   = HydroState::dimensions;
            const auto plm_theta  = state_.metadata.plm_theta;

            // create stencil for reconstruction around this face
            auto stencil = make_stencil<rec_t>(prims_, coord, dir_);
            auto [left_states, right_states] = stencil.neighbor_values();

            // reconstruct left and right states at face
            auto pl = reconstruct_left<rec_t>(left_states, plm_theta);
            auto pr = reconstruct_right<rec_t>(right_states, plm_theta);

            // normal vector for this dimension
            auto nhat = unit_vectors::ehat<dims>(dir_);

            // face velocity
            auto vface = mesh::face_velocity(coord, dir_, state_.mesh);

            // solve riemann problem
            return solver(
                pl,
                pr,
                nhat,
                vface,
                state_.metadata.gamma,
                state_.metadata.shock_smoother
            );
        }

      private:
        constexpr static auto get_riemann_solver()
        {
            using primitive_t = typename HydroState::primitive_t;
            if constexpr (solver_t == Solver::HLLC) {
                return hydro::hllc_flux<primitive_t>;
            }
            else if constexpr (solver_t == Solver::HLLE) {
                return hydro::hlle_flux<primitive_t>;
            }
            else if constexpr (solver_t == Solver::HLLD) {
                return hydro::rmhd::hlld_flux<primitive_t>;
            }
            else {
                static_assert(false, "unsupported solver type");
            }
        }
    };

    // =================================================================
    // factory functions - clean public interface
    // =================================================================

    // create domain operation expressions
    template <typename HydroState>
    constexpr auto flux_divergence(const HydroState& state, real dt)
    {
        return domain_operation_expr_t<
            decltype(state.mesh.domain),
            flux_divergence_op_t<HydroState>>(
            state.mesh.domain,
            flux_divergence_op_t<HydroState>{state, dt}
        );
    }

    template <typename HydroState>
    constexpr auto gravity_sources(const HydroState& state, real dt)
    {
        return domain_operation_expr_t<
            decltype(state.mesh.domain),
            gravity_sources_op_t<HydroState>>(
            state.mesh.domain,
            gravity_sources_op_t<HydroState>{state, dt}
        );
    }

    template <typename HydroState>
    constexpr auto hydro_sources(const HydroState& state, real dt)
    {
        return domain_operation_expr_t<
            decltype(state.mesh.domain),
            hydro_sources_op_t<HydroState>>(
            state.mesh.domain,
            hydro_sources_op_t<HydroState>{state, dt}
        );
    }

    template <typename HydroState>
    constexpr auto geometric_sources(const HydroState& state, real dt)
    {
        return domain_operation_expr_t<
            decltype(state.mesh.domain),
            geometric_sources_op_t<HydroState>>(
            state.mesh.domain,
            geometric_sources_op_t<HydroState>{state, dt}
        );
    }

    template <typename HydroState>
    constexpr auto compute_fluxes(const HydroState& state, std::uint64_t dir)
    {
        iarray<HydroState::dimensions> expand_amount{0};
        expand_amount[dir]   = 1;
        auto flux_domain     = expand(state.mesh.domain, expand_amount);
        const auto prim_view = state.prim[state.mesh.domain];

        return domain_operation_expr_t<
            decltype(flux_domain),
            compute_fluxes_op_t<HydroState, decltype(prim_view)>>(
            flux_domain,
            compute_fluxes_op_t<HydroState, decltype(prim_view)>{
              state,
              prim_view,
              dir
            }
        );
    }

    // create field expressions
    template <typename Field>
    constexpr auto make_expr(const Field& field)
    {
        return field_expr_t<Field>{field};
    }

    // create domain expressions
    template <std::uint64_t Dims>
    constexpr auto make_expr(domain_t<Dims> domain)
    {
        return domain_expr_t<Dims>{domain};
    }

    // map factory function
    template <typename MapFunc>
    constexpr auto map(MapFunc&& func)
    {
        return [func = std::forward<MapFunc>(func)](auto&& domain_expr) {
            return map_expr_t<
                decltype(domain_expr.domain()),
                std::decay_t<MapFunc>>(domain_expr.domain(), func);
        };
    }

    // =================================================================
    // integration
    // =================================================================
    // terminal operations
    template <typename Target>
    struct commit_to_t {
        Target& target_;

        template <typename Source>
        void operator()(Source&& source) const
        {
            auto domain     = target_.domain();
            auto range_view = fp::range(domain.size());
            auto src_it     = std::begin(source);

            for (auto idx : range_view) {
                auto coord     = domain.linear_to_coord(idx);
                target_[coord] = *src_it++;
            }
        }
    };

    template <typename Target>
    struct accumulate_to_t {
        Target& target_;

        template <typename Source>
        void operator()(Source&& source) const
        {
            auto domain     = target_.domain();
            auto range_view = fp::range(domain.size());
            auto src_it     = std::begin(source);

            for (auto idx : range_view) {
                auto coord = domain.linear_to_coord(idx);
                target_[coord] += *src_it++;
            }
        }
    };

    template <typename Target>
    auto commit_to(Target& target)
    {
        return commit_to_t<Target>{target};
    }

    template <typename Target>
    auto accumulate_to(Target& target)
    {
        return accumulate_to_t<Target>{target};
    }

    // assignment function
    template <typename Field, typename Expr>
    void assign(Field& field, const Expr& expr)
    {
        auto domain = make_domain(field.shape());
        for (auto coord : domain) {
            field[coord] = expr.eval(coord);
        }
    }

    // evaluate expression directly into field
    template <typename Field, typename Expr>
    void evaluate_into(Field& field, const Expr& expr)
    {
        assign(field, expr);
    }

    // enable pipeline syntax on domains
    template <std::uint64_t Dims, expression_operation Op>
    constexpr DUAL auto operator|(domain_t<Dims> domain, Op&& op)
    {
        return make_expr(domain) | std::forward<Op>(op);
    }

    template <std::uint64_t Dims, typename Op>
    constexpr DUAL auto operator|(domain_t<Dims> domain, Op&& op)
        requires(!expr::expression_operation<Op>) && requires { op(domain); }
    {
        return std::forward<Op>(op)(domain);
    }

    // scalar multiplication
    template <typename Expr, typename Scalar>
    constexpr DUAL auto operator*(Scalar scalar, Expr&& expr)
        requires std::is_arithmetic_v<Scalar>
    {
        return std::forward<Expr>(expr) * scalar;
    }

}   // namespace simbi::expr

#endif   // SIMBI_EXPRESSIONS_HPP
