// =============================================================================
// express_t.hpp
//
// encapsulates a full, evaluatable expression for use in the simulation.
// this file defines `expression_t`, a class that bundles together all the
// components of a mathematical expression: the dag, parameters, and the
// linearized instruction sequence for the vm. it provides `apply` methods
// to evaluate the expression as a source term or boundary condition.
//
// usage:
//   auto expr = expression_t::from_config(config);
//   conserved_t source = expr.apply(coords, prim_state, time, gamma);
// =============================================================================
#pragma once

#include "base/concepts.hpp"
#include "build_config.hpp"
#include "containers/store.hpp"
#include "containers/vector.hpp"
#include "dag/evaluator.hpp"
#include "dag/exp_load.hpp"
#include "dag/expression.hpp"
#include "dag/linearizer.hpp"
#include "decorators.hpp"
#include "physics/hydro/physics.hpp"
#include "utility/config_dict.hpp"
#include "xpu/mem/managed.hpp"

#include <cstdint>
#include <utility>

namespace simbi::state {

    struct hydro_source_tag;
    struct gravity_source_tag;

    template <std::uint64_t Rank>
    struct expression_t : public managed_t
    {
        bool                                 enabled;
        std::int64_t                         register_count;
        store_t<expression::ExprNode>        nodes;
        store_t<std::int64_t>                output_indices;
        store_t<std::int64_t>                output_indices_mapped;
        store_t<real>                        parameters;
        store_t<expression::LinearExprInstr> linear_instructions;

        template <concepts::is_hydro_conserved_c conserved_t>
        DEV conserved_t
        apply(const vector_t<real, Rank> coords, const conserved_t& cons, real time = 0.0) const
        {
            if (!enabled) {
                return conserved_t{}; // return zeroed conserved state
            }
            conserved_t       result{};
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                local_coords[ii] = coords[ii];
            }

            expression::evaluate_linear_expr(
                linear_instructions.data(),
                linear_instructions.size(),
                output_indices_mapped.data(),
                output_indices.size(),
                local_coords[0],
                local_coords[1],
                local_coords[2],
                time,
                cons.data(),
                result.data()
            );

            return result;
        }

        template <concepts::is_hydro_primitive_c primitive_t>
        DEV typename primitive_t::counterpart_t apply(
            const vector_t<real, Rank> coords,
            const primitive_t&         prim,
            real                       time,
            real                       gamma
        ) const
        {
            using conserved_t = typename primitive_t::counterpart_t;
            if (!enabled) {
                return conserved_t{};
            }

            vector_t<real, Rank> local_vector{0.0};
            vector_t<real, 3>    local_coords{0.0, 0.0, 0.0};
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                local_coords[ii] = coords[ii];
            }

            expression::evaluate_linear_expr(
                linear_instructions.data(),
                linear_instructions.size(),
                output_indices_mapped.data(),
                output_indices.size(),
                local_coords[0],
                local_coords[1],
                local_coords[2],
                time,
                nullptr,
                local_vector.data()
            );

            // gravity source specialization
            const auto den   = hydro::labframe_density(prim);
            const auto dp_dt = den * local_vector;
            const auto v_old = prim.vel;
            const auto v_new = (hydro::linear_momentum(prim, gamma) + dp_dt) / den;
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE_dt = vecops::dot(dp_dt, v_avg);

            if constexpr (is_mhd_conserved_c<conserved_t>) {
                return conserved_t{
                    0.0,   // density source term is zero
                    dp_dt, // momentum source term is the force
                    dE_dt, // energy source term is the power
                    {},    // magnetic source term is zero
                    0.0    // chi source term is zero
                };
            }
            else {
                return conserved_t{
                    0.0,   // density source term is zero
                    dp_dt, // momentum source term is the force
                    dE_dt, // energy source term is the power
                    0.0    // chi source term is zero
                };
            }
        }

        // boundary condition variant: evaluates expression to primitive state directly
        // used for dynamic inflow boundaries where we set state values, not apply sources
        template <concepts::is_hydro_primitive_c primitive_t>
        DEV primitive_t
        apply(const vector_t<real, Rank> coords, const primitive_t& edge_state, real time) const
        {
            if (!enabled) {
                return edge_state;
            }

            primitive_t       result{};
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                local_coords[ii] = coords[ii];
            }

            expression::evaluate_linear_expr(
                linear_instructions.data(),
                linear_instructions.size(),
                output_indices_mapped.data(),
                output_indices.size(),
                local_coords[0],
                local_coords[1],
                local_coords[2],
                time,
                edge_state.data(),
                result.data()
            );

            return result;
        }

        static expression_t from_config(const config_dict_t& config)
        {
            expression_t expr;

            if (config.empty()) {
                expr.enabled = false;
                return expr;
            }

            auto [nodes, output_indices, params] = expression::load_expression_data(config);
            auto [linear_instrs, mapped_output] =
                expression::linearize_expression_tree(nodes, output_indices);

            expr.enabled               = true;
            expr.register_count        = expression::get_max_register(linear_instrs);
            expr.nodes                 = std::move(nodes);
            expr.linear_instructions   = std::move(linear_instrs);
            expr.output_indices        = std::move(output_indices);
            expr.output_indices_mapped = std::move(mapped_output);
            expr.parameters            = std::move(params);

            return expr;
        }
    };

} // namespace simbi::state
