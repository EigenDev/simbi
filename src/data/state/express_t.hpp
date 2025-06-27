#ifndef SIMBI_STATE_EXPRESSION_HPP
#define SIMBI_STATE_EXPRESSION_HPP

#include "compute/math/evaluator.hpp"
#include "compute/math/exp_load.hpp"
#include "compute/math/expression.hpp"
#include "compute/math/linearizer.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/utility/config_dict.hpp"
#include "core/utility/managed.hpp"
#include "data/containers/ndarray.hpp"
#include "data/containers/vector.hpp"
#include "physics/hydro/physics.hpp"
#include <cstdint>
#include <utility>

namespace simbi::state {
    using namespace nd;

    struct hydro_source_tag;
    struct gravity_source_tag;

    template <std::uint64_t Dims>
    struct expression_t : public Managed<platform::is_gpu> {
        bool enabled;
        std::int64_t register_count;
        ndarray_t<expression::ExprNode> nodes;
        ndarray_t<std::int64_t> output_indices;
        ndarray_t<std::int64_t> output_indices_mapped;
        ndarray_t<real> parameters;
        ndarray_t<expression::LinearExprInstr> linear_instructions;

        template <concepts::is_hydro_conserved_c conserved_t>
        DEV auto apply(
            const vector_t<real, Dims> coords,
            const conserved_t& cons,
            real time = 0.0,
            real dt   = 0.0
        ) const
        {
            if (!enabled) {
                return conserved_t{};   // return zeroed conserved state
            }
            conserved_t result{};
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
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
                dt,
                cons.data(),
                result.data()
            );

            return result;
        }

        template <concepts::is_hydro_primitive_c primitive_t>
        DEV auto apply(
            const vector_t<real, Dims> coords,
            const primitive_t& prim,
            real time,
            real gamma
        ) const
        {
            using conserved_t = typename primitive_t::counterpart_t;
            if (!enabled) {
                return conserved_t{};   // return zeroed
                                        // conserved state
            }

            vector_t<real, Dims> local_vector{0.0};
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
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
                0.0,   // dt not used for gravity sources
                nullptr,
                local_vector.data()
            );

            // this is a specialziation for gravity sources
            const auto den   = hydro::labframe_density(prim);
            const auto dp_dt = den * local_vector;
            const auto v_old = prim.vel;
            const auto v_new =
                (hydro::spatial_momentum(prim, gamma) + dp_dt) / den;
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE_dt = vecops::dot(dp_dt, v_avg);

            if constexpr (is_mhd_conserved_c<conserved_t>) {
                return conserved_t{
                  0.0,     // density source term is zero
                  dp_dt,   // momentum source term is the force
                  dE_dt,   // energy source term is the power
                  {},      // magnetic source term is zero
                  0.0      // chi source term is zero
                };
            }
            else {
                return conserved_t{
                  0.0,     // density source term is zero
                  dp_dt,   // momentum source term is the force
                  dE_dt,   // energy source term is the power
                  0.0      // chi source term is zero
                };
            }
        }

        static expression_t from_config(const config_dict_t& config)
        {
            expression_t expr;

            if (config.empty()) {
                expr.enabled = false;
                return expr;
            }

            auto [nodes, output_indices, params] =
                expression::load_expression_data(config);
            auto [linear_instrs, mapped_output] =
                expression::linearize_expression_tree(nodes, output_indices);

            expr.enabled               = true;
            expr.nodes                 = std::move(nodes);
            expr.linear_instructions   = std::move(linear_instrs);
            expr.output_indices        = std::move(output_indices);
            expr.output_indices_mapped = std::move(mapped_output);
            expr.parameters            = std::move(params);
            expr.register_count = expression::get_max_register(linear_instrs);

            if constexpr (platform::is_gpu) {
                expr.nodes.sync_to_device();
                expr.output_indices.sync_to_device();
                expr.parameters.sync_to_device();
                expr.linear_instructions.sync_to_device();
                expr.output_indices_mapped.sync_to_device();
            }

            return expr;
        }
    };

    // template <std::uint64_t Dims>
    // auto make_source(
    //     const expression_t<Dims>& expr,
    //     const index_space_t<Dims>& domain
    // )
    // {
    //     return source_expr_t<Dims>{expr, domain};
    // }

}   // namespace simbi::state

#endif
