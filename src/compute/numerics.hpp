// =============================================================================
// numerics.hpp
//
// physics and numerics-specific functors for use with computation_t.
// these functors encode domain knowledge (hydro, mhd, eos) unlike the
// pure mathematical operations in functional/fp.hpp.
//
// usage:
//   cons_field = prim_field.map(to_conserved_t{gamma}).with(exec);
//   prim_field = cons_field.map(to_primitive_t{gamma}).with(exec);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "ecs/geometry_visitor.hpp"
#include "functional/fp.hpp"
#include "functional/monad/maybe.hpp"
#include "geometry/block_geometry.hpp"
#include "grid/domain.hpp"
#include "physics/hydro/conversion.hpp"
#include "physics/hydro/physics.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "utility/helpers.hpp"
#include "xpu/xpu.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace simbi::numerics {

    

    struct to_conserved_t
    {
        real gamma;

        template <typename Prim>
        constexpr DEV typename Prim::counterpart_t operator()(const Prim& prim) const
        {
            return hydro::to_conserved(prim, gamma);
        }
    };

    template <typename T>
    struct to_primitive_t
    {
        real gamma;
        T    block_geo;

        template <typename Cons>
        constexpr DEV maybe_t<typename Cons::counterpart_t>
                      operator()(auto coord, const Cons& cons) const
        {
            const auto dvscale = block_geo.extensive_scaling(coord);
            return hydro::to_primitive(cons, gamma, dvscale);
        }
    };

    

    // forward euler step: u^{n+1} = u^n + dt * L(u^n)
    // for moving mesh: dudt is intensive, volume_scale converts to extensive rate
    // extensive: \tilde{U}^{n+1} = \tilde{U}^n + dt * a^3 * L(u^n)
    struct euler_step_t
    {
        real dt;

        template <typename State>
        constexpr DEV State operator()(const State& u, const State& dudt) const
        {
            using namespace simbi::structs;
            return u | add_gas(dudt * dt);
        }
    };

    // linear time interpolation: u = (1-alpha)*u_n + alpha*u_curr
    // used for temporal refinement in amr ghost cell filling
    struct time_interpolate_t
    {
        real alpha;

        template <typename State>
        constexpr DEV State operator()(const State& u_n, const State& u_curr) const
        {
            using namespace simbi::structs;
            return u_n | scale_gas(1.0 - alpha) | add_gas(u_curr | scale_gas(alpha));
        }
    };

    
    template <
        typename PrimField,
        typename Geometry,
        std::uint64_t          Rank,
        xpu::execution_space_c ExecutionSpace>
    real compute_partition_timestep(
        const PrimField&                 prim,
        const grid::domain_t<Rank>&      domain,
        const Geometry&                  geometry,
        real                             cfl,
        real                             gamma,
        xpu::executor_t<ExecutionSpace>& exec
    )
    {
        if (domain.empty()) {
            return std::numeric_limits<real>::max();
        }

        auto kernel = [=] DEV(const iarray<Rank>& coord) -> real {
            const auto p = prim(coord);

            // scale factors h_i account for curvilinear coordinates
            // and moving mesh expansion (motion.a is already factored in)
            const auto h = geometry.scale_factors(coord);

            // get cell widths in coordinate space from the metric's maps
            const auto cell_widths = geometry.metric.cell_widths(coord);

            real min_dt = std::numeric_limits<real>::max();

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                const auto ehat = unit_vectors::ehat<Rank>(dd);
                // get grid velocity for moving mesh
                const auto offset = unit_vectors::array_offset<Rank>(dd);
                const real vfaceL = geometry.face_grid_velocity(coord, dd);
                const real vfaceR = geometry.face_grid_velocity(coord + offset, dd);
                // use average face velocity as cell-centered grid velocity
                const real v_c = 0.5 * (vfaceL + vfaceR);

                const auto ws = hydro::wave_speeds(p, ehat, gamma);

                // effective wave speeds include mesh motion
                const real s_left_eff  = std::abs(ws.left - v_c);
                const real s_right_eff = std::abs(ws.right - v_c);
                const real s_max       = helpers::my_max(s_left_eff, s_right_eff);

                // physical cell width = scale_factor * coordinate_width
                const real dx = h[dd] * cell_widths[dd];

                if (s_max > 0.0) {
                    min_dt = helpers::my_min(min_dt, cfl * dx / s_max);
                }
            }
            return min_dt;
        };

        return exec.reduce(domain, std::numeric_limits<real>::max(), kernel, fp::min_op);
    }

    
    template <typename Sim>
    real compute_level_timestep(Sim& sim, std::uint64_t lvl, const geometry::motion_state_t& motion)
    {
        const auto& meta     = sim.metadata();
        const auto& mesh_cfg = sim.mesh(lvl);

        real dt_min = std::numeric_limits<real>::max();

        // build geometry and compute timestep inside visitor
        ecs::with_block_geometry<Sim::coord_system>(mesh_cfg, motion, [&](const auto& block_geo) {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                real local_dt = compute_partition_timestep(
                    fields.prim.view(),
                    part.owned_domain,
                    block_geo,
                    meta.cfl,
                    meta.gamma,
                    exec
                );

                dt_min = std::min(dt_min, local_dt);
            }
        });

        return dt_min;
    }

} // namespace simbi::numerics
