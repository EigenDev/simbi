#ifndef UPDATE_TIMESTEP_HPP
#define UPDATE_TIMESTEP_HPP

// =============================================================================
// timestep.hpp
//
// cfl-based adaptive timestep computation using heterogeneous execution.
// computes minimum dt over all cells accounting for curvilinear geometry.
//
// usage:
//   auto dt = compute_level_timestep(sim, lvl, motion);
// =============================================================================

#include "compat.hpp"
#include "containers/vector.hpp"
#include "ecs/geometry_visitor.hpp"
#include "geometry/block_geometry.hpp"
#include "grid/domain.hpp"
#include "hesi/exec/executor.hpp"
#include "hesi/exec/reduce.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "utility/helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace simbi::timestep {

    // =========================================================================
    // compute_partition_timestep
    //
    // computes minimum dt over a partition's owned domain using proper
    // geometry. block_geometry provides scale factors and coordinate maps.
    // =========================================================================
    template <typename PrimField, typename Geometry, std::uint64_t Rank>
    real compute_partition_timestep(
        const PrimField& prim,
        const grid::domain_t<Rank>& domain,
        const Geometry& geometry,
        real cfl,
        real gamma,
        het::exec::executor_t& exec
    )
    {
        if (domain.empty()) {
            return std::numeric_limits<real>::max();
        }

        auto kernel = [=] DUAL(const iarray<Rank>& coord) {
            const auto p = prim(coord);

            // scale factors h_i account for curvilinear coordinates
            // and moving mesh expansion (motion.a is already factored in)
            const auto h = geometry.scale_factors(coord);

            // get cell widths in coordinate space from the metric's maps
            const auto cell_widths = geometry.metric.cell_widths(coord);

            real min_dt = std::numeric_limits<real>::max();

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                const auto ehat = unit_vectors::ehat<Rank>(dd);
                const auto ws   = hydro::wave_speeds(p, ehat, gamma);

                // max signal speed in this direction
                const real s_max =
                    helpers::my_max(std::abs(ws.left), std::abs(ws.right));

                // physical cell width = scale_factor * coordinate_width
                const real dx = h[dd] * cell_widths[dd];

                if (s_max > 0.0) {
                    min_dt = helpers::my_min(min_dt, cfl * dx / s_max);
                }
            }

            return min_dt;
        };

        return het::exec::reduce_sync(
            exec,
            domain,
            std::numeric_limits<real>::max(),
            kernel,
            [] DEV(real a, real b) { return helpers::my_min(a, b); },
            std::numeric_limits<real>::max()
        );
    }

    // =========================================================================
    // compute_level_timestep
    //
    // computes minimum dt across all partitions of a level.
    // uses proper curvilinear geometry from mesh config.
    // =========================================================================
    template <typename Sim>
    real compute_level_timestep(
        Sim& sim,
        std::uint64_t lvl,
        const geometry::motion_state_t& motion
    )
    {
        const auto& meta     = sim.metadata();
        const auto& mesh_cfg = sim.mesh(lvl);

        real dt_min = std::numeric_limits<real>::max();

        // build geometry and compute timestep inside visitor
        ecs::with_block_geometry<Sim::coord_system>(
            mesh_cfg,
            motion,
            [&](const auto& block_geo) {
                for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                    auto& fields = sim.partition_hydro(lvl, pp);
                    auto& part   = sim.partition(lvl, pp);
                    auto exec    = sim.partition_executor(lvl, pp);

                    real local_dt = compute_partition_timestep(
                        fields.prim,
                        part.owned_domain,
                        block_geo,
                        meta.cfl,
                        meta.gamma,
                        exec
                    );

                    dt_min = std::min(dt_min, local_dt);
                }
            }
        );

        return dt_min;
    }

}   // namespace simbi::timestep

#endif   // UPDATE_TIMESTEP_HPP
