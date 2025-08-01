#ifndef ADAPTIVE_TIMESTEP_HPP
#define ADAPTIVE_TIMESTEP_HPP

#include "compute/field.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "execution/executor.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/wave_speeds.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace simbi {
    // pure timestep computation function (unchanged)
    template <typename prim_t, std::uint64_t Dims>
    real compute_local_timestep(
        const prim_t& prim,
        const vector_t<real, Dims>& cell_widths,
        real gamma,
        real cfl
    )
    {
        real min_dt = std::numeric_limits<real>::max();

        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            // unit vector in this dimension
            const auto ehat = unit_vectors::ehat<Dims>(dim);

            // compute characteristic wave speeds
            const auto speeds = hydro::wave_speeds(prim, ehat, gamma);

            // maximum wave speed in this direction
            const auto max_speed =
                std::max(std::abs(speeds.left), std::abs(speeds.right));

            // CFL timestep constraint for this dimension
            real dt_dim = cfl * cell_widths[dim] / max_speed;
            min_dt      = std::min(min_dt, dt_dim);
        }

        return min_dt;
    }

    // ============================================================================
    // FIELD-BASED TIMESTEP COMPUTATION
    // ============================================================================
    template <typename HydroState, typename MeshConfig>
    auto create_timestep_field(const HydroState& state, const MeshConfig& mesh)
    {
        const auto gamma       = state.metadata.gamma;
        const auto cfl         = state.metadata.cfl;
        const auto domain      = mesh.full_domain;
        const auto* prim_field = state.prim.data();

        // create lazy timestep field using compute_field_t
        auto timestep_func = [=](const auto& coord) -> real {
            auto idx = domain.coord_to_linear(coord);   // ensure coord is valid
            auto prim   = prim_field[idx];
            auto widths = mesh::cell_widths(coord, mesh);
            return compute_local_timestep(prim, widths, gamma, cfl);
        };

        return field(domain, timestep_func);
    }

    template <typename HydroState, typename MeshConfig>
    auto compute_timestep_async(const HydroState& state, const MeshConfig& mesh)
    {
        // create lazy timestep field
        auto timestep_field = create_timestep_field(state, mesh);

        // use executor to reduce over the field's domain
        auto executor = async::default_executor();

        return executor.reduce(
            timestep_field.domain(),
            std::numeric_limits<real>::max(),
            [&timestep_field](auto coord) { return timestep_field(coord); },
            [](real a, real b) { return std::min(a, b); }
        );
    }

    template <typename HydroState, typename MeshConfig>
    void update_timestep(HydroState& state, const MeshConfig& mesh)
    {
        auto dt_future    = compute_timestep_async(state, mesh);
        state.metadata.dt = dt_future.wait();
    }

}   // namespace simbi

#endif   // ADAPTIVE_TIMESTEP_HPP
