#ifndef ADAPTIVE_TIMESTEP_HPP
#define ADAPTIVE_TIMESTEP_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "execution/executor.hpp"
#include "execution/future.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "utility/helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace simbi {
    template <typename prim_t, std::uint64_t Dims>
    DEV real compute_local_timestep(
        const prim_t& prim,
        const vector_t<real, Dims>& cell_widths,
        real gamma,
        real cfl
    )
    {
        using namespace simbi::helpers;
        real min_dt = std::numeric_limits<real>::max();

        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            const auto ehat   = unit_vectors::ehat<Dims>(dim);
            const auto ws     = hydro::wave_speeds(prim, ehat, gamma);
            const auto ms     = my_max(std::abs(ws.left), std::abs(ws.right));
            const real dt_dim = cfl * cell_widths[dim] / ms;
            min_dt            = my_min(min_dt, dt_dim);
        }

        return min_dt;
    }

    template <typename PrimField, typename MeshConfig>
    struct timestep_op_t {
        PrimField prims;
        real gamma;
        real cfl;
        MeshConfig mesh;

        DEV constexpr auto
        operator()(coordinate_t<PrimField::dimensions> coord) const
        {
            const auto prim   = prims[coord];
            const auto widths = mesh::cell_widths(coord, mesh);
            return compute_local_timestep(prim, widths, gamma, cfl);
        }
    };

    template <typename PrimField, typename MeshConfig>
    auto create_timestep_field(
        const PrimField& prim,
        const MeshConfig& mesh,
        real gamma,
        real cfl
    )
    {
        return compute_field_t{
          timestep_op_t{prim[mesh.domain], gamma, cfl, mesh},
          mesh.domain
        };
    }

    template <typename PrimField, typename MeshConfig>
    exec::future_t<real> compute_timestep(
        const PrimField& state,
        const MeshConfig& mesh,
        real gamma,
        real cfl
    )
    {
        auto timestep_at = create_timestep_field(state, mesh, gamma, cfl);

        return exec::default_executor().reduce(
            timestep_at.domain(),
            std::numeric_limits<real>::max(),
            [timestep_at] DEV(auto coord) { return timestep_at(coord); },
            [](real a, real b) { return std::min(a, b); }
        );
    }

    template <typename SimState>
    void update_timestep(SimState& sim, std::uint64_t level_id)
    {
        auto& mesh     = sim.mesh(level_id);
        auto& prim     = sim.hydro(level_id).prim;
        auto& meta     = sim.metadata();
        auto dt_future = compute_timestep(prim, mesh, meta.gamma, meta.cfl);
        meta.dt        = dt_future.wait();
    }

}   // namespace simbi

#endif   // ADAPTIVE_TIMESTEP_HPP
