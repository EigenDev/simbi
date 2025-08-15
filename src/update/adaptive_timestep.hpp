#ifndef ADAPTIVE_TIMESTEP_HPP
#define ADAPTIVE_TIMESTEP_HPP

#include "compute/field.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "execution/executor.hpp"
#include "execution/future.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "utility/helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <ctime>
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

    template <typename HydroState, typename MeshConfig>
    struct timestep_op_t {
        HydroState state;
        MeshConfig mesh;

        DEV constexpr auto
        operator()(coordinate_t<HydroState::dimensions> coord) const
        {
            const auto gamma  = state.metadata.gamma;
            const auto cfl    = state.metadata.cfl;
            const auto prim   = state.prim[coord];
            const auto widths = mesh::cell_widths(coord, mesh);
            return compute_local_timestep(prim, widths, gamma, cfl);
        }
    };

    template <typename HydroState, typename MeshConfig>
    auto create_timestep_field(const HydroState& state, const MeshConfig& mesh)
    {
        return compute_field_t{timestep_op_t{state, mesh}, mesh.full_domain};
    }

    template <typename HydroState, typename MeshConfig>
    exec::future_t<real>
    compute_timestep_async(const HydroState& state, const MeshConfig& mesh)
    {
        auto timestep_at = create_timestep_field(state, mesh);
        auto executor    = exec::default_executor();

        return executor.reduce(
            timestep_at.domain(),
            std::numeric_limits<real>::max(),
            [timestep_at] DEV(auto coord) { return timestep_at(coord); },
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
