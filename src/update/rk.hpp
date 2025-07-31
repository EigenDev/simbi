#ifndef RK_KUTTA_HPP
#define RK_KUTTA_HPP

#include "bcs.hpp"
#include "containers/state_ops.hpp"
#include "flux.hpp"
#include "physics/em/perm.hpp"
#include "prim_recovery.hpp"

namespace simbi::cfd {
    template <typename HydroState, typename MeshConfig>
    auto godunov_op(const HydroState& state, const MeshConfig& mesh);
}

namespace simbi {
    // RK2 step
    namespace rk {
        using namespace simbi::structs;

        template <typename HydroState, typename MeshConfig, typename CfdOps>
        auto rk2_step(
            HydroState& workspace,
            const MeshConfig& mesh,
            const CfdOps& ops
        )
        {
            const auto dt = workspace.metadata.dt;
            update_staggered_fields(workspace, ops, mesh);

            // === initial state u_n ===
            auto un = workspace.cons.clone();

            // === k1 evaluation ===
            const auto k1 = cfd::godunov_op(workspace, mesh);

            auto u1  = workspace.cons.clone();
            auto u1c = u1[mesh.domain];
            u1c      = u1c.map([k1, dt](const auto coord, const auto u) {
                return u | add_gas(k1(coord) * dt);
            });

            // === intermediate state u_star ===
            workspace.cons = u1.map([](auto u) { return u; });
            if constexpr (HydroState::is_mhd) {
                // correct energy density from CT algorithm
                em::update_energy_density(workspace, mesh);
            }
            boundary::apply_boundary_conditions(workspace, mesh);

            // === k2 evaluation ===
            hydro::recover_primitives(workspace);
            update_staggered_fields(
                workspace,
                ops,
                mesh,
                {.advance_bfields = false}
            );
            auto k2 = cfd::godunov_op(workspace, mesh);

            auto unc = un[mesh.domain];
            unc      = unc.map(
                [k2, dt, us = u1[mesh.domain]](const auto coord, const auto u) {
                    return u | scale_gas(0.5) | add_gas(0.5 * us[coord]) |
                           add_gas(0.5 * dt * k2(coord));
                }
            );

            // final update
            workspace.cons = un.map([](auto u) { return u; });
            if constexpr (HydroState::is_mhd) {
                // correct energy density from CT algorithm
                em::update_energy_density(workspace, mesh);
            }
            boundary::apply_boundary_conditions(workspace, mesh);
            hydro::recover_primitives(workspace);
            update_timestep(workspace, mesh);

            workspace.metadata.time += dt;
        }
    }   // namespace rk
}   // namespace simbi
#endif
