#ifndef SYSTEMS_HPP
#define SYSTEMS_HPP

#include "compute/cfd.hpp"
#include "containers/state_ops.hpp"
#include "ecs/components.hpp"
#include "mesh/fmr/flux_correction.hpp"
#include "mesh/fmr/transfer.hpp"
#include "physics/ib/collection.hpp"
#include "physics/ib/diagnostics.hpp"
#include "update/bcs.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <memory>

namespace simbi::ecs {
    using namespace simbi::cfd;
    using namespace simbi::mesh::fmr;

    template <typename... Components>
    struct optional_system_t {
        template <typename Sim>
        bool has_requirements(const Sim& sim) const
        {
            return (sim.registry.template has<Components>(sim.global) && ...);
        }
    };

    struct sink_cache_t {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            update_sink_cache(sim);
        }
    };

    struct staggered_fields_system_t {
        bool advance_bfields = true;

        template <typename Sim, typename Ops>
        void operator()(Sim& sim, Ops ops, std::uint64_t lvl) const
        {
            auto& hydro       = sim.hydro(lvl);
            auto& mesh        = sim.mesh(lvl);
            auto& meta        = sim.metadata();
            const auto& prims = hydro.prim[mesh.domain];

            // compute fluxes in each direction
            for (std::uint64_t dir = 0; dir < Sim::dimensions; ++dir) {
                auto ff   = compute_fluxes(prims, mesh, ops, meta, dir);
                auto flux = hydro.flux[dir][mesh.face_domain[dir]];
                flux      = flux.coord_map(ff);
            }

            if constexpr (Sim::is_mhd) {
                // if we're doing MHD, then we need to make sure
                // that quantities that are staggered (i.e, the fluxes)
                // correctly have the perpendicular boundary conditions
                // applied since we do not save the edge-centered
                // electric fields but rather compute them
                // on-the-fly.
                if (advance_bfields) {
                    boundary::apply_stagg_bcs(
                        hydro,
                        mesh,
                        meta.boundary_conditions
                    );
                    em::update_magnetic_fields(hydro, mesh, meta.dt);
                }
            }
        }
    };

    struct c2p_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            auto& hydro = sim.hydro(lvl);
            recover_primitives(hydro.prim, hydro.cons, sim.metadata().gamma);
        }
    };

    struct timestep_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            if (lvl < sim.num_levels() - 1) {
                return;
            }
            update_timestep(sim, lvl);
            sim.metadata().time += sim.metadata().dt;
        }
    };

    struct ghost_fill_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            if (lvl == 0) {
                boundary::apply_boundary_conditions(sim);
                return;
            }
            prolongate_level_data(sim, lvl);
        }
    };

    // system that requires body components
    template <std::uint64_t Dims>
    struct body_effects_system_t
        : optional_system_t<immersed_bodies_t<Dims>, body_info_t<Dims>> {
        template <typename Sim>
        auto operator()(Sim& sim, std::uint64_t lvl) const
        {
            using base_t =
                optional_system_t<immersed_bodies_t<Dims>, body_info_t<Dims>>;

            auto& hydro      = sim.hydro(lvl);
            auto& mesh       = sim.mesh(lvl);
            const auto& meta = sim.metadata();
            // only run if we have the required components
            if (!base_t::has_requirements(sim)) {
                return body_effects(
                    hydro.prim[mesh.domain],
                    mesh,
                    body::body_collection_t<Dims>{},
                    std::unique_ptr<body::body_diagnostics_t<Dims>>{},
                    meta.gamma,
                    meta.dt
                );
            }

            // now we can safely use
            auto effects = body_effects(
                hydro.prim[mesh.domain],
                mesh,
                sim.bodies(),
                sim.diagnostics(),
                meta.gamma,
                meta.dt
            );
            return effects;
        }
    };

    template <typename Ops>
    struct euler_system_t {
        Ops ops;

        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            auto& hydro   = sim.hydro(lvl);
            auto& mesh    = sim.mesh(lvl);
            auto& meta    = sim.metadata();
            auto& sources = sim.sources();

            // flux computation
            sink_cache_t{}(sim);
            staggered_fields_system_t{}(sim, ops, lvl);

            // base Godunov operator
            const auto ell = godunov_op(hydro, mesh, meta, sources);

            // optional body effects
            auto be = body_effects_system_t<Sim::dimensions>{}(sim, lvl);

            auto u   = hydro.cons[mesh.domain];
            auto u_p = u.enum_map([&](auto coord, auto u) {
                return u | add_gas((ell(coord) + be(coord)) * meta.dt);
            });
            u        = u_p;

            if constexpr (Sim::is_mhd) {
                em::update_energy_density(hydro.cons, hydro.bfield, mesh);
            }
        }
    };

    template <typename Sim>
    struct rk_workspace_t {
        using cons_field_t = decltype(std::declval<Sim>().hydro(0).cons);

        cons_field_t u_n;      // init state
        cons_field_t u_star;   // intermediate state

        rk_workspace_t(const Sim& sim, std::uint64_t lvl)
            : u_n(sim.hydro(lvl).cons.clone()),
              u_star(sim.hydro(lvl).cons.clone())
        {
        }
    };

    template <typename Ops>
    struct rk2_system_t {
        Ops ops;

        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto& hydro   = sim.hydro(lvl);
            auto& mesh    = sim.mesh(lvl);
            auto& meta    = sim.metadata();
            auto& sources = sim.sources();
            const auto dt = sim.metadata().dt;

            sink_cache_t{}(sim);
            staggered_fields_system_t{}(sim, ops, lvl);
            auto body1 = body_effects_system_t<Sim::dimensions>{}(sim, lvl);

            // create workspace for RK stages
            rk_workspace_t workspace(sim, lvl);

            // === First Stage (k1) ===
            auto k1 = godunov_op(hydro, mesh, meta, sources);

            // update to intermediate state u*
            auto u1 = workspace.u_star[mesh.domain];
            u1      = u1.enum_map([k1, body1, dt](auto coord, auto u) {
                return u | add_gas((k1(coord) + body1(coord)) * dt);
            });

            hydro.cons = workspace.u_star.map([](auto u) { return u; });
            if constexpr (Sim::is_mhd) {
                em::update_energy_density(hydro.cons, hydro.bfield, mesh);
            }

            // apply BCs and recover primitives for stage 2
            c2p_system_t{}(sim, lvl);
            sink_cache_t{}(sim);
            staggered_fields_system_t{.advance_bfields = false}(sim, ops, lvl);
            auto body2 = body_effects_system_t<Sim::dimensions>{}(sim, lvl);

            // === Second Stage (k2) ===
            auto k2 = godunov_op(hydro, mesh, meta, sources);

            // final update
            auto unc = workspace.u_n[mesh.domain];
            unc      = unc.enum_map([u1, k2, dt, body2](auto coord, auto u) {
                return u | scale_gas(0.5) | add_gas(0.5 * u1[coord]) |
                       add_gas(0.5 * dt * (k2(coord) + body2(coord)));
            });

            hydro.cons = workspace.u_n.map([](auto u) { return u; });
            if constexpr (Sim::is_mhd) {
                em::update_energy_density(hydro.cons, hydro.bfield, mesh);
            }
        }
    };

    struct integration_system_t {
        template <typename Sim, typename Ops>
        void operator()(Sim& sim, std::uint64_t lvl, const Ops& ops) const
        {
            if (sim.metadata().timestepping == Timestepping::EULER) {
                euler_system_t{ops}(sim, lvl);
            }
            else if (sim.metadata().timestepping == Timestepping::RK2) {
                rk2_system_t{ops}(sim, lvl);
            }
            else {
                throw std::runtime_error(
                    "That timestepping method is not implemented."
                );
            }
        }
    };

    struct flux_correction_system_t {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            if (sim.num_levels() == 1) {
                return;
            }
            auto& hierarchy = sim.hierarchy();
            for (std::uint64_t lvl = hierarchy.num_levels - 1; lvl > 0; --lvl) {
                auto& coarser_hydro = sim.hydro(lvl);
                auto& finer_hydro   = sim.hydro(lvl - 1);
                correct_level_fluxes(
                    coarser_hydro.flux,
                    finer_hydro.flux,
                    hierarchy,
                    lvl
                );
            }
        }
    };

}   // namespace simbi::ecs

#endif
