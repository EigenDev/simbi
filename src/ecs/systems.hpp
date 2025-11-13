#ifndef SYSTEMS_HPP
#define SYSTEMS_HPP

#include "compat.hpp"
#include "compute/cfd.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "ecs/components.hpp"
#include "io/exceptions.hpp"
#include "mesh/fmr/flux_correction.hpp"
#include "mesh/fmr/hierarchy.hpp"
#include "mesh/fmr/prolongation.hpp"
#include "mesh/fmr/restriction.hpp"
#include "physics/em/ct_updater.hpp"
#include "physics/ib/collection.hpp"
#include "physics/ib/diagnostics.hpp"
#include "update/adaptive_timestep.hpp"
#include "update/bcs.hpp"
#include "update/prim_recovery.hpp"

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

    struct sink_cache_system_t {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            update_sink_cache(sim);
        }
    };

    struct zero_flux_buffer_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            // only run if we have refinement (and thus, flux_avg buffers)
            if (!sim.has_refinement()) {
                return;
            }

            auto& hydro = sim.hydro(lvl);

            // loop over each spatial direction
            for (std::uint64_t dir = 0; dir < Sim::dimensions; ++dir) {
                // get the entire flux_avg field (not just a domain view)
                auto& ff_avg = hydro.flux_avg[dir];
                ff_avg       = ff_avg.map([](auto f) {
                    return f | structs::scale_gas(0.0);
                });
            }
        }
    };

    struct staggered_fields_system_t {
        bool advance_bfields   = true;
        bool accumulate_fluxes = false;

        template <typename Sim, typename Ops>
        void operator()(Sim& sim, Ops ops, std::uint64_t lvl) const
        {
            auto& hydro       = sim.hydro(lvl);
            auto& mesh        = sim.mesh(lvl);
            auto& meta        = sim.metadata();
            const auto& prims = hydro.prim[mesh.domain];

            const real dt_lvl = meta.level_dts[lvl];

            // compute fluxes in each direction
            for (std::uint64_t dir = 0; dir < Sim::dimensions; ++dir) {
                auto ff   = compute_fluxes(prims, mesh, ops, meta, dir);
                auto flux = hydro.flux[dir][mesh.face_domain[dir]];
                flux      = flux.coord_map(ff);

                if (accumulate_fluxes && sim.has_refinement()) {
                    auto flux_avg = hydro.flux_avg[dir][mesh.face_domain[dir]];

                    // accumulate the time-weighted flux
                    flux_avg =
                        flux_avg.enum_map([ff, dt_lvl](auto coord, auto avg) {
                            return avg + ff(coord) * dt_lvl;
                        });
                }
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
                    em::update_magnetic_fields(
                        hydro,
                        mesh,
                        meta.level_dts[lvl]
                    );
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
            const auto nlvls = sim.num_levels();
            if (lvl < nlvls - 1) {
                return;
            }
            update_timestep(sim, lvl);
            auto& meta = sim.metadata();

            // propagate dt to all levels
            for (std::int64_t ell = nlvls - 2; ell >= 0; --ell) {
                // get the refinement ratio from the *finer* level (L+1)
                const auto ref_ratio = sim.level_info(ell + 1).refinement_ratio;

                meta.level_dts[ell] = meta.level_dts[ell + 1] * ref_ratio;
            }
            meta.global_dt = meta.level_dts[0];
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
            const auto& map = sim.level_mapping(lvl);
            auto& fine      = sim.hydro(lvl);
            auto& coarse    = sim.hydro(lvl - 1);

            // porlongate
            prolongate_ghosts_conservative(coarse.cons, fine.cons, map);
        }
    };

    struct time_interpolated_ghost_fill_system_t {
        // 'alpha' is the fractional time to interpolate to.
        real alpha = 0.5;

        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            using namespace structs;
            if (lvl == 0) {
                return;
            }

            // get components
            const auto& coarse_rk    = sim.rk_workspace(lvl - 1);
            const auto& coarse_hydro = sim.hydro(lvl - 1);
            auto& fine_hydro         = sim.hydro(lvl);
            const auto& map          = sim.level_mapping(lvl);

            prolongate_ghosts_time_interpolated(
                coarse_rk.u_n,       //  u_n
                coarse_hydro.cons,   //  u_star
                alpha,               //  interpolation factor
                fine_hydro.cons,     //  destination field
                map
            );
        }
    };

    // system that requires body components
    template <std::uint64_t Dims>
    struct body_effects_system_t
        : optional_system_t<immersed_bodies_t<Dims>, body_info_t<Dims>> {
        std::unique_ptr<body::body_diagnostics_t<Dims>> def_diag{nullptr};

        std::uint64_t find_level_containing_position(
            const vector_t<real, Dims>& position,
            const mesh_hierarchy_t<Dims>& hierarchy,
            const auto& get_mesh_func
        ) const
        {
            // start from finest level and work backwards
            for (std::int64_t lvl = hierarchy.num_levels - 1; lvl >= 0; --lvl) {
                const auto& mesh = get_mesh_func(lvl);

                // check if position is within this level's physical bounds
                bool inside = true;
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    if (position[d] < mesh.bounds_min[d] ||
                        position[d] >= mesh.bounds_max[d]) {
                        inside = false;
                        break;
                    }
                }

                if (inside) {
                    return lvl;
                }
            }

            // fallback: return base level
            return 0;
        }

        template <typename Sim>
        auto operator()(Sim& sim, std::uint64_t lvl) const
        {
            using base_t =
                optional_system_t<immersed_bodies_t<Dims>, body_info_t<Dims>>;

            const auto& hydro = sim.hydro(lvl);
            const auto& mesh  = sim.mesh(lvl);
            const auto& meta  = sim.metadata();
            // only run if we have the required components
            if (!base_t::has_requirements(sim)) {
                return body_effects(
                    hydro.prim[mesh.domain],
                    mesh,
                    body::body_collection_t<Dims>{},
                    def_diag,
                    meta.gamma,
                    meta.level_dts[lvl]
                );
            }

            // check if ANY body is on this level
            bool this_level_has_body = false;
            auto& bodies             = sim.bodies();

            if (!sim.has_refinement()) {
                this_level_has_body = (lvl == 0);
            }
            else {
                bodies.visit_all([&](const auto& body) {
                    auto body_level = find_level_containing_position(
                        body.position,
                        sim.hierarchy(),
                        [&](auto l) -> const auto& { return sim.mesh(l); }
                    );
                    if (lvl == body_level) {
                        this_level_has_body = true;
                    }
                });
            }

            // apply effects for bodies on this level
            return body_effects(
                hydro.prim[mesh.domain],
                mesh,
                bodies,
                this_level_has_body ? sim.diagnostics() : def_diag,
                meta.gamma,
                meta.level_dts[lvl]
            );
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

            // base Godunov operator
            auto ell = godunov_op(hydro, mesh, meta, sources);

            // optional body effects
            auto be = body_effects_system_t<Sim::dimensions>{}(sim, lvl);

            auto u_p = hydro.cons[mesh.domain];
            u_p      = u_p.enum_map([&](auto coord, auto u) {
                return u |
                       add_gas((ell(coord) + be(coord)) * meta.level_dts[lvl]);
            });

            if constexpr (Sim::is_mhd) {
                em::update_energy_density(hydro.cons, hydro.bfield, mesh);
            }
        }
    };

    template <typename Ops>
    struct rk2_stage1_system_t {
        Ops ops;

        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto& hydro   = sim.hydro(lvl);
            auto& mesh    = sim.mesh(lvl);
            auto& meta    = sim.metadata();
            auto& sources = sim.sources();
            const auto dt = meta.level_dts[lvl];

            // assume fluxes already computed and corrected
            auto k1   = godunov_op(hydro, mesh, meta, sources);
            auto body = body_effects_system_t<Sim::dimensions>{}(sim, lvl);

            if (!sim.has_rk_workspace(lvl)) {
                sim.create_rk_workspace(lvl);
            }
            auto& workspace = sim.rk_workspace(lvl);

            // store u^n
            workspace.u_n = hydro.cons.map([](auto u) { return u; });

            // advance to u^*
            auto u_star = hydro.cons[mesh.domain];
            u_star      = u_star.enum_map([k1, body, dt](auto coord, auto u) {
                return u | add_gas((k1(coord) + body(coord)) * dt);
            });

            if constexpr (Sim::is_mhd) {
                em::update_energy_density(hydro.cons, hydro.bfield, mesh);
            }
        }
    };

    template <typename Ops>
    struct rk2_stage2_system_t {
        Ops ops;

        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto& hydro   = sim.hydro(lvl);
            auto& mesh    = sim.mesh(lvl);
            auto& meta    = sim.metadata();
            auto& sources = sim.sources();
            const auto dt = meta.level_dts[lvl];

            auto& workspace = sim.rk_workspace(lvl);

            // assume fluxes already computed and corrected for u^*
            auto k2   = godunov_op(hydro, mesh, meta, sources);
            auto body = body_effects_system_t<Sim::dimensions>{}(sim, lvl);

            // get current u^* and original u^n
            auto u_star = hydro.cons[mesh.domain];
            auto u_n    = workspace.u_n[mesh.domain];

            // RK2 combination: u^(n+1) = 0.5*u^n + 0.5*(u^* + dt*k2)
            u_n = u_n.enum_map([u_star, k2, body, dt](auto coord, auto u) {
                return u | scale_gas(0.5) | add_gas(0.5 * u_star(coord)) |
                       add_gas(0.5 * dt * (k2(coord) + body(coord)));
            });

            // write final solution
            hydro.cons = workspace.u_n.map([](auto u) { return u; });

            if constexpr (Sim::is_mhd) {
                em::update_energy_density(hydro.cons, hydro.bfield, mesh);
            }
        }
    };

    struct flux_correction_system_t {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            if (!sim.has_refinement()) {
                return;
            }

            auto& hierarchy = sim.hierarchy();
            auto& meta      = sim.metadata();
            for (std::uint64_t lvl = hierarchy.num_levels - 1; lvl > 0; --lvl) {
                auto& coarser_hydro  = sim.hydro(lvl - 1);
                auto& finer_hydro    = sim.hydro(lvl);
                auto& map            = sim.level_mapping(lvl);
                const real dt_coarse = meta.level_dts[lvl - 1];
                correct_level_fluxes(
                    coarser_hydro.flux,
                    finer_hydro.flux_avg,
                    map,
                    dt_coarse
                );
            }
        }
    };

    struct restriction_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            const auto& map = sim.level_mapping(lvl);
            auto& fine      = sim.hydro(lvl);
            auto& coarse    = sim.hydro(lvl - 1);
            restrict_conservative(fine.cons, coarse.cons, map);
        }
    };

}   // namespace simbi::ecs

#endif   // SYSTEMS_HPP
