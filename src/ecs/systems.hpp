#ifndef SYSTEMS_HPP
#define SYSTEMS_HPP

// =============================================================================
// systems.hpp
//
// ecs systems for partition-aware multi-device simulations.
// each system operates on partitions, using executors for kernel dispatch.
//
// key changes from systems.hpp:
//   - uses partition_hydro(lvl, part) instead of hydro(lvl)
//   - uses partition_executor(lvl, part) for kernel dispatch
//   - uses grid/amr/* for amr operations
//   - geometry uses motion_state_t snapshots
// =============================================================================

#include "compat.hpp"
#include "compute/cfd.hpp"
#include "compute/numerics.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "ecs/components.hpp"
#include "ecs/geometry_visitor.hpp"
#include "functional/fp.hpp"
#include "geometry/block_geometry.hpp"
#include "geometry/boundary/driver.hpp"
#include "grid/amr/api.hpp"
#include "grid/connectivity.hpp"
#include "io/exceptions.hpp"
#include "physics/em/api.hpp"
#include "physics/hydro/boundary_policy.hpp"
#include "physics/ib/collection.hpp"
#include "physics/ib/diagnostics.hpp"
#include "update/prim_recovery.hpp"
#include "update/timestep.hpp"
#include "utility/enums.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace simbi::ecs {

    using namespace simbi::cfd;

    // =========================================================================
    // helper: get motion state for current time
    // =========================================================================
    template <typename Sim>
    geometry::motion_state_t get_motion_state(const Sim& sim)
    {
        if (sim.registry.template has<mesh_motion_config_t>(sim.global)) {
            const auto& motion = sim.registry.template get<mesh_motion_config_t>(sim.global);
            return motion.snapshot(sim.metadata().time);
        }
        return mesh_motion_config_t::static_mesh();
    }

    // =========================================================================
    // body effects system
    //
    // computes gravity and accretion effects for a single partition.
    // caller loops over partitions. returns source term for integration.
    // only finest level at body position gets real diagnostics.
    // =========================================================================
    template <std::uint64_t Rank>
    struct body_effects_system_t
    {
        std::unique_ptr<body::body_diagnostics_t<Rank>> null_diag{nullptr};
        bool                                            update_diagnostics{true};

        template <typename MeshConfig>
        static bool partition_contains(const vector_t<real, Rank>& pos, const MeshConfig& mesh_cfg)
        {
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                if (pos[dd] < mesh_cfg.geometry.dims[dd].start ||
                    pos[dd] >= mesh_cfg.geometry.dims[dd].end) {
                    return false;
                }
            }
            return true;
        }

        template <typename Sim>
        std::uint64_t find_finest_level_at(const Sim& sim, const vector_t<real, Rank>& pos) const
        {
            for (std::int64_t lvl = sim.num_levels() - 1; lvl >= 0; --lvl) {
                for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                    const auto& mesh_cfg = sim.mesh(lvl);
                    if (partition_contains(pos, mesh_cfg)) {
                        return lvl;
                    }
                }
            }
            return 0;
        }

        template <typename Sim>
        bool is_authoritative(const Sim& sim, std::uint64_t lvl) const
        {
            if (!sim.has_bodies()) {
                return false;
            }

            const auto& bodies   = sim.bodies();
            const auto& mesh_cfg = sim.mesh(lvl);

            bool authoritative = false;
            bodies.visit_all([&](const auto& body) {
                auto body_level = find_finest_level_at(sim, body.position);
                if (body_level == lvl && partition_contains(body.position, mesh_cfg)) {
                    authoritative = true;
                }
            });

            return authoritative;
        }

        template <typename Sim, typename Geometry>
        auto operator()(
            Sim&            sim,
            const Geometry& block_geo,
            std::uint64_t   lvl,
            std::uint64_t   pp,
            real            dt
        ) const
        {
            auto& meta   = sim.metadata();
            auto& fields = sim.partition_hydro(lvl, pp);
            auto& part   = sim.partition(lvl, pp);
            if (!sim.has_bodies()) {
                // no bodies: return zero source
                auto& part = sim.partition(lvl, pp);
                return cfd::body_effects(
                    fields.prim[part.owned_domain],
                    part.owned_domain,
                    block_geo,
                    body_collection_t<Rank>{},
                    null_diag.get(),
                    meta.gamma,
                    dt
                );
            }

            const auto& bodies = sim.bodies();
            const bool  auth   = is_authoritative(sim, lvl);

            auto* diag = auth ? (update_diagnostics ? sim.diagnostics().get() : null_diag.get())
                              : null_diag.get();

            return cfd::body_effects(
                fields.prim[part.owned_domain],
                part.owned_domain,
                block_geo,
                bodies,
                diag,
                meta.gamma,
                dt
            );
        }
    };

    // =========================================================================
    // timestep system
    // computes dt for all levels, applies subcycling logic
    // =========================================================================
    struct timestep_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            const auto nlvls = sim.num_levels();
            auto&      meta  = sim.metadata();

            // compute cfl timestep for each level
            for (std::uint64_t lvl = 0; lvl < nlvls; ++lvl) {
                compute_level_dt(sim, lvl);
            }

            // apply subcycling logic
            if (nlvls > 1) {
                apply_subcycling(sim);
            }

            meta.global_dt = meta.level_dts[0];
        }

      private:
        template <typename Sim>
        void compute_level_dt(Sim& sim, std::uint64_t lvl) const
        {
            auto& meta   = sim.metadata();
            auto  motion = get_motion_state(sim);

            meta.level_dts[lvl] = timestep::compute_level_timestep(sim, lvl, motion);
        }

        template <typename Sim>
        void apply_subcycling(Sim& sim) const
        {
            auto&      meta  = sim.metadata();
            const auto nlvls = sim.num_levels();

            if (meta.subcycling_mode == subcycling_mode_t::NONE) {
                // all levels use global minimum
                real dt_min = meta.level_dts[0];
                for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                    dt_min = std::min(dt_min, meta.level_dts[lvl]);
                }
                for (std::uint64_t lvl = 0; lvl < nlvls; ++lvl) {
                    meta.level_dts[lvl] = dt_min;
                }
            }
            else if (meta.subcycling_mode == subcycling_mode_t::STANDARD) {
                subcycle_standard(sim);
            }
            else if (meta.subcycling_mode == subcycling_mode_t::MANUAL) {
                subcycle_manual(sim);
            }
            else if (meta.subcycling_mode == subcycling_mode_t::ADAPTIVE) {
                subcycle_adaptive(sim);
            }
        }

        template <typename Sim>
        void subcycle_standard(Sim& sim) const
        {
            auto&      meta  = sim.metadata();
            const auto nlvls = sim.num_levels();

            // find most restrictive scaled timestep
            real dt_min_scaled = meta.level_dts[0];
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                real cumulative_ratio = 1;
                for (std::uint64_t kk = 1; kk <= lvl; ++kk) {
                    cumulative_ratio *= sim.level_info(kk).refinement_ratio;
                }
                dt_min_scaled = std::min(dt_min_scaled, meta.level_dts[lvl] * cumulative_ratio);
            }

            // set timesteps respecting refinement ratios
            meta.level_dts[0] = dt_min_scaled;
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                const auto ref_ratio = sim.level_info(lvl).refinement_ratio;
                meta.level_dts[lvl]  = meta.level_dts[lvl - 1] / ref_ratio;
            }
        }

        template <typename Sim>
        void subcycle_manual(Sim& sim) const
        {
            auto&      meta  = sim.metadata();
            const auto nlvls = sim.num_levels();

            // use user-provided manual_substeps
            // find global dt from most restrictive level
            real dt_global = meta.level_dts[0];
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                const auto nsteps = meta.level_substeps[lvl];
                dt_global         = std::min(dt_global, meta.level_dts[lvl] * nsteps);
            }

            // set timesteps based on manual substeps
            meta.level_dts[0] = dt_global;
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                const auto nsteps   = meta.level_substeps[lvl];
                meta.level_dts[lvl] = dt_global / nsteps;
            }
        }

        template <typename Sim>
        void subcycle_adaptive(Sim& sim) const
        {
            auto&      meta  = sim.metadata();
            const auto nlvls = sim.num_levels();

            real dt_min = meta.level_dts[0];
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                dt_min = std::min(dt_min, meta.level_dts[lvl]);
            }

            for (std::uint64_t lvl = 0; lvl < nlvls; ++lvl) {
                int nsteps = std::max(1, static_cast<int>(std::ceil(dt_min / meta.level_dts[lvl])));
                meta.level_substeps[lvl] = nsteps;
                meta.level_dts[lvl]      = dt_min / nsteps;
            }

            meta.global_dt = dt_min;
        }
    };

    // =========================================================================
    // conservative to primitive recovery
    // =========================================================================
    struct c2p_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);
                recover_primitives(exec, fields.prim, fields.cons, sim.metadata().gamma);
            }
        }
    };

    // =========================================================================
    // boundary condition system
    // uses geometry/boundary/driver.hpp for physical boundaries
    // =========================================================================
    struct ghost_fill_system_t
    {
        // if true, prolongate from workspace.u_n instead of cons
        bool use_coarse_u_n{false};
        // time interpolation weight: u = (1-alpha)*u_n + alpha*u_current
        // alpha < 0 disables interpolation
        real alpha{-1.0};

        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            if (lvl == 0) {
                // base level: apply physical bcs
                apply_physical_bcs(sim, lvl);
            }
            else {
                // refined levels: prolongate from coarser
                prolongate_from_coarse(sim, lvl);
            }

            // exchange halos between partitions
            sim.exchange_halos(lvl);
        }

        template <typename Sim>
        void prolongate_from_coarse(Sim& sim, std::uint64_t fine_lvl) const
        {
            constexpr auto Rank       = Sim::rank;
            const auto     coarse_lvl = fine_lvl - 1;
            const auto     ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // for each fine partition, prolongate from overlapping coarse
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl); ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                auto& fine_part   = sim.partition(fine_lvl, fp);
                auto& exec        = sim.partition_executor(fine_lvl, fp);

                // find overlapping coarse partition
                std::uint64_t cp = 0;
                if (sim.num_partitions(coarse_lvl) > 1) {
                    cp = fp % sim.num_partitions(coarse_lvl);
                }

                if (cp >= sim.num_partitions(coarse_lvl)) {
                    continue;
                }

                auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);

                // time interpolation enabled and workspace exists?
                const bool do_interpolation = alpha >= 0.0 && sim.has_workspace(coarse_lvl, cp);

                if (do_interpolation) {
                    // interpolate between u_n and current state
                    auto& coarse_ws = sim.workspace(coarse_lvl, cp);
                    auto  u_interp =
                        coarse_ws.u_n.zip(coarse_fields.cons, numerics::time_interpolate_t{alpha})
                            .with(exec);

                    grid::amr::fill_fine_ghosts(
                        fine_fields.cons,
                        u_interp,
                        fine_part.owned_domain,
                        ratio,
                        exec
                    );
                }
                else if (use_coarse_u_n && sim.has_workspace(coarse_lvl, cp)) {
                    // use stored u^n from workspace
                    auto& coarse_ws = sim.workspace(coarse_lvl, cp);
                    grid::amr::fill_fine_ghosts(
                        fine_fields.cons,
                        coarse_ws.u_n,
                        fine_part.owned_domain,
                        ratio,
                        exec
                    );
                }
                else {
                    // use current state in cons
                    grid::amr::fill_fine_ghosts(
                        fine_fields.cons,
                        coarse_fields.cons,
                        fine_part.owned_domain,
                        ratio,
                        exec
                    );
                }
            }
        }

      private:
        template <typename Sim>
        void apply_physical_bcs(Sim& sim, std::uint64_t lvl) const
        {
            constexpr std::uint64_t Rank   = Sim::rank;
            constexpr bool          is_mhd = Sim::is_mhd;

            // create boundary policy for this physics
            auto& mesh_cfg = sim.mesh(lvl);
            auto& geo      = mesh_cfg.geometry;

            // extract theta bounds for spherical coordinate handling
            // theta is x2 (logical), which is at array index Rank-2
            real theta_min = 0.0;
            real theta_max = std::numbers::pi;
            if constexpr (Rank >= 2) {
                constexpr std::uint64_t theta_idx = Rank - 2;
                if (geo.dims.size() > theta_idx) {
                    theta_min = geo.dims[theta_idx].start;
                    theta_max = geo.dims[theta_idx].end;
                }
            }

            auto policy =
                hydro::make_boundary_policy<is_mhd, Rank>(geo.metric, theta_min, theta_max);

            // simple context (no dynamic expressions for now)
            geometry::simple_context_t context;

            auto& decomp = sim.decomposition(lvl);

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                // apply boundaries using the driver
                geometry::boundary_driver_t::apply_boundaries(
                    fields.cons,
                    part.block.id,
                    decomp.skeleton,
                    mesh_cfg,
                    policy,
                    context,
                    exec
                );
            }
        }
    };

    // =========================================================================
    // compute electric fields system
    // computes edge-centered E-fields from fluxes and primitives
    // and stores them in partition_fields.efield
    // =========================================================================
    struct compute_efield_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                // computes E = avg(Flux) + contact_terms
                // stores into fields.efield
                em::compute_edge_efields(
                    exec,
                    fields,
                    part.edge_domains,
                    part.face_domains,
                    part.owned_domain
                );
            }
        }
    };

    // =========================================================================
    // flux computation system
    // =========================================================================
    struct flux_system_t
    {
        template <typename Sim, typename Ops, typename Geometry>
        void
        operator()(Sim& sim, const Ops& ops, const Geometry& block_geo, std::uint64_t lvl) const
        {
            constexpr std::uint64_t rank = Sim::rank;

            auto& meta = sim.metadata();

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                // compute fluxes for each direction
                for (std::uint64_t dir = 0; dir < rank; ++dir) {
                    // note: passing owned_domain for primitives, but iterating
                    // over flux domain (includes ghosts). this works because
                    // primitive view allows access to underlying storage.
                    auto flux_comp = cfd::compute_fluxes(
                        fields.prim[part.owned_domain],
                        // full dace domain included transverse ghosts
                        fields.flux[dir].domain(),
                        block_geo,
                        ops,
                        meta.gamma,
                        meta.plm_theta,
                        meta.viscosity,
                        meta.shock_smoother,
                        dir
                    );

                    // execute and store
                    fields.flux[dir] = flux_comp.with(exec);
                }
            }
        }
    };

    // =========================================================================
    // zero flux buffer for a level (before subcycling)
    // =========================================================================

    // =========================================================================
    // snapshot u^n before subcycling (needed for time interpolation)
    // =========================================================================
    struct snapshot_u_n_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                if (!sim.has_workspace(lvl, pp)) {
                    sim.create_workspace(lvl, pp);
                }
                auto& ws = sim.workspace(lvl, pp);

                // store u^n for time interpolation
                ws.u_n = fields.cons.map(fp::identity).with(exec);
            }
        }
    };

    // =========================================================================
    // euler time integration
    // =========================================================================
    template <typename Ops>
    struct euler_system_t
    {
        Ops ops;

        template <typename Sim, typename Geometry>
        void operator()(Sim& sim, const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto&      meta    = sim.metadata();
            auto&      sources = sim.sources();
            const auto dt      = meta.level_dts[lvl];

            if constexpr (Sim::is_mhd) {
                compute_efield_system_t{}(sim, lvl);
            }

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                // godunov operator L(u)
                auto ell = cfd::godunov_op(fields, part.owned_domain, block_geo, meta, sources);

                auto be = body_effects_system_t<Sim::rank>{}(
                    sim,
                    block_geo,
                    lvl,
                    pp,
                    meta.level_dts[lvl]
                );

                // u^{n+1} = u^n + dt * L(u^n)
                auto u_view = fields.cons[part.owned_domain];
                u_view      = u_view.zip(ell, numerics::euler_step_t{dt})
                             .zip(be, numerics::euler_step_t{dt})
                             .with(exec);

                if constexpr (Sim::is_mhd) {
                    em::update_magnetic_fields(
                        exec,
                        fields,
                        block_geo,
                        part.face_domains,
                        part.owned_domain,
                        dt
                    );
                }
            }
        }
    };

    // =========================================================================
    // rk2 stage 1: u^n -> u*
    // =========================================================================
    template <typename Ops>
    struct rk2_stage1_system_t
    {
        Ops ops;

        template <typename Sim, typename Geometry>
        void operator()(Sim& sim, const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto&      meta    = sim.metadata();
            auto&      sources = sim.sources();
            const auto dt      = meta.level_dts[lvl];
            if constexpr (Sim::is_mhd) {
                compute_efield_system_t{}(sim, lvl);
            }

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                // ensure workspace exists
                if (!sim.has_workspace(lvl, pp)) {
                    sim.create_workspace(lvl, pp);
                }
                auto& ws = sim.workspace(lvl, pp);

                // store u^n
                ws.u_n = fields.cons.map(fp::identity).with(exec);

                // compute L(u^n)
                auto k1 = cfd::godunov_op(fields, part.owned_domain, block_geo, meta, sources);

                auto be = body_effects_system_t<Sim::rank>{
                    .update_diagnostics = false
                }(sim, block_geo, lvl, pp, meta.level_dts[lvl]);

                // u* = u^n + dt * L(u^n)
                auto u_star = fields.cons[part.owned_domain];
                u_star      = u_star.zip(k1, numerics::euler_step_t{dt})
                             .zip(be, numerics::euler_step_t{dt})
                             .with(exec);

                if constexpr (Sim::is_mhd) {
                    // save E^n for rk2 stage 2
                    for (std::uint64_t dd = 0; dd < Sim::rank; ++dd) {
                        ws.e_n[dd] = fields.efield[dd].map(fp::identity).with(exec);
                    }
                }
            }
        }
    };

    // =========================================================================
    // rk2 stage 2 functor (local to this header)
    // =========================================================================
    template <std::uint64_t Rank, typename UStarComp, typename K2Comp, typename BEComp>
    struct rk2_final_stage_t
    {
        UStarComp u_star;
        K2Comp    k2;
        BEComp    be;
        real      dt;

        template <typename ConsT>
        DEV ConsT operator()(const iarray<Rank>& coord, const ConsT& u) const
        {
            using namespace simbi::structs;
            return u | scale_gas(0.5) | add_gas(0.5 * u_star(coord)) |
                   add_gas(0.5 * dt * k2(coord)) | add_gas(0.5 * dt * be(coord));
        }
    };

    // =========================================================================
    // rk2 stage 2 system
    // =========================================================================
    template <typename Ops>
    struct rk2_stage2_system_t
    {
        Ops ops;

        template <typename Sim, typename Geometry>
        void operator()(Sim& sim, const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto&      meta    = sim.metadata();
            auto&      sources = sim.sources();
            const auto dt      = meta.level_dts[lvl];

            if constexpr (Sim::is_mhd) {
                // compute E^{*}
                compute_efield_system_t{}(sim, lvl);
            }

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto& ws     = sim.workspace(lvl, pp);
                auto& exec   = sim.partition_executor(lvl, pp);

                // compute L(u*)
                auto k2 = cfd::godunov_op(fields, part.owned_domain, block_geo, meta, sources);
                auto be = body_effects_system_t<Sim::rank>{
                    .update_diagnostics = true
                }(sim, block_geo, lvl, pp, meta.level_dts[lvl]);

                // u^{n+1} = 0.5 * u^n + 0.5 * (u* + dt * L(u*))
                auto u_n    = ws.u_n[part.owned_domain];
                auto u_star = fields.cons[part.owned_domain];

                rk2_final_stage_t<Sim::rank, decltype(u_star), decltype(k2), decltype(be)>
                    rk2_combine{u_star, k2, be, dt};

                u_n = u_n.enum_map(rk2_combine).with(exec);

                // copy result back
                fields.cons = ws.u_n.map(fp::identity).with(exec);

                if constexpr (Sim::is_mhd) {
                    // compute E^{n+1/2} = 0.5 * (E^n + E^*)
                    for (std::uint64_t dd = 0; dd < Sim::rank; ++dd) {
                        auto en = fields.efield[dd];
                        en      = ws.e_n[dd].zip(en, fp::average_op).with(exec);
                    }

                    // update magnetic fields using rk2 E-fields
                    em::update_magnetic_fields(
                        exec,
                        fields,
                        block_geo,
                        part.face_domains,
                        part.owned_domain,
                        dt
                    );
                }
            }
        }
    };

    // =========================================================================
    // restriction (fine -> coarse)
    // averages fine cells onto overlapping coarse cells
    // =========================================================================
    struct restriction_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return; // nothing coarser
            }

            constexpr auto Rank       = Sim::rank;
            const auto     coarse_lvl = fine_lvl - 1;
            const auto     ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // for each fine partition, restrict to overlapping coarse partition
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl); ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);

                // find overlapping coarse partition
                std::uint64_t cp = find_coarse_partition(sim, fine_lvl, fp);

                if (cp >= sim.num_partitions(coarse_lvl)) {
                    continue; // no valid coarse partition found
                }

                auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);
                auto& fine_part     = sim.partition(fine_lvl, fp);

                // use coarse executor since we're writing to coarse field
                auto& exec = sim.partition_executor(coarse_lvl, cp);

                // restrict fine owned cells -> coarse owned cells (no ghosts)
                auto fine_owned_cons = fine_fields.cons[fine_part.owned_domain];

                grid::amr::restrict_to_coarse(coarse_fields.cons, fine_owned_cons, ratio, exec);
            }
        }

      private:
        // find which coarse partition overlaps the given fine partition
        template <typename Sim>
        static std::uint64_t
        find_coarse_partition(Sim& sim, std::uint64_t fine_lvl, std::uint64_t fine_part_idx)
        {
            constexpr auto Rank       = Sim::rank;
            const auto     coarse_lvl = fine_lvl - 1;

            // for single partition, always 0
            if (sim.num_partitions(coarse_lvl) == 1) {
                return 0;
            }

            // spatial overlap detection via domain intersection
            const auto   ref_ratio = sim.level_info(fine_lvl).refinement_ratio;
            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            auto& fine_part = sim.partition(fine_lvl, fine_part_idx);
            // map fine domain to coarse coordinates
            auto fine_in_coarse_coords =
                grid::amr::scale_domain_down(fine_part.owned_domain, ratio);

            // find coarse partition with spatial overlap
            for (std::uint64_t cp = 0; cp < sim.num_partitions(coarse_lvl); ++cp) {
                auto& coarse_part = sim.partition(coarse_lvl, cp);
                using namespace grid::domain_algebra;
                auto overlap = intersection(coarse_part.owned_domain, fine_in_coarse_coords);
                if (!overlap.empty()) {
                    return cp;
                }
            }

            // no overlap found - this should not happen with proper nesting
            throw std::runtime_error(
                "find_coarse_partition: no coarse partition overlaps fine "
                "partition"
            );
        }
    };

    // =========================================================================
    // prolongation (coarse -> fine ghosts)
    // fills fine level ghost cells by interpolating from coarse
    // =========================================================================
    struct prolongation_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return; // base level uses physical bcs
            }

            constexpr auto Rank       = Sim::rank;
            const auto     coarse_lvl = fine_lvl - 1;
            const auto     ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // for each fine partition, prolongate from coarse
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl); ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                auto& fine_part   = sim.partition(fine_lvl, fp);
                auto& exec        = sim.partition_executor(fine_lvl, fp);

                // find overlapping coarse partition
                std::uint64_t cp = find_coarse_partition(sim, fine_lvl, fp);

                if (cp >= sim.num_partitions(coarse_lvl)) {
                    continue;
                }

                auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);

                // fill fine ghosts from coarse
                grid::amr::fill_fine_ghosts(
                    fine_fields.cons,
                    coarse_fields.cons,
                    fine_part.owned_domain,
                    ratio,
                    exec
                );
            }
        }

      private:
        template <typename Sim>
        static std::uint64_t
        find_coarse_partition(Sim& sim, std::uint64_t fine_lvl, std::uint64_t fine_part_idx)
        {
            const auto coarse_lvl = fine_lvl - 1;

            if (sim.num_partitions(coarse_lvl) == 1) {
                return 0;
            }

            return fine_part_idx % sim.num_partitions(coarse_lvl);
        }
    };

    // =========================================================================
    // synchronize all partitions of a level
    // =========================================================================
    struct synchronize_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& exec = sim.partition_executor(lvl, pp);
                exec.sync();
            }
        }
    };

    // =========================================================================
    // flux register initialization
    // creates flux registers for coarse-fine boundaries
    // =========================================================================
    struct init_flux_registers_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return; // no coarser level
            }
            if (!sim.has_refinement()) {
                return; // no refinement in sim
            }

            constexpr auto Rank = Sim::rank;
            using cons_t        = typename Sim::conserved_t;

            const auto coarse_lvl = fine_lvl - 1;
            const auto ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // get or create flux register component for the fine level
            // auto& decomp = sim.decomposition(fine_lvl);

            if (!sim.has_flux_register(fine_lvl)) {
                flux_register_component_t<cons_t, Rank> flux_regs;
                flux_regs.ratio = ratio;

                // create one register per coarse partition that borders
                // fine
                auto& coarse_decomp = sim.decomposition(coarse_lvl);
                for (std::uint64_t cp = 0; cp < coarse_decomp.num_partitions(); ++cp) {
                    auto& coarse_part = coarse_decomp.partitions[cp];

                    // runtime locality check: ensure that the coarse
                    // partition's field storage locality matches the partition
                    // stream locality. this guards against scheduling kernel
                    // execution on an executor that cannot directly write into
                    // the register backing storage (important for single-device
                    // correctness).
                    flux_regs.registers.emplace_back(coarse_part.owned_domain, ratio);
                }

                flux_regs.initialized = true;
                sim.registry.add(sim.level_entity(fine_lvl), std::move(flux_regs));
            }
        }
    };

    // =========================================================================
    // zero flux registers
    // must be called at the start of each coarse timestep
    // =========================================================================
    struct zero_flux_registers_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return;
            }

            if (!sim.has_flux_register(fine_lvl)) {
                return;
            }

            auto&      flux_regs  = sim.flux_register(fine_lvl);
            const auto coarse_lvl = fine_lvl - 1;

            // zero all registers using executor from first coarse partition
            for (std::uint64_t cp = 0; cp < sim.num_partitions(coarse_lvl); ++cp) {
                auto& exec = sim.partition_executor(coarse_lvl, cp);
                flux_regs.registers[cp].zero_all(exec);
            }
        }
    };

    // =========================================================================
    // flux register accumulation (coarse side)
    // accumulates -F_coarse * dt into registers
    // =========================================================================
    struct accumulate_coarse_flux_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t coarse_lvl, std::uint64_t fine_lvl, real dt) const
        {
            constexpr auto Rank = Sim::rank;
            if (!sim.has_refinement()) {
                return;
            }

            // get flux register for the fine level
            if (!sim.has_flux_register(fine_lvl)) {
                return;
            }

            auto& flux_regs = sim.flux_register(fine_lvl);
            auto& mesh_cfg  = sim.mesh(coarse_lvl);
            auto  motion    = get_motion_state(sim);

            // build geometry for coarse level
            with_block_geometry<Sim::coord_system>(mesh_cfg, motion, [&](const auto& block_geo) {
                // for each coarse partition that borders fine region
                for (std::uint64_t cp = 0; cp < sim.num_partitions(coarse_lvl); ++cp) {
                    auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);
                    auto& exec          = sim.partition_executor(coarse_lvl, cp);

                    // accumulate coarse flux for each dimension
                    for (std::uint64_t dim = 0; dim < Rank; ++dim) {
                        // left face
                        flux_regs.registers[cp].accumulate_coarse(
                            exec,
                            coarse_fields.flux[dim],
                            block_geo,
                            dim,
                            grid::side_t::left,
                            dt
                        );

                        // right face
                        flux_regs.registers[cp].accumulate_coarse(
                            exec,
                            coarse_fields.flux[dim],
                            block_geo,
                            dim,
                            grid::side_t::right,
                            dt
                        );
                    }
                }
            });
        }
    };

    // =========================================================================
    // flux register accumulation (fine side)
    // accumulates +average(F_fine) * dt into registers
    // =========================================================================
    struct accumulate_fine_flux_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl, real dt) const
        {
            constexpr auto Rank = Sim::rank;

            if (fine_lvl == 0) {
                return;
            }

            // get flux register
            if (!sim.has_flux_register(fine_lvl)) {
                return;
            }

            auto& flux_regs = sim.flux_register(fine_lvl);

            const auto coarse_lvl = fine_lvl - 1;
            auto&      mesh_cfg   = sim.mesh(fine_lvl);
            auto       motion     = get_motion_state(sim);

            // build geometry for fine level
            with_block_geometry<Sim::coord_system>(mesh_cfg, motion, [&](const auto& block_geo) {
                // for each fine partition
                for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl); ++fp) {
                    auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                    auto& exec        = sim.partition_executor(fine_lvl, fp);

                    // determine which coarse partition this fine partition
                    // overlaps for single-partition case, it's always 0
                    std::uint64_t cp = 0;
                    if (sim.num_partitions(coarse_lvl) > 1) {
                        // multi-partition: need to find overlapping coarse
                        // partition for now, assume 1:1 mapping
                        // (simplification)
                        cp = fp % sim.num_partitions(coarse_lvl);
                    }

                    // accumulate fine flux for each dimension
                    for (std::uint64_t dim = 0; dim < Rank; ++dim) {
                        flux_regs.registers[cp].accumulate_fine(
                            exec,
                            fine_fields.flux[dim],
                            block_geo,
                            dim,
                            grid::side_t::left,
                            dt
                        );

                        flux_regs.registers[cp].accumulate_fine(
                            exec,
                            fine_fields.flux[dim],
                            block_geo,
                            dim,
                            grid::side_t::right,
                            dt
                        );
                    }
                }
            });
        }
    };

    // =========================================================================
    // reflux system
    // applies accumulated flux mismatch to coarse level conserved variables
    // call after fine level completes all subcycles
    // =========================================================================
    struct reflux_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return;
            }

            // get flux register
            if (!sim.has_flux_register(fine_lvl)) {
                return;
            }

            auto& flux_regs = sim.flux_register(fine_lvl);

            const auto coarse_lvl = fine_lvl - 1;
            auto&      mesh_cfg   = sim.mesh(coarse_lvl);
            auto       motion     = get_motion_state(sim);

            // build geometry and apply correction
            with_block_geometry<Sim::coord_system>(mesh_cfg, motion, [&](const auto& block_geo) {
                for (std::uint64_t cp = 0; cp < sim.num_partitions(coarse_lvl); ++cp) {
                    auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);
                    auto& exec          = sim.partition_executor(coarse_lvl, cp);

                    grid::amr::apply_flux_correction(
                        coarse_fields.cons,
                        flux_regs.registers[cp],
                        block_geo,
                        exec
                    );
                }
            });
        }
    };

    // =======================================================================
    // sink cache system`
    // =======================================================================
    struct sink_cache_system_t
    {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            body::update_sink_cache(sim);
        }
    };

} // namespace simbi::ecs

#endif // SYSTEMS_HPP
