#ifndef SYSTEMS_HPP
#define SYSTEMS_HPP

// =============================================================================
// nsystems.hpp
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
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "ecs/components.hpp"
#include "ecs/geometry_visitor.hpp"
#include "geometry/block_geometry.hpp"
#include "geometry/boundary/driver.hpp"
#include "grid/amr/api.hpp"
#include "grid/amr/flux_correction.hpp"
#include "grid/connectivity.hpp"
#include "io/exceptions.hpp"
#include "physics/em/ct_updater.hpp"
#include "physics/hydro/boundary_policy.hpp"
#include "physics/ib/diagnostics.hpp"
#include "update/prim_recovery.hpp"
#include "update/timestep.hpp"
#include "utility/enums.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

namespace simbi::ecs {

    using namespace simbi::cfd;

    // =========================================================================
    // helper: get motion state for current time
    // =========================================================================
    template <typename Sim>
    geometry::motion_state_t get_motion_state(const Sim& sim)
    {
        if (sim.registry.template has<mesh_motion_config_t>(sim.global)) {
            const auto& motion =
                sim.registry.template get<mesh_motion_config_t>(sim.global);
            return motion.snapshot(sim.metadata().time);
        }
        return mesh_motion_config_t::static_mesh();
    }

    // =========================================================================
    // timestep system
    // computes dt for all levels, applies subcycling logic
    // =========================================================================
    struct timestep_system_t {
        template <typename Sim>
        void operator()(Sim& sim) const
        {
            const auto nlvls = sim.num_levels();
            auto& meta       = sim.metadata();

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
            auto& meta  = sim.metadata();
            auto motion = get_motion_state(sim);

            meta.level_dts[lvl] =
                timestep::compute_level_timestep(sim, lvl, motion);
        }

        template <typename Sim>
        void apply_subcycling(Sim& sim) const
        {
            auto& meta       = sim.metadata();
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
            else if (meta.subcycling_mode == subcycling_mode_t::ADAPTIVE) {
                subcycle_adaptive(sim);
            }
        }

        template <typename Sim>
        void subcycle_standard(Sim& sim) const
        {
            auto& meta       = sim.metadata();
            const auto nlvls = sim.num_levels();

            // find most restrictive scaled timestep
            real dt_min_scaled = meta.level_dts[0];
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                real cumulative_ratio = 1;
                for (std::uint64_t kk = 1; kk <= lvl; ++kk) {
                    cumulative_ratio *= sim.level_info(kk).refinement_ratio;
                }
                dt_min_scaled = std::min(
                    dt_min_scaled,
                    meta.level_dts[lvl] * cumulative_ratio
                );
            }

            // set timesteps respecting refinement ratios
            meta.level_dts[0] = dt_min_scaled;
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                const auto ref_ratio = sim.level_info(lvl).refinement_ratio;
                meta.level_dts[lvl]  = meta.level_dts[lvl - 1] / ref_ratio;
            }
        }

        template <typename Sim>
        void subcycle_adaptive(Sim& sim) const
        {
            auto& meta       = sim.metadata();
            const auto nlvls = sim.num_levels();

            real dt_min = meta.level_dts[0];
            for (std::uint64_t lvl = 1; lvl < nlvls; ++lvl) {
                dt_min = std::min(dt_min, meta.level_dts[lvl]);
            }

            for (std::uint64_t lvl = 0; lvl < nlvls; ++lvl) {
                int nsteps = std::max(
                    1,
                    static_cast<int>(std::ceil(dt_min / meta.level_dts[lvl]))
                );
                meta.level_substeps[lvl] = nsteps;
                meta.level_dts[lvl]      = dt_min / nsteps;
            }

            meta.global_dt = dt_min;
        }
    };

    // =========================================================================
    // conservative to primitive recovery
    // =========================================================================
    struct c2p_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);
                recover_primitives(
                    exec,
                    fields.prim,
                    fields.cons,
                    sim.metadata().gamma
                );
            }
        }
    };

    // =========================================================================
    // boundary condition system
    // uses geometry/boundary/driver.hpp for physical boundaries
    // =========================================================================
    struct ghost_fill_system_t {
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

      private:
        template <typename Sim>
        void apply_physical_bcs(Sim& sim, std::uint64_t lvl) const
        {
            constexpr std::uint64_t Rank = Sim::rank;
            constexpr bool is_mhd        = Sim::is_mhd;

            // create boundary policy for this physics
            auto policy = hydro::make_boundary_policy<is_mhd, Rank>();

            // simple context (no dynamic expressions for now)
            geometry::simple_context_t context;

            auto& decomp = sim.decomposition(lvl);

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);

                // get mesh config for this partition
                auto& mesh_cfg = sim.mesh(lvl);

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

        template <typename Sim>
        void prolongate_from_coarse(Sim& sim, std::uint64_t fine_lvl) const
        {
            constexpr auto Rank   = Sim::rank;
            const auto coarse_lvl = fine_lvl - 1;
            const auto ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // for each fine partition, prolongate from overlapping coarse
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl);
                 ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                auto& fine_part   = sim.partition(fine_lvl, fp);
                auto exec         = sim.partition_executor(fine_lvl, fp);

                // find overlapping coarse partition (simplified mapping)
                std::uint64_t cp = 0;
                if (sim.num_partitions(coarse_lvl) > 1) {
                    cp = fp % sim.num_partitions(coarse_lvl);
                }

                if (cp >= sim.num_partitions(coarse_lvl)) {
                    continue;
                }

                auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);

                grid::amr::fill_fine_ghosts(
                    fine_fields.cons,
                    coarse_fields.cons,
                    fine_part.owned_domain,
                    ratio,
                    exec
                );
            }
        }
    };

    // =========================================================================
    // flux computation system
    // =========================================================================
    struct flux_system_t {
        bool accumulate_fluxes = false;

        template <typename Sim, typename Ops, typename Geometry>
        void operator()(
            Sim& sim,
            const Ops& ops,
            const Geometry& block_geo,
            std::uint64_t lvl
        ) const
        {
            constexpr std::uint64_t rank = Sim::rank;

            auto& meta    = sim.metadata();
            const auto dt = meta.level_dts[lvl];

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);

                // compute fluxes for each direction
                for (std::uint64_t dir = 0; dir < rank; ++dir) {
                    // face domain for this direction
                    auto face_domain = part.owned_domain;
                    face_domain.fin[dir] += 1;

                    auto flux_comp = cfd::compute_fluxes(
                        fields.prim[part.allocated_domain],
                        face_domain,
                        block_geo,
                        ops,
                        meta.gamma,
                        meta.plm_theta,
                        meta.viscosity,
                        meta.shock_smoother,
                        dir
                    );

                    // execute and store
                    fields.flux[dir][face_domain] = flux_comp.with(exec);

                    if (accumulate_fluxes && sim.has_refinement()) {
                        // accumulate time-weighted flux for reflux
                        auto flux_view = fields.flux[dir][face_domain];
                        auto avg_view  = fields.flux_avg[dir][face_domain];
                        avg_view       = avg_view
                                       .zip(
                                           flux_view,
                                           [dt](auto avg, auto f) {
                                               return avg + f * dt;
                                           }
                                       )
                                       .with(exec);
                    }
                }
            }
        }
    };

    // =========================================================================
    // zero flux buffer for a level (before subcycling)
    // =========================================================================
    struct zero_flux_buffer_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            std::cout << "Zeroing flux buffers at level " << lvl << "\n";
            std::cout << "Has refinement: " << sim.has_refinement() << "\n";
            if (!sim.has_refinement()) {
                return;
            }

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);

                for (std::uint64_t dir = 0; dir < Sim::rank; ++dir) {
                    fields.flux_avg[dir] =
                        fields.flux_avg[dir].map(
                                                [](auto f) { return f * 0.0; }
                        ).with(exec);
                }
            }
        }
    };

    // =========================================================================
    // euler time integration
    // =========================================================================
    template <typename Ops>
    struct euler_system_t {
        Ops ops;

        template <typename Sim, typename Geometry>
        void
        operator()(Sim& sim, const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto& meta    = sim.metadata();
            auto& sources = sim.sources();
            const auto dt = meta.level_dts[lvl];

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);

                // godunov operator L(u)
                auto ell = cfd::godunov_op(
                    fields,
                    part.owned_domain,
                    block_geo,
                    meta,
                    sources
                );

                // u^{n+1} = u^n + dt * L(u^n)
                auto u_view = fields.cons[part.owned_domain];
                u_view      = u_view
                             .enum_map([ell, dt](auto coord, auto u) {
                                 return u | add_gas(ell(coord) * dt);
                             })
                             .with(exec);

                if constexpr (Sim::is_mhd) {
                    // update magnetic fields via constrained transport
                    em::update_magnetic_fields(
                        exec,
                        fields,
                        block_geo,
                        part.face_domains,
                        part.owned_domain,
                        dt
                    );

                    // correct energy density for updated B field
                    em::update_energy_density(
                        exec,
                        fields.cons,
                        fields.bfield,
                        block_geo,
                        part.face_domains,
                        part.owned_domain
                    );
                }
            }
        }
    };

    // =========================================================================
    // rk2 stage 1: u^n -> u*
    // =========================================================================
    template <typename Ops>
    struct rk2_stage1_system_t {
        Ops ops;

        template <typename Sim, typename Geometry>
        void
        operator()(Sim& sim, const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto& meta    = sim.metadata();
            auto& sources = sim.sources();
            const auto dt = meta.level_dts[lvl];

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);

                // ensure workspace exists
                if (!sim.has_workspace(lvl, pp)) {
                    sim.create_workspace(lvl, pp);
                }
                auto& workspace = sim.workspace(lvl, pp);

                // store u^n
                workspace.u_n =
                    fields.cons.map([](auto u) { return u; }).with(exec);

                // compute L(u^n)
                auto k1 = cfd::godunov_op(
                    fields,
                    part.owned_domain,
                    block_geo,
                    meta,
                    sources
                );

                // u* = u^n + dt * L(u^n)
                auto u_star = fields.cons[part.owned_domain];
                u_star      = u_star
                             .enum_map([k1, dt](auto coord, auto u) {
                                 return u | add_gas(k1(coord) * dt);
                             })
                             .with(exec);

                if constexpr (Sim::is_mhd) {
                    // update magnetic fields via constrained transport (stage
                    // 1)
                    em::update_magnetic_fields(
                        exec,
                        fields,
                        block_geo,
                        part.face_domains,
                        part.owned_domain,
                        dt
                    );

                    // correct energy density for updated B field
                    em::update_energy_density(
                        exec,
                        fields.cons,
                        fields.bfield,
                        block_geo,
                        part.face_domains,
                        part.owned_domain
                    );
                }
            }
        }
    };

    // =========================================================================
    // rk2 stage 2: u* -> u^{n+1}
    // =========================================================================
    template <typename Ops>
    struct rk2_stage2_system_t {
        Ops ops;

        template <typename Sim, typename Geometry>
        void
        operator()(Sim& sim, const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace simbi::structs;

            auto& meta    = sim.metadata();
            auto& sources = sim.sources();
            const auto dt = meta.level_dts[lvl];

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields    = sim.partition_hydro(lvl, pp);
                auto& part      = sim.partition(lvl, pp);
                auto& workspace = sim.workspace(lvl, pp);
                auto exec       = sim.partition_executor(lvl, pp);

                // compute L(u*)
                auto k2 = cfd::godunov_op(
                    fields,
                    part.owned_domain,
                    block_geo,
                    meta,
                    sources
                );

                // u^{n+1} = 0.5 * u^n + 0.5 * (u* + dt * L(u*))
                auto u_n    = workspace.u_n[part.owned_domain];
                auto u_star = fields.cons[part.owned_domain];

                u_n = u_n.enum_map(
                             [u_star, k2, dt](auto coord, auto u) {
                                 return u | scale_gas(0.5) |
                                        add_gas(0.5 * u_star(coord)) |
                                        add_gas(0.5 * dt * k2(coord));
                             }
                ).with(exec);

                // copy result back
                fields.cons =
                    workspace.u_n.map([](auto u) { return u; }).with(exec);

                if constexpr (Sim::is_mhd) {
                    // update magnetic fields via constrained transport (stage
                    // 2)
                    em::update_magnetic_fields(
                        exec,
                        fields,
                        block_geo,
                        part.face_domains,
                        part.owned_domain,
                        dt
                    );

                    // correct energy density for updated B field
                    em::update_energy_density(
                        exec,
                        fields.cons,
                        fields.bfield,
                        block_geo,
                        part.face_domains,
                        part.owned_domain
                    );
                }
            }
        }
    };

    // =========================================================================
    // restriction (fine -> coarse)
    // averages fine cells onto overlapping coarse cells
    // =========================================================================
    struct restriction_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return;   // nothing coarser
            }

            constexpr auto Rank   = Sim::rank;
            const auto coarse_lvl = fine_lvl - 1;
            const auto ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // for each fine partition, restrict to overlapping coarse
            // partition
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl);
                 ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                // auto& fine_part   = sim.partition(fine_lvl, fp);
                auto exec = sim.partition_executor(fine_lvl, fp);

                // find overlapping coarse partition
                // for aligned grids, we can compute this from topology
                std::uint64_t cp = find_coarse_partition(sim, fine_lvl, fp);

                if (cp >= sim.num_partitions(coarse_lvl)) {
                    continue;   // no valid coarse partition found
                }

                auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);

                // restrict fine -> coarse
                grid::amr::restrict_to_coarse(
                    coarse_fields.cons,
                    fine_fields.cons,
                    ratio,
                    exec
                );
            }
        }

      private:
        // find which coarse partition overlaps the given fine partition
        template <typename Sim>
        static std::uint64_t find_coarse_partition(
            Sim& sim,
            std::uint64_t fine_lvl,
            std::uint64_t fine_part_idx
        )
        {
            const auto coarse_lvl = fine_lvl - 1;

            // for single partition, always 0
            if (sim.num_partitions(coarse_lvl) == 1) {
                return 0;
            }

            // for aligned multi-partition grids, the fine partition
            // maps to coarse partition based on refinement ratio and
            // topology simplified: assume 1:1 correspondence
            return fine_part_idx % sim.num_partitions(coarse_lvl);
        }
    };

    // =========================================================================
    // prolongation (coarse -> fine ghosts)
    // fills fine level ghost cells by interpolating from coarse
    // =========================================================================
    struct prolongation_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return;   // base level uses physical bcs
            }

            constexpr auto Rank   = Sim::rank;
            const auto coarse_lvl = fine_lvl - 1;
            const auto ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // for each fine partition, prolongate from coarse
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl);
                 ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                auto& fine_part   = sim.partition(fine_lvl, fp);
                auto exec         = sim.partition_executor(fine_lvl, fp);

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
        static std::uint64_t find_coarse_partition(
            Sim& sim,
            std::uint64_t fine_lvl,
            std::uint64_t fine_part_idx
        )
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
    struct synchronize_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t lvl) const
        {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& part = sim.partition(lvl, pp);
                part.stream.synchronize();
            }
        }
    };

    // =========================================================================
    // flux register initialization
    // creates flux registers for coarse-fine boundaries
    // =========================================================================
    struct init_flux_registers_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            if (fine_lvl == 0) {
                return;   // no coarser level
            }

            constexpr auto Rank = Sim::rank;
            using cons_t        = typename Sim::conserved_t;

            const auto coarse_lvl = fine_lvl - 1;
            const auto ref_ratio  = sim.level_info(fine_lvl).refinement_ratio;

            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(ref_ratio));

            // get or create flux register component for the fine level
            auto& decomp = sim.decomposition(fine_lvl);

            if (!sim.registry
                     .template has<flux_register_component_t<cons_t, Rank>>(
                         sim.level_entity(fine_lvl)
                     )) {
                flux_register_component_t<cons_t, Rank> flux_regs;
                flux_regs.ratio = ratio;

                // create one register per coarse partition that borders
                // fine
                auto& coarse_decomp = sim.decomposition(coarse_lvl);
                for (std::uint64_t cp = 0; cp < coarse_decomp.num_partitions();
                     ++cp) {
                    auto& coarse_part = coarse_decomp.partitions[cp];
                    flux_regs.registers.emplace_back(
                        coarse_part.owned_domain,
                        ratio
                    );
                }

                flux_regs.initialized = true;
                sim.registry.add(
                    sim.level_entity(fine_lvl),
                    std::move(flux_regs)
                );
            }
        }
    };

    // =========================================================================
    // flux register accumulation (coarse side)
    // accumulates -F_coarse * dt into registers
    // =========================================================================
    struct accumulate_coarse_flux_system_t {
        template <typename Sim>
        void operator()(
            Sim& sim,
            std::uint64_t coarse_lvl,
            std::uint64_t fine_lvl,
            real dt
        ) const
        {
            constexpr auto Rank = Sim::rank;
            using cons_t        = typename Sim::conserved_t;

            // get flux register for the fine level
            if (!sim.registry
                     .template has<flux_register_component_t<cons_t, Rank>>(
                         sim.level_entity(fine_lvl)
                     )) {
                return;
            }

            auto& flux_regs =
                sim.registry
                    .template get<flux_register_component_t<cons_t, Rank>>(
                        sim.level_entity(fine_lvl)
                    );

            // for each coarse partition that borders fine region
            for (std::uint64_t cp = 0; cp < sim.num_partitions(coarse_lvl);
                 ++cp) {
                auto& coarse_fields = sim.partition_hydro(coarse_lvl, cp);
                auto exec           = sim.partition_executor(coarse_lvl, cp);

                // accumulate coarse flux for each dimension
                for (std::uint64_t dim = 0; dim < Rank; ++dim) {
                    // left face
                    flux_regs.registers[cp].accumulate_coarse(
                        exec,
                        coarse_fields.flux[dim],
                        dim,
                        grid::side_t::left,
                        dt
                    );

                    // right face
                    flux_regs.registers[cp].accumulate_coarse(
                        exec,
                        coarse_fields.flux[dim],
                        dim,
                        grid::side_t::right,
                        dt
                    );
                }
            }
        }
    };

    // =========================================================================
    // flux register accumulation (fine side)
    // accumulates +average(F_fine) * dt into registers
    // =========================================================================
    struct accumulate_fine_flux_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl, real dt) const
        {
            constexpr auto Rank = Sim::rank;
            using cons_t        = typename Sim::conserved_t;

            if (fine_lvl == 0) {
                return;
            }

            // get flux register
            if (!sim.registry
                     .template has<flux_register_component_t<cons_t, Rank>>(
                         sim.level_entity(fine_lvl)
                     )) {
                return;
            }

            auto& flux_regs =
                sim.registry
                    .template get<flux_register_component_t<cons_t, Rank>>(
                        sim.level_entity(fine_lvl)
                    );

            const auto coarse_lvl = fine_lvl - 1;

            // for each fine partition
            for (std::uint64_t fp = 0; fp < sim.num_partitions(fine_lvl);
                 ++fp) {
                auto& fine_fields = sim.partition_hydro(fine_lvl, fp);
                auto exec         = sim.partition_executor(fine_lvl, fp);

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
                        dim,
                        grid::side_t::left,
                        dt
                    );

                    flux_regs.registers[cp].accumulate_fine(
                        exec,
                        fine_fields.flux[dim],
                        dim,
                        grid::side_t::right,
                        dt
                    );
                }
            }
        }
    };

    // =========================================================================
    // reflux system
    // applies accumulated flux mismatch to coarse level conserved variables
    // call after fine level completes all subcycles
    // =========================================================================
    struct reflux_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            constexpr auto Rank = Sim::rank;
            using cons_t        = typename Sim::conserved_t;

            if (fine_lvl == 0) {
                return;
            }

            // get flux register
            if (!sim.registry
                     .template has<flux_register_component_t<cons_t, Rank>>(
                         sim.level_entity(fine_lvl)
                     )) {
                return;
            }

            auto& flux_regs =
                sim.registry
                    .template get<flux_register_component_t<cons_t, Rank>>(
                        sim.level_entity(fine_lvl)
                    );

            const auto coarse_lvl = fine_lvl - 1;
            auto& mesh_cfg        = sim.mesh(coarse_lvl);
            auto motion           = get_motion_state(sim);

            // build geometry and apply correction
            with_block_geometry<Sim::coord_system>(
                mesh_cfg,
                motion,
                [&](const auto& block_geo) {
                    for (std::uint64_t cp = 0;
                         cp < sim.num_partitions(coarse_lvl);
                         ++cp) {
                        auto& coarse_fields =
                            sim.partition_hydro(coarse_lvl, cp);
                        auto exec = sim.partition_executor(coarse_lvl, cp);

                        grid::amr::apply_flux_correction(
                            coarse_fields.cons,
                            flux_regs.registers[cp],
                            block_geo,
                            exec
                        );
                    }
                }
            );
        }
    };

    // =========================================================================
    // clear flux registers (before new coarse step)
    // =========================================================================
    struct clear_flux_registers_system_t {
        template <typename Sim>
        void operator()(Sim& sim, std::uint64_t fine_lvl) const
        {
            constexpr auto Rank = Sim::rank;
            using cons_t        = typename Sim::conserved_t;

            if (fine_lvl == 0) {
                return;
            }

            if (!sim.registry
                     .template has<flux_register_component_t<cons_t, Rank>>(
                         sim.level_entity(fine_lvl)
                     )) {
                return;
            }

            // re-initialize registers (cheapest way to clear)
            auto& flux_regs =
                sim.registry
                    .template get<flux_register_component_t<cons_t, Rank>>(
                        sim.level_entity(fine_lvl)
                    );

            const auto coarse_lvl = fine_lvl - 1;
            auto& coarse_decomp   = sim.decomposition(coarse_lvl);

            for (std::uint64_t cp = 0; cp < coarse_decomp.num_partitions();
                 ++cp) {
                auto& coarse_part = coarse_decomp.partitions[cp];
                flux_regs.registers[cp] =
                    grid::amr::flux_register_t<cons_t, Rank>(
                        coarse_part.owned_domain,
                        flux_regs.ratio
                    );
            }
        }
    };

    // =========================================================================
    // body effects system (multi-partition aware)
    //
    // key design:
    //   - all levels compute body effects (gravity is long-range)
    //   - only the finest level containing each body passes real diagnostics
    //   - other levels use null diagnostic sink to avoid double-counting
    // =========================================================================
    template <std::uint64_t Rank>
    struct body_effects_system_t {
        // null diagnostic sink for non-authoritative levels
        std::unique_ptr<body::body_diagnostics_t<Rank>> null_diag{nullptr};

        // check if a position falls within a partition's physical bounds
        template <typename MeshConfig>
        static bool partition_contains(
            const vector_t<real, Rank>& pos,
            const MeshConfig& mesh_cfg
        )
        {
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                if (pos[dd] < mesh_cfg.geometry.dims[dd].start ||
                    pos[dd] >= mesh_cfg.geometry.dims[dd].end) {
                    return false;
                }
            }
            return true;
        }

        // find the finest level containing a position
        template <typename Sim>
        std::uint64_t find_finest_level_at(
            const Sim& sim,
            const vector_t<real, Rank>& pos
        ) const
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

        // check if this level/partition is authoritative for any body
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
                if (body_level == lvl &&
                    partition_contains(body.position, mesh_cfg)) {
                    authoritative = true;
                }
            });

            return authoritative;
        }

        // main entry point: compute body effects for a level
        template <typename Sim, typename Geometry>
        void operator()(
            Sim& sim,
            const Geometry& block_geo,
            std::uint64_t lvl,
            real dt
        ) const
        {
            if (!sim.has_bodies()) {
                return;
            }

            const auto& bodies = sim.bodies();
            auto& meta         = sim.metadata();

            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto& fields = sim.partition_hydro(lvl, pp);
                auto& part   = sim.partition(lvl, pp);
                auto exec    = sim.partition_executor(lvl, pp);

                // only authoritative partition passes real diagnostics
                bool auth  = is_authoritative(sim, lvl);
                auto* diag = auth ? sim.diagnostics().get() : null_diag.get();

                // compute body effects using cfd module
                auto effects = cfd::body_effects(
                    fields.prim[part.allocated_domain],
                    part.owned_domain,
                    block_geo,
                    bodies,
                    diag,
                    meta.gamma,
                    dt
                );

                // apply to conservative variables
                auto u_view = fields.cons[part.owned_domain];
                u_view      = u_view
                             .enum_map([effects](auto coord, auto u) {
                                 return u | add_gas(effects(coord));
                             })
                             .with(exec);
            }
        }
    };

}   // namespace simbi::ecs

#endif   // SYSTEMS_HPP
