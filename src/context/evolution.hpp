// =============================================================================
// evolution.hpp
//
// evolution driver for partition-aware multi-device simulations.
// uses nsystems.hpp and ntiming.hpp infrastructure.
//
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "checkpoint.hpp"
#include "ecs/systems.hpp"
#include "io/console/dprintb.hpp"
#include "io/diagnostics.hpp"
#include "io/exceptions.hpp"
#include "physics/ib/motion.hpp"
#include "progress.hpp"
#include "timing.hpp"
#include "utility/enums.hpp"
#include "utility/helpers.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace simbi::evolution {
    template <typename Sim>
    std::uint64_t count_weighted_zones(const Sim& sim);

    struct evolution_state_t
    {
        timing::timing_stats_t            stats;
        progress::progress_state_t        progress;
        checkpoint::checkpoint_schedule_t schedule;
        bool                              should_stop{false};
    };

    template <typename Sim>
    evolution_state_t initialize(Sim& sim)
    {
        auto& meta = sim.metadata();

        auto state = evolution_state_t{
            .stats    = timing::timing_stats_t{},
            .progress = progress::initialize(meta.regime),
            .schedule =
                checkpoint::checkpoint_schedule_t{
                    .checkpoint_time     = meta.checkpoint_interval,
                    .checkpoint_interval = meta.checkpoint_interval,
                    .dlogt               = meta.dlogt,
                    .tstart              = meta.initial_time,
                    .checkpoint_index    = meta.checkpoint_index
                },
            .should_stop = false
        };

        // mirror all display messages to a log file in the data directory
        if (!meta.data_dir.empty()) {
            auto log_path = std::filesystem::path(meta.data_dir) / "simbi.log";
            state.progress.table.set_log_file(log_path);
        }

        return state;
    }

    template <typename Sim, typename PhysicsStep>
    void run(Sim& sim, PhysicsStep&& step, evolution_state_t& state)
    {
        auto& meta = sim.metadata();

        // get executor from partition 0 level 0 for timing
        auto& exec = sim.partition_executor(0, 0);

        // initial checkpoint
        try {
            if (meta.time == 0.0 || meta.checkpoint_index == 0) {
                if (sim.in_failure_state) {
                    throw exception::SimulationFailureException();
                }
                checkpoint::save(sim, state.progress);
            }
        }
        catch (exception::SimulationFailureException& e) {
            // diagnose and report detailed failure information
            diagnostics::diagnose_cons2prim_failure(sim, state.progress.table);
            throw;
        }

        state.schedule.advance(meta.time, meta.checkpoint_index);
        meta.advance_schedule(state.schedule);

        while (meta.time < meta.tend && !state.should_stop) {
            try {
                // create scoped timer
                std::uint64_t          nzones_weighted = count_weighted_zones(sim);
                timing::scoped_timer_t timer(exec, state.stats, nzones_weighted);

                // run physics step
                step(sim);

                // update progress periodically
                if (meta.iteration % 100 == 0) {
                    auto elapsed = timer.timer_.elapsed_so_far();
                    auto speed   = nzones_weighted / elapsed;
                    progress::update(
                        state.progress,
                        meta.iteration,
                        meta.time,
                        meta.global_dt,
                        meta.tend,
                        speed
                    );
                }

                if (state.schedule.should_checkpoint(meta.time)) {
                    checkpoint::save(sim, state.progress);
                    state.schedule.advance(meta.time, meta.checkpoint_index);
                    meta.advance_schedule(state.schedule);
                }

                meta.iteration++;
                helpers::catch_signals();
            }
            catch (exception::InterruptException& e) {
                sim.was_interrupted = true;
                state.should_stop   = true;
                state.progress.table.post_error(std::string("Interrupted: ") + e.what());
                checkpoint::save(sim, state.progress);
            }
            catch (exception::SimulationFailureException& e) {
                sim.in_failure_state = true;
                state.should_stop    = true;

                // diagnose and report detailed failure information
                diagnostics::diagnose_cons2prim_failure(sim, state.progress.table);

                checkpoint::save(sim, state.progress);
            }
        }
        if (!state.should_stop) {
            state.progress.table.set_progress(100);
        }
        bool successful_sim = !sim.in_failure_state && !sim.was_interrupted;
        progress::finalize(state.progress, successful_sim);

        if (state.stats.count > 0) {
            io::writeln(
                "Average zone update/sec for {:>5} iterations was {:>5.2e} "
                "zones/sec",
                meta.iteration,
                state.stats.average()
            );
        }
    }

    template <typename Sim>
    std::uint64_t count_weighted_zones(const Sim& sim)
    {
        const auto& meta      = sim.metadata();
        const real  dt_coarse = meta.level_dts[0];

        std::uint64_t total = 0;
        for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
            // sum zones across all partitions
            std::uint64_t level_zones = 0;
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                const auto& part = sim.partition(lvl, pp);
                level_zones += part.owned_domain.size();
            }

            // weight by timestep ratio
            const real weight = dt_coarse / meta.level_dts[lvl];
            total += static_cast<std::uint64_t>(level_zones * weight);
        }

        return total;
    }

    template <typename Sim, typename Ops>
    struct hydro_pipeline_t
    {
        Sim&       sim;
        const Ops& ops;

      private:
        // ---------------------------------------------------------------------
        // euler driver
        // ---------------------------------------------------------------------
        void advance_level_euler(std::uint64_t lvl) const
        {
            using namespace ecs;
            auto& meta     = sim.metadata();
            auto  motion   = sim.motion_state();
            auto& mesh_cfg = sim.mesh(lvl);

            with_block_geometry<Sim::coord_system>(mesh_cfg, motion, [&](const auto& block_geo) {
                c2p_system_t{}(sim, lvl, block_geo);
                ghost_fill_system_t{}(sim, lvl);

                // snapshot prim^n after ghost fill so ghost cells are
                // consistent with interior data at t^n
                if (lvl < sim.num_levels() - 1) {
                    const auto nsteps = get_substeps(lvl + 1);
                    if (nsteps > 1) {
                        snapshot_u_n_system_t{}(sim, lvl);
                    }
                }

                sink_cache_system_t{}(sim, lvl);

                flux_system_t{}(sim, ops, block_geo, lvl);

                // accumulate coarse flux before advancing coarse level
                if (lvl < sim.num_levels() - 1) {
                    const auto fine_level = lvl + 1;
                    const auto dt_coarse  = meta.level_dts[lvl];

                    zero_flux_registers_system_t{}(sim, fine_level);
                    accumulate_coarse_flux_system_t{}(sim, lvl, fine_level, dt_coarse);
                }

                euler_system_t<Ops>{ops}(sim, block_geo, lvl);

                // coarse level cons now at u^{n+1}
                // update primitives so time interpolation has correct endpoint
                if (lvl < sim.num_levels() - 1) {
                    const auto nsteps = get_substeps(lvl + 1);
                    if (nsteps > 1) {
                        c2p_system_t{}(sim, lvl, block_geo);
                        // refresh coarse ghost prims so the prolongation slope
                        // stencil sees consistent t^{n+1} data at owned-domain
                        // edges
                        ghost_fill_system_t{}(sim, lvl);
                    }
                }
            });

            if (lvl < sim.num_levels() - 1) {
                const auto fine_level = lvl + 1;
                const auto nsteps     = get_substeps(fine_level);
                const auto dt_fine    = meta.level_dts[fine_level];

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    // time interpolation between prim^n (stored) and prim^{n+1} (current)
                    // alpha = substep/nsteps gives boundary at t^n + substep*dt_fine
                    real alpha =
                        nsteps > 1 ? static_cast<real>(substep) / static_cast<real>(nsteps) : -1.0;

                    if (sim.has_bodies() && alpha >= 0.0) {
                        sim.bodies().interpolate_to(alpha);
                    }

                    ghost_fill_system_t{.alpha = alpha}(sim, fine_level);

                    advance_level_euler(fine_level);

                    // accumulate fine flux at interface
                    accumulate_fine_flux_system_t{}(sim, fine_level, dt_fine);
                }

                // restore bodies to t^n for consistency before restriction
                if (sim.has_bodies()) {
                    sim.bodies().restore_from_snapshot();
                }

                // restriction: inject fine interior back to coarse
                restriction_system_t{}(sim, fine_level);

                // apply flux correction to coarse level
                reflux_system_t{}(sim, fine_level);
            }

            // sync partitions
            synchronize_system_t{}(sim, lvl);
        }

        void advance_level_rk2(std::uint64_t lvl) const
        {
            using namespace ecs;
            auto&      meta     = sim.metadata();
            auto&      mesh_cfg = sim.mesh(lvl);
            auto       motion   = sim.motion_state();
            const auto dt       = meta.level_dts[lvl];

            with_block_geometry<Sim::coord_system>(mesh_cfg, motion, [&](const auto& block_geo) {
                // === STAGE 1: u^n -> u* ===
                c2p_system_t{}(sim, lvl, block_geo);
                ghost_fill_system_t{}(sim, lvl);

                // snapshot prim^n after ghost fill so ghost cells are
                // consistent with interior data at t^n
                if (lvl < sim.num_levels() - 1) {
                    const auto nsteps = get_substeps(lvl + 1);
                    if (nsteps > 1) {
                        snapshot_u_n_system_t{}(sim, lvl);
                    }
                }

                sink_cache_system_t{}(sim, lvl);

                flux_system_t{}(sim, ops, block_geo, lvl); // F(u^n)

                // accumulate this level's flux into parent's register
                if (lvl > 0) {
                    accumulate_fine_flux_system_t{}(sim, lvl, 0.5 * dt);
                }

                if (lvl < sim.num_levels() - 1) {
                    const auto fine_level = lvl + 1;
                    zero_flux_registers_system_t{}(sim, fine_level);
                    // accumulate coarse F(u^n) weighted by 0.5*dt (trapezoidal)
                    accumulate_coarse_flux_system_t{}(sim, lvl, fine_level, 0.5 * dt);
                }

                rk2_stage1_system_t<Ops>{ops}(sim, block_geo, lvl);

                // === STAGE 2: u* -> u^{n+1} ===
                c2p_system_t{}(sim, lvl, block_geo);
                ghost_fill_system_t{}(sim, lvl);
                sink_cache_system_t{}(sim, lvl);

                flux_system_t{}(sim, ops, block_geo, lvl); // F(u*)

                // accumulate this level's flux into parent's register
                if (lvl > 0) {
                    accumulate_fine_flux_system_t{}(sim, lvl, 0.5 * dt);
                }

                if (lvl < sim.num_levels() - 1) {
                    const auto fine_level = lvl + 1;
                    // accumulate coarse F(u*) weighted by 0.5*dt (trapezoidal)
                    accumulate_coarse_flux_system_t{}(sim, lvl, fine_level, 0.5 * dt);
                }
                rk2_stage2_system_t<Ops>{ops}(sim, block_geo, lvl);

                // coarse level cons now at u^{n+1}
                // update prim so time interpolation has correct endpoint (prim^{n+1})
                if (lvl < sim.num_levels() - 1) {
                    const auto nsteps = get_substeps(lvl + 1);
                    if (nsteps > 1) {
                        c2p_system_t{}(sim, lvl, block_geo);
                        // refresh coarse ghost prims so the prolongation slope
                        // stencil sees consistent t^{n+1} data at owned-domain
                        // edges. without this, ghost prims remain at t^n,
                        // creating a time discontinuity that slope-based
                        // prolongation amplifies into artifacts
                        ghost_fill_system_t{}(sim, lvl);
                    }
                }

                // === SUBCYCLE FINE LEVELS ===
                // now that coarse has completed its rk2 step, we have prim^n (stored)
                // and prim^{n+1} (current). subcycle fine with time interpolation.
                if (lvl < sim.num_levels() - 1) {
                    const auto fine_level = lvl + 1;
                    const auto nsteps     = get_substeps(fine_level);

                    for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                        // time interpolation: alpha = substep/nsteps
                        // alpha=0 -> prim^n, alpha=1 -> prim^{n+1}
                        real alpha = nsteps > 1
                                         ? static_cast<real>(substep) / static_cast<real>(nsteps)
                                         : -1.0;

                        if (sim.has_bodies() && alpha >= 0.0) {
                            sim.bodies().interpolate_to(alpha);
                        }

                        ghost_fill_system_t{.alpha = alpha}(sim, fine_level);

                        advance_level_rk2(fine_level);
                    }

                    // restore bodies to t^n for consistency before reflux/restriction
                    if (sim.has_bodies()) {
                        sim.bodies().restore_from_snapshot();
                    }

                    // restriction first: inject fine interior back to coarse
                    restriction_system_t{}(sim, fine_level);

                    // reflux after restriction: correct coarse-fine interface conservation
                    reflux_system_t{}(sim, fine_level);
                }
            });

            synchronize_system_t{}(sim, lvl);
        }

        std::uint64_t get_substeps(std::uint64_t child_lvl) const
        {
            auto& meta = sim.metadata();

            if (meta.subcycling_mode == subcycling_mode_t::NONE) {
                return 1;
            }
            else if (meta.subcycling_mode == subcycling_mode_t::STANDARD) {
                return sim.level_info(child_lvl).refinement_ratio;
            }
            else if (meta.subcycling_mode == subcycling_mode_t::MANUAL) {
                return meta.level_substeps[child_lvl];
            }
            else { // ADAPTIVE
                return meta.level_substeps[child_lvl];
            }
        }

      public:
        // ---------------------------------------------------------------------
        // configure (called before evolution loop)
        // ---------------------------------------------------------------------
        void configure() const
        {
            using namespace ecs;
            for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
                auto& mesh_cfg = sim.mesh(lvl);
                auto  motion   = sim.motion_state();
                with_block_geometry<Sim::coord_system>(
                    mesh_cfg,
                    motion,
                    [&](const auto& block_geo) {
                        c2p_system_t{}(sim, lvl, block_geo);
                        ghost_fill_system_t{}(sim, lvl);
                    }
                );
                if (sim.has_refinement()) {
                    init_flux_registers_system_t{}(sim, lvl);
                }
            }
            timestep_system_t{}(sim);
        }

        // ---------------------------------------------------------------------
        // step_all (one coarse timestep)
        // ---------------------------------------------------------------------
        void step_all() const
        {
            using namespace ecs;
            auto& meta = sim.metadata();

            timestep_system_t{}(sim);

            // snapshot body positions for subcycle interpolation
            if (sim.has_bodies() && sim.has_refinement()) {
                auto advanced = body::compute_advanced_bodies(sim);
                sim.bodies().snapshot(advanced);
            }

            if (meta.timestepping == timestepping_t::RK2) {
                advance_level_rk2(0);
            }
            else if (meta.timestepping == timestepping_t::EULER) {
                advance_level_euler(0);
            }
            else {
                throw std::runtime_error("that timestepping method is not implemented.");
            }

            meta.time += meta.global_dt;

            if (sim.has_bodies()) {
                body::evolve_bodies(sim);
            }
        }
    };

} // namespace simbi::evolution
