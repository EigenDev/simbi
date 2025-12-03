#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP

// =============================================================================
// evolution.hpp
//
// evolution driver for partition-aware multi-device simulations.
// uses nsystems.hpp and ntiming.hpp infrastructure.
//
// key changes from evolution.hpp:
//   - timer requires executor (from partition 0)
//   - zone counting accounts for multiple partitions
//   - pipeline uses partition-aware systems
// =============================================================================

#include "checkpoint.hpp"
#include "compat.hpp"
#include "ecs/systems.hpp"
#include "io/console/printb.hpp"
#include "io/exceptions.hpp"
#include "physics/ib/motion.hpp"
#include "progress.hpp"
#include "timing.hpp"
#include "utility/enums.hpp"
#include "utility/helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace simbi::evolution {
    template <typename Sim>
    std::uint64_t count_weighted_zones(const Sim& sim);

    // =========================================================================
    // evolution state
    // =========================================================================
    struct evolution_state_t {
        timing::timing_stats_t stats;
        progress::progress_state_t progress;
        checkpoint::checkpoint_schedule_t schedule;
        bool should_stop{false};
    };

    // =========================================================================
    // initialization
    // =========================================================================
    template <typename Sim>
    evolution_state_t initialize(Sim& sim, const char* title = "Simulation")
    {
        auto& meta = sim.metadata();

        return {
          .stats    = timing::timing_stats_t{},
          .progress = progress::initialize(title),
          .schedule =
              checkpoint::checkpoint_schedule_t{
                .checkpoint_time     = meta.checkpoint_interval,
                .checkpoint_interval = meta.checkpoint_interval,
                .dlogt               = meta.dlogt,
                .checkpoint_index    = meta.checkpoint_index
              },
          .should_stop = false
        };
    }

    // =========================================================================
    // main evolution loop
    // =========================================================================
    template <typename Sim, typename PhysicsStep>
    void run(Sim& sim, PhysicsStep&& step, evolution_state_t& state)
    {
        auto& meta = sim.metadata();

        // get executor from partition 0 level 0 for timing
        auto exec = sim.partition_executor(0, 0);

        // initial checkpoint
        state.progress.table.refresh();
        if (meta.time == 0.0 || meta.checkpoint_index == 0) {
            if (sim.in_failure_state) {
                throw exception::SimulationFailureException();
            }
            checkpoint::save(sim, state.progress);
        }
        state.schedule.advance(meta.time);
        meta.advance_schedule(state.schedule);

        while (meta.time < meta.tend && !state.should_stop) {
            try {
                // create scoped timer
                std::uint64_t nzones_weighted = count_weighted_zones(sim);
                timing::scoped_timer_t timer(
                    exec,
                    state.stats,
                    nzones_weighted
                );

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
                    state.schedule.advance(meta.time);
                    meta.advance_schedule(state.schedule);
                }

                meta.iteration++;
                helpers::catch_signals();
            }
            catch (exception::InterruptException& e) {
                sim.was_interrupted = true;
                state.should_stop   = true;
                state.progress.table.post_error(
                    std::string("Interrupted: ") + e.what()
                );
                checkpoint::save(sim, state.progress);
            }
            catch (exception::SimulationFailureException& e) {
                sim.in_failure_state = true;
                state.should_stop    = true;
                state.progress.table.post_error(
                    std::string("Failed: ") + e.what()
                );
                checkpoint::save(sim, state.progress);
            }
        }
        if (!state.should_stop) {
            state.progress.table.set_progress(100);
        }
        progress::finalize(state.progress);

        if (state.stats.count > 0) {
            util::writeln(
                "Average zone update/sec for {:>5} iterations was {:>5.2e} "
                "zones/sec",
                meta.iteration,
                state.stats.average()
            );
        }
    }

    // =========================================================================
    // zone counting helper
    // =========================================================================
    template <typename Sim>
    std::uint64_t count_weighted_zones(const Sim& sim)
    {
        const auto& meta     = sim.metadata();
        const real dt_coarse = meta.level_dts[0];

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

    // =========================================================================
    // hydro pipeline
    // orchestrates the time integration with amr subcycling
    // =========================================================================
    template <typename Sim, typename Ops>
    struct hydro_pipeline_t {
        Sim& sim;
        const Ops& ops;

      private:
        // ---------------------------------------------------------------------
        // euler driver
        // ---------------------------------------------------------------------
        void advance_level_euler(std::uint64_t lvl) const
        {
            using namespace ecs;
            auto& meta     = sim.metadata();
            auto motion    = get_motion_state(sim);
            auto& mesh_cfg = sim.mesh(lvl);

            with_block_geometry<Sim::coord_system>(
                mesh_cfg,
                motion,
                [&](const auto& block_geo) {
                    ghost_fill_system_t{}(sim, lvl);
                    c2p_system_t{}(sim, lvl);

                    flux_system_t{}(sim, ops, block_geo, lvl);

                    euler_system_t<Ops>{ops}(sim, block_geo, lvl);
                }
            );

            if (lvl < sim.num_levels() - 1) {
                const auto nsteps     = get_substeps(lvl + 1);
                const auto fine_level = lvl + 1;
                const auto dt_fine    = meta.level_dts[fine_level];
                const auto dt_coarse  = meta.level_dts[lvl];

                // zero flux registers at start of coarse timestep
                zero_flux_registers_system_t{}(sim, fine_level);

                // accumulate coarse flux at interface
                accumulate_coarse_flux_system_t{}(
                    sim,
                    lvl,
                    fine_level,
                    dt_coarse
                );

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    if (nsteps > 1) {
                        // ghost fill logic for subcycling...
                        real alpha = (static_cast<real>(substep) + 1.0) /
                                     static_cast<real>(nsteps);
                        time_interpolated_ghost_fill_system_t{
                          alpha
                        }(sim, fine_level);
                    }
                    advance_level_euler(fine_level);

                    // accumulate fine flux at interface
                    accumulate_fine_flux_system_t{}(sim, fine_level, dt_fine);
                }

                // restriction: inject fine interior back to coarse
                restriction_system_t{}(sim, fine_level);

                // apply flux correction to coarse level
                reflux_system_t{}(sim, fine_level);
            }

            // sync partitions
            synchronize_system_t{}(sim, lvl);
        }

        // =============================================================================
        // rk2 driver (berger-colella)
        // =============================================================================
        void advance_level_rk2(std::uint64_t lvl) const
        {
            using namespace ecs;
            auto& meta     = sim.metadata();
            auto& mesh_cfg = sim.mesh(lvl);
            auto motion    = get_motion_state(sim);
            const auto dt  = meta.level_dts[lvl];

            with_block_geometry<Sim::coord_system>(
                mesh_cfg,
                motion,
                [&](const auto& block_geo) {
                    // === STAGE 1: u^n -> u* ===
                    ghost_fill_system_t{.use_coarse_u_n = true}(sim, lvl);
                    c2p_system_t{}(sim, lvl);

                    flux_system_t{}(sim, ops, block_geo, lvl);   // F(u^n)

                    // accumulate this level's flux into parent's register
                    if (lvl > 0) {
                        accumulate_fine_flux_system_t{}(sim, lvl, 0.5 * dt);
                    }

                    if (lvl < sim.num_levels() - 1) {
                        const auto fine_level = lvl + 1;
                        zero_flux_registers_system_t{}(sim, fine_level);
                        // accumulate coarse F(u^n) (0.5 * dt)
                        accumulate_coarse_flux_system_t{}(
                            sim,
                            lvl,
                            fine_level,
                            0.5 * dt
                        );
                    }

                    rk2_stage1_system_t<Ops>{ops}(sim, block_geo, lvl);

                    // === RECURSION ===
                    if (lvl < sim.num_levels() - 1) {
                        const auto fine_level = lvl + 1;
                        const auto nsteps     = get_substeps(fine_level);

                        for (std::uint64_t substep = 0; substep < nsteps;
                             ++substep) {
                            // not: use simple ghost fill if nsteps == 1 (NONE
                            // mode)
                            if (nsteps > 1) {
                                // ghost fill logic for subcycling...
                                real alpha =
                                    (static_cast<real>(substep) + 0.5) /
                                    static_cast<real>(nsteps);
                                time_interpolated_ghost_fill_system_t{
                                  alpha
                                }(sim, fine_level);
                            }
                            else {
                                // for none mode, just let the next recursive
                                // call handle ghosts, or call ghost_fill here
                                // if needed manually. Standard practice: The
                                // recursive call starts with ghost_fill.
                            }

                            advance_level_rk2(fine_level);
                        }
                    }

                    // === STAGE 2: u* -> u^{n+1} ===
                    ghost_fill_system_t{.use_coarse_u_n = true}(sim, lvl);
                    c2p_system_t{}(sim, lvl);

                    flux_system_t{}(sim, ops, block_geo, lvl);   // F(u*)

                    // accumulate this level's flux into parent's register
                    if (lvl > 0) {
                        accumulate_fine_flux_system_t{}(sim, lvl, 0.5 * dt);
                    }

                    if (lvl < sim.num_levels() - 1) {
                        const auto fine_level = lvl + 1;
                        // accumulate coarse F(u*) (0.5 * dt)
                        accumulate_coarse_flux_system_t{}(
                            sim,
                            lvl,
                            fine_level,
                            0.5 * dt
                        );
                    }

                    rk2_stage2_system_t<Ops>{ops}(sim, block_geo, lvl);

                    // === REFLUX AND SYNCHRONIZE ===
                    if (lvl < sim.num_levels() - 1) {
                        // apply reflux to fix conservation at boundaries
                        reflux_system_t{}(sim, lvl + 1);

                        // perform restriction to sync the grids for the
                        // next step. The coarse grid has finished its RK2 step,
                        // so it is safe to overwrite.
                        restriction_system_t{}(sim, lvl + 1);
                    }
                }
            );

            synchronize_system_t{}(sim, lvl);
        }

        std::uint64_t get_substeps(std::uint64_t child_lvl) const
        {
            auto& meta = sim.metadata();

            if (meta.subcycling_mode == subcycling_mode_t::NONE) {
                return 1;
            }
            else if (meta.subcycling_mode == subcycling_mode_t::ADAPTIVE) {
                return meta.level_substeps[child_lvl];
            }
            else {   // STANDARD
                return sim.level_info(child_lvl).refinement_ratio;
            }
        }

      public:
        // ---------------------------------------------------------------------
        // configure (called before evolution loop)
        // ---------------------------------------------------------------------
        void configure(std::uint64_t lvl) const
        {
            using namespace ecs;
            ghost_fill_system_t{}(sim, lvl);
            c2p_system_t{}(sim, lvl);
            init_flux_registers_system_t{}(sim, lvl);
        }

        // ---------------------------------------------------------------------
        // step_all (one coarse timestep)
        // ---------------------------------------------------------------------
        void step_all() const
        {
            using namespace ecs;
            auto& meta = sim.metadata();

            timestep_system_t{}(sim);

            sink_cache_system_t{}(sim);

            if (meta.timestepping == timestepping_t::RK2) {
                advance_level_rk2(0);
            }
            else if (meta.timestepping == timestepping_t::EULER) {
                advance_level_euler(0);
            }
            else {
                throw std::runtime_error(
                    "that timestepping method is not implemented."
                );
            }

            meta.time += meta.global_dt;

            if (sim.has_bodies()) {
                body::evolve_bodies(sim);
            }
        }
    };

}   // namespace simbi::evolution

#endif   // EVOLUTION_HPP
