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
        template <typename Geometry>
        void
        advance_level_euler(const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace ecs;

            // prep: fill ghosts, recover primitives
            ghost_fill_system_t{}(sim, lvl);
            c2p_system_t{}(sim, lvl);

            // compute fluxes
            flux_system_t{
              .accumulate_fluxes = (lvl > 0)
            }(sim, ops, block_geo, lvl);
            // subcycle finer level
            if (lvl < sim.num_levels() - 1) {
                const auto nsteps = get_substeps(lvl + 1);

                zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    advance_level_euler(block_geo, lvl + 1);
                    restriction_system_t{}(sim, lvl + 1);
                }

                reflux_system_t{}(sim, lvl + 1);
            }

            // advance this level
            euler_system_t<Ops>{ops}(sim, block_geo, lvl);

            // apply body effects (all levels compute, only authoritative passes
            // diagnostics)
            body_effects_system_t<Sim::rank>{}(
                sim,
                block_geo,
                lvl,
                sim.metadata().level_dts[lvl]
            );

            // sync partitions
            synchronize_system_t{}(sim, lvl);
        }

        // ---------------------------------------------------------------------
        // rk2 driver (berger-colella)
        // ---------------------------------------------------------------------
        template <typename Geometry>
        void
        advance_level_rk2(const Geometry& block_geo, std::uint64_t lvl) const
        {
            using namespace ecs;

            // === stage 1: u^n -> u* ===
            ghost_fill_system_t{}(sim, lvl);
            c2p_system_t{}(sim, lvl);
            flux_system_t{.accumulate_fluxes = false}(sim, ops, block_geo, lvl);
            rk2_stage1_system_t<Ops>{ops}(sim, block_geo, lvl);

            // === subcycle fine level ===
            if (lvl < sim.num_levels() - 1) {
                const auto nsteps = get_substeps(lvl + 1);
                zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    advance_level_rk2(block_geo, lvl + 1);
                    restriction_system_t{}(sim, lvl + 1);
                }

                reflux_system_t{}(sim, lvl + 1);
            }

            // === stage 2: u* -> u^{n+1} ===
            ghost_fill_system_t{}(sim, lvl);
            c2p_system_t{}(sim, lvl);
            flux_system_t{
              .accumulate_fluxes = (lvl > 0)
            }(sim, ops, block_geo, lvl);

            rk2_stage2_system_t<Ops>{ops}(sim, block_geo, lvl);

            // apply body effects (all levels compute, only authoritative passes
            // diagnostics)
            body_effects_system_t<Sim::rank>{}(
                sim,
                block_geo,
                lvl,
                sim.metadata().level_dts[lvl]
            );

            // sync partitions
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
            timestep_system_t{}(sim);
        }

        // ---------------------------------------------------------------------
        // step_all (one coarse timestep)
        // ---------------------------------------------------------------------
        void step_all() const
        {
            using namespace ecs;
            auto& meta = sim.metadata();

            // compute all level timesteps
            timestep_system_t{}(sim);

            // build geometry once per timestep and advance
            auto motion    = get_motion_state(sim);
            auto& mesh_cfg = sim.mesh(0);

            with_block_geometry<Sim::coord_system>(
                mesh_cfg,
                motion,
                [&](const auto& block_geo) {
                    if (meta.timestepping == timestepping_t::RK2) {
                        advance_level_rk2(block_geo, 0);
                    }
                    else if (meta.timestepping == timestepping_t::EULER) {
                        advance_level_euler(block_geo, 0);
                    }
                    else {
                        throw std::runtime_error(
                            "that timestepping method is not implemented."
                        );
                    }
                }
            );

            // advance global time
            meta.time += meta.global_dt;

            // evolve bodies (if any)
            if (sim.has_bodies()) {
                body::evolve_bodies(sim);
            }
        }
    };

}   // namespace simbi::evolution

#endif   // EVOLUTION_HPP
