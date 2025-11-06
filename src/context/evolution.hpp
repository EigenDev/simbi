#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP

#include "checkpoint.hpp"
#include "ecs/entity.hpp"
#include "ecs/systems.hpp"
#include "functional/fp.hpp"
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
#include <vector>

namespace simbi::evolution {

    struct evolution_state_t {
        timing::timer_t timer;
        timing::timing_stats_t stats;
        progress::progress_state_t progress;
        checkpoint::checkpoint_schedule_t schedule;
        bool should_stop{false};
    };

    template <typename Sim>
    evolution_state_t initialize(Sim& sim, const char* title = "Simulation")
    {
        auto& meta = sim.metadata();

        return {
          .timer    = timing::timer_t{},
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

    template <typename Sim, typename PhysicsStep>
    void run(Sim& sim, PhysicsStep&& step, evolution_state_t& state)
    {
        auto& meta = sim.metadata();

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
                state.timer.start();

                // run physics
                step(sim);

                auto duration        = state.timer.elapsed_seconds();
                std::uint64_t nzones = 0;
                for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
                    const auto shape = sim.hydro(lvl).cons.domain().shape();
                    nzones += fp::product(shape);
                }
                state.stats.record(duration, nzones);

                // update progress periodically
                if (meta.iteration % 100 == 0) {
                    auto speed = nzones / duration;
                    progress::update(
                        state.progress,
                        meta.iteration,
                        meta.time,
                        meta.dt,
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

    template <typename Sim, typename Ops>
    struct hydro_pipeline_t {
        Sim& sim;
        const Ops& ops;

        void configure(std::uint64_t lvl) const
        {
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::timestep_system_t{}(sim, lvl);
        }

        void step_all_rk2(
            const std::vector<ecs::entity_t>& levels,
            const ecs::registry_t&
        ) const
        {
            auto& meta = sim.metadata();

            // ========== STAGE 1 SETUP ==========
            ecs::ghost_fill_system_t{}(sim, 0);
            for (std::size_t lvl = 1; lvl < levels.size(); ++lvl) {
                ecs::ghost_fill_system_t{}(sim, lvl);
            }

            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::c2p_system_t{}(sim, lvl);
            }

            // calc timestep (only at finest level)
            ecs::timestep_system_t{}(sim, levels.size() - 1);

            ecs::sink_cache_system_t{}(sim);

            // calc fluxes for u^n
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::staggered_fields_system_t{}(sim, ops, lvl);
            }

            // flux correction for Stage 1
            ecs::flux_correction_system_t{}(sim);

            // advance to u^* using Stage 1
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::rk2_stage1_system_t{ops}(sim, lvl);
            }

            // fill ghosts for u^*
            ecs::ghost_fill_system_t{}(sim, 0);
            for (std::size_t lvl = 1; lvl < levels.size(); ++lvl) {
                ecs::ghost_fill_system_t{}(sim, lvl);
            }

            // recover primitives for u^*
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::c2p_system_t{}(sim, lvl);
            }

            // updatre sink cache for u^*
            ecs::sink_cache_system_t{}(sim);

            // calc fluxes for u^*
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::staggered_fields_system_t{
                  .advance_bfields = false
                }(sim, ops, lvl);
            }

            ecs::flux_correction_system_t{}(sim);

            // calc final u^(n+1) using Stage 2
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::rk2_stage2_system_t{ops}(sim, lvl);
            }

            // restriction: copy fine solution to coarse
            for (std::size_t lvl = levels.size() - 1; lvl > 0; --lvl) {
                ecs::restriction_system_t{}(sim, lvl);
            }

            meta.time += meta.dt;

            if (sim.has_bodies()) {
                body::evolve_bodies(sim);
            }
        }

        void step_all_euler(
            const std::vector<ecs::entity_t>& levels,
            const ecs::registry_t&
        ) const
        {
            auto& meta = sim.metadata();

            // apply boundary conditions on base level
            ecs::ghost_fill_system_t{}(sim, 0);

            // prolongation: fill fine ghost zones from coarse
            for (std::size_t lvl = 1; lvl < levels.size(); ++lvl) {
                ecs::ghost_fill_system_t{}(sim, lvl);
            }

            // recover primitives on all levels
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::c2p_system_t{}(sim, lvl);
            }

            // compute timestep (only at finest level)
            ecs::timestep_system_t{}(sim, levels.size() - 1);

            // sink cache houses the body properties (like R_BH and Mdot_target)
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::staggered_fields_system_t{}(sim, ops, lvl);
            }
            ecs::flux_correction_system_t{}(sim);

            // advance all levels by the same dt
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                ecs::euler_system_t{ops}(sim, lvl);
            }

            // restriction: copy fine solution to coarse (finest to coarsest)
            for (std::size_t lvl = levels.size() - 1; lvl > 0; --lvl) {
                ecs::restriction_system_t{}(sim, lvl);
            }
            meta.time += meta.dt;

            if (sim.has_bodies()) {
                body::evolve_bodies(sim);
            }
        }

        void step_all(
            const std::vector<ecs::entity_t>& levels,
            const ecs::registry_t& registry
        ) const
        {
            if (sim.metadata().timestepping == Timestepping::RK2) {
                step_all_rk2(levels, registry);
            }
            else if (sim.metadata().timestepping == Timestepping::EULER) {
                step_all_euler(levels, registry);
            }
            else {
                throw std::runtime_error(
                    "That timestepping method is not implemented."
                );
            }
        }
    };

}   // namespace simbi::evolution

#endif
