#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP

#include "checkpoint.hpp"
#include "ecs/components.hpp"
#include "ecs/entity.hpp"
#include "ecs/systems.hpp"
#include "functional/fp.hpp"
#include "io/console/printb.hpp"
#include "io/exceptions.hpp"
#include "progress.hpp"
#include "timing.hpp"
#include "utility/helpers.hpp"

#include <cstddef>
#include <cstdint>
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
            checkpoint::save(sim, state.progress);
            state.schedule.advance(meta.time);
            meta.advance_schedule(state.schedule);
        }
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
                state.should_stop = true;
                state.progress.table.post_error(
                    std::string("Interrupted: ") + e.what()
                );
                checkpoint::save(sim, state.progress);
            }
            catch (exception::SimulationFailureException& e) {
                state.should_stop = true;
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

        // single level step
        void step_level(std::uint64_t lvl) const
        {
            // pipeline of systems
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::timestep_system_t{}(sim, lvl);
            ecs::integration_system_t{}(sim, lvl, ops);
        }

        // all levels with subcycling
        void step_all(
            const std::vector<ecs::entity_t>& levels,
            ecs::registry_t& registry
        ) const
        {
            for (std::size_t lvl = 0; lvl < levels.size(); ++lvl) {
                auto& level_info = registry.get<ecs::level_info_t>(levels[lvl]);
                auto nsteps      = level_info.refinement_ratio;

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    step_level(lvl);
                }
            }

            ecs::flux_correction_system_t{}(sim);
        }
    };

}   // namespace simbi::evolution

#endif
