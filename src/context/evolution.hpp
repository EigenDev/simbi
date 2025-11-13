#ifndef EVOLUTION_HPP
#define EVOLUTION_HPP

#include "checkpoint.hpp"
#include "compat.hpp"
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

                auto duration                 = state.timer.elapsed_seconds();
                std::uint64_t nzones_weighted = 0;
                const real dt_coarse          = meta.level_dts[0];
                for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
                    const auto shape = sim.hydro(lvl).cons.domain().shape();
                    const auto zones = fp::product(shape);

                    const real weight = dt_coarse / meta.level_dts[lvl];
                    nzones_weighted +=
                        static_cast<std::uint64_t>(zones * weight);
                }
                state.stats.record(duration, nzones_weighted);

                // update progress periodically
                if (meta.iteration % 100 == 0) {
                    auto speed = nzones_weighted / duration;
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

    template <typename Sim, typename Ops>
    struct hydro_pipeline_t {
        Sim& sim;
        const Ops& ops;

      private:
        void correct_fluxes(std::uint64_t lvl) const
        {
            // only correct if this is a fine level (L > 0)
            if (lvl == 0) {
                return;
            }

            auto& meta          = sim.metadata();
            auto& coarser_hydro = sim.hydro(lvl - 1);
            auto& finer_hydro   = sim.hydro(lvl);
            const auto& map     = sim.level_mapping(lvl);

            // get the timestep of the *coarse* level (L-1)
            const real dt_coarse = meta.level_dts[lvl - 1];

            // call the flux correction function directly.
            // this reads from finer_hydro.flux_avg and corrects
            // coarser_hydro.flux
            mesh::fmr::correct_level_fluxes(
                coarser_hydro.flux,
                finer_hydro.flux_avg,   // read from the time-weighted buffer
                map,
                dt_coarse   // pass coarse dt for normalization
            );
        }

        void advance_level_euler(std::uint64_t lvl) const
        {
            // get primitives from u^n
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes from u^n and accumulate for parent
            ecs::staggered_fields_system_t{
              .accumulate_fluxes = (lvl > 0)
            }(sim, ops, lvl);

            // if fine level exists: sub-cycle first, then correct flux
            if (lvl < sim.num_levels() - 1) {
                const auto ref_ratio = sim.level_info(lvl + 1).refinement_ratio;

                // zero fine level's flux accumulator
                ecs::zero_flux_buffer_system_t{}(sim, lvl + 1);

                // sub-cycle fine level
                for (std::uint64_t substep = 0; substep < ref_ratio;
                     ++substep) {
                    advance_level_euler(lvl + 1);
                }

                // correct this level's flux using fine level's time-averaged
                // flux
                auto& meta = sim.metadata();
                mesh::fmr::correct_level_fluxes(
                    sim.hydro(lvl).flux,
                    sim.hydro(lvl + 1).flux_avg,
                    sim.level_mapping(lvl + 1),
                    meta.level_dts[lvl]
                );
            }

            // advance u^n -> u^{n+1} using (potentially corrected) flux
            ecs::euler_system_t{ops}(sim, lvl);

            // restrict fine solution onto coarse covered cells
            if (lvl < sim.num_levels() - 1) {
                ecs::restriction_system_t{}(sim, lvl + 1);
            }
        }

        void advance_level_rk2(std::uint64_t lvl) const
        {
            if (lvl > 0) {
                ecs::zero_flux_buffer_system_t{}(sim, lvl);
            }

            // === STAGE 1: u^n -> u* ===

            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes from u^n
            ecs::staggered_fields_system_t{
              .advance_bfields   = true,
              .accumulate_fluxes = (lvl > 0),
            }(sim, ops, lvl);

            // advance to u* before sub-cycling
            ecs::rk2_stage1_system_t{ops}(sim, lvl);

            // === sub-cycle between stages ===

            if (lvl < sim.num_levels() - 1) {
                const auto ref_ratio = sim.level_info(lvl + 1).refinement_ratio;
                ecs::zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < ref_ratio;
                     ++substep) {
                    // time-interpolate for ALL substeps
                    const real alpha = static_cast<real>(substep) / ref_ratio;
                    ecs::time_interpolated_ghost_fill_system_t{
                      alpha
                    }(sim, lvl + 1);

                    // advance fine level
                    advance_level_rk2(lvl + 1);

                    // restrict
                    ecs::restriction_system_t{}(sim, lvl + 1);
                }
            }

            // === STAGE 2: u* -> u^{n+1} ===

            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes from u*
            ecs::staggered_fields_system_t{
              .advance_bfields   = false,
              .accumulate_fluxes = (lvl > 0),
            }(sim, ops, lvl);

            // advance to u^{n+1} (NO flux correction for stage 2)
            ecs::rk2_stage2_system_t{ops}(sim, lvl);
        }

      public:
        void configure(std::uint64_t lvl) const
        {
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::timestep_system_t{}(sim, lvl);
        }

        void step_all() const
        {
            ecs::sink_cache_system_t{}(sim);
            auto& meta = sim.metadata();

            // calculate all level timesteps (L_max -> L=0)
            ecs::timestep_system_t{}(sim, sim.num_levels() - 1);

            // branch to the correct recursive driver
            if (meta.timestepping == Timestepping::RK2) {
                advance_level_rk2(0);
            }
            else if (meta.timestepping == Timestepping::EULER) {
                advance_level_euler(0);
            }
            else {
                throw std::runtime_error(
                    "That timestepping method is not implemented."
                );
            }

            // advance global time by the *coarsest* step
            meta.time += meta.global_dt;

            // evolve bodies (if any)
            if (sim.has_bodies()) {
                body::evolve_bodies(sim);
            }
        }
    };

}   // namespace simbi::evolution

#endif
