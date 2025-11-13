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
            // prep: get u^n state
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute provisional fluxes
            // calculate fluxes from u^n and accumulate for parent
            ecs::staggered_fields_system_t{
              .accumulate_fluxes = (lvl > 0)
            }(sim, ops, lvl);

            // sub-cycle (if fine level exists)
            if (lvl < sim.num_levels() - 1) {
                const auto ref_ratio = sim.level_info(lvl + 1).refinement_ratio;

                // zero fine level's accumulator once
                ecs::zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < ref_ratio;
                     ++substep) {
                    advance_level_euler(lvl + 1);   // recurse
                }

                // correct this level's fluxes
                // now that lvl+1's flux_avg is full, correct *this* level's
                // flux buffer (hydro(lvl).flux)
                auto& meta         = sim.metadata();
                auto& this_hydro   = sim.hydro(lvl);
                auto& finer_hydro  = sim.hydro(lvl + 1);
                const auto& map    = sim.level_mapping(lvl + 1);
                const real dt_this = meta.level_dts[lvl];

                mesh::fmr::correct_level_fluxes(
                    this_hydro.flux,        // correct this level's flux
                    finer_hydro.flux_avg,   // using fine level's average
                    map,
                    dt_this   // normalize by this level's dt
                );
            }

            // advance this level
            // advance u^n -> u^{n+1} using the now-corrected fluxes
            ecs::euler_system_t{ops}(sim, lvl);

            // restrict (if fine level exists)
            // now that this level is at u^{n+1}, overwrite the
            // covered cells with the more-accurate fine grid solution.
            if (lvl < sim.num_levels() - 1) {
                ecs::restriction_system_t{}(sim, lvl + 1);
            }

            // correct parent (if parent exists)
            // this level's flux_avg is now full, so we can
            // correct the parent's flux buffer.
            if (lvl > 0) {
                correct_fluxes(lvl);
            }
        }

        void advance_level_rk2(std::uint64_t lvl) const
        {
            // zero this level's flux accumulator for the entire rk2 step
            // this buffer will be used by the parent (l-1)
            if (lvl > 0) {
                ecs::zero_flux_buffer_system_t{}(sim, lvl);
            }

            // === stage 1: u^n -> u* ===

            // prolongate u^n from parent, compute primitives
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute provisional fluxes from u^n
            ecs::staggered_fields_system_t{
              .advance_bfields   = true,
              .accumulate_fluxes = (lvl > 0),   // accumulate for parent
            }(sim, ops, lvl);

            // === sub-cycle between stages ===

            if (lvl < sim.num_levels() - 1) {
                const auto ref_ratio = sim.level_info(lvl + 1).refinement_ratio;
                const auto& map      = sim.level_mapping(lvl + 1);

                // zero fine level's accumulator once for its entire rk2 cycle
                ecs::zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < ref_ratio;
                     ++substep) {
                    // advance to u* *before* the first substep
                    if (substep == 0) {
                        // advance to u* (and save u^n in workspace)
                        // this uses the provisional (uncorrected) fluxes
                        ecs::rk2_stage1_system_t{ops}(sim, lvl);
                    }

                    // time-interpolate ghosts for substeps 1, 2, ...
                    if (substep > 0) {
                        const real alpha =
                            static_cast<real>(substep) / ref_ratio;
                        ecs::time_interpolated_ghost_fill_system_t{
                          alpha
                        }(sim, lvl + 1);
                    }

                    // recursively advance fine level (full rk2)
                    advance_level_rk2(lvl + 1);

                    // restriction must happen *inside* the loop
                    // this overwrites coarse u* with fine u^{n+1}
                    ecs::restriction_system_t{}(sim, lvl + 1);
                }

                // now that the sub-cycle is done, correct *this level's* fluxes
                // note: we correct the stage 1 fluxes *after* they were used
                // this is the standard Berger-Colella method. the error
                // is corrected in the final stage.
                mesh::fmr::correct_level_fluxes(
                    sim.hydro(lvl).flux,
                    sim.hydro(lvl + 1).flux_avg,
                    map,
                    // normalize by this level's dt
                    sim.metadata().level_dts[lvl]
                );
            }
            else {
                // if this is the finest level, there's no sub-cycle
                // just advance to u*
                ecs::rk2_stage1_system_t{ops}(sim, lvl);
            }

            // === stage 2: u* -> u^{n+1} ===

            // compute primitives from u* (which was overwritten by restriction)
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes from u*
            ecs::staggered_fields_system_t{
              .advance_bfields   = false,       // already advanced in stage 1
              .accumulate_fluxes = (lvl > 0),   // continue accumulating
            }(sim, ops, lvl);

            // flux correction for stage 2
            if (lvl < sim.num_levels() - 1) {
                // correct the stage 2 fluxes before they are used
                const auto& map = sim.level_mapping(lvl + 1);
                mesh::fmr::correct_level_fluxes(
                    sim.hydro(lvl).flux,
                    sim.hydro(lvl + 1)
                        .flux_avg,   // use the same total avg flux
                    map,
                    sim.metadata().level_dts[lvl]
                );
            }

            // advance from u* to u^{n+1} using corrected stage 2 fluxes
            ecs::rk2_stage2_system_t{ops}(sim, lvl);
        }

      public:
        void configure(std::uint64_t lvl) const
        {
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::timestep_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);
        }

        void step_all() const
        {
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
