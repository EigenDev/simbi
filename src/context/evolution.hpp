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
        // ---
        // euler driver
        //
        // the order is:
        // > compute provisional fluxes for coarse level (l) from u^n
        // > sub-cycle and restrict fine level (l+1)
        // > correct coarse level (l) fluxes using fine flux_avg
        // > advance coarse level (l) from u^n to u^{n+1}
        // ---
        void advance_level_euler(std::uint64_t lvl) const
        {
            // prep: prolongate ghosts, compute primitives
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute provisional fluxes for this level
            ecs::staggered_fields_system_t{
              .accumulate_fluxes = (lvl > 0)
            }(sim, ops, lvl);

            // sub-cycle finer level first (if it exists)
            if (lvl < sim.num_levels() - 1) {
                const auto& map = sim.level_mapping(lvl + 1);

                // get the actual number of substeps
                const auto nsteps = get_substeps(lvl + 1);

                // zero fine level's accumulator once
                ecs::zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    // recurse
                    advance_level_euler(lvl + 1);
                    // restriction must happen *inside* the loop
                    ecs::restriction_system_t{}(sim, lvl + 1);
                }

                // correct this level's fluxes using accumulated fine flux
                mesh::fmr::correct_level_fluxes(
                    sim.hydro(lvl).flux,
                    sim.hydro(lvl + 1).flux_avg,
                    map,
                    sim.metadata().level_dts[lvl]
                );
            }

            // advance this level using the (now corrected) fluxes
            ecs::euler_system_t{ops}(sim, lvl);
        }

        // ---
        // rk2 driver (berger-colella)
        //
        // the order is:
        // > compute fluxes for l from u^n
        // > advance l to u* (using uncorrected fluxes)
        // > sub-cycle l+1 (interpolating between u^n and u* from l)
        // > re-compute fluxes for l from u* (which was modified by
        // restriction)
        // > correct stage 2 fluxes for l using fine flux_avg
        // > advance l from u* to u^{n+1}
        // ---
        void advance_level_rk2(std::uint64_t lvl) const
        {
            // === stage 1: u^n -> u* ===
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes from u^n
            ecs::staggered_fields_system_t{
              .advance_bfields   = true,
              .accumulate_fluxes = false,
            }(sim, ops, lvl);

            // advance coarse to u* BEFORE subcycling
            ecs::rk2_stage1_system_t{ops}(sim, lvl);

            // === sub-cycle fine level ===

            if (lvl < sim.num_levels() - 1) {
                // Get the actual number of substeps (from timestep system)
                const auto nsteps = get_substeps(lvl + 1);

                // zero child's accumulator
                ecs::zero_flux_buffer_system_t{}(sim, lvl + 1);

                for (std::uint64_t substep = 0; substep < nsteps; ++substep) {
                    // time-interpolate boundaries between u^n and u*
                    const real alpha =
                        (static_cast<real>(substep) + 0.5) / nsteps;
                    ecs::time_interpolated_ghost_fill_system_t{
                      alpha
                    }(sim, lvl + 1);

                    // fine level does full RK2
                    advance_level_rk2(lvl + 1);

                    // restrict after each fine step
                    ecs::restriction_system_t{}(sim, lvl + 1);
                }
            }

            // === stage 2: u* -> u^{n+1} ===
            ecs::c2p_system_t{}(sim, lvl);
            ecs::sink_cache_system_t{}(sim);

            // compute fluxes from u*
            ecs::staggered_fields_system_t{
              .advance_bfields   = false,
              .accumulate_fluxes = (lvl > 0),
            }(sim, ops, lvl);

            // correct stage 2 fluxes using fine's accumulated flux
            if (lvl < sim.num_levels() - 1) {
                mesh::fmr::correct_level_fluxes(
                    sim.hydro(lvl).flux,
                    sim.hydro(lvl + 1).flux_avg,
                    sim.level_mapping(lvl + 1),
                    sim.metadata().level_dts[lvl]
                );
            }

            // advance to u^{n+1} using corrected stage 2 fluxes
            ecs::rk2_stage2_system_t{ops}(sim, lvl);
        }

        std::uint64_t get_substeps(std::uint64_t child_lvl) const
        {
            auto& meta = sim.metadata();

            if (meta.subcycling_mode == subcycling_mode_t::NONE) {
                return 1;   // no subcycling
            }
            else if (meta.subcycling_mode == subcycling_mode_t::ADAPTIVE) {
                return meta.level_substeps[child_lvl];
            }
            else {   // STANDARD
                return sim.level_info(child_lvl).refinement_ratio;
            }
        }

      public:
        void configure(std::uint64_t lvl) const
        {
            ecs::ghost_fill_system_t{}(sim, lvl);
            ecs::c2p_system_t{}(sim, lvl);
            ecs::timestep_system_t{}(sim);
        }

        void step_all() const
        {
            auto& meta = sim.metadata();

            // calculate all level timesteps (l_max -> l=0)
            ecs::timestep_system_t{}(sim);

            if (meta.timestepping == Timestepping::RK2) {
                advance_level_rk2(0);
            }
            else if (meta.timestepping == Timestepping::EULER) {
                advance_level_euler(0);
            }
            else {
                throw std::runtime_error(
                    "that timestepping method is not implemented."
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
