// =============================================================================
// checkpoint.hpp
//
// utilities for managing and saving simulation checkpoints.
// this file defines the `checkpoint_schedule_t` struct for determining when
// to save checkpoints (supporting both linear and logarithmic intervals) and
// the main `save` function that orchestrates writing checkpoint files.
//
// usage:
//   checkpoint_schedule_t schedule = { ... };
//   if (schedule.should_checkpoint(current_time)) {
//     checkpoint::save(sim, progress);
//     schedule.advance(current_time, n);
//   }
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "io/checkpoint.hpp"
#include "io/diagnostic_writer.hpp"
#include "io/write_policy.hpp"
#include "progress.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>

namespace simbi::checkpoint {

    struct checkpoint_schedule_t
    {
        real          checkpoint_time;
        real          checkpoint_interval;
        real          dlogt;
        real          tstart;
        std::uint64_t checkpoint_index;

        bool should_checkpoint(real current_time) const
        {
            return current_time >= checkpoint_time;
        }

        void advance(real time, std::uint64_t n)
        {
            // set the initial time interval based on the current time,
            // advanced by the checkpoint interval to the nearest place
            // in the log10 scale. if dlogt is 0 then the interval is set
            // to the current time shifted towards the nearest checkpoint
            // interval. if the checkpoint interval is 0 then the interval
            // is set to the current time.
            if (dlogt != 0) {
                checkpoint_time = tstart * std::pow(10.0, (n + 1) * dlogt);
            }
            else {
                auto round_place = 1.0 / checkpoint_interval;
                checkpoint_time =
                    checkpoint_interval + std::floor(time * round_place + 0.5) / round_place;
            }
            checkpoint_index += 1;
        }

        auto checkpoint_identifier() const
        {
            return dlogt != 0.0 ? checkpoint_index : checkpoint_time;
        }
    };

    template <typename Sim>
    void save(Sim& sim, progress::progress_state_t& progress, const io::write_policy_t& policy = {})
    {
        auto&      meta     = sim.metadata();
        const auto filename = io::compute_checkpoint_filename(
            meta.data_dir,
            meta.checkpoint_identifier(),
            meta.checkpoint_index,
            meta.checkpoint_zones,
            meta.time,
            meta.dlogt,
            sim.was_interrupted,
            sim.in_failure_state
        );

        io::write_checkpoint(sim, filename, policy);

        // update prev_checkpoint_time after successful write
        meta.prev_checkpoint_time = meta.time;

        std::filesystem::path p(filename);
        progress.table.post_success("Checkpoint: " + p.filename().string());
        progress.table.print();
    }

    template <typename Sim>
    void save_diagnostics(Sim& sim, progress::progress_state_t& progress)
    {
        auto& meta = sim.metadata();

        // build filename: {data_dir}/diagnostics/{zones}.diag.{id}.h5
        auto diag_dir = std::filesystem::path(meta.data_dir) / "diagnostics";
        std::filesystem::create_directories(diag_dir);

        auto tnow = helpers::format_real(meta.diagnostic_identifier());
        auto filename =
            diag_dir / helpers::string_format("%d.diag.%s.h5", meta.checkpoint_zones, tnow.c_str());

        io::write_diagnostic_checkpoint(sim, filename.string());

        meta.prev_diagnostic_time = meta.time;

        // reset accumulators only on diagnostic writes
        if constexpr (requires { sim.diagnostics(); }) {
            if (sim.has_bodies()) {
                sim.diagnostics()->reset();
            }
        }

        progress.table.post_success("Diagnostic: " + filename.filename().string());
        progress.table.print();
    }

} // namespace simbi::checkpoint
