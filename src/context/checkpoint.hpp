#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "compat.hpp"
#include "io/checkpoint.hpp"
#include "io/write_policy.hpp"
#include "progress.hpp"

#include <cmath>
#include <cstdint>

namespace simbi::checkpoint {

    struct checkpoint_schedule_t
    {
        real          checkpoint_time;
        real          checkpoint_interval;
        real          dlogt;
        std::uint64_t checkpoint_index;

        bool should_checkpoint(real current_time) const
        {
            return current_time >= checkpoint_time;
        }

        void advance(real time)
        {
            // set the initial time interval based on the current time,
            // advanced by the checkpoint interval to the nearest place
            // in the log10 scale. if dlogt is 0 then the interval is set
            // to the current time shifted towards the nearest checkpoint
            // interval. if the checkpoint interval is 0 then the interval
            // is set to the current time.
            if (dlogt != 0) {
                checkpoint_time = time * std::pow(10.0, std::floor(std::log10(time) + dlogt));
            }
            else {
                static auto round_place = 1.0 / checkpoint_interval;
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

        progress.table.post_info("[Writing checkpoint to path: " + filename + "]");
        progress.table.print();

        io::write_checkpoint(sim, filename, policy);

        // update prev_checkpoint_time after successful write
        meta.prev_checkpoint_time = meta.time;

        // reset diagnostics for next checkpoint interval
        if constexpr (requires { sim.diagnostics(); }) {
            if (sim.has_bodies()) {
                sim.diagnostics()->reset();
            }
        }
    }

} // namespace simbi::checkpoint

#endif
