#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "build_config.hpp"
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
                std::cout << "tstart: " << tstart << "\n";
                std::cout << "dlogt: " << dlogt << "\n";
                std::cout << "n: " << n << "\n";
                std::cout << "checkpoint time: " << checkpoint_time << std::endl;
                std::cin.get();
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

        io::write_checkpoint(sim, filename, policy);

        // update prev_checkpoint_time after successful write
        meta.prev_checkpoint_time = meta.time;

        // reset diagnostics for next checkpoint interval
        if constexpr (requires { sim.diagnostics(); }) {
            if (sim.has_bodies()) {
                sim.diagnostics()->reset();
            }
        }
        progress.table.post_success("Checkpoint saved: " + filename);
        progress.table.print();
    }

} // namespace simbi::checkpoint

#endif
