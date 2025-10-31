#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "compat.hpp"
#include "io/serializer.hpp"
#include "progress.hpp"

#include <cmath>
#include <cstdint>

namespace simbi::checkpoint {

    struct checkpoint_schedule_t {
        real checkpoint_time;
        real checkpoint_interval;
        real dlogt;
        std::uint64_t checkpoint_index;

        bool should_checkpoint(real current_time) const
        {
            return current_time >= checkpoint_time;
        }

        void advance(real time)
        {
            // Set the initial time interval
            // based on the current time, advanced
            // by the checkpoint interval to the nearest
            // place in the log10 scale. If dlogt is 0
            // then the interval is set to the current time
            // shifted towards the nearest checkpoint interval
            // if the checkpoint interval is 0 then the interval
            // is set to the current time
            if (dlogt != 0) {
                checkpoint_time =
                    time * std::pow(10.0, std::floor(std::log10(time) + dlogt));
            }
            else {
                static auto round_place = 1.0 / checkpoint_interval;
                checkpoint_time =
                    checkpoint_interval +
                    std::floor(time * round_place + 0.5) / round_place;
            }
            checkpoint_index += 1;
        }

        auto checkpoint_identifier() const
        {
            return dlogt != 0.0 ? checkpoint_index : checkpoint_time;
        }
    };

    template <typename Sim>
    void save(Sim& sim, progress::progress_state_t& progress)
    {
        const auto filename = io::h5::compute_checkpoint_filename(sim);
        progress.table.post_info(
            "[Writing checkpoint to path: " + filename + "]"
        );
        progress.table.print();

        io::serialize_sim_state(sim, filename);
    }

}   // namespace simbi::checkpoint

#endif
