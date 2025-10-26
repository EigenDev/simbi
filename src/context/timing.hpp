#ifndef TIMING_HPP
#define TIMING_HPP

#include "hetero/adapter.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace simbi::timing {

    struct timer_t {
        hetero::event start_event;
        hetero::event stop_event;
        hetero::stream stream;

        timer_t()
            : start_event(hetero::device::create_event()),
              stop_event(hetero::device::create_event()),
              stream(hetero::device::create_stream())
        {
        }

        void start() { start_event.record(stream); }

        double elapsed_seconds()
        {
            stop_event.record(stream);
            return stop_event.elapsed_time_ms(start_event) * 1e-3;
        }
    };

    // accumulator for statistics
    struct timing_stats_t {
        double total_time{0.0};
        double zone_updates{0.0};
        double min_time{std::numeric_limits<double>::max()};
        double max_time{0.0};
        std::uint64_t count{0};

        void record(double duration, std::uint64_t nzones)
        {
            total_time += duration;
            min_time = std::min(min_time, duration);
            max_time = std::max(max_time, duration);
            zone_updates += nzones / duration;

            count++;
        }

        double average() const
        {
            return count > 0 ? zone_updates / count : 0.0;
        }
    };

}   // namespace simbi::timing

#endif
