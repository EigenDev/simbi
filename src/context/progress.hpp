#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include "compat.hpp"
#include "io/tabulate/table.hpp"

#include <chrono>
#include <cstdint>
#include <string>

namespace simbi::progress {
    using clock_t      = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;

    struct progress_state_t {
        io::Table table;
        time_point_t start_time;
        std::uint64_t last_emit_iteration{0};
    };

    progress_state_t initialize(const char* title);

    std::string format_time(std::int64_t total_seconds);

    void update(
        progress_state_t& state,
        std::uint64_t iteration,
        real time,
        real dt,
        real tend,
        double speed
    );

    void finalize(progress_state_t& state);

}   // namespace simbi::progress

#endif
