// =============================================================================
// progress.hpp
//
// utilities for displaying simulation progress to the console.
// this file defines the `progress_state_t` struct to hold display-related
// state and provides functions for initializing, updating, and finalizing
// a formatted progress table during a simulation run.
//
// usage:
//   auto state = progress::initialize(regime);
//   // ... in loop ...
//   progress::update(state, iter, time, dt, tend, speed);
//   // ... after loop ...
//   progress::finalize(state, success);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "io/display/table.hpp"
#include "utility/enums.hpp"

#include <chrono>
#include <cstdint>
#include <string>

namespace simbi::progress {
    using clock_t      = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;

    struct progress_state_t
    {
        display::table_t table;
        time_point_t     start_time;
        std::uint64_t    last_emit_iteration{0};
    };

    progress_state_t initialize(regime_t regime);

    const char* regime_display_name(regime_t regime);

    std::string format_time(std::int64_t total_seconds);

    void update(
        progress_state_t& state,
        std::uint64_t     iteration,
        real              time,
        real              dt,
        real              tend,
        double            speed
    );

    void finalize(progress_state_t& state, bool successful_sim);

} // namespace simbi::progress
