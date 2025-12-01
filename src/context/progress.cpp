#include "progress.hpp"
#include "compat.hpp"
#include "io/console/statistics.hpp"
#include "io/tabulate/table.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace simbi::progress {

    progress_state_t initialize(const char* title)
    {
        std::cout << std::string(5, '\n');
        statistics::display_system_info();

        auto table = io::table_factory_t::create_elegant_table(
            title,
            io::display_mode_t::Dynamic,
            io::progress_bar_t::Enabled
        );

        table.set_header(
            {"Iteration", "Time", "dt", "Speed", "Elapsed", "ETA"}
        );
        table.add_row({"0", "0.0", "0.0", "0.0", "00:00:00", "00:00:00"});
        table.print();
        return {.table = std::move(table), .start_time = clock_t::now()};
    }

    std::string format_time(std::int64_t total_seconds)
    {
        std::int64_t hours   = total_seconds / 3600;
        std::int64_t minutes = (total_seconds % 3600) / 60;
        std::int64_t seconds = total_seconds % 60;

        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << hours << ":" << std::setw(2)
            << std::setfill('0') << minutes << ":" << std::setw(2)
            << std::setfill('0') << seconds;
        return oss.str();
    }

    void update(
        progress_state_t& state,
        std::uint64_t iteration,
        real time,
        real dt,
        real tend,
        double speed
    )
    {
        using namespace std::chrono;

        auto elapsed     = clock_t::now() - state.start_time;
        auto elapsed_sec = duration_cast<seconds>(elapsed).count();
        auto eta_sec =
            static_cast<std::int64_t>(elapsed_sec * (tend / time - 1));

        auto format_sci = [](real value) {
            std::stringstream ss;
            ss << std::scientific << std::setprecision(2) << value;
            return ss.str();
        };

        state.table.update_row(
            1,
            {std::to_string(iteration),
             format_sci(time),
             format_sci(dt),
             format_sci(speed),
             format_time(elapsed_sec),
             format_time(eta_sec)}
        );

        state.table.set_progress(
            static_cast<std::int64_t>((time / tend) * 100.0)
        );
        state.table.refresh();

        state.last_emit_iteration = iteration;
    }

    void finalize(progress_state_t& state)
    {
        state.table.refresh();
        std::cout << "Simulation completed.\n";
    }

}   // namespace simbi::progress
