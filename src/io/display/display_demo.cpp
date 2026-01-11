// =============================================================================
// display_demo.cpp
//
// standalone test for the display system.
// simulates a hydrodynamics simulation with realistic data updates.
//
// usage:
//   meson compile display_demo
//   ./build/src/io/display/display_demo
//
// resize your terminal and watch the columns adapt!
// =============================================================================

#include "table.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

std::string format_sci(double value)
{
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << value;
    return ss.str();
}

std::string format_time(std::int64_t total_seconds)
{
    std::int64_t hours   = total_seconds / 3600;
    std::int64_t minutes = (total_seconds % 3600) / 60;
    std::int64_t seconds = total_seconds % 60;

    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hours << ":" << std::setw(2) << std::setfill('0')
        << minutes << ":" << std::setw(2) << std::setfill('0') << seconds;
    return oss.str();
}

int main()
{
    using namespace simbi::display;

    std::cout << "\n=== Display System Demo ===\n";
    std::cout << "Try resizing your terminal to see columns adapt!\n";
    std::cout << "Press Ctrl+C to exit.\n\n";

    std::this_thread::sleep_for(std::chrono::seconds(2));

    // create display in dynamic mode
    table_t display("Relativistic Hydrodynamics Simulation", true);

    // set table header
    display.set_header({"Iteration", "Time", "dt", "Speed", "Elapsed", "ETA"});

    // simulate 200 iterations
    for (std::int64_t ii = 0; ii < 200; ++ii) {
        // simulate realistic data
        double       time    = ii * 1.234e-4;
        double       dt      = 1.234e-4 * (0.9 + 0.2 * (ii % 10) / 10.0); // varying timestep
        double       speed   = 1.15e8 + (ii * 2.3e5);                     // increasing speed
        std::int64_t elapsed = ii * 3;                                    // seconds
        std::int64_t eta     = (200 - ii) * 3;

        // update display
        display.update_row(
            {std::to_string(ii * 150),
             format_sci(time),
             format_sci(dt),
             format_sci(speed),
             format_time(elapsed),
             format_time(eta)}
        );

        display.set_progress(ii / 2); // 0-100%

        // post messages at key milestones
        if (ii == 5) {
            display.post_info("AMR grid initialized with 3 levels");
        }
        if (ii == 15) {
            display.post_success("First checkpoint saved: output/hydro_chk_0001.h5");
        }
        if (ii == 30) {
            display.post_info("Adaptive timestep engaged");
        }
        if (ii == 45) {
            display.post_warning("CFL condition approaching limit (0.47)");
        }
        if (ii == 50) {
            display.post_success("Second checkpoint saved: output/hydro_chk_0002.h5");
        }
        if (ii == 65) {
            display.post_info("Refinement triggered in shock region");
        }
        if (ii == 80) {
            display.post_success("Third checkpoint saved: output/hydro_chk_0003.h5");
        }
        if (ii == 95) {
            display.post_warning("High density contrast detected (ratio: 1e5)");
        }
        if (ii == 110) {
            display.post_success("Fourth checkpoint saved: output/hydro_chk_0004.h5");
        }
        if (ii == 125) {
            display.post_info("Grid rebalancing complete");
        }
        if (ii == 140) {
            display.post_success("Fifth checkpoint saved: output/hydro_chk_0005.h5");
        }
        if (ii == 155) {
            display.post_info("Approaching final time");
        }
        if (ii == 170) {
            display.post_success("Sixth checkpoint saved: output/hydro_chk_0006.h5");
        }
        if (ii == 185) {
            display.post_info("Finalizing simulation");
        }

        // render
        display.refresh();

        // slow down for visual effect (50ms per frame)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // final state
    display.set_progress(100);
    display.post_success("Simulation completed successfully!");
    display.refresh();

    std::cout << "\n\nDemo completed. Check how columns filled your terminal width!\n";

    return 0;
}
