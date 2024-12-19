/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       logger.hpp
 * @brief      the logger "context" which executes most of the I/O on the host
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "build_options.hpp"   // for real, Platform, global::BuildPlatform, luint
#include "common/exceptions.hpp"   // for SimulationFailureException
#include "common/helpers.hpp"      // for get_real_idx, catch_signals, Inter...
#include "common/traits.hpp"       // for is_relativistic
#include "device_api.hpp"          // for gpuEventCreate, gpuEventDestroy
#include "printb.hpp"              // for writeln, writefl
#include "progress.hpp"            // for progress_bar
#include <chrono>                  // for time_point, high_resolution_clock
#include <cmath>                   // for INFINITY, pow
#include <format>                  // for format
#include <iostream>                // for operator<<, char_traits, basic_ost...
#include <memory>                  // for allocator
#include <type_traits>             // for conditional_t

using namespace std::chrono;

namespace simbi {
    namespace detail {
        class Timer
        {
            using time_type = std::conditional_t<
                global::on_gpu,
                devEvent_t,
                high_resolution_clock::time_point>;
            using duration_type =
                std::conditional_t<global::on_gpu, float, double>;
            time_type tstart, tstop;
            duration_type duration;

          public:
            Timer()
            {
                create_event(tstart);
                create_event(tstop);
            }

            ~Timer()
            {
                destroy_event(tstart);
                destroy_event(tstop);
            }

            void startTimer() { recordEvent(tstart); }

            template <global::Platform P = global::BuildPlatform, typename T>
            void create_event(T& stamp)
            {
                if constexpr (P == global::Platform::GPU) {
                    gpu::api::gpuEventCreate(&stamp);
                }
            }

            template <global::Platform P = global::BuildPlatform, typename T>
            void destroy_event(T& stamp)
            {
                if constexpr (P == global::Platform::GPU) {
                    gpu::api::gpuEventDestroy(stamp);
                }
            }

            template <typename T>
            void recordEvent(T& stamp)
            {
                if constexpr (std::is_same_v<
                                  T,
                                  high_resolution_clock::time_point>) {
                    stamp = high_resolution_clock::now();
                }
                else {
                    gpu::api::gpuEventRecord(stamp);
                }
            }

            template <typename T, typename U>
            void recordDuration(T& dt, U t1, U t2)
            {
                if constexpr (std::is_same_v<
                                  U,
                                  high_resolution_clock::time_point>) {
                    dt =
                        static_cast<std::chrono::duration<real>>(t2 - t1).count(
                        );
                }
                else {
                    gpu::api::gpuEventSynchronize(t2);
                    gpu::api::gpuEventElapsedTime(&dt, t1, t2);
                    // time output from GPU automatically in ms so convert to
                    // seconds
                    dt *= 1e-3;
                }
            }

            duration_type get_duration()
            {
                recordEvent(tstop);
                recordDuration(duration, tstart, tstop);
                return duration;
            }
        };

        namespace logger {
            class PrettyTable
            {
              private:
                std::string title;
                std::vector<std::vector<std::string>> rows;
                std::vector<int> columnWidths;
                double progress = 0.0;
                std::time_t startTime;
                int printedLines = 0;

                const std::string RESET   = "\033[0m";
                const std::string BOLD    = "\033[1m";
                const std::string RED     = "\033[31m";
                const std::string GREEN   = "\033[32m";
                const std::string YELLOW  = "\033[33m";
                const std::string BLUE    = "\033[34m";
                const std::string MAGENTA = "\033[35m";
                const std::string CYAN    = "\033[36m";
                const std::string WHITE   = "\033[37m";

                void calculateColumnWidths()
                {
                    columnWidths.clear();
                    for (const auto& row : rows) {
                        for (size_t i = 0; i < row.size(); ++i) {
                            if (columnWidths.size() <= i) {
                                columnWidths.push_back(row[i].size());
                            }
                            else {
                                columnWidths[i] = std::max(
                                    columnWidths[i],
                                    static_cast<int>(row[i].size())
                                );
                            }
                        }
                    }
                }

                void printSeparator() const
                {
                    std::cout << BOLD << WHITE;
                    for (const auto& width : columnWidths) {
                        std::cout << "╠" << std::string(width + 2, '=');
                    }
                    std::cout << "╣" << RESET << std::endl;
                }

                void saveCursorPosition() const
                {
                    std::cout << "\033[s";   // Save cursor position
                }

                void restoreCursorPosition() const
                {
                    std::cout << "\033[u";   // Restore cursor position
                }

                void moveCursorUp(int lines) const
                {
                    if (lines > 0) {
                        std::cout << "\033[" << lines << "A";
                    }
                }

                void printProgressBar() const
                {
                    int tableWidth = 0;
                    for (const auto& width : columnWidths) {
                        tableWidth +=
                            width + 3;   // Column width + padding + border
                    }
                    tableWidth += 1;   // Right border

                    int barWidth = tableWidth - 2;   // Adjust for borders
                    std::cout << BOLD << WHITE << "╠";
                    int pos = barWidth * progress;
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) {
                            std::cout << "█";
                        }
                        else if (i == pos) {
                            std::cout << ">";
                        }
                        else {
                            std::cout << " ";
                        }
                    }
                    std::cout << "╣ " << static_cast<int>(progress * 100.0)
                              << "%" << RESET << std::endl;
                }

              public:
                PrettyTable(const std::string& title)
                    : title(title), startTime(std::time(nullptr))
                {
                }

                void addRow(const std::vector<std::string>& row)
                {
                    rows.push_back(row);
                    calculateColumnWidths();
                }

                void
                updateRow(size_t index, const std::vector<std::string>& row)
                {
                    if (index < rows.size()) {
                        rows[index] = row;
                        calculateColumnWidths();
                    }
                }

                void setProgress(double value) { progress = value; }

                void print()
                {
                    if (printedLines > 0) {
                        moveCursorUp(printedLines);
                    }
                    else {
                        saveCursorPosition();
                    }

                    std::cout << BOLD << CYAN << title << RESET << std::endl;
                    printSeparator();
                    for (const auto& row : rows) {
                        for (size_t i = 0; i < row.size(); ++i) {
                            std::cout << BOLD << WHITE << "║ " << RESET
                                      << std::setw(columnWidths[i]) << std::left
                                      << row[i] << " ";
                        }
                        std::cout << BOLD << WHITE << "║" << RESET << std::endl;
                        printSeparator();
                    }

                    printProgressBar();

                    if (printedLines == 0) {
                        printedLines =
                            2 + rows.size() * 2 +
                            1;   // Title + separators and rows + progress bar
                    }
                }
            };

            struct Logger {
                int n, nfold, ncheck;
                real speed, zu_avg, delta_t;
                Logger()
                    : n(0),
                      nfold(100),
                      ncheck(0),
                      speed(0),
                      zu_avg(0),
                      delta_t(0) {};

                ~Logger()
                {   // Show the cursor
                    std::cout << "\033[?25h";
                };

                void print_avg_speed()
                {
                    if (ncheck > 0) {
                        util::writeln(
                            "Average zone update/sec for {:>5} "
                            "iterations was {:>5.2e} zones/sec",
                            n,
                            zu_avg / ncheck
                        );
                    }
                }

                template <typename sim_state_t>
                void emit_troubled_cells(sim_state_t& sim_state)
                {
                    const luint nx             = sim_state.nx;
                    const luint ny             = sim_state.ny;
                    const luint nz             = sim_state.nz;
                    const luint radius         = sim_state.radius;
                    const luint xag            = sim_state.xag;
                    const luint yag            = sim_state.yag;
                    const luint zag            = sim_state.zag;
                    const luint total_zones    = nx * ny * nz;
                    const auto& troubled_cells = sim_state.troubled_cells;
                    auto& prims                = sim_state.prims;
                    for (luint gid = 0; gid < total_zones; gid++) {
                        if (troubled_cells[gid] != 0) {
                            const luint kk   = get_height(gid, nx, ny);
                            const luint jj   = get_row(gid, nx, ny, kk);
                            const luint ii   = get_column(gid, nx, ny, kk);
                            const lint ireal = get_real_idx(ii, radius, xag);
                            const lint jreal = get_real_idx(jj, radius, yag);
                            const lint kreal = get_real_idx(kk, radius, zag);
                            const auto cell =
                                sim_state.cell_factors(ireal, jreal, kreal);
                            const real x1mean = cell.x1mean;
                            const real x2mean = cell.x2mean;
                            const real x3mean = cell.x3mean;
                            prims[gid].error_at(x1mean, x2mean, x3mean);
                        }
                    }
                }

                template <typename sim_state_t>
                void emit_exception(sim_state_t& sim_state, auto& err)
                {
                    auto print_pattern = [](auto pattern) {
                        for (int i = 0; i < 40; ++i) {
                            std::cerr << pattern;
                        }
                        std::cerr << std::endl;
                    };
                    print_pattern("-+");
                    std::cerr << err.what() << '\n';
                    print_pattern("-+");
                    sim_state.troubled_cells.copyFromGpu();
                    sim_state.cons.copyFromGpu();
                    sim_state.prims.copyFromGpu();
                    sim_state.hasCrashed = true;
                    write_to_file(sim_state);
                    emit_troubled_cells(sim_state);
                }

                template <typename sim_state_t>
                void print_gpu_info(
                    int n,
                    const sim_state_t& sim_state,
                    double speed,
                    double delta_t
                )
                {
                    const real gpu_emperical_bw = helpers::getFlops<
                        typename sim_state_t::conserved_t,
                        typename sim_state_t::primitive_t>(
                        sim_state_t::dimensions,
                        sim_state.radius,
                        sim_state.total_zones,
                        sim_state.active_zones,
                        delta_t
                    );

                    std::cout << "\033[s";    // Save cursor position
                    std::cout << "\033[1B";   // Move cursor down one line
                    util::writefl<Color::LIGHT_MAGENTA>(
                        "iteration:{:>06}  dt: {:>08.2e}  time: {:>08.2e}  "
                        "zones/sec: {:>08.2e}  ebw(%): {:>04.2f} ",
                        n,
                        sim_state.dt,
                        sim_state.t,
                        speed,
                        100.0 * gpu_emperical_bw / gpu_theoretical_bw
                    );
                    std::cout << "\033[K\r";   // Clear the line and move to
                                               // the beginning
                    std::cout << "\033[u";     // Restore cursor position
                }

                template <typename sim_state_t>
                void print_cpu_info(
                    int n,
                    const sim_state_t& sim_state,
                    double speed
                )
                {
                    std::cout << "\033[s";    // Save cursor position
                    std::cout << "\033[3A";   // Move cursor up two lines
                    util::writefl<Color::LIGHT_MAGENTA>(
                        "iteration:{:>06}    dt: {:>08.2e}    time: "
                        "{:>08.2e}  "
                        "  "
                        "zones/sec: {:>08.2e} ",
                        n,
                        sim_state.dt,
                        sim_state.t,
                        speed
                    );
                    std::cout << "\033[K\r";   // Clear the line and move to
                                               // the beginning
                    std::cout << "\033[u";     // Restore cursor position
                }

                template <typename sim_state_t>
                void print_progress_bar(
                    const sim_state_t& sim_state,
                    double end_time,
                    steady_clock::time_point start_time
                )
                {
                    auto elapsed_time = steady_clock::now() - start_time;
                    auto elapsed_seconds =
                        duration_cast<seconds>(elapsed_time).count();
                    auto estimated_time_left = static_cast<int>(
                        elapsed_seconds * (end_time / sim_state.t - 1)
                    );

                    auto format_time = [](int total_seconds) {
                        int hours   = total_seconds / 3600;
                        int minutes = (total_seconds % 3600) / 60;
                        int seconds = total_seconds % 60;
                        std::ostringstream oss;
                        oss << std::setw(2) << std::setfill('0') << hours << ":"
                            << std::setw(2) << std::setfill('0') << minutes
                            << ":" << std::setw(2) << std::setfill('0')
                            << seconds;
                        return oss.str();
                    };

                    std::cout << "\033[s";    // Save cursor position
                    std::cout << "\033[1A";   // Move cursor up one line
                    util::writefl<Color::LIGHT_YELLOW>(
                        "Elapsed time: {}    Estimated time left: {}",
                        format_time(elapsed_seconds),
                        format_time(estimated_time_left)
                    );
                    std::cout << "\033[K\r";   // Clear the line and move to
                                               // the beginning
                    std::cout << "\033[u";     // Restore cursor position

                    std::cout << "\033[s";      // Save cursor position
                    std::cout << "\033[999B";   // Move cursor down two lines
                    helpers::progress_bar(sim_state.t / end_time);
                    std::cout << "\033[u";   // Restore cursor position
                }

                // Print the benchmark results
                template <typename sim_state_t>
                void emit_benchmarks(
                    PrettyTable& table,
                    int n,
                    const sim_state_t& sim_state,
                    double speed,
                    double delta_t,
                    double end_time,
                    steady_clock::time_point start_time
                )
                {
                    const auto elapsed_time = steady_clock::now() - start_time;
                    const auto elapsed_seconds =
                        duration_cast<seconds>(elapsed_time).count();
                    const auto estimated_time_left = static_cast<int>(
                        elapsed_seconds * (end_time / sim_state.t - 1)
                    );

                    auto format_time = [](int total_seconds) {
                        int hours   = total_seconds / 3600;
                        int minutes = (total_seconds % 3600) / 60;
                        int seconds = total_seconds % 60;
                        std::ostringstream oss;
                        oss << std::setw(2) << std::setfill('0') << hours << ":"
                            << std::setw(2) << std::setfill('0') << minutes
                            << ":" << std::setw(2) << std::setfill('0')
                            << seconds;
                        return oss.str();
                    };
                    table.updateRow(
                        1,
                        {std::to_string(n),
                         std::format("{:.2e}", sim_state.t),
                         std::format("{:.2e}", delta_t),
                         std::format("{:.2e}", speed),
                         format_time(elapsed_seconds),
                         format_time(estimated_time_left)}
                    );
                    table.setProgress(sim_state.t / end_time);
                    table.print();
                };
            };

            template <typename sim_state_t, typename F>
            void with_logger(sim_state_t& sim_state, real end_time, F&& f)
            {
                auto timer    = Timer();
                auto logger   = Logger();
                auto& n       = logger.n;
                auto& nfold   = logger.nfold;
                auto& ncheck  = logger.ncheck;
                auto& speed   = logger.speed;
                auto& zu_avg  = logger.zu_avg;
                auto& delta_t = logger.delta_t;

                auto start_time = steady_clock::now();

                // use pretty table to print the results
                PrettyTable table("Simulation Benchmarks");
                table.addRow(
                    {"Iteration",
                     "Time",
                     "Delta T",
                     "Speed",
                     "Time Elapsed",
                     "Time Remaining"}
                );
                table.addRow({"0", "0", "0", "0", "0", "0"});

                // Hide the cursor
                std::cout << "\033[?25l";
                while ((sim_state.t < end_time) && (!sim_state.inFailureState)
                ) {
                    try {
                        //============== Compute benchmarks
                        if (sim_state.use_rk1) {
                            timer.startTimer();
                            f();
                            delta_t = timer.get_duration();
                        }
                        else {
                            timer.startTimer();
                            f();
                            f();
                            delta_t = timer.get_duration();
                        }
                        //=================== Record Benchmarks
                        if (n % nfold == 0) {
                            ncheck += 1;
                            speed = sim_state.total_zones / delta_t;
                            zu_avg += speed;
                            logger.emit_benchmarks(
                                table,
                                n,
                                sim_state,
                                speed,
                                delta_t,
                                end_time,
                                start_time
                            );
                        }
                        n++;
                        sim_state.global_iter = n;

                        // Write to a file at every checkpoint interval
                        if (sim_state.t >= sim_state.t_interval &&
                            sim_state.t != INFINITY) {
                            helpers::write_to_file(sim_state);
                            if (sim_state.dlogt != 0) {
                                sim_state.t_interval *=
                                    std::pow(10.0, sim_state.dlogt);
                            }
                            else {
                                sim_state.t_interval +=
                                    sim_state.chkpt_interval;
                            }
                        }

                        if (sim_state.inFailureState) {
                            throw exception::SimulationFailureException();
                        }
                        // Listen to kill signals
                        helpers::catch_signals();
                    }
                    catch (exception::InterruptException& e) {
                        sim_state.inFailureState = true;
                        sim_state.wasInterrupted = true;
                        logger.emit_exception(sim_state, e);
                    }
                    catch (exception::SimulationFailureException& e) {
                        logger.emit_exception(sim_state, e);
                    }
                }
                logger.emit_benchmarks(
                    table,
                    n,
                    sim_state,
                    speed,
                    delta_t,
                    end_time,
                    start_time
                );
                logger.print_avg_speed();
            };
        }   // namespace logger
    }   // namespace detail

}   // namespace simbi
#endif
