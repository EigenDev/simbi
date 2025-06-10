/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            logger.hpp
 *  * @brief           logger "context" for the emitting simulation state
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "adapter/device_adapter_api.hpp"   // for gpuEventCreate, gpuEventDestroy
#include "config.hpp"   // for real, Platform, global::BuildPlatform, luint
#include "io/console/printb.hpp"    // for writeln, writefl
#include "io/exceptions.hpp"        // for SimulationFailureException
#include "io/hdf5/checkpoint.hpp"   // for write_to_file
#include "io/tabulate/table.hpp"    // for Table, etc
#include "statistics.hpp"           // for display_system_info
#include "util/tools/helpers.hpp"   // for catch_signals, Inter...
#include <chrono>                   // for time_point, high_resolution_clock
#include <csignal>                  // for signal handling
#include <iostream>                 // for operator<<, char_traits, basic_ost...
#include <type_traits>              // for conditional_t

using namespace std::chrono;

namespace simbi {
    namespace io {
        class Timer
        {
            using time_type = std::conditional_t<
                platform::is_gpu,
                adapter::event_t<>,
                high_resolution_clock::time_point>;
            using duration_type =
                std::conditional_t<platform::is_gpu, float, double>;
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

            void start_timer() { record_event(tstart); }

            template <global::Platform P = global::BuildPlatform, typename T>
            void create_event(T& stamp)
            {
                if constexpr (P == global::Platform::GPU) {
                    gpu::api::event_create(&stamp);
                }
            }

            template <global::Platform P = global::BuildPlatform, typename T>
            void destroy_event(T& stamp)
            {
                if constexpr (P == global::Platform::GPU) {
                    gpu::api::event_destroy(stamp);
                }
            }

            template <typename T>
            void record_event(T& stamp)
            {
                if constexpr (std::is_same_v<
                                  T,
                                  high_resolution_clock::time_point>) {
                    stamp = high_resolution_clock::now();
                }
                else {
                    gpu::api::event_record(stamp);
                }
            }

            template <typename T, typename U>
            void record_duration(T& dt, U t1, U t2)
            {
                if constexpr (std::is_same_v<
                                  U,
                                  high_resolution_clock::time_point>) {
                    dt = static_cast<std::chrono::duration<real>>(t2 - t1)
                             .count();
                }
                else {
                    gpu::api::event_synchronize(t2);
                    gpu::api::event_elapsed_time(&dt, t1, t2);
                    // time output from GPU automatically in ms so convert to
                    // seconds
                    dt *= 1e-3;
                }
            }

            duration_type get_duration()
            {
                record_event(tstop);
                record_duration(duration, tstart, tstop);
                return duration;
            }
        };

        namespace logger {

            class CursorManager
            {
              public:
                CursorManager()
                {
                    // Hide the cursor
                    std::cout << "\033[?25l";
                    // Set up signal handlers
                    std::signal(SIGINT, signalHandler);
                    std::signal(SIGTERM, signalHandler);
                    std::signal(SIGABRT, signalHandler);
                    std::signal(SIGSEGV, signalHandler);
                    std::signal(SIGQUIT, signalHandler);
                }

                ~CursorManager()
                {
                    // Show the cursor
                    std::cout << "\033[?25h";
                }

                static void signalHandler(int signal)
                {
                    // Show the cursor
                    std::cout << "\033[?25h";
                    // Exit the program
                    std::exit(signal);
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
                void
                emit_troubled_cells(sim_state_t& sim_state, io::Table& table)
                {
                    auto unwravel_idx = [&](luint idx,
                                            const std::vector<luint>& shape) {
                        std::vector<luint> coords(shape.size());

                        if (global::col_major) {
                            // Column-major order (Fortran-style)
                            luint stride = 1;
                            for (luint ii = 0; ii < shape.size(); ++ii) {
                                coords[ii] = (idx / stride) % shape[ii];
                                stride *= shape[ii];
                            }
                        }
                        else {
                            // Row-major order (C-style)
                            luint stride = 1;
                            for (int ii = shape.size() - 1; ii >= 0; --ii) {
                                coords[ii] = (idx / stride) % shape[ii];
                                stride *= shape[ii];
                            }
                        }

                        return std::make_tuple(coords[0], coords[1], coords[2]);
                    };

                    for (size_t idx = 0;
                         const auto& prim : sim_state.primitives()) {
                        prim.unwrap_or_else([&]() {
                            std::vector<luint> shape;
                            shape.push_back(sim_state.nz());
                            shape.push_back(sim_state.ny());
                            shape.push_back(sim_state.nx());
                            // unravel the index
                            auto [kk, jj, ii] = unwravel_idx(idx, shape);
                            auto cell         = sim_state.mesh()
                                            .get_cell_from_indices(ii, jj, kk);
                            prim->error_at(
                                ii,
                                jj,
                                kk,
                                cell.centroid_coordinate(0),
                                cell.centroid_coordinate(1),
                                cell.centroid_coordinate(2),
                                prim.error_code(),
                                table
                            );

                            return typename sim_state_t::primitive_t{};
                        });
                        ++idx;
                    }
                }

                template <typename sim_state_t>
                void emit_exception(
                    sim_state_t& sim_state,
                    auto& err,
                    io::Table& table
                )
                {
                    table.post_error(std::string("Exception: ") + err.what());
                    sim_state.sync_to_host();
                    sim_state.has_crashed();
                    io::write_to_file(sim_state, table);
                    emit_troubled_cells(sim_state, table);
                }

                // Print the benchmark results
                template <typename sim_state_t>
                void emit_benchmarks(
                    io::Table& table,
                    int n,
                    const sim_state_t& sim_state,
                    double speed,
                    double end_time,
                    steady_clock::time_point start_time
                )
                {
                    const auto elapsed_time = steady_clock::now() - start_time;
                    const auto elapsed_seconds =
                        duration_cast<seconds>(elapsed_time).count();
                    const auto estimated_time_left = static_cast<int>(
                        elapsed_seconds * (end_time / sim_state.time() - 1)
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
                    table.update_row(
                        1,
                        {std::to_string(n),
                         [&]() {
                             std::stringstream ss;
                             ss << std::scientific << std::setprecision(2)
                                << sim_state.time();
                             return ss.str();
                         }(),
                         [&]() {
                             std::stringstream ss;
                             ss << std::scientific << std::setprecision(2)
                                << sim_state.dt();
                             return ss.str();
                         }(),
                         [&]() {
                             std::stringstream ss;
                             ss << std::scientific << std::setprecision(2)
                                << speed;
                             return ss.str();
                         }(),
                         format_time(elapsed_seconds),
                         format_time(estimated_time_left)}
                    );
                    table.set_progress(
                        static_cast<int>((sim_state.time() / end_time) * 100.0)
                    );
                    table.refresh();
                }
            };

            template <typename sim_state_t, typename F>
            void with_logger(sim_state_t& sim_state, real end_time, F&& f)
            {
                auto timer      = Timer();
                auto logger     = Logger();
                auto& n         = logger.n;
                auto& nfold     = logger.nfold;
                auto& ncheck    = logger.ncheck;
                auto& speed     = logger.speed;
                auto& zu_avg    = logger.zu_avg;
                auto& delta_t   = logger.delta_t;
                auto start_time = steady_clock::now();
                CursorManager cursor_manager;

                statistics::display_system_info();

                // use pretty table to print the results
                auto table = simbi::io::TableFactory::create_elegant_table(
                    "Simulation Benchmarks",
                    io::DisplayMode::Dynamic,
                    io::ProgressBar::Enabled
                );
                table.set_header(
                    {"Iteration", "Time", "dt", "Speed", "Elapsed", "ETA"}
                );
                // add initial row with placeholders
                table.add_row(
                    {"0", "0.0", "0.0", "0.0", "00:00:00", "00:00:00"}
                );

                // print the initial table
                table.print();

                // if at the very beginning of the simulation
                // write the initial state to a file
                try {
                    if (sim_state.is_in_initial_primitive_state()) {
                        if (sim_state.is_in_failure_state()) {
                            throw exception::SimulationFailureException();
                        }
                        io::write_to_file(sim_state, table);
                    }
                }
                catch (exception::SimulationFailureException& e) {
                    sim_state.has_failed();
                    sim_state.has_crashed();
                    logger.emit_exception(sim_state, e, table);
                }

                while ((sim_state.time() < end_time) &&
                       (!sim_state.is_in_failure_state())) {
                    try {
                        //============== Compute benchmarks
                        if (sim_state.using_rk1()) {
                            timer.start_timer();
                            f();
                            delta_t = timer.get_duration();
                        }
                        else {
                            timer.start_timer();
                            f();
                            f();
                            delta_t = timer.get_duration();
                        }

                        //=================== Record Benchmarks
                        if (n % nfold == 0) {
                            ncheck += 1;
                            speed = sim_state.total_zones() / delta_t;
                            zu_avg += speed;
                            logger.emit_benchmarks(
                                table,
                                n,
                                sim_state,
                                speed,
                                end_time,
                                start_time
                            );
                        }
                        n++;
                        sim_state.io().increment_iter();

                        if (sim_state.is_in_failure_state()) {
                            throw exception::SimulationFailureException();
                        }
                        // Write to a file at every checkpoint interval
                        if (sim_state.time_to_write_checkpoint()) {
                            table.set_progress(
                                static_cast<int>(
                                    (sim_state.time() / end_time) * 100.0
                                )
                            );
                            io::write_to_file(sim_state, table);
                            sim_state.time_manager()
                                .update_next_checkpoint_time();
                        }
                        // Listen to kill signals
                        helpers::catch_signals();
                    }
                    catch (exception::InterruptException& e) {
                        sim_state.has_failed();
                        sim_state.was_interrupted();
                        logger.emit_exception(sim_state, e, table);
                    }
                    catch (exception::SimulationFailureException& e) {
                        logger.emit_exception(sim_state, e, table);
                    }
                }
                logger.emit_benchmarks(
                    table,
                    n,
                    sim_state,
                    speed,
                    end_time,
                    start_time
                );
                logger.print_avg_speed();
            };
        }   // namespace logger
    }   // namespace io

}   // namespace simbi
#endif
