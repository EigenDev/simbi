#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "functional/monad/serializer.hpp"
#include "io/console/printb.hpp"
#include "io/console/statistics.hpp"
#include "io/exceptions.hpp"
#include "io/tabulate/table.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/conversion.hpp"
#include "utility/helpers.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::chrono;

namespace simbi {
    class timer_t
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
        timer_t()
        {
            create_event(tstart);
            create_event(tstop);
        }

        ~timer_t()
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
            if constexpr (std::
                              is_same_v<T, high_resolution_clock::time_point>) {
                stamp = high_resolution_clock::now();
            }
            else {
                gpu::api::event_record(stamp);
            }
        }

        template <typename T, typename U>
        void record_duration(T& dt, U t1, U t2)
        {
            if constexpr (std::
                              is_same_v<U, high_resolution_clock::time_point>) {
                dt = static_cast<std::chrono::duration<real>>(t2 - t1).count();
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

    template <typename State, typename Mesh>
    struct simulation_context_t {
        State& state_;
        Mesh& mesh_;
        real end_time_;
        std::uint64_t iteration_ = 0;
        timer_t timer_;
        io::Table table_;
        steady_clock::time_point start_time_;
        real zone_update_bench{0.0};
        real speed_{0.0};
        std::uint64_t nemits_{0};

        // RAII constructor - sets up the "with" context
        simulation_context_t(
            State& state,
            Mesh& mesh,
            real end_time,
            const char* title = "Simulation"
        )
            : state_(state),
              mesh_(mesh),
              end_time_(end_time),
              table_(
                  io::TableFactory::create_elegant_table(
                      title,
                      io::DisplayMode::Dynamic,
                      io::ProgressBar::Enabled
                  )
              )
        {
            // create a buffer b/w the c++ tables and the python ones
            std::cout << std::string(5, '\n');
            statistics::display_system_info();

            table_.set_header(
                {"Iteration", "Time", "dt", "Speed", "Elapsed", "ETA"}
            );
            table_.add_row({"0", "0.0", "0.0", "0.0", "00:00:00", "00:00:00"});
            table_.print();
            start_time_ = steady_clock::now();
        }

        // RAII destructor - cleanup
        ~simulation_context_t()
        {
            table_.set_progress(100);
            emit_benchmarks();
            print_avg_speed();
            std::cout << "Simulation Completed." << std::endl;
        }

        template <typename PhysicsStep>
        void evolve(PhysicsStep&& physics_step)
        {
            auto& meta = state_.metadata;
            try {
                if (meta.time == 0.0 || meta.checkpoint_index == 0) {
                    if (state_.in_failure_state) {
                        throw exception::SimulationFailureException();
                    }
                    io::serialize_hydro_state(state_, mesh_, table_);
                }
            }
            catch (exception::SimulationFailureException& e) {
                emit_exception(e);
            }

            while (meta.time < end_time_ && !state_.in_failure_state) {
                try {
                    timer_.start_timer();

                    // run the physics computation
                    physics_step(state_);

                    auto duration = timer_.get_duration();

                    // handle I/O effects
                    if (iteration_ % 100 == 0) {
                        speed_ = mesh_.full_domain.size() / duration;
                        zone_update_bench += speed_;
                        nemits_++;
                        emit_benchmarks();
                    }

                    if (meta.time >= meta.checkpoint_time) {
                        table_.set_progress(
                            static_cast<std::int64_t>(
                                (meta.time / meta.tend) * 100.0
                            )
                        );
                        io::serialize_hydro_state(state_, mesh_, table_);
                    }

                    iteration_++;
                    // listen to kill signals
                    helpers::catch_signals();
                }
                catch (exception::InterruptException& e) {
                    state_.in_failure_state = true;
                    state_.was_interrupted  = true;
                    emit_exception(e);
                }
                catch (exception::SimulationFailureException& e) {
                    emit_exception(e);
                }
            }
        }

      private:
        void emit_benchmarks()
        {
            const auto meta         = state_.metadata;
            const auto elapsed_time = steady_clock::now() - start_time_;
            const auto elapsed_seconds =
                duration_cast<seconds>(elapsed_time).count();
            const auto estimated_time_left = static_cast<std::int64_t>(
                elapsed_seconds * (meta.tend / meta.time - 1)
            );

            auto format_time = [](std::int64_t total_seconds) {
                std::int64_t hours   = total_seconds / 3600;
                std::int64_t minutes = (total_seconds % 3600) / 60;
                std::int64_t seconds = total_seconds % 60;
                std::ostringstream oss;
                oss << std::setw(2) << std::setfill('0') << hours << ":"
                    << std::setw(2) << std::setfill('0') << minutes << ":"
                    << std::setw(2) << std::setfill('0') << seconds;
                return oss.str();
            };
            table_.update_row(
                1,
                {std::to_string(iteration_),
                 [&]() {
                     std::stringstream ss;
                     ss << std::scientific << std::setprecision(2) << meta.time;
                     return ss.str();
                 }(),
                 [&]() {
                     std::stringstream ss;
                     ss << std::scientific << std::setprecision(2) << meta.dt;
                     return ss.str();
                 }(),
                 [&]() {
                     std::stringstream ss;
                     ss << std::scientific << std::setprecision(2) << speed_;
                     return ss.str();
                 }(),
                 format_time(elapsed_seconds),
                 format_time(estimated_time_left)}
            );
            table_.set_progress(
                static_cast<std::int64_t>((meta.time / meta.tend) * 100.0)
            );
            table_.refresh();
        }

        void emit_exception(const auto& err)
        {
            table_.post_error(std::string("Exception: ") + err.what());
            // state_.sync_to_host();
            state_.in_failure_state = true;
            io::serialize_hydro_state(state_, mesh_, table_);
            emit_troubled_cells();
        }

        void emit_troubled_cells()
        {

            std::vector<std::pair<coordinate_t<State::dimensions>, ErrorCode>>
                crash_regions;

            // we simply try to recover primitives again, but this time
            // we capture the coordinate and the error code
            const auto domain = state_.prim.domain();
            const auto gamma  = state_.metadata.gamma;
            for (std::uint64_t ii = 0; ii < domain.size(); ii++) {
                const auto coord      = domain.linear_to_coord(ii);
                const auto& cons      = state_.cons(coord);
                const auto maybe_prim = hydro::to_primitive(cons, gamma);
                if (!maybe_prim.has_value()) {
                    crash_regions.emplace_back(coord, maybe_prim.error_code());
                }
            }

            for (const auto& v : crash_regions) {
                const auto& coord     = v.first;
                const auto error_code = v.second;
                if (error_code != ErrorCode::NONE) {
                    error_at(coord, error_code);
                }
            }
        }

        void
        error_at(coordinate_t<State::dimensions> coord, ErrorCode error_code)
        {
            constexpr auto Dims = State::dimensions;
            std::ostringstream oss;
            oss << "Primitives in non-physical state.\n";
            if (error_code != ErrorCode::NONE) {
                oss << "reason: " << helpers::error_code_to_string(error_code)
                    << "\n";
            }
            if constexpr (Dims == 1) {
                auto x1 = mesh::centroid(coord, mesh_)[0];
                oss << "location: (" << x1 << "): \n";
            }
            else if constexpr (Dims == 2) {
                if (mesh_.domain.shape()[0] == 1) {   // an effective  1D run
                    auto x1 = mesh::centroid(coord, mesh_)[1];
                    oss << "location: (" << x1 << "): \n";
                    oss << "index: [" << coord[0] << "]\n";
                }
                else {
                    auto [x2, x1] = mesh::centroid(coord, mesh_);
                    oss << "location: (" << x1 << ", " << x2 << "): \n";
                    oss << "indices: [" << coord[1] << ", " << coord[0]
                        << "]\n";
                }
            }
            else {
                if (mesh_.domain.shape()[1] == 1) {   // an effective  1D run
                    auto x1 = mesh::centroid(coord, mesh_)[2];
                    oss << "location: (" << x1 << "): \n";
                    oss << "indicies: [" << coord[2] << "]\n";
                }
                else if (mesh_.domain.shape()[0] ==
                         1) {   // an effective 2D run
                    auto [x3, x2, x1] = mesh::centroid(coord, mesh_);
                    oss << "location: (" << x1 << ", " << x2 << "): \n";
                    oss << "indices: [" << coord[2] << ", " << coord[1]
                        << "]\n";
                }
                else {
                    auto [x3, x2, x1] = mesh::centroid(coord, mesh_);
                    oss << "location: (" << x1 << ", " << x2 << ", " << x3
                        << "): \n";
                    oss << "indices: [" << coord[2] << ", " << coord[1] << ", "
                        << coord[0] << "]\n";
                }
            }
            table_.post_error(oss.str());
        }

        void print_avg_speed()
        {
            if (nemits_ > 0) {
                util::writeln(
                    "Average zone update/sec for {:>5} "
                    "iterations was {:>5.2e} zones/sec",
                    iteration_,
                    zone_update_bench / nemits_
                );
            }
        }
    };

    // Python-style RAII b/c it's fun
    template <typename State, typename Mesh, typename F>
    void with_simulation(State& state, Mesh& mesh, F&& computation)
    {
        simulation_context_t context{state, mesh, state.metadata.tend};
        computation(context);
    }
}   // namespace simbi
#endif
