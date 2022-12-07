#include "build_options.hpp"
#include "common/helpers.hpp"
#include "common/helpers.hip.hpp"
#include "printb.hpp"

namespace simbi
{
    namespace detail
    {
        class Timer
        {
            using time_type     = std::conditional_t<BuildPlatform == Platform::GPU, anyGpuEvent_t, std::chrono::high_resolution_clock::time_point>;
            using duration_type = std::conditional_t<BuildPlatform == Platform::GPU, float, double>;
            time_type tstart, tstop;
            duration_type duration;
            public:
            Timer() {
                create_event(tstart);
                create_event(tstop);
            }
            ~Timer(){
                destroy_event(tstart);
                destroy_event(tstop);
            }

            void startTimer() {
                recordEvent(tstart);
            }

            template<Platform P = BuildPlatform, typename T>
            void create_event(T &stamp) {
                if constexpr(P == Platform::GPU) {
                    gpu::api::gpuEventCreate(&stamp);
                }
            }

            template<Platform P = BuildPlatform, typename T>
            void destroy_event(T &stamp) {
                if constexpr(P == Platform::GPU) {
                    gpu::api::gpuEventDestroy(stamp);
                }
            }

            template<typename T>
            void recordEvent(T &stamp) {
                if constexpr(std::is_same_v<T,std::chrono::high_resolution_clock::time_point>) {
                    stamp = std::chrono::high_resolution_clock::now();
                } else {
                    gpu::api::gpuEventRecord(stamp);
                }
            }

            template<typename T, typename U>
            void recordDuration(T &dt, U t1, U t2) {
                if constexpr(std::is_same_v<U,std::chrono::high_resolution_clock::time_point>) {
                    dt = static_cast<std::chrono::duration<real>>(t2 - t1).count();
                } else {
                    gpu::api::gpuEventSynchronize(t2);
                    gpu::api::gpuEventElapsedTime(&dt, t1, t2);
                    // time output from GPU automatically in ms so convert to seconds
                    dt *= 1e-3;
                }
            }

            duration_type get_duration() {
                recordEvent(tstop);
                recordDuration(duration, tstart, tstop);
                return duration;
            }
        };

        namespace logger {
            struct Logger {
                int n, nfold, ncheck;
                real speed, zu_avg, delta_t;
                Logger() : n(0), nfold(100), ncheck(0), speed(0), zu_avg(0), delta_t(0) {};
                ~Logger(){};
            };

            inline void print_avg_speed(Logger &logger) {
                if (logger.ncheck > 0) {
                    util::writeln("Average zone update/sec for {:>5} iterations was {:>5.2e} zones/sec", 
                    logger.n, logger.zu_avg / logger.ncheck);
                }
            }

            template <typename sim_state_t, typename F>
            void with_logger(sim_state_t &sim_state, real end_time, F &&f) {
                auto timer  = Timer();
                auto logger = Logger();
                using conserved_t = typename sim_state_t::conserved_t;
                using primitive_t = typename sim_state_t::primitive_t;
                auto&n       = logger.n;
                auto&nfold   = logger.nfold;
                auto&ncheck  = logger.ncheck;
                auto&speed   = logger.speed;
                auto&zu_avg  = logger.zu_avg;
                auto&delta_t = logger.delta_t;
                constexpr auto write2file = helpers::write_to_file<typename sim_state_t::primitive_soa_t, sim_state_t::dimensions, sim_state_t>;
                
                while (sim_state.t < end_time & !sim_state.inFailureState)
                {
                    if constexpr(is_relativistic<sim_state_t>::value) {
                        if constexpr(sim_state_t::dimensions == 1) {
                            // Fill outer zones if user-defined conservative functions provided
                            if (sim_state.all_outer_bounds && (sim_state.mesh_motion)) {
                                const real dV  = get_cell_volume(sim_state.active_zones - 1, sim_state.geometry);
                                sim_state.outer_zones[0] = conserved_t{d_outer(sim_state.x1max), s_outer(sim_state.x1max), e_outer(sim_state.x1max)} * dV;
                                sim_state.outer_zones.copyToGpu();
                            }
                        } else if constexpr(sim_state_t::dimensions == 2) {
                            // Fill outer zones if user-defined conservative functions provided
                            if (sim_state.all_outer_bounds && sim_state.mesh_motion)
                            {
                                // #pragma omp parallel for 
                                for (luint jj = 0; jj < sim_state.ny; jj++) {
                                    const auto jreal = helpers::get_real_idx(jj, sim_state.radius, sim_state.yphysical_grid);
                                    const real dV    = get_cell_volume(sim_state.xphysical_grid - 1, jreal, sim_state.geometry);
                                    sim_state.outer_zones[jj]  = conserved_t{d_outer(sim_state.x1max, sim_state.x2[jreal]), 
                                    s1_outer(sim_state.x1max, sim_state.x2[jreal]), s2_outer(sim_state.x1max, sim_state.x2[jreal]), e_outer(sim_state.x1max, sim_state.x2[jreal])} * dV;
                                }
                                sim_state.outer_zones.copyToGpu();
                            }
                        }
                    }
                    // Using a sigmoid decay function to represent when the source terms turn off.
                    sim_state.decay_constant = helpers::sigmoid(sim_state.t, sim_state.engine_duration);
                    try {
                        //============== Compute benchmarks
                        int fold_count = 0;
                        if (sim_state.first_order) {
                            timer.startTimer();
                            do
                            {
                                f();
                            } while (++fold_count < nfold && sim_state.t < sim_state.t_interval);
                            delta_t = timer.get_duration();
                        } else {
                            timer.startTimer();
                            do
                            {
                                f();
                                f();
                            } while (++fold_count < nfold && sim_state.t < sim_state.t_interval);
                            delta_t = timer.get_duration();
                        }

                        //=================== Record Benchmarks
                        n      += fold_count;
                        ncheck += 1;
                        speed   = fold_count * sim_state.total_zones / delta_t;
                        zu_avg += speed;
                        if constexpr(BuildPlatform == Platform::GPU) {
                            const real gpu_emperical_bw = getFlops<conserved_t, primitive_t>(sim_state.pseudo_radius, sim_state.total_zones, sim_state.active_zones, delta_t);
                            util::writefl<Color::LIGHT_MAGENTA>("\riteration:{:>06}  dt: {:>08.2e}  time: {:>08.2e}  zones/sec: {:>08.2e}  ebw(%): {:>04.2f}", 
                            n, sim_state.dt, sim_state.t, speed, static_cast<real>(100.0) * fold_count * gpu_emperical_bw / gpu_theoretical_bw);
                        } else {
                            util::writefl<Color::LIGHT_MAGENTA>("\riteration:{:>06}    dt: {:>08.2e}    time: {:>08.2e}    zones/sec: {:>08.2e}", n, sim_state.dt, sim_state.t, speed);
                        }
                        
                        // Write to a file at every checkpoint interval
                        if (sim_state.t >= sim_state.t_interval && sim_state.t != INFINITY)
                        {
                            write2file(sim_state, sim_state.setup, sim_state.data_directory, 
                            sim_state.t, sim_state.t_interval, sim_state.chkpt_interval, sim_state.checkpoint_zones);
                            if (sim_state.dlogt != 0) {
                                sim_state.t_interval *= std::pow(10, sim_state.dlogt);
                            } else {
                                sim_state.t_interval += sim_state.chkpt_interval;
                            }
                        }
                        // Listen to kill signals
                        helpers::catch_signals();
                    } catch (helpers::InterruptException &e) {
                        util::writeln("{}", e.what());
                        sim_state.inFailureState = true;
                        write2file(sim_state, sim_state.setup, sim_state.data_directory, 
                        sim_state.t, INFINITY, sim_state.chkpt_interval, sim_state.checkpoint_zones);
                    }  
                }
                print_avg_speed(logger);              
            };
        } // namespace logger 
    } // namespace detail
    
} // namespace simbi
