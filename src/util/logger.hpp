#include "build_options.hpp"
#include "common/helpers.hpp"
#include "common/helpers.hip.hpp"
#include "printb.hpp"

namespace simbi
{
    namespace detail
    {
        namespace logger {
            static luint n       = 0;
            static luint nfold   = 0;
            static luint ncheck  = 0;
            static real  zu_avg  = 0.0;
            static real delta_t  = 0.0;
        };

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
                    anyGpuEventCreate(&stamp);
                }
            }

            template<Platform P = BuildPlatform, typename T>
            void destroy_event(T &stamp) {
                if constexpr(P == Platform::GPU) {
                    anyGpuEventDestroy(stamp);
                }
            }

            template<typename T>
            void recordEvent(T &stamp) {
                if constexpr(std::is_same_v<T,std::chrono::high_resolution_clock::time_point>) {
                    stamp = std::chrono::high_resolution_clock::now();
                } else {
                    anyGpuEventRecord(stamp);
                }
            }

            template<typename T, typename U>
            void recordDuration(T &dt, U t1, U t2) {
                if constexpr(std::is_same_v<U,std::chrono::high_resolution_clock::time_point>) {
                    dt = static_cast<std::chrono::duration<real>>(t2 - t1).count();
                } else {
                    anyGpuEventSynchronize(t2);
                    anyGpuEventElapsedTime(&dt, t1, t2);
                    dt *= 1e-3;
                }
            }

            duration_type get_duration() {
                recordEvent(tstop);
                recordDuration(duration, tstart, tstop);
                return duration;
            }
        };

        template <typename sim_state_t, typename F>
        void with_logger(sim_state_t &sim_state, F &&f) {
            using conserved_t = typename sim_state_t::conserved_t;
            using primitive_t = typename sim_state_t::primitive_t;
            constexpr auto write2file = helpers::write_to_file<typename sim_state_t::primitive_soa_t, sim_state_t::dimensions, sim_state_t>;
            
             // Some benchmarking tools 
            using namespace logger;
            static auto timer    = Timer();
            try {
                if (sim_state.first_order) {
                    timer.startTimer();
                    f();
                    delta_t = timer.get_duration();
                } else {
                    timer.startTimer();
                    f();
                    f();
                    delta_t = timer.get_duration();
                }
                sim_state.t += sim_state.dt;
                if (n >= nfold){
                    ncheck += 1;
                    zu_avg += sim_state.total_zones / delta_t;
                    if constexpr(BuildPlatform == Platform::GPU) {
                        const real gpu_emperical_bw = getFlops<conserved_t, primitive_t>(sim_state.pseudo_radius, sim_state.total_zones, sim_state.active_zones, delta_t);
                        util::writefl("\riteration:{:>06} dt:{:>08.2e} time:{:>08.2e} zones/sec:{:>08.2e} ebw(%):{:>04.2f}", 
                        n, sim_state.dt, sim_state.t, sim_state.total_zones/delta_t, static_cast<real>(100.0) * gpu_emperical_bw / gpu_theoretical_bw);
                    } else {
                        util::writefl("\riteration:{:>06}    dt: {:>08.2e}    time: {:>08.2e}    zones/sec: {:>08.2e}", n, sim_state.dt, sim_state.t, sim_state.total_zones/delta_t);
                    }
                    nfold += 100;
                }

                // Write to a file at every checkpoint interval
                if (sim_state.t >= sim_state.t_interval && sim_state.t != INFINITY)
                {
                    write2file(sim_state, sim_state.setup, sim_state.data_directory, sim_state.t, sim_state.t_interval, sim_state.chkpt_interval, sim_state.checkpoint_zones);
                    if (sim_state.dlogt != 0) {
                        sim_state.t_interval *= std::pow(10, sim_state.dlogt);
                    } else {
                        sim_state.t_interval += sim_state.chkpt_interval;
                    }
                }
                n++;
                // Listen to kill signals
                helpers::catch_signals();
            } catch (helpers::InterruptException &e) {
                util::writeln("{}", e.what());
                sim_state.inFailureState = true;
                write2file(sim_state, sim_state.setup, sim_state.data_directory, sim_state.t, INFINITY, sim_state.chkpt_interval, sim_state.checkpoint_zones);
            }
        }
    } // namespace detail
    
} // namespace simbi
