#ifndef TIMING_HPP
#define TIMING_HPP

#include "hesi/adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

// timing.hpp
namespace simbi::timing {

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

    // manual
    struct timer_t {
        het::executor_t& exec_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop_;

        explicit timer_t(het::executor_t& exec) : exec_(exec) {}

        void start()
        {
            exec_.stream().synchronize();   // wait for pending work
            start_ = std::chrono::high_resolution_clock::now();
        }

        void stop()
        {
            exec_.stream().synchronize();   // wait for work to finish
            stop_ = std::chrono::high_resolution_clock::now();
        }

        double elapsed_seconds() const
        {
            auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    stop_ - start_
                );
            return duration.count();
        }

        // elapsed time from start to now (for mid-timing queries)
        double elapsed_so_far() const
        {
            exec_.stream().synchronize();
            auto now = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    now - start_
                );
            return duration.count();
        }
    };

    // RAII scope timer
    struct scoped_timer_t {
        timer_t timer_;
        timing_stats_t* stats_;
        std::uint64_t nzones_;

        scoped_timer_t(
            het::executor_t& exec,
            timing_stats_t& stats,
            std::uint64_t nzones
        )
            : timer_(exec), stats_(&stats), nzones_(nzones)
        {
            timer_.start();
        }

        ~scoped_timer_t()
        {
            timer_.stop();
            if (stats_) {
                stats_->record(timer_.elapsed_seconds(), nzones_);
            }
        }
    };

}   // namespace simbi::timing

#endif
