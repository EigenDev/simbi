// =============================================================================
// timing.hpp
//
// timing utilities for performance measurement
// uses xpu executor synchronization for accurate gpu timing
//
// usage:
//   timer_t timer(exec);
//   timer.start();
//   // ... work ...
//   timer.stop();
//   double seconds = timer.elapsed_seconds();
//
//   // or use raii:
//   {
//     scoped_timer_t timer(exec, stats, nzones);
//     // ... work ...
//   } // automatically records timing
// =============================================================================

#ifndef TIMING_HPP
#define TIMING_HPP

#include "compat.hpp"
#include "xpu/xpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

namespace simbi::timing {

    // -------------------------------------------------------------------------
    // timing statistics accumulator
    // -------------------------------------------------------------------------
    struct timing_stats_t
    {
        double        total_time{0.0};
        double        zone_updates{0.0};
        double        min_time{std::numeric_limits<double>::max()};
        double        max_time{0.0};
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

    // -------------------------------------------------------------------------
    // manual timer
    // -------------------------------------------------------------------------
    struct timer_t
    {
        xpu::executor_t<xpu::default_space>&                        exec_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop_;

        explicit timer_t(xpu::executor_t<xpu::default_space>& exec) : exec_(exec) {}

        void start()
        {
            exec_.wait();
            start_ = std::chrono::high_resolution_clock::now();
        }

        void stop()
        {
            exec_.wait();
            stop_ = std::chrono::high_resolution_clock::now();
        }

        double elapsed_seconds() const
        {
            auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(stop_ - start_);
            return duration.count();
        }

        // elapsed time from start to now
        double elapsed_so_far() const
        {
            exec_.wait();
            auto now      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_);
            return duration.count();
        }
    };

    // -------------------------------------------------------------------------
    // raii scoped timer
    // -------------------------------------------------------------------------
    struct scoped_timer_t
    {
        timer_t         timer_;
        timing_stats_t* stats_;
        std::uint64_t   nzones_;

        scoped_timer_t(
            xpu::executor_t<xpu::default_space>& exec,
            timing_stats_t&                      stats,
            std::uint64_t                        nzones
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

} // namespace simbi::timing

#endif // TIMING_HPP
