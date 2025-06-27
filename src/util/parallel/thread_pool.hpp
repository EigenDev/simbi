/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            thread_pool.hpp
 *  * @brief           custom thread pool implementation aside from OpenMP
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
#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include "config.hpp"
#include <atomic>               // for atomic
#include <condition_variable>   // for condition_variable
#include <cstdlib>              // for getenv
#include <functional>           // for function
#include <mutex>                // for mutex, unique_lock
#include <queue>                // for queue
#include <stdexcept>
#include <string>    // for allocator, stoul, string
#include <thread>    // for jthread
#include <utility>   // for move, swap
#include <vector>

#if __cplusplus >= 202002L && !defined(__clang__)
constexpr bool need_join = false;
using std_thread         = std::jthread;
#else
constexpr bool need_join = true;
using std_thread         = std::thread;
#endif

namespace simbi::pooling {
    /**
     * @brief A custom thread pool implementation.
     *
     * Practicing thread pooling based on these resources:
     * https://stackoverflow.com/a/32593825/13874039
     * https://www.cnblogs.com/sinkinben/p/16064857.html
     * https://gist.github.com/tonykero/9512f2fb7f47d1ee687ae8595b17666e
     * https://stackoverflow.com/questions/23896421/efficiently-waiting-for-all-tasks-in-a-threadpool-to-finish
     */
    class thread_pool_t
    {
      public:
        static thread_pool_t& instance(const std::uint64_t nthreads)
        {
            static thread_pool_t singleton(nthreads);
            return singleton;
        }

        void queue_up(const std::function<void()>& job)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (should_terminate) {
                    return;
                }
                jobs.push(job);
            }
            cv_task.notify_one();
        }

        template <typename index_type, typename F>
        void parallel_for(
            const index_type start,
            const index_type stop,
            const F& func
        )
        {
            if (global::use_omp) {
#pragma omp parallel for schedule(static)
                for (index_type idx = start; idx < stop; idx++) {
                    func(idx);
                }
                return;
            }
            auto batch_size = static_cast<index_type>(
                (stop - start + nthreads - 1) / nthreads
            );
            index_type block_start = start - batch_size;
            index_type block_end   = start;

            auto step = [&] {
                block_start += batch_size;
                block_end += batch_size;
                block_end = (block_end > stop) ? stop : block_end;
            };
            step();

            for ([[maybe_unused]] auto& worker : threads) {
                queue_up([block_start, block_end, func] {
                    for (index_type q = block_start; q < block_end; q++) {
                        func(q);
                    }
                });
                step();
            }

            wait_until_finished();
        }

        template <typename index_type, typename F>
        void parallel_for(const index_type stop, const F& func)
        {
            parallel_for(static_cast<index_type>(0), stop, func);
        }

        template <typename index_type, typename F>
        void parallel_for(
            const index_type start,
            const index_type stop,
            const index_type step,
            const F& func
        )
        {
            if (global::use_omp) {
#pragma omp parallel for schedule(static)
                for (index_type idx = start; idx < stop; idx += step) {
                    func(idx);
                }
                return;
            }
            auto batch_size = static_cast<index_type>(
                (stop - start + nthreads - 1) / nthreads
            );
            index_type block_start = start - batch_size;
            index_type block_end   = start;

            auto blockStep = [&] {
                block_start += batch_size;
                block_end += batch_size;
                block_end = (block_end > stop) ? stop : block_end;
            };
            blockStep();

            for ([[maybe_unused]] auto& worker : threads) {
                queue_up([block_start, block_end, step, func] {
                    for (index_type q = block_start; q < block_end; q += step) {
                        func(q);
                    }
                });
                blockStep();
            }

            wait_until_finished();
        }

        bool poolBusy()
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            return !(jobs.empty() && busy == 0);
        }

      private:
        thread_pool_t(const thread_pool_t&)            = delete;
        thread_pool_t& operator=(const thread_pool_t&) = delete;

        thread_pool_t(const std::uint64_t nthreads)
            : nthreads(nthreads), should_terminate(false), busy(0)
        {
            if (nthreads == 0) {
                throw std::invalid_argument(
                    "Number of threads must be greater than zero."
                );
            }
            threads.reserve(nthreads);
            for (std::uint64_t i = 0; i < nthreads; i++) {
                threads.emplace_back(
                    std_thread(&thread_pool_t::spawn_thread_proc, this)
                );
            }
        }

        ~thread_pool_t()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                should_terminate = true;
            }
            cv_task.notify_all();
            if constexpr (need_join) {
                for (std_thread& active_thread : threads) {
                    active_thread.join();
                }
            }
            threads.clear();
        }

        void spawn_thread_proc()
        {
            while (true) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> latch(queue_mutex);
                    cv_task.wait(latch, [this] {
                        return !jobs.empty() || should_terminate;
                    });
                    if (should_terminate) {
                        return;
                    }
                    job = std::move(jobs.front());
                    jobs.pop();
                    busy.fetch_add(1, std::memory_order_relaxed);
                }
                try {
                    job();
                }
                catch (...) {
                    // Handle exceptions if necessary
                }
                {
                    std::unique_lock<std::mutex> latch(queue_mutex);
                    busy.fetch_sub(1, std::memory_order_relaxed);
                    cv_finished.notify_one();
                }
            }
        }

        void wait_until_finished()
        {
            std::unique_lock<std::mutex> latch(queue_mutex);
            cv_finished.wait(latch, [this] {
                return jobs.empty() &&
                       busy.load(std::memory_order_relaxed) == 0;
            });
        }

        std::uint64_t nthreads;
        bool should_terminate;
        std::mutex queue_mutex;
        std::condition_variable cv_task;
        std::condition_variable cv_finished;
        std::vector<std_thread> threads;
        std::queue<std::function<void()>> jobs;
        std::atomic<std::uint64_t> busy;
    };

    inline auto get_nthreads() -> std::uint64_t
    {
        if (const char* thread_env = std::getenv("NTHREADS")) {
            return static_cast<std::uint64_t>(
                std::stoul(std::string(thread_env))
            );
        }

        if (const char* thread_env = std::getenv("OMP_NUM_THREADS")) {
            return static_cast<std::uint64_t>(
                std::stoul(std::string(thread_env))
            );
        }

        return std_thread::hardware_concurrency();
    };

    // Global accessor function
    inline thread_pool_t& get_thread_pool()
    {
        static thread_pool_t& pool = thread_pool_t::instance(get_nthreads());
        return pool;
    }

    template <typename T>
    T fetch_minimum(std::atomic<T>& a, T val)
    {
        T old = a.load(std::memory_order_relaxed);
        while (old > val &&
               !a.compare_exchange_weak(old, val, std::memory_order_relaxed)) {
            // Add a backoff strategy if necessary
        }
        return old;
    }

    template <typename T>
    void update_minimum(std::atomic<T>& a, T const& value) noexcept
    {
        T old = a.load(std::memory_order_relaxed);
        while (
            old > value &&
            !a.compare_exchange_weak(old, value, std::memory_order_relaxed)
        ) {
            // Add a backoff strategy if necessary
        }
    }
}   // namespace simbi::pooling

#endif   // THREAD_POOL_HPP
