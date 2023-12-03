/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 * @file       thread_pool.hpp
 * @brief      implements a custom thread pool using STL
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */
#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <algorithm>            // for copy
#include <atomic>               // for atomic
#include <cmath>                // for ceil
#include <condition_variable>   // for condition_variable
#include <cstdlib>              // for getenv
#include <functional>           // for function
#include <mutex>                // for mutex, unique_lock
#include <queue>                // for queue
#include <string>               // for allocator, stoul, string
#include <thread>               // for jthread
#include <utility>              // for move, swap
#include <vector>               // for vector

namespace simbi {
    namespace pooling {
#if __cplusplus >= 202002L && !defined(__clang__)
        constexpr bool need_join = false;
        using std_thread         = std::jthread;
#else
        constexpr bool need_join = true;
        using std_thread         = std::thread;
#endif
        /**
         * Practicing ThreadPooling based on these resources:
         * https://stackoverflow.com/a/32593825/13874039
         * https://www.cnblogs.com/sinkinben/p/16064857.html
         * https://gist.github.com/tonykero/9512f2fb7f47d1ee687ae8595b17666e
         * https://stackoverflow.com/questions/23896421/efficiently-waiting-for-all-tasks-in-a-threadpool-to-finish
         */
        class ThreadPool
        {
          public:
            static ThreadPool& instance(const unsigned int nthreads)
            {
                static ThreadPool singleton(nthreads);
                return singleton;
            }

            void queueUp(const std::function<void()>& job)
            {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    jobs.push(job);
                }
                cv_task.notify_one();
            }

            template <typename index_type, typename F>
            void parallel_for(const index_type start,
                              const index_type stop,
                              const F& func)
            {
                if (global::use_omp) {
#pragma omp parallel for schedule(static)
                    for (auto idx = start; idx < stop; idx++) {
                        func(idx);
                    }
                    return;
                }
                static unsigned batch_size =
                    std::ceil((float) (stop - start) / (float) nthreads);
                int block_start = start - batch_size;
                int block_end   = start;

                auto step = [&] {
                    block_start += batch_size;
                    block_end += batch_size;
                    block_end =
                        ((index_type) block_end > stop) ? stop : block_end;
                };
                step();

                for ([[gnu::unused]] auto& worker : threads) {
                    queueUp([block_start, block_end, func] {
                        for (auto q = block_start; q < block_end; q++) {
                            func(q);
                        }
                    });
                    step();
                }

                waitUntilFinished();
            }

            bool poolBusy()
            {
                bool poolbusy;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    poolbusy = !(jobs.empty() && busy == 0);
                }
                return poolbusy;
            };

          private:
            ThreadPool(const ThreadPool&)            = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;

            ThreadPool(const unsigned int nthreads)
                : nthreads(nthreads), should_terminate(false), busy(0)
            {
                threads.reserve(nthreads);
                for (unsigned i = 0; i < nthreads; i++) {
                    threads.emplace_back(
                        std_thread(&ThreadPool::spawn_thread_proc, this));
                }
            }

            ~ThreadPool()
            {
                // Stop the thread pool and notify all threads ot finish the
                // remaining tasks
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
                    std::unique_lock<std::mutex> latch(queue_mutex);
                    cv_task.wait(latch, [this] {
                        return !jobs.empty() || should_terminate;
                    });
                    // pop a job from queue and execute it
                    {
                        if (should_terminate) {
                            return;
                        }

                        // got work. set busy.
                        ++busy;
                        job = std::move(jobs.front());
                        jobs.pop();
                    }
                    latch.unlock();
                    job();
                    latch.lock();
                    --busy;
                    cv_finished.notify_one();
                }
            }

            void spawn_thread()
            {
                while (true) {
                    std::function<void()> job;
                    // pop a job from queue and execute it
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
                    }
                    job();
                }
            }

            // waits until the queue is empty.
            void waitUntilFinished()
            {
                std::unique_lock<std::mutex> latch(queue_mutex);
                cv_finished.wait(latch,
                                 [this] { return jobs.empty() && busy == 0; });
            }

            unsigned int nthreads;
            bool should_terminate;    // Tells threads to stop looking for jobs
            std::mutex queue_mutex;   // Prevents data races to the job queue
            std::condition_variable
                cv_task;   // Allows threads to wait on new jobs or termination
            std::condition_variable cv_finished;
            std::vector<std_thread> threads;
            std::queue<std::function<void()>> jobs;
            unsigned int busy;
        };

        inline auto get_nthreads = ([] {
            if (const char* thread_env = std::getenv("NTHREADS")) {
                return static_cast<unsigned int>(
                    std::stoul(std::string(thread_env)));
            }

            if (const char* thread_env = std::getenv("OMP_NUM_THREADS")) {
                return static_cast<unsigned int>(
                    std::stoul(std::string(thread_env)));
            }

            return std_thread::hardware_concurrency();
        });

        template <typename T> T fetch_minimum(std::atomic<T>& a, T val)
        {
            T old = a;
            while (old > val && !a.compare_exchange_weak(old, val)) {
            }
            return old;
        }

        template <typename T>
        void update_minimum(std::atomic<T>& a, T const& value) noexcept
        {
            T old = a;
            while (old > value && !a.compare_exchange_weak(old, value)) {
            }
        }
    }   // namespace pooling
}   // namespace simbi
#endif