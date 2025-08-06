#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace simbi::async {
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

        return std::thread::hardware_concurrency();
    };

    // minimal thread pool implementation
    class thread_pool_t
    {
      private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_;

      public:
        explicit thread_pool_t(std::size_t threads) : stop_(false)
        {
            for (std::size_t i = 0; i < threads; ++i) {
                workers_.emplace_back([this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex_);
                            condition_.wait(lock, [this] {
                                return stop_ || !tasks_.empty();
                            });
                            if (stop_ && tasks_.empty()) {
                                return;
                            }
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
            }
        }

        template <typename Func>
        void submit(Func&& func)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (stop_) {
                    return;
                }
                tasks_.emplace(std::forward<Func>(func));
            }
            condition_.notify_one();
        }

        bool try_execute_one_task()
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (tasks_.empty()) {
                    return false;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();   // execute the task
            return true;
        }

        // work-stealing wait
        template <typename Predicate>
        void wait_while_working(Predicate&& pred)
        {
            while (!pred()) {
                if (!try_execute_one_task()) {
                    std::this_thread::yield();
                }
            }
        }

        ~thread_pool_t()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                stop_ = true;
            }
            condition_.notify_all();
            for (std::thread& worker : workers_) {
                worker.join();
            }
        }
    };

    // singleton thread pool manager - lazy initialization
    class thread_pool_manager_t
    {
      public:
        static thread_pool_t& get_pool()
        {
            static thread_pool_t singleton(get_nthreads());
            return singleton;
        }

        static std::size_t get_nthreads()
        {
            return ::simbi::async::get_nthreads();
        }
    };
}   // namespace simbi::async
#endif
