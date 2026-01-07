// =============================================================================
// thread_pool.hpp
//
// a simple thread pool for asynchronous cpu execution.
//
// design principles:
//   - fixed-size pool of worker threads.
//   - tasks are submitted as std::function objects.
//   - returns std::future to track task completion.
// =============================================================================

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace simbi::xpu::detail {

    class thread_pool_t
    {
      public:
        thread_pool_t(size_t);
        template <class F, class... Args>
        auto enqueue(F&& f, Args&&... args)
            -> std::future<typename std::invoke_result<F, Args...>::type>;
        ~thread_pool_t();

      private:
        std::vector<std::thread>          workers;
        std::queue<std::function<void()>> tasks;
        std::mutex                        queue_mutex;
        std::condition_variable           condition;
        bool                              stop;
    };

    inline thread_pool_t::thread_pool_t(size_t threads) : stop(false)
    {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto thread_pool_t::enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped thread_pool_t");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    inline thread_pool_t::~thread_pool_t()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

} // namespace simbi::xpu::detail
