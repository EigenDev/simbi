#ifndef COMPLETION_HPP
#define COMPLETION_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "thread_pool.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace simbi::async {
    /**
     * Completion context for waiting on futures or tasks.
     * Provides different strategies for waiting based on the execution context.
     *
     */
    struct completion_context_t {
        std::function<void(
            const std::atomic<bool>&,
            std::mutex&,
            std::condition_variable&
        )>
            wait_fn;

        // constructor for work-stealing
        static completion_context_t
        work_stealing(thread_pool_t* pool = &thread_pool_manager_t::get_pool())
        {
            return {[pool](const auto& ready, auto& /*mutex*/, auto& /*cv*/) {
                pool->wait_while_working([&ready] { return ready.load(); });
            }};
        }

        // constructor for direct waiting
        static completion_context_t direct()
        {
            return {[](const auto& ready, auto& mutex, auto& cv) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&ready] { return ready.load(); });
            }};
        }

        // constructor for gpu streams
        static completion_context_t gpu_stream(adapter::stream_t<> stream)
        {
            return {
              [stream](const auto& /*ready*/, auto& /*mutex*/, auto& /*cv*/) {
                  if (stream) {
                      gpu::api::stream_synchronize(stream);
                  }
              }
            };
        }
    };

}   // namespace simbi::async
#endif
