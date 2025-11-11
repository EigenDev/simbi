#ifndef FUTURE_HPP
#define FUTURE_HPP

#include "completion.hpp"
#include "hetero/adapter.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace simbi::exec {
    // forward declarations
    class cpu_executor_t;
    class par_cpu_executor_t;
    class gpu_executor_t;
    class omp_executor_t;

    template <typename T>
    struct future_t {
      private:
        struct future_state_t {
            std::atomic<bool> ready{false};
            std::atomic<bool> has_error{false};
            alignas(T) std::byte result_storage[sizeof(T)];
            std::exception_ptr exception;
            hetero::stream stream{};
            hetero::event event{};
            std::condition_variable cv;
            std::mutex mutex;
            completion_context_t completion_context;
            std::vector<hetero::event> completion_events;

            T& result() { return *reinterpret_cast<T*>(result_storage); }

            const T& result() const
            {
                return *reinterpret_cast<const T*>(result_storage);
            }

            template <typename... Args>
            void construct_result(Args&&... args)
            {
                new (result_storage) T(std::forward<Args>(args)...);
            }

            void destroy_result()
            {
                if (ready.load() && !has_error.load()) {
                    result().~T();
                }
            }

            ~future_state_t() { destroy_result(); }
        };
        std::shared_ptr<future_state_t> state_;

      public:
        template <typename ExecutorType>
        friend class executor_base_t;

        // grant access to the specific executor classes
        friend class cpu_executor_t;
        friend class par_cpu_executor_t;
        friend class omp_executor_t;
        friend class gpu_executor_t;

        future_t(const future_t&)            = delete;
        future_t& operator=(const future_t&) = delete;
        future_t(future_t&&)                 = default;
        future_t& operator=(future_t&&)      = default;

        const T& wait() const
        {
            if (!state_->ready.load()) {
                wait_impl();
            }

            if (state_->has_error.load()) {
                std::rethrow_exception(state_->exception);
            }

            return state_->result();
        }

        bool is_ready() const
        {
            if (state_->ready.load()) {
                return true;
            }
            return check_completion();
        }

        const T& get_unsafe() const { return state_->result(); }

      private:
        explicit future_t(std::shared_ptr<future_state_t> state)
            : state_(std::move(state))
        {
        }

        void wait_impl() const
        {
            if (!state_->completion_events.empty()) {
                for (auto& event : state_->completion_events) {
                    event.synchronize();
                }
                state_->ready.store(true);
            }
            else {
                state_->completion_context
                    .wait_fn(state_->ready, state_->mutex, state_->cv);
            }
        }

        bool check_completion() const
        {
            if (state_->stream) {
                if (state_->stream.query_complete()) {
                    bool expected = false;
                    state_->ready.compare_exchange_strong(expected, true);
                    return true;
                }
            }
            return false;
        }
    };

    // void specialization
    template <>
    struct future_t<void> {
      private:
        struct future_state_t {
            std::atomic<bool> ready{false};
            std::atomic<bool> has_error{false};
            std::exception_ptr exception;
            hetero::stream stream{};
            hetero::event event{};
            std::condition_variable cv;
            std::mutex mutex;
            completion_context_t completion_context;
            std::vector<hetero::event> completion_events;

            ~future_state_t() = default;
        };
        std::shared_ptr<future_state_t> state_;

      public:
        template <typename ExecutorType>
        friend class executor_base_t;

        friend class cpu_executor_t;
        friend class par_cpu_executor_t;
        friend class omp_executor_t;
        friend class gpu_executor_t;

        future_t(const future_t&)            = delete;
        future_t& operator=(const future_t&) = delete;
        future_t(future_t&&)                 = default;
        future_t& operator=(future_t&&)      = default;

        void wait() const
        {
            if (!state_->ready.load()) {
                wait_impl();
            }

            if (state_->has_error.load()) {
                std::rethrow_exception(state_->exception);
            }
        }

        bool is_ready() const
        {
            if (state_->ready.load()) {
                return true;
            }

            if (state_->stream) {
                if (state_->stream.query_complete()) {
                    state_->ready.store(true);
                    return true;
                }
            }
            return false;
        }

      private:
        explicit future_t(std::shared_ptr<future_state_t> state)
            : state_(std::move(state))
        {
        }

        void wait_impl() const
        {
            if (!state_->completion_events.empty()) {
                for (auto& event : state_->completion_events) {
                    event.synchronize();
                }
                state_->ready.store(true);
            }
            else {
                state_->completion_context
                    .wait_fn(state_->ready, state_->mutex, state_->cv);
            }
        }
    };

}   // namespace simbi::exec

#endif   // FUTURE_HPP
