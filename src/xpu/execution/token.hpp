// =============================================================================
// token.hpp
//
// phase 2 spec-compliant token implementation for xpu framework.
// provides type-safe async operation tokens with execution space parameter,
// preserving hesi async semantics while using clean xpu abstractions.
//
// design principles:
//   - template on execution space for compile-time dispatch
//   - direct event handle management (no futures internally)
//   - hesi-compatible api with record/wait/sync semantics
//   - raii resource management for events
//
// usage:
//   token_t<cuda_space> token = token_t<cuda_space>::create();
//   token.record(executor);
//   token.wait();
// =============================================================================

#pragma once

#include "cpu_space.hpp"
#include "cuda_space.hpp"
#include "execution_space.hpp"
#include "xpu/device/detail/event_wrapper.hpp"

#include <utility>

namespace simbi::xpu {

    // forward declaration
    template <execution_space ExecutionSpace>
    class executor_t;

    // =============================================================================
    // token implementation
    // =============================================================================

    template <execution_space ExecutionSpace>
    class token_t
    {
      public:
        using execution_space_type = ExecutionSpace;
        using event_handle_type    = typename ExecutionSpace::event_handle_type;

      private:
        detail::event_wrapper_t<ExecutionSpace> event_;
        bool                                    owns_resource_ = true;
        bool                                    is_ready_      = false;

      public:
        // =============================================================================
        // construction and destruction
        // =============================================================================

        // private constructor - use factory functions
        explicit token_t(detail::event_wrapper_t<ExecutionSpace>&& event, bool owns = true)
            : event_(std::move(event)), owns_resource_(owns), is_ready_(false)
        {
        }

        // private constructor for immediate tokens
        explicit token_t(bool ready) : owns_resource_(false), is_ready_(ready) {}

        ~token_t() = default;

        // move-only (preserves hesi semantics)
        token_t(token_t&& other) noexcept
            : event_(std::move(other.event_)),
              owns_resource_(std::exchange(other.owns_resource_, false)),
              is_ready_(std::exchange(other.is_ready_, false))
        {
        }

        token_t& operator=(token_t&& other) noexcept
        {
            if (this != &other) {
                event_         = std::move(other.event_);
                owns_resource_ = std::exchange(other.owns_resource_, false);
                is_ready_      = std::exchange(other.is_ready_, false);
            }
            return *this;
        }

        // no copy (hesi semantics)
        token_t(const token_t&)            = delete;
        token_t& operator=(const token_t&) = delete;

        // =============================================================================
        // factory functions (preserve hesi semantics)
        // =============================================================================

        static token_t create()
        {
            auto event = detail::make_event<ExecutionSpace>();
            return token_t{std::move(event), true};
        }

        static token_t immediate()
        {
            return token_t{true}; // no-op token that's immediately ready
        }

        // =============================================================================
        // async operations (preserve hesi semantics)
        // =============================================================================

        void record(const executor_t<ExecutionSpace>& exec)
        {
            if (!is_ready_ && owns_resource_) {
                event_.record(exec.stream());
                is_ready_ = false; // recorded but not necessarily complete
            }
        }

        void wait_on(const executor_t<ExecutionSpace>& exec) const
        {
            if (!is_ready_ && owns_resource_) {
                event_.wait_on(exec.stream());
            }
        }

        // cross-device wait: make another executor wait for this token's work
        // useful for multi-gpu synchronization (e.g., gpu1 waits for gpu0)
        template <typename OtherSpace>
        void wait_on_cross_device(const executor_t<OtherSpace>& other_exec) const
        {
#ifdef XPU_CUDA_AVAILABLE
            if constexpr (std::is_same_v<ExecutionSpace, cuda_space> &&
                          std::is_same_v<OtherSpace, cuda_space>) {
                if (!is_ready_ && owns_resource_) {
                    // make other_exec's stream wait for this token's event
                    cudaEvent_t event = native_handle();
                    cudaStreamWaitEvent(other_exec.stream(), event, 0);
                }
            }
            else {
                // fallback: explicit sync for cpu or mixed spaces
                sync();
            }
#else
            (void) other_exec;
            sync();
#endif
        }

        void sync()
        {
            if (!is_ready_) {
                if (owns_resource_) {
                    event_.sync();
                }
                is_ready_ = true;
            }
        }

        bool ready() const
        {
            if (is_ready_) {
                return true;
            }

            if (!owns_resource_) {
                return true; // immediate tokens are always ready
            }

            bool event_ready = event_.ready();
            if (event_ready) {
                const_cast<token_t*>(this)->is_ready_ = true;
            }
            return event_ready;
        }

        void wait() const
        {
            if (!is_ready_) {
                const_cast<token_t*>(this)->sync();
            }
        }

        // =============================================================================
        // resource access
        // =============================================================================

        event_handle_type native_handle() const
        {
            if (owns_resource_) {
                return event_.native_handle();
            }
            else {
                return {}; // immediate token has no event
            }
        }

        bool owns_resource() const noexcept
        {
            return owns_resource_;
        }

        constexpr execution_space_type execution_space() const
        {
            return {};
        }

        // =============================================================================
        // utility
        // =============================================================================

        explicit operator bool() const
        {
            return ready();
        }

        void mark_ready()
        {
            is_ready_ = true;
        }

        bool is_immediate() const noexcept
        {
            return !owns_resource_;
        }

        // =============================================================================
        // space-specific accessors
        // =============================================================================

#ifdef XPU_USE_CUDA
        // cuda-specific event access
        cudaEvent_t cuda_event() const noexcept
            requires std::same_as<ExecutionSpace, cuda_space>
        {
            if (owns_resource_) {
                return event_.template cuda_event<ExecutionSpace>();
            }
            else {
                return nullptr; // immediate token
            }
        }
#endif

        // cpu-specific completion check
        bool is_cpu_ready() const noexcept
            requires std::same_as<ExecutionSpace, cpu_space>
        {
            // cpu events are typically always ready
            return true;
        }

        // =============================================================================
        // advanced operations
        // =============================================================================

        // chain with another token (dependency)
        template <typename Kernel, typename... Args>
        token_t then(executor_t<ExecutionSpace>& exec, Kernel&& kernel, Args&&... args) const
        {
            return exec.then(*this, std::forward<Kernel>(kernel), std::forward<Args>(args)...);
        }

        // join with other tokens
        static token_t join(const std::vector<token_t>& tokens)
        {
            if (tokens.empty()) {
                return immediate();
            }

            // create new token for the join operation
            auto result = create();

            // wait for all tokens (simple implementation)
            for (const auto& token : tokens) {
                token.wait();
            }

            result.mark_ready();
            return result;
        }

        // cross-device join: wait for tokens from different devices
        template <typename OtherSpace>
        static token_t join_cross_device(
            const std::vector<token_t<ExecutionSpace>>& local_tokens,
            const std::vector<token_t<OtherSpace>>&     remote_tokens
        )
        {
            // wait for all local tokens
            for (const auto& token : local_tokens) {
                token.wait();
            }

            // wait for all remote tokens
            for (const auto& token : remote_tokens) {
                token.wait();
            }

            auto result = create();
            result.mark_ready();
            return result;
        }
    };

    // =============================================================================
    // free functions for token operations
    // =============================================================================

    template <execution_space ExecutionSpace>
    token_t<ExecutionSpace> make_ready_token()
    {
        return token_t<ExecutionSpace>::immediate();
    }

    template <execution_space ExecutionSpace>
    token_t<ExecutionSpace> make_token()
    {
        return token_t<ExecutionSpace>::create();
    }

    template <execution_space ExecutionSpace>
    void wait_all(const std::vector<token_t<ExecutionSpace>>& tokens)
    {
        for (const auto& token : tokens) {
            token.wait();
        }
    }

    template <execution_space ExecutionSpace>
    bool all_ready(const std::vector<token_t<ExecutionSpace>>& tokens)
    {
        for (const auto& token : tokens) {
            if (!token.ready()) {
                return false;
            }
        }
        return true;
    }

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using cpu_token = token_t<cpu_space>;

#ifdef XPU_USE_CUDA
    using cuda_token = token_t<cuda_space>;
#endif

} // namespace simbi::xpu
