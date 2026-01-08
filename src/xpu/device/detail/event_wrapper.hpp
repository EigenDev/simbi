// =============================================================================
// event_wrapper.hpp
//
// space-specific event management with raii semantics for phase 2.
// provides type-safe event creation, destruction, and synchronization.
// preserves hesi semantics while using execution space abstractions.
//
// design principles:
//   - raii: automatic resource management
//   - type-safe: template-based space dispatch
//   - zero-overhead: compile-time space selection
//   - hesi-compatible: preserves original async semantics
//
// usage:
//   event_wrapper_t<cuda_space> event;
//   event.record(stream);
//   event.wait();
// =============================================================================

#pragma once

#include "stream_wrapper.hpp"
#include "xpu/execution/execution_space.hpp"

#include <utility>

namespace simbi::xpu::detail {

    // =============================================================================
    // event wrapper implementation
    // =============================================================================

    template <execution_space ExecutionSpace>
    class event_wrapper_t
    {
      public:
        using execution_space_type = ExecutionSpace;
        using event_handle_type    = typename ExecutionSpace::event_handle_type;
        using stream_handle_type   = typename ExecutionSpace::stream_handle_type;

      private:
        event_handle_type handle_;
        bool              owns_resource_ = true;

      public:
        // =============================================================================
        // construction and destruction
        // =============================================================================

        event_wrapper_t()
        {
            handle_ = ExecutionSpace::create_event();
        }

        // construct from existing handle (non-owning)
        explicit event_wrapper_t(event_handle_type handle, bool owns = false)
            : handle_(handle), owns_resource_(owns)
        {
        }

        ~event_wrapper_t()
        {
            if (owns_resource_) {
                if constexpr (requires { ExecutionSpace::destroy_event(handle_); }) {
                    ExecutionSpace::destroy_event(handle_);
                }
                else if constexpr (requires { handle_ != event_handle_type{}; }) {
                    // fallback for spaces without explicit destroy
                    if (handle_ != event_handle_type{}) {
                        // space-specific cleanup would go here
                    }
                }
            }
        }

        // move semantics (preserves hesi move-only semantics)
        event_wrapper_t(event_wrapper_t&& other) noexcept
            : handle_(std::exchange(other.handle_, {})),
              owns_resource_(std::exchange(other.owns_resource_, false))
        {
        }

        event_wrapper_t& operator=(event_wrapper_t&& other) noexcept
        {
            if (this != &other) {
                if (owns_resource_) {
                    if constexpr (requires { ExecutionSpace::destroy_event(handle_); }) {
                        ExecutionSpace::destroy_event(handle_);
                    }
                }
                handle_        = std::exchange(other.handle_, {});
                owns_resource_ = std::exchange(other.owns_resource_, false);
            }
            return *this;
        }

        // no copy (hesi semantics)
        event_wrapper_t(const event_wrapper_t&)            = delete;
        event_wrapper_t& operator=(const event_wrapper_t&) = delete;

        // =============================================================================
        // event operations
        // =============================================================================

        void record(stream_handle_type stream)
        {
            if constexpr (requires { ExecutionSpace::record_event(handle_, stream); }) {
                ExecutionSpace::record_event(handle_, stream);
            }
            else {
                // fallback for spaces without explicit record
                // cpu spaces typically don't need event recording
            }
        }

        void record(const stream_wrapper_t<ExecutionSpace>& stream)
        {
            record(stream.native_handle());
        }

        void wait() const
        {
            if constexpr (requires { ExecutionSpace::wait_for_event(handle_); }) {
                ExecutionSpace::wait_for_event(handle_);
            }
            else if constexpr (requires { ExecutionSpace::synchronize_event(handle_); }) {
                ExecutionSpace::synchronize_event(handle_);
            }
            else {
                // fallback: no-op for spaces without event sync
            }
        }

        void wait_on(stream_handle_type stream) const
        {
            if constexpr (requires { ExecutionSpace::stream_wait_event(stream, handle_); }) {
                ExecutionSpace::stream_wait_event(stream, handle_);
            }
            else {
                // fallback: just wait for event completion
                wait();
            }
        }

        void wait_on(const stream_wrapper_t<ExecutionSpace>& stream) const
        {
            wait_on(stream.native_handle());
        }

        bool ready() const
        {
            if constexpr (requires { ExecutionSpace::is_event_ready(handle_); }) {
                return ExecutionSpace::is_event_ready(handle_);
            }
            else {
                // fallback: assume ready if no query method
                return true;
            }
        }

        void sync() const
        {
            wait();
        }

        // =============================================================================
        // resource access
        // =============================================================================

        event_handle_type native_handle() const noexcept
        {
            return handle_;
        }

        event_handle_type get() const noexcept
        {
            return handle_;
        }

        bool owns_resource() const noexcept
        {
            return owns_resource_;
        }

        // =============================================================================
        // utility
        // =============================================================================

        explicit operator bool() const noexcept
        {
            if constexpr (std::is_pointer_v<event_handle_type>) {
                return handle_ != nullptr;
            }
            else if constexpr (requires { handle_ != event_handle_type{}; }) {
                return handle_ != event_handle_type{};
            }
            else {
                return true; // assume valid if can't check
            }
        }

        // release ownership without destroying
        event_handle_type release() noexcept
        {
            owns_resource_ = false;
            return std::exchange(handle_, {});
        }

        // reset with new handle
        void reset(event_handle_type new_handle = {}, bool owns = true)
        {
            if (owns_resource_ && handle_) {
                if constexpr (requires { ExecutionSpace::destroy_event(handle_); }) {
                    ExecutionSpace::destroy_event(handle_);
                }
            }
            handle_        = new_handle;
            owns_resource_ = owns;
        }

        // =============================================================================
        // space-specific optimizations
        // =============================================================================

#ifdef XPU_USE_CUDA
        // cuda-specific event operations
        template <typename Space = ExecutionSpace>
        auto cuda_event() const noexcept
            -> std::enable_if_t<std::is_same_v<Space, cuda_space>, cudaEvent_t>
        {
            static_assert(std::is_same_v<Space, ExecutionSpace>);
            return handle_;
        }
#endif

        // cpu-specific completion check
        template <typename Space = ExecutionSpace>
        auto is_cpu_ready() const noexcept
            -> std::enable_if_t<std::is_same_v<Space, cpu_space>, bool>
        {
            static_assert(std::is_same_v<Space, ExecutionSpace>);
            // cpu events are typically always ready
            return true;
        }
    };

    // =============================================================================
    // event factory functions
    // =============================================================================

    template <execution_space ExecutionSpace>
    event_wrapper_t<ExecutionSpace> make_event()
    {
        return event_wrapper_t<ExecutionSpace>{};
    }

    // create immediate (ready) event
    template <execution_space ExecutionSpace>
    event_wrapper_t<ExecutionSpace> make_immediate_event()
    {
        auto event = event_wrapper_t<ExecutionSpace>{};
        // immediate events are ready without recording
        return event;
    }

    // create non-owning wrapper around existing handle
    template <execution_space ExecutionSpace>
    event_wrapper_t<ExecutionSpace> wrap_event(typename ExecutionSpace::event_handle_type handle)
    {
        return event_wrapper_t<ExecutionSpace>{handle, false};
    }

} // namespace simbi::xpu::detail
