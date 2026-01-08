// =============================================================================
// stream_wrapper.hpp
//
// space-specific stream management with raii semantics for phase 2.
// provides type-safe stream creation, destruction, and operations.
// preserves hesi semantics while using execution space abstractions.
//
// design principles:
//   - raii: automatic resource management
//   - type-safe: template-based space dispatch
//   - zero-overhead: compile-time space selection
//   - hesi-compatible: preserves original async semantics
//
// usage:
//   stream_wrapper_t<cuda_space> stream(device_id);
//   stream.sync();
//   auto handle = stream.native_handle();
// =============================================================================

#pragma once

#include "xpu/execution/execution_space.hpp"

#include <thread>
#include <utility>

namespace simbi::xpu::detail {

    // =============================================================================
    // stream wrapper implementation
    // =============================================================================

    template <execution_space ExecutionSpace>
    class stream_wrapper_t
    {
      public:
        using execution_space_type = ExecutionSpace;
        using stream_handle_type   = typename ExecutionSpace::stream_handle_type;

      private:
        stream_handle_type handle_;
        int                device_id_;
        bool               owns_resource_ = true;

      public:
        // =============================================================================
        // construction and destruction
        // =============================================================================

        explicit stream_wrapper_t(std::int64_t device_id = 0) : device_id_(device_id)
        {
            if constexpr (requires { ExecutionSpace::set_device(device_id); }) {
                ExecutionSpace::set_device(device_id);
            }
            handle_ = ExecutionSpace::create_stream();
        }

        // construct from existing handle (non-owning)
        stream_wrapper_t(stream_handle_type handle, std::int64_t device_id, bool owns = false)
            : handle_(handle), device_id_(device_id), owns_resource_(owns)
        {
        }

        ~stream_wrapper_t()
        {
            if (owns_resource_) {
                ExecutionSpace::destroy_stream(handle_);
            }
        }

        // move semantics (preserves hesi move-only semantics)
        stream_wrapper_t(stream_wrapper_t&& other) noexcept
            : handle_(std::exchange(other.handle_, {})), device_id_(other.device_id_),
              owns_resource_(std::exchange(other.owns_resource_, false))
        {
        }

        stream_wrapper_t& operator=(stream_wrapper_t&& other) noexcept
        {
            if (this != &other) {
                if (owns_resource_) {
                    ExecutionSpace::destroy_stream(handle_);
                }
                handle_        = std::exchange(other.handle_, {});
                device_id_     = other.device_id_;
                owns_resource_ = std::exchange(other.owns_resource_, false);
            }
            return *this;
        }

        // no copy (hesi semantics)
        stream_wrapper_t(const stream_wrapper_t&)            = delete;
        stream_wrapper_t& operator=(const stream_wrapper_t&) = delete;

        // =============================================================================
        // stream operations
        // =============================================================================

        void sync()
        {
            ExecutionSpace::synchronize_stream(handle_);
        }

        bool ready() const
        {
            if constexpr (requires { ExecutionSpace::is_stream_ready(handle_); }) {
                return ExecutionSpace::is_stream_ready(handle_);
            }
            else {
                // fallback: assume ready if no query method
                return true;
            }
        }

        void wait() const
        {
            sync();
        }

        // =============================================================================
        // resource access
        // =============================================================================

        stream_handle_type native_handle() const noexcept
        {
            return handle_;
        }

        stream_handle_type get() const noexcept
        {
            return handle_;
        }

        std::int64_t device_id() const noexcept
        {
            return device_id_;
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
            if constexpr (std::is_pointer_v<stream_handle_type>) {
                return handle_ != nullptr;
            }
            else if constexpr (requires { handle_ != stream_handle_type{}; }) {
                return handle_ != stream_handle_type{};
            }
            else {
                return true; // assume valid if can't check
            }
        }

        // release ownership without destroying
        stream_handle_type release() noexcept
        {
            owns_resource_ = false;
            return std::exchange(handle_, {});
        }

        // reset with new handle
        void reset(stream_handle_type new_handle = {}, bool owns = true)
        {
            if (owns_resource_ && handle_) {
                ExecutionSpace::destroy_stream(handle_);
            }
            handle_        = new_handle;
            owns_resource_ = owns;
        }

        // =============================================================================
        // space-specific optimizations
        // =============================================================================

#ifdef XPU_USE_CUDA
        // cuda-specific stream operations
        template <typename Space = ExecutionSpace>
        auto cuda_stream() const noexcept
            -> std::enable_if_t<std::is_same_v<Space, cuda_space>, cudaStream_t>
        {
            static_assert(std::is_same_v<Space, ExecutionSpace>);
            return handle_;
        }
#endif

        // cpu-specific thread id
        template <typename Space = ExecutionSpace>
        auto thread_id() const noexcept
            -> std::enable_if_t<std::is_same_v<Space, cpu_space>, std::thread::id>
        {
            static_assert(std::is_same_v<Space, ExecutionSpace>);
            return handle_;
        }
    };

    // =============================================================================
    // stream factory functions
    // =============================================================================

    template <execution_space ExecutionSpace>
    stream_wrapper_t<ExecutionSpace> make_stream(std::int64_t device_id = 0)
    {
        return stream_wrapper_t<ExecutionSpace>{device_id};
    }

    // create non-owning wrapper around existing handle
    template <execution_space ExecutionSpace>
    stream_wrapper_t<ExecutionSpace>
    wrap_stream(typename ExecutionSpace::stream_handle_type handle, std::int64_t device_id = 0)
    {
        return stream_wrapper_t<ExecutionSpace>{handle, device_id, false};
    }

} // namespace simbi::xpu::detail
