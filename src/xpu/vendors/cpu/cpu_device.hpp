// =============================================================================
// cpu_device.hpp
//
// cpu device implementation for heterogeneous xpu abstraction.
// implements hetero_device concept for cpu execution using openmp.
// provides zero-overhead vendor abstraction while maintaining cpu
// performance for massive simulation workloads.
//
// design principles:
//   - concept-driven: satisfies core::hetero_device concept
//   - performance-first: zero overhead abstractions
//   - production-ready: proper error handling and resource management
//   - extensible: easy to add new cpu features
//
// usage:
//   cpu_device_t device{0};  // cpu device
//   auto stream = device.create_stream();
//   auto ptr = device.allocate(1024);
// =============================================================================

#pragma once

#include "../../core/device_concepts.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <omp.h>
#include <string_view>
#include <thread>

namespace simbi::xpu::vendors::cpu {

    // =============================================================================
    // cpu device handles
    // =============================================================================

    struct cpu_memory_handle_t
    {
        void* ptr = nullptr;

        bool operator==(const cpu_memory_handle_t& other) const noexcept
        {
            return ptr == other.ptr;
        }

        explicit operator bool() const noexcept
        {
            return ptr != nullptr;
        }
    };

    struct cpu_stream_handle_t
    {
        std::thread::id thread_id = std::this_thread::get_id();

        bool operator==(const cpu_stream_handle_t& other) const noexcept
        {
            return thread_id == other.thread_id;
        }

        bool operator==(const std::thread::id& other) const noexcept
        {
            return thread_id == other;
        }

        operator std::thread::id() const noexcept
        {
            return thread_id;
        }
    };

    struct cpu_event_handle_t
    {
        void* state = nullptr; // trivially copyable placeholder

        cpu_event_handle_t() = default;

        explicit cpu_event_handle_t(std::future<void>&& f) : state(reinterpret_cast<void*>(0x1))
        {
            (void) f;
        }

        bool operator==(const cpu_event_handle_t& other) const noexcept
        {
            return state == other.state;
        }

        bool operator==(std::nullptr_t) const noexcept
        {
            return state == nullptr;
        }

        bool operator!=(const cpu_event_handle_t& other) const noexcept
        {
            return state != other.state;
        }

        bool operator!=(std::nullptr_t) const noexcept
        {
            return state != nullptr;
        }

        explicit operator bool() const noexcept
        {
            return state != nullptr;
        }

        bool ready() const noexcept
        {
            return true; // cpu events are always ready
        }
    };

    // =============================================================================
    // cpu device implementation
    // =============================================================================

    class cpu_device_t
    {
      public:
        // concept requirements
        using memory_handle_type = cpu_memory_handle_t;
        using stream_handle_type = cpu_stream_handle_t;
        using event_handle_type  = cpu_event_handle_t;

        // device properties
        static constexpr bool        is_gpu_device       = false;
        static constexpr bool        is_cpu_device       = true;
        static constexpr std::size_t preferred_alignment = alignof(std::max_align_t);

        static constexpr std::string_view vendor_name()
        {
            return "host";
        }

        // construction
        explicit cpu_device_t(std::int64_t device_id = 0) : device_id_(device_id) {}

        ~cpu_device_t() = default;

        // copyable and movable
        cpu_device_t(const cpu_device_t&)            = default;
        cpu_device_t& operator=(const cpu_device_t&) = default;
        cpu_device_t(cpu_device_t&&)                 = default;
        cpu_device_t& operator=(cpu_device_t&&)      = default;

        // =============================================================================
        // memory allocation (device_memory_allocator concept)
        // =============================================================================

        memory_handle_type allocate(std::size_t bytes)
        {
            void* ptr = std::aligned_alloc(
                preferred_alignment,
                (bytes + preferred_alignment - 1) & ~(preferred_alignment - 1)
            );
            if (!ptr && bytes > 0) {
                throw std::bad_alloc{};
            }
            return memory_handle_type{ptr};
        }

        void deallocate(memory_handle_type handle)
        {
            if (handle.ptr) {
                std::free(handle.ptr);
            }
        }

        std::size_t memory_alignment() const
        {
            return preferred_alignment;
        }

        std::size_t max_allocation_size() const
        {
            return SIZE_MAX;
        }

        bool is_accessible_from_host(memory_handle_type) const
        {
            return true; // cpu memory always accessible from host
        }

        // =============================================================================
        // stream management (device_stream_manager concept)
        // =============================================================================

        stream_handle_type create_stream()
        {
            return stream_handle_type{std::this_thread::get_id()};
        }

        void destroy_stream(stream_handle_type)
        {
            // no-op for cpu streams
        }

        void synchronize_stream(stream_handle_type)
        {
            // cpu work is inherently synchronous within thread
        }

        bool is_stream_ready(stream_handle_type) const
        {
            return true; // cpu streams are always ready
        }

        stream_handle_type default_stream()
        {
            return stream_handle_type{std::this_thread::get_id()};
        }

        // =============================================================================
        // event management (device_event_manager concept)
        // =============================================================================

        event_handle_type create_event()
        {
            std::promise<void> promise;
            auto               future = promise.get_future();
            promise.set_value(); // immediately ready
            return event_handle_type{std::move(future)};
        }

        void destroy_event(event_handle_type)
        {
            // no-op - future handles its own lifetime
        }

        void record_event(event_handle_type, stream_handle_type)
        {
            // no-op - cpu events are immediately ready
        }

        bool is_event_ready(event_handle_type event) const
        {
            (void) event;
            return true; // cpu events are always ready
        }

        void synchronize_event(event_handle_type event)
        {
            // cpu events are immediately ready, nothing to wait for
            (void) event;
        }

        void stream_wait_event(stream_handle_type, event_handle_type event)
        {
            // cpu work is synchronous, nothing to wait for
            (void) event;
        }

        // =============================================================================
        // memory transfer (device_memory_transfer concept)
        // =============================================================================

        void copy_host_to_device(const void* src, memory_handle_type dst, std::size_t bytes)
        {
            if (bytes > 0 && src && dst.ptr) {
                std::memcpy(dst.ptr, src, bytes);
            }
        }

        void copy_device_to_host(memory_handle_type src, void* dst, std::size_t bytes)
        {
            if (bytes > 0 && src.ptr && dst) {
                std::memcpy(dst, src.ptr, bytes);
            }
        }

        void
        copy_device_to_device(memory_handle_type src, memory_handle_type dst, std::size_t bytes)
        {
            if (bytes > 0 && src.ptr && dst.ptr) {
                std::memcpy(dst.ptr, src.ptr, bytes);
            }
        }

        void copy_host_to_device_async(
            const void*        src,
            memory_handle_type dst,
            std::size_t        bytes,
            stream_handle_type
        )
        {
            copy_host_to_device(src, dst, bytes); // cpu copies are synchronous
        }

        void copy_device_to_host_async(
            memory_handle_type src,
            void*              dst,
            std::size_t        bytes,
            stream_handle_type
        )
        {
            copy_device_to_host(src, dst, bytes); // cpu copies are synchronous
        }

        // =============================================================================
        // device properties (device_properties concept)
        // =============================================================================

        std::int64_t device_id() const
        {
            return device_id_;
        }

        std::string_view device_name() const
        {
            return "cpu";
        }

        std::size_t total_memory() const
        {
            // approximate system memory - could query /proc/meminfo on linux
            return SIZE_MAX;
        }

        std::size_t available_memory() const
        {
            // conservative estimate
            return SIZE_MAX;
        }

        double memory_bandwidth_gb_per_sec() const
        {
            // typical ddr4 bandwidth: 25-50 gb/s, use conservative estimate
            return 30.0;
        }

        std::size_t compute_units() const
        {
            return static_cast<std::size_t>(omp_get_max_threads());
        }

        std::size_t max_threads_per_block() const
        {
            return 1; // cpu doesn't have blocks, single thread per "block"
        }

        std::size_t warp_size() const
        {
            return 1; // cpu doesn't have warps
        }

        bool supports_unified_memory() const
        {
            return true; // cpu memory is unified with host
        }

        bool supports_peer_to_peer() const
        {
            return false; // no peer-to-peer for cpu
        }

        bool supports_async_memory_ops() const
        {
            return false; // cpu memory ops are synchronous
        }

      private:
        std::int64_t device_id_;
    };

} // namespace simbi::xpu::vendors::cpu
