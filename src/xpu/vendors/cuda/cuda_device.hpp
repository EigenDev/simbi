// =============================================================================
// cuda_device.hpp
//
// cuda device implementation for heterogeneous xpu abstraction.
// implements hetero_device concept for cuda execution using cuda runtime.
// provides zero-overhead vendor abstraction while maintaining cuda
// performance for massive simulation workloads.
//
// design principles:
//   - concept-driven: satisfies core::hetero_device concept
//   - performance-first: zero overhead abstractions
//   - production-ready: proper error handling and resource management
//   - extensible: easy to add new cuda features
//
// usage:
//   cuda_device_t device{0};  // cuda device 0
//   auto stream = device.create_stream();
//   auto ptr = device.allocate(1024);
// =============================================================================

#pragma once

#include "xpu/core/device_concepts.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifdef XPU_CUDA_AVAILABLE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace simbi::xpu::vendors::cuda {

    // =============================================================================
    // cuda device handles - trivially copyable for concept compliance
    // =============================================================================

    struct cuda_memory_handle_t
    {
        void* ptr = nullptr;

        bool operator==(const cuda_memory_handle_t& other) const noexcept
        {
            return ptr == other.ptr;
        }

        bool operator==(std::nullptr_t) const noexcept
        {
            return ptr == nullptr;
        }

        bool operator!=(const cuda_memory_handle_t& other) const noexcept
        {
            return ptr != other.ptr;
        }

        bool operator!=(std::nullptr_t) const noexcept
        {
            return ptr != nullptr;
        }

        explicit operator bool() const noexcept
        {
            return ptr != nullptr;
        }
    };

    struct cuda_stream_handle_t
    {
#ifdef XPU_CUDA_AVAILABLE
        cudaStream_t stream = nullptr;
#else
        void* stream = nullptr;
#endif

        bool operator==(const cuda_stream_handle_t& other) const noexcept
        {
            return stream == other.stream;
        }

        bool operator==(std::nullptr_t) const noexcept
        {
            return stream == nullptr;
        }

        bool operator!=(const cuda_stream_handle_t& other) const noexcept
        {
            return stream != other.stream;
        }

        bool operator!=(std::nullptr_t) const noexcept
        {
            return stream != nullptr;
        }

        explicit operator bool() const noexcept
        {
            return stream != nullptr;
        }

        cuda_stream_handle_t& operator=(std::nullptr_t) noexcept
        {
            stream = nullptr;
            return *this;
        }
    };

    struct cuda_event_handle_t
    {
#ifdef XPU_CUDA_AVAILABLE
        cudaEvent_t event = nullptr;
#else
        void* event = nullptr;
#endif

        bool operator==(const cuda_event_handle_t& other) const noexcept
        {
            return event == other.event;
        }

        bool operator==(std::nullptr_t) const noexcept
        {
            return event == nullptr;
        }

        bool operator!=(const cuda_event_handle_t& other) const noexcept
        {
            return event != other.event;
        }

        bool operator!=(std::nullptr_t) const noexcept
        {
            return event != nullptr;
        }

        explicit operator bool() const noexcept
        {
            return event != nullptr;
        }
    };

    // cuda kernel handle for device_kernel_executor concept
    struct cuda_kernel_handle_t
    {
#ifdef XPU_CUDA_AVAILABLE
        void*       kernel_ptr = nullptr;
        std::string kernel_name;
#else
        void*       kernel_ptr = nullptr;
        std::string kernel_name;
#endif

        bool operator==(const cuda_kernel_handle_t& other) const noexcept
        {
            return kernel_ptr == other.kernel_ptr;
        }

        bool operator!=(const cuda_kernel_handle_t& other) const noexcept
        {
            return kernel_ptr != other.kernel_ptr;
        }

        explicit operator bool() const noexcept
        {
            return kernel_ptr != nullptr;
        }
    };

    // =============================================================================
    // cuda device implementation
    // =============================================================================

    class cuda_device_t
    {
      public:
        // concept requirements
        using memory_handle_type = cuda_memory_handle_t;
        using stream_handle_type = cuda_stream_handle_t;
        using event_handle_type  = cuda_event_handle_t;
        using kernel_handle_type = cuda_kernel_handle_t;

        // device properties
        static constexpr bool        is_gpu_device       = true;
        static constexpr bool        is_cpu_device       = false;
        static constexpr std::size_t preferred_alignment = 256;

        static constexpr std::string_view vendor_name()
        {
            return "nvidia";
        }

        // construction
        explicit cuda_device_t(std::int64_t device_id = 0) : device_id_(device_id)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaSetDevice(device_id_);
            cudaGetDeviceProperties(&props_, device_id_);
            initialized_ = true;
#endif
        }

        ~cuda_device_t()
        {
#ifdef XPU_CUDA_AVAILABLE
            // cleanup streams and events managed by this device
            for (auto stream : managed_streams_) {
                if (stream.stream) {
                    cudaStreamDestroy(stream.stream);
                }
            }
            for (auto event : managed_events_) {
                if (event.event) {
                    cudaEventDestroy(event.event);
                }
            }
#endif
        }

        // copyable and movable
        cuda_device_t(const cuda_device_t&)            = default;
        cuda_device_t& operator=(const cuda_device_t&) = default;
        cuda_device_t(cuda_device_t&&)                 = default;
        cuda_device_t& operator=(cuda_device_t&&)      = default;

        // =============================================================================
        // device information (device_properties concept)
        // =============================================================================

        std::int64_t device_id() const noexcept
        {
            return device_id_;
        }

        std::string_view device_name() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                device_name_cache_ = std::string(props_.name);
                return device_name_cache_;
            }
#endif
            return "CUDA Device (unavailable)";
        }

        std::size_t total_memory() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                return props_.totalGlobalMem;
            }
#endif
            return 0;
        }

        std::size_t available_memory() const
        {
#ifdef XPU_CUDA_AVAILABLE
            std::size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            return free_bytes;
#else
            return 0;
#endif
        }

        double memory_bandwidth_gb_per_sec() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                // estimate bandwidth based on bus width
                // typical gddr6: ~14 gbps per pin, ddr so *2
                double estimated_clock_ghz = 7.0; // conservative estimate
                double bandwidth           = estimated_clock_ghz * (props_.memoryBusWidth / 8) * 2;
                return bandwidth;
            }
#endif
            return 0.0;
        }

        std::size_t compute_units() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                return props_.multiProcessorCount;
            }
#endif
            return 0;
        }

        std::size_t max_threads_per_block() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                return props_.maxThreadsPerBlock;
            }
#endif
            return 1024;
        }

        std::size_t warp_size() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                return props_.warpSize;
            }
#endif
            return 32;
        }

        bool supports_unified_memory() const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (initialized_) {
                return props_.managedMemory;
            }
#endif
            return false;
        }

        bool supports_peer_to_peer() const
        {
#ifdef XPU_CUDA_AVAILABLE
            // check if any other devices support p2p with this device
            int device_count;
            cudaGetDeviceCount(&device_count);
            for (int i = 0; i < device_count; ++i) {
                if (i != device_id_) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, device_id_, i);
                    if (can_access) {
                        return true;
                    }
                }
            }
#endif
            return false;
        }

        bool supports_async_memory_ops() const
        {
            return true; // cuda supports async memory operations
        }

        // =============================================================================
        // memory allocation (device_memory_allocator concept)
        // =============================================================================

        memory_handle_type allocate(std::size_t bytes)
        {
#ifdef XPU_CUDA_AVAILABLE
            void*       ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                return {};
            }
            return {ptr};
#else
            (void) bytes;
            return {};
#endif
        }

        void deallocate(memory_handle_type handle)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (handle.ptr) {
                cudaFree(handle.ptr);
            }
#else
            (void) handle;
#endif
        }

        memory_handle_type allocate_async(std::size_t bytes, stream_handle_type stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            void*       ptr = nullptr;
            cudaError_t err = cudaMallocAsync(&ptr, bytes, stream.stream);
            if (err != cudaSuccess) {
                return {};
            }
            return {ptr};
#else
            (void) bytes;
            (void) stream;
            return {};
#endif
        }

        void deallocate_async(memory_handle_type handle, stream_handle_type stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (handle.ptr) {
                cudaFreeAsync(handle.ptr, stream.stream);
            }
#else
            (void) handle;
            (void) stream;
#endif
        }

        std::size_t memory_alignment() const noexcept
        {
            return preferred_alignment;
        }

        std::size_t max_allocation_size() const
        {
            return total_memory();
        }

        bool is_accessible_from_host(memory_handle_type handle) const
        {
            (void) handle;
            return false; // device memory not host accessible by default
        }

        // =============================================================================
        // stream management (device_stream_manager concept)
        // =============================================================================

        stream_handle_type create_stream()
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaStream_t stream;
            cudaError_t  err = cudaStreamCreate(&stream);
            if (err != cudaSuccess) {
                return {};
            }
            stream_handle_type handle{stream};
            managed_streams_.push_back(handle);
            return handle;
#else
            return {};
#endif
        }

        void destroy_stream(stream_handle_type stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (stream.stream) {
                cudaStreamDestroy(stream.stream);
                auto it = std::find_if(
                    managed_streams_.begin(),
                    managed_streams_.end(),
                    [stream](const auto& s) { return s.stream == stream.stream; }
                );
                if (it != managed_streams_.end()) {
                    managed_streams_.erase(it);
                }
            }
#else
            (void) stream;
#endif
        }

        void synchronize_stream(stream_handle_type stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (stream.stream) {
                cudaStreamSynchronize(stream.stream);
            }
#else
            (void) stream;
#endif
        }

        bool is_stream_ready(stream_handle_type stream) const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (stream.stream) {
                cudaError_t err = cudaStreamQuery(stream.stream);
                return err == cudaSuccess;
            }
#else
            (void) stream;
#endif
            return true;
        }

        stream_handle_type default_stream() const
        {
#ifdef XPU_CUDA_AVAILABLE
            return {cudaStreamDefault};
#else
            return {};
#endif
        }

        // =============================================================================
        // event management (device_event_manager concept)
        // =============================================================================

        event_handle_type create_event()
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaEvent_t event;
            cudaError_t err = cudaEventCreate(&event);
            if (err != cudaSuccess) {
                return {};
            }
            event_handle_type handle{event};
            managed_events_.push_back(handle);
            return handle;
#else
            return {};
#endif
        }

        void destroy_event(event_handle_type event)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (event.event) {
                cudaEventDestroy(event.event);
                auto it = std::find_if(
                    managed_events_.begin(),
                    managed_events_.end(),
                    [event](const auto& e) { return e.event == event.event; }
                );
                if (it != managed_events_.end()) {
                    managed_events_.erase(it);
                }
            }
#else
            (void) event;
#endif
        }

        void record_event(event_handle_type event, stream_handle_type stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (event.event && stream.stream) {
                cudaEventRecord(event.event, stream.stream);
            }
#else
            (void) event;
            (void) stream;
#endif
        }

        bool is_event_ready(event_handle_type event) const
        {
#ifdef XPU_CUDA_AVAILABLE
            if (event.event) {
                cudaError_t err = cudaEventQuery(event.event);
                return err == cudaSuccess;
            }
#else
            (void) event;
#endif
            return true;
        }

        void synchronize_event(event_handle_type event)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (event.event) {
                cudaEventSynchronize(event.event);
            }
#else
            (void) event;
#endif
        }

        void stream_wait_event(stream_handle_type stream, event_handle_type event)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (stream.stream && event.event) {
                cudaStreamWaitEvent(stream.stream, event.event, 0);
            }
#else
            (void) stream;
            (void) event;
#endif
        }

        // =============================================================================
        // memory transfer (device_memory_transfer concept)
        // =============================================================================

        void copy_host_to_device(const void* src, memory_handle_type dst, std::size_t bytes)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpy(dst.ptr, src, bytes, cudaMemcpyHostToDevice);
#else
            (void) src;
            (void) dst;
            (void) bytes;
#endif
        }

        void copy_device_to_host(memory_handle_type src, void* dst, std::size_t bytes)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpy(dst, src.ptr, bytes, cudaMemcpyDeviceToHost);
#else
            (void) src;
            (void) dst;
            (void) bytes;
#endif
        }

        void
        copy_device_to_device(memory_handle_type src, memory_handle_type dst, std::size_t bytes)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpy(dst.ptr, src.ptr, bytes, cudaMemcpyDeviceToDevice);
#else
            (void) src;
            (void) dst;
            (void) bytes;
#endif
        }

        void copy_host_to_device_async(
            const void*        src,
            memory_handle_type dst,
            std::size_t        bytes,
            stream_handle_type stream
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice, stream.stream);
#else
            (void) src;
            (void) dst;
            (void) bytes;
            (void) stream;
#endif
        }

        void copy_device_to_host_async(
            memory_handle_type src,
            void*              dst,
            std::size_t        bytes,
            stream_handle_type stream
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpyAsync(dst, src.ptr, bytes, cudaMemcpyDeviceToHost, stream.stream);
#else
            (void) src;
            (void) dst;
            (void) bytes;
            (void) stream;
#endif
        }

        // =============================================================================
        // kernel execution (device_kernel_executor concept)
        // =============================================================================

        template <typename... Args>
        event_handle_type launch_kernel(
            kernel_handle_type kernel,
            std::size_t        grid_size,
            std::size_t        block_size,
            stream_handle_type stream,
            Args&&... args
        )
        {
            // placeholder - actual kernel launching would require nvcc compilation
            (void) kernel;
            (void) grid_size;
            (void) block_size;
            (void) stream;
            ((void) args, ...);

            auto event = create_event();
            record_event(event, stream);
            return event;
        }

      private:
        int                 device_id_;
        bool                initialized_ = false;
        mutable std::string device_name_cache_;

        // resource management
        std::vector<stream_handle_type> managed_streams_;
        std::vector<event_handle_type>  managed_events_;

#ifdef XPU_CUDA_AVAILABLE
        cudaDeviceProp props_;
#endif
    };

    // =============================================================================
    // concept verification
    // =============================================================================

    // verify that cuda_device_t satisfies hetero_device concept
    static_assert(core::hetero_device<cuda_device_t>);
    static_assert(core::async_memory_allocator<cuda_device_t>);

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using nvidia_device_t = cuda_device_t;

} // namespace simbi::xpu::vendors::cuda
