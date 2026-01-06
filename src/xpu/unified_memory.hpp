// =============================================================================
// unified_memory.hpp
//
// unified memory space implementation using cuda managed memory.
// provides both host and device accessible memory allocation.
// implements memory_space concept for unified data storage.
//
// usage:
//   auto block = allocate<unified_memory, float>(1000);
//   shared_buffer_t<int, unified_memory> buffer(n);
//   bool accessible = unified_memory::is_accessible_from<host_memory>();
// =============================================================================

#pragma once

#include "memory_space.hpp"

#include <cstring>
#include <string_view>

#ifdef XPU_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace xpu {

    // =============================================================================
    // unified memory space
    // =============================================================================

    struct unified_memory
    {
        // =============================================================================
        // type requirements for memory_space concept
        // =============================================================================

        using pointer_type       = void*;
        using const_pointer_type = const void*;
        using size_type          = std::size_t;

        // =============================================================================
        // compile-time properties
        // =============================================================================

        static constexpr std::string_view space_name()
        {
            return "unified";
        }

        static constexpr std::string_view name() // legacy
        {
            return "unified";
        }

        static constexpr bool        is_device_accessible = true;
        static constexpr bool        is_host_accessible   = true;
        static constexpr bool        is_unified           = true;
        static constexpr std::size_t preferred_alignment  = 256; // gpu alignment for unified memory

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            // unified memory typically limited by pcie bandwidth
            return 16.0; // pcie gen3 x16 theoretical
        }

        // =============================================================================
        // allocation interface
        // =============================================================================

        static void* allocate(std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (size == 0) {
                return nullptr;
            }

            void*       ptr;
            cudaError_t err = cudaMallocManaged(&ptr, size);
            if (err != cudaSuccess) {
                return nullptr; // allocation failed
            }

            // record allocation for debugging
            stats::record_allocation(size);
            return ptr;
#else
            // fallback to host allocation when cuda not available
            return std::aligned_alloc(64, (size + 63) & ~63);
#endif
        }

        static void deallocate(void* ptr, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr) {
                cudaFree(ptr);
                stats::record_deallocation(size);
            }
#else
            if (ptr) {
                std::free(ptr);
            }
#endif
        }

        // =============================================================================
        // accessibility queries
        // =============================================================================

        template <memory_space OtherSpace>
        static constexpr bool is_accessible_from()
        {
            // unified memory is accessible from all spaces
            return true;
        }

        // specialized accessibility queries for known spaces
        static constexpr bool is_accessible_from_host()
        {
            return true;
        }
        static constexpr bool is_accessible_from_device()
        {
            return true;
        }
        static constexpr bool is_accessible_from_unified()
        {
            return true;
        }

        // =============================================================================
        // memory operations
        // =============================================================================

        static void memset(void* ptr, int value, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            // can use either host or device memset for unified memory
            std::memset(ptr, value, size);
#else
            std::memset(ptr, value, size);
#endif
        }

        static void memcpy(void* dest, const void* src, std::size_t size)
        {
            // unified memory can be copied with standard memcpy
            std::memcpy(dest, src, size);
        }

        // =============================================================================
        // memory hints and prefetching
        // =============================================================================

        static void prefetch_to_device(void* ptr, std::size_t size, int device_id = -1)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr == nullptr || size == 0) {
                return;
            }

// use modern cudaMemLocation API for CUDA 12+
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = (device_id >= 0) ? device_id : 0;

            cudaError_t err = cudaMemPrefetchAsync(ptr, size, location, 0);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#else
            // fallback for older cuda versions using device id directly
            int         target_device = (device_id >= 0) ? device_id : 0;
            cudaError_t err           = cudaMemPrefetchAsync(ptr, size, target_device);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#endif
#endif
        }

        static void prefetch_to_host(void* ptr, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr == nullptr || size == 0) {
                return;
            }

// use modern cudaMemLocation API for CUDA 12+
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeHost;
            location.id   = 0;

            cudaError_t err = cudaMemPrefetchAsync(ptr, size, location, 0);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#else
            // fallback for older cuda versions using cudaCpuDeviceId
            cudaError_t err = cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#endif
#endif
        }

        static void advise_read_mostly(void* ptr, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr == nullptr || size == 0) {
                return;
            }

// use modern cudaMemLocation API for CUDA 12+
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = 0; // current device

            cudaError_t err = cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, location);
            if (err != cudaSuccess) {
                // advise failed, continue without error
            }
#else
            // fallback for older cuda versions using device id directly
            cudaError_t err = cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, 0);
            if (err != cudaSuccess) {
                // advise failed, continue without error
            }
#endif
#endif
        }

        static void advise_preferred_location(void* ptr, std::size_t size, int device_id)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr == nullptr || size == 0) {
                return;
            }

// use modern cudaMemLocation API for CUDA 12+
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = (device_id >= 0) ? device_id : 0;

            cudaError_t err = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, location);
            if (err != cudaSuccess) {
                // advise failed, continue without error
            }
#else
            // fallback for older cuda versions using device id directly
            int         target_device = (device_id >= 0) ? device_id : 0;
            cudaError_t err =
                cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, target_device);
            if (err != cudaSuccess) {
                // advise failed, continue without error
            }
#endif
#endif
        }

        // =============================================================================
        // synchronization
        // =============================================================================

        static void synchronize()
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaDeviceSynchronize();
#endif
        }

        static void synchronize_stream(cudaStream_t stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaStreamSynchronize(stream);
#endif
        }

        // =============================================================================
        // async prefetch with stream support
        // =============================================================================

        static void prefetch_to_device_async(
            void*        ptr,
            std::size_t  size,
            cudaStream_t stream,
            int          device_id = -1
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr == nullptr || size == 0) {
                return;
            }

// use modern cudaMemLocation API for CUDA 12+
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = (device_id >= 0) ? device_id : 0;

            cudaError_t err = cudaMemPrefetchAsync(ptr, size, location, 0, stream);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#else
            // fallback for older cuda versions using device id directly
            int         target_device = (device_id >= 0) ? device_id : 0;
            cudaError_t err           = cudaMemPrefetchAsync(ptr, size, target_device, stream);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#endif
#endif
        }

        static void prefetch_to_host_async(void* ptr, std::size_t size, cudaStream_t stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr == nullptr || size == 0) {
                return;
            }

// use modern cudaMemLocation API for CUDA 12+
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeHost;
            location.id   = 0;

            cudaError_t err = cudaMemPrefetchAsync(ptr, size, location, 0, stream);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#else
            // fallback for older cuda versions using cudaCpuDeviceId
            cudaError_t err = cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream);
            if (err != cudaSuccess) {
                // prefetch failed, continue without error
            }
#endif
#endif
        }

        // =============================================================================
        // memory info and queries
        // =============================================================================

        static bool is_managed_pointer(const void* ptr)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaPointerAttributes attrs;
            cudaError_t           err = cudaPointerGetAttributes(&attrs, ptr);
            return err == cudaSuccess && attrs.type == cudaMemoryTypeManaged;
#else
            return true; // assume all pointers are "managed" without cuda
#endif
        }

        static std::size_t get_available_memory()
        {
#ifdef XPU_CUDA_AVAILABLE
            std::size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            return free_mem;
#else
            return SIZE_MAX; // unlimited host memory assumption
#endif
        }

        // =============================================================================
        // performance hints
        // =============================================================================

        struct allocation_hints
        {
            static constexpr bool        supports_concurrent_access = true;
            static constexpr bool        supports_cache_coherency   = true;
            static constexpr bool        requires_explicit_sync     = false;
            static constexpr std::size_t preferred_alignment        = 256; // cuda managed alignment
        };

        // =============================================================================
        // debug and introspection
        // =============================================================================

        static bool is_valid_pointer(const void* ptr)
        {
            if (ptr == nullptr) {
                return false;
            }
#ifdef XPU_CUDA_AVAILABLE
            cudaPointerAttributes attrs;
            cudaError_t           err = cudaPointerGetAttributes(&attrs, ptr);
            return err == cudaSuccess;
#else
            return true;
#endif
        }

        static std::size_t get_alignment()
        {
            return allocation_hints::preferred_alignment;
        }

        // =============================================================================
        // statistics (for debugging and monitoring)
        // =============================================================================

        struct stats
        {
            static std::size_t total_allocated;
            static std::size_t total_deallocated;
            static std::size_t current_usage;

            static void record_allocation(std::size_t size)
            {
                total_allocated += size;
                current_usage += size;
            }

            static void record_deallocation(std::size_t size)
            {
                total_deallocated += size;
                current_usage = (current_usage >= size) ? current_usage - size : 0;
            }

            static void reset()
            {
                total_allocated   = 0;
                total_deallocated = 0;
                current_usage     = 0;
            }
        };
    };

    // =============================================================================
    // static member definitions
    // =============================================================================

    inline std::size_t unified_memory::stats::total_allocated   = 0;
    inline std::size_t unified_memory::stats::total_deallocated = 0;
    inline std::size_t unified_memory::stats::current_usage     = 0;

    // =============================================================================
    // default memory space specialization when cuda available
    // =============================================================================

#ifdef XPU_CUDA_AVAILABLE
    template <>
    struct default_memory_space_selector<true>
    {
        using type = unified_memory;
    };
#endif

    // static assertion to verify concept compliance
    static_assert(memory_space<unified_memory>);

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using unified_block_t = memory_block_t<unified_memory>;

    template <typename T>
    using unified_buffer_t = memory_block_t<unified_memory>;

} // namespace xpu
