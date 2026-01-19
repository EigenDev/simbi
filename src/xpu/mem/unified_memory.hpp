// =============================================================================
// unified_memory.hpp
//
// memory space implementation for cuda/hip unified memory.
// defines `unified_memory_t`, which implements the `memory_space` concept for
// unified memory, accessible from both host and device. it uses
// `cudaMallocManaged` or a host-side fallback. provides memory advising and
// prefetching capabilities.
//
// usage:
//   using unified_block = block_t<unified_memory_t>;
//   unified_block my_block(1024);
// =============================================================================
#pragma once

#include "memory_space.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#ifdef XPU_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace simbi::xpu::mem {

    struct unified_memory_t
    {

        using pointer_type       = void*;
        using const_pointer_type = const void*;
        using size_type          = std::size_t;

        static constexpr std::string_view space_name()
        {
            return "unified";
        }

        static constexpr std::string_view name()
        {
            return space_name();
        }

        static constexpr bool        is_host_accessible   = true;
        static constexpr bool        is_device_accessible = true;
        static constexpr bool        is_unified           = true;
        static constexpr bool        requires_staging     = false;
        static constexpr std::size_t preferred_alignment  = 64;

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            return 16.0; // conservative estimate for pcie gen3
        }

        static void* allocate(std::size_t size)
        {
            if (size == 0) {
                return nullptr;
            }

#ifdef XPU_CUDA_AVAILABLE
            void*       ptr;
            cudaError_t err = cudaMallocManaged(&ptr, size);
            if (err != cudaSuccess) {
                return nullptr;
            }
            // cudaMallocManaged already zero-initializes
            stats::record_allocation(size);
            return ptr;
#else
            // cpu fallback: aligned host allocation with zero-init
            std::size_t aligned_size = (size + 63) & ~63;
            void*       ptr          = std::aligned_alloc(64, aligned_size);
            if (ptr) {
                // zero-initialize to match cudaMallocManaged behavior
                std::memset(ptr, 0, aligned_size);
                stats::record_allocation(size);
            }
            return ptr;
#endif
        }

        static void deallocate(void* ptr, std::size_t size)
        {
            if (!ptr) {
                return;
            }

#ifdef XPU_CUDA_AVAILABLE
            cudaFree(ptr);
#else
            std::free(ptr);
#endif
            stats::record_deallocation(size);
        }

        static void memset(void* ptr, int value, std::size_t size)
        {
            std::memset(ptr, value, size);
        }

        static void memcpy(void* dest, const void* src, std::size_t size)
        {
            std::memcpy(dest, src, size);
        }

        template <typename MemorySpace>
        static constexpr bool is_accessible_from()
        {
            return true; // unified memory accessible from all spaces
        }

#ifdef XPU_CUDA_AVAILABLE
        static void prefetch_to_device(void* ptr, std::size_t size, std::int64_t device_id = -1)
        {
            if (!ptr || size == 0) {
                return;
            }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = (device_id >= 0) ? device_id : 0;
            cudaMemPrefetchAsync(ptr, size, location, 0, nullptr);
#else
            int target_device = (device_id >= 0) ? device_id : 0;
            cudaMemPrefetchAsync(ptr, size, target_device, nullptr);
#endif
        }

        static void prefetch_to_host(void* ptr, std::size_t size)
        {
            if (!ptr || size == 0) {
                return;
            }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeHost;
            location.id   = 0;
            cudaMemPrefetchAsync(ptr, size, location, 0, nullptr);
#else
            cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, nullptr);
#endif
        }

        static void advise_read_mostly(void* ptr, std::size_t size, std::int64_t device_id = 0)
        {
            if (!ptr || size == 0) {
                return;
            }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = device_id;
            cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, location);
#else
            cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device_id);
#endif
        }

        static void advise_preferred_location(void* ptr, std::size_t size, std::int64_t device_id)
        {
            if (!ptr || size == 0) {
                return;
            }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id   = device_id;
            cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, location);
#else
            cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
#endif
        }

        static void synchronize_stream(cudaStream_t stream)
        {
            cudaStreamSynchronize(stream);
        }
#endif // XPU_CUDA_AVAILABLE

        static void synchronize()
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaDeviceSynchronize();
#endif
            // no-op for cpu-only
        }

        static bool is_managed_pointer(const void* ptr)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaPointerAttributes attrs;
            cudaError_t           err = cudaPointerGetAttributes(&attrs, ptr);
            return err == cudaSuccess && attrs.type == cudaMemoryTypeManaged;
#else
            (void) ptr;
            return true; // all host pointers are "managed" in cpu-only mode
#endif
        }

        static std::size_t get_available_memory()
        {
#ifdef XPU_CUDA_AVAILABLE
            std::size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            return free_mem;
#else
            return std::numeric_limits<std::int64_t>::max(); // no practical limit for host memory
#endif
        }

        static bool is_valid_pointer(const void* ptr)
        {
            if (!ptr) {
                return false;
            }

#ifdef XPU_CUDA_AVAILABLE
            cudaPointerAttributes attrs;
            cudaError_t           err = cudaPointerGetAttributes(&attrs, ptr);
            return err == cudaSuccess;
#else
            return true; // assume all non-null host pointers are valid
#endif
        }

        static std::size_t get_alignment()
        {
            return 64; // cache line alignment
        }

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

    inline std::size_t unified_memory_t::stats::total_allocated   = 0;
    inline std::size_t unified_memory_t::stats::total_deallocated = 0;
    inline std::size_t unified_memory_t::stats::current_usage     = 0;

    template <bool has_cuda>
    struct default_memory_space_selector
    {
        using type = host_memory;
    };

#ifdef XPU_CUDA_AVAILABLE
    template <>
    struct default_memory_space_selector<true>
    {
        using type = unified_memory_t;
    };
#endif

    static_assert(memory_space_c<unified_memory_t>);

    using unified_block_t = block_t<unified_memory_t>;

    template <typename T>
    using unified_buffer_t = block_t<unified_memory_t>;

} // namespace simbi::xpu::mem
