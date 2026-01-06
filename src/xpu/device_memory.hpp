// =============================================================================
// device_memory.hpp
//
// device memory space implementation using cuda memory management.
// provides gpu-accessible memory allocation and management.
// implements memory_space concept for device-side data storage.
//
// usage:
//   auto block = allocate<device_memory, float>(1000);
//   shared_buffer_t<int, device_memory> buffer(n);
//   bool accessible = device_memory::is_accessible_from<host_memory>();
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
    // device memory space
    // =============================================================================

    struct device_memory
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
            return "device";
        }

        static constexpr std::string_view name() // legacy
        {
            return "device";
        }

        static constexpr bool        is_device_accessible = true;
        static constexpr bool        is_host_accessible   = false;
        static constexpr bool        is_unified           = false;
        static constexpr std::size_t preferred_alignment  = 256; // gpu memory alignment

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            // conservative estimate for gpu memory bandwidth
            return 300.0; // typical for mid-range gpus
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
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                return nullptr; // allocation failed
            }

            // record allocation for debugging
            stats::record_allocation(size);
            return ptr;
#else
            static_assert(sizeof(size) == 0, "device_memory requires CUDA compilation");
#endif
        }

        static void deallocate(void* ptr, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (ptr) {
                cudaFree(ptr);
                stats::record_deallocation(size);
            }
#endif
        }

        // =============================================================================
        // accessibility queries
        // =============================================================================

        template <memory_space OtherSpace>
        static constexpr bool is_accessible_from()
        {
            // device memory is accessible from device and unified spaces only
            return OtherSpace::is_device_accessible;
        }

        // specialized accessibility queries for known spaces
        static constexpr bool is_accessible_from_host()
        {
            return false;
        }
        static constexpr bool is_accessible_from_device()
        {
            return true;
        }
        static constexpr bool is_accessible_from_unified()
        {
            return true; // unified memory can access device
        }

        // =============================================================================
        // memory operations
        // =============================================================================

        static void memset(void* ptr, int value, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemset(ptr, value, size);
#endif
        }

        static void memcpy_from_host(void* dest, const void* src, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
#endif
        }

        static void memcpy_to_host(void* dest, const void* src, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
#endif
        }

        static void memcpy_device_to_device(void* dest, const void* src, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
#endif
        }

        // =============================================================================
        // peer-to-peer memory operations (multi-gpu)
        // =============================================================================

        // enable peer access between two devices
        static bool enable_peer_access(int src_device, int dst_device)
        {
#ifdef XPU_CUDA_AVAILABLE
            int         can_access = 0;
            cudaError_t err        = cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
            if (err != cudaSuccess || !can_access) {
                return false;
            }

            // enable peer access (ignore error if already enabled)
            err = cudaDeviceEnablePeerAccess(src_device, 0);
            return err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled;
#else
            return false;
#endif
        }

        // disable peer access between two devices
        static bool disable_peer_access(int src_device)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaError_t err = cudaDeviceDisablePeerAccess(src_device);
            return err == cudaSuccess;
#else
            return false;
#endif
        }

        // check if peer access is possible between devices
        static bool can_access_peer(int src_device, int dst_device)
        {
#ifdef XPU_CUDA_AVAILABLE
            int         can_access = 0;
            cudaError_t err        = cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
            return err == cudaSuccess && can_access;
#else
            return false;
#endif
        }

        // peer-to-peer copy (device-to-device across different gpus)
        static bool
        memcpy_peer(void* dst, int dst_device, const void* src, int src_device, std::size_t size)
        {
#ifdef XPU_CUDA_AVAILABLE
            if (size == 0) {
                return true;
            }

            // attempt to enable peer access if not already enabled
            enable_peer_access(src_device, dst_device);

            // perform peer copy
            cudaError_t err = cudaMemcpyPeer(dst, dst_device, src, src_device, size);

            return err == cudaSuccess;
#else
            return false;
#endif
        }

        // async peer-to-peer copy
        static bool memcpy_peer_async(
            void*        dst,
            int          dst_device,
            const void*  src,
            int          src_device,
            std::size_t  size,
            cudaStream_t stream
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            if (size == 0) {
                return true;
            }

            // attempt to enable peer access if not already enabled
            enable_peer_access(src_device, dst_device);

            // perform async peer copy
            cudaError_t err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, size, stream);

            return err == cudaSuccess;
#else
            return false;
#endif
        }

        // =============================================================================
        // async memory operations
        // =============================================================================

        static void memcpy_from_host_async(
            void*        dest,
            const void*  src,
            std::size_t  size,
            cudaStream_t stream = 0
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, stream);
#endif
        }

        static void
        memcpy_to_host_async(void* dest, const void* src, std::size_t size, cudaStream_t stream = 0)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, stream);
#endif
        }

        static void memcpy_device_to_device_async(
            void*        dest,
            const void*  src,
            std::size_t  size,
            cudaStream_t stream = 0
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice, stream);
#endif
        }

        // =============================================================================
        // device management
        // =============================================================================

        static int get_current_device()
        {
#ifdef XPU_CUDA_AVAILABLE
            int device;
            cudaGetDevice(&device);
            return device;
#else
            return -1;
#endif
        }

        static bool set_device(int device_id)
        {
#ifdef XPU_CUDA_AVAILABLE
            return cudaSetDevice(device_id) == cudaSuccess;
#else
            return false;
#endif
        }

        static std::size_t get_free_memory()
        {
#ifdef XPU_CUDA_AVAILABLE
            std::size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            return free_mem;
#else
            return 0;
#endif
        }

        static std::size_t get_total_memory()
        {
#ifdef XPU_CUDA_AVAILABLE
            std::size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            return total_mem;
#else
            return 0;
#endif
        }

        // =============================================================================
        // performance hints
        // =============================================================================

        struct allocation_hints
        {
            static constexpr bool        supports_concurrent_access = true;
            static constexpr bool        supports_cache_coherency   = false;
            static constexpr bool        requires_explicit_sync     = true;
            static constexpr std::size_t preferred_alignment        = 256; // gpu alignment
        };

        // =============================================================================
        // debug and introspection
        // =============================================================================

        static bool is_valid_pointer(const void* ptr)
        {
#ifdef XPU_CUDA_AVAILABLE
            cudaPointerAttributes attrs;
            cudaError_t           err = cudaPointerGetAttributes(&attrs, ptr);
            return err == cudaSuccess && attrs.type == cudaMemoryTypeDevice;
#else
            return false;
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

    inline std::size_t device_memory::stats::total_allocated   = 0;
    inline std::size_t device_memory::stats::total_deallocated = 0;
    inline std::size_t device_memory::stats::current_usage     = 0;

    // static assertion to verify concept compliance
    static_assert(memory_space<device_memory>);

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using device_block_t = memory_block_t<device_memory>;

    template <typename T>
    using device_buffer_t = memory_block_t<device_memory>;

} // namespace xpu
