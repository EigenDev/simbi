// =============================================================================
// device_queries.hpp
//
// cuda device query api for xpu framework.
// compatible with cuda 12+ using cudaDeviceGetAttribute instead of deprecated
// cudaGetDeviceProperties where appropriate.
//
// design principles:
//   - use modern cuda 12+ api (cudaDeviceGetAttribute)
//   - fallback to cudaGetDeviceProperties for properties not in attributes
//   - simple c-style api (no exceptions, no complex abstractions)
//   - header-only for ease of use
//
// usage:
//   int count = xpu::vendors::cuda::get_device_count();
//   auto props = xpu::vendors::cuda::get_properties(0);
// =============================================================================

#ifndef XPU_VENDORS_CUDA_DEVICE_QUERIES_HPP
#define XPU_VENDORS_CUDA_DEVICE_QUERIES_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef XPU_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace simbi::xpu::vendors::cuda {

    // =========================================================================
    // device properties structure
    // =========================================================================
    struct device_properties_t
    {
        std::string  name;
        std::size_t  total_memory;
        std::size_t  shared_memory_per_block;
        std::int32_t max_threads_per_block;
        std::int32_t max_threads_per_sm;
        std::int32_t multiprocessor_count;
        std::int32_t warp_size;
        std::int32_t max_grid_size[3];
        std::int32_t max_block_dims[3];
        std::int32_t compute_capability_major;
        std::int32_t compute_capability_minor;
        std::int32_t clock_rate_khz;
        std::int32_t memory_clock_rate_khz;
        std::int32_t memory_bus_width_bits;
        std::int32_t l2_cache_size;
        bool         unified_addressing;
        bool         concurrent_kernels;
        bool         ecc_enabled;
    };

    // =========================================================================
    // device count query
    // =========================================================================
    inline int get_device_count()
    {
#ifdef XPU_CUDA_AVAILABLE
        int         count = 0;
        cudaError_t err   = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            return 0;
        }
        return count;
#else
        return 0;
#endif
    }

    // =========================================================================
    // current device query
    // =========================================================================
    inline int get_current_device()
    {
#ifdef XPU_CUDA_AVAILABLE
        int         device = 0;
        cudaError_t err    = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            return 0;
        }
        return device;
#else
        return 0;
#endif
    }

    // =========================================================================
    // set active device
    // =========================================================================
    inline bool set_device(std::int64_t device_id)
    {
#ifdef XPU_CUDA_AVAILABLE
        cudaError_t err = cudaSetDevice(device_id);
        return (err == cudaSuccess);
#else
        (void) device_id;
        return false;
#endif
    }

    // =========================================================================
    // device properties query (cuda 12+ compatible)
    // =========================================================================
    inline device_properties_t get_properties(std::int64_t device_id)
    {
        device_properties_t props{};

#ifdef XPU_CUDA_AVAILABLE
        // use cudaGetDeviceProperties as primary source (still supported in cuda 12+)
        // for properties that don't have attribute equivalents
        cudaDeviceProp cuda_props;
        cudaError_t    err = cudaGetDeviceProperties(&cuda_props, static_cast<int>(device_id));
        if (err != cudaSuccess) {
            return props;
        }

        // basic properties from cudaDeviceProp
        props.name                     = cuda_props.name;
        props.total_memory             = cuda_props.totalGlobalMem;
        props.shared_memory_per_block  = cuda_props.sharedMemPerBlock;
        props.max_threads_per_block    = cuda_props.maxThreadsPerBlock;
        props.max_threads_per_sm       = cuda_props.maxThreadsPerMultiProcessor;
        props.multiprocessor_count     = cuda_props.multiProcessorCount;
        props.warp_size                = cuda_props.warpSize;
        props.max_grid_size[0]         = cuda_props.maxGridSize[0];
        props.max_grid_size[1]         = cuda_props.maxGridSize[1];
        props.max_grid_size[2]         = cuda_props.maxGridSize[2];
        props.max_block_dims[0]        = cuda_props.maxThreadsDim[0];
        props.max_block_dims[1]        = cuda_props.maxThreadsDim[1];
        props.max_block_dims[2]        = cuda_props.maxThreadsDim[2];
        props.compute_capability_major = cuda_props.major;
        props.compute_capability_minor = cuda_props.minor;
        props.memory_bus_width_bits    = cuda_props.memoryBusWidth;
        props.l2_cache_size            = cuda_props.l2CacheSize;
        props.unified_addressing       = cuda_props.unifiedAddressing;
        props.concurrent_kernels       = cuda_props.concurrentKernels;
        props.ecc_enabled              = cuda_props.ECCEnabled;

        // use cudaDeviceGetAttribute for attributes (cuda 12+ preferred method)
        int clock_rate = 0;
        if (cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device_id) == cudaSuccess) {
            props.clock_rate_khz = clock_rate;
        }

        int mem_clock_rate = 0;
        if (cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, device_id) ==
            cudaSuccess) {
            props.memory_clock_rate_khz = mem_clock_rate;
        }
#endif
        (void) device_id;
        return props;
    }

    // =========================================================================
    // memory info query
    // =========================================================================
    struct memory_info_t
    {
        std::size_t free_bytes;
        std::size_t total_bytes;
    };

    inline memory_info_t get_memory_info(std::int64_t device_id)
    {
        memory_info_t info{};

#ifdef XPU_CUDA_AVAILABLE
        // save current device
        int current_device = 0;
        cudaGetDevice(&current_device);

        // switch to target device
        if (cudaSetDevice(device_id) == cudaSuccess) {
            cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
            // restore original device
            cudaSetDevice(current_device);
        }
#else
        (void) device_id;
#endif

        return info;
    }

    // =========================================================================
    // device synchronization
    // =========================================================================
    inline bool synchronize(std::int64_t device_id)
    {
#ifdef XPU_CUDA_AVAILABLE
        // save current device
        int current_device = 0;
        cudaGetDevice(&current_device);

        // switch to target device
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return false;
        }

        // synchronize
        err = cudaDeviceSynchronize();

        // restore original device
        cudaSetDevice(current_device);

        return (err == cudaSuccess);
#else
        (void) device_id;
        return false;
#endif
    }

    // =========================================================================
    // peer-to-peer access query
    // =========================================================================
    inline bool can_access_peer(std::int64_t device_id, int peer_device_id)
    {
#ifdef XPU_CUDA_AVAILABLE
        int         can_access = 0;
        cudaError_t err        = cudaDeviceCanAccessPeer(&can_access, device_id, peer_device_id);
        return (err == cudaSuccess && can_access != 0);
#else
        (void) device_id;
        (void) peer_device_id;
        return false;
#endif
    }

    // =========================================================================
    // enable peer-to-peer access
    // =========================================================================
    inline bool enable_peer_access(std::int64_t device_id, int peer_device_id)
    {
#ifdef XPU_CUDA_AVAILABLE
        // save current device
        int current_device = 0;
        cudaGetDevice(&current_device);

        // switch to device_id
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return false;
        }

        // enable access to peer
        err = cudaDeviceEnablePeerAccess(peer_device_id, 0);

        // restore original device
        cudaSetDevice(current_device);

        // cudaErrorPeerAccessAlreadyEnabled is not a failure
        return (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
#else
        (void) device_id;
        (void) peer_device_id;
        return false;
#endif
    }

} // namespace simbi::xpu::vendors::cuda

#endif // XPU_VENDORS_CUDA_DEVICE_QUERIES_HPP
