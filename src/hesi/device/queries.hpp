#ifndef HET_DEVICE_QUERIES_HPP
#define HET_DEVICE_QUERIES_HPP

#include "compat.hpp"
#include "hesi/core/error_handling.hpp"
#include "hesi/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace simbi::het::device {

    // device properties - unified across backends
    struct properties_t {
        std::string name;
        std::size_t total_memory;   // bytes
        std::size_t shared_memory_per_block;
        std::int32_t max_threads_per_block;
        std::int32_t max_threads_per_sm;   // streaming multiprocessor
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
        bool unified_addressing;   // UVM support
        bool concurrent_kernels;
        bool ecc_enabled;
    };

    // get number of devices for a backend
    inline std::int64_t get_device_count(backend_type_t backend)
    {
        if (backend == backend_type_t::cpu) {
            return 1;   // CPU is always device 0
        }

#if defined(CUDA_ENABLED)
        if (backend == backend_type_t::cuda) {
            int count = 0;
            check_error<cuda_backend_t>(
                cudaGetDeviceCount(&count),
                "get device count"
            );
            return static_cast<std::int64_t>(count);
        }
#elif defined(HIP_ENABLED)
        if (backend == backend_type_t::hip) {
            int count = 0;
            check_error<hip_backend_t>(
                hipGetDeviceCount(&count),
                "get device count"
            );
            return static_cast<std::int64_t>(count);
        }
#endif

        return 0;
    }

    // get current device ID
    inline std::int64_t get_current_device(backend_type_t backend)
    {
        if (backend == backend_type_t::cpu) {
            return 0;
        }

#if defined(CUDA_ENABLED)
        if (backend == backend_type_t::cuda) {
            int device = 0;
            check_error<cuda_backend_t>(
                cudaGetDevice(&device),
                "get current device"
            );
            return static_cast<std::int64_t>(device);
        }
#elif defined(HIP_ENABLED)
        if (backend == backend_type_t::hip) {
            int device = 0;
            check_error<hip_backend_t>(
                hipGetDevice(&device),
                "get current device"
            );
            return static_cast<std::int64_t>(device);
        }
#endif

        return 0;
    }

    // get active device
    inline void set_device(locality_t loc)
    {
        if (loc.backend == backend_type_t::cpu) {
            return;
        }

#if defined(CUDA_ENABLED)
        if (loc.backend == backend_type_t::cuda) {
            check_error<cuda_backend_t>(
                cudaSetDevice(static_cast<int>(loc.device_id)),
                "set device"
            );
        }
#elif defined(HIP_ENABLED)
        if (loc.backend == backend_type_t::hip) {
            check_error<hip_backend_t>(
                hipSetDevice(static_cast<int>(loc.device_id)),
                "set device"
            );
        }
#endif
    }

    // get device properties
    inline properties_t get_properties(locality_t loc)
    {
        properties_t props{};

        if (loc.backend == backend_type_t::cpu) {
            props.name                     = "CPU";
            props.total_memory             = 0;   // query system RAM if needed
            props.max_threads_per_block    = 1;
            props.warp_size                = 1;
            props.multiprocessor_count     = 1;
            props.compute_capability_major = 0;
            props.compute_capability_minor = 0;
            props.unified_addressing       = true;
            props.concurrent_kernels       = false;
            return props;
        }

#if defined(CUDA_ENABLED)
        if (loc.backend == backend_type_t::cuda) {
            cudaDeviceProp cuda_props;
            check_error<cuda_backend_t>(
                cudaGetDeviceProperties(
                    &cuda_props,
                    static_cast<int>(loc.device_id)
                ),
                "get device properties"
            );
            int clockRateKHz = 0;
            cudaDeviceGetAttribute(
                &clockRateKHz,
                cudaDevAttrClockRate,
                static_cast<int>(loc.device_id)
            );

            props.name                    = cuda_props.name;
            props.total_memory            = cuda_props.totalGlobalMem;
            props.shared_memory_per_block = cuda_props.sharedMemPerBlock;
            props.max_threads_per_block   = cuda_props.maxThreadsPerBlock;
            props.max_threads_per_sm   = cuda_props.maxThreadsPerMultiProcessor;
            props.multiprocessor_count = cuda_props.multiProcessorCount;
            props.warp_size            = cuda_props.warpSize;
            props.max_grid_size[0]     = cuda_props.maxGridSize[0];
            props.max_grid_size[1]     = cuda_props.maxGridSize[1];
            props.max_grid_size[2]     = cuda_props.maxGridSize[2];
            props.max_block_dims[0]    = cuda_props.maxThreadsDim[0];
            props.max_block_dims[1]    = cuda_props.maxThreadsDim[1];
            props.max_block_dims[2]    = cuda_props.maxThreadsDim[2];
            props.compute_capability_major = cuda_props.major;
            props.compute_capability_minor = cuda_props.minor;
            props.clock_rate_khz           = clockRateKHz;
            props.memory_clock_rate_khz    = clockRateKHz;
            props.memory_bus_width_bits    = cuda_props.memoryBusWidth;
            props.l2_cache_size            = cuda_props.l2CacheSize;
            props.unified_addressing       = cuda_props.unifiedAddressing;
            props.concurrent_kernels       = cuda_props.concurrentKernels;
            props.ecc_enabled              = cuda_props.ECCEnabled;

            return props;
        }
#elif defined(HIP_ENABLED)
        if (loc.backend == backend_type_t::hip) {
            hipDeviceProp_t hip_props;
            check_error<hip_backend_t>(
                hipGetDeviceProperties(
                    &hip_props,
                    static_cast<int>(loc.device_id)
                ),
                "get device properties"
            );

            props.name                    = hip_props.name;
            props.total_memory            = hip_props.totalGlobalMem;
            props.shared_memory_per_block = hip_props.sharedMemPerBlock;
            props.max_threads_per_block   = hip_props.maxThreadsPerBlock;
            props.max_threads_per_sm   = hip_props.maxThreadsPerMultiProcessor;
            props.multiprocessor_count = hip_props.multiProcessorCount;
            props.warp_size            = hip_props.warpSize;
            props.max_grid_size[0]     = hip_props.maxGridSize[0];
            props.max_grid_size[1]     = hip_props.maxGridSize[1];
            props.max_grid_size[2]     = hip_props.maxGridSize[2];
            props.max_block_dims[0]    = hip_props.maxThreadsDim[0];
            props.max_block_dims[1]    = hip_props.maxThreadsDim[1];
            props.max_block_dims[2]    = hip_props.maxThreadsDim[2];
            props.compute_capability_major = hip_props.major;
            props.compute_capability_minor = hip_props.minor;
            props.clock_rate_khz           = hip_props.clockRate;
            props.memory_clock_rate_khz    = hip_props.memoryClockRate;
            props.memory_bus_width_bits    = hip_props.memoryBusWidth;
            props.l2_cache_size            = hip_props.l2CacheSize;
            props.unified_addressing       = hip_props.unifiedAddressing;
            props.concurrent_kernels       = hip_props.concurrentKernels;
            props.ecc_enabled              = hip_props.ECCEnabled;

            return props;
        }
#endif

        return props;
    }

    // get available memory on device (current free/total)
    struct memory_info_t {
        std::size_t free_bytes;
        std::size_t total_bytes;
    };

    inline memory_info_t get_memory_info(locality_t loc)
    {
        memory_info_t info{};

        if (loc.backend == backend_type_t::cpu) {
            // could query system memory here if needed
            return info;
        }

#if defined(CUDA_ENABLED)
        if (loc.backend == backend_type_t::cuda) {
            check_error<cuda_backend_t>(
                cudaSetDevice(static_cast<int>(loc.device_id)),
                "set device for memory query"
            );
            check_error<cuda_backend_t>(
                cudaMemGetInfo(&info.free_bytes, &info.total_bytes),
                "get memory info"
            );
            return info;
        }
#elif defined(HIP_ENABLED)
        if (loc.backend == backend_type_t::hip) {
            check_error<hip_backend_t>(
                hipSetDevice(static_cast<int>(loc.device_id)),
                "set device for memory query"
            );
            check_error<hip_backend_t>(
                hipMemGetInfo(&info.free_bytes, &info.total_bytes),
                "get memory info"
            );
            return info;
        }
#endif

        return info;
    }

    // syncrhonize device (wait for all operations)
    inline void synchronize(locality_t loc)
    {
        if (loc.backend == backend_type_t::cpu) {
            return;
        }

#if defined(CUDA_ENABLED)
        if (loc.backend == backend_type_t::cuda) {
            check_error<cuda_backend_t>(
                cudaSetDevice(static_cast<int>(loc.device_id)),
                "set device for synchronize"
            );
            check_error<cuda_backend_t>(
                cudaDeviceSynchronize(),
                "device synchronize"
            );
        }
#elif defined(HIP_ENABLED)
        if (loc.backend == backend_type_t::hip) {
            check_error<hip_backend_t>(
                hipSetDevice(static_cast<int>(loc.device_id)),
                "set device for synchronize"
            );
            check_error<hip_backend_t>(
                hipDeviceSynchronize(),
                "device synchronize"
            );
        }
#endif
    }

    // calc optimal block size for a kernel
    // retuns: (grid_size, block_size)
    inline dim3_t calculate_occupancy(
        std::size_t total_threads,
        std::size_t shared_mem_per_block,
        locality_t loc
    )
    {
        auto props = get_properties(loc);

        // Start with maximum threads per block
        std::int32_t block_size = props.max_threads_per_block;

        // Clamp to warp-aligned values
        if (block_size > 1024) {
            block_size = 1024;
        }
        if (block_size > 512) {
            block_size = 512;
        }
        if (block_size > 256) {
            block_size = 256;
        }

        // adjust for shared memory constraints
        if (shared_mem_per_block > 0) {
            std::int32_t max_blocks_by_smem =
                props.shared_memory_per_block / shared_mem_per_block;
            if (max_blocks_by_smem < 1) {
                // shared memory requirement too large
                block_size = 128;   // fallback
            }
        }

        // calc grid size
        std::uint64_t grid_size = (total_threads + block_size - 1) / block_size;

        return dim3_t{grid_size, 1, 1};
    }

}   // namespace simbi::het::device

#endif   // HETERO_DEVICE_QUERIES_HPP
