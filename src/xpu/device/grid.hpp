// =============================================================================
// grid.hpp
//
// vendor-agnostic grid and thread intrinsics for device code.
// provides unified interface for thread/block/grid indexing across vendors.
//
// design:
//   - compile-time dispatch via platform detection
//   - zero overhead - resolves to native vendor intrinsics
//   - supports 1d, 2d, 3d grid configurations
//   - cpu fallback returns sensible defaults
//
// usage:
//   DEV void kernel() {
//       int tid = xpu::get_thread_id_x();
//       int bid = xpu::get_block_id();
//       int idx = xpu::get_global_thread_id();
//   }
// =============================================================================

#ifndef XPU_DEVICE_GRID_HPP
#define XPU_DEVICE_GRID_HPP

#include "compat.hpp"

#include <cstdint>

namespace simbi::xpu {

    // =============================================================================
    // thread indices
    // =============================================================================

    DEV inline constexpr std::int64_t get_thread_id_x()
    {
#if defined(__CUDA_ARCH__)
        return threadIdx.x;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipThreadIdx_x;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_local_id(0);
#else
        return 0;
#endif
    }

    DEV inline constexpr std::int64_t get_thread_id_y()
    {
#if defined(__CUDA_ARCH__)
        return threadIdx.y;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipThreadIdx_y;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_local_id(1);
#else
        return 0;
#endif
    }

    DEV inline constexpr std::int64_t get_thread_id_z()
    {
#if defined(__CUDA_ARCH__)
        return threadIdx.z;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipThreadIdx_z;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
#else
        return 0;
#endif
    }

    // =============================================================================
    // block indices
    // =============================================================================

    DEV inline constexpr std::int64_t get_block_id_x()
    {
#if defined(__CUDA_ARCH__)
        return blockIdx.x;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipBlockIdx_x;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group(0);
#else
        return 0;
#endif
    }

    DEV inline constexpr std::int64_t get_block_id_y()
    {
#if defined(__CUDA_ARCH__)
        return blockIdx.y;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipBlockIdx_y;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_group(1);
#else
        return 0;
#endif
    }

    DEV inline std::int64_t constexpr get_block_id_z()
    {
#if defined(__CUDA_ARCH__)
        return blockIdx.z;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipBlockIdx_z;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2);
#else
        return 0;
#endif
    }

    // =============================================================================
    // block dimensions
    // =============================================================================

    DEV inline constexpr std::int64_t get_block_dim_x()
    {
#if defined(__CUDA_ARCH__)
        return blockDim.x;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipBlockDim_x;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_local_range(0);
#else
        return 1;
#endif
    }

    DEV inline constexpr std::int64_t get_block_dim_y()
    {
#if defined(__CUDA_ARCH__)
        return blockDim.y;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipBlockDim_y;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_local_range(1);
#else
        return 1;
#endif
    }

    DEV inline constexpr std::int64_t get_block_dim_z()
    {
#if defined(__CUDA_ARCH__)
        return blockDim.z;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipBlockDim_z;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2);
#else
        return 1;
#endif
    }

    // =============================================================================
    // grid dimensions
    // =============================================================================

    DEV inline constexpr std::int64_t get_grid_dim_x()
    {
#if defined(__CUDA_ARCH__)
        return gridDim.x;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipGridDim_x;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group_range(0);
#else
        return 1;
#endif
    }

    DEV inline constexpr std::int64_t get_grid_dim_y()
    {
#if defined(__CUDA_ARCH__)
        return gridDim.y;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipGridDim_y;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<2>().get_group_range(1);
#else
        return 1;
#endif
    }

    DEV inline constexpr std::int64_t get_grid_dim_z()
    {
#if defined(__CUDA_ARCH__)
        return gridDim.z;
#elif defined(__HIP_DEVICE_COMPILE__)
        return hipGridDim_z;
#elif defined(__SYCL_DEVICE_ONLY__)
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_range(2);
#else
        return 1;
#endif
    }

    // =============================================================================
    // composite indices - commonly used patterns
    // =============================================================================

    // linear block id across 3d grid
    DEV inline constexpr std::int64_t get_block_id()
    {
        const std::int64_t bx = get_block_id_x();
        const std::int64_t by = get_block_id_y();
        const std::int64_t bz = get_block_id_z();
        const std::int64_t gx = get_grid_dim_x();
        const std::int64_t gy = get_grid_dim_y();

        return bx + by * gx + bz * gx * gy;
    }

    // linear thread id within block
    DEV inline constexpr std::int64_t get_thread_id()
    {
        const std::int64_t tx = get_thread_id_x();
        const std::int64_t ty = get_thread_id_y();
        const std::int64_t tz = get_thread_id_z();
        const std::int64_t bx = get_block_dim_x();
        const std::int64_t by = get_block_dim_y();

        return tx + ty * bx + tz * bx * by;
    }

    // global thread id across entire grid (1d)
    DEV inline constexpr std::int64_t get_global_thread_id()
    {
        return get_block_id_x() * get_block_dim_x() + get_thread_id_x();
    }

    // global thread id in x dimension
    DEV inline constexpr std::int64_t get_global_thread_id_x()
    {
        return get_block_id_x() * get_block_dim_x() + get_thread_id_x();
    }

    // global thread id in y dimension
    DEV inline constexpr std::int64_t get_global_thread_id_y()
    {
        return get_block_id_y() * get_block_dim_y() + get_thread_id_y();
    }

    // global thread id in z dimension
    DEV inline constexpr std::int64_t get_global_thread_id_z()
    {
        return get_block_id_z() * get_block_dim_z() + get_thread_id_z();
    }

    // =============================================================================
    // synchronization
    // =============================================================================

    DEV inline void sync_threads()
    {
#if defined(__CUDA_ARCH__)
        __syncthreads();
#elif defined(__HIP_DEVICE_COMPILE__)
        __syncthreads();
#elif defined(__SYCL_DEVICE_ONLY__)
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group());
#else
        // cpu: no-op
#endif
    }

    DEV inline void sync_warp()
    {
#if defined(__CUDA_ARCH__)
        __syncwarp();
#elif defined(__HIP_DEVICE_COMPILE__)
        __syncthreads(); // hip doesn't have explicit warp sync
#elif defined(__SYCL_DEVICE_ONLY__)
        // sycl sub-group barrier
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#else
        // cpu: no-op
#endif
    }

} // namespace simbi::xpu

#endif
