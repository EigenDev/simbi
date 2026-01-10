// =============================================================================
// cuda_dispatch.hpp
//
// cuda execution space dispatch implementation for parallel domain iteration.
// header-only implementation to avoid linking issues with device lambdas.
// uses grid-stride loop pattern for scalability across multiple gpu generations.
//
// design principles:
//   - header-only for device lambda compatibility
//   - grid-stride loops for multi-gpu scalability
//   - automatic launch configuration based on domain size
//   - extended device lambdas (c++20 + cuda 13)
//
// usage:
//   // internal use only - called from executor_t::dispatch()
//   cuda_dispatch(domain, [=] __device__ (auto idx) { /* work */ }, stream);
// =============================================================================

#pragma once

#include "grid/domain.hpp"

#ifdef XPU_CUDA_AVAILABLE

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace simbi::xpu {

    // =============================================================================
    // launch configuration helpers
    // =============================================================================

    // compute optimal grid/block dimensions for 1d work
    inline dim3 compute_launch_config_1d(std::int64_t total_size)
    {
        constexpr int threads_per_block = 256;
        const int     blocks            = std::min(
            static_cast<int>((total_size + threads_per_block - 1) / threads_per_block),
            65535
        ); // max grid dimension
        return dim3(blocks, 1, 1);
    }

    // compute optimal grid/block dimensions for 2d work
    inline dim3 compute_launch_config_2d(std::int64_t size_x, std::int64_t size_y)
    {
        constexpr int block_x = 16;
        constexpr int block_y = 16;

        const int grid_x = std::min(static_cast<int>((size_x + block_x - 1) / block_x), 65535);
        const int grid_y = std::min(static_cast<int>((size_y + block_y - 1) / block_y), 65535);

        return dim3(grid_x, grid_y, 1);
    }

    // compute optimal grid/block dimensions for 3d work
    inline dim3
    compute_launch_config_3d(std::int64_t size_x, std::int64_t size_y, std::int64_t size_z)
    {
        constexpr int block_x = 8;
        constexpr int block_y = 8;
        constexpr int block_z = 8;

        const int grid_x = std::min(static_cast<int>((size_x + block_x - 1) / block_x), 65535);
        const int grid_y = std::min(static_cast<int>((size_y + block_y - 1) / block_y), 65535);
        const int grid_z = std::min(static_cast<int>((size_z + block_z - 1) / block_z), 65535);

        return dim3(grid_x, grid_y, grid_z);
    }

    // =============================================================================
    // cuda dispatch kernels - grid-stride pattern
    // note: these must only be compiled with nvcc (not regular c++ compiler)
    // =============================================================================

#ifdef __CUDACC__
    // 1d grid-stride kernel
    template <typename Func>
    __global__ void dispatch_kernel_1d(grid::domain_t<1> domain, Func func)
    {
        const std::int64_t total_size = domain.size();
        const std::int64_t stride     = blockDim.x * gridDim.x;
        const std::int64_t start_idx  = blockIdx.x * blockDim.x + threadIdx.x;

        for (std::uint64_t linear = start_idx; linear < total_size; linear += stride) {
            auto coord = domain.linear_to_coord(linear);
            func(coord);
        }
    }

    // 2d grid-stride kernel
    template <typename Func>
    __global__ void dispatch_kernel_2d(grid::domain_t<2> domain, Func func)
    {
        const auto shape    = domain.shape();
        const auto stride_x = blockDim.x * gridDim.x;
        const auto stride_y = blockDim.y * gridDim.y;
        const auto start_x  = blockIdx.x * blockDim.x + threadIdx.x;
        const auto start_y  = blockIdx.y * blockDim.y + threadIdx.y;

        for (std::int64_t yy = start_y; yy < shape[0]; yy += stride_y) {
            for (std::int64_t xx = start_x; xx < shape[1]; xx += stride_x) {
                typename grid::domain_t<2>::coord_t coord{
                    yy + domain.start[0],
                    xx + domain.start[1]
                };
                func(coord);
            }
        }
    }

    // 3d grid-stride kernel
    template <typename Func>
    __global__ void dispatch_kernel_3d(grid::domain_t<3> domain, Func func)
    {
        const auto shape    = domain.shape();
        const auto stride_x = blockDim.x * gridDim.x;
        const auto stride_y = blockDim.y * gridDim.y;
        const auto stride_z = blockDim.z * gridDim.z;
        const auto start_x  = blockIdx.x * blockDim.x + threadIdx.x;
        const auto start_y  = blockIdx.y * blockDim.y + threadIdx.y;
        const auto start_z  = blockIdx.z * blockDim.z + threadIdx.z;

        for (std::int64_t zz = start_z; zz < shape[0]; zz += stride_z) {
            for (std::int64_t yy = start_y; yy < shape[1]; yy += stride_y) {
                for (std::int64_t xx = start_x; xx < shape[2]; xx += stride_x) {
                    typename grid::domain_t<3>::coord_t coord{
                        zz + domain.start[0],
                        yy + domain.start[1],
                        xx + domain.start[2]
                    };
                    func(coord);
                }
            }
        }
    }
#endif // __CUDACC__

    // =============================================================================
    // cuda dispatch implementations by rank
    // note: these must only be compiled with nvcc for kernel launch syntax
    // =============================================================================

#ifdef __CUDACC__
    // 1d dispatch
    template <typename Func>
    inline void cuda_dispatch_1d(const grid::domain_t<1>& domain, Func&& func, cudaStream_t stream)
    {
        const auto     total_size = domain.size();
        const auto     grid       = compute_launch_config_1d(total_size);
        constexpr dim3 block(256, 1, 1);

        dispatch_kernel_1d<<<grid, block, 0, stream>>>(domain, func);
    }

    // 2d dispatch
    template <typename Func>
    inline void cuda_dispatch_2d(const grid::domain_t<2>& domain, Func&& func, cudaStream_t stream)
    {
        const auto     shape = domain.shape();
        const auto     grid  = compute_launch_config_2d(shape[1], shape[0]);
        constexpr dim3 block(16, 16, 1);

        dispatch_kernel_2d<<<grid, block, 0, stream>>>(domain, func);
    }

    // 3d dispatch
    template <typename Func>
    inline void cuda_dispatch_3d(const grid::domain_t<3>& domain, Func&& func, cudaStream_t stream)
    {
        const auto     shape = domain.shape();
        const auto     grid  = compute_launch_config_3d(shape[2], shape[1], shape[0]);
        constexpr dim3 block(4, 4, 4);

        dispatch_kernel_3d<<<grid, block, 0, stream>>>(domain, func);
    }
#endif // __CUDACC__

    // =============================================================================
    // generic cuda dispatch - delegates to rank-specific implementation
    // =============================================================================

#ifdef __CUDACC__
    template <std::uint64_t Rank, typename Func>
    inline void cuda_dispatch(const grid::domain_t<Rank>& domain, Func&& func, cudaStream_t stream)
    {
        if constexpr (Rank == 1) {
            cuda_dispatch_1d(domain, std::forward<Func>(func), stream);
        }
        else if constexpr (Rank == 2) {
            cuda_dispatch_2d(domain, std::forward<Func>(func), stream);
        }
        else if constexpr (Rank == 3) {
            cuda_dispatch_3d(domain, std::forward<Func>(func), stream);
        }
        else {
            // fallback for higher ranks - linearized dispatch
            const auto     total_size = domain.size();
            const auto     grid       = compute_launch_config_1d(total_size);
            constexpr dim3 block(256, 1, 1);

            dispatch_kernel_1d<<<grid, block, 0, stream>>>(domain, func);
        }
    }
#else
    // fallback stub when not compiled with nvcc (e.g., regular c++ compiler)
    // this allows non-cuda tests to compile but will never be called at runtime
    template <std::uint64_t Rank, typename Func>
    inline void cuda_dispatch(const grid::domain_t<Rank>&, Func&&, cudaStream_t)
    {
        // stub - never called since dispatch_impl is constexpr-if guarded
    }
#endif // __CUDACC__

    // =============================================================================
    // cuda reduction kernels - two-phase block reduction
    // =============================================================================

#ifdef __CUDACC__
    // trait to detect large types (>32 bytes triggers global memory path)
    template <typename T>
    constexpr bool use_global_reduce_v = sizeof(T) > 32;

    // warp reduction using shared memory for small types
    template <typename T, typename ReduceOp>
    __device__ T warp_reduce(T value, ReduceOp reduce_op)
    {
        __shared__ T warp_shared[1024];

        const unsigned int lane     = threadIdx.x % 32;
        const unsigned int warp_id  = threadIdx.x / 32;
        T*                 warp_mem = &warp_shared[warp_id * 32];

        warp_mem[lane] = value;
        __syncwarp();

        for (unsigned int offset = 16; offset > 0; offset /= 2) {
            if (lane < offset) {
                warp_mem[lane] = reduce_op(warp_mem[lane], warp_mem[lane + offset]);
            }
            __syncwarp();
        }

        return warp_mem[0];
    }

    // global memory reduction for large types
    template <typename T, std::uint64_t Rank, typename MapFunc, typename ReduceOp>
    __global__ void reduce_kernel_global(
        grid::domain_t<Rank> domain,
        T                    init_value,
        MapFunc              map_func,
        ReduceOp             reduce_op,
        T*                   thread_results,
        int                  total_threads
    )
    {
        const std::int64_t total_size = domain.size();
        const std::int64_t stride     = blockDim.x * gridDim.x;
        const std::int64_t start_idx  = blockIdx.x * blockDim.x + threadIdx.x;
        const std::int64_t tid        = start_idx;

        T thread_result = init_value;
        for (std::uint64_t linear = start_idx; linear < total_size; linear += stride) {
            auto coord        = domain.linear_to_coord(linear);
            T    mapped_value = map_func(coord);
            thread_result     = reduce_op(thread_result, mapped_value);
        }

        if (tid < total_threads) {
            thread_results[tid] = thread_result;
        }
    }

    // block-level reduction kernel - phase 1 (small types only)
    template <typename T, std::uint64_t Rank, typename MapFunc, typename ReduceOp>
    __global__ void reduce_kernel_phase1(
        grid::domain_t<Rank> domain,
        T                    init_value,
        MapFunc              map_func,
        ReduceOp             reduce_op,
        T*                   block_results
    )
    {
        constexpr unsigned int block_size = 128;
        __shared__ T           shared[block_size / 32];

        const std::int64_t total_size = domain.size();
        const std::int64_t stride     = blockDim.x * gridDim.x;
        const std::int64_t start_idx  = blockIdx.x * blockDim.x + threadIdx.x;

        T thread_result = init_value;
        for (std::uint64_t linear = start_idx; linear < total_size; linear += stride) {
            auto coord        = domain.linear_to_coord(linear);
            T    mapped_value = map_func(coord);
            thread_result     = reduce_op(thread_result, mapped_value);
        }

        const unsigned int lane    = threadIdx.x % 32;
        const unsigned int warp_id = threadIdx.x / 32;
        thread_result              = warp_reduce(thread_result, reduce_op);

        if (lane == 0) {
            shared[warp_id] = thread_result;
        }
        __syncthreads();

        if (warp_id == 0) {
            T warp_result = (lane < blockDim.x / 32) ? shared[lane] : init_value;
            warp_result   = warp_reduce(warp_result, reduce_op);

            if (lane == 0) {
                block_results[blockIdx.x] = warp_result;
            }
        }
    }

    // final reduction kernel - phase 2 (small types only)
    template <typename T, typename ReduceOp>
    __global__ void reduce_kernel_phase2(
        T*       block_results,
        int      num_blocks,
        T        init_value,
        ReduceOp reduce_op,
        T*       final_result
    )
    {
        constexpr unsigned int block_size = 128;
        __shared__ T           shared[block_size / 32];

        T thread_result = init_value;
        for (int ii = threadIdx.x; ii < num_blocks; ii += blockDim.x) {
            thread_result = reduce_op(thread_result, block_results[ii]);
        }

        const unsigned int lane    = threadIdx.x % 32;
        const unsigned int warp_id = threadIdx.x / 32;
        thread_result              = warp_reduce(thread_result, reduce_op);

        if (lane == 0) {
            shared[warp_id] = thread_result;
        }
        __syncthreads();

        if (warp_id == 0) {
            T warp_result = (lane < blockDim.x / 32) ? shared[lane] : init_value;
            warp_result   = warp_reduce(warp_result, reduce_op);

            if (lane == 0) {
                *final_result = warp_result;
            }
        }
    }

    // cuda reduce implementation
    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    T cuda_reduce(
        const grid::domain_t<Rank>& domain,
        T                           init_value,
        MapFunc&&                   map_func,
        ReduceOp&&                  reduce_op,
        cudaStream_t                stream
    )
    {
        const auto total_size = domain.size();

        if constexpr (use_global_reduce_v<T>) {
            // large types: use global memory reduction (no shared memory)
            constexpr int threads_per_block = 128;
            const int     num_blocks        = std::min(
                static_cast<int>((total_size + threads_per_block - 1) / threads_per_block),
                1024
            );
            const int total_threads = num_blocks * threads_per_block;

            T* d_thread_results = nullptr;
            cudaMalloc(&d_thread_results, total_threads * sizeof(T));

            reduce_kernel_global<<<num_blocks, threads_per_block, 0, stream>>>(
                domain,
                init_value,
                map_func,
                reduce_op,
                d_thread_results,
                total_threads
            );

            // cpu-side reduction of thread results
            std::vector<T> h_thread_results(total_threads);
            cudaMemcpyAsync(
                h_thread_results.data(),
                d_thread_results,
                total_threads * sizeof(T),
                cudaMemcpyDeviceToHost,
                stream
            );
            cudaStreamSynchronize(stream);
            cudaFree(d_thread_results);

            T final_result = init_value;
            for (const auto& val : h_thread_results) {
                final_result = reduce_op(final_result, val);
            }
            return final_result;
        }
        else {
            // small types: use shared memory reduction (fast path)
            constexpr int threads_per_block = 128;
            const int     num_blocks        = std::min(
                static_cast<int>((total_size + threads_per_block - 1) / threads_per_block),
                1024
            );

            T* d_block_results = nullptr;
            T* d_final_result  = nullptr;
            cudaMallocAsync(&d_block_results, num_blocks * sizeof(T), stream);
            cudaMallocAsync(&d_final_result, sizeof(T), stream);

            reduce_kernel_phase1<<<num_blocks, threads_per_block, 0, stream>>>(
                domain,
                init_value,
                map_func,
                reduce_op,
                d_block_results
            );

            reduce_kernel_phase2<<<1, threads_per_block, 0, stream>>>(
                d_block_results,
                num_blocks,
                init_value,
                reduce_op,
                d_final_result
            );

            T final_result;
            cudaMemcpyAsync(
                &final_result,
                d_final_result,
                sizeof(T),
                cudaMemcpyDeviceToHost,
                stream
            );
            cudaStreamSynchronize(stream);

            cudaFreeAsync(d_block_results, stream);
            cudaFreeAsync(d_final_result, stream);

            return final_result;
        }
    }
#else
    // fallback stub for cuda_reduce when not compiled with nvcc
    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    inline T
    cuda_reduce(const grid::domain_t<Rank>&, T init_value, MapFunc&&, ReduceOp&&, cudaStream_t)
    {
        return init_value;
    }
#endif // __CUDACC__

} // namespace simbi::xpu

#endif // XPU_CUDA_AVAILABLE
