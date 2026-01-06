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

#include "../domain.hpp"

#ifdef XPU_CUDA_AVAILABLE

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace xpu {

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
    __global__ void dispatch_kernel_1d(domain_t<1> domain, Func func)
    {
        const std::int64_t total_size = domain.size();
        const std::int64_t stride     = blockDim.x * gridDim.x;
        const std::int64_t start_idx  = blockIdx.x * blockDim.x + threadIdx.x;

        for (std::int64_t linear = start_idx; linear < total_size; linear += stride) {
            auto coord = domain.linear_to_coord(linear);
            func(coord);
        }
    }

    // 2d grid-stride kernel
    template <typename Func>
    __global__ void dispatch_kernel_2d(domain_t<2> domain, Func func)
    {
        const auto shape    = domain.shape();
        const auto stride_x = blockDim.x * gridDim.x;
        const auto stride_y = blockDim.y * gridDim.y;
        const auto start_x  = blockIdx.x * blockDim.x + threadIdx.x;
        const auto start_y  = blockIdx.y * blockDim.y + threadIdx.y;

        for (std::int64_t yy = start_y; yy < shape[0]; yy += stride_y) {
            for (std::int64_t xx = start_x; xx < shape[1]; xx += stride_x) {
                typename domain_t<2>::coord_t coord{yy + domain.start[0], xx + domain.start[1]};
                func(coord);
            }
        }
    }

    // 3d grid-stride kernel
    template <typename Func>
    __global__ void dispatch_kernel_3d(domain_t<3> domain, Func func)
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
                    typename domain_t<3>::coord_t coord{
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
    inline void cuda_dispatch_1d(const domain_t<1>& domain, Func&& func, cudaStream_t stream)
    {
        const auto     total_size = domain.size();
        const auto     grid       = compute_launch_config_1d(total_size);
        constexpr dim3 block(256, 1, 1);

        dispatch_kernel_1d<<<grid, block, 0, stream>>>(domain, func);
    }

    // 2d dispatch
    template <typename Func>
    inline void cuda_dispatch_2d(const domain_t<2>& domain, Func&& func, cudaStream_t stream)
    {
        const auto     shape = domain.shape();
        const auto     grid  = compute_launch_config_2d(shape[1], shape[0]);
        constexpr dim3 block(16, 16, 1);

        dispatch_kernel_2d<<<grid, block, 0, stream>>>(domain, func);
    }

    // 3d dispatch
    template <typename Func>
    inline void cuda_dispatch_3d(const domain_t<3>& domain, Func&& func, cudaStream_t stream)
    {
        const auto     shape = domain.shape();
        const auto     grid  = compute_launch_config_3d(shape[2], shape[1], shape[0]);
        constexpr dim3 block(8, 8, 8);

        dispatch_kernel_3d<<<grid, block, 0, stream>>>(domain, func);
    }
#endif // __CUDACC__

    // =============================================================================
    // generic cuda dispatch - delegates to rank-specific implementation
    // =============================================================================

#ifdef __CUDACC__
    template <std::uint64_t Rank, typename Func>
    inline void cuda_dispatch(const domain_t<Rank>& domain, Func&& func, cudaStream_t stream)
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
    inline void cuda_dispatch(const domain_t<Rank>&, Func&&, cudaStream_t)
    {
        // stub - never called since dispatch_impl is constexpr-if guarded
    }
#endif // __CUDACC__

    // =============================================================================
    // cuda reduction kernels - two-phase block reduction
    // =============================================================================

#ifdef __CUDACC__
    // warp-level reduction using shuffle operations (modern cuda)
    template <typename T, typename ReduceOp>
    __device__ T warp_reduce(T value, ReduceOp reduce_op)
    {
        constexpr unsigned int warp_size = 32;
        for (unsigned int offset = warp_size / 2; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, value, offset);
            value   = reduce_op(value, other);
        }
        return value;
    }

    // block-level reduction kernel - phase 1
    // each block reduces its portion and writes result to block_results
    template <typename T, std::uint64_t Rank, typename MapFunc, typename ReduceOp>
    __global__ void reduce_kernel_phase1(
        domain_t<Rank> domain,
        T              init_value,
        MapFunc        map_func,
        ReduceOp       reduce_op,
        T*             block_results
    )
    {
        constexpr unsigned int block_size = 256;
        __shared__ T           shared[block_size / 32]; // one per warp

        const std::int64_t total_size = domain.size();
        const std::int64_t stride     = blockDim.x * gridDim.x;
        const std::int64_t start_idx  = blockIdx.x * blockDim.x + threadIdx.x;

        // phase 1: grid-stride map and thread-local reduction
        T thread_result = init_value;
        for (std::int64_t linear = start_idx; linear < total_size; linear += stride) {
            auto coord        = domain.linear_to_coord(linear);
            T    mapped_value = map_func(coord);
            thread_result     = reduce_op(thread_result, mapped_value);
        }

        // phase 2: warp-level reduction
        const unsigned int lane    = threadIdx.x % 32;
        const unsigned int warp_id = threadIdx.x / 32;
        thread_result              = warp_reduce(thread_result, reduce_op);

        // first thread in each warp writes to shared memory
        if (lane == 0) {
            shared[warp_id] = thread_result;
        }
        __syncthreads();

        // phase 3: final reduction within block (first warp only)
        if (warp_id == 0) {
            T warp_result = (lane < blockDim.x / 32) ? shared[lane] : init_value;
            warp_result   = warp_reduce(warp_result, reduce_op);

            // first thread writes block result
            if (lane == 0) {
                block_results[blockIdx.x] = warp_result;
            }
        }
    }

    // final reduction kernel - phase 2
    // reduces block results to final value
    template <typename T, typename ReduceOp>
    __global__ void reduce_kernel_phase2(
        T*       block_results,
        int      num_blocks,
        T        init_value,
        ReduceOp reduce_op,
        T*       final_result
    )
    {
        constexpr unsigned int block_size = 256;
        __shared__ T           shared[block_size / 32];

        // load block results
        T thread_result = init_value;
        for (int ii = threadIdx.x; ii < num_blocks; ii += blockDim.x) {
            thread_result = reduce_op(thread_result, block_results[ii]);
        }

        // warp reduction
        const unsigned int lane    = threadIdx.x % 32;
        const unsigned int warp_id = threadIdx.x / 32;
        thread_result              = warp_reduce(thread_result, reduce_op);

        if (lane == 0) {
            shared[warp_id] = thread_result;
        }
        __syncthreads();

        // final reduction
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
        const domain_t<Rank>& domain,
        T                     init_value,
        MapFunc&&             map_func,
        ReduceOp&&            reduce_op,
        cudaStream_t          stream
    )
    {
        const auto total_size = domain.size();

        // launch configuration
        constexpr int threads_per_block = 256;
        const int     num_blocks        = std::min(
            static_cast<int>((total_size + threads_per_block - 1) / threads_per_block),
            1024
        );

        // allocate device memory for block results
        T* d_block_results = nullptr;
        T* d_final_result  = nullptr;
        cudaMalloc(&d_block_results, num_blocks * sizeof(T));
        cudaMalloc(&d_final_result, sizeof(T));

        // phase 1: map and block-level reduction
        reduce_kernel_phase1<<<num_blocks, threads_per_block, 0, stream>>>(
            domain,
            init_value,
            map_func,
            reduce_op,
            d_block_results
        );

        // phase 2: final reduction
        reduce_kernel_phase2<<<1, threads_per_block, 0, stream>>>(
            d_block_results,
            num_blocks,
            init_value,
            reduce_op,
            d_final_result
        );

        // copy result back to host
        T final_result;
        cudaMemcpyAsync(&final_result, d_final_result, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // cleanup
        cudaFree(d_block_results);
        cudaFree(d_final_result);

        return final_result;
    }
#else
    // fallback stub for cuda_reduce when not compiled with nvcc
    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    inline T cuda_reduce(const domain_t<Rank>&, T init_value, MapFunc&&, ReduceOp&&, cudaStream_t)
    {
        return init_value;
    }
#endif // __CUDACC__

} // namespace xpu

#endif // XPU_CUDA_AVAILABLE
