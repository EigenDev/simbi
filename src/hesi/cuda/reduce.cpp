#ifndef HET_BACKEND_CUDA_REDUCE_HPP
#define HET_BACKEND_CUDA_REDUCE_HPP

#ifdef CUDA_ENABLED
#include "grid/domain.hpp"
#include "hesi/core/error_handling.hpp"
#include "hesi/cuda/primitives.hpp"
#include <cuda_runtime.h>

namespace simbi::het::backend::cuda {

    // =======================================================================
    // PASS 1 KERNEL: Reduce input to per-block partial results
    // =======================================================================

    template <typename T, typename BinaryOp>
    __global__ void reduce_range_kernel_pass1(
        const T* input,
        T* partial_results,   // output: one value per block
        std::uint64_t n,
        BinaryOp op,
        T identity
    )
    {
        extern __shared__ char smem_bytes[];
        T* shared = reinterpret_cast<T*>(smem_bytes);

        auto ctx = dgrid::this_thread();
        auto grp = dgrid::this_sub_group();

        // grid-stride accumulation
        T acc = identity;
        for (std::uint64_t ii = ctx.global_linear_id(); ii < n;
             ii += ctx.total_threads()) {
            acc = op(acc, input[ii]);
        }

        // warp reduce
        T warp_val = warp_reduce(acc, op);

        // store warp results in shared memory
        std::uint64_t warp_id = ctx.thread_idx.x / grp.size();
        if (grp.rank() == 0) {
            shared[warp_id] = warp_val;
        }
        __syncthreads();

        // first warp reduces all warp results
        std::uint64_t num_warps = ctx.block_dim.x / grp.size();
        if (ctx.thread_idx.x < num_warps) {
            T val = shared[ctx.thread_idx.x];
            val   = warp_reduce(val, op);

            // block leader writes to global memory
            if (ctx.thread_idx.x == 0) {
                partial_results[ctx.block_idx.x] = val;
            }
        }
    }

    // =======================================================================
    // PASS 2 KERNEL: Reduce partial results to final answer
    // =======================================================================

    template <typename T, typename BinaryOp>
    __global__ void reduce_range_kernel_pass2(
        const T* partial_results,
        T* final_output,
        std::uint64_t num_partials,
        BinaryOp op,
        T identity,
        T init   // initial value to combine with
    )
    {
        extern __shared__ char smem_bytes[];
        T* shared = reinterpret_cast<T*>(smem_bytes);

        auto ctx = dgrid::this_thread();
        auto grp = dgrid::this_sub_group();

        // each thread processes subset of partial results
        T acc = identity;
        for (std::uint64_t ii = ctx.thread_idx.x; ii < num_partials;
             ii += ctx.block_dim.x) {
            acc = op(acc, partial_results[ii]);
        }

        // warp reduce
        T warp_val = warp_reduce(acc, op);

        std::uint64_t warp_id = ctx.thread_idx.x / grp.size();
        if (grp.rank() == 0) {
            shared[warp_id] = warp_val;
        }
        __syncthreads();

        // final reduction
        std::uint64_t num_warps = ctx.block_dim.x / grp.size();
        if (ctx.thread_idx.x < num_warps) {
            T val = shared[ctx.thread_idx.x];
            val   = warp_reduce(val, op);

            if (ctx.thread_idx.x == 0) {
                *final_output = op(init, val);   // combine with init
            }
        }
    }

    // =======================================================================
    // TRANSFORM-REDUCE PASS 1: Compute + reduce
    // =======================================================================

    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp>
    __global__ void transform_reduce_kernel_pass1(
        Computation comp,
        T* partial_results,
        TransformOp transform,
        BinaryOp op,
        T identity
    )
    {
        extern __shared__ char smem_bytes[];
        T* shared = reinterpret_cast<T*>(smem_bytes);

        auto ctx        = dgrid::this_thread();
        auto grp        = dgrid::this_sub_group();
        auto domain     = comp.domain();
        std::uint64_t n = domain.size();

        // grid-stride accumulation with coordinate mapping
        T acc = identity;
        for (std::uint64_t ii = ctx.global_linear_id(); ii < n;
             ii += ctx.total_threads()) {
            auto coord = domain.linear_to_coord(ii);
            acc        = op(acc, transform(comp(coord)));
        }

        // warp reduce
        T warp_val = warp_reduce(acc, op);

        std::uint64_t warp_id = ctx.thread_idx.x / grp.size();
        if (grp.rank() == 0) {
            shared[warp_id] = warp_val;
        }
        __syncthreads();

        // block reduce
        std::uint64_t num_warps = ctx.block_dim.x / grp.size();
        if (ctx.thread_idx.x < num_warps) {
            T val = shared[ctx.thread_idx.x];
            val   = warp_reduce(val, op);

            if (ctx.thread_idx.x == 0) {
                partial_results[ctx.block_idx.x] = val;
            }
        }
    }

    // =======================================================================
    // TWO-PASS REDUCE: Contiguous range
    // =======================================================================

    template <typename T, typename BinaryOp>
    void reduce_range(
        cudaStream_t stream,
        const T* input,
        T* output,
        std::uint64_t n,
        T init,
        BinaryOp op,
        T identity
    )
    {
        if (n == 0) {
            return;
        }

        constexpr std::uint64_t block_size = 256;
        std::uint64_t grid_size =
            std::min((n + block_size - 1) / block_size, 1024ULL);
        std::size_t smem = (block_size / 32) * sizeof(T);

        // allocate temp buffer for partial results
        T* partial_results;
        check_error<cuda_backend_t>(
            cudaMalloc(&partial_results, grid_size * sizeof(T)),
            "allocate partial results"
        );

        // pass 1: reduce to per-block results
        reduce_range_kernel_pass1<<<grid_size, block_size, smem, stream>>>(
            input,
            partial_results,
            n,
            op,
            identity
        );

        check_error<cuda_backend_t>(cudaGetLastError(), "reduce pass 1 launch");

        // pass 2: reduce partial results (single block)
        reduce_range_kernel_pass2<<<1, block_size, smem, stream>>>(
            partial_results,
            output,
            grid_size,
            op,
            identity,
            init
        );

        check_error<cuda_backend_t>(cudaGetLastError(), "reduce pass 2 launch");

        // cleanup (async - will happen after kernels complete)
        cudaFreeAsync(partial_results, stream);
    }

    // =======================================================================
    // TWO-PASS TRANSFORM-REDUCE: Domain computation
    // =======================================================================

    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp>
    void transform_reduce(
        cudaStream_t stream,
        const Computation& comp,
        T* result,
        T init,
        TransformOp transform,
        BinaryOp op,
        T identity
    )
    {
        std::uint64_t n = comp.domain().size();
        if (n == 0) {
            return;
        }

        constexpr std::uint64_t block_size = 256;
        std::uint64_t grid_size =
            std::min((n + block_size - 1) / block_size, 1024ULL);
        std::size_t smem = (block_size / 32) * sizeof(T);

        // allocate temp buffer
        T* partial_results;
        check_error<cuda_backend_t>(
            cudaMalloc(&partial_results, grid_size * sizeof(T)),
            "allocate partial results"
        );

        // pass 1: transform + reduce
        transform_reduce_kernel_pass1<<<grid_size, block_size, smem, stream>>>(
            comp,
            partial_results,
            transform,
            op,
            identity
        );

        check_error<cuda_backend_t>(
            cudaGetLastError(),
            "transform_reduce pass 1 launch"
        );

        // pass 2: final reduce
        reduce_range_kernel_pass2<<<1, block_size, smem, stream>>>(
            partial_results,
            result,
            grid_size,
            op,
            identity,
            init
        );

        check_error<cuda_backend_t>(
            cudaGetLastError(),
            "transform_reduce pass 2 launch"
        );

        // cleanup
        cudaFreeAsync(partial_results, stream);
    }

}   // namespace simbi::het::backend::cuda

#endif   // CUDA_ENABLED
#endif
