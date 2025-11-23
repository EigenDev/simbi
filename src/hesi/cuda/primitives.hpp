#ifndef HET_BACKEND_CUDA_PRIMITIVES_HPP
#define HET_BACKEND_CUDA_PRIMITIVES_HPP

#ifdef CUDA_ENABLED
#include "compat.hpp"
#include "hesi/device/context.hpp"

namespace simbi::het::backend::cuda {

    // warp-level reduction (UNCHANGED from before)
    template <typename T, typename BinaryOp>
    DEV T warp_reduce(T value, BinaryOp op)
    {
        auto grp = dgrid::this_sub_group();

        for (int offset = grp.size() / 2; offset > 0; offset /= 2) {
            T other = grp.shuffle_down(value, offset);
            value   = op(value, other);
        }

        return value;
    }

    // block-level reduction (UNCHANGED from before)
    template <typename T, typename BinaryOp>
    DEV T block_reduce(T value, BinaryOp op, T identity)
    {
        extern __shared__ char smem_bytes[];
        T* shared = reinterpret_cast<T*>(smem_bytes);

        auto ctx = dgrid::this_thread();
        auto grp = dgrid::this_sub_group();

        // warp reduce
        T warp_val = warp_reduce(value, op);

        // first thread in warp writes to shared
        std::uint64_t warp_id = ctx.thread_idx.x / grp.size();
        if (grp.rank() == 0) {
            shared[warp_id] = warp_val;
        }
        __syncthreads();

        // first warp reduces across warp results
        std::uint64_t num_warps = ctx.block_dim.x / grp.size();
        if (ctx.thread_idx.x < num_warps) {
            T block_val = shared[ctx.thread_idx.x];
            block_val   = warp_reduce(block_val, op);

            if (ctx.thread_idx.x == 0) {
                shared[0] = block_val;
            }
        }
        __syncthreads();

        return shared[0];
    }

    // atomic update (UNCHANGED from before)
    template <typename T, typename BinaryOp>
    DEV void atomic_update(T* address, T value, BinaryOp op)
    {
        if constexpr (std::is_same_v<BinaryOp, std::plus<T>> ||
                      std::is_same_v<BinaryOp, std::plus<>>) {
            atomicAdd(address, value);
        }
        else {
            T old = atomicAdd(address, T{0});   // atomic load
            T assumed;
            do {
                assumed    = old;
                T computed = op(assumed, value);
                old        = atomicCAS(address, assumed, computed);
            } while (assumed != old);
        }
    }

}   // namespace simbi::het::backend::cuda

#endif   // CUDA_ENABLED
#endif
