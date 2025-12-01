#ifndef HET_CUDA_PARALLEL_FOR_HPP
#define HET_CUDA_PARALLEL_FOR_HPP

#ifdef CUDA_ENABLED
#include "compat.hpp"
#include "grid/domain.hpp"
#include "hesi/core/error_handling.hpp"
#include "hesi/device/context.hpp"
#include "hesi/exec/policy.hpp"
#include <cuda_runtime.h>

namespace simbi::het::backend::cuda {

    // grid-stride kernel for arbitrary rank
    template <std::uint64_t Rank, typename Functor>
    __global__ void parallel_for_kernel(grid::domain_t<Rank> domain, Functor f)
    {
        // get thread context
        auto ctx = dgrid::ctx();

        if constexpr (Rank == 1) {
            const std::uint64_t stride = ctx.block_dim_x() * ctx.grid_dim_x();
            const std::uint64_t start =
                ctx.block_x() * ctx.block_dim_x() + ctx.thread_x();

            for (std::int64_t ii = start; ii < domain.shape()[0];
                 ii += stride) {
                iarray<1> coord{ii + domain.start[0]};
                f(coord);
            }
        }
        else if constexpr (Rank == 2) {
            const std::uint64_t stride_x = ctx.block_dim_x() * ctx.grid_dim_x();
            const std::uint64_t stride_y = ctx.block_dim_y() * ctx.grid_dim_y();

            const std::uint64_t start_x =
                ctx.block_x() * ctx.block_dim_x() + ctx.thread_x();
            const std::uint64_t start_y =
                ctx.block_y() * ctx.block_dim_y() + ctx.thread_y();

            for (std::int64_t yy = start_y; yy < domain.shape()[0];
                 yy += stride_y) {
                for (std::int64_t xx = start_x; xx < domain.shape()[1];
                     xx += stride_x) {
                    iarray<2> coord{
                      static_cast<std::int64_t>(yy) + domain.start[0],
                      static_cast<std::int64_t>(xx) + domain.start[1]
                    };
                    f(coord);
                }
            }
        }
        else if constexpr (Rank == 3) {
            const std::uint64_t stride_x = ctx.block_dim_x() * ctx.grid_dim_x();
            const std::uint64_t stride_y = ctx.block_dim_y() * ctx.grid_dim_y();
            const std::uint64_t stride_z = ctx.block_dim_z() * ctx.grid_dim_z();

            const std::uint64_t start_x =
                ctx.block_x() * ctx.block_dim_x() + ctx.thread_x();
            const std::uint64_t start_y =
                ctx.block_y() * ctx.block_dim_y() + ctx.thread_y();
            const std::uint64_t start_z =
                ctx.block_z() * ctx.block_dim_z() + ctx.thread_z();

            for (std::uint64_t zz = start_z; zz < domain.shape()[0];
                 zz += stride_z) {
                for (std::uint64_t yy = start_y; yy < domain.shape()[1];
                     yy += stride_y) {
                    for (std::uint64_t xx = start_x; xx < domain.shape()[2];
                         xx += stride_x) {
                        iarray<3> coord{
                          static_cast<std::int64_t>(zz) + domain.start[0],
                          static_cast<std::int64_t>(yy) + domain.start[1],
                          static_cast<std::int64_t>(xx) + domain.start[2]
                        };
                        f(coord);
                    }
                }
            }
        }
    }

    // launch helper
    template <std::uint64_t Rank, typename Functor>
    void parallel_for(
        cudaStream_t stream,
        const exec::launch_policy_t& policy,
        const grid::domain_t<Rank>& domain,
        Functor&& f
    )
    {
        if (domain.empty()) {
            return;   // nothing to do
        }
        const dim3 grid  = {policy.grid.x, policy.grid.y, policy.grid.z};
        const dim3 block = {policy.block.x, policy.block.y, policy.block.z};

        parallel_for_kernel<Rank, Functor>
            <<<grid, block, policy.shared_mem_bytes, stream>>>(
                domain,
                std::forward<Functor>(f)
            );

        // check launch error
        check_error<cuda_backend_t>(
            cudaGetLastError(),
            "parallel_for kernel launch"
        );
    }

}   // namespace simbi::het::backend::cuda

#endif   // CUDA_ENABLED
#endif
