#ifndef HET_LAUNCHER_HPP
#define HET_LAUNCHER_HPP

#include "compat.hpp"
#include "hesi/core/primitives.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/policy.hpp"
#include "hesi/exec/stream.hpp"

#include <cstdint>

namespace simbi::het::exec::detail {

    // -------------------------------------------------------------------------
    // CPU Emulation State (Internal)
    // -------------------------------------------------------------------------
    struct dim3_state_t {
        std::uint64_t x = 0, y = 0, z = 0;
    };

    // Thread Local Storage to mimic hardware registers
    // This is safe for OpenMP as each thread gets its own instance
    struct cpu_execution_state_t {
        dim3_state_t thread_idx;
        dim3_state_t block_idx;
        dim3_state_t block_dim;
        dim3_state_t grid_dim;
    };

    inline thread_local cpu_execution_state_t cpu_state;

    // -------------------------------------------------------------------------
    // gpu kernel wrapper
    // -------------------------------------------------------------------------
    // we need a global entry point to call the functor
    template <typename Functor>
    KERNEL void kernel_entry_point(Functor f)
    {
        f();
    }

    // -------------------------------------------------------------------------
    // dispatch mechanism
    // -------------------------------------------------------------------------
    template <typename Functor>
    void
    launch(const stream_t& stream, const launch_policy_t& policy, Functor f)
    {

        if (stream.backend() == backend_type_t::cpu) {
            cpu_state
                .block_dim = {policy.block.x, policy.block.y, policy.block.z};
            cpu_state.grid_dim = {policy.grid.x, policy.grid.y, policy.grid.z};

#pragma omp parallel for collapse(3)
            for (std::uint64_t bz = 0; bz < policy.grid.z; ++bz) {
                for (std::uint64_t by = 0; by < policy.grid.y; ++by) {
                    for (std::uint64_t bx = 0; bx < policy.grid.x; ++bx) {

                        // update TLS for Block ID
                        // note: in OpenMP, this updates the 'local' thread's
                        // copy
                        cpu_state.block_idx = {bx, by, bz};

                        // block loops (Serial/Tiled)
                        // these run serially to maximize L1 cache locality
                        // (tiling). this mimics the behavior of a CUDA
                        // streaming multiprocessor (SM) processing a
                        // warp/block.
                        for (std::uint64_t tz = 0; tz < policy.block.z; ++tz) {
                            for (std::uint64_t ty = 0; ty < policy.block.y;
                                 ++ty) {
                                for (std::uint64_t tx = 0; tx < policy.block.x;
                                     ++tx) {

                                    // updt=ate TLS for thread id
                                    cpu_state.thread_idx = {tx, ty, tz};

                                    // execute Kernel
                                    f();
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (stream.backend() == backend_type_t::cuda ||
                 stream.backend() == backend_type_t::hip) {
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
            // unpack dimensions
            dim3 grid(policy.grid.x, policy.grid.y, policy.grid.z);
            dim3 block(policy.block.x, policy.block.y, policy.block.z);

            // syntax is identical for cuda and hip
            kernel_entry_point<<<
                grid,
                block,
                policy.shared_mem_bytes,
                stream.native()>>>(f);
#if defined(CUDA_ENABLED)
            check_error<cuda_backend_t>(cudaGetLastError(), "kernel launch");
#elif defined(HIP_ENABLED)
            check_error<hip_backend_t>(hipGetLastError(), "kernel launch");
#endif
            (void) policy;
#endif
        }
    }

}   // namespace simbi::het::exec::detail

#endif   // HETERO_LAUNCHER_HPP
