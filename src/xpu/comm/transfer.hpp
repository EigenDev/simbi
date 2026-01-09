// =============================================================================
// comm/transfer.hpp
//
// communication transfer operations for multi-device data movement
// provides single-node peer-to-peer transfers; stubs for mpi
//
// design:
//   - local transfers use device_memory peer copy (already implemented)
//   - cross-node transfers error until mpi is implemented
//   - clean api that's ready for mpi when needed
//
// usage:
//   // local peer copy (same node, different gpus)
//   transfer_sync(src_rank, src_ptr, dst_rank, dst_ptr, size);
//
//   // async with executor
//   auto token = transfer_async(src_rank, src_ptr, dst_rank, dst_ptr, size, exec);
// =============================================================================

#ifndef XPU_TRANSFER_HPP
#define XPU_TRANSFER_HPP

#include "types.hpp"
#include "xpu/execution/execution_space.hpp"
#include "xpu/execution/token.hpp"
#include "xpu/mem/device_memory.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace simbi::xpu::comm {

    // =========================================================================
    // synchronous transfers
    // =========================================================================

    // synchronous transfer between two ranks
    // dispatches based on locality:
    //   - same device: no-op
    //   - same node: peer copy
    //   - different nodes: error (mpi not implemented)
    inline void transfer_sync(
        const rank_id_t& src_rank,
        const void*      src_ptr,
        const rank_id_t& dst_rank,
        void*            dst_ptr,
        std::size_t      bytes
    )
    {
        auto strategy = get_transfer_strategy(src_rank, dst_rank);

        switch (strategy) {
            case transfer_strategy_t::none:
                // same device, no transfer needed
                return;

            case transfer_strategy_t::peer_copy: {
                // same node, different devices: use peer copy
                bool success = device_memory_t::memcpy_peer(
                    dst_ptr,
                    dst_rank.device_id,
                    src_ptr,
                    src_rank.device_id,
                    bytes
                );
                if (!success) {
                    throw std::runtime_error("peer-to-peer copy failed");
                }
                return;
            }

            case transfer_strategy_t::host_staged: {
                // fallback when peer access unavailable: copy via host
                std::vector<char> staging(bytes);

// device -> host
#ifdef XPU_CUDA_AVAILABLE
                cudaMemcpy(staging.data(), src_ptr, bytes, cudaMemcpyDeviceToHost);
                // host -> device
                cudaMemcpy(dst_ptr, staging.data(), bytes, cudaMemcpyHostToDevice);
#else
                std::memcpy(staging.data(), src_ptr, bytes);
                std::memcpy(dst_ptr, staging.data(), bytes);
#endif
                return;
            }

            case transfer_strategy_t::mpi_send:
                // cross-node: requires mpi
                throw std::runtime_error("cross-node transfer requires mpi (not implemented)");
        }
    }

    // =========================================================================
    // asynchronous transfers
    // =========================================================================

    // async transfer between two ranks
    // returns token for synchronization
    //
    // IMPORTANT: executor stream must remain valid until token.sync() completes
    // the stream is used for async operations but not retained by the token
    template <execution_space ExecutionSpace>
    token_t<ExecutionSpace> transfer_async(
        const rank_id_t&            src_rank,
        const void*                 src_ptr,
        const rank_id_t&            dst_rank,
        void*                       dst_ptr,
        std::size_t                 bytes,
        executor_t<ExecutionSpace>& exec
    )
    {
        auto strategy = get_transfer_strategy(src_rank, dst_rank);

        switch (strategy) {
            case transfer_strategy_t::none: {
                // same device, no transfer needed
                auto token = token_t<ExecutionSpace>::create();
                token.mark_ready();
                return token;
            }

            case transfer_strategy_t::peer_copy: {
                // same node, different devices: use async peer copy
                auto token = token_t<ExecutionSpace>::create();

                if constexpr (std::is_same_v<ExecutionSpace, cuda_space>) {
#ifdef XPU_CUDA_AVAILABLE
                    device_memory_t::memcpy_peer_async(
                        dst_ptr,
                        dst_rank.device_id,
                        src_ptr,
                        src_rank.device_id,
                        bytes,
                        exec.stream()
                    );
                    token.record(exec);
#endif
                }
                else {
                    (void) exec; // suppress unused warning
                    // cpu space: just do synchronous copy
                    transfer_sync(src_rank, src_ptr, dst_rank, dst_ptr, bytes);
                    token.mark_ready();
                }

                return token;
            }

            case transfer_strategy_t::host_staged:
                throw std::runtime_error(
                    "host-staged async transfer not implemented (peer copy should work)"
                );

            case transfer_strategy_t::mpi_send:
                throw std::runtime_error(
                    "cross-node async transfer requires mpi (not implemented)"
                );
        }

        // unreachable
        auto token = token_t<ExecutionSpace>::create();
        token.mark_ready();
        return token;
    }

    // =========================================================================
    // halo exchange helpers (for multi-partition grids)
    // =========================================================================

    // describes a single halo transfer operation
    // used to build halo exchange graphs without executing immediately
    struct halo_transfer_t
    {
        rank_id_t   src_rank;
        const void* src_ptr;
        rank_id_t   dst_rank;
        void*       dst_ptr;
        std::size_t bytes;

        // execute this transfer synchronously
        void execute_sync() const
        {
            transfer_sync(src_rank, src_ptr, dst_rank, dst_ptr, bytes);
        }

        // execute this transfer asynchronously
        template <execution_space ExecutionSpace>
        token_t<ExecutionSpace> execute_async(executor_t<ExecutionSpace>& exec) const
        {
            return transfer_async(src_rank, src_ptr, dst_rank, dst_ptr, bytes, exec);
        }
    };

    // =========================================================================
    // region-based transfer helpers for field views
    // =========================================================================

    // synchronous region-based transfer for field views
    // copies data from src_view[src_region] to dst_view[dst_region]
    // uses element-wise copy to work with any view type
    template <typename DstView, typename SrcView, typename Region>
    void transfer_region_sync(
        DstView       dst_view,
        const Region& dst_region,
        SrcView       src_view,
        const Region& src_region
    )
    {
        // element-wise copy for cpu-only builds
        // iterate over the region and copy element by element
        for (std::size_t idx = 0; idx < src_region.size(); ++idx) {
            auto local_coord = src_region.linear_to_coord(idx);
            auto src_coord   = src_region.start + local_coord;
            auto dst_coord   = dst_region.start + local_coord;

            // get mutable references from both views
            // views should provide mutable access even if value_type appears const
            auto& dst_ref = const_cast<typename std::remove_const<
                typename std::remove_reference<decltype(dst_view(dst_coord))>::type>::type&>(
                dst_view(dst_coord)
            );
            auto& src_ref = const_cast<typename std::remove_const<
                typename std::remove_reference<decltype(src_view(src_coord))>::type>::type&>(
                src_view(src_coord)
            );

            dst_ref = src_ref;
        }
    }

    // async region-based transfer (for future gpu support)
    template <execution_space ExecutionSpace, typename DstView, typename SrcView, typename Region>
    token_t<ExecutionSpace> transfer_region_async(
        executor_t<ExecutionSpace>& /*exec*/,
        DstView       dst_view,
        const Region& dst_region,
        SrcView       src_view,
        const Region& src_region
    )
    {
        // for cpu space, just do sync transfer and return ready token
        transfer_region_sync(dst_view, dst_region, src_view, src_region);

        auto token = token_t<ExecutionSpace>::create();
        token.mark_ready();
        return token;
    }

} // namespace simbi::xpu::comm

#endif // XPU_TRANSFER_HPP
