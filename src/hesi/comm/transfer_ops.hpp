#ifndef HET_COMM_TRANSFER_OPS_HPP
#define HET_COMM_TRANSFER_OPS_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "hesi/exec/executor.hpp"
#include "hesi/exec/for_each.hpp"
#include "hesi/exec/token.hpp"
#include "hesi/mem/view.hpp"

#include <cstdint>
namespace simbi::het::comm {

    struct transfer_request_t
    {
        exec::token_t gpu_token;
#ifdef MPI_ENABLED
        std::vector<MPI_Request> mpi_requests;
#endif

        void wait(exec::stream_t& stream)
        {
            gpu_token.synchronize();
#ifdef MPI_ENABLED
            if (!mpi_requests.empty()) {
                MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);
                mpi_requests.clear();
            }
#else
            (void) stream;
#endif
        }
    };

    // pack region into contiguous buffer
    template <typename T, std::uint64_t Rank>
    exec::token_t pack_region(
        exec::executor_t&           exec,
        T*                          buffer,
        const mem::view_t<T, Rank>& view,
        const grid::domain_t<Rank>& region
    )
    {
        return exec::parallel_for(exec::gpu_t{}, exec, region, [=] DEV(iarray<Rank> coord) {
            std::uint64_t idx = region.coord_to_linear(coord);
            buffer[idx]       = view(coord);
        });
    }

    // unpack buffer into region
    template <typename T, std::uint64_t Rank>
    exec::token_t unpack_region(
        exec::executor_t&           exec,
        mem::view_t<T, Rank>&       view,
        const grid::domain_t<Rank>& region,
        const T*                    buffer
    )
    {
        return exec::parallel_for(exec::gpu_t{}, exec, region, [=] DEV(iarray<Rank> coord) {
            std::uint64_t idx = region.coord_to_linear(coord);
            view(coord)       = buffer[idx];
        });
    }

    // local device copy (same gpu)
    template <std::uint64_t Rank, typename DstView, typename SrcView>
    exec::token_t local_copy_region(
        exec::executor_t&           exec,
        DstView                     dst_view,
        const grid::domain_t<Rank>& dst_region,
        const SrcView&              src_view,
        const grid::domain_t<Rank>& src_region
    )
    {
        return exec::parallel_for(
            exec::gpu_t{},
            exec,
            src_region,
            [=] DEV(iarray<Rank> src_coord) mutable {
                auto offset         = src_coord - src_region.start;
                auto dst_coord      = dst_region.start + offset;
                dst_view[dst_coord] = src_view(src_coord);
            }
        );
    }

} // namespace simbi::het::comm

#endif
