#ifndef HET_COMM_COMMUNICATOR_HPP
#define HET_COMM_COMMUNICATOR_HPP

#include "hesi/core/types.hpp"
#include "hesi/exec/executor.hpp"
#include "hesi/mem/block.hpp"
#include "hesi/mem/view.hpp"
#include "mpi_backend.hpp"
#include "region_utils.hpp"
#include "transfer_ops.hpp"
#include "transfer_strat.hpp"
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace simbi::het::comm {

    struct communicator_t {
        mpi_backend_t mpi_;
        rank_id_t my_rank_;
        bool cuda_aware_mpi_;

        // staging buffers for non-cuda-aware mpi
        std::unordered_map<std::string, mem::block_t> staging_buffers_;

        communicator_t()
            : my_rank_{mpi_.rank(), -1},   // default: cpu only
              cuda_aware_mpi_(false)
        {
// detect cuda-aware mpi at runtime
#if defined(CUDA_ENABLED) && defined(MPIX_CUDA_AWARE_SUPPORT)
            cuda_aware_mpi_ = (MPIX_CUDA_AWARE_SUPPORT != 0);
#endif
        }

        void set_device(std::int32_t device_id) { my_rank_.device = device_id; }

        // generic region exchange
        template <typename T, std::uint64_t Rank>
        transfer_request_t exchange_region(
            exec::executor_t& exec,
            mem::view_t<T, Rank>& local_view,
            const region_descriptor_t<Rank>& send_region,
            rank_id_t dest_rank,
            mem::view_t<T, Rank>& recv_view,
            const region_descriptor_t<Rank>& recv_region,
            rank_id_t src_rank,
            message_tag_t /*tag*/
        )
        {
            locality_t local_loc = exec.stream().locality();

            // determine send strategy
            auto send_strategy = transfer_strategy_t::select(
                local_loc,
                local_loc,   // dest locality unknown, assume same backend
                my_rank_,
                dest_rank,
                cuda_aware_mpi_
            );

            transfer_request_t request;

            // handle send
            if (dest_rank != my_rank_) {
                std::size_t send_bytes = send_region.size() * sizeof(T);

                if (send_strategy.requires_staging) {
                    // gpu -> staging buffer -> mpi
                    auto& staging = get_staging_buffer(
                        "send",
                        send_bytes,
                        locality_t::host()
                    );

                    // pack into staging
                    auto send_domain = to_domain(send_region, /* dummy */ {});
                    auto pack_token  = pack_region(
                        exec,
                        staging.template as<T>(),
                        local_view,
                        send_domain
                    );
                    pack_token.synchronize();

// mpi send from staging
#ifdef MPI_ENABLED
                    auto mpi_req = mpi_.isend(
                        staging.data(),
                        send_region.size(),
                        mpi_type_map<T>::get(),
                        dest_rank.node,
                        tag.encode()
                    );
                    request.mpi_requests.push_back(mpi_req);
#endif
                }
                else {
// direct send (cuda-aware mpi)
#ifdef MPI_ENABLED
                    auto send_domain = to_domain(send_region, {});
                    T* send_ptr      = &local_view(send_domain.start);

                    auto mpi_req = mpi_.isend(
                        send_ptr,
                        send_region.size(),
                        mpi_type_map<T>::get(),
                        dest_rank.node,
                        tag.encode()
                    );
                    request.mpi_requests.push_back(mpi_req);
#endif
                }
            }

            // handle recv
            if (src_rank != my_rank_) {
                std::size_t recv_bytes = recv_region.size() * sizeof(T);

                if (send_strategy.requires_staging) {
                    // mpi -> staging buffer -> gpu
                    auto& staging = get_staging_buffer(
                        "recv",
                        recv_bytes,
                        locality_t::host()
                    );

#ifdef MPI_ENABLED
                    auto mpi_req = mpi_.irecv(
                        staging.data(),
                        recv_region.size(),
                        mpi_type_map<T>::get(),
                        src_rank.node,
                        tag.encode()
                    );
                    request.mpi_requests.push_back(mpi_req);
#endif

// wait for mpi, then unpack
// note: this blocks. for full async, need callback system
#ifdef MPI_ENABLED
                    MPI_Wait(&request.mpi_requests.back(), MPI_STATUS_IGNORE);
#endif

                    auto recv_domain  = to_domain(recv_region, {});
                    request.gpu_token = unpack_region(
                        exec,
                        recv_view,
                        recv_domain,
                        staging.template as<T>()
                    );
                }
                else {
// direct recv
#ifdef MPI_ENABLED
                    auto recv_domain = to_domain(recv_region, {});
                    T* recv_ptr      = &recv_view(recv_domain.start);

                    auto mpi_req = mpi_.irecv(
                        recv_ptr,
                        recv_region.size(),
                        mpi_type_map<T>::get(),
                        src_rank.node,
                        tag.encode()
                    );
                    request.mpi_requests.push_back(mpi_req);
#endif
                }
            }

            return request;
        }

        // halo exchange (common case)
        template <typename field_t>
        transfer_request_t exchange_halo(
            exec::executor_t& exec,
            field_t& field,
            std::uint64_t dim,
            rank_id_t left_neighbor,
            rank_id_t right_neighbor
        )
        {
            using T             = typename field_t::value_type;
            constexpr auto Rank = field_t::rank;

            std::int64_t halo_width =
                field.domain().bounds.rules.elems[dim].second;

            if (halo_width == 0) {
                return transfer_request_t{};
            }

            // build regions
            auto send_left =
                get_interior_boundary(field.domain(), dim, halo_width, false);
            auto send_right =
                get_interior_boundary(field.domain(), dim, halo_width, true);
            auto recv_left =
                get_halo_zone(field.domain(), dim, halo_width, true);
            auto recv_right =
                get_halo_zone(field.domain(), dim, halo_width, false);

            // issue sends and recvs
            message_tag_t left_tag{dim, 0, 0};
            message_tag_t right_tag{dim, 1, 0};

            auto req1 = exchange_region<T, Rank>(
                exec,
                field.view(),
                send_left,
                left_neighbor,
                field.view(),
                recv_right,
                right_neighbor,
                right_tag
            );

            auto req2 = exchange_region<T, Rank>(
                exec,
                field.view(),
                send_right,
                right_neighbor,
                field.view(),
                recv_left,
                left_neighbor,
                left_tag
            );

            // merge requests
            req1.mpi_requests.insert(
                req1.mpi_requests.end(),
                req2.mpi_requests.begin(),
                req2.mpi_requests.end()
            );

            return req1;
        }

      private:
        mem::block_t& get_staging_buffer(
            const std::string& key,
            std::size_t bytes,
            locality_t loc
        )
        {
            auto it = staging_buffers_.find(key);
            if (it == staging_buffers_.end() || it->second.size() < bytes) {
                staging_buffers_[key] =
                    mem::block_t(bytes, loc, memory_type_t::pinned);
            }
            return staging_buffers_[key];
        }
    };

}   // namespace simbi::het::comm

#endif
