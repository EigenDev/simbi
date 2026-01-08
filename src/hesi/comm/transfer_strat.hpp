#ifndef HET_COMM_TRANSFER_STRATEGY_HPP
#define HET_COMM_TRANSFER_STRATEGY_HPP

#include "hesi/core/types.hpp"
#include "hesi/mem/transfer.hpp"
#include "types.hpp"

#include <stdexcept>

namespace simbi::het::comm {

    struct transfer_strategy_t {
        transfer_mode_t mode;
        bool requires_staging;

        static transfer_strategy_t select(
            locality_t src_loc,
            locality_t dst_loc,
            rank_id_t src_rank,
            rank_id_t dst_rank,
            bool cuda_aware_mpi
        )
        {
            // same device
            if (src_rank == dst_rank && src_loc == dst_loc) {
                return {transfer_mode_t::local_copy, false};
            }

            // same node, different devices
            if (src_rank.node == dst_rank.node) {
                bool both_gpu = src_loc.backend != backend_type_t::cpu &&
                                dst_loc.backend != backend_type_t::cpu;

                if (both_gpu && mem::can_access_peer(src_loc, dst_loc)) {
                    return {transfer_mode_t::peer_gpu, false};
                }

                return {transfer_mode_t::host_only, false};
            }

// different nodes
#ifdef MPI_ENABLED
            bool involves_gpu = src_loc.backend != backend_type_t::cpu ||
                                dst_loc.backend != backend_type_t::cpu;

            if (involves_gpu) {
                if (cuda_aware_mpi) {
                    return {transfer_mode_t::mpi_direct, false};
                }
                else {
                    return {transfer_mode_t::mpi_staged, true};
                }
            }

            return {transfer_mode_t::host_only, false};
#else
            (void) cuda_aware_mpi;
            throw std::runtime_error("mpi required for cross-node transfer");
#endif
        }
    };

}   // namespace simbi::het::comm

#endif
