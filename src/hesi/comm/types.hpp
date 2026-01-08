#ifndef HET_COMM_TYPES_HPP
#define HET_COMM_TYPES_HPP

#include "containers/vector.hpp"

#include <cstdint>

namespace simbi::het::comm {

    // identifies a compute resource
    struct rank_id_t {
        std::int32_t node;     // mpi rank
        std::int32_t device;   // gpu index within node (-1 for cpu)

        bool operator==(const rank_id_t&) const = default;
    };

    // communication pattern
    enum class transfer_mode_t {
        local_copy,   // same device
        peer_gpu,     // different gpu, same node
        mpi_staged,   // different node: gpu->host->mpi->host->gpu
        mpi_direct,   // different node: gpu->mpi (cuda-aware mpi)
        host_only     // cpu-only transfer
    };

    // describes a rectangular slab of a domain
    template <std::uint64_t Rank>
    struct region_descriptor_t {
        iarray<Rank> start;
        iarray<Rank> extent;   // size in each dimension

        std::uint64_t size() const
        {
            std::uint64_t n = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                n *= extent[ii];
            }
            return n;
        }
    };

    // mpi message tag builder
    struct message_tag_t {
        std::uint64_t dimension;
        std::uint64_t direction;   // 0=left, 1=right
        std::uint64_t phase;       // for multistage exchanges

        std::uint64_t encode() const
        {
            return (phase << 16) | (dimension << 8) | direction;
        }
    };

}   // namespace simbi::het::comm

#endif
