// =============================================================================
// comm/types.hpp
//
// minimal communication type definitions for multi-device coordination
// provides rank identification and locality queries without mpi dependency
//
// design:
//   - rank_id_t: identifies a compute resource (node + device)
//   - locality queries: determine if ranks are local, same device, etc.
//   - transfer descriptors: describe data movement between ranks
//
// usage:
//   rank_id_t rank{0, 2};  // node 0, device 2
//   if (rank.is_local()) { /* single-node */ }
//   if (same_device(rank1, rank2)) { /* no copy needed */ }
// =============================================================================

#pragma once

#include <cstdint>
#include <functional>

namespace simbi::xpu::comm {

    // =========================================================================
    // rank identification
    // =========================================================================

    // identifies a compute resource within a distributed system
    // node_id = mpi rank (or 0 for single-node)
    // device_id = gpu id within that node
    struct rank_id_t
    {
        std::int64_t node_id   = 0;
        std::int64_t device_id = 0;

        // check if this rank is on the current node
        bool is_local() const
        {
            // for single-node: always local
            // for mpi: would compare to current mpi rank
            return node_id == 0;
        }

        // equality
        bool operator==(const rank_id_t& other) const
        {
            return node_id == other.node_id && device_id == other.device_id;
        }

        bool operator!=(const rank_id_t& other) const
        {
            return !(*this == other);
        }
    };

    // =========================================================================
    // locality queries
    // =========================================================================

    // check if two ranks are on the same physical node
    inline bool same_node(const rank_id_t& a, const rank_id_t& b)
    {
        return a.node_id == b.node_id;
    }

    // check if two ranks are the same device
    inline bool same_device(const rank_id_t& a, const rank_id_t& b)
    {
        return a.node_id == b.node_id && a.device_id == b.device_id;
    }

    // check if two ranks require inter-node communication
    inline bool requires_mpi(const rank_id_t& a, const rank_id_t& b)
    {
        return a.node_id != b.node_id;
    }

    // check if two ranks can use peer-to-peer gpu copy
    inline bool can_use_peer_copy(const rank_id_t& a, const rank_id_t& b)
    {
        return same_node(a, b) && !same_device(a, b);
    }

    // =========================================================================
    // transfer strategy
    // =========================================================================

    enum class transfer_strategy_t : std::uint8_t {
        none,        // no transfer needed (same device)
        peer_copy,   // gpu peer-to-peer (same node, different gpus)
        host_staged, // via host memory (same node, peer disabled)
        mpi_send,    // cross-node (requires mpi)
    };

    // determine optimal transfer strategy for two ranks
    inline transfer_strategy_t get_transfer_strategy(const rank_id_t& src, const rank_id_t& dst)
    {
        if (same_device(src, dst)) {
            return transfer_strategy_t::none;
        }

        if (same_node(src, dst)) {
            // assume peer-to-peer is available
            // could query cudaDeviceCanAccessPeer in actual implementation
            return transfer_strategy_t::peer_copy;
        }

        return transfer_strategy_t::mpi_send;
    }

} // namespace simbi::xpu::comm

// hash support for rank_id_t (for use in std::unordered_map)
namespace std {
    template <>
    struct hash<simbi::xpu::comm::rank_id_t>
    {
        std::size_t operator()(const simbi::xpu::comm::rank_id_t& rank) const noexcept
        {
            return std::hash<std::int64_t>{}(rank.node_id) ^
                   (std::hash<std::int64_t>{}(rank.device_id) << 1);
        }
    };
} // namespace std
