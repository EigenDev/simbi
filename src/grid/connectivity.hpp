#ifndef GRID_CONNECTIVITY_HPP
#define GRID_CONNECTIVITY_HPP

#include "boundary.hpp"
#include "patch_id.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>
#include <vector>

namespace simbi::grid {
    // -------------------------------------------------------------------------
    // face definition
    // -------------------------------------------------------------------------
    enum class side_t : std::uint8_t {
        left  = 0,
        right = 1
    };

    struct face_id_t {
        std::uint8_t dimension;   // 2=x, 1=y, 0=z
        side_t side;              // left/right

        // linear index for array storage (0..2*Rank-1)
        constexpr std::size_t linear_index() const
        {
            return static_cast<std::size_t>(dimension) * 2 +
                   static_cast<std::size_t>(side);
        }
    };

    // -------------------------------------------------------------------------
    // connection info
    // represents what exists at a specific face of a block
    // -------------------------------------------------------------------------
    struct connection_t {
        boundary_type_t type;

        // amr upgrade: list of neighbors instead of single optional
        // - empty: physical boundary
        // - size 1: conforming or coarse-fine (1 parent)
        // - size >1: coarse-fine (many children)
        std::vector<patch_id_t> neighbors;

        // ---------------------------------------------------------------------
        // factories
        // ---------------------------------------------------------------------
        static connection_t physical(boundary_type_t bc) { return {bc, {}}; }

        static connection_t internal(patch_id_t neighbor)
        {
            return {boundary_type_t::partition, {neighbor}};
        }

        static connection_t internal(std::vector<patch_id_t> neighbors)
        {
            return {boundary_type_t::partition, std::move(neighbors)};
        }

        // ---------------------------------------------------------------------
        // queries
        // ---------------------------------------------------------------------
        bool is_physical() const
        {
            // treat periodic as logical connection usually, but structurally
            // it behaves like a boundary that wraps.
            // here we define 'physical' as having no explicit neighbors stored.
            return neighbors.empty();
        }

        bool is_connected() const { return !neighbors.empty(); }

        bool is_conforming() const { return neighbors.size() == 1; }

        bool is_refined() const { return neighbors.size() > 1; }

        boundary_type_t boundary_type() const { return type; }

        // helper for the 90% case (1 neighbor)
        const patch_id_t& single_neighbor() const { return neighbors[0]; }
    };

    // -------------------------------------------------------------------------
    // ostream operator for debugging
    // -------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const connection_t& conn)
    {
        if (conn.is_connected()) {
            os << "Connected to ";
            for (const auto& neighbor : conn.neighbors) {
                os << neighbor << " ";
            }
        }
        else {
            os << "Physical BC (" << static_cast<int>(conn.type) << ")";
        }
        return os;
    }

}   // namespace simbi::grid

#endif   // GRID_CONNECTIVITY_HPP
