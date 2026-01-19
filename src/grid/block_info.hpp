// =============================================================================
// block_info.hpp
//
// descriptor for a single grid block (patch).
// defines `block_info_t`, a struct that holds all metadata for a single
// block in the grid hierarchy, including its unique `patch_id_t`, its
// geometric domain, and its face connectivity to neighboring blocks or
// physical boundaries.
//
// usage:
//   block_info_t block;
//   block.id = ...;
//   block.geometry = ...;
//   block.connect(0, side_t::left, neighbor_id);
// =============================================================================
#pragma once

#include "boundary.hpp"
#include "connectivity.hpp"
#include "containers/vector.hpp"
#include "domain.hpp"
#include "patch_id.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>

namespace simbi::grid {
    template <std::uint64_t Rank>
    struct block_info_t
    {
        // identity
        patch_id_t id;

        // geometry (the subset of z^rank this block owns)
        domain_t<Rank> geometry;

        // topology (connectivity)
        // stores the state of all 2*Rank faces
        vector_t<connection_t, 2 * Rank> faces;

        // ---------------------------------------------------------------------
        // accessors
        // ---------------------------------------------------------------------
        const connection_t& get_face(std::size_t dim, side_t side) const
        {
            face_id_t f{static_cast<std::uint8_t>(dim), side};
            return faces[f.linear_index()];
        }

        connection_t& get_face(std::size_t dim, side_t side)
        {
            face_id_t f{static_cast<std::uint8_t>(dim), side};
            return faces[f.linear_index()];
        }

        // helper to set physical BCs quickly
        void set_boundary(std::size_t dim, side_t side, boundary_type_t bc)
        {
            get_face(dim, side) = connection_t::physical(bc);
        }

        // helper to link neighbors
        void connect(std::size_t dim, side_t side, patch_id_t neighbor)
        {
            get_face(dim, side) = connection_t::internal(neighbor);
        }
    };

    // ostream operator for debugging
    template <std::uint64_t Rank>
    std::ostream& operator<<(std::ostream& os, const block_info_t<Rank> block)
    {
        os << "Block ID: " << block.id << "\n";
        os << "Geometry: [";
        for (std::uint64_t d = 0; d < Rank; ++d) {
            os << block.geometry.start[d] << ":" << block.geometry.fin[d];
            if (d < Rank - 1) {
                os << ", ";
            }
        }
        os << "]\n";

        os << "Faces:\n";
        for (std::uint64_t d = 0; d < Rank; ++d) {
            for (auto side : {side_t::left, side_t::right}) {
                const auto& conn = block.get_face(d, side);
                os << "  Dim " << d << " " << ((side == side_t::left) ? "Left" : "Right") << ": ";
                if (conn.is_connected()) {
                    os << "Connected to ";
                    for (const auto& neighbor : conn.neighbors) {
                        os << neighbor << " ";
                    }
                }
                else {
                    os << "Physical BC (" << static_cast<int>(conn.type) << ")";
                }
                os << "\n";
            }
        }

        return os;
    }

} // namespace simbi::grid
