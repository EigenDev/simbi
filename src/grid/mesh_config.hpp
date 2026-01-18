// =============================================================================
// mesh_config.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "boundary.hpp"
#include "containers/vector.hpp"
#include "geometry/api.hpp" // for geometry_config_t

#include <cstdint>

namespace simbi::grid {

    // -------------------------------------------------------------------------
    // boundary configuration
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct boundary_config_t
    {
        // rules for the 2*Rank faces of the global domain
        boundary_set_t<Rank> active_boundaries;
    };

    // -------------------------------------------------------------------------
    // master mesh configuration
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct mesh_config_t
    {
        static constexpr std::uint64_t rank = Rank;
        // topology
        iarray<Rank> global_cells; // total cells (e.g. 1024, 1024)
        iarray<Rank> block_size;   // per block (e.g. 128, 128)

        // geometry (comoving coordinates, fixed throughout simulation)
        geometry::geometry_config_t<Rank> geometry;
        boundary_set_t<Rank>              boundaries;

        // ghost cells
        std::int64_t halo_width;
    };

} // namespace simbi::grid


