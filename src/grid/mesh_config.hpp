#ifndef GRID_MESH_CONFIG_HPP
#define GRID_MESH_CONFIG_HPP

#include "boundary.hpp"
#include "build_config.hpp"
#include "containers/vector.hpp"
#include "geometry/api.hpp" // for geometry_config_t

#include <cstdint>
#include <functional>

namespace simbi::grid {

    // -------------------------------------------------------------------------
    // motion configuration
    // describes how the mesh moves over time (e.g., homologous expansion)
    // -------------------------------------------------------------------------
    struct motion_config_t
    {
        bool enabled    = false;
        bool homologous = false; // if true, v_grid = H(t) * r

        // initial scale factor (usually 1.0)
        real scale_factor_0 = 1.0;

        // expansion history function a(t) and adot(t)
        // used by the host to update the frame state
        std::function<real(real)> scale_func;
        std::function<real(real)> rate_func;
    };

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

        // physics / geometry
        geometry::geometry_config_t<Rank> geometry;
        motion_config_t                   motion;
        boundary_set_t<Rank>              boundaries;

        // ghost cells
        std::int64_t halo_width;
    };

} // namespace simbi::grid

#endif // GRID_MESH_CONFIG_HPP
