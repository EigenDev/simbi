// =============================================================================
// api.hpp
//
// public api for the geometry service.
// defines configuration types (`geometry_config_t`, `dimension_config_t`) and
// the `geometry_service_t`, which is responsible for creating coordinate
// maps for different parts of the grid based on the simulation's geometry.
//
// usage:
//   geometry_service_t<2> service{config};
//   auto map_variant = service.create_map(dim, topo_coord, level);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "containers/vector.hpp"
#include "geometry/coordinate_map.hpp"
#include "utility/bimap.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <variant>
#include <vector>

namespace simbi::geometry {

    // -------------------------------------------------------------------------
    // configuration types
    // -------------------------------------------------------------------------
    enum class map_type_t {
        uniform,
        log
    };
    enum class metric_type_t {
        cartesian,
        spherical,
        cylindrical
    };

    struct dimension_config_t
    {
        map_type_t type;
        real       start;
        real       end;
    };

    template <std::size_t Rank>
    struct geometry_config_t
    {
        metric_type_t                   metric;
        std::vector<dimension_config_t> dims;

        // we need to know the "resolution" of a block
        // assuming uniform block size across the grid (e.g. 128^3)
        iarray<Rank> block_size_cells;
    };

    // -------------------------------------------------------------------------
    // the "variant" map
    // holds either a uniform map or a log map
    // this allows us to store heterogeneous maps in a single container
    // -------------------------------------------------------------------------
    using any_map_t = std::variant<uniform_map_t, log_map_t>;

    // -------------------------------------------------------------------------
    // the service
    // -------------------------------------------------------------------------
    template <std::size_t Rank>
    struct geometry_service_t
    {
        geometry_config_t<Rank> config_;
        iarray<Rank>            root_blocks_;

        // construct a specific map for a dimension, level, and coordinate
        any_map_t create_map(std::size_t dim, std::int64_t topo_coord, std::int64_t level) const
        {
            const auto& dconf = config_.dims[dim];

            // calculate global physical span
            real global_len = dconf.end - dconf.start;

            //  calculate the scale factor for this level
            // level 0 has 'root_blocks'
            // level L has 'root_blocks * 2^L' effective blocks
            std::int64_t level_factor = 1 << level;
            std::int64_t total_blocks = root_blocks_[dim] * level_factor;

            // calculate the physical width of this specific block
            real block_width_phys = global_len / static_cast<real>(total_blocks);

            // calculate the physical start of this specific block
            real block_start_phys = dconf.start + static_cast<real>(topo_coord) * block_width_phys;

            // calculate cell spacing info
            std::int64_t n_cells = config_.block_size_cells[dim];

            if (dconf.type == map_type_t::uniform) {
                real dx = block_width_phys / static_cast<real>(n_cells);
                return uniform_map_t(block_start_phys, dx);
            }
            else {
                // log map
                // start = block_start_phys
                // end = block_start_phys + block_width_phys
                // log_slope = log10(end / start) / n_cells
                real end_phys = block_start_phys + block_width_phys;

                // safety check for log(0)
                if (block_start_phys <= 0) {
                    // handle pole singularity or configuration error
                    // usually log grids start at > 0
                }

                real log_slope =
                    std::log10(end_phys / block_start_phys) / static_cast<real>(n_cells);
                std::cout << "log_slope: " << log_slope << "\n";
                std::cin.get();
                return log_map_t(block_start_phys, log_slope);
            }
        }

        // helper to unpack the variant into the metric template
        // this requires a "visit" pattern in the physics kernel launch
        // or we return a struct that the kernel can hold.
        // since GPU kernels can't easily hold std::variant, we usually
        // instantiate the kernel *inside* the visit.
    };

} // namespace simbi::geometry

namespace simbi {
    REGISTER_ENUM_BIMAP(
        geometry::map_type_t,
        {geometry::map_type_t::uniform, "linear"},
        {geometry::map_type_t::log, "log"},
    );

    REGISTER_ENUM_BIMAP(
        geometry::metric_type_t,
        {geometry::metric_type_t::cartesian, "cartesian"},
        {geometry::metric_type_t::spherical, "spherical"},
        {geometry::metric_type_t::cylindrical, "cylindrical"}
    );
} // namespace simbi
