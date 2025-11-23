#ifndef GRID_CREATION_GEOMETRY_BUILDER_HPP
#define GRID_CREATION_GEOMETRY_BUILDER_HPP

#include "compat.hpp"
#include "ecs/blueprints.hpp"
#include "geometry/api.hpp"
#include "geometry/coordinate_map.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace simbi::grid::creation {

    template <std::uint64_t Rank>
    struct geometry_builder_t {

        static std::map<patch_id_t, std::vector<geometry::any_map_t>>
        build_maps(
            const std::vector<skeleton_t<Rank>>& hierarchy,
            const ecs::mesh_blueprint_t<Rank>& root_bp
        )
        {
            std::map<patch_id_t, std::vector<geometry::any_map_t>> all_maps;

            for (std::uint64_t lvl = 0; lvl < hierarchy.size(); ++lvl) {
                build_level_maps(hierarchy[lvl], root_bp, lvl, all_maps);
            }

            return all_maps;
        }

      private:
        static void build_level_maps(
            const skeleton_t<Rank>& skeleton,
            const ecs::mesh_blueprint_t<Rank>& root_bp,
            std::uint64_t level,
            std::map<patch_id_t, std::vector<geometry::any_map_t>>& out_maps
        )
        {
            for (const auto& [id, block] : skeleton) {
                std::vector<geometry::any_map_t> maps;

                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    maps.push_back(
                        create_dimension_map(root_bp, dd, id.coords[dd], level)
                    );
                }

                out_maps[id] = std::move(maps);
            }
        }

        static geometry::any_map_t create_dimension_map(
            const ecs::mesh_blueprint_t<Rank>& root_bp,
            std::uint64_t dim,
            std::int64_t block_topo_coord,
            std::uint64_t level
        )
        {
            using namespace geometry;

            real phys_start = root_bp.bounds[dim].first;
            real phys_end   = root_bp.bounds[dim].second;
            real global_len = phys_end - phys_start;

            std::int64_t level_factor = 1 << level;
            std::int64_t root_blocks  = 1;
            std::int64_t total_blocks = root_blocks * level_factor;

            real block_width_phys =
                global_len / static_cast<real>(total_blocks);
            real block_start_phys =
                phys_start +
                static_cast<real>(block_topo_coord) * block_width_phys;

            std::int64_t n_cells =
                root_bp.active_resolution[dim] * level_factor;

            auto map_type = deserialize<map_type_t>(root_bp.spacing[dim]);

            if (map_type == map_type_t::uniform) {
                real dx = block_width_phys / static_cast<real>(n_cells);
                return uniform_map_t(block_start_phys, dx);
            }
            else {
                real end_phys  = block_start_phys + block_width_phys;
                real log_slope = std::log10(end_phys / block_start_phys) /
                                 static_cast<real>(n_cells);
                return log_map_t(block_start_phys, log_slope);
            }
        }
    };

}   // namespace simbi::grid::creation

#endif
