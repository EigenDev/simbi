#ifndef GRID_CREATION_TOPOLOGY_BUILDER_HPP
#define GRID_CREATION_TOPOLOGY_BUILDER_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "ecs/blueprints.hpp"
#include "grid/block_info.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "setup.hpp"

#include <cstdint>
#include <vector>

namespace simbi::grid::creation {

    template <std::uint64_t Rank>
    struct topology_builder_t {

        static std::vector<skeleton_t<Rank>> build_hierarchy(
            const ecs::mesh_blueprint_t<Rank>& root_bp,
            const ecs::amr_blueprint_t& amr_bp
        )
        {
            std::vector<skeleton_t<Rank>> hierarchy;

            // level 0: root
            auto root_config = mesh_setup_t<Rank>::create_config(root_bp);
            hierarchy.push_back(
                skeleton_builder_t<Rank>::build_single_block(root_config)
            );

            if (!amr_bp.enabled || amr_bp.max_levels <= 1) {
                return hierarchy;
            }

            // build refined levels iteratively
            for (std::uint64_t lvl = 1; lvl < amr_bp.max_levels; ++lvl) {
                std::uint64_t ratio =
                    (lvl - 1 < amr_bp.refinement_ratios.size())
                        ? amr_bp.refinement_ratios[lvl - 1]
                        : 2;

                const auto& region = amr_bp.static_refinement_regions[lvl - 1];

                hierarchy.push_back(
                    build_refined_level(root_bp, region, ratio, lvl)
                );
            }

            return hierarchy;
        }

      private:
        static skeleton_t<Rank> build_refined_level(
            const ecs::mesh_blueprint_t<Rank>& root_bp,
            const std::vector<real>& refinement_region,
            std::uint64_t refinement_ratio,
            std::uint64_t child_level
        )
        {
            skeleton_t<Rank> child_skeleton;

            // convert physical bounds to parent index space
            auto parent_domain = physical_to_parent_domain(
                refinement_region,
                root_bp,
                child_level - 1
            );

            // scale to child index space
            auto child_domain =
                scale_domain_up(parent_domain, refinement_ratio);

            // create block
            patch_id_t id;
            id.level = child_level;
            id.coords.fill(0);

            block_info_t<Rank> block;
            block.id       = id;
            block.geometry = child_domain;

            // boundaries: internal to parent (will be filled by prolongation)
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                block
                    .set_boundary(dd, side_t::left, boundary_type_t::partition);
                block.set_boundary(
                    dd,
                    side_t::right,
                    boundary_type_t::partition
                );
            }

            child_skeleton.add_block(block);
            return child_skeleton;
        }

        static domain_t<Rank> physical_to_parent_domain(
            const std::vector<real>& bounds,
            const ecs::mesh_blueprint_t<Rank>& root_bp,
            std::uint64_t parent_level
        )
        {
            iarray<Rank> start, fin;

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                real phys_start = bounds[2 * dd];
                real phys_end   = bounds[2 * dd + 1];

                real domain_start = root_bp.bounds[dd].first;
                real domain_end   = root_bp.bounds[dd].second;
                real domain_len   = domain_end - domain_start;

                std::int64_t root_cells   = root_bp.active_resolution[dd];
                std::int64_t parent_cells = root_cells * (1 << parent_level);

                real cell_size = domain_len / static_cast<real>(parent_cells);

                start[dd] = static_cast<std::int64_t>(
                    (phys_start - domain_start) / cell_size
                );
                fin[dd] = static_cast<std::int64_t>(
                    (phys_end - domain_start) / cell_size
                );
            }

            return domain_t<Rank>{start, fin};
        }

        static domain_t<Rank>
        scale_domain_up(const domain_t<Rank>& d, std::uint64_t ratio)
        {
            return domain_t<Rank>{
              d.start * static_cast<std::int64_t>(ratio),
              d.fin * static_cast<std::int64_t>(ratio)
            };
        }
    };

}   // namespace simbi::grid::creation

#endif
