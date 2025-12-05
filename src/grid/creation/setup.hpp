#ifndef GRID_CREATION_SETUP_HPP
#define GRID_CREATION_SETUP_HPP

#include "containers/vector.hpp"
#include "ecs/blueprints.hpp"
#include "geometry/api.hpp"
#include "grid/block_info.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/mesh_config.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "utility/bimap.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::grid::creation {
    // -------------------------------------------------------------------------
    // mesh configuration builder
    // maps blueprint -> mesh_config_t
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct mesh_setup_t
    {

        static mesh_config_t<Rank> create_config(const ecs::mesh_blueprint_t<Rank>& bp)
        {
            using namespace simbi::geometry;
            mesh_config_t<Rank> config;

            // topology
            config.global_cells = bp.active_resolution;
            config.halo_width   = bp.halo_width;

            // boundaries
            // blueprint provides a vector of strings.
            // we map them to the [dim][side] structure.
            // order is: x3_L, x13_R, x2_L, x2_R...
            const auto& bc_strs = bp.boundary_conditions;
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                // map input index 0 -> dimension (Rank-1)
                // example 3d: idx 0 -> dim 2 (z)
                // example 2d: idx 0 -> dim 1 (y)

                // vector is packed [left, right, left, right...]
                std::size_t vec_offset = dd * 2;

                auto left_type  = deserialize<boundary_type_t>(bc_strs[vec_offset]);
                auto right_type = deserialize<boundary_type_t>(bc_strs[vec_offset + 1]);

                config.boundaries.set_left(dd, left_type);
                config.boundaries.set_right(dd, right_type);
            }

            config.geometry.metric = deserialize<metric_type_t>(bp.coord_system);

            // geometry configuration
            // we unpack pairs into the config vectors
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                if (dd < bp.bounds.size()) {
                    auto               start = bp.bounds[dd].first;
                    auto               end   = bp.bounds[dd].second;
                    auto               stype = deserialize<map_type_t>(bp.spacing[dd]);
                    dimension_config_t dim_conf{stype, start, end};
                    config.geometry.dims.push_back(dim_conf);
                }
            }

            // resolution per block
            config.geometry.block_size_cells = bp.active_resolution;

            // motion
            config.motion.enabled    = bp.moving_mesh;
            config.motion.homologous = bp.homologous_expansion;

            return config;
        }
    };

    // -------------------------------------------------------------------------
    // skeleton builder (initial topology)
    // creates the initial block layout (single block or simple decomposition)
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct skeleton_builder_t
    {

        static skeleton_t<Rank> build_single_block(const mesh_config_t<Rank>& config)
        {
            skeleton_t<Rank> skeleton;

            // define identity
            // root level (0), origin coordinates (0,0,0)
            patch_id_t id;
            id.level = 0;
            id.coords.fill(0);

            // define geometry (domain)
            // for a single block, it owns the entire global index space
            block_info_t<Rank> block;
            block.id       = id;
            block.geometry = domain_t<Rank>{
                iarray<Rank>{},     // start at 0
                config.global_cells // end at N
            };

            // apply physical boundaries
            // since this is a single block, ALL faces are physical boundaries
            // unless they are periodic (which is also handled as a BC type
            // here)
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                block.set_boundary(dd, side_t::left, config.boundaries.left(dd));
                block.set_boundary(dd, side_t::right, config.boundaries.right(dd));

                // attach metric info for spherical theta boundaries
                if constexpr (Rank >= 2) {
                    constexpr std::uint64_t theta_array_dim = Rank - 2;
                    if (dd == theta_array_dim &&
                        config.geometry.metric == geometry::metric_type_t::spherical) {
                        // theta is x2 in logical coords, at array index Rank-2
                        real theta_min = config.geometry.dims[theta_array_dim].start;
                        real theta_max = config.geometry.dims[theta_array_dim].end;

                        // attach metric info to both left and right theta faces
                        block.get_face(dd, side_t::left).set_metric_info(theta_min, theta_max);
                        block.get_face(dd, side_t::right).set_metric_info(theta_min, theta_max);
                    }
                }
            }

            skeleton.add_block(block);
            return skeleton;
        }
    };

} // namespace simbi::grid::creation

#endif // GRID_CREATION_SETUP_HPP
