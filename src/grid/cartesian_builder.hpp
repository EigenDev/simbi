// =============================================================================
// cartesian_builder.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "block_info.hpp"
#include "boundary.hpp"
#include "connectivity.hpp"
#include "decomposition.hpp"
#include "domain.hpp"
#include "patch_id.hpp"
#include "skeleton.hpp"

#include <cstdint>

namespace simbi::grid {

    // a stateless service class that builds skeletons for cartesian grids
    struct cartesian_builder_t {

        template <std::uint64_t Rank>
        static void build(
            skeleton_t<Rank>& skeleton,
            const domain_t<Rank>& global_domain,
            const topology_t& topo,
            std::uint64_t my_rank,
            const boundary_set_t<Rank>& global_bcs
        )
        {
            // decompose geometry
            domain_t<Rank> local_geom =
                decomposer_t::decompose(global_domain, topo, my_rank);

            // construct identity
            patch_id_t my_id;
            my_id.level  = 0;
            my_id.coords = topo.coords(my_rank);

            // construct block info
            block_info_t<Rank> info;
            info.id       = my_id;
            info.geometry = local_geom;

            // 4. resolve connectivity
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                std::int64_t coord     = my_id.coords[dd];
                std::int64_t max_coord = topo.dims[dd];

                // left face
                if (coord > 0) {
                    patch_id_t neighbor = my_id;
                    neighbor.coords[dd]--;
                    info.connect(dd, side_t::left, neighbor);
                }
                else {
                    if (global_bcs.left(dd) == boundary_type_t::periodic) {
                        patch_id_t neighbor = my_id;
                        neighbor.coords[dd] = max_coord - 1;
                        info.connect(dd, side_t::left, neighbor);
                    }
                    else {
                        info.set_boundary(
                            dd,
                            side_t::left,
                            global_bcs.left(dd)
                        );
                    }
                }

                // right face
                if (coord < max_coord - 1) {
                    patch_id_t neighbor = my_id;
                    neighbor.coords[dd]++;
                    info.connect(dd, side_t::right, neighbor);
                }
                else {
                    if (global_bcs.right(dd) == boundary_type_t::periodic) {
                        patch_id_t neighbor = my_id;
                        neighbor.coords[dd] = 0;
                        info.connect(dd, side_t::right, neighbor);
                    }
                    else {
                        info.set_boundary(
                            dd,
                            side_t::right,
                            global_bcs.right(dd)
                        );
                    }
                }
            }

            // 5. add to skeleton
            skeleton.add_block(info);
        }
    };

}   // namespace simbi::grid


