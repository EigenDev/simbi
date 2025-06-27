/**
 * block_tiling.hpp
 * simple block-based domain decomposition
 */

#ifndef SIMBI_CORE_PARALLEL_BLOCK_TILING_HPP
#define SIMBI_CORE_PARALLEL_BLOCK_TILING_HPP

#include "core/base/concepts.hpp"
#include "core/base/coordinate.hpp"
#include "core/mesh/grid_topology.hpp"
#include "system/parallel/tiling/tiling_strategy.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace simbi::parallel {

    /**
     * block_tiling_t - decomposes domain into fixed-size blocks
     */
    template <std::uint64_t Dims>
        requires concepts::valid_dimension<Dims>
    class block_tiling_t : public tiling_strategy_t<Dims>
    {
      public:
        // construct with specific block sizes
        explicit block_tiling_t(const base::coordinate_t<Dims>& block_sizes);

        // create with fixed size in all dimensions
        static std::shared_ptr<block_tiling_t<Dims>>
        create_uniform(std::size_t block_size);

        // divide topology into tiles
        std::vector<mesh::grid_topology_t<Dims>> create_tiles(
            const mesh::grid_topology_t<Dims>& topology,
            const base::coordinate_t<Dims>& ghost_zones = {}
        ) const override
        {
            std::vector<mesh::grid_topology_t<Dims>> tiles;

            auto extent = topology.physical_extent();
            base::coordinate_t<Dims> current_origin =
                topology.physical_origin();

            // recursive tile generation
            std::function<void(base::coordinate_t<Dims>&, std::size_t)>
                generate_tiles = [&](base::coordinate_t<Dims>& origin,
                                     std::size_t dim) {
                    if (dim == Dims) {
                        // create tile with ghost zones
                        auto tile_extent = block_sizes_;
                        for (std::size_t i = 0; i < Dims; ++i) {
                            tile_extent[i] = std::min(
                                tile_extent[i],
                                extent[i] -
                                    (origin[i] - topology.physical_origin()[i])
                            );
                        }

                        tiles.emplace_back(origin, tile_extent, ghost_zones);
                        return;
                    }

                    for (std::int64_t pos = origin[dim];
                         pos < topology.physical_origin()[dim] + extent[dim];
                         pos += block_sizes_[dim]) {
                        origin[dim] = pos;
                        generate_tiles(origin, dim + 1);
                    }
                };

            generate_tiles(current_origin, 0);
            return tiles;
        }

      private:
        base::coordinate_t<Dims> block_sizes_;

        // helper to add ghost zones to a tile
        mesh::grid_topology_t<Dims> add_ghost_zones(
            const mesh::grid_topology_t<Dims>& tile,
            const base::coordinate_t<Dims>& ghost_zones,
            const mesh::grid_topology_t<Dims>& full_topology
        ) const;

        // helper to advance multi-dimensional index
        bool advance_index(
            base::coordinate_t<Dims>& idx,
            const base::coordinate_t<Dims>& max_idx
        ) const;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_BLOCK_TILING_HPP
