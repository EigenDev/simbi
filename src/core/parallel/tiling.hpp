/**
 * tiling.hpp
 * strategies for domain decomposition
 */

#ifndef SIMBI_PARALLEL_TILING_HPP
#define SIMBI_PARALLEL_TILING_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/types/alias/alias.hpp"
#include "domain.hpp"
#include <cmath>
#include <memory>
#include <vector>

namespace simbi::parallel {

    /**
     * base class for tiling strategies
     */
    template <size_type Dims>
    class tiling_strategy_t
    {
      public:
        virtual ~tiling_strategy_t() = default;

        // divide domain into tiles
        virtual std::vector<domain_t<Dims>> create_tiles(
            const domain_t<Dims>& input_domain,
            const array_t<size_type, Dims>& halo_sizes
        ) const = 0;
    };

    /**
     * simple block decomposition strategy
     * divides domain into fixed-size blocks
     */
    template <size_type Dims>
    class blocks_strategy_t : public tiling_strategy_t<Dims>
    {
      public:
        explicit blocks_strategy_t(const array_t<size_type, Dims>& tile_size)
            : tile_size_(tile_size)
        {
        }

        std::vector<domain_t<Dims>> create_tiles(
            const domain_t<Dims>& input_domain,
            const array_t<size_type, Dims>& halo_sizes
        ) const override
        {
            std::vector<domain_t<Dims>> result;
            const auto& shape = input_domain.shape();

            // Calculate number of tiles in each dimension
            array_t<size_type, Dims> num_tiles;
            for (size_type i = 0; i < Dims; ++i) {
                num_tiles[i] = (shape[i] + tile_size_[i] - 1) / tile_size_[i];
            }

            // Create tiles
            array_t<size_type, Dims> tile_idx{};
            do {
                // Calculate start and end coordinates for this tile
                array_t<size_type, Dims> tile_start;
                array_t<size_type, Dims> tile_end;

                for (size_type i = 0; i < Dims; ++i) {
                    tile_start[i] = tile_idx[i] * tile_size_[i];
                    tile_end[i] =
                        std::min(tile_start[i] + tile_size_[i], shape[i]);
                }

                // Create the tile domain
                auto tile = input_domain.subregion(tile_start, tile_end);

                // Add halo regions
                auto expanded_tile = add_halo(tile, halo_sizes, input_domain);
                result.push_back(expanded_tile);

                // Advance to next tile
            } while (advance_index(tile_idx, num_tiles));

            return result;
        }

      private:
        // Increment multi-dimensional index with carry
        static constexpr bool advance_index(
            array_t<size_type, Dims>& idx,
            const array_t<size_type, Dims>& max_idx
        )
        {
            for (size_type dim = 0; dim < Dims; ++dim) {
                ++idx[dim];
                if (idx[dim] < max_idx[dim]) {
                    return true;
                }
                idx[dim] = 0;
            }
            return false;   // wrapped around completely
        }

        // Add halo to a tile, clamping at domain boundaries
        domain_t<Dims> add_halo(
            const domain_t<Dims>& tile,
            const array_t<size_type, Dims>& halo_sizes,
            const domain_t<Dims>& full_domain
        ) const
        {
            const auto tile_global_start = tile.offset();
            const auto tile_shape        = tile.shape();

            array_t<size_type, Dims> new_start;
            array_t<size_type, Dims> new_end;

            for (size_type i = 0; i < Dims; ++i) {
                // Clamp the halo at domain boundaries
                new_start[i] = tile_global_start[i] >= halo_sizes[i]
                                   ? tile_global_start[i] - halo_sizes[i]
                                   : 0;

                new_end[i] = std::min(
                    tile_global_start[i] + tile_shape[i] + halo_sizes[i],
                    full_domain.shape()[i]
                );
            }

            // Create a new domain with the expanded region
            return domain_t<Dims>(full_domain.subregion(new_start, new_end));
        }

        array_t<size_type, Dims> tile_size_;
    };

    /**
     * CPU cache-optimized tiling strategy
     */
    template <size_type Dims>
    class cpu_cache_strategy_t : public tiling_strategy_t<Dims>
    {
      public:
        explicit cpu_cache_strategy_t(
            size_type bytes_per_element = sizeof(real)
        )
            : bytes_per_element_(bytes_per_element)
        {
        }

        std::vector<domain_t<Dims>> create_tiles(
            const domain_t<Dims>& input_domain,
            const array_t<size_type, Dims>& halo_sizes
        ) const override
        {
            // Calculate optimal tile size based on L1/L2 cache size
            constexpr size_type target_cache_size =
                32 * 1024;   // Aim for 32KB tiles (typical L1 size)

            // Calculate optimal elements per tile
            size_type elements_per_tile =
                target_cache_size / bytes_per_element_;

            // Calculate tile dimensions for balanced tiles
            array_t<size_type, Dims> tile_size;
            size_type dim_size = static_cast<size_type>(
                std::pow(static_cast<double>(elements_per_tile), 1.0 / Dims)
            );

            for (size_type i = 0; i < Dims; ++i) {
                tile_size[i] = dim_size;
            }

            // Use block strategy with these tile sizes
            blocks_strategy_t<Dims> block_strategy_(tile_size);
            return block_strategy_.create_tiles(input_domain, halo_sizes);
        }

      private:
        size_type bytes_per_element_;
    };

    /**
     * GPU block-optimized tiling strategy
     */
    template <size_type Dims>
    class gpu_strategy_t : public tiling_strategy_t<Dims>
    {
      public:
        std::vector<domain_t<Dims>> create_tiles(
            const domain_t<Dims>& input_domain,
            const array_t<size_type, Dims>& halo_sizes
        ) const override
        {
            // For GPU, we want tiles that map well to thread blocks
            array_t<size_type, Dims> tile_size;

            if constexpr (Dims == 1) {
                tile_size[0] = 256;   // One thread block
            }
            else if constexpr (Dims == 2) {
                tile_size[0] = 32;   // Warp size
                tile_size[1] = 8;    // 256 threads per block
            }
            else if constexpr (Dims == 3) {
                tile_size[0] = 16;
                tile_size[1] = 8;
                tile_size[2] = 2;   // 256 threads per block
            }

            // Use block strategy with these tile sizes
            blocks_strategy_t<Dims> block_strategy(tile_size);
            return block_strategy.create_tiles(input_domain, halo_sizes);
        }
    };

    /**
     * Factory function to get optimal strategy for hardware
     */
    template <size_type Dims>
    inline std::unique_ptr<tiling_strategy_t<Dims>> get_optimal_strategy()
    {
        if constexpr (platform::is_gpu) {
            return std::make_unique<gpu_strategy_t<Dims>>();
        }
        else {
            return std::make_unique<cpu_cache_strategy_t<Dims>>();
        }
    }

}   // namespace simbi::parallel

#endif   // SIMBI_PARALLEL_TILING_HPP
