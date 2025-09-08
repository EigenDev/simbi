#ifndef SIMBI_TILING_HPP
#define SIMBI_TILING_HPP

#include "base/concepts.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace simbi::tiling {
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    struct tile_t {
        domain_t<Dims> domain;
        iarray<Dims> tile_index;

        constexpr auto size() const noexcept { return domain.size(); }
        constexpr auto start() const noexcept { return domain.start; }
        constexpr auto end() const noexcept { return domain.end; }
    };

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto default_tile_size() noexcept -> iarray<Dims>
    {
        if constexpr (Dims == 1) {
            return iarray<1>{1024};
        }
        else if constexpr (Dims == 2) {
            return iarray<2>{128, 128};
        }
        else if constexpr (Dims == 3) {
            return iarray<3>{16, 16, 16};
        }
    }

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto tile_bounds(
        const domain_t<Dims>& domain,
        const iarray<Dims>& tile_index,
        const iarray<Dims>& tile_size
    ) noexcept -> domain_t<Dims>
    {
        iarray<Dims> start, end;

        for (std::size_t dd = 0; dd < Dims; ++dd) {
            start[dd] = domain.start[dd] + tile_index[dd] * tile_size[dd];
            end[dd]   = std::min(start[dd] + tile_size[dd], domain.end[dd]);
        }

        return domain_t<Dims>{start, end};
    }

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    auto make_tiles(
        const domain_t<Dims>& domain,
        const iarray<Dims>& tile_size = default_tile_size<Dims>()
    ) -> std::vector<tile_t<Dims>>
    {
        iarray<Dims> num_tiles;
        for (std::size_t dd = 0; dd < Dims; ++dd) {
            const auto extent = domain.end[dd] - domain.start[dd];
            num_tiles[dd]     = (extent + tile_size[dd] - 1) / tile_size[dd];
        }

        std::int64_t total_tiles = 1;
        for (std::size_t dd = 0; dd < Dims; ++dd) {
            total_tiles *= num_tiles[dd];
        }

        std::vector<tile_t<Dims>> tiles;
        tiles.reserve(total_tiles);

        // generate tiles with simple nested loops
        if constexpr (Dims == 1) {
            for (std::int64_t ii = 0; ii < num_tiles[0]; ++ii) {
                iarray<1> tile_idx{ii};
                tiles.emplace_back(
                    tile_bounds(domain, tile_idx, tile_size),
                    tile_idx
                );
            }
        }
        else if constexpr (Dims == 2) {
            for (std::int64_t ii = 0; ii < num_tiles[0]; ++ii) {
                for (std::int64_t jj = 0; jj < num_tiles[1]; ++jj) {
                    iarray<2> tile_idx{ii, jj};
                    tiles.emplace_back(
                        tile_bounds(domain, tile_idx, tile_size),
                        tile_idx
                    );
                }
            }
        }
        else if constexpr (Dims == 3) {
            for (std::int64_t ii = 0; ii < num_tiles[0]; ++ii) {
                for (std::int64_t jj = 0; jj < num_tiles[1]; ++jj) {
                    for (std::int64_t kk = 0; kk < num_tiles[2]; ++kk) {
                        iarray<3> tile_idx{ii, jj, kk};
                        tiles.emplace_back(
                            tile_bounds(domain, tile_idx, tile_size),
                            tile_idx
                        );
                    }
                }
            }
        }

        return tiles;
    }

    template <std::uint64_t Dims, typename Func>
        requires valid_dimension<Dims>
    void for_each_tile(
        const domain_t<Dims>& domain,
        Func&& func,
        const iarray<Dims>& tile_size = default_tile_size<Dims>()
    )
    {
        const auto tiles = make_tiles(domain, tile_size);
        for (const auto& tile : tiles) {
            func(tile);
        }
    }

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    auto add_ghost_zones(
        const std::vector<tile_t<Dims>>& tiles,
        const iarray<Dims>& halo_width,
        const domain_t<Dims>& global_domain
    ) -> std::vector<tile_t<Dims>>
    {
        std::vector<tile_t<Dims>> ghost_tiles;
        ghost_tiles.reserve(tiles.size());

        for (const auto& tile : tiles) {
            iarray<Dims> ghost_start, ghost_end;

            for (std::size_t dd = 0; dd < Dims; ++dd) {
                ghost_start[dd] = std::max(
                    tile.start()[dd] - halo_width[dd],
                    global_domain.start[dd]
                );
                ghost_end[dd] = std::min(
                    tile.end()[dd] + halo_width[dd],
                    global_domain.end[dd]
                );
            }

            domain_t<Dims> ghost_domain{ghost_start, ghost_end};
            ghost_tiles.emplace_back(ghost_domain, tile.tile_index);
        }

        return ghost_tiles;
    }

}   // namespace simbi::tiling

#endif   // SIMBI_TILING_HPP
