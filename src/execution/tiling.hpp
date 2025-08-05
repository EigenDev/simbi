#ifndef SIMBI_TILING_HPP
#define SIMBI_TILING_HPP

#include "base/concepts.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <span>
#include <vector>

namespace simbi::tiling {

    // forward declarations
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    struct tile_t;

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    struct tile_range_t;

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto tile_bounds(
        const domain_t<Dims>& domain,
        const iarray<Dims>& tile_index,
        const iarray<Dims>& tile_size
    ) noexcept -> domain_t<Dims>;

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto
    cpu_tile_size(std::size_t element_size = sizeof(real)) noexcept
        -> iarray<Dims>;

    // hardware target selection
    enum class hardware_target {
        cpu_cache,
        gpu_blocks,
        auto_detect
    };

    // simple tile representation - just a domain slice
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    struct tile_t {
        domain_t<Dims> domain;
        iarray<Dims> tile_index;   // which tile this is in the grid

        constexpr auto size() const noexcept { return domain.size(); }
        constexpr auto start() const noexcept { return domain.start; }
        constexpr auto end() const noexcept { return domain.end; }
    };

    // calculate optimal tile size for target hardware
    template <std::uint64_t Dims, typename T = real>
        requires valid_dimension<Dims>
    constexpr auto optimal_tile_size(
        std::size_t element_size = sizeof(T),
        hardware_target target   = hardware_target::auto_detect
    ) noexcept -> iarray<Dims>
    {
        if constexpr (global::on_gpu) {
            if (target == hardware_target::auto_detect ||
                target == hardware_target::gpu_blocks) {
                return gpu_tile_size<Dims>();
            }
        }

        // default to CPU cache optimization
        return cpu_tile_size<Dims>(element_size);
    }

    // CPU cache-optimized tile sizing
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto cpu_tile_size(std::size_t element_size) noexcept
        -> iarray<Dims>
    {
        // target L2 cache (typical 256KB-1MB), use conservative 256KB
        constexpr std::size_t target_cache_size = 256 * 1024;
        const std::size_t elements_per_tile = target_cache_size / element_size;

        if constexpr (Dims == 1) {
            return iarray<1>{static_cast<std::int64_t>(elements_per_tile)};
        }
        else if constexpr (Dims == 2) {
            // square tiles for better locality
            const auto side_length =
                static_cast<std::int64_t>(std::sqrt(elements_per_tile));
            return iarray<2>{side_length, side_length};
        }
        else if constexpr (Dims == 3) {
            // cube tiles
            const auto side_length =
                static_cast<std::int64_t>(std::cbrt(elements_per_tile));
            return iarray<3>{side_length, side_length, side_length};
        }
    }

    // GPU block-optimized tile sizing
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto gpu_tile_size() noexcept -> iarray<Dims>
    {
        if constexpr (Dims == 1) {
            return iarray<1>{1024};   // max threads per block
        }
        else if constexpr (Dims == 2) {
            return iarray<2>{32, 32};   // 1024 threads in 2D
        }
        else if constexpr (Dims == 3) {
            return iarray<3>{16, 16, 4};   // 1024 threads in 3D
        }
    }

    // create tiles from domain with specified tile size
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    auto make_tiles(const domain_t<Dims>& domain, const iarray<Dims>& tile_size)
        -> std::vector<tile_t<Dims>>
    {
        std::vector<tile_t<Dims>> tiles;

        // calculate number of tiles in each dimension
        iarray<Dims> num_tiles;
        for (std::size_t d = 0; d < Dims; ++d) {
            const auto extent = domain.end[d] - domain.start[d];
            num_tiles[d]      = (extent + tile_size[d] - 1) /
                           tile_size[d];   // Ceiling division
        }

        // generate all tile combinations
        const auto total_tiles = [&] {
            std::int64_t total = 1;
            for (std::size_t d = 0; d < Dims; ++d) {
                total *= num_tiles[d];
            }
            return total;
        }();

        tiles.reserve(total_tiles);

        // multi-dimensional tile iteration
        std::function<void(iarray<Dims>, std::size_t)> generate_tiles =
            [&](iarray<Dims> tile_idx, std::size_t dim) {
                if (dim == Dims) {
                    tiles.emplace_back(
                        tile_bounds(domain, tile_idx, tile_size),
                        tile_idx
                    );
                    return;
                }

                for (std::int64_t i = 0; i < num_tiles[dim]; ++i) {
                    tile_idx[dim] = i;
                    generate_tiles(tile_idx, dim + 1);
                }
            };

        iarray<Dims> start_idx{};   // zero-initialized
        generate_tiles(start_idx, 0);

        return tiles;
    }

    // create tiles with optimal size for target hardware
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    auto make_tiles(
        const domain_t<Dims>& domain,
        std::size_t element_size = sizeof(real),
        hardware_target target   = hardware_target::auto_detect
    ) -> std::vector<tile_t<Dims>>
    {
        const auto tile_size = optimal_tile_size<Dims>(element_size, target);
        return make_tiles(domain, tile_size);
    }

    // calc bounds for a specific tile
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto tile_bounds(
        const domain_t<Dims>& domain,
        const iarray<Dims>& tile_index,
        const iarray<Dims>& tile_size
    ) noexcept -> domain_t<Dims>
    {
        iarray<Dims> start, end;

        for (std::size_t d = 0; d < Dims; ++d) {
            start[d] = domain.start[d] + tile_index[d] * tile_size[d];
            end[d]   = std::min(start[d] + tile_size[d], domain.end[d]);
        }

        return domain_t<Dims>{start, end};
    }

    // add ghost zones to tiles
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    auto add_ghost_zones(
        std::span<const tile_t<Dims>> tiles,
        const iarray<Dims>& halo_width,
        const domain_t<Dims>& global_domain
    ) -> std::vector<tile_t<Dims>>
    {
        std::vector<tile_t<Dims>> ghost_tiles;
        ghost_tiles.reserve(tiles.size());

        for (const auto& tile : tiles) {
            iarray<Dims> ghost_start, ghost_end;

            for (std::size_t d = 0; d < Dims; ++d) {
                ghost_start[d] = std::max(
                    tile.start()[d] - halo_width[d],
                    global_domain.start[d]
                );
                ghost_end[d] = std::min(
                    tile.end()[d] + halo_width[d],
                    global_domain.end[d]
                );
            }

            domain_t<Dims> ghost_domain{ghost_start, ghost_end};
            ghost_tiles.emplace_back(ghost_domain, tile.tile_index);
        }

        return ghost_tiles;
    }

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    struct tile_range_t {
        domain_t<Dims> domain_;
        iarray<Dims> tile_size_;

        class iterator
        {
            domain_t<Dims> domain_;
            iarray<Dims> tile_size_;
            iarray<Dims> tile_index_;
            iarray<Dims> max_indices_;
            bool at_end_;

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = tile_t<Dims>;
            using difference_type   = std::ptrdiff_t;
            using pointer           = void;
            using reference         = value_type;

            constexpr iterator() noexcept : at_end_(true) {}

            constexpr iterator(
                const domain_t<Dims>& domain,
                const iarray<Dims>& tile_size,
                const iarray<Dims>& start_idx
            ) noexcept
                : domain_(domain),
                  tile_size_(tile_size),
                  tile_index_(start_idx),
                  at_end_(false)
            {

                // calculate max indices for bounds checking
                for (std::size_t d = 0; d < Dims; ++d) {
                    const auto extent = domain.end[d] - domain.start[d];
                    max_indices_[d] =
                        (extent + tile_size[d] - 1) / tile_size[d];
                }

                // check if we start at end
                if (tile_index_[0] >= max_indices_[0]) {
                    at_end_ = true;
                }
            }

            constexpr value_type operator*() const noexcept
            {
                const auto tile_domain =
                    tile_bounds(domain_, tile_index_, tile_size_);
                return tile_t<Dims>{tile_domain, tile_index_};
            }

            constexpr iterator& operator++() noexcept
            {
                if (at_end_) {
                    return *this;
                }

                // increment multi-dimensional index
                for (std::size_t d = Dims - 1; d != SIZE_MAX; --d) {
                    ++tile_index_[d];
                    if (tile_index_[d] < max_indices_[d]) {
                        return *this;
                    }
                    tile_index_[d] = 0;
                }

                at_end_ = true;
                return *this;
            }

            constexpr iterator operator++(int) noexcept
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }

            constexpr bool operator==(const iterator& other) const noexcept
            {
                return at_end_ == other.at_end_ &&
                       (at_end_ || tile_index_ == other.tile_index_);
            }

            constexpr bool operator!=(const iterator& other) const noexcept
            {
                return !(*this == other);
            }
        };

        constexpr iterator begin() const noexcept
        {
            return iterator{domain_, tile_size_, iarray<Dims>{}};
        }

        constexpr iterator end() const noexcept { return iterator{}; }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto tile_range(
        const domain_t<Dims>& domain,
        const iarray<Dims>& tile_size
    ) noexcept -> tile_range_t<Dims>
    {
        return tile_range_t<Dims>{domain, tile_size};
    }

    // overload with auto tile sizing
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    constexpr auto tile_range(
        const domain_t<Dims>& domain,
        std::size_t element_size = sizeof(real),
        hardware_target target   = hardware_target::auto_detect
    ) noexcept -> tile_range_t<Dims>
    {
        const auto tile_size = optimal_tile_size<Dims>(element_size, target);
        return tile_range_t<Dims>{domain, tile_size};
    }

    template <std::uint64_t Dims, typename Func>
        requires valid_dimension<Dims> &&
                 std::invocable<Func, const tile_t<Dims>&>
    void for_each_tile(
        const domain_t<Dims>& domain,
        Func&& func,
        const iarray<Dims>& tile_size
    )
    {
        tile_range(domain, tile_size) | fp::for_each(std::forward<Func>(func));
    }

    template <std::uint64_t Dims, typename Func>
        requires valid_dimension<Dims> &&
                 std::invocable<Func, const tile_t<Dims>&>
    void for_each_tile(
        const domain_t<Dims>& domain,
        Func&& func,
        std::size_t element_size = sizeof(real),
        hardware_target target   = hardware_target::auto_detect
    )
    {
        tile_range(domain, element_size, target) |
            fp::for_each(std::forward<Func>(func));
    }

}   // namespace simbi::tiling

#endif   // SIMBI_TILING_HPP
