// =============================================================================
// domain.hpp
//
// minimal n-dimensional domain abstraction for xpu parallel dispatch.
// represents half-open interval [start, end) with linear indexing support
// for grid-stride cuda kernels.
//
// design principles:
//   - minimal surface area (only what phase 3 needs)
//   - compile-time rank (template parameter)
//   - linear ↔ multi-dimensional index conversion
//   - constexpr for device-side use
//
// usage:
//   auto domain = xpu::extents<3>({100, 200, 50});
//   executor.dispatch(domain, [=](auto idx) { /* work */ });
//
// integration note:
//   when integrating with simbi, this will be replaced by or adapted to
//   simbi::grid::domain_t which has additional features (boundaries, slicing,
//   intersection). for now, this minimal version proves the dispatch pattern.
// =============================================================================

#pragma once

#include <array>
#include <cstdint>

namespace xpu {

    // =============================================================================
    // domain_t - n-dimensional iteration space
    // =============================================================================

    template <std::uint64_t Rank>
    struct domain_t
    {
        using coord_t = std::array<std::int64_t, Rank>;

        coord_t start;
        coord_t end;

        // =============================================================================
        // geometric queries
        // =============================================================================

        constexpr coord_t shape() const
        {
            coord_t result{};
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                result[ii] = end[ii] - start[ii];
            }
            return result;
        }

        constexpr std::int64_t size() const
        {
            std::int64_t result = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                result *= (end[ii] - start[ii]);
            }
            return result;
        }

        constexpr bool empty() const
        {
            return size() <= 0;
        }

        // =============================================================================
        // indexing conversions (critical for grid-stride kernels)
        // =============================================================================

        // convert linear index to multi-dimensional coordinate
        // uses row-major ordering (last dimension varies fastest)
        constexpr coord_t linear_to_coord(std::int64_t linear) const
        {
            coord_t coord{};
            for (std::int64_t ii = Rank - 1; ii >= 0; --ii) {
                auto dim_size = end[ii] - start[ii];
                coord[ii]     = start[ii] + (linear % dim_size);
                linear /= dim_size;
            }
            return coord;
        }

        // convert multi-dimensional coordinate to linear index
        constexpr std::int64_t coord_to_linear(const coord_t& coord) const
        {
            std::int64_t linear = 0;
            std::int64_t stride = 1;
            for (std::int64_t ii = Rank - 1; ii >= 0; --ii) {
                linear += (coord[ii] - start[ii]) * stride;
                stride *= (end[ii] - start[ii]);
            }
            return linear;
        }

        // =============================================================================
        // comparison operators
        // =============================================================================

        constexpr bool operator==(const domain_t& other) const
        {
            return start == other.start && end == other.end;
        }

        constexpr bool operator!=(const domain_t& other) const
        {
            return !(*this == other);
        }
    };

    // =============================================================================
    // factory functions
    // =============================================================================

    // create domain from shape (starts at origin)
    template <std::uint64_t Rank>
    constexpr domain_t<Rank> extents(std::array<std::int64_t, Rank> shape)
    {
        domain_t<Rank> result;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            result.start[ii] = 0;
            result.end[ii]   = shape[ii];
        }
        return result;
    }

    // create domain with explicit start and end
    template <std::uint64_t Rank>
    constexpr domain_t<Rank>
    make_domain(std::array<std::int64_t, Rank> start, std::array<std::int64_t, Rank> end)
    {
        return domain_t<Rank>{start, end};
    }

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using domain1d = domain_t<1>;
    using domain2d = domain_t<2>;
    using domain3d = domain_t<3>;

} // namespace xpu
