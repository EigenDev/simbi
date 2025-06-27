#ifndef SIMBI_UTILITY_DOMAIN_HPP
#define SIMBI_UTILITY_DOMAIN_HPP

#include "config.hpp"
#include "coordinate.hpp"
#include <array>
#include <cassert>
#include <cstddef>

namespace simbi::base {
    // =============================================================================
    // Domain and Slicing System
    // =============================================================================
    template <std::uint64_t Dims>
    struct domain_t {
        std::array<std::size_t, Dims> extents;

        domain_t() = default;

        constexpr domain_t(const std::array<std::size_t, Dims>& ext)
            : extents(ext)
        {
        }

        // convenience constructors
        constexpr domain_t(std::size_t nx)
            requires(Dims == 1)
            : extents{nx}
        {
        }
        constexpr domain_t(std::size_t nx, std::size_t ny)
            requires(Dims == 2)
            : extents{nx, ny}
        {
        }
        constexpr domain_t(std::size_t nx, std::size_t ny, std::size_t nz)
            requires(Dims == 3)
            : extents{nx, ny, nz}
        {
        }

        DUAL constexpr std::size_t extent(std::size_t dim) const
        {
            return extents[dim];
        }

        DUAL constexpr std::size_t total_size() const
        {
            std::size_t total = 1;
            for (std::size_t i = 0; i < Dims; ++i) {
                total *= extents[i];
            }
            return total;
        }

        // Original linear indexing functions
        DUAL constexpr std::size_t
        linear_index(const std::array<std::size_t, Dims>& point) const
        {
            std::size_t idx = 0, stride = 1;
            for (std::int64_t i = Dims - 1; i >= 0; --i) {
                idx += point[i] * stride;
                stride *= extents[i];
            }
            return idx;
        }

        // New functions for coordinate compatibility
        DUAL constexpr std::size_t
        linear_index(const base::coordinate_t<Dims>& coord) const
        {
            std::size_t idx = 0, stride = 1;
            for (std::int64_t i = Dims - 1; i >= 0; --i) {
                // Ensure coordinate is non-negative
                assert(coord[i] >= 0);
                idx += static_cast<std::size_t>(coord[i]) * stride;
                stride *= extents[i];
            }
            return idx;
        }

        // Check if a coordinate is valid within this domain
        DUAL constexpr bool
        contains(const base::coordinate_t<Dims>& coord) const
        {
            for (std::size_t i = 0; i < Dims; ++i) {
                if (coord[i] < 0 ||
                    static_cast<std::size_t>(coord[i]) >= extents[i]) {
                    return false;
                }
            }
            return true;
        }

        // Convert domain postd::int64_t to coordinate
        DUAL constexpr base::coordinate_t<Dims>
        to_coordinate(const std::array<std::size_t, Dims>& point) const
        {
            base::coordinate_t<Dims> coord;
            for (std::size_t i = 0; i < Dims; ++i) {
                coord[i] = static_cast<std::int64_t>(point[i]);
            }
            return coord;
        }

        // Convert coordinate to domain point
        DUAL constexpr std::array<std::size_t, Dims>
        from_coordinate(const base::coordinate_t<Dims>& coord) const
        {
            std::array<std::size_t, Dims> point;
            for (std::size_t i = 0; i < Dims; ++i) {
                assert(coord[i] >= 0);
                point[i] = static_cast<std::size_t>(coord[i]);
            }
            return point;
        }
    };

    template <std::uint64_t Dims>
    struct slice_t {
        std::array<size_t, Dims> start;
        std::array<size_t, Dims> end;   // exclusive
        std::array<size_t, Dims> stride;

        constexpr slice_t(
            const std::array<size_t, Dims>& s,
            const std::array<size_t, Dims>& e
        )
            : start(s), end(e)
        {
            std::fill(stride.begin(), stride.end(), 1);
        }

        constexpr slice_t(
            const std::array<size_t, Dims>& s,
            const std::array<size_t, Dims>& e,
            const std::array<size_t, Dims>& str
        )
            : start(s), end(e), stride(str)
        {
        }

        DUAL constexpr bool
        contains(const std::array<size_t, Dims>& point) const
        {
            for (size_t i = 0; i < Dims; ++i) {
                if (point[i] < start[i] || point[i] >= end[i]) {
                    return false;
                }
                if ((point[i] - start[i]) % stride[i] != 0) {
                    return false;
                }
            }
            return true;
        }

        DUAL constexpr size_t total_size() const
        {
            size_t total = 1;
            for (size_t i = 0; i < Dims; ++i) {
                total *= (end[i] - start[i] + stride[i] - 1) / stride[i];
            }
            return total;
        }

        DUAL constexpr std::array<size_t, Dims> extents() const
        {
            std::array<size_t, Dims> ext;
            for (size_t i = 0; i < Dims; ++i) {
                ext[i] = (end[i] - start[i] + stride[i] - 1) / stride[i];
            }
            return ext;
        }
    };

    // convenience slice constructors
    template <std::uint64_t Dims>
    constexpr auto interior_domain(const domain_t<Dims>& domain)
    {
        std::array<size_t, Dims> start, end;
        for (size_t i = 0; i < Dims; ++i) {
            start[i] = 1;
            end[i]   = domain.extent(i) - 1;
        }
        return slice_t<Dims>{start, end};
    }

    template <std::uint64_t Dims>
    constexpr auto
    boundary_slice(const domain_t<Dims>& domain, size_t dim, size_t face)
    {
        std::array<size_t, Dims> start, end;
        for (size_t i = 0; i < Dims; ++i) {
            if (i == dim) {
                start[i] = (face == 0) ? 0 : domain.extent(i) - 1;
                end[i]   = start[i] + 1;
            }
            else {
                start[i] = 0;
                end[i]   = domain.extent(i);
            }
        }
        return slice_t<Dims>{start, end};
    }

    using domain1d_t = domain_t<1>;
    using domain2d_t = domain_t<2>;
    using domain3d_t = domain_t<3>;

    using slice1d_t = slice_t<1>;
    using slice2d_t = slice_t<2>;
    using slice3d_t = slice_t<3>;
}   // namespace simbi::base

#endif   // SIMBI_UTILITY_DOMAIN_HPP
