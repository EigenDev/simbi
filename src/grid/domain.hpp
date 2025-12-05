#ifndef GRID_DOMAIN_HPP
#define GRID_DOMAIN_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"

#include <cstdint>
#include <iostream>
#include <ostream>

namespace simbi::grid {
    // domain: pure topology defined by integer vectors
    // represents the half-open interval [start, end)
    template <std::uint64_t Rank>
    struct domain_t
    {
        // strict 64-bit indexing
        using coord_t = vector_t<std::int64_t, Rank>;
        // local geometry
        coord_t start;
        coord_t fin;

        // geometric queries
        DUAL constexpr coord_t shape() const
        {
            return fin - start;
        }
        DUAL constexpr std::int64_t size() const
        {
            return fp::product(shape());
        }
        DUAL constexpr bool empty() const
        {
            return size() <= 0;
        }

        DUAL constexpr bool contains(const coord_t& c) const
        {
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                if (c[ii] < start[ii] || c[ii] >= fin[ii]) {
                    return false;
                }
            }
            return true;
        }

        // intersection logic
        // purely geometric clip, preserves boundaries of 'this'
        DUAL constexpr domain_t intersect(const domain_t& other) const
        {
            domain_t res = *this;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                res.start[ii] = (start[ii] > other.start[ii]) ? start[ii] : other.start[ii];
                res.fin[ii]   = (fin[ii] < other.fin[ii]) ? fin[ii] : other.fin[ii];
            }
            return res;
        }

        DUAL constexpr domain_t contract(std::int64_t width) const
        {
            domain_t res = *this;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                res.start[ii] += width;
                res.fin[ii] -= width;
            }
            return res;
        }

        constexpr auto linear_to_coord(std::uint64_t linear) const
        {
            iarray<Rank> coord{};
            for (std::int64_t ii = Rank - 1; ii >= 0; --ii) {
                auto dim_size = fin[ii] - start[ii];
                coord[ii]     = start[ii] + (linear % dim_size);
                linear /= dim_size;
            }
            return coord;
        }

        constexpr auto coord_to_linear(const iarray<Rank>& coord) const
        {
            std::uint64_t linear = 0;
            for (std::int64_t ii = Rank - 1; ii >= 0; --ii) {
                linear *= (fin[ii] - start[ii]);
                linear += coord[ii] - start[ii];
            }
            return linear;
        }

        // mitosis: the critical logic for decomposition
        // returns a new domain clipped to [split_start, split_end)
        // automatically downgrades boundaries to 'partition' if detached from
        // global edge
        DUAL constexpr domain_t
        slice(std::int64_t axis, std::int64_t split_start, std::int64_t split_fin) const
        {
            domain_t shard = *this;

            // geometric clip
            shard.start[axis] = (start[axis] > split_start) ? start[axis] : split_start;
            shard.fin[axis]   = (fin[axis] < split_fin) ? fin[axis] : split_fin;

            return shard;
        }

        DUAL constexpr bool operator==(const domain_t& other) const
        {
            return (start == other.start) && (fin == other.fin);
        }
        DUAL constexpr bool operator!=(const domain_t& other) const
        {
            return !(*this == other);
        }
    };

    template <std::uint64_t Rank>
    constexpr auto extents(const iarray<Rank>& shape)
    {
        return domain_t<Rank>{iarray<Rank>{}, shape};
    }

    template <std::uint64_t Rank>
    std::ostream& operator<<(std::ostream& os, const domain_t<Rank>& d)
    {
        os << "Domain(";
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            os << d.start[ii] << ":" << d.fin[ii];
            if constexpr (Rank > 1) {
                if (ii < Rank - 1) {
                    os << ", ";
                }
            }
        }
        os << ")";
        return os;
    }

} // namespace simbi::grid

#endif // GRID_DOMAIN_HPP
