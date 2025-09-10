#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "containers/vector.hpp"
#include "functional/fp.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

namespace simbi {
    template <std::uint64_t Dims>
    struct domain_t;

    template <std::uint64_t Dims>
    auto subdivide(const domain_t<Dims>& domain, const iarray<Dims>& divisions)
    {
        // reasonable max for most cases
        vector_t<domain_t<Dims>, 64> subdomains;
        std::size_t count = 0;

        auto chunk_sizes = domain.shape();
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            chunk_sizes[ii] =
                (chunk_sizes[ii] + divisions[ii] - 1) / divisions[ii];
        }

        // generate all subdivision combinations
        iarray<Dims> div_coord{};
        do {
            iarray<Dims> sub_start, sub_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                sub_start[ii] =
                    domain.start[ii] + div_coord[ii] * chunk_sizes[ii];
                sub_end[ii] =
                    std::min(sub_start[ii] + chunk_sizes[ii], domain.end[ii]);
            }

            if (sub_start != sub_end) {   // non-empty subdomain
                subdomains[count++] = domain_t<Dims>{sub_start, sub_end};
            }

        } while (increment_coord(div_coord, divisions));

        return std::pair{subdomains, count};
    }

    template <std::uint64_t Dims>
    struct domain_t {
        static constexpr auto dimensions = Dims;
        iarray<Dims> start{0}, end{0};

        constexpr auto linear_to_coord(std::uint64_t linear) const
        {
            iarray<Dims> coord{};
            for (std::int64_t ii = Dims - 1; ii >= 0; --ii) {
                auto dim_size = end[ii] - start[ii];
                coord[ii]     = start[ii] + (linear % dim_size);
                linear /= dim_size;
            }
            return coord;
        }

        constexpr auto coord_to_linear(const iarray<Dims>& coord) const
        {
            std::uint64_t linear = 0;
            for (std::int64_t ii = Dims - 1; ii >= 0; --ii) {
                linear *= (end[ii] - start[ii]);
                linear += coord[ii] - start[ii];
            }
            return linear;
        }

        // pure coordinate space queries
        constexpr std::uint64_t size() const
        {
            std::uint64_t result = 1;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                auto dim_size = end[ii] - start[ii];
                if (dim_size <= 0) {
                    return 0;
                }
                result *= static_cast<std::uint64_t>(dim_size);
            }
            return result;
        }

        constexpr auto shape() const { return end - start; }
        constexpr bool empty() const { return size() == 0; }

        constexpr bool contains(const iarray<Dims>& coord) const
        {
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (coord[ii] < start[ii] || coord[ii] >= end[ii]) {
                    return false;
                }
            }
            return true;
        }

        constexpr domain_t partition(
            std::uint64_t nparts,
            std::uint64_t part,
            std::uint64_t axis = 0
        ) const
        {
            // for single-axis partition, create divisions array
            auto divisions =
                fp::range(Dims) |
                fp::map([=](auto dim) { return (dim == axis) ? nparts : 1; }) |
                fp::collect<iarray<Dims>>;

            auto [subdomains, count] = subdivide(*this, divisions);
            return (part < count) ? subdomains[part] : domain_t<Dims>{};
        }
    };

    // factory functions
    template <std::uint64_t Dims>
    constexpr auto make_domain(const iarray<Dims>& shape)
    {
        return domain_t<Dims>{iarray<Dims>{}, shape};
    }

    template <std::uint64_t Dims>
    constexpr auto
    make_domain(const iarray<Dims>& start, const iarray<Dims>& end)
    {
        return domain_t<Dims>{start, end};
    }

    template <std::uint64_t Dims>
    bool increment_coord(iarray<Dims>& coord, const iarray<Dims>& bounds)
    {
        for (std::int64_t ii = Dims - 1; ii >= 0; --ii) {
            if (++coord[ii] < bounds[ii]) {
                return true;
            }
            coord[ii] = 0;
        }
        return false;   // overflow - done
    }

    template <std::uint64_t Dims>
    std::ostream& operator<<(std::ostream& os, const domain_t<Dims>& d)
    {
        os << "Domain(";
        for (std::uint64_t i = 0; i < Dims; ++i) {
            os << d.start[i] << ":" << d.end[i];
            if (i < Dims - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

}   // namespace simbi

#endif
