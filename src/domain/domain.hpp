#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "containers/vector.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>

namespace simbi {

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
