#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

namespace simbi {
    template <std::uint64_t Dims>
    struct domain_t;

    namespace domain_algebra {
        template <std::uint64_t Dims>
        constexpr auto
        contract(const domain_t<Dims>& domain, const iarray<Dims>& contraction);
    }

    template <std::uint64_t Dims>
    struct physical_region_t {
        vector_t<real, Dims> min;
        vector_t<real, Dims> max;
    };

    template <std::uint64_t Dims>
    domain_t<Dims> to_index_space(
        const physical_region_t<Dims>& phys_region,
        const vector_t<real, Dims>& bounds_min,
        const vector_t<real, Dims>& bounds_max,
        const iarray<Dims>& base_resolution
    )
    {
        iarray<Dims> start, end;

        // convert physical coordinates to indices
        for (std::uint64_t dd = 0; dd < Dims; ++dd) {
            real dx   = (bounds_max[dd] - bounds_min[dd]) / base_resolution[dd];
            start[dd] = static_cast<std::int64_t>(
                (phys_region.min[dd] - bounds_min[dd]) / dx
            );
            end[dd] = static_cast<std::int64_t>(
                (phys_region.max[dd] - bounds_min[dd]) / dx
            );
        }

        return domain_t<Dims>{start, end};
    }

    template <std::uint64_t Dims>
    physical_region_t<Dims> to_physical_space(
        const domain_t<Dims>& index_domain,
        const vector_t<real, Dims>& bounds_min,
        const vector_t<real, Dims>& bounds_max,
        const iarray<Dims>& base_resolution
    )
    {
        vector_t<real, Dims> phys_min, phys_max;

        // Convert indices back to physical coordinates
        for (std::uint64_t dd = 0; dd < Dims; ++dd) {
            real dx = (bounds_max[dd] - bounds_min[dd]) / base_resolution[dd];
            phys_min[dd] = bounds_min[dd] + index_domain.start[dd] * dx;
            phys_max[dd] = bounds_min[dd] + index_domain.fin[dd] * dx;
        }

        return physical_region_t<Dims>{phys_min, phys_max};
    }

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
                    std::min(sub_start[ii] + chunk_sizes[ii], domain.fin[ii]);
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
        iarray<Dims> start{0}, fin{0};

        constexpr auto linear_to_coord(std::uint64_t linear) const
        {
            iarray<Dims> coord{};
            for (std::int64_t ii = Dims - 1; ii >= 0; --ii) {
                auto dim_size = fin[ii] - start[ii];
                coord[ii]     = start[ii] + (linear % dim_size);
                linear /= dim_size;
            }
            return coord;
        }

        constexpr auto coord_to_linear(const iarray<Dims>& coord) const
        {
            std::uint64_t linear = 0;
            for (std::int64_t ii = Dims - 1; ii >= 0; --ii) {
                linear *= (fin[ii] - start[ii]);
                linear += coord[ii] - start[ii];
            }
            return linear;
        }

        // pure coordinate space queries
        constexpr std::uint64_t size() const
        {
            std::uint64_t result = 1;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                auto dim_size = fin[ii] - start[ii];
                if (dim_size <= 0) {
                    return 0;
                }
                result *= static_cast<std::uint64_t>(dim_size);
            }
            return result;
        }

        constexpr auto shape() const { return fin - start; }
        constexpr bool empty() const { return size() == 0; }

        constexpr bool contains(const iarray<Dims>& coord) const
        {
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (coord[ii] < start[ii] || coord[ii] >= fin[ii]) {
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

        domain_t contract(const iarray<Dims>& amount) const
        {
            return domain_algebra::contract(*this, amount);
        }
        template <std::integral T>
        domain_t contract(T amount) const
        {
            return domain_algebra::contract(
                *this,
                ones<Dims, std::int64_t>() * static_cast<std::int64_t>(amount)
            );
        }

        // iterator support
        struct iterator {
            const domain_t<Dims>* domain;
            iarray<Dims> current;
            bool at_end;

            iterator(const domain_t<Dims>* d, bool end_iter = false)
                : domain(d), current(d->start), at_end(end_iter || d->empty())
            {
            }

            iarray<Dims> operator*() const { return current; }

            iterator& operator++()
            {
                if (at_end) {
                    return *this;
                }

                // increment like odometer
                for (std::int64_t dim = Dims - 1; dim >= 0; --dim) {
                    if (++current[dim] < domain->fin[dim]) {
                        return *this;
                    }
                    current[dim] = domain->start[dim];
                }
                at_end = true;   // wrapped all dimensions
                return *this;
            }

            iterator operator++(int)
            {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            bool operator==(const iterator& other) const
            {
                if (at_end && other.at_end) {
                    return true;
                }
                if (at_end != other.at_end) {
                    return false;
                }
                return current == other.current;
            }

            bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }
        };

        iterator begin() const { return iterator(this, false); }
        iterator end() const { return iterator(this, true); }
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
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            os << d.start[ii] << ":" << d.fin[ii];
            if (ii < Dims - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

}   // namespace simbi

#endif
