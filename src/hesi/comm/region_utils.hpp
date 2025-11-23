#ifndef HET_COMM_REGION_UTILS_HPP
#define HET_COMM_REGION_UTILS_HPP

#include "grid/domain.hpp"
#include "types.hpp"

#include <cstdint>

namespace simbi::het::comm {

    template <std::uint64_t Rank>
    grid::domain_t<Rank> to_domain(
        const region_descriptor_t<Rank>& region,
        const grid::domain_t<Rank>& /*parent*/
    )
    {
        grid::domain_t<Rank> result;
        result.start = region.start;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            result.fin[ii] = region.start[ii] + region.extent[ii];
        }
        return result;
    }

    // extract interior edge (source for halo send)
    template <std::uint64_t Rank>
    region_descriptor_t<Rank> get_interior_boundary(
        const grid::domain_t<Rank>& domain,
        std::uint64_t dim,
        std::int64_t width,
        bool right_side
    )
    {
        region_descriptor_t<Rank> region;
        region.extent = domain.shape();
        region.start  = domain.start;

        // shrink to interior (exclude halos)
        std::int64_t interior_start = domain.start[dim] + width;
        std::int64_t interior_size  = domain.shape()[dim] - 2 * width;

        if (right_side) {
            region.start[dim] = interior_start + interior_size - width;
        }
        else {
            region.start[dim] = interior_start;
        }
        region.extent[dim] = width;

        return region;
    }

    // extract halo region (destination for halo recv)
    template <std::uint64_t Rank>
    region_descriptor_t<Rank> get_halo_zone(
        const grid::domain_t<Rank>& domain,
        std::uint64_t dim,
        std::int64_t width,
        bool left_side
    )
    {
        region_descriptor_t<Rank> region;
        region.extent = domain.shape();
        region.start  = domain.start;

        if (left_side) {
            region.start[dim] = domain.start[dim];
        }
        else {
            region.start[dim] = domain.fin[dim] - width;
        }
        region.extent[dim] = width;

        return region;
    }

}   // namespace simbi::het::comm

#endif
