#ifndef GHOST_ANALYSIS_HPP
#define GHOST_ANALYSIS_HPP

#include "algebra.hpp"
#include "containers/vector.hpp"
#include "domain.hpp"
#include <cstddef>
#include <cstdint>

namespace simbi::boundary {

    enum class face_side_t : std::uint8_t {
        minus,   // ghost is before active domain in this dimension
        plus,    // ghost is after active domain in this dimension
        none     // ghost doesn't contact active domain in this dimension
    };

    enum class ghost_type_t : std::uint8_t {
        face,    // touches active domain on exactly one face
        edge,    // touches active domain on exactly one edge (2 faces)
        corner   // touches active domain on exactly one corner (3+ faces)
    };

    template <std::uint64_t Dims>
    struct ghost_region_t {
        domain_t<Dims> domain;
        ghost_type_t type;
        vector_t<face_side_t, Dims> directions;
    };

    template <std::uint64_t Dims>
    struct ghost_set_t {
        // maximum possible ghost regions from difference operation
        static constexpr std::size_t max_regions =
            domain_algebra::difference_set_t<Dims>::max_regions;

        vector_t<ghost_region_t<Dims>, max_regions> regions;
        std::size_t count = 0;

        auto begin() { return regions.begin(); }
        auto end() { return regions.begin() + count; }
        auto begin() const { return regions.begin(); }
        auto end() const { return regions.begin() + count; }

        bool empty() const { return count == 0; }
    };

    // classify ghost region by contact type with active domain
    template <std::uint64_t Dims>
    constexpr ghost_type_t classify_ghost_type(
        const domain_t<Dims>& ghost,
        const domain_t<Dims>& active
    )
    {
        std::uint64_t contact_count = 0;
        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            if (ghost.start[dim] == active.end[dim] ||
                ghost.end[dim] == active.start[dim]) {
                contact_count++;
            }
        }

        if (contact_count == 1) {
            return ghost_type_t::face;
        }
        if (contact_count == 2) {
            return ghost_type_t::edge;
        }
        return ghost_type_t::corner;
    }

    // determine which faces the ghost region contacts
    template <std::uint64_t Dims>
    constexpr auto
    ghost_direction(const domain_t<Dims>& ghost, const domain_t<Dims>& active)
    {
        vector_t<face_side_t, Dims> directions;
        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            if (ghost.end[dim] == active.start[dim]) {
                directions[dim] = face_side_t::minus;
            }
            else if (ghost.start[dim] == active.end[dim]) {
                directions[dim] = face_side_t::plus;
            }
            else {
                directions[dim] = face_side_t::none;
            }
        }
        return directions;
    }

    // main analysis function - returns classified ghost regions
    template <std::uint64_t Dims>
    auto analyze_ghost_regions(
        const domain_t<Dims>& full_domain,
        const domain_t<Dims>& active_domain
    )
    {
        using namespace domain_algebra;

        // get geometric difference
        auto raw_regions = difference(full_domain, active_domain);

        ghost_set_t<Dims> result;

        // classify each geometric region
        for (const auto& region : raw_regions) {
            auto& ghost      = result.regions[result.count++];
            ghost.domain     = region;
            ghost.type       = classify_ghost_type(region, active_domain);
            ghost.directions = ghost_direction(region, active_domain);
        }

        return result;
    }

}   // namespace simbi::boundary

#endif
