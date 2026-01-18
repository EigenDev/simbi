// =============================================================================
// ghost.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "algebra.hpp"
#include "containers/vector.hpp"
#include "domain.hpp"
#include <cstddef>
#include <cstdint>

namespace simbi::grid::boundary {

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

    template <std::uint64_t Rank>
    struct ghost_region_t {
        domain_t<Rank> domain;
        ghost_type_t type;
        vector_t<face_side_t, Rank> directions;
    };

    template <std::uint64_t Rank>
    struct ghost_set_t {
        // maximum possible ghost regions from difference operation
        static constexpr std::size_t max_regions =
            domain_algebra::difference_set_t<Rank>::max_regions;

        vector_t<ghost_region_t<Rank>, max_regions> regions;
        std::size_t count = 0;

        auto begin() { return regions.begin(); }
        auto end() { return regions.begin() + count; }
        auto begin() const { return regions.begin(); }
        auto end() const { return regions.begin() + count; }

        bool empty() const { return count == 0; }
    };

    // classify ghost region by contact type with active domain
    template <std::uint64_t Rank>
    constexpr ghost_type_t classify_ghost_type(
        const domain_t<Rank>& ghost,
        const domain_t<Rank>& active
    )
    {
        std::uint64_t contact_count = 0;
        for (std::uint64_t dim = 0; dim < Rank; ++dim) {
            if (ghost.start[dim] == active.fin[dim] ||
                ghost.fin[dim] == active.start[dim]) {
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
    template <std::uint64_t Rank>
    constexpr auto
    ghost_direction(const domain_t<Rank>& ghost, const domain_t<Rank>& active)
    {
        vector_t<face_side_t, Rank> directions;
        for (std::uint64_t dim = 0; dim < Rank; ++dim) {
            if (ghost.fin[dim] == active.start[dim]) {
                directions[dim] = face_side_t::minus;
            }
            else if (ghost.start[dim] == active.fin[dim]) {
                directions[dim] = face_side_t::plus;
            }
            else {
                directions[dim] = face_side_t::none;
            }
        }
        return directions;
    }

    // main analysis function - returns classified ghost regions
    template <std::uint64_t Rank>
    auto analyze_ghost_regions(
        const domain_t<Rank>& full_domain,
        const domain_t<Rank>& active_domain
    )
    {
        using namespace domain_algebra;

        // get geometric difference
        auto raw_regions = difference(full_domain, active_domain);

        ghost_set_t<Rank> result;

        // classify each geometric region
        for (const auto& region : raw_regions) {
            auto& ghost      = result.regions[result.count++];
            ghost.domain     = region;
            ghost.type       = classify_ghost_type(region, active_domain);
            ghost.directions = ghost_direction(region, active_domain);
        }

        return result;
    }

}   // namespace simbi::grid::boundary


