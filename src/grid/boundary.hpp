// =============================================================================
// boundary.hpp
//
// defines types for specifying and managing grid boundaries.
// contains enums like `boundary_type_t` (periodic, outflow, reflect) and
// `grid_location_t`, as well as structs like `boundary_set_t` for holding the
// boundary conditions for all faces of a domain.
//
// usage:
//   boundary_set_t<3> bcs;
//   bcs.set_left(0, boundary_type_t::periodic);
// =============================================================================
#pragma once

#include "containers/vector.hpp"
#include "utility/bimap.hpp"

#include <cstdint>
#include <utility>

namespace simbi::grid {
    // -------------------------------------------------------------------------
    // the staggering (where is the data?)
    // -------------------------------------------------------------------------
    enum class grid_location_t {
        center, // i, j, k (Cell Center)
        face_x, // i+1/2, j, k
        face_y, // i, j+1/2, k
        face_z, // i, j, k+1/2
        node    // i+1/2, j+1/2, k+1/2 (Vertices)
    };

    // -------------------------------------------------------------------------
    // the rules (what happens at the edge?)
    // -------------------------------------------------------------------------
    enum class boundary_type_t {
        periodic,  // wraps around: index -1 => N-1
        outflow,   // dirichlet: value is constant
        reflect,   // mirror: index -1 => 0
        dynamic,   // user-defined function
        partition, // internal partition (inter-block)
    };

    // -------------------------------------------------------------------------
    // boundary state
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct boundary_set_t
    {
        // 2 boundaries per dimension (left/right, bottom/top, back/front)
        vector_t<std::pair<boundary_type_t, boundary_type_t>, Rank> rules;

        constexpr boundary_type_t left(int dim) const
        {
            return rules[dim].first;
        }
        constexpr boundary_type_t right(int dim) const
        {
            return rules[dim].second;
        }

        // mutator for slicing
        constexpr void set_left(std::int64_t dim, boundary_type_t t)
        {
            rules[dim].first = t;
        }
        constexpr void set_right(std::int64_t dim, boundary_type_t t)
        {
            rules[dim].second = t;
        }
    };

    template <std::uint64_t Rank>
    struct boundary_rules_t
    {
        vector_t<std::pair<boundary_type_t, std::int64_t>, Rank> elems;
    };

} // namespace simbi::grid

namespace simbi {
    REGISTER_ENUM_BIMAP(
        grid::boundary_type_t,
        {grid::boundary_type_t::dynamic, "dynamics"},
        {grid::boundary_type_t::outflow, "outflow"},
        {grid::boundary_type_t::periodic, "periodic"},
        {grid::boundary_type_t::reflect, "reflecting"},
        {grid::boundary_type_t::partition, "partition"}
    );

    // ostream overload for boundary_type_t
    inline std::ostream& operator<<(std::ostream& os, const grid::boundary_type_t& bt)
    {
        switch (bt) {
            case grid::boundary_type_t::dynamic:
                os << "dynamic";
                break;
            case grid::boundary_type_t::outflow:
                os << "outflow";
                break;
            case grid::boundary_type_t::periodic:
                os << "periodic";
                break;
            case grid::boundary_type_t::reflect:
                os << "reflecting";
                break;
            case grid::boundary_type_t::partition:
                os << "partition";
                break;
            default:
                os << "unknown";
                break;
        }
        return os;
    }
} // namespace simbi
