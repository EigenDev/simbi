// =============================================================================
// policy.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "decorators.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"

#include <concepts>
#include <cstdint>

namespace simbi::grid {

    // -------------------------------------------------------------------------
    // boundary policy concept
    // any physics module must provide a type satisfying this to the driver
    // -------------------------------------------------------------------------
    template <typename T, typename P>
    concept boundary_policy_c = requires(
        const P&        policy,
        const T&        val,
        std::uint64_t   dim,
        side_t          side,
        boundary_type_t type
    ) {
        { policy.apply(val, dim, side, type) } -> std::convertible_to<T>;
    };

    // -------------------------------------------------------------------------
    // default/noop policy
    // just copies values (scalar advection behavior)
    // -------------------------------------------------------------------------
    struct default_boundary_policy_t
    {
        template <typename T>
        DUAL T apply(
            const T& val,
            std::uint64_t /*dim*/,
            side_t /*side*/,
            boundary_type_t /*type*/
        ) const
        {
            return val;
        }
    };

} // namespace simbi::grid


