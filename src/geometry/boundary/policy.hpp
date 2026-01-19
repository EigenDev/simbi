// =============================================================================
// policy.hpp
//
// defines the concept for boundary condition policies.
// this file specifies the `boundary_policy_c` concept that physics modules
// must satisfy to handle boundary conditions. it also provides a
// `default_boundary_policy_t` that performs a simple copy.
//
// usage:
//   struct my_policy {
//     template <typename T>
//     T apply(const T& val, ...) const { ... }
//   };
//   static_assert(boundary_policy_c<my_type, my_policy>);
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
