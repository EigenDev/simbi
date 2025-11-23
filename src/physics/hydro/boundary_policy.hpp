#ifndef PHYSICS_HYDRO_BOUNDARY_POLICY_HPP
#define PHYSICS_HYDRO_BOUNDARY_POLICY_HPP

// =============================================================================
// boundary_policy.hpp
//
// hydro-specific boundary policy for geometry/boundary/driver.hpp
// handles velocity reflection, pressure extrapolation, etc.
// =============================================================================

#include "compat.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"

#include <cstdint>

namespace simbi::hydro {

    // =========================================================================
    // hydro_boundary_policy_t
    //
    // transforms primitive/conserved states at boundaries.
    // reflecting: flip normal velocity component
    // outflow: copy (no change)
    // periodic: copy (index remapping handles position)
    // =========================================================================
    template <std::uint64_t Rank>
    struct hydro_boundary_policy_t {
        // index of velocity component in state vector
        // for newtonian: [rho, vx, vy, vz, p] -> velocity starts at 1
        // momentum index = dim + 1 for [rho, v1, v2, v3, ...]
        static constexpr std::uint64_t velocity_offset = 1;

        template <typename T>
        DUAL T apply(
            const T& state,
            std::uint64_t dim,
            grid::side_t /*side*/,
            grid::boundary_type_t type
        ) const
        {
            T result = state;

            switch (type) {
                case grid::boundary_type_t::reflect: {
                    // flip normal velocity component
                    // velocity index: for dim d in [0, Rank),
                    // the velocity component is at index (Rank - 1 - d) +
                    // offset because we store in reverse order (vz, vy, vx)
                    // typically but let's use standard ordering: v[dim] =
                    // state[1 + dim]

                    // actually, the state layout depends on regime
                    // for now, assume velocity at indices [1, 1+Rank)
                    // normal component for dimension 'dim' is at 1 + (Rank - 1
                    // - dim) in typical (k,j,i) -> (z,y,x) ordering

                    const std::uint64_t vel_idx =
                        velocity_offset + (Rank - 1 - dim);
                    result[vel_idx] = -result[vel_idx];
                    break;
                }
                case grid::boundary_type_t::outflow:
                case grid::boundary_type_t::periodic:
                case grid::boundary_type_t::dynamic:
                default:
                    // no transformation needed
                    break;
            }

            return result;
        }
    };

    // =========================================================================
    // mhd_boundary_policy_t
    //
    // extends hydro policy with magnetic field handling
    // =========================================================================
    template <std::uint64_t Rank>
    struct mhd_boundary_policy_t {
        static constexpr std::uint64_t velocity_offset = 1;
        // after p
        static constexpr std::uint64_t bfield_offset = 1 + Rank + 1;

        template <typename T>
        DUAL T apply(
            const T& state,
            std::uint64_t dim,
            grid::side_t /*side*/,
            grid::boundary_type_t type
        ) const
        {
            T result = state;

            switch (type) {
                case grid::boundary_type_t::reflect: {
                    // flip normal velocity
                    const std::uint64_t vel_idx =
                        velocity_offset + (Rank - 1 - dim);
                    result[vel_idx] = -result[vel_idx];

                    // flip normal magnetic field (for perfect conductor)
                    // tangential B unchanged, normal B flips
                    if constexpr (T::nmem > bfield_offset) {
                        const std::uint64_t b_idx =
                            bfield_offset + (Rank - 1 - dim);
                        result[b_idx] = -result[b_idx];
                    }
                    break;
                }
                default: break;
            }

            return result;
        }
    };

    // =========================================================================
    // factory to select policy based on regime
    // =========================================================================
    template <bool IsMHD, std::uint64_t Rank>
    auto make_boundary_policy()
    {
        if constexpr (IsMHD) {
            return mhd_boundary_policy_t<Rank>{};
        }
        else {
            return hydro_boundary_policy_t<Rank>{};
        }
    }

}   // namespace simbi::hydro

#endif   // PHYSICS_HYDRO_BOUNDARY_POLICY_HPP
