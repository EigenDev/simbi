#ifndef PHYSICS_HYDRO_BOUNDARY_POLICY_HPP
#define PHYSICS_HYDRO_BOUNDARY_POLICY_HPP

// =============================================================================
// boundary_policy.hpp
//
// hydro-specific boundary policy for geometry/boundary/driver.hpp
// handles velocity reflection, pressure extrapolation, etc.
//
// coordinate conventions:
//   array indexing: row-major where index 0 is slowest, index Rank-1 is fastest
//   for 2D spherical: dim=0 -> theta, dim=1 -> r
//   for 3D spherical: dim=0 -> phi, dim=1 -> theta, dim=2 -> r
//
//   logical dimensions: x1=r, x2=theta, x3=phi
//   array index = Rank - 1 - logical_index
//
//   state layout: [rho, v1, v2, v3, p, ...]
//   velocities in logical order: v1=vr, v2=vtheta, v3=vphi
//
// spherical polar boundaries:
//   theta boundaries at 0 or pi are axis singularities (poles)
//   reflecting at poles flips v_theta AND v_phi (both tangent to pole)
//   theta boundaries at other values (e.g., pi/2) are true walls
//   reflecting at walls only flips v_theta (normal component)
// =============================================================================

#include "build_config.hpp"
#include "decorators.hpp"
#include "geometry/api.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "utility/helpers.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>

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
    struct hydro_boundary_policy_t
    {
        // state layout: [rho, v1, v2, v3, p]
        // velocity starts at index 1, v_dim is at index (1 + dim)
        static constexpr std::uint64_t velocity_offset = 1;

        // coordinate system (needed for spherical pole handling)
        geometry::metric_type_t metric = geometry::metric_type_t::cartesian;

        // theta bounds (only relevant for spherical, dim=1)
        real theta_min = 0.0;
        real theta_max = std::numbers::pi;

        template <typename T>
        DUAL T apply(
            const T&              state,
            std::uint64_t         dim,
            grid::side_t          side,
            grid::boundary_type_t type
        ) const
        {
            T result = state;

            // convert array-index dim to logical dim for velocity access
            // array dim 0 is slowest (highest logical), dim Rank-1 is fastest
            // (x1)
            const std::uint64_t logical_dim = Rank - 1 - dim;
            const std::uint64_t vel_idx     = velocity_offset + logical_dim;

            // theta is x2, which in array-index is at position Rank-2
            constexpr std::uint64_t theta_array_dim = Rank - 2;

            switch (type) {
                case grid::boundary_type_t::reflect: {
                    if constexpr (Rank >= 2) {
                        if (metric == geometry::metric_type_t::spherical &&
                            dim == theta_array_dim) {
                            // theta boundary in spherical coordinates
                            // check if this is a pole or a wall
                            bool           is_pole  = false;
                            constexpr real pole_tol = 1e-10;
                            constexpr real pi_half  = 0.5 * std::numbers::pi;
                            if (side == grid::side_t::left) {
                                is_pole = (theta_min < pole_tol);
                            }
                            else {
                                is_pole = (std::abs(theta_max - std::numbers::pi) < pole_tol);
                            }

                            if (!is_pole && helpers::goes_to_zero(theta_max - pi_half)) {
                                // at non-pole wall (e.g., theta = pi/2)
                                // only flip normal component (v_theta)
                                result[vel_idx] = -result[vel_idx];
                            }
                        }
                        else {
                            // cartesian or non-theta spherical dimension
                            // standard reflection: flip normal velocity
                            result[vel_idx] = -result[vel_idx];
                        }
                    }
                    else {
                        // 1D case: standard reflection
                        result[vel_idx] = -result[vel_idx];
                    }
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
    struct mhd_boundary_policy_t
    {
        static constexpr std::uint64_t velocity_offset = 1;
        // magnetic field after pressure: [rho, v1, v2, v3, p, b1, b2, b3]
        static constexpr std::uint64_t bfield_offset = 1 + Rank + 1;

        geometry::metric_type_t metric    = geometry::metric_type_t::cartesian;
        real                    theta_min = 0.0;
        real                    theta_max = std::numbers::pi;

        template <typename T>
        DUAL T apply(
            const T&              state,
            std::uint64_t         dim,
            grid::side_t          side,
            grid::boundary_type_t type
        ) const
        {
            T result = state;

            // convert array-index dim to logical dim for velocity/bfield access
            const std::uint64_t logical_dim = Rank - 1 - dim;
            const std::uint64_t vel_idx     = velocity_offset + logical_dim;
            const std::uint64_t b_idx       = bfield_offset + logical_dim;

            // theta is x2, which in array-index is at position Rank-2
            constexpr std::uint64_t theta_array_dim = Rank - 2;

            switch (type) {
                case grid::boundary_type_t::reflect: {
                    if constexpr (Rank >= 2) {
                        if (metric == geometry::metric_type_t::spherical &&
                            dim == theta_array_dim) {
                            constexpr real pole_tol = 1e-10;
                            bool           is_pole  = false;

                            if (side == grid::side_t::left) {
                                is_pole = (theta_min < pole_tol);
                            }
                            else {
                                is_pole = (std::abs(theta_max - std::numbers::pi) < pole_tol);
                            }

                            if (is_pole) {
                                // at poles: flip tangential components
                                // v_theta at velocity_offset + 1 (logical x2)
                                result[velocity_offset + 1] = -result[velocity_offset + 1];
                                if constexpr (Rank > 2) {
                                    // v_phi at velocity_offset + 2 (logical x3)
                                    result[velocity_offset + 2] = -result[velocity_offset + 2];
                                }
                                // magnetic field: same treatment
                                if constexpr (T::nmem > bfield_offset) {
                                    result[bfield_offset + 1] = -result[bfield_offset + 1];
                                    if constexpr (Rank > 2) {
                                        result[bfield_offset + 2] = -result[bfield_offset + 2];
                                    }
                                }
                            }
                            else {
                                result[vel_idx] = -result[vel_idx];
                                if constexpr (T::nmem > bfield_offset) {
                                    result[b_idx] = -result[b_idx];
                                }
                            }
                        }
                        else {
                            // standard reflection
                            result[vel_idx] = -result[vel_idx];
                            if constexpr (T::nmem > bfield_offset) {
                                result[b_idx] = -result[b_idx];
                            }
                        }
                    }
                    else {
                        // 1D case: standard reflection
                        result[vel_idx] = -result[vel_idx];
                        if constexpr (T::nmem > bfield_offset) {
                            result[b_idx] = -result[b_idx];
                        }
                    }
                    break;
                }
                default:
                    break;
            }

            return result;
        }
    };

    // =========================================================================
    // factory to select policy based on regime
    // =========================================================================
    template <bool IsMHD, std::uint64_t Rank>
    auto make_boundary_policy(
        geometry::metric_type_t metric    = geometry::metric_type_t::cartesian,
        real                    theta_min = 0.0,
        real                    theta_max = std::numbers::pi
    )
    {
        if constexpr (IsMHD) {
            return mhd_boundary_policy_t<Rank>{metric, theta_min, theta_max};
        }
        else {
            return hydro_boundary_policy_t<Rank>{metric, theta_min, theta_max};
        }
    }

} // namespace simbi::hydro

#endif // PHYSICS_HYDRO_BOUNDARY_POLICY_HPP
