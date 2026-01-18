// =============================================================================
// volume_scaling.hpp
//
// volume scaling factors for moving mesh across all coordinate systems.
// handles conversion between intensive and extensive variables for homologous
// mesh expansion.
//
// key concept:
//   - physical volume: V_phys(t) = scale_factor(geometry, a(t)) * V_comoving
//   - extensive vars: store U_ext = U_intensive * scale_factor
//   - intensive vars: U_intensive = U_ext / scale_factor
//
// usage:
//   auto scaler = volume_scaler_t<geometry_t::SPHERICAL, 1>{};
//   real factor = scaler.scaling_factor(motion.a);
//   U_ext = U_int * factor;
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "decorators.hpp"
#include "utility/enums.hpp"

#include <cstdint>

namespace simbi::geometry {

    

    template <geometry_t G, std::uint64_t Rank>
    struct volume_scaler_t
    {
        // scaling factor for physical volume relative to comoving volume
        // V_physical = scaling_factor * V_comoving
        DUAL constexpr real scaling_factor(real a) const
        {
            if constexpr (G == geometry_t::CARTESIAN) {
                // cartesian: V ~ x * y * z
                // each dimension scales with a
                // 1D: V ~ x ~ a
                // 2D: V ~ x*y ~ a²
                // 3D: V ~ x*y*z ~ a³
                return power(a, static_cast<real>(Rank));
            }
            else if constexpr (G == geometry_t::SPHERICAL) {
                // spherical: V ~ r^3 regardless of dimensionality
                // 1D: V = 4π/3 * r^3 ~ a^3
                // 2D: V = 2π * r^3 * (cos\theta_1 - cos\theta_2) ~ a^3
                // 3D: V = r³ * \Omega ~ a^3
                return a * a * a;
            }
            else if constexpr (G == geometry_t::CYLINDRICAL) {
                // cylindrical: V ~ r² * z
                // 1D (r-only): V = 2π * r² * L ~ a²
                // 2D (r-z): V = 2π * r² * z ~ a³
                // 3D (r-φ-z): V = r² * φ * z ~ a³
                if constexpr (Rank == 1) {
                    return a * a;
                }
                else {
                    return a * a * a;
                }
            }
            else if constexpr (G == geometry_t::AXIS_CYLINDRICAL) {
                // axis cylindrical (r-z with axisymmetry)
                // V ~ r² * z ~ a³
                return a * a * a;
            }
            else if constexpr (G == geometry_t::PLANAR_CYLINDRICAL) {
                // planar cylindrical (r-φ, z-integrated)
                // V ~ r² ~ a²
                return a * a;
            }
            else {
                // fallback: assume full 3D scaling
                return a * a * a;
            }
        }

        // volume expansion rate: (1/V) * dV/dt = expansion_rate * H
        // where H = a_dot / a is the hubble parameter
        DUAL constexpr real expansion_rate() const
        {
            if constexpr (G == geometry_t::CARTESIAN) {
                // dV/dt / V = d/dt(aⁿ) / aⁿ = n * (da/dt) / a = n * H
                return static_cast<real>(Rank);
            }
            else if constexpr (G == geometry_t::SPHERICAL) {
                // dV/dt / V = d/dt(a³) / a³ = 3 * H
                return 3.0;
            }
            else if constexpr (G == geometry_t::CYLINDRICAL) {
                if constexpr (Rank == 1) {
                    return 2.0;
                }
                else {
                    return 3.0;
                }
            }
            else if constexpr (G == geometry_t::AXIS_CYLINDRICAL) {
                return 3.0;
            }
            else if constexpr (G == geometry_t::PLANAR_CYLINDRICAL) {
                return 2.0;
            }
            else {
                return 3.0;
            }
        }

        // inverse scaling factor (for converting extensive → intensive)
        DUAL constexpr real inverse_scaling_factor(real a) const
        {
            return 1.0 / scaling_factor(a);
        }

      private:
        // compile-time power function
        DUAL constexpr real power(real base, real exponent) const
        {
            if (exponent == 1.0) {
                return base;
            }
            else if (exponent == 2.0) {
                return base * base;
            }
            else if (exponent == 3.0) {
                return base * base * base;
            }
            else {
                // should not reach here with our enum values
                return base * base * base;
            }
        }
    };

    

    // get scaling factor at runtime based on geometry enum
    template <std::uint64_t Rank>
    DUAL inline real get_scaling_factor(geometry_t geom, real a)
    {
        switch (geom) {
            case geometry_t::CARTESIAN:
                return volume_scaler_t<geometry_t::CARTESIAN, Rank>{}.scaling_factor(a);
            case geometry_t::SPHERICAL:
                return volume_scaler_t<geometry_t::SPHERICAL, Rank>{}.scaling_factor(a);
            case geometry_t::CYLINDRICAL:
                return volume_scaler_t<geometry_t::CYLINDRICAL, Rank>{}.scaling_factor(a);
            case geometry_t::AXIS_CYLINDRICAL:
                return volume_scaler_t<geometry_t::AXIS_CYLINDRICAL, Rank>{}.scaling_factor(a);
            case geometry_t::PLANAR_CYLINDRICAL:
                return volume_scaler_t<geometry_t::PLANAR_CYLINDRICAL, Rank>{}.scaling_factor(a);
            default:
                return a * a * a;
        }
    }

    // get expansion rate at runtime based on geometry enum
    template <std::uint64_t Rank>
    DUAL inline real get_expansion_rate(geometry_t geom)
    {
        switch (geom) {
            case geometry_t::CARTESIAN:
                return volume_scaler_t<geometry_t::CARTESIAN, Rank>{}.expansion_rate();
            case geometry_t::SPHERICAL:
                return volume_scaler_t<geometry_t::SPHERICAL, Rank>{}.expansion_rate();
            case geometry_t::CYLINDRICAL:
                return volume_scaler_t<geometry_t::CYLINDRICAL, Rank>{}.expansion_rate();
            case geometry_t::AXIS_CYLINDRICAL:
                return volume_scaler_t<geometry_t::AXIS_CYLINDRICAL, Rank>{}.expansion_rate();
            case geometry_t::PLANAR_CYLINDRICAL:
                return volume_scaler_t<geometry_t::PLANAR_CYLINDRICAL, Rank>{}.expansion_rate();
            default:
                return 3.0;
        }
    }

    

    template <geometry_t G, std::uint64_t Rank>
    struct to_extensive_t
    {
        real a;

        template <typename conserved_t>
        DUAL constexpr conserved_t operator()(const conserved_t& u_intensive) const
        {
            constexpr auto scaler = volume_scaler_t<G, Rank>{};
            const real     factor = scaler.scaling_factor(a);
            return u_intensive * factor;
        }
    };

    

    template <geometry_t G, std::uint64_t Rank>
    struct to_intensive_t
    {
        real a;

        template <typename conserved_t>
        DUAL constexpr conserved_t operator()(const conserved_t& u_extensive) const
        {
            constexpr auto scaler     = volume_scaler_t<G, Rank>{};
            const real     inv_factor = scaler.inverse_scaling_factor(a);
            return u_extensive * inv_factor;
        }
    };

    

    // check if geometry and rank combination is valid
    constexpr bool is_valid_geometry_rank(geometry_t geom, std::uint64_t rank)
    {
        switch (geom) {
            case geometry_t::CARTESIAN:
                return rank >= 1 && rank <= 3;
            case geometry_t::SPHERICAL:
                return rank >= 1 && rank <= 3;
            case geometry_t::CYLINDRICAL:
                return rank >= 1 && rank <= 3;
            case geometry_t::AXIS_CYLINDRICAL:
                return rank == 2;
            case geometry_t::PLANAR_CYLINDRICAL:
                return rank == 2;
            default:
                return false;
        }
    }

} // namespace simbi::geometry


