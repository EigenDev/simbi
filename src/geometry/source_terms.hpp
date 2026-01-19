// =============================================================================
// source_terms.hpp
//
// computes geometric source terms for curvilinear coordinates.
// this file provides the implementation for non-conservative source terms
// that arise from the grid geometry in curvilinear coordinate systems, such as
// spherical or cylindrical coordinates, which are necessary for correct momentum
// and energy evolution.
//
// usage:
//   auto sources = geometric_source_terms(prims, gamma, idx, block_geo);
// =============================================================================
#pragma once

#include "base/concepts.hpp"
#include "build_config.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "geometry/metrics.hpp"
#include "physics/hydro/physics.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <type_traits>

namespace simbi::geometry {

    // -------------------------------------------------------------------------
    // geometric source term provider
    // computes the non-conservative updates required for curvilinear coords
    // e.g. spherical: adds (rho*v_theta^2 + 2*p)/r to radial momentum
    // -------------------------------------------------------------------------
    template <is_hydro_primitive_c prim_t, typename block_geo_t, std::uint64_t Rank>
    DUAL typename prim_t::counterpart_t geometric_source_terms(
        const prim_t&       prim,
        real                gamma,
        const iarray<Rank>& idx,
        const block_geo_t&  block_geo
    )
    {
        using namespace hydro;
        using cons_t                 = typename prim_t::counterpart_t;
        constexpr std::uint64_t rank = prim_t::rank;

        // cartesian (flat space)
        // no source terms. compiler optimizes this function away completely.
        if constexpr (is_cartesian_c<typename block_geo_t::metric_type>) {
            return cons_t{};
        }
        // spherical (r, theta, phi)
        else if constexpr (is_spherical_c<typename block_geo_t::metric_type>) {
            // retrieve coordinates
            auto centroid = block_geo.labframe_centroid(idx);
            real r        = centroid[0];
            real theta    = [=]() {
                if constexpr (Rank > 1) {
                    return centroid[1];
                }
                else {
                    return 0.5 * std::numbers::pi;
                }
            }();

            // singularity guard
            if (r < build::epsilon) {
                return cons_t{};
            }
            const auto sin_t = std::sin(theta);

            real cot = (std::abs(sin_t) > build::epsilon) ? std::cos(theta) / sin_t : 0.0;

            // unpack primitives
            const real v1    = proper_velocity(prim, 1);
            const real v2    = proper_velocity(prim, 2);
            const real v3    = proper_velocity(prim, 3);
            const real pt    = total_pressure(prim);
            const auto bmu   = magnetic_four_vector(prim);
            const real wt    = enthalpy_density(prim, gamma);
            const real gam2  = lorentz_factor_squared(prim);
            const real wgam2 = wt * gam2;

            // compute terms
            cons_t src{};

            // radial momentum: (rho(vt^2 + vp^2) + 2P) / r
            // the '2P/r' comes from the area expansion dA/dr
            const auto rs = unit_vectors::array_offset<rank>(rank - 1);
            const auto aL = block_geo.labframe_face_area(idx /**/, rank - 1);
            const auto aR = block_geo.labframe_face_area(idx + rs, rank - 1);
            const auto dv = block_geo.labframe_volume(idx);
            src.mom[0]    = pt * (aR - aL) / dv + wgam2 * (v2 * v2 + v3 * v3) / r -
                         (bmu[2] * bmu[2] + bmu[3] * bmu[3]) / r;

            // theta momentum
            if constexpr (Rank > 1) {
                const auto ts = unit_vectors::array_offset<rank>(rank - 2);
                const auto aL = block_geo.labframe_face_area(idx /**/, rank - 2);
                const auto aR = block_geo.labframe_face_area(idx + ts, rank - 2);
                src.mom[1]    = pt * (aR - aL) / dv - wgam2 * (v2 * v1 - v3 * v3 * cot) / r +
                             (bmu[2] * bmu[1] - bmu[3] * bmu[3] * cot) / r;
            }

            // phi momentum: -(rho*vr*vp)/r - (rho*vt*vp*cot)/r
            if constexpr (Rank > 2) {
                src.mom[2] =
                    -wgam2 * v3 * (v1 + cot * v2) / r + bmu[3] * (bmu[1] + cot * bmu[2]) / r;
            }

            return src;
        }

        // cylindrical (r, phi, z)
        else if constexpr (is_cylindrical_c<typename block_geo_t::metric_type>) {
            auto centroid = block_geo.labframe_centroid(idx);
            real r        = centroid[0];
            if (r < build::epsilon) {
                return cons_t{};
            }

            // unpack primitives
            real v1    = proper_velocity(prim, 1);
            real v2    = proper_velocity(prim, 2);
            real pt    = total_pressure(prim);
            auto bmu   = magnetic_four_vector(prim);
            real wt    = enthalpy_density(prim, gamma);
            real gam2  = lorentz_factor_squared(prim);
            real wgam2 = wt * gam2;

            cons_t src{};

            // radial momentum
            using cyl_type = typename block_geo_t::metric_type::cyl_type;
            if constexpr (std::is_same_v<cyl_type, axis_cylindrical_tag>) {
                src.mom[0] = (pt - bmu[1] * bmu[1]) / r;
            }
            else {
                src.mom[0] = (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / r;
            }

            if constexpr (Rank > 1) {
                if constexpr (!std::is_same_v<cyl_type, axis_cylindrical_tag>) {
                    src.mom[1] = -(wgam2 * v1 * v2 - bmu[1] * bmu[2]) / r;
                }
            }

            return src;
        }

        return cons_t{};
    }

    // -------------------------------------------------------------------------
    // mesh motion source term
    // accounts for volume change in moving mesh for homologous expansion
    //
    // for homologous expansion r_phys = a(t) * r_comoving:
    //   - 1D spherical: V ~ r³ → dV/dt / V = 3H
    //   - 1D cylindrical: V ~ r² → dV/dt / V = 2H
    //   - 1D cartesian: V ~ r → dV/dt / V = H
    //   - 2D/3D: depends on how many dims expand
    //
    // where H = a_dot / a is the hubble parameter
    // source term: S = -u * (dV/dt) / V
    // -------------------------------------------------------------------------
    template <typename conserved_t, typename metric_t>
    DUAL conserved_t mesh_motion_source_term(const conserved_t& u, real a, real a_dot)
    {
        if (a_dot == 0.0 || a == 0.0) {
            return conserved_t{};
        }

        // hubble parameter: H = a_dot / a
        const real H = a_dot / a;

        // dimension factor depends on coordinate system
        real dim_factor = 3.0; // default: full 3D or spherical 1D

        if constexpr (is_spherical_c<metric_t>) {
            // spherical: V ~ r³ regardless of dimension
            dim_factor = 3.0;
        }
        else if constexpr (is_cylindrical_c<metric_t>) {
            // cylindrical 1D: V ~ r²; 2D: V ~ r²z; 3D: V ~ r²z
            constexpr std::uint64_t rank = conserved_t::rank;
            dim_factor                   = (rank == 1) ? 2.0 : 3.0;
        }
        else if constexpr (is_cartesian_c<metric_t>) {
            // cartesian: dimension = number of expanding directions
            constexpr std::uint64_t rank = conserved_t::rank;
            dim_factor                   = static_cast<real>(rank);
        }

        // volume change rate: dV/dt / V = dim_factor * H
        const real volume_rate = dim_factor * H;

        // source term: S = -u * (dV/dt) / V
        return u * (-volume_rate);
    }

} // namespace simbi::geometry
