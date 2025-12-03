#ifndef GEOMETRY_NCT_GEOM_HPP
#define GEOMETRY_NCT_GEOM_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "geometry/metrics.hpp"
#include "utility/enums.hpp"

#include <cmath>
#include <cstdint>

namespace simbi::em {

    // ========================================================================
    // discrete curl for constrained transport (geometry-based)
    // computes -dB/dt = curl(E) using face/edge EMFs
    // ========================================================================

    // cartesian discrete curl
    template <magnetic_comp_t MagComp, std::uint64_t Rank, typename Geometry>
    DEV real discrete_curl_cartesian(
        const vector_t<vector_t<real, 2>, 2>& edge_emfs,
        const iarray<Rank>& face_coord,
        const Geometry& geo
    )
    {
        // get cell widths from geometry
        const auto h = geo.metric.cell_widths(face_coord);

        if constexpr (MagComp == magnetic_comp_t::K) {   // Bz (comp 0)
            // for Bz: t1=1 (y), t2=2 (x)
            // edge_emfs[0] = Ey, edge_emfs[1] = Ex
            const auto& iedge = edge_emfs[0];   // Ex
            const auto& jedge = edge_emfs[1];   // Ey
            const real ej_l   = jedge[0];
            const real ej_r   = jedge[1];
            const real ei_l   = iedge[0];
            const real ei_r   = iedge[1];

            const real dxi = h[2];   // i-direction (x)
            const real dxj = h[1];   // j-direction (y)

            // curl(E)_z = dEy/dx - dEx/dy
            return ((ej_r - ej_l) / dxi) - ((ei_r - ei_l) / dxj);
        }
        else if constexpr (MagComp == magnetic_comp_t::J) {   // By (comp 1)
            // for By: t1=0 (z), t2=2 (x)
            // edge_emfs[0] = Ez, edge_emfs[1] = Ex
            const auto& kedge = edge_emfs[0];   // Ez
            const auto& iedge = edge_emfs[1];   // Ex
            const real ek_l   = kedge[0];
            const real ek_r   = kedge[1];
            const real ei_l   = iedge[0];
            const real ei_r   = iedge[1];

            const real dxk = h[0];   // k-direction (z)
            const real dxi = h[2];   // i-direction (x)

            // curl(E)_y = dEx/dz - dEz/dx
            return ((ei_r - ei_l) / dxk) - ((ek_r - ek_l) / dxi);
        }
        else {   // Bx (comp 2)
            // for Bx: t1=1 (y), t2=0 (z)
            // edge_emfs[0] = Ey, edge_emfs[1] = Ez
            const auto& jedge = edge_emfs[0];   // Ey
            const auto& kedge = edge_emfs[1];   // Ez
            const real ej_l   = jedge[0];
            const real ej_r   = jedge[1];
            const real ek_l   = kedge[0];
            const real ek_r   = kedge[1];

            const real dxj = h[1];   // j-direction (y)
            const real dxk = h[0];   // k-direction (z)

            // curl(E)_x = dEz/dy - dEy/dz
            return ((ek_r - ek_l) / dxj) - ((ej_r - ej_l) / dxk);
        }
    }

    // spherical discrete curl
    template <magnetic_comp_t MagComp, std::uint64_t Rank, typename Geometry>
    DEV real discrete_curl_spherical(
        const vector_t<vector_t<real, 2>, 2>& edge_emfs,
        const iarray<Rank>& face_coord,
        const Geometry& geo
    )
    {
        const auto position = geo.centroid(face_coord);
        const real r        = position[0];
        const real theta    = position[1];

        if constexpr (MagComp == magnetic_comp_t::I) {   // Br
            const auto& jedge = edge_emfs[0];
            const auto& kedge = edge_emfs[1];

            const real tl = geo.metric.face_position(face_coord, 1);
            auto tr_coord = face_coord;
            tr_coord[1] += 1;
            const real tr = geo.metric.face_position(tr_coord, 1);

            const real ej_l = jedge[0];
            const real ej_r = jedge[1];
            const real ek_l = kedge[0] * std::sin(tl);
            const real ek_r = kedge[1] * std::sin(tr);

            auto pk_coord = face_coord;
            pk_coord[0] += 1;
            const real dxk = geo.metric.face_position(pk_coord, 0) -
                             geo.metric.face_position(face_coord, 0);
            const real dxj = tr - tl;

            return (1.0 / (r * std::sin(theta))) *
                   (((ek_r - ek_l) / dxj) - ((ej_r - ej_l) / dxk));
        }
        else if constexpr (MagComp == magnetic_comp_t::J) {   // Btheta
            const auto& kedge = edge_emfs[0];
            const auto& iedge = edge_emfs[1];

            const real rl = geo.metric.face_position(face_coord, 2);
            auto rr_coord = face_coord;
            rr_coord[2] += 1;
            const real rr = geo.metric.face_position(rr_coord, 2);

            const real ei_l = iedge[0];
            const real ei_r = iedge[1];
            const real ek_l = kedge[0] * rl;
            const real ek_r = kedge[1] * rr;

            auto pk_coord = face_coord;
            pk_coord[0] += 1;
            const real dxk = (geo.metric.face_position(pk_coord, 0) -
                              geo.metric.face_position(face_coord, 0)) /
                             std::sin(theta);
            const real dxi = rr - rl;

            return (1.0 / r) * (((ei_r - ei_l) / dxk) - ((ek_r - ek_l) / dxi));
        }
        else {   // Bphi
            const auto& iedge = edge_emfs[0];
            const auto& jedge = edge_emfs[1];

            const real rl = geo.metric.face_position(face_coord, 2);
            auto rr_coord = face_coord;
            rr_coord[2] += 1;
            const real rr = geo.metric.face_position(rr_coord, 2);

            const real ei_l = iedge[0];
            const real ei_r = iedge[1];
            const real ej_l = jedge[0] * rl;
            const real ej_r = jedge[1] * rr;

            auto pj_coord = face_coord;
            pj_coord[1] += 1;
            const real dxj = geo.metric.face_position(pj_coord, 1) -
                             geo.metric.face_position(face_coord, 1);
            const real dxi = rr - rl;

            return (1.0 / r) * (((ej_r - ej_l) / dxi) - ((ei_r - ei_l) / dxj));
        }
    }

    // cylindrical discrete curl
    template <magnetic_comp_t MagComp, std::uint64_t Rank, typename Geometry>
    DEV real discrete_curl_cylindrical(
        const vector_t<vector_t<real, 2>, 2>& edge_emfs,
        const iarray<Rank>& face_coord,
        const Geometry& geo
    )
    {
        const auto position = geo.centroid(face_coord);
        const real r        = position[0];

        if constexpr (MagComp == magnetic_comp_t::I) {   // Br
            const auto& jedge = edge_emfs[0];
            const auto& kedge = edge_emfs[1];

            const real ej_l = jedge[0] * r;
            const real ej_r = jedge[1] * r;
            const real ek_l = kedge[0];
            const real ek_r = kedge[1];

            auto pk_coord = face_coord;
            pk_coord[0] += 1;
            const real dxk = geo.metric.face_position(pk_coord, 0) -
                             geo.metric.face_position(face_coord, 0);
            auto pj_coord = face_coord;
            pj_coord[1] += 1;
            const real dxj = geo.metric.face_position(pj_coord, 1) -
                             geo.metric.face_position(face_coord, 1);

            return (1.0 / r) * (ek_r - ek_l) / dxj - (ej_r - ej_l) / dxk;
        }
        else if constexpr (MagComp == magnetic_comp_t::J) {   // Bphi
            const auto& kedge = edge_emfs[0];
            const auto& iedge = edge_emfs[1];

            const real ei_l = iedge[0];
            const real ei_r = iedge[1];
            const real ek_l = kedge[0];
            const real ek_r = kedge[1];

            auto pk_coord = face_coord;
            pk_coord[0] += 1;
            const real dxk = geo.metric.face_position(pk_coord, 0) -
                             geo.metric.face_position(face_coord, 0);
            auto pi_coord = face_coord;
            pi_coord[1] += 1;
            const real dxi = geo.metric.face_position(pi_coord, 1) -
                             geo.metric.face_position(face_coord, 1);

            return ((ei_r - ei_l) / dxk) - ((ek_r - ek_l) / dxi);
        }
        else {   // Bz
            const auto& iedge = edge_emfs[0];
            const auto& jedge = edge_emfs[1];

            const real rl = geo.metric.face_position(face_coord, 2);
            auto rr_coord = face_coord;
            rr_coord[2] += 1;
            const real rr = geo.metric.face_position(rr_coord, 2);

            const real ei_l = iedge[0];
            const real ei_r = iedge[1];
            const real ej_l = jedge[0] * rl;
            const real ej_r = jedge[1] * rr;

            auto pj_coord = face_coord;
            pj_coord[1] += 1;
            const real dxj = geo.metric.face_position(pj_coord, 1) -
                             geo.metric.face_position(face_coord, 1);
            const real dxi = rr - rl;

            return (1.0 / r) * (((ej_r - ej_l) / dxi) - ((ei_r - ei_l) / dxj));
        }
    }

    // ========================================================================
    // unified discrete curl dispatcher
    // ========================================================================
    template <magnetic_comp_t MagComp, std::uint64_t Rank, typename Geometry>
    DEV real discrete_curl(
        const vector_t<vector_t<real, 2>, 2>& edge_emfs,
        const iarray<Rank>& face_coord,
        const Geometry& geo
    )
    {
        using metric_t = typename Geometry::metric_type;

        if constexpr (geometry::is_cartesian_c<metric_t>) {
            return discrete_curl_cartesian<MagComp>(edge_emfs, face_coord, geo);
        }
        else if constexpr (geometry::is_spherical_c<metric_t>) {
            return discrete_curl_spherical<MagComp>(edge_emfs, face_coord, geo);
        }
        else if constexpr (geometry::is_cylindrical_c<metric_t>) {
            return discrete_curl_cylindrical<MagComp>(
                edge_emfs,
                face_coord,
                geo
            );
        }
        else {
            // fallback to cartesian
            return discrete_curl_cartesian<MagComp>(edge_emfs, face_coord, geo);
        }
    }

}   // namespace simbi::em

#endif   // GEOMETRY_NCT_GEOM_HPP
