#ifndef SIMBI_GEOMETRY_CT_EXTENSIONS_HPP
#define SIMBI_GEOMETRY_CT_EXTENSIONS_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "mesh/mesh_config.hpp"
#include "mesh/mesh_ops.hpp"
#include "utility/enums.hpp"
#include <cmath>
#include <cstdint>

namespace simbi::em {
    using namespace simbi::mesh;

    // ========================================================================
    // DISCRETE CURL IMPLEMENTATION
    // ========================================================================

    template <magnetic_comp_t MagComp, std::uint64_t Dims, Geometry G>
    real discrete_curl(
        const vector_t<vector_t<real, 2>, 2>& edge_emfs,
        const iarray<Dims>& face_coord,
        const mesh::mesh_config_t<Dims, G>& config
    )
    {
        if constexpr (G == Geometry::CARTESIAN) {
            const auto widths = cell_widths(face_coord, config);

            if constexpr (MagComp == magnetic_comp_t::K) {   // Bz
                // ∇ × E for Bz: (∂Ey/∂x - ∂Ex/∂y)
                const auto& iedge = edge_emfs[0];   // Ex(i, j\pm 1/2, k-1/2)
                const auto& jedge = edge_emfs[1];   // Ey(i\pm 1/2, j, k-1/2)
                const real ei_l   = iedge[0];       // Ex(i, j - 1/2, k-1/2)
                const real ei_r   = iedge[1];       // Ex(i, j + 1/2, k-1/2)
                const real ej_l   = jedge[0];       // Ey(i - 1/2, j, k-1/2)
                const real ej_r   = jedge[1];       // Ey(i + 1/2, j, k-1/2)

                const real dxi = widths[2];   // i-direction (x)
                const real dxj = widths[1];   // j-direction (y)

                return ((ej_r - ej_l) / dxi) - ((ei_r - ei_l) / dxj);
            }
            else if constexpr (MagComp == magnetic_comp_t::J) {   // By
                // ∇ × E for By: (∂Ex/∂z - ∂Ez/∂x)
                const auto& kedge = edge_emfs[0];   // Ez(i \pm 1/2, j - 1/2, k)
                const auto& iedge = edge_emfs[1];   // Ex(i, j - 1/2, k \pm 1/2)
                const real ei_l   = iedge[0];       // Ex(i, j - 1/2, k - 1/2)
                const real ei_r   = iedge[1];       // Ex(i, j - 1/2, k + 1/2)
                const real ek_l   = kedge[0];       // Ez(i - 1/2, j - 1/2, k)
                const real ek_r   = kedge[1];       // Ez(i + 1/2, j - 1/2, k)

                const real dxk = widths[0];   // k-direction (z)
                const real dxi = widths[2];   // i-direction (x)

                return ((ei_r - ei_l) / dxk) - ((ek_r - ek_l) / dxi);
            }
            else {   // Bx
                // ∇ × E for Bx: (∂Ez/∂y - ∂Ey/∂z)
                const auto& jedge = edge_emfs[0];   // Ey(i-1/2, j\pm 1/2, k)
                const auto& kedge = edge_emfs[1];   // Ey(i-1/2, j, k\pm 1/2)
                const real ek_l   = kedge[0];       // Ez(i-1/2, j - 1/2, k)
                const real ek_r   = kedge[1];       // Ez(i-1/2, j + 1/2, k)
                const real ej_l   = jedge[0];       // Ey(i-1/2, j, k - 1/2)
                const real ej_r   = jedge[1];       // Ey(i-1/2, j, k + 1/2)

                const real dxj = widths[1];   // j-direction (y)
                const real dxk = widths[0];   // k-direction (z)

                return ((ek_r - ek_l) / dxj) - ((ej_r - ej_l) / dxk);
            }
        }
        else if constexpr (G == Geometry::SPHERICAL) {
            const auto position = centroid(face_coord, config);
            const real r        = position[2];
            const real theta    = position[1];

            if constexpr (MagComp == magnetic_comp_t::I) {   // Br (x-like)
                // (∇ × E)_r = (1/r) * [∂(E_φ sin θ)/∂θ - ∂E_θ/∂φ]
                const auto& jedge = edge_emfs[0];   // E_θ at φ\pm 1/2 edges
                const auto& kedge = edge_emfs[1];   // E_φ at θ\pm 1/2 edges

                const real tl = face_position(face_coord, 1, Dir::W, config);
                const real tr = face_position(face_coord, 1, Dir::E, config);

                // E_θ (h_θ = r, but r is constant here)
                const real ej_l = jedge[0];
                const real ej_r = jedge[1];                  // E_θ
                const real ek_l = kedge[0] * std::sin(tl);   // E_φ * sin θ
                const real ek_r = kedge[1] * std::sin(tr);   // E_φ * sin θ

                const real dxk = face_position(face_coord, 0, Dir::E, config) -
                                 face_position(face_coord, 0, Dir::W, config);
                const real dxj = tr - tl;   // θ difference

                return (1.0 / (r * std::sin(theta))) *
                       (((ek_r - ek_l) / dxj) - ((ej_r - ej_l) / dxk));
            }
            else if constexpr (MagComp == magnetic_comp_t::J) {   // Bθ (y-like)
                // (∇ × E)_θ = (1/r) * [∂E_r/∂φ/(sin θ) - ∂(r E_φ)/∂r]
                const auto& kedge = edge_emfs[0];   // E_φ at r±1/2 edges
                const auto& iedge = edge_emfs[1];   // E_r at φ±1/2 edges

                const real rl = face_position(face_coord, 2, Dir::W, config);
                const real rr = face_position(face_coord, 2, Dir::E, config);

                const real ei_l = iedge[0];        // E_r (h_r = 1)
                const real ei_r = iedge[1];        // E_r
                const real ek_l = kedge[0] * rl;   // E_φ * r
                const real ek_r = kedge[1] * rr;   // E_φ * r

                const real dxk =
                    (face_position(face_coord, 0, Dir::E, config) -
                     face_position(face_coord, 0, Dir::W, config)) /
                    std::sin(theta);
                const real dxi = rr - rl;

                return (1.0 / r) *
                       (((ei_r - ei_l) / dxk) - ((ek_r - ek_l) / dxi));
            }
            else {   // Bφ (z-like)
                // (∇ × E)_φ = (1/r) * [∂(r E_θ)/∂r - ∂E_r/∂θ]
                const auto& iedge = edge_emfs[0];   // E_r at θ±1/2 edges
                const auto& jedge = edge_emfs[1];   // E_θ at r±1/2 edges

                const real rl = face_position(face_coord, 2, Dir::W, config);
                const real rr = face_position(face_coord, 2, Dir::E, config);

                const real ei_l = iedge[0];        // E_r (h_r = 1)
                const real ei_r = iedge[1];        // E_r
                const real ej_l = jedge[0] * rl;   // E_θ * r
                const real ej_r = jedge[1] * rr;   // E_θ * r

                const real dxj = face_position(face_coord, 1, Dir::E, config) -
                                 face_position(face_coord, 1, Dir::W, config);
                const real dxi = rr - rl;

                return (1.0 / r) *
                       (((ej_r - ej_l) / dxi) - ((ei_r - ei_l) / dxj));
            }
        }
        else {   // cylindrical
            const auto position = centroid(face_coord, config);
            const real r        = position[2];

            if constexpr (MagComp == magnetic_comp_t::I) {   // Br (x-like)
                // (∇ × E)_r = (1/r) * [∂E_z/∂φ - ∂(r E_φ)/∂z]
                const auto& jedge = edge_emfs[0];   // E_φ at z±1/2 edges
                const auto& kedge = edge_emfs[1];   // E_z at φ±1/2 edges

                const real ej_l = jedge[0] * r;   // E_φ * r
                const real ej_r = jedge[1] * r;   // E_φ * r
                const real ek_l = kedge[0];       // E_z (h_z = 1)
                const real ek_r = kedge[1];       // E_z

                const real dxk = face_position(face_coord, 0, Dir::E, config) -
                                 face_position(face_coord, 0, Dir::W, config);
                const real dxj = face_position(face_coord, 1, Dir::E, config) -
                                 face_position(face_coord, 1, Dir::W, config);

                return (1.0 / r) * (ek_r - ek_l) / dxj - (ej_r - ej_l) / dxk;
            }
            else if constexpr (MagComp == magnetic_comp_t::J) {   // Bφ (y-like)
                // (∇ × E)_φ = ∂E_r/∂z - ∂E_z/∂r
                const auto& kedge = edge_emfs[0];   // E_r at z\pm1/2 edges
                const auto& iedge = edge_emfs[1];   // E_z at r\pm1/2 edges

                const real ei_l = iedge[0];   // E_r (h_r = 1)
                const real ei_r = iedge[1];   // E_r
                const real ek_l = kedge[0];   // E_z (h_z = 1)
                const real ek_r = kedge[1];   // E_z

                const real dxk = face_position(face_coord, 0, Dir::E, config) -
                                 face_position(face_coord, 0, Dir::W, config);
                const real dxi = face_position(face_coord, 1, Dir::E, config) -
                                 face_position(face_coord, 1, Dir::W, config);

                return ((ei_r - ei_l) / dxk) - ((ek_r - ek_l) / dxi);
            }
            else {   // Bz (z-like)
                // (∇ × E)_z = (1/r) * [∂(r E_φ)/∂r - ∂E_r/∂φ]
                const auto& iedge = edge_emfs[0];   // E_r at φ±1/2 edges
                const auto& jedge = edge_emfs[1];   // E_φ at r±1/2 edges

                const real rl = face_position(face_coord, 2, Dir::W, config);
                const real rr = face_position(face_coord, 2, Dir::E, config);

                const real ei_l = iedge[0];        // E_r (h_r = 1)
                const real ei_r = iedge[1];        // E_r
                const real ej_l = jedge[0] * rl;   // E_φ * r
                const real ej_r = jedge[1] * rr;   // E_φ * r

                const real dxj = face_position(face_coord, 1, Dir::E, config) -
                                 face_position(face_coord, 1, Dir::W, config);
                const real dxi = rr - rl;

                return (1.0 / r) *
                       (((ej_r - ej_l) / dxi) - ((ei_r - ei_l) / dxj));
            }
        }
    }
}   // namespace simbi::em

#endif   // SIMBI_GEOMETRY_CT_EXTENSIONS_HPP
