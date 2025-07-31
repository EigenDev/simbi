#ifndef MESH_OPS_HPP
#define MESH_OPS_HPP

#include "base/concepts.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "mesh_config.hpp"
#include "physics/hydro/physics.hpp"
#include "utility/enums.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace simbi::mesh {
    using namespace simbi::concepts;
    using namespace simbi::hydro;
    // ========================================================================
    // forward declarations
    template <std::uint64_t Dims, Geometry G>
    DEV constexpr real face_position(
        const iarray<Dims>& coord,
        std::uint64_t direction,
        Dir dir,
        const mesh_config_t<Dims, G>& config
    );

    template <std::uint64_t Dims, Geometry G>
    DEV constexpr vector_t<real, Dims>
    centroid(const iarray<Dims>& coord, const mesh_config_t<Dims, G>& config);

    template <std::uint64_t Dims, Geometry G>
    DEV constexpr vector_t<real, Dims> cell_widths(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, G>& config
    );

    template <std::uint64_t Dims, Geometry G>
    DEV real
    volume(const iarray<Dims>& coord, const mesh_config_t<Dims, G>& config);

    template <std::uint64_t Dims, Geometry G>
    DEV real face_area(
        const iarray<Dims>& coord,
        std::uint64_t comp,
        Dir dir,
        const mesh_config_t<Dims, G>& config
    );

    template <std::uint64_t Dims, Geometry G>
    DEV bool
    at_pole(const iarray<Dims>& coord, const mesh_config_t<Dims, G>& config);

    template <std::uint64_t Dims, Geometry G>
    DEV constexpr vector_t<real, Dims> to_cartesian(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, G>& config
    );

    template <is_hydro_primitive_c prim_t, std::uint64_t Dims, Geometry G>
    DEV constexpr auto geometric_source_terms(
        const prim_t& prim,
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, G>& config,
        real gamma
    );

    //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // VOLUME SPECIALIZATIONS
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    template <std::uint64_t Dims>
    DEV real volume(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::CARTESIAN>& config
    )
    {
        // Cartesian volume is simply the product of cell widths
        const auto widths = cell_widths(coord, config);
        return fp::product(widths);
    }

    template <std::uint64_t Dims>
    DEV real volume(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::CYLINDRICAL>& config
    )
    {
        // Cylindrical volume calculation
        const real rl  = face_position(coord, Dims - 1, Dir::W, config);
        const real rr  = face_position(coord, Dims - 1, Dir::E, config);
        const real rdr = (rr * rr - rl * rl) / 2.0;

        real dphi = 2.0 * std::numbers::pi;
        if constexpr (Dims > 1) {
            dphi = face_position(coord, Dims - 2, Dir::E, config) -
                   face_position(coord, Dims - 2, Dir::W, config);
        }
        real dz = 1.0;
        if constexpr (Dims > 2) {
            dz = face_position(coord, Dims - 3, Dir::E, config) -
                 face_position(coord, Dims - 3, Dir::W, config);
        }

        return rdr * dphi * dz;
    }

    template <std::uint64_t Dims>
    DEV real volume(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::SPHERICAL>& config
    )
    {
        // Spherical volume calculation
        const real rl = face_position(coord, Dims - 1, Dir::W, config);
        const real rr = face_position(coord, Dims - 1, Dir::E, config);
        const real dr = (rr * rr * rr - rl * rl * rl) / 3.0;

        real dtheta = 2.0;
        if constexpr (Dims > 1) {
            const real tl = face_position(coord, Dims - 2, Dir::W, config);
            const real tr = face_position(coord, Dims - 2, Dir::E, config);
            dtheta        = std::cos(tl) - std::cos(tr);
        }

        real dphi = 2.0 * std::numbers::pi;
        if constexpr (Dims > 2) {
            dphi = face_position(coord, Dims - 3, Dir::E, config) -
                   face_position(coord, Dims - 3, Dir::W, config);
        }

        return dr * dtheta * dphi;
    }

    template <std::uint64_t Dims>
    DEV real volume(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::AXIS_CYLINDRICAL>& config
    )
    {
        // axisymmetric cylindrical volume calculation
        const real rl  = face_position(coord, Dims - 1, Dir::W, config);
        const real rr  = face_position(coord, Dims - 1, Dir::E, config);
        const real rdr = (rr * rr - rl * rl) / 2.0;

        const real dphi = 2.0 * std::numbers::pi;
        const real dz   = face_position(coord, Dims - 3, Dir::E, config) -
                        face_position(coord, Dims - 3, Dir::W, config);

        return rdr * dphi * dz;
    }

    template <std::uint64_t Dims>
    DEV real volume(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::PLANAR_CYLINDRICAL>& config
    )
    {
        // planar cylindrical volume calculation
        const real rl  = face_position(coord, Dims - 1, Dir::W, config);
        const real rr  = face_position(coord, Dims - 1, Dir::E, config);
        const real rdr = (rr * rr - rl * rr) / 2.0;

        const real dphi = face_position(coord, Dims - 2, Dir::E, config) -
                          face_position(coord, Dims - 2, Dir::W, config);
        const real dz = 1.0;

        return rdr * dphi * dz;
    }

    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // FACE AREA SPECIALIZATIONS
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    template <std::uint64_t Dims>
    DEV real face_area(
        const iarray<Dims>& coord,
        std::uint64_t comp,
        Dir /*dir*/,
        const mesh_config_t<Dims, Geometry::CARTESIAN>& config
    )
    {
        const auto widths = cell_widths(coord, config);
        real area         = 1.0;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            if (ii != comp) {
                area *= widths[ii];
            }
        }
        return area;
    }

    template <std::uint64_t Dims>
    DEV real face_area(
        const iarray<Dims>& coord,
        std::uint64_t comp,
        Dir dir,
        const mesh_config_t<Dims, Geometry::SPHERICAL>& config
    )
    {
        if (comp == Dims - 1) {
            // radial face
            const real r = face_position(coord, Dims - 1, dir, config);

            real dtheta = 2.0;
            if constexpr (Dims > 1) {
                const real tl = face_position(coord, Dims - 2, Dir::W, config);
                const real tr = face_position(coord, Dims - 2, Dir::E, config);
                dtheta        = std::cos(tl) - std::cos(tr);
            }

            real dphi = 2.0 * std::numbers::pi;
            if constexpr (Dims > 2) {
                dphi = face_position(coord, Dims - 3, Dir::E, config) -
                       face_position(coord, Dims - 3, Dir::W, config);
            }

            return r * r * dtheta * dphi;
        }
        else if (comp == Dims - 2 && Dims > 1) {
            // theta face
            const real rl    = face_position(coord, Dims - 1, Dir::W, config);
            const real rr    = face_position(coord, Dims - 1, Dir::E, config);
            const real theta = face_position(coord, Dims - 2, dir, config);

            real dphi = 2.0 * std::numbers::pi;
            if constexpr (Dims > 2) {
                dphi = face_position(coord, Dims - 3, Dir::E, config) -
                       face_position(coord, Dims - 3, Dir::W, config);
            }

            return 0.5 * (rr * rr - rl * rl) * std::sin(theta) * dphi;
        }
        else if (comp == Dims - 3 && Dims > 2) {
            // phi face
            const real rl     = face_position(coord, Dims - 1, Dir::W, config);
            const real rr     = face_position(coord, Dims - 1, Dir::E, config);
            const real tl     = face_position(coord, Dims - 2, Dir::W, config);
            const real tr     = face_position(coord, Dims - 2, Dir::E, config);
            const real dtheta = std::cos(tl) - std::cos(tr);

            return 0.5 * (rr * rr - rl * rl) * dtheta;
        }
        return 0.0;
    }

    template <std::uint64_t Dims>
    DEV real face_area(
        const iarray<Dims>& coord,
        std::uint64_t comp,
        Dir dir,
        const mesh_config_t<Dims, Geometry::CYLINDRICAL>& config
    )
    {
        // cylindrical area element is r * dr * dphi
        if (comp == Dims - 1) {
            // radial face
            const real r = face_position(coord, Dims - 1, dir, config);
            real dphi    = 2.0 * std::numbers::pi;
            if constexpr (Dims > 1) {
                dphi = face_position(coord, Dims - 2, Dir::E, config) -
                       face_position(coord, Dims - 2, Dir::W, config);
            }
            real dz = 1.0;
            if constexpr (Dims > 2) {
                dz = face_position(coord, Dims - 3, Dir::E, config) -
                     face_position(coord, Dims - 3, Dir::W, config);
            }
            return r * dz * dphi;
        }
        else if (comp == Dims - 2 && Dims > 1) {
            // phi face
            const real rl = face_position(coord, Dims - 1, Dir::W, config);
            const real rr = face_position(coord, Dims - 1, Dir::E, config);

            real dz = 1.0;
            if constexpr (Dims > 2) {
                dz = face_position(coord, Dims - 3, Dir::E, config) -
                     face_position(coord, Dims - 3, Dir::W, config);
            }
            return (rr - rl) * dz;
        }
        else {
            const real rl   = face_position(coord, Dims - 1, Dir::W, config);
            const real rr   = face_position(coord, Dims - 1, Dir::E, config);
            const real dphi = face_position(coord, Dims - 2, Dir::E, config) -
                              face_position(coord, Dims - 2, Dir::W, config);
            return 0.5 * (rr * rr - rl * rl) * dphi;
        }
    }

    template <std::uint64_t Dims>
    DEV real face_area(
        const iarray<Dims>& coord,
        std::uint64_t comp,
        Dir dir,
        const mesh_config_t<Dims, Geometry::AXIS_CYLINDRICAL>& config
    )
    {
        // axisymmetric cylindrical area element is r * dr * dphi
        if (comp == Dims - 1) {
            // radial face
            const real r    = face_position(coord, Dims - 1, dir, config);
            const real dphi = 2.0 * std::numbers::pi;
            const real dz   = face_position(coord, Dims - 2, Dir::E, config) -
                            face_position(coord, Dims - 2, Dir::W, config);
            return r * dz * dphi;
        }
        else {
            // z face
            const real rl   = face_position(coord, Dims - 1, Dir::W, config);
            const real rr   = face_position(coord, Dims - 1, Dir::E, config);
            const real rm   = centroid(coord, config)[Dims - 1];
            const real dphi = 2.0 * std::numbers::pi;
            return (rr - rl) * rm * dphi;
        }
    }

    template <std::uint64_t Dims>
    DEV real face_area(
        const iarray<Dims>& coord,
        std::uint64_t comp,
        Dir dir,
        const mesh_config_t<Dims, Geometry::PLANAR_CYLINDRICAL>& config
    )
    {
        // planar cylindrical area element is r * dr * dphi
        if (comp == Dims - 1) {
            // radial face
            const real r    = face_position(coord, Dims - 1, dir, config);
            const real dphi = face_position(coord, Dims - 2, Dir::E, config) -
                              face_position(coord, Dims - 2, Dir::W, config);
            const real dz = 1.0;
            return r * dphi * dz;
        }
        else {
            // phi face
            const real rl = face_position(coord, Dims - 1, Dir::W, config);
            const real rr = face_position(coord, Dims - 1, Dir::E, config);
            const real dz = 1.0;
            return (rr - rl) * dz;
        }
    }

    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // QUERY FUNCTIONS
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    template <std::uint64_t Dims>
    DEV bool at_pole(
        const iarray<Dims>&,
        const mesh_config_t<Dims, Geometry::CARTESIAN>&
    )
    {
        return false;   // Cartesian geometry has no poles
    }

    template <std::uint64_t Dims>
    DEV bool at_pole(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::SPHERICAL>& config
    )
    {
        if constexpr (Dims < 2) {
            return false;   // no poles in 1D or lower (kinda)
        }

        constexpr real POLAR_TOL = 1.e-10;
        const real tl = face_position(coord, Dims - 2, Dir::W, config);
        const real tr = face_position(coord, Dims - 2, Dir::E, config);

        return std::abs(std::sin(tl)) < POLAR_TOL ||
               std::abs(std::sin(tr)) < POLAR_TOL;
    }

    template <std::uint64_t Dims>
    DEV bool at_pole(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::CYLINDRICAL>& config
    )
    {
        if constexpr (Dims < 2) {
            return false;   // no poles in 1D or lower (kinda)
        }

        constexpr real POLAR_TOL = 1.e-10;
        const real rl = face_position(coord, Dims - 1, Dir::W, config);
        const real rr = face_position(coord, Dims - 1, Dir::E, config);

        return std::abs(rl) < POLAR_TOL || std::abs(rr) < POLAR_TOL;
    }

    template <std::uint64_t Dims>
    DEV bool at_pole(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::AXIS_CYLINDRICAL>& config
    )
    {
        if constexpr (Dims < 2) {
            return false;   // no poles in 1D or lower (kinda)
        }

        constexpr real POLAR_TOL = 1.e-10;
        const real rl = face_position(coord, Dims - 1, Dir::W, config);
        const real rr = face_position(coord, Dims - 1, Dir::E, config);

        return std::abs(rl) < POLAR_TOL || std::abs(rr) < POLAR_TOL;
    }

    template <std::uint64_t Dims>
    DEV bool at_pole(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::PLANAR_CYLINDRICAL>& config
    )
    {
        if constexpr (Dims < 2) {
            return false;   // no poles in 1D or lower (kinda)
        }

        constexpr real POLAR_TOL = 1.e-10;
        const real rl = face_position(coord, Dims - 1, Dir::W, config);
        const real rr = face_position(coord, Dims - 1, Dir::E, config);

        return std::abs(rl) < POLAR_TOL || std::abs(rr) < POLAR_TOL;
    }

    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // GEOMETRY SPECIALIZATIONS
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims> to_cartesian(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::CARTESIAN>& config
    )
    {
        // Cartesian coordinates are already in Cartesian space
        return centroid(coord, config);
    }

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims> to_cartesian(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::CYLINDRICAL>& config
    )
    {
        return vecops::cylindrical_to_cartesian(centroid(coord, config));
    }

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims> to_cartesian(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::AXIS_CYLINDRICAL>& config
    )
    {
        return vecops::cylindrical_to_cartesian(centroid(coord, config));
    }

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims> to_cartesian(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::PLANAR_CYLINDRICAL>& config
    )
    {
        return vecops::cylindrical_to_cartesian(centroid(coord, config));
    }

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims> to_cartesian(
        const iarray<Dims>& coord,
        const mesh_config_t<Dims, Geometry::SPHERICAL>& config
    )
    {
        return vecops::spherical_to_cartesian(centroid(coord, config));
    }

    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // GEOMETRIC SOURCE TERMS
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    template <is_hydro_primitive_c prim_t, std::uint64_t Dims>
    DEV constexpr auto geometric_source_terms(
        const prim_t&,
        const iarray<Dims>&,
        const mesh_config_t<Dims, Geometry::CARTESIAN>&,
        real
    )
    {
        return typename prim_t::counterpart_t{};
    }

    template <is_hydro_primitive_c prim_t, std::uint64_t Dims>
    DEV constexpr auto geometric_source_terms(
        const prim_t& prim,
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, Geometry::SPHERICAL>& config,
        real gamma
    )
    {
        const auto cent  = centroid(coords, config);
        const auto theta = Dims >= 2 ? cent[Dims - 2] : 0.5 * std::numbers::pi;
        const auto r     = cent[Dims - 1];
        const real sint  = std::sin(theta);
        const real cot   = std::cos(theta) / sint;

        // grab central primitives
        const real v1    = proper_velocity(prim, 1);
        const real v2    = proper_velocity(prim, 2);
        const real v3    = proper_velocity(prim, 3);
        const real pt    = total_pressure(prim);
        const auto bmu   = magnetic_four_vector(prim);
        const real wt    = enthalpy_density(prim, gamma);
        const real gam2  = lorentz_factor_squared(prim);
        const real wgam2 = wt * gam2;

        const auto dv = volume(coords, config);

        // geometric source terms in momentum
        using cons_t = typename prim_t::counterpart_t;
        cons_t cons;
        for (std::uint64_t qq = 0; qq < Dims; qq++) {
            if (qq == 0) {
                const auto aL = face_area(coords, Dims - 1, Dir::W, config);
                const auto aR = face_area(coords, Dims - 1, Dir::E, config);
                cons.mom[qq]  = pt * (aR - aL) / dv +
                               wgam2 * (v2 * v2 + v3 * v3) / r -
                               (bmu[1] * bmu[1] + bmu[3] * bmu[3]) / r;
            }
            else if (qq == 1) {
                const auto aL = face_area(coords, Dims - 2, Dir::W, config);
                const auto aR = face_area(coords, Dims - 2, Dir::E, config);

                cons.mom[qq] = pt * (aR - aL) / dv -
                               wgam2 * (v2 * v1 - v3 * v3 * cot) / r +
                               (bmu[1] * bmu[0] - bmu[3] * bmu[3] * cot) / r;
            }
            else {
                cons.mom[qq] = -wgam2 * v3 * (v1 + cot * v2) / r +
                               bmu[3] * (bmu[0] + cot * bmu[1]) / r;
            }
        }
        return cons;
    }

    template <is_hydro_primitive_c prim_t, std::uint64_t Dims>
    DEV constexpr auto geometric_source_terms(
        const prim_t& prim,
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, Geometry::CYLINDRICAL>& config,
        real gamma
    )
    {
        const real v1    = proper_velocity(prim, 1);
        const real v2    = proper_velocity(prim, 2);
        const real pt    = total_pressure(prim);
        const auto bmu   = magnetic_four_vector(prim);
        const real wt    = enthalpy_density(prim, gamma);
        const real gam2  = lorentz_factor_squared(prim);
        const real wgam2 = wt * gam2;
        const auto r     = centroid(coords, config)[Dims - 1];

        using cons_t = typename prim_t::counterpart_t;
        cons_t cons;
        for (std::uint64_t qq = 0; qq < Dims; qq++) {
            if (qq == 0) {
                cons.mom[qq] = (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / r;
            }
            else if (qq == 1) {
                cons.mom[qq] = -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) / r;
            }
        }
        return cons;
    }

    template <is_hydro_primitive_c prim_t, std::uint64_t Dims>
    DEV constexpr auto geometric_source_terms(
        const prim_t& prim,
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, Geometry::AXIS_CYLINDRICAL>& config,
        real /*gamma*/
    )
    {
        const real pt  = total_pressure(prim);
        const auto bmu = magnetic_four_vector(prim);
        const auto r   = centroid(coords, config)[Dims - 1];

        using cons_t = typename prim_t::counterpart_t;
        cons_t cons;
        for (std::uint64_t qq = 0; qq < Dims; qq++) {
            if (qq == 0) {
                cons.mom[qq] = (pt - bmu[1] * bmu[1]) / r;
            }
        }
        return cons;
    }

    template <is_hydro_primitive_c prim_t, std::uint64_t Dims>
    DEV constexpr auto geometric_source_terms(
        const prim_t& prim,
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, Geometry::PLANAR_CYLINDRICAL>& config,
        real gamma
    )
    {
        const real v1    = proper_velocity(prim, 1);
        const real v2    = proper_velocity(prim, 2);
        const real pt    = total_pressure(prim);
        const auto bmu   = magnetic_four_vector(prim);
        const real wt    = enthalpy_density(prim, gamma);
        const real gam2  = lorentz_factor_squared(prim);
        const real wgam2 = wt * gam2;
        const auto r     = centroid(coords, config)[Dims - 1];

        using cons_t = typename prim_t::counterpart_t;
        cons_t cons;
        for (std::uint64_t qq = 0; qq < Dims; qq++) {
            if (qq == 0) {
                cons.mom[qq] = (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / r;
            }
            else if (qq == 1) {
                cons.mom[qq] = -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) / r;
            }
        }
        return cons;   // No v3 component in planar cylindrical case
    }

    //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    // MAX / MIN CELL WIDTH
    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    template <std::uint64_t Dims, Geometry G>
    DEV real min_cell_width(
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, G>& config
    )
    {
        const auto widths = cell_widths(coords, config);
        real min_width    = widths[0];
        for (std::uint64_t ii = 1; ii < Dims; ++ii) {
            if (widths[ii] < min_width) {
                min_width = widths[ii];
            }
        }
        return min_width;
    }

    template <std::uint64_t Dims, Geometry G>
    DEV real max_cell_width(
        const iarray<Dims>& coords,
        const mesh_config_t<Dims, G>& config
    )
    {
        const auto widths = cell_widths(coords, config);
        real max_width    = widths[0];
        for (std::uint64_t ii = 1; ii < Dims; ++ii) {
            if (widths[ii] > max_width) {
                max_width = widths[ii];
            }
        }
        return max_width;
    }

    // ============================================================================
    // SHARED COORDINATE FUNCTIONS (geometry-independent)
    // ============================================================================
    template <std::uint64_t Dims, Geometry G>
    DEV constexpr real face_position(
        const iarray<Dims>& coord,
        std::uint64_t direction,
        Dir dir,
        const mesh_config_t<Dims, G>& config
    )
    {
        if (direction >= Dims) {
            return 0.0;
        }

        const auto bounds_min  = config.current_bounds_min();
        const auto bounds_max  = config.current_bounds_max();
        const auto active_dims = config.shape;

        const real min_val      = bounds_min[direction];
        const real max_val      = bounds_max[direction];
        const std::uint64_t idx = coord[direction];

        if (config.spacing_types[direction] == Cellspacing::LINEAR) {
            const real dx = (max_val - min_val) / active_dims[direction];
            return dir == Dir::E ? min_val + (idx + 1) * dx
                                 : min_val + idx * dx;
        }
        else {
            const real dlogx =
                std::log10(max_val / min_val) / active_dims[direction];
            return dir == Dir::E ? min_val * std::pow(10.0, (idx + 1) * dlogx)
                                 : min_val * std::pow(10.0, (idx + 0) * dlogx);
        }
    }

    template <std::uint64_t Dims, Geometry G>
    DEV constexpr vector_t<real, Dims>
    centroid(const iarray<Dims>& coord, const mesh_config_t<Dims, G>& config)
    {
        vector_t<real, Dims> result;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            const real left  = face_position(coord, ii, Dir::W, config);
            const real right = face_position(coord, ii, Dir::E, config);
            result[ii]       = (config.spacing_types[ii] == Cellspacing::LOG)
                                   ? std::sqrt(left * right)
                                   : 0.5 * (left + right);
        }
        return result;
    }

    template <std::uint64_t Dims, Geometry G>
    DEV constexpr vector_t<real, Dims>
    cell_widths(const iarray<Dims>& coord, const mesh_config_t<Dims, G>& config)
    {
        vector_t<real, Dims> widths;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            widths[ii] = face_position(coord, ii, Dir::E, config) -
                         face_position(coord, ii, Dir::W, config);
        }
        return widths;
    }

    template <std::uint64_t Dims, Geometry G>
    DEV constexpr real face_velocity(
        const iarray<Dims>& coord,
        std::uint64_t direction,
        const mesh_config_t<Dims, G>& config,
        Dir side = Dir::W
    )
    {
        if (!config.mesh_motion) {
            return 0.0;
        }
        const real face_pos = face_position(coord, direction, side, config);

        if (config.homologous) {
            // homologous velocity: v = H * r
            return config.expansion_rate * face_pos;
        }
        else {
            // uniform velocity
            return config.expansion_rate;
        }
    }

}   // namespace simbi::mesh

#endif   // COMPILE_TIME_GEOMETRY_HPP
