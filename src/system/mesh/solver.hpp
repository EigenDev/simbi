

#ifndef COMPILE_TIME_GEOMETRY_HPP
#define COMPILE_TIME_GEOMETRY_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/base/coordinate.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"
#include "data/containers/vector.hpp"
#include "mesh_config.hpp"
#include "physics/hydro/physics.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace simbi::mesh {
    using namespace simbi::base;
    using namespace simbi::concepts;
    using namespace simbi::hydro;
    // ========================================================================
    // forward declarations
    template <std::uint64_t Dims>
    DEV constexpr real face_position(
        const uarray<Dims>& coord,
        std::uint64_t direction,
        Dir dir,
        const mesh_config_t<Dims>& config
    );

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims>
    centroid(const uarray<Dims>& coord, const mesh_config_t<Dims>& config);

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims>
    cell_widths(const uarray<Dims>& coord, const mesh_config_t<Dims>& config);

    // base template - never instantiated
    template <Geometry G, std::uint64_t Dims>
    struct geometry_functions;

    // SPHERICAL SPECIALIZATION
    template <std::uint64_t Dims>
    struct geometry_functions<Geometry::SPHERICAL, Dims> {

        DEV static constexpr real
        volume(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            const real rl = face_position(coord, 0, Dir::W, config);
            const real rr = face_position(coord, 0, Dir::E, config);
            const real dr = (rr * rr * rr - rl * rl * rl) / 3.0;

            real dtheta = 2.0;
            if constexpr (Dims > 1) {
                const real tl = face_position(coord, 1, Dir::W, config);
                const real tr = face_position(coord, 1, Dir::E, config);
                dtheta        = std::cos(tl) - std::cos(tr);
            }

            real dphi = 2.0 * std::numbers::pi;
            if constexpr (Dims > 2) {
                dphi = face_position(coord, 2, Dir::E, config) -
                       face_position(coord, 2, Dir::W, config);
            }

            return dr * dtheta * dphi;
        }

        DEV static constexpr real face_area(
            const uarray<Dims>& coord,
            std::uint64_t face_dir,
            Dir dir,
            const mesh_config_t<Dims>& config
        )
        {
            if (face_dir == 0) {
                // radial face
                const real r = face_position(coord, 0, dir, config);

                real dtheta = 2.0;
                if constexpr (Dims > 1) {
                    const real tl = face_position(coord, 1, Dir::W, config);
                    const real tr = face_position(coord, 1, Dir::E, config);
                    dtheta        = std::cos(tl) - std::cos(tr);
                }

                real dphi = 2.0 * std::numbers::pi;
                if constexpr (Dims > 2) {
                    dphi = face_position(coord, 2, Dir::E, config) -
                           face_position(coord, 2, Dir::W, config);
                }

                return r * r * dtheta * dphi;
            }
            else if (face_dir == 1 && Dims > 1) {
                // Theta face
                const real rl    = face_position(coord, 0, Dir::E, config);
                const real rr    = face_position(coord, 0, Dir::W, config);
                const real theta = face_position(coord, 1, dir, config);

                real dphi = 2.0 * std::numbers::pi;
                if constexpr (Dims > 2) {
                    dphi = face_position(coord, 2, Dir::E, config) -
                           face_position(coord, 2, Dir::W, config);
                }

                return 0.5 * (rr * rr - rl * rl) * std::sin(theta) * dphi;
            }
            else if (face_dir == 2 && Dims > 2) {
                // Phi face
                const real rl     = face_position(coord, 0, Dir::W, config);
                const real rr     = face_position(coord, 0, Dir::E, config);
                const real tl     = face_position(coord, 1, Dir::W, config);
                const real tr     = face_position(coord, 1, Dir::E, config);
                const real dtheta = std::cos(tl) - std::cos(tr);

                return 0.5 * (rr * rr - rl * rl) * dtheta;
            }
            return 0.0;
        }

        DEV static constexpr auto to_cartesian(
            const uarray<Dims>& coord,
            const mesh_config_t<Dims>& config
        )
        {
            return vecops::spherical_to_cartesian(centroid(coord, config));
        }

        DEV static constexpr bool
        at_pole(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            if constexpr (Dims < 2) {
                return false;
            }

            constexpr real POLAR_TOL = 1.e-10;
            const real tl            = face_position(coord, 1, Dir::W, config);
            const real tr            = face_position(coord, 1, Dir::E, config);

            return std::abs(std::sin(tl)) < POLAR_TOL ||
                   std::abs(std::sin(tr)) < POLAR_TOL;
        }

        template <is_hydro_primitive_c prim_t>
        DEV static constexpr auto geometric_source_terms(
            const prim_t& prim,
            const uarray<Dims>& coords,
            const mesh_config_t<Dims>& config,
            real gamma
        )
        {
            const auto cent  = centroid(coords, config);
            const auto theta = cent[1];
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
                const auto r = cent[0];
                if (qq == 0) {
                    const auto aL    = face_area(coords, 0, Dir::W, config);
                    const auto aR    = face_area(coords, 0, Dir::E, config);
                    cons.mom[qq + 1] = pt * (aR - aL) / dv +
                                       wgam2 * (v2 * v2 + v3 * v3) / r -
                                       (bmu[1] * bmu[1] + bmu[3] * bmu[3]) / r;
                }
                else if (qq == 1) {
                    const auto aL = face_area(coords, 1, Dir::W, config);
                    const auto aR = face_area(coords, 1, Dir::E, config);
                    cons.mom[qq + 1] =
                        pt * (aR - aL) / dv -
                        wgam2 * (v2 * v1 - v3 * v3 * cot) / r +
                        (bmu[1] * bmu[0] - bmu[3] * bmu[3] * cot) / r;
                }
                else {
                    cons.mom[qq + 1] = -wgam2 * v3 * (v1 + cot * v2) / r +
                                       bmu[3] * (bmu[0] + cot * bmu[1]) / r;
                }
            }
            return cons;
        }
    };

    // CARTESIAN SPECIALIZATION
    template <std::uint64_t Dims>
    struct geometry_functions<Geometry::CARTESIAN, Dims> {

        DEV static constexpr real
        volume(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            const auto widths = cell_widths(coord, config);
            real vol          = 1.0;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                vol *= widths[ii];
            }
            return vol;
        }

        DEV static constexpr real face_area(
            const uarray<Dims>& coord,
            std::uint64_t face_dir,
            Dir,
            const mesh_config_t<Dims>& config
        )
        {
            const auto widths = cell_widths(coord, config);
            real area         = 1.0;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (ii != face_dir) {
                    area *= widths[ii];
                }
            }
            return area;
        }

        DEV static constexpr real to_cartesian(
            const uarray<Dims>& coord,
            const mesh_config_t<Dims>& config
        )
        {
            return centroid(coord, config);
        }

        DEV static constexpr bool
        at_pole(const uarray<Dims>&, const mesh_config_t<Dims>&)
        {
            return false;   // no poles in Cartesian
        }

        template <is_hydro_primitive_c prim_t>
        DEV static constexpr auto geometric_source_terms(
            const prim_t& /*prim*/,
            const uarray<Dims>& /*coords*/,
            const mesh_config_t<Dims>& /*config*/,
            real /*gamma*/
        )
        {
            // Cartesian geometry has no geometric source terms
            return typename prim_t::counterpart_t{};
        }
    };

    // CYLINDRICAL SPECIALIZATION
    template <std::uint64_t Dims>
    struct geometry_functions<Geometry::CYLINDRICAL, Dims> {

        DEV static constexpr real
        volume(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            const real rl  = face_position(coord, 0, Dir::W, config);
            const real rr  = face_position(coord, 0, Dir::E, config);
            const real rdr = 0.5 * (rr * rr - rl * rl);

            real dphi = 2.0 * std::numbers::pi;
            if constexpr (Dims > 1) {
                dphi = face_position(coord, 1, Dir::E, config) -
                       face_position(coord, 1, Dir::W, config);
            }
            real dz = 1.0;
            if constexpr (Dims > 1) {
                dz = face_position(coord, 2, Dir::E, config) -
                     face_position(coord, 2, Dir::W, config);
            }

            return rdr * dphi * dz;
        }

        DEV static constexpr real face_area(
            const uarray<Dims>& coord,
            std::uint64_t face_dir,
            Dir dir,
            const mesh_config_t<Dims>& config
        )
        {
            // cylindrical area element is r * dr * dphi
            // static_assert(
            //     face_dir < Dims,
            //     "face direction must be less than or equal to Dims"
            // );
            if (face_dir == 0) {
                // radial face
                const real r = face_position(coord, 0, dir, config);
                real dphi    = 2.0 * std::numbers::pi;
                if constexpr (Dims > 1) {
                    dphi = face_position(coord, 1, Dir::E, config) -
                           face_position(coord, 1, Dir::W, config);
                }
                real dz = 1.0;
                if constexpr (Dims > 2) {
                    dz = face_position(coord, 2, Dir::E, config) -
                         face_position(coord, 2, Dir::W, config);
                }
                return r * dz * dphi;
            }
            else if (face_dir == 1 && Dims > 1) {
                // phi face
                const real rl = face_position(coord, 0, Dir::W, config);
                const real rr = face_position(coord, 0, Dir::E, config);

                real dz = 1.0;
                if constexpr (Dims > 2) {
                    dz = face_position(coord, 2, Dir::E, config) -
                         face_position(coord, 2, Dir::W, config);
                }
                return (rr - rl) * dz;
            }
            else {
                const real rmean = centroid(coord, config)[0];
                const real rl    = face_position(coord, 0, Dir::W, config);
                const real rr    = face_position(coord, 0, Dir::E, config);

                const real dphi = face_position(coord, 1, Dir::E, config) -
                                  face_position(coord, 1, Dir::W, config);
                return rmean * dphi * (rr - rl);
            }
        }

        DEV static constexpr auto to_cartesian(
            const uarray<Dims>& coord,
            const mesh_config_t<Dims>& config
        )
        {
            return vecops::cylindrical_to_cartesian(centroid(coord, config));
        }

        DEV static constexpr bool
        at_pole(const uarray<Dims>& coords, const mesh_config_t<Dims>& config)
        {
            const auto rmean = centroid(coords, config)[0];
            return helpers::goes_to_zero(rmean);
        }

        template <is_hydro_primitive_c prim_t>
        DEV static constexpr auto geometric_source_terms(
            const prim_t& prim,
            const uarray<Dims>& coords,
            const mesh_config_t<Dims>& config,
            real gamma
        )
        {
            // special care must be taken for axisymmetry
            // or cylindrical-polar coordinates. In axisymmetry,
            // the phi velocity is zero, but out of convenience, we
            // store the z component in the second component of the
            // velocity vector. This is done to avoid having to
            // rearrange the velocity vector in the axisymmetric
            // case
            const real v1    = proper_velocity(prim, 1);
            const real v2    = proper_velocity(prim, 2);
            const real pt    = total_pressure(prim);
            const auto bmu   = magnetic_four_vector(prim);
            const real wt    = enthalpy_density(prim, gamma);
            const real gam2  = lorentz_factor_squared(prim);
            const real wgam2 = wt * gam2;
            const auto cent  = centroid(coords, config);

            using cons_t = typename prim_t::counterpart_t;
            cons_t cons;
            for (std::uint64_t qq = 0; qq < Dims; qq++) {
                if (qq == 0) {
                    cons.mom[qq + 1] =
                        (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / cent[0];
                }
                else if (qq == 1) {
                    cons.mom[qq + 1] =
                        -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) / cent[0];
                }
            }
            return cons;
        }
    };

    // AXIS-CYLINDRICAL SPECIALIZATION (r-z plane, no dphi)
    template <std::uint64_t Dims>
    struct geometry_functions<Geometry::AXIS_CYLINDRICAL, Dims> {

        DEV static constexpr real
        volume(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            // Axisymmetric volume calculation
            const real rl  = face_position(coord, 0, Dir::W, config);
            const real rr  = face_position(coord, 0, Dir::E, config);
            const real rdr = 0.5 * (rr * rr - rl * rl);

            const real dz = face_position(coord, 1, Dir::E, config) -
                            face_position(coord, 1, Dir::W, config);

            return rdr * dz;   // No dphi in axisymmetric case
        }

        DEV static constexpr real face_area(
            const uarray<Dims>& coord,
            std::uint64_t face_dir,
            Dir dir,
            const mesh_config_t<Dims>& config
        )
        {
            // static_assert(
            //     face_dir < Dims,
            //     "face direction must be less than Dims"
            // );
            if (face_dir == 0) {
                // radial face
                const real r  = face_position(coord, 0, dir, config);
                const real dz = face_position(coord, 1, Dir::E, config) -
                                face_position(coord, 1, Dir::W, config);
                return r * dz;   // no dphi in axisymmetric case
            }
            else {
                // z face
                const real rl = face_position(coord, 0, Dir::W, config);
                const real rr = face_position(coord, 0, Dir::E, config);
                return 0.5 * (rr * rr - rl * rl);
            }
        }

        DEV static constexpr auto to_cartesian(
            const uarray<Dims>& coord,
            const mesh_config_t<Dims>& config
        )
        {
            return vecops::cylindrical_to_cartesian(centroid(coord, config));
        }

        DEV static constexpr bool
        at_pole(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            const auto rmean = centroid(coord, config);
            return helpers::goes_to_zero(rmean);
        }

        template <is_hydro_primitive_c prim_t>
        DEV static constexpr auto geometric_source_terms(
            const prim_t& prim,
            const uarray<Dims>& coords,
            const mesh_config_t<Dims>& config,
            real gamma
        )
        {
            const real v1    = proper_velocity(prim, 1);
            const real v2    = 0.0;   // v2 (vphi) is zero in axisymmetric case
            const real pt    = total_pressure(prim);
            const auto bmu   = magnetic_four_vector(prim);
            const real wt    = enthalpy_density(prim, gamma);
            const real gam2  = lorentz_factor_squared(prim);
            const real wgam2 = wt * gam2;
            const auto cent  = centroid(coords, config);

            using cons_t = typename prim_t::counterpart_t;
            cons_t cons;
            for (std::uint64_t qq = 0; qq < Dims; qq++) {
                if (qq == 0) {
                    cons.mom[qq + 1] =
                        (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / cent[0];
                }
                else if (qq == 1) {
                    cons.mom[qq + 1] =
                        -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) / cent[0];
                }
            }
            return cons;   // No v3 component in axisymmetric case
        }
    };

    // PLANAR-CYLINDRICAL SPECIALIZATION (r-phi plane, no dz)
    template <std::uint64_t Dims>
    struct geometry_functions<Geometry::PLANAR_CYLINDRICAL, Dims> {
        DEV static constexpr real
        volume(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            // Planar cylindrical volume calculation
            const real rl  = face_position(coord, 0, Dir::W, config);
            const real rr  = face_position(coord, 0, Dir::E, config);
            const real rdr = 0.5 * (rr * rr - rl * rl);

            const real dphi = face_position(coord, 1, Dir::E, config) -
                              face_position(coord, 1, Dir::W, config);

            return rdr * dphi;   // No dz in planar cylindrical case
        }

        DEV static constexpr real face_area(
            const uarray<Dims>& coord,
            std::uint64_t face_dir,
            Dir dir,
            const mesh_config_t<Dims>& config
        )
        {
            return geometry_functions<Geometry::CYLINDRICAL, Dims>::face_area(
                coord,
                face_dir,
                dir,
                config
            );
        }

        DEV static constexpr auto to_cartesian(
            const uarray<Dims>& coord,
            const mesh_config_t<Dims>& config
        )
        {
            return vecops::cylindrical_to_cartesian(centroid(coord, config));
        }

        DEV static constexpr bool
        at_pole(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
        {
            const auto rmean = centroid(coord, config);
            return helpers::goes_to_zero(rmean);
        }

        template <is_hydro_primitive_c prim_t>
        DEV static constexpr auto geometric_source_terms(
            const prim_t& prim,
            const uarray<Dims>& coords,
            const mesh_config_t<Dims>& config,
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
            const auto cent  = centroid(coords, config);

            using cons_t = typename prim_t::counterpart_t;
            cons_t cons;
            for (std::uint64_t qq = 0; qq < Dims; qq++) {
                if (qq == 0) {
                    cons.mom[qq + 1] =
                        (wgam2 * v2 * v2 - bmu[1] * bmu[1] + pt) / cent[0];
                }
                else if (qq == 1) {
                    cons.mom[qq + 1] =
                        -(wgam2 * v1 * v2 - bmu[0] * bmu[1]) / cent[0];
                }
            }
            return cons;   // No v3 component in planar cylindrical case
        }
    };

    // ============================================================================
    // SHARED COORDINATE FUNCTIONS (geometry-independent)
    // ============================================================================
    template <std::uint64_t Dims>
    DEV constexpr real face_position(
        const uarray<Dims>& coord,
        std::uint64_t direction,
        Dir dir,
        const mesh_config_t<Dims>& config
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
                                 : min_val * std::pow(10.0, idx * dlogx);
        }
    }

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims>
    centroid(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
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

    template <std::uint64_t Dims>
    DEV constexpr vector_t<real, Dims>
    cell_widths(const uarray<Dims>& coord, const mesh_config_t<Dims>& config)
    {
        vector_t<real, Dims> widths;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            widths[ii] = face_position(coord, ii, Dir::E, config) -
                         face_position(coord, ii, Dir::W, config);
        }
        return widths;
    }

    template <std::uint64_t Dims>
    DEV constexpr real face_velocity(
        const uarray<Dims>& coord,
        std::uint64_t direction,
        Dir side,
        const mesh_config_t<Dims>& config
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

    // ============================================================================
    // COMPILE-TIME GEOMETRY SOLVER TEMPLATE
    // ============================================================================
    template <std::uint64_t Dims, Geometry G>
    struct geometry_solver_t {
        using geom = geometry_functions<G, Dims>;
        mesh_config_t<Dims> config;

        DEV real volume(const uarray<Dims>& coord) const
        {
            return geom::volume(coord, config);
        }

        DEV real face_area(
            const uarray<Dims>& coord,
            std::uint64_t face_dir,
            Dir dir
        ) const
        {
            return geom::face_area(coord, face_dir, dir, config);
        }

        DEV real
        distance(const uarray<Dims>& coord, const uarray<Dims>& target) const
        {
            return geom::distance(coord, target, config);
        }

        DEV bool at_pole(const uarray<Dims>& coord) const
        {
            return geom::at_pole(coord, config);
        }

        DEV vector_t<real, Dims> centroid(const uarray<Dims>& coord) const
        {
            return simbi::mesh::centroid(coord, config);
        }

        DEV vector_t<real, Dims> cell_widths(const uarray<Dims>& coord) const
        {
            return simbi::mesh::cell_widths(coord, config);
        }

        // update mesh state
        void advance_mesh(real dt, real expansion_rate)
        {
            if (config.mesh_motion) {
                config.expansion_factor += dt * expansion_rate;
            }
        }

        DEV real min_cell_width(const vector_t<real, Dims>& coords) const
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

        DEV real max_cell_width(const vector_t<real, Dims>& coords) const
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

        DEV vector_t<real, Dims>
        cartesian_centroid(const uarray<Dims>& coord) const
        {
            return geom::to_cartesian(coord, config);
        }

        template <is_hydro_primitive_c prim_t>
        DEV auto geometric_sources(
            const uarray<Dims>& coords,
            const prim_t& prim,
            real gamma
        ) const
        {
            return geom::geometric_source_terms(prim, coords, config, gamma);
        }

        DEV real face_velocity(
            const uarray<Dims>& coord,
            std::uint64_t direction,
            Dir side = Dir::W
        ) const
        {
            return simbi::mesh::face_velocity(coord, direction, side, config);
        }

        auto expand_mesh(real time, real dt)
        {
            if (!config.mesh_motion) {
                return;
            }

            // update expansion factor based on time and dt
            config = config.update_expansion(time, dt);
        }

        // accessors
        static constexpr Geometry geometry() { return G; }
        static constexpr std::uint64_t dimensions() { return Dims; }
    };

    // ============================================================================
    // TYPE ALIASES FOR COMMON GEOMETRIES
    // ============================================================================

    template <std::uint64_t Dims>
    using spherical_t = geometry_solver_t<Dims, Geometry::SPHERICAL>;

    template <std::uint64_t Dims>
    using cartesian_t = geometry_solver_t<Dims, Geometry::CARTESIAN>;

    template <std::uint64_t Dims>
    using cylindrical_t = geometry_solver_t<Dims, Geometry::CYLINDRICAL>;

    template <std::uint64_t Dims>
    using axis_cylindrical_t =
        geometry_solver_t<Dims, Geometry::AXIS_CYLINDRICAL>;

    template <std::uint64_t Dims>
    using planar_cylindrical_t =
        geometry_solver_t<Dims, Geometry::PLANAR_CYLINDRICAL>;

}   // namespace simbi::mesh

#endif   // COMPILE_TIME_GEOMETRY_HPP
