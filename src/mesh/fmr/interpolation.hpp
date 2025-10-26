#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include "base/concepts.hpp"
#include "compat.hpp"              // for real type
#include "compute/field.hpp"       // for field_t
#include "containers/vector.hpp"   // for vector_t, coordinate_t, unit_vectors
#include "domain/domain.hpp"       // for domain_t
#include "functional/fp.hpp"       // for fp::partial
#include "utility/helpers.hpp"

#include <cmath>       // for std::abs
#include <cstdint>     // for std::uint64_t
#include <stdexcept>   // for std::runtime_error

namespace simbi::mesh::fmr {
    template <typename T, std::uint64_t Dims>
    struct interpolation_context_t {
        const field_t<T, Dims>& coarse_field;   // source field
        domain_t<Dims> coarse_domain;           // domain in coarse coordinates
        domain_t<Dims> fine_domain;             // domain in fine coordinates
        // refinement ratios per dimension
        std::uint64_t ref_ratio;

        iarray<Dims> coarse_offset;   // coarse domain origin
        iarray<Dims> fine_offset;     // fine domain origin

        // validate context
        constexpr bool is_valid() const
        {
            if (coarse_domain.empty() || fine_domain.empty()) {
                return false;
            }
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (ref_ratio <= 0) {
                    return false;
                }
            }
            return true;
        }

        // map fine coordinate to containing coarse coordinate
        DUAL coordinate_t<Dims>
        to_coarse_coord(const coordinate_t<Dims>& fine_coord) const
        {
            coordinate_t<Dims> coarse_coord;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                // translate to zero-based, scale down, translate to coarse
                // origin
                auto fine_local = fine_coord[ii] - fine_offset[ii];
                coarse_coord[ii] =
                    coarse_offset[ii] +
                    fine_local / static_cast<std::int64_t>(ref_ratio);
            }
            return coarse_coord;
        }

        // get fine cell position within coarse cell
        DUAL coordinate_t<Dims>
        fine_cell_offset(const coordinate_t<Dims>& fine_coord) const
        {
            coordinate_t<Dims> offset;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                auto fine_local = fine_coord[ii] - fine_offset[ii];
                offset[ii] = fine_local % static_cast<std::int64_t>(ref_ratio);
            }
            return offset;
        }
    };

    // linear interpolation transform
    template <typename T>
    struct linear_interpolation_t {
        template <std::uint64_t Dims>
        DUAL T operator()(
            const interpolation_context_t<T, Dims>& ctx,
            coordinate_t<Dims> fine_coord
        ) const
        {
            // get containing coarse cell and offset
            const auto coarse_coord = ctx.to_coarse_coord(fine_coord);
            const auto offset       = ctx.fine_cell_offset(fine_coord);

            if constexpr (Dims == 1) {
                // 1D linear interpolation
                const real x  = static_cast<real>(offset[0]) / ctx.ref_ratio;
                const auto vL = ctx.coarse_field(coarse_coord);
                const auto vR = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0)
                );
                return vL + x * (vR - vL);
            }
            else if constexpr (Dims == 2) {
                // bilinear interpolation using tensor product
                const vector_t<real, Dims> t{
                  static_cast<real>(offset[0]) / ctx.ref_ratio,
                  static_cast<real>(offset[1]) / ctx.ref_ratio
                };

                // get corner values
                const auto v00 = ctx.coarse_field(coarse_coord);
                const auto v10 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0)
                );
                const auto v01 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(1)
                );
                const auto v11 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0) +
                    unit_vectors::array_offset<Dims>(1)
                );

                // bilinear interpolation
                return (1.0 - t[0]) * (1.0 - t[1]) * v00 +
                       t[0] * (1.0 - t[1]) * v10 + (1.0 - t[0]) * t[1] * v01 +
                       t[0] * t[1] * v11;
            }
            else if constexpr (Dims == 3) {
                // trilinear interpolation
                const vector_t<real, Dims> t{
                  static_cast<real>(offset[0]) / ctx.ref_ratio,
                  static_cast<real>(offset[1]) / ctx.ref_ratio,
                  static_cast<real>(offset[2]) / ctx.ref_ratio
                };

                // get corner values (using your unit vector utilities)
                const auto v000 = ctx.coarse_field(coarse_coord);
                const auto v100 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0)
                );
                const auto v010 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(1)
                );
                const auto v110 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0) +
                    unit_vectors::array_offset<Dims>(1)
                );
                const auto v001 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(2)
                );
                const auto v101 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0) +
                    unit_vectors::array_offset<Dims>(2)
                );
                const auto v011 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(1) +
                    unit_vectors::array_offset<Dims>(2)
                );
                const auto v111 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0) +
                    unit_vectors::array_offset<Dims>(1) +
                    unit_vectors::array_offset<Dims>(2)
                );

                // trilinear interpolation
                return (1.0 - t[0]) * (1.0 - t[1]) * (1.0 - t[2]) * v000 +
                       t[0] * (1.0 - t[1]) * (1.0 - t[2]) * v100 +
                       (1.0 - t[0]) * t[1] * (1.0 - t[2]) * v010 +
                       t[0] * t[1] * (1.0 - t[2]) * v110 +
                       (1.0 - t[0]) * (1.0 - t[1]) * t[2] * v001 +
                       t[0] * (1.0 - t[1]) * t[2] * v101 +
                       (1.0 - t[0]) * t[1] * t[2] * v011 +
                       t[0] * t[1] * t[2] * v111;
            }
        }
    };

    // conservative interpolation transform
    template <typename T>
    struct conservative_interpolation_t {
        template <std::uint64_t Dims>
        DUAL T operator()(
            const interpolation_context_t<T, Dims>& ctx,
            coordinate_t<Dims> fine_coord
        ) const
        {
            // get containing coarse cell and offset
            const auto coarse_coord = ctx.to_coarse_coord(fine_coord);
            const auto offset       = ctx.fine_cell_offset(fine_coord);

            if constexpr (Dims == 1) {
                // compute limited slope for conservation
                const auto vm = ctx.coarse_field(
                    coarse_coord - unit_vectors::array_offset<Dims>(0)
                );
                const auto v0 = ctx.coarse_field(coarse_coord);
                const auto vp = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0)
                );

                // compute limited slope (minmod limiter)
                const auto slope_l = v0 - vm;
                const auto slope_r = vp - v0;
                const auto slope   = limit_slope(slope_l, slope_r);

                // compute conservatively interpolated value
                const real dx = 1.0 / ctx.ref_ratio;
                const real x =
                    (offset[0] + 0.5) * dx - 0.5;   // cell center offset
                return v0 + x * slope;
            }
            else if constexpr (Dims == 2) {
                // 2D conservative interpolation using slope limiting in each
                // direction
                const auto v00 = ctx.coarse_field(coarse_coord);

                // compute x-direction slopes
                const auto vm0 = ctx.coarse_field(
                    coarse_coord - unit_vectors::array_offset<Dims>(0)
                );
                const auto vp0 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0)
                );
                const auto slope_x = limit_slope(v00 - vm0, vp0 - v00);

                // compute y-direction slopes
                const auto v0m = ctx.coarse_field(
                    coarse_coord - unit_vectors::array_offset<Dims>(1)
                );
                const auto v0p = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(1)
                );
                const auto slope_y = limit_slope(v00 - v0m, v0p - v00);

                // compute offsets from coarse cell center
                const vector_t<real, Dims> dx{
                  1.0 / ctx.ref_ratio,
                  1.0 / ctx.ref_ratio
                };

                const vector_t<real, Dims> x{
                  (offset[0] + 0.5) * dx[0] - 0.5,
                  (offset[1] + 0.5) * dx[1] - 0.5
                };

                // conservative interpolation with limited slopes
                return v00 + x[0] * slope_x + x[1] * slope_y;
            }
            else if constexpr (Dims == 3) {
                // 3D conservative interpolation
                const auto v000 = ctx.coarse_field(coarse_coord);

                // compute slopes in each direction
                const auto vm00 = ctx.coarse_field(
                    coarse_coord - unit_vectors::array_offset<Dims>(0)
                );
                const auto vp00 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(0)
                );
                const auto slope_x = limit_slope(v000 - vm00, vp00 - v000);

                const auto v0m0 = ctx.coarse_field(
                    coarse_coord - unit_vectors::array_offset<Dims>(1)
                );
                const auto v0p0 = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(1)
                );
                const auto slope_y = limit_slope(v000 - v0m0, v0p0 - v000);

                const auto v00m = ctx.coarse_field(
                    coarse_coord - unit_vectors::array_offset<Dims>(2)
                );
                const auto v00p = ctx.coarse_field(
                    coarse_coord + unit_vectors::array_offset<Dims>(2)
                );
                const auto slope_z = limit_slope(v000 - v00m, v00p - v000);

                // compute offsets
                const vector_t<real, Dims> dx{
                  1.0 / ctx.ref_ratio,
                  1.0 / ctx.ref_ratio,
                  1.0 / ctx.ref_ratio
                };

                const vector_t<real, Dims> x{
                  (offset[0] + 0.5) * dx[0] - 0.5,
                  (offset[1] + 0.5) * dx[1] - 0.5,
                  (offset[2] + 0.5) * dx[2] - 0.5
                };

                // conservative interpolation with limited slopes
                return v000 + x[0] * slope_x + x[1] * slope_y + x[2] * slope_z;
            }
        }

      private:
        DUAL static real limit_slope_component(real slope_l, real slope_r)
        {
            using namespace simbi::helpers;
            // [TODO]: make theta configurable
            constexpr real theta = 2.0;   // van Leer limiter parameter

            const auto r =
                (std::abs(slope_l) < global::epsilon) ? 1.0 : slope_r / slope_l;

            if (r <= 0) {
                return real{0};   // opposite signs - return zero slope
            }

            // van Leer limiter with theta parameter
            return slope_l *
                   my_max(
                       0.0,
                       my_min(theta * r, my_min((1.0 + r) * 0.5, theta))
                   );
        }
        // slope limiter to prevent oscillations
        DUAL static T limit_slope(const T& slope_l, const T& slope_r)
        {
            T new_slope;
            for (std::uint64_t ii = 0; ii << T::nmem; ++ii) {
                new_slope[ii] = limit_slope_component(slope_l[ii], slope_r[ii]);
            }
            return new_slope;
        }
    };

    // create interpolation field from context
    template <typename T, std::uint64_t Dims>
    auto make_interpolation_field(
        const interpolation_context_t<T, Dims>& ctx,
        bool /*conservative*/ = true
    )
    {
        if (!ctx.is_valid()) {
            throw std::runtime_error("invalid interpolation context");
        }

        // choose interpolation method
        // if (conservative) {
        //     auto transform = conservative_interpolation_t<T>{};
        //     return field(ctx.fine_domain, fp::partial(transform, ctx));
        // }
        // else {
        //     auto transform = linear_interpolation_t<T>{};
        //     return field(ctx.fine_domain, fp::partial(transform, ctx));
        // }
        // default to conservative interpolation
        auto transform = conservative_interpolation_t<T>{};
        return field(ctx.fine_domain, fp::partial(transform, ctx));
    }

}   // namespace simbi::mesh::fmr

#endif   // INTERPOLATION_HPP
