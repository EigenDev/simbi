#ifndef PROLONGATION_HPP
#define PROLONGATION_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "domain/ghost.hpp"
#include "level_mapping.hpp"
#include "utility/helpers.hpp"

#include <cstdint>
#include <iostream>

namespace simbi::mesh::fmr {
    // === Interpolation Strategies ===

    // piecewise constant (zeroth-order)
    template <typename T, std::uint64_t Dims>
    struct constant_interpolator_t {
        DUAL T operator()(
            const field_t<T, Dims>& coarse,
            const coordinate_t<Dims>& fine_coord,
            const level_mapping_t<Dims>& map
        ) const
        {
            auto coarse_coord = map.fine_to_coarse(fine_coord);

            // clamp to valid coarse domain if out of bounds
            if (!map.coarse_full.contains(coarse_coord)) {
                // std::cout << "Warning: coarse_coord " << coarse_coord
                //           << " out of bounds "
                //           << "coarse_full: " << map.coarse_full << "\n";
                auto clamped = coarse_coord;
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    clamped[d] = std::clamp(
                        coarse_coord[d],
                        map.coarse_full.start[d],
                        map.coarse_full.fin[d] - 1
                    );
                }
                return coarse(clamped);
            }

            return coarse(coarse_coord);
        }
    };

    // conservative with slope limiting (first-order)
    template <typename T, std::uint64_t Dims>
    struct conservative_interpolator_t {
        DUAL T operator()(
            const field_t<T, Dims>& coarse,
            const coordinate_t<Dims>& fine_coord,
            const level_mapping_t<Dims>& map
        ) const
        {
            auto coarse_coord = map.fine_to_coarse(fine_coord);

            // boundary handling
            if (!map.coarse_full.contains(coarse_coord)) {
                auto clamped = coarse_coord;
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    clamped[d] = std::clamp(
                        coarse_coord[d],
                        map.coarse_full.start[d],
                        map.coarse_full.fin[d] - 1
                    );
                }
                return coarse(clamped);
            }

            // get base value
            auto v0 = coarse(coarse_coord);
            vector_t<T, Dims> slopes;

            for (std::uint64_t d = 0; d < Dims; ++d) {
                auto left_coord  = coarse_coord;
                auto right_coord = coarse_coord;
                left_coord[d] -= 1;
                right_coord[d] += 1;

                bool has_left  = map.coarse_full.contains(left_coord);
                bool has_right = map.coarse_full.contains(right_coord);

                if (has_left && has_right) {
                    // --- centered difference ---
                    auto v_left  = coarse(left_coord);
                    auto v_right = coarse(right_coord);
                    slopes[d]    = limit_slope(v0 - v_left, v_right - v0);
                }
                else if (has_left) {
                    // --- backward difference (at upper boundary) ---
                    auto v_left = coarse(left_coord);
                    slopes[d]   = limit_slope(v0 - v_left, v_left - v0);
                }
                else if (has_right) {
                    // --- forward difference (at lower boundary) ---
                    auto v_right = coarse(right_coord);
                    slopes[d]    = limit_slope(v_right - v0, v_right - v0);
                }
                else {
                    // --- no neighbors (isolated cell) ---
                    slopes[d] = T{};
                }
            }

            // compute position within coarse cell
            auto offset = map.fine_offset_in_coarse(fine_coord);
            vector_t<real, Dims> normalized_pos;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                normalized_pos[d] = (static_cast<real>(offset[d]) + 0.5) /
                                        static_cast<real>(map.ratio) -
                                    0.5;
            }

            // linear reconstruction
            T result = v0;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                result = result + slopes[d] * normalized_pos[d];
            }

            return result;
        }

      private:
        DUAL static T limit_slope(const T& slope_left, const T& slope_right)
        {
            using namespace simbi::helpers;
            T limited;

            for (std::uint64_t i = 0; i < T::nmem; ++i) {
                const auto sl = slope_left[i];
                const auto sr = slope_right[i];

                if (sl * sr <= 0) {
                    limited[i] = 0;   // opp signs
                }
                else {
                    // Van Leer limiter
                    const auto r =
                        (std::abs(sl) < global::epsilon) ? 1.0 : sr / sl;
                    constexpr real theta = 2.0;
                    limited[i] =
                        sl *
                        my_max(
                            0.0,
                            my_min(theta * r, my_min((1.0 + r) * 0.5, theta))
                        );
                }
            }

            return limited;
        }
    };

    // parabolic, cell-average-preserving (second-order)
    // Romain Teyssier taught me that because of the fine-coarse
    // interface, prolongation across the refinement boundaries
    // reduces ones solution to N - 1 order accurace, where N is
    // the order of the scheme. Thus, I use a parabolic interpolator
    // here to achieve overall second-order accuracy.
    template <typename T, std::uint64_t Dims>
    struct parabolic_interpolator_t {
        DUAL T operator()(
            const field_t<T, Dims>& coarse,
            const coordinate_t<Dims>& fine_coord,
            const level_mapping_t<Dims>& map
        ) const
        {
            auto coarse_coord = map.fine_to_coarse(fine_coord);

            // boundary handling (clamp to valid domain)
            if (!map.coarse_full.contains(coarse_coord)) {
                auto clamped = coarse_coord;
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    clamped[d] = std::clamp(
                        coarse_coord[d],
                        map.coarse_full.start[d],
                        map.coarse_full.fin[d] - 1
                    );
                }
                return coarse(clamped);   // fallback to constant
            }

            // get base value and normalized position
            auto v0 = coarse(coarse_coord);

            // compute position within coarse cell (normalized to [-0.5, 0.5])
            auto offset = map.fine_offset_in_coarse(fine_coord);
            vector_t<real, Dims> x_norm;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                x_norm[d] = (static_cast<real>(offset[d]) + 0.5) /
                                static_cast<real>(map.ratio) -
                            0.5;
            }

            // multiD reconstruction
            //    we start with the base value (v0) and add the
            //    1st and 2nd order corrections from each direction.
            T result = v0;

            for (std::uint64_t d = 0; d < Dims; ++d) {
                auto left_coord  = coarse_coord;
                auto right_coord = coarse_coord;
                left_coord[d] -= 1;
                right_coord[d] += 1;

                if (map.coarse_full.contains(left_coord) &&
                    map.coarse_full.contains(right_coord)) {
                    // === We have a 3-point stencil ===
                    auto v_left  = coarse(left_coord);
                    auto v_right = coarse(right_coord);

                    // standard centered-difference slopes
                    auto slope_left  = v0 - v_left;
                    auto slope_right = v_right - v0;

                    // compute limited slope (same as 1st-order)
                    auto slope_limited = limit_slope(slope_left, slope_right);

                    // compute limited 2nd derivative (curvature)
                    auto d2v = v_right - 2.0 * v0 + v_left;

                    // monotocity check: if curvature and slope have
                    // opposite signs, flatten the parabola.
                    auto d2v_limited = d2v;
                    for (std::uint64_t i = 0; i < T::nmem; ++i) {
                        if (d2v[i] * slope_limited[i] < 0.0) {
                            d2v_limited[i] = 0.0;
                        }
                    }

                    // add 1st-order (linear) term
                    result = result + slope_limited * x_norm[d];

                    // Add 2nd-order (parabolic) term
                    // The 1/2 is from Taylor expansion, the 1/12 is from
                    // enforcing cell-average conservation.
                    // f(x) = ... + (d2v/2) * (x^2 - 1/12)
                    result =
                        result + (d2v_limited * 0.5) *
                                     (x_norm[d] * x_norm[d] - (1.0 / 12.0));
                }
                else {
                    // === At a boundary, fall back to 1st-order ===
                    // (that is, we can't build a parabola)

                    bool has_left  = map.coarse_full.contains(left_coord);
                    bool has_right = map.coarse_full.contains(right_coord);

                    T slope_limited{};
                    if (has_left) {
                        auto v_left   = coarse(left_coord);
                        slope_limited = limit_slope(v0 - v_left, v0 - v_left);
                    }
                    else if (has_right) {
                        auto v_right  = coarse(right_coord);
                        slope_limited = limit_slope(v_right - v0, v_right - v0);
                    }

                    // add 1st-order (linear) term only
                    result = result + slope_limited * x_norm[d];
                }
            }

            return result;
        }

      private:
        // this is the same limiter you already have
        DUAL static T limit_slope(const T& slope_left, const T& slope_right)
        {
            using namespace simbi::helpers;
            T limited;

            for (std::uint64_t i = 0; i < T::nmem; ++i) {
                const auto sl = slope_left[i];
                const auto sr = slope_right[i];

                if (sl * sr <= 0) {
                    limited[i] = 0;   // opp signs
                }
                else {
                    // van Leer limiter
                    const auto r =
                        (std::abs(sl) < global::epsilon) ? 1.0 : sr / sl;
                    constexpr real theta = 2.0;
                    limited[i] =
                        sl *
                        my_max(
                            0.0,
                            my_min(theta * r, my_min((1.0 + r) * 0.5, theta))
                        );
                }
            }
            return limited;
        }
    };

    template <typename T, std::uint64_t Dims, typename SpatialInterpolator>
    struct time_varying_interpolator_t {
        const field_t<T, Dims>& u_n;
        const field_t<T, Dims>& u_star;
        real alpha;
        SpatialInterpolator
            spatial_interp;   // the wrapped spatial interpolator

        // this ignores the 'coarse' field passed by the driver
        // and uses its own u_n and u_star fields instead.
        DUAL T operator()(
            const field_t<T, Dims>& /* dummy_coarse */,
            const coordinate_t<Dims>& fine_coord,
            const level_mapping_t<Dims>& map
        ) const
        {
            // spatially interpolate u_n to the fine coord
            T v_n = spatial_interp(u_n, fine_coord, map);

            // spatially interpolate u_star to the fine coord
            T v_star = spatial_interp(u_star, fine_coord, map);

            // time interpolate the two spatially-interpolated values
            //    (Using the operators from your systems.hpp)
            return v_n | structs::scale_gas(1.0 - alpha) |
                   structs::add_gas(v_star | structs::scale_gas(alpha));
        }
    };

    // fill entire fine field from coarse field
    template <typename T, std::uint64_t Dims, typename Interpolator>
    void prolongate_full(
        const field_t<T, Dims>& coarse,
        field_t<T, Dims>& fine,
        const level_mapping_t<Dims>& map,
        Interpolator interpolator = Interpolator{}
    )
    {
        // iterate over entire fine domain (active + ghosts)
        for (const auto& fine_coord : map.fine_full) {
            fine(fine_coord) = interpolator(coarse, fine_coord, map);
        }
    }

    // convenience wrappers
    template <typename T, std::uint64_t Dims>
    void prolongate_constant(
        const field_t<T, Dims>& coarse,
        field_t<T, Dims>& fine,
        const level_mapping_t<Dims>& map
    )
    {
        prolongate_full(coarse, fine, map, constant_interpolator_t<T, Dims>{});
    }

    template <typename T, std::uint64_t Dims>
    void prolongate_conservative(
        const field_t<T, Dims>& coarse,
        field_t<T, Dims>& fine,
        const level_mapping_t<Dims>& map
    )
    {
        prolongate_full(
            coarse,
            fine,
            map,
            conservative_interpolator_t<T, Dims>{}
        );
    }

    // fill only ghost zones (for evolution)
    template <typename T, std::uint64_t Dims, typename Interpolator>
    void prolongate_ghosts_only(
        const field_t<T, Dims>& coarse,
        field_t<T, Dims>& fine,
        const level_mapping_t<Dims>& map,
        Interpolator interpolator = Interpolator{}
    )
    {
        auto ghost_regions =
            boundary::analyze_ghost_regions(map.fine_full, map.fine_active);

        for (const auto& ghost_region : ghost_regions) {
            for (const auto& fine_coord : ghost_region.domain) {
                fine(fine_coord) = interpolator(coarse, fine_coord, map);
            }
        }
    }

    template <typename T, std::uint64_t Dims>
    void prolongate_ghosts_conservative(
        const field_t<T, Dims>& coarse,
        field_t<T, Dims>& fine,
        const level_mapping_t<Dims>& map
    )
    {
        prolongate_ghosts_only(
            coarse,
            fine,
            map,
            parabolic_interpolator_t<T, Dims>{}
        );
    }

    template <typename T, std::uint64_t Dims>
    void prolongate_ghosts_time_interpolated(
        const field_t<T, Dims>& coarse_u_n,
        const field_t<T, Dims>& coarse_u_star,
        real alpha,
        field_t<T, Dims>& fine,
        const level_mapping_t<Dims>& map
    )
    {
        auto spatial_interpolator = parabolic_interpolator_t<T, Dims>{};

        auto time_space_interpolator = time_varying_interpolator_t<
            T,
            Dims,
            decltype(spatial_interpolator)>{
          coarse_u_n,
          coarse_u_star,
          alpha,
          spatial_interpolator
        };

        prolongate_ghosts_only(coarse_u_n, fine, map, time_space_interpolator);
    }

}   // namespace simbi::mesh::fmr

#endif
