#ifndef PROLONGATION_HPP
#define PROLONGATION_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "domain/ghost.hpp"
#include "level_mapping.hpp"
#include "utility/helpers.hpp"

#include <cstdint>

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

            // Boundary handling
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

            // compute limited slopes in each direction
            vector_t<T, Dims> slopes;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                auto left_coord  = coarse_coord;
                auto right_coord = coarse_coord;
                left_coord[d] -= 1;
                right_coord[d] += 1;

                if (map.coarse_full.contains(left_coord) &&
                    map.coarse_full.contains(right_coord)) {
                    auto v_left  = coarse(left_coord);
                    auto v_right = coarse(right_coord);
                    slopes[d]    = limit_slope(v0 - v_left, v_right - v0);
                }
                else {
                    slopes[d] = T{};   // zero slope at boundaries
                }
            }

            // compute position within coarse cell (normalized to [-0.5, 0.5])
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
            conservative_interpolator_t<T, Dims>{}
        );
    }

}   // namespace simbi::mesh::fmr

#endif
