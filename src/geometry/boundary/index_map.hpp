#ifndef GEOMETRY_BOUNDARY_MAPS_HPP
#define GEOMETRY_BOUNDARY_MAPS_HPP

#include "compat.hpp"
#include "containers/vector.hpp"

#include <cstdint>

namespace simbi::geometry {

    // -------------------------------------------------------------------------
    // clamp map (outflow / zero-gradient)
    // maps any index outside [min, max] to the nearest edge
    // -------------------------------------------------------------------------
    struct clamp_map_t {
        std::uint64_t dim_;
        std::int64_t min_val_;   // global start index of active domain
        std::int64_t max_val_;   // global end index (exclusive) - 1

        DUAL constexpr clamp_map_t(
            std::uint64_t dim,
            std::int64_t min_val,
            std::int64_t max_val
        )
            : dim_(dim), min_val_(min_val), max_val_(max_val - 1)
        {
        }

        template <std::uint64_t Rank>
        DUAL auto operator()(const iarray<Rank>& coord) const
        {
            auto ret = coord;
            if (ret[dim_] < min_val_) {
                ret[dim_] = min_val_;
            }
            else if (ret[dim_] > max_val_) {
                ret[dim_] = max_val_;
            }
            return ret;
        }
    };

    // -------------------------------------------------------------------------
    // mirror map (reflection)
    // pivots coordinate around a face
    // formula: src = 2 * pivot - 1 - dst
    // -------------------------------------------------------------------------
    struct mirror_map_t {
        std::uint64_t dim_;
        std::int64_t pivot_term_;   // precomputed: 2 * face_index - 1

        DUAL constexpr mirror_map_t(std::uint64_t dim, std::int64_t face_idx)
            : dim_(dim), pivot_term_(2 * face_idx - 1)
        {
        }

        template <std::uint64_t Rank>
        DUAL auto operator()(const iarray<Rank>& coord) const
        {
            auto ret = coord;
            // standard reflection formula for 0-based indexing
            ret[dim_] = pivot_term_ - ret[dim_];
            return ret;
        }
    };

    // -------------------------------------------------------------------------
    // periodic map (wrap)
    // wraps coordinate into [start, start + len)
    // -------------------------------------------------------------------------
    struct periodic_map_t {
        std::uint64_t dim_;
        std::int64_t start_;
        std::int64_t len_;

        DUAL constexpr periodic_map_t(
            std::uint64_t dim,
            std::int64_t start,
            std::int64_t len
        )
            : dim_(dim), start_(start), len_(len)
        {
        }

        template <std::uint64_t Rank>
        DUAL auto operator()(const iarray<Rank>& coord) const
        {
            auto ret         = coord;
            std::int64_t val = ret[dim_] - start_;

            // handle negative wrap
            val = val % len_;
            if (val < 0) {
                val += len_;
            }

            ret[dim_] = start_ + val;
            return ret;
        }
    };

}   // namespace simbi::geometry

#endif   // GRID_BOUNDARY_MAPS_HPP
