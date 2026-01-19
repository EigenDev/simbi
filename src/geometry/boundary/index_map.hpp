// =============================================================================
// index_map.hpp
//
// coordinate mapping functors for boundary conditions.
// this file provides various structs (`clamp_map_t`, `mirror_map_t`,
// `periodic_map_t`, etc.) that are used as functors to remap out-of-bounds
// ghost cell coordinates to valid interior coordinates, forming the core
// mechanism for applying boundary conditions on the grid.
//
// usage:
//   auto map = clamp_map_t<1>{0, 0, 100};
//   auto remapped_computation = field.remap(map);
// =============================================================================
#pragma once

#include "containers/vector.hpp"
#include "decorators.hpp"

#include <cstdint>

namespace simbi::geometry {

    // -------------------------------------------------------------------------
    // clamp map (outflow / zero-gradient)
    // maps any index outside [min, max] to the nearest edge
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct clamp_map_t
    {
        using value_type                    = iarray<Rank>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        std::uint64_t dim_;
        std::int64_t  min_val_; // global start index of active domain
        std::int64_t  max_val_; // global end index (exclusive) - 1

        DEV constexpr clamp_map_t(std::uint64_t dim, std::int64_t min_val, std::int64_t max_val)
            : dim_(dim), min_val_(min_val), max_val_(max_val - 1)
        {
        }

        DEV value_type operator()(argument_type coord) const
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
    template <std::uint64_t Rank>
    struct mirror_map_t
    {
        using value_type                    = iarray<Rank>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        std::uint64_t dim_;
        std::int64_t  pivot_term_; // precomputed: 2 * face_index - 1

        DEV constexpr mirror_map_t(std::uint64_t dim, std::int64_t face_idx)
            : dim_(dim), pivot_term_(2 * face_idx - 1)
        {
        }

        DEV value_type operator()(argument_type coord) const
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
    template <std::uint64_t Rank>
    struct periodic_map_t
    {
        using value_type                    = iarray<Rank>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        std::uint64_t dim_;
        std::int64_t  start_;
        std::int64_t  len_;

        DEV constexpr periodic_map_t(std::uint64_t dim, std::int64_t start, std::int64_t len)
            : dim_(dim), start_(start), len_(len)
        {
        }

        DEV value_type operator()(argument_type coord) const
        {
            auto         ret = coord;
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

    // -------------------------------------------------------------------------
    // multi-dimensional boundary map
    // applies boundary conditions in multiple dimensions simultaneously
    // used for edges and corners
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct multidim_map_t
    {
        using value_type                    = iarray<Rank>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        iarray<Rank> active_dims_; // 1 if this dimension applies a map
        iarray<Rank> map_types_;   // 0=none, 1=periodic, 2=mirror, 3=clamp
        iarray<Rank> starts_;      // domain starts for periodic
        iarray<Rank> lens_;        // domain lengths for periodic
        iarray<Rank> pivots_;      // pivot points for mirror
        iarray<Rank> clamp_vals_;  // clamp values for outflow

        DEV value_type operator()(argument_type coord) const
        {
            auto ret = coord;
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                if (!active_dims_[dd]) {
                    continue;
                }

                switch (map_types_[dd]) {
                    case 1: { // periodic
                        std::int64_t val = ret[dd] - starts_[dd];
                        val              = val % lens_[dd];
                        if (val < 0) {
                            val += lens_[dd];
                        }
                        ret[dd] = starts_[dd] + val;
                        break;
                    }
                    case 2: { // mirror
                        ret[dd] = pivots_[dd] - ret[dd];
                        break;
                    }
                    case 3: { // clamp
                        ret[dd] = clamp_vals_[dd];
                        break;
                    }
                }
            }
            return ret;
        }
    };

    // -------------------------------------------------------------------------
    // spherical pole map
    // handles reflection across theta=0 or theta=pi poles in spherical coords
    //
    // geometry:
    //   - mirrors theta across pole: theta -> -theta or 2π - theta
    //   - rotates phi by π (opposite hemisphere)
    //   - works for arbitrary ghost depth
    //
    // coordinate layout (array indices):
    //   2D: [theta, r]           -> theta_dim=0, phi_dim=invalid
    //   3D: [phi, theta, r]      -> theta_dim=1, phi_dim=0
    //
    // only used for Rank >= 2, guarded at call site
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct spherical_pole_map_t
    {
        using value_type                    = iarray<Rank>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        std::int64_t theta_pivot_; // precomputed: 2*pole_idx - 1
        std::int64_t phi_start_;   // for wrapping phi after rotation
        std::int64_t phi_len_;     // phi domain length

        DEV constexpr spherical_pole_map_t(
            std::int64_t theta_pole_idx,
            std::int64_t phi_domain_start = 0,
            std::int64_t phi_domain_len   = 0
        )
            : theta_pivot_(2 * theta_pole_idx - 1), phi_start_(phi_domain_start),
              phi_len_(phi_domain_len)
        {
        }

        DEV value_type operator()(argument_type coord) const
        {
            auto ret = coord;

            if constexpr (Rank >= 2) {
                constexpr std::uint64_t theta_dim = Rank - 2;
                ret[theta_dim]                    = theta_pivot_ - ret[theta_dim];

                if constexpr (Rank >= 3) {
                    constexpr std::uint64_t phi_dim = Rank - 3;

                    std::int64_t phi_val = ret[phi_dim] - phi_start_;
                    phi_val += phi_len_ / 2;

                    phi_val = phi_val % phi_len_;
                    if (phi_val < 0) {
                        phi_val += phi_len_;
                    }

                    ret[phi_dim] = phi_start_ + phi_val;
                }
            }

            return ret;
        }
    };

} // namespace simbi::geometry
