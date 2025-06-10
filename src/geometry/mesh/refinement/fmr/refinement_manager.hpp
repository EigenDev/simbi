/**
 * @file            refinement_manager.hpp
 * @brief           manages fixed mesh refinement using functional paradigms
 */

#ifndef REFINEMENT_MANAGER_HPP
#define REFINEMENT_MANAGER_HPP

#include "config.hpp"
#include "core/functional/fp.hpp"
#include "core/types/containers/array.hpp"
#include <functional>
#include <vector>

namespace simbi::refinement {
    /**
     * refinement_region - defines a rectangular region for mesh refinement
     */
    template <size_type Dims>
    struct refinement_region {
        // minimum coordinates of region (inclusive)
        array_t<size_type, Dims> min_bounds;
        // maximum coordinates of region (exclusive)
        array_t<size_type, Dims> max_bounds;

        // check if a point is within this region using fp concept
        DUAL bool contains(const array_t<size_type, Dims>& coords) const
        {
            // create array of comparison results for each dimension
            array_t<bool, Dims> dim_checks;

            for (size_type ii = 0; ii < Dims; ++ii) {
                dim_checks[ii] = (coords[ii] >= min_bounds[ii]) &&
                                 (coords[ii] < max_bounds[ii]);
            }

            return fp::all_of(dim_checks, [](bool x) { return x; });
        }

        // get the size of this region in cells
        DUAL array_t<size_type, Dims> size() const
        {
            array_t<size_type, Dims> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = max_bounds[ii] - min_bounds[ii];
            }
            return result;
        }

        // get total number of cells in this region
        DUAL size_type cell_count() const
        {
            auto s          = size();
            size_type count = 1;
            for (size_type ii = 0; ii < Dims; ++ii) {
                count *= s[ii];
            }
            return count;
        }
    };

    /**
     * functional transformation between grid levels
     */
    struct level_transforms {
        // transform from coarse to fine coordinates
        template <size_type Dims>
        DUAL static array_t<size_type, Dims>
        coarse_to_fine(const array_t<size_type, Dims>& coords, size_type rf)
        {
            return fp::map(coords, [rf](size_type x) { return x * rf; });
        }

        // transform from fine to coarse coordinates
        template <size_type Dims>
        DUAL static array_t<size_type, Dims>
        fine_to_coarse(const array_t<size_type, Dims>& coords, size_type rf)
        {
            return fp::map(coords, [rf](size_type x) { return x / rf; });
        }

        // get the fine cell offset within its parent coarse cell
        template <size_type Dims>
        DUAL static array_t<size_type, Dims> fine_cell_offset(
            const array_t<size_type, Dims>& fine_coords,
            size_type rf
        )
        {
            return fp::map(fine_coords, [rf](size_type x) { return x % rf; });
        }
    };

    /**
     * refinement_manager - manages refinement levels and regions
     */
    template <size_type Dims>
    struct refinement_manager {
        // immutable configuration properties
        const size_type max_level;   // maximum refinement level (0 = base grid)
        const size_type refinement_factor;   // refinement ratio between levels
        // whether to apply flux correction at  boundaries
        const bool flux_correction;

        // vector of regions per level
        const std::vector<std::vector<refinement_region<Dims>>> regions;

        // ctors
        // default
        refinement_manager()
            : max_level(0),
              refinement_factor(1),
              flux_correction(false),
              regions(std::vector<std::vector<refinement_region<Dims>>>{})
        {
        }

        // check if a point is within any refined region at the given level
        DUAL bool is_refined(
            size_type level,
            const array_t<size_type, Dims>& coords
        ) const
        {
            if (level >= max_level || level >= regions.size()) {
                return false;
            }

            // check all regions at this level
            for (const auto& region : regions[level]) {
                if (region.contains(coords)) {
                    return true;
                }
            }

            return false;
        }

        // get the refinement level at a position
        DUAL size_type level_at(const array_t<size_type, Dims>& coords) const
        {
            for (size_type level = max_level; level > 0; --level) {
                if (is_refined(level - 1, coords)) {
                    return level;
                }
            }
            return 0;   // base level
        }

        // transform from coarse to fine coordinates
        DUAL array_t<size_type, Dims>
        coarse_to_fine(const array_t<size_type, Dims>& coords) const
        {
            return level_transforms::coarse_to_fine(coords, refinement_factor);
        }

        // transform from fine to coarse coordinates
        DUAL array_t<size_type, Dims>
        fine_to_coarse(const array_t<size_type, Dims>& coords) const
        {
            return level_transforms::fine_to_coarse(coords, refinement_factor);
        }

        // check if a coarse cell has fine cells that overlap with it
        DUAL bool has_fine_overlap(
            size_type level,
            const array_t<size_type, Dims>& coords
        ) const
        {
            if (level >= max_level) {
                return false;
            }
            return is_refined(level, coords);
        }

        // check if a fine cell is at a coarse-fine boundary
        DUAL bool is_at_level_boundary(
            size_type level,
            const array_t<size_type, Dims>& coords
        ) const
        {
            if (level == 0) {
                return false;   // base level has no level boundaries
            }

            // check if any neighboring coarse cell doesn't have fine cells
            for (size_type dim = 0; dim < Dims; ++dim) {
                array_t<size_type, Dims> coarse_coords = fine_to_coarse(coords);

                // check neighboring coarse cells in this dimension
                for (int offset = -1; offset <= 1; offset += 2) {
                    array_t<size_type, Dims> neighbor = coarse_coords;
                    neighbor[dim] += offset;

                    // if the neighbor isn't refined, we're at a boundary
                    if (!is_refined(level - 1, neighbor)) {
                        return true;
                    }
                }
            }

            return false;
        }
    };

}   // namespace simbi::refinement

#endif   // REFINEMENT_MANAGER_HPP
