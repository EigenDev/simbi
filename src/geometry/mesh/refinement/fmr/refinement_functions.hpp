/**
 * @file            refinement_functions.hpp
 * @brief           functional utilities for mesh refinement operations
 */

#ifndef REFINEMENT_FUNCTIONS_HPP
#define REFINEMENT_FUNCTIONS_HPP

#include "build_options.hpp"
#include "core/functional/fp.hpp"
#include "core/types/containers/ndarray.hpp"
#include "refinement_manager.hpp"
#include "util/parallel/parallel_for.hpp"
#include <array>
#include <functional>

namespace simbi::refinement {
    /**
     * prolongate_value - interpolate a value from coarse to fine using
     * linear interpolation
     */
    template <typename T>
    DUAL T prolongate_value(const T& coarse_value)
    {
        // simple copy for now
        // TODO: could be extended with gradients for higher
        // order?
        return coarse_value;
    }

    /**
     * restrict_values - average fine values to get a coarse value
     */
    template <typename T>
    DUAL T restrict_values(const std::vector<T>& fine_values)
    {
        return fp::sum(fine_values) / fine_values.size();
    }

    /**
     * prolongation operation for a single cell
     */
    template <typename T, size_type Dims>
    struct prolongate_cell {
        const ndarray<T, Dims>& coarse_data;
        ndarray<T, Dims>& fine_data;
        size_type refinement_factor;

        DUAL void operator()(const array_t<size_type, Dims>& fine_coords) const
        {
            // compute corresponding coarse indices
            array_t<size_type, Dims> coarse_coords =
                level_transforms::fine_to_coarse(
                    fine_coords,
                    refinement_factor
                );

            // get coarse value
            T coarse_value;
            if constexpr (Dims == 1) {
                coarse_value = coarse_data.at(coarse_coords[0]);
            }
            else if constexpr (Dims == 2) {
                coarse_value =
                    coarse_data.at(coarse_coords[0], coarse_coords[1]);
            }
            else {
                coarse_value = coarse_data.at(
                    coarse_coords[0],
                    coarse_coords[1],
                    coarse_coords[2]
                );
            }

            // interpolate and set fine value
            T fine_value = prolongate_value(coarse_value);

            if constexpr (Dims == 1) {
                fine_data.at(fine_coords[0]) = fine_value;
            }
            else if constexpr (Dims == 2) {
                fine_data.at(fine_coords[0], fine_coords[1]) = fine_value;
            }
            else {
                fine_data.at(fine_coords[0], fine_coords[1], fine_coords[2]) =
                    fine_value;
            }
        }
    };

    /**
     * prolongation - interpolates data from coarse to fine grid
     *
     * @param coarse_data    data on the coarse grid
     * @param fine_data      data on the fine grid (will be modified)
     * @param region         the region to refine
     * @param rf             refinement factor
     * @param policy         execution policy for parallel operations
     */
    template <typename T, size_type Dims>
    void prolongate(
        const ndarray<T, Dims>& coarse_data,
        ndarray<T, Dims>& fine_data,
        const refinement_region<Dims>& region,
        size_type rf,
        const ExecutionPolicy<>& policy = ExecutionPolicy<>()
    )
    {
        // convert region bounds to fine grid coordinates
        array_t<size_type, Dims> fine_min, fine_max;
        for (size_type ii = 0; ii < Dims; ++ii) {
            fine_min[ii] = region.min_bounds[ii] * rf;
            fine_max[ii] = region.max_bounds[ii] * rf;
        }

        // create operator for cell prolongation
        prolongate_cell<T, Dims> op{coarse_data, fine_data, rf};

        // parallel implementation for 1D case
        if constexpr (Dims == 1) {
            parallel_for(
                policy,
                fine_min[0],
                fine_max[0],
                [op] DEV(size_type ii) { op({ii}); }
            );
        }
        // sequential implementation for higher dimensions for now
        // TODO: extend to parallel for higher dimensions
        else {
            for (size_type ii = fine_min[0]; ii < fine_max[0]; ++ii) {
                for (size_type jj = (Dims > 1 ? fine_min[1] : 0);
                     jj < (Dims > 1 ? fine_max[1] : 1);
                     ++jj) {
                    for (size_type kk = (Dims > 2 ? fine_min[2] : 0);
                         kk < (Dims > 2 ? fine_max[2] : 1);
                         ++kk) {

                        if constexpr (Dims == 2) {
                            op({ii, jj});
                        }
                        else if constexpr (Dims == 3) {
                            op({ii, jj, kk});
                        }
                    }
                }
            }
        }
    }

    /**
     * restriction operation for a single coarse cell
     */
    template <typename T, size_type Dims>
    struct restrict_cell {
        const ndarray<T, Dims>& fine_data;
        ndarray<T, Dims>& coarse_data;
        size_type refinement_factor;

        DUAL void operator()(const array_t<size_type, Dims>& coarse_coords
        ) const
        {
            std::vector<T> fine_values;
            fine_values.reserve(std::pow(refinement_factor, Dims));

            // collect all fine values in this coarse cell
            array_t<size_type, Dims> fine_start =
                level_transforms::coarse_to_fine(
                    coarse_coords,
                    refinement_factor
                );

            // iterate through fine cells
            for (size_type ii_offset = 0; ii_offset < refinement_factor;
                 ++ii_offset) {
                for (size_type jj_offset = 0;
                     jj_offset < (Dims > 1 ? refinement_factor : 1);
                     ++jj_offset) {
                    for (size_type kk_offset = 0;
                         kk_offset < (Dims > 2 ? refinement_factor : 1);
                         ++kk_offset) {

                        array_t<size_type, Dims> fine_coords = fine_start;
                        fine_coords[0] += ii_offset;
                        if constexpr (Dims > 1) {
                            fine_coords[1] += jj_offset;
                        }
                        if constexpr (Dims > 2) {
                            fine_coords[2] += kk_offset;
                        }

                        // get fine value
                        T fine_value;
                        if constexpr (Dims == 1) {
                            fine_value = fine_data.at(fine_coords[0]);
                        }
                        else if constexpr (Dims == 2) {
                            fine_value =
                                fine_data.at(fine_coords[0], fine_coords[1]);
                        }
                        else {
                            fine_value = fine_data.at(
                                fine_coords[0],
                                fine_coords[1],
                                fine_coords[2]
                            );
                        }

                        fine_values.push_back(fine_value);
                    }
                }
            }

            // restrict fine values to coarse value
            T coarse_value = restrict_values(fine_values);

            // set coarse value
            if constexpr (Dims == 1) {
                coarse_data.at(coarse_coords[0]) = coarse_value;
            }
            else if constexpr (Dims == 2) {
                coarse_data.at(coarse_coords[0], coarse_coords[1]) =
                    coarse_value;
            }
            else {
                coarse_data.at(
                    coarse_coords[0],
                    coarse_coords[1],
                    coarse_coords[2]
                ) = coarse_value;
            }
        }
    };

    /**
     * restriction - averages data from fine to coarse grid
     *
     * @param fine_data      data on the fine grid
     * @param coarse_data    data on the coarse grid (will be modified)
     * @param region         the region to restrict
     * @param rf             refinement factor
     * @param policy         execution policy for parallel operations
     */
    template <typename T, size_type Dims>
    void restrict(
        const ndarray<T, Dims>& fine_data,
        ndarray<T, Dims>& coarse_data,
        const refinement_region<Dims>& region,
        size_type rf,
        const ExecutionPolicy<>& policy = ExecutionPolicy<>()
    )
    {
        // create operator for cell restriction
        restrict_cell<T, Dims> op{fine_data, coarse_data, rf};

        // parallel implementation for 1D case
        if constexpr (Dims == 1) {
            parallel_for(
                policy,
                region.min_bounds[0],
                region.max_bounds[0],
                [op] DEV(size_type ii) { op({ii}); }
            );
        }
        // sequential implementation for higher dimensions for now
        else {
            for (size_type ii = region.min_bounds[0]; ii < region.max_bounds[0];
                 ++ii) {
                for (size_type jj = (Dims > 1 ? region.min_bounds[1] : 0);
                     jj < (Dims > 1 ? region.max_bounds[1] : 1);
                     ++jj) {
                    for (size_type kk = (Dims > 2 ? region.min_bounds[2] : 0);
                         kk < (Dims > 2 ? region.max_bounds[2] : 1);
                         ++kk) {

                        if constexpr (Dims == 2) {
                            op({ii, jj});
                        }
                        else if constexpr (Dims == 3) {
                            op({ii, jj, kk});
                        }
                    }
                }
            }
        }
    }
}   // namespace simbi::refinement

#endif   // REFINEMENT_FUNCTIONS_HPP
