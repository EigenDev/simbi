/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            array_props.hpp
 *  * @brief           a struct to hold properties of an array-type
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-21
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-21      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef ARRAY_PROPS_HPP
#define ARRAY_PROPS_HPP

#include "build_options.hpp"
#include "core/types/alias/alias.hpp"
#include "util/tools/helpers.hpp"

namespace simbi {
    template <typename T, size_type Dims>
    struct array_properties {
        // Ctor
        DUAL array_properties(
            const uarray<Dims>& shape,
            const uarray<Dims>& strides,
            const uarray<Dims>& offsets,
            size_type size
        )
            : shape_(shape), strides_(strides), offsets_(offsets), size_(size)
        {
        }

        // default ctor
        array_properties() = default;

        // copy ctor
        array_properties(const array_properties&) = default;
        // move ctor
        array_properties(array_properties&&) = default;

        // // copy assignment
        array_properties& operator=(const array_properties&) = default;

        // helper to compute local coordinates from linear index
        DEV uarray<Dims> get_local_coords(size_type idx) const
        {
            return helpers::memory_layout_coordinates<Dims>(idx, shape_);
        }

        DEV uarray<Dims> get_local_coords(size_type idx, const auto shape) const
        {
            return helpers::memory_layout_coordinates<Dims>(idx, shape);
        }

        DUAL static size_type compute_size(const uarray<Dims>& dims)
        {
            size_type size = 1;
            for (size_type ii = 0; ii < Dims; ++ii) {
                size *= dims[ii];
            }
            return size;
        }

        DUAL static auto compute_strides(const uarray<Dims>& dims)
            -> simbi::array_t<size_type, Dims>
        {
            uarray<Dims> strides;

            if constexpr (global::col_major) {
                // Column major (i,j,k): k fastest
                strides[Dims - 1] = 1;   // k stride
                for (size_type ii = Dims - 2; ii == 0; --ii) {
                    strides[ii] = strides[ii + 1] * dims[ii + 1];
                }
                // Result: strides = {nj*nk, nk, 1}
                // For (i,j,k) input order
            }
            else {
                // Row major (i,j,k): i fastest
                strides[0] = 1;   // i stride
                for (size_type ii = 1; ii < Dims; ++ii) {
                    strides[ii] = strides[ii - 1] * dims[Dims - ii];
                }
                // Result: strides = {1, ni, ni*nj}
                // For (i,j,k) input order
            }

            return strides;
        };

        DUAL size_type compute_offset(const uarray<Dims>& offsets) const
        {
            size_type offset = 0;
            for (size_type ii = 0; ii < Dims; ++ii) {
                offset += offsets[ii] * strides_[ii];
            }
            return offset;
        }

        DUAL auto strides() const -> uarray<Dims> { return strides_; };

        DUAL auto offsets() const -> uarray<Dims> { return offsets_; };

        DUAL auto shape() const -> uarray<Dims> { return shape_; };

        DUAL auto size() const -> size_type { return size_; };

      protected:
        uarray<Dims> shape_;
        uarray<Dims> strides_;
        uarray<Dims> offsets_;
        size_type size_{0};
    };
}   // namespace simbi

#endif
