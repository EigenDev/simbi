/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            array_view.hpp
 *  * @brief           array view for ndarray data, similar to numpy
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

#ifndef ARRAY_VIEW_HPP
#define ARRAY_VIEW_HPP

#include "config.hpp"                          // for DUAL, real
#include "core/managers/array_props.hpp"       // for array_properties
#include "core/traits.hpp"                     // for is_maybe_v, has_value_type
#include "core/utility/operation_traits.hpp"   // for OperationTraits, StencilOp
#include "util/parallel/exec_policy.hpp"       // for ExecutionPolicy
#include <span>

namespace simbi {
    // forward declarations
    template <typename T, size_type Dims>
    class ndarray;

    template <typename T, size_type Dims>
    class array_view : public array_properties<T, Dims>
    {
      public:
        using raw_type = T;
        using value_type =
            typename std::conditional_t<is_maybe_v<T>, get_value_type_t<T>, T>;
        static constexpr size_type dim = Dims;

        DUAL array_view(
            const ndarray_t<T, Dims>& source,
            T* data,
            const uarray<Dims>& shape,
            const uarray<Dims>& strides,
            const uarray<Dims>& offsets
        )
            : array_properties<T, Dims>(
                  shape,
                  strides,
                  offsets,
                  this->compute_size(shape)
              ),
              source_(&source),
              data_(data, source.size())
        {
        }

        // Allow copying but track source
        array_view(const array_view&)            = default;
        array_view& operator=(const array_view&) = default;

        DUAL value_type& operator[](size_type ii)
        {
            // bounds check
            assert(ii < this->size_ && "Index out of boumds");
            return access(data_[ii] + this->compute_offset(this->offsets_));
        }
        DUAL value_type& operator[](size_type ii) const
        {
            // bounds check
            assert(ii < this->size_ && "Index out of bounds");
            return access(data_[ii] + this->compute_offset(this->offsets_));
        }
        template <typename... Indices>
        DUAL T& at(Indices... indices)
        {
            static_assert(sizeof...(Indices) == Dims);
            uarray<Dims> idx{static_cast<size_type>(indices)...};
            size_type offset = 0;

            if constexpr (global::col_major) {
                // Column major (k,j,i)
                for (size_type d = 0; d < Dims; ++d) {
                    if (idx[d] >= this->shape_[d]) {
                        return data_[0];   // bounds check
                    }
                    offset += idx[d] * this->strides_[d];
                }
            }
            else {
                // Row major (i,j,k)
                for (size_type d = Dims - 1; d == 0; --d) {
                    if (idx[d] >= this->shape_[d]) {
                        return data_[0];   // bounds check
                    }
                    offset += idx[d] * this->strides_[d];
                }
            }
            return access(data_[offset]);
        }
        DUAL auto data() const -> T* { return data_.data(); }
        DUAL auto data() -> T* { return data_.data(); }
        DUAL auto& access(T& val)
        {
            if constexpr (is_maybe_v<T>) {
                return val.value();
            }
            else {
                return val;
            }
        }
        DUAL auto& access(T& val) const
        {
            if constexpr (is_maybe_v<T>) {
                return val.value();
            }
            else {
                return val;
            }
        }

        // transform using stencil views of dependent ndarrays views
        template <typename... DependentViews, typename F>
        void stencil_transform(
            F op,
            const ExecutionPolicy<>& policy,
            const DependentViews&... arrays
        )
        {
            OperationTraits<StencilOp>::execute(this, op, policy, arrays...);
        }

        DUAL auto source_size() const { return data_.size(); }

        DUAL auto source_shape() const { return source_->shape(); }

        // Position-aware element that can do relative indexing
        template <typename DT = T>
        class stencil_view
        {
          public:
            DUAL stencil_view(
                std::span<DT> data,
                const uarray<Dims>& center_pos,
                const uarray<Dims>& shape,
                const uarray<Dims>& strides,
                const uarray<Dims>& offsets
            )
                : data_(data),
                  center_(center_pos),
                  shape_(shape),
                  strides_(strides),
                  offsets_(offsets)
            {
            }

            // Relative indexing from center
            DUAL get_value_type_t<DT>& at(int i, int j = 0, int k = 0) const
            {
                iarray<Dims> coords;
                size_type idx = 0;

                // center_ is in interior coordinates, convert to global by
                // adding offset
                if constexpr (global::col_major) {
                    // Column major: (i,j,k) -> k fastest
                    // Add offset to get global position
                    coords[0] = center_[0] + offsets_[0] + i;
                    if constexpr (Dims >= 2) {
                        coords[1] = center_[1] + offsets_[1] + j;
                    }
                    if constexpr (Dims >= 3) {
                        coords[2] = center_[2] + offsets_[2] + k;
                    }
                }
                else {
                    // Row major: (i,j,k) -> i fastest
                    coords[0] = center_[0] + offsets_[Dims - 1] + i;
                    if constexpr (Dims >= 2) {
                        coords[1] = center_[1] + offsets_[Dims - 2] + j;
                    }
                    if constexpr (Dims >= 3) {
                        coords[2] = center_[2] + offsets_[Dims - 3] + k;
                    }
                }

                // Calculate global linear index
                for (size_type d = 0; d < Dims; ++d) {
                    idx += coords[d] * strides_[d];
                }

                return access(data_[idx]);
            }

            // Get center position
            DUAL auto position() const
            {
                uarray<3> pos3d = {0, 0, 0};
                pos3d[0]        = center_[0];
                if constexpr (Dims > 1) {
                    pos3d[1] = center_[1];
                }
                if constexpr (Dims > 2) {
                    pos3d[2] = center_[2];
                }
                return pos3d;
            }

            // get global position
            DUAL auto global_position() const
            {
                uarray<3> pos3d = {0, 0, 0};
                pos3d[0]        = center_[0] + offsets_[Dims - 1];
                if constexpr (Dims > 1) {
                    pos3d[1] = center_[1] + offsets_[Dims - 2];
                }
                if constexpr (Dims > 2) {
                    pos3d[2] = center_[2] + offsets_[Dims - 3];
                }
                return pos3d;
            }

            // Direct value access at center
            DUAL get_value_type_t<DT>& value() const { return at(0); }

            // structured binding support
            template <size_type I>
            DUAL auto get() const
            {
                if constexpr (I < Dims) {
                    return position()[I];
                }
                else {
                    throw size_type{0};
                }
            }

            // we also need special access method in case we get maybe type
            DUAL auto& access(DT& val) const
            {
                if constexpr (has_value_type<DT>::value) {
                    return val.value();
                }
                else {
                    return val;
                }
            }

          private:
            // DT* data_;
            std::span<DT> data_;
            uarray<Dims> center_;
            uarray<Dims> shape_;
            uarray<Dims> strides_;
            uarray<Dims> offsets_;
        };

        template <typename DT>
        class device_stencil_view_t
        {
          public:
            DUAL device_stencil_view_t(
                DT* data,
                const uarray<Dims>& center_pos,
                const uarray<Dims>& shape,
                const uarray<Dims>& strides,
                const uarray<Dims>& offsets
            )
                : data_(data),
                  center_(center_pos),
                  shape_(shape),
                  strides_(strides),
                  offsets_(offsets)
            {
            }

            // Relative indexing from center
            DUAL get_value_type_t<DT>& at(int i, int j = 0, int k = 0) const
            {
                iarray<Dims> coords;
                size_type idx = 0;

                if constexpr (global::col_major) {
                    // Column major: (i,j,k) -> k fastest
                    coords[0] = center_[0] + offsets_[0] + i;
                    if constexpr (Dims >= 2) {
                        coords[1] = center_[1] + offsets_[1] + j;
                    }
                    if constexpr (Dims >= 3) {
                        coords[2] = center_[2] + offsets_[2] + k;
                    }
                }
                else {
                    // Row major: (i,j,k) -> i fastest
                    coords[0] = center_[0] + offsets_[Dims - 1] + i;
                    if constexpr (Dims >= 2) {
                        coords[1] = center_[1] + offsets_[Dims - 2] + j;
                    }
                    if constexpr (Dims >= 3) {
                        coords[2] = center_[2] + offsets_[Dims - 3] + k;
                    }
                }

                // Calculate global linear index
                for (size_type d = 0; d < Dims; ++d) {
                    idx += coords[d] * strides_[d];
                }

                return access(data_[idx]);
            }

            // Get center position
            DUAL auto position() const
            {
                uarray<3> pos3d = {0, 0, 0};
                pos3d[0]        = center_[0];
                if constexpr (Dims > 1) {
                    pos3d[1] = center_[1];
                }
                if constexpr (Dims > 2) {
                    pos3d[2] = center_[2];
                }
                return pos3d;
            }

            // get global position
            DUAL auto global_position() const
            {
                uarray<3> pos3d = {0, 0, 0};
                pos3d[0]        = center_[0] + offsets_[Dims - 1];
                if constexpr (Dims > 1) {
                    pos3d[1] = center_[1] + offsets_[Dims - 2];
                }
                if constexpr (Dims > 2) {
                    pos3d[2] = center_[2] + offsets_[Dims - 3];
                }
                return pos3d;
            }

            // Direct value access at center
            DUAL get_value_type_t<DT>& value() const { return at(0); }

            // we also need special access method in case we get maybe type
            DUAL auto& access(DT& val) const
            {
                if constexpr (has_value_type<DT>::value) {
                    return val.value();
                }
                else {
                    return val;
                }
            }

          private:
            DT* data_;   // Raw pointer instead of std::span
            uarray<Dims> center_;
            uarray<Dims> shape_;
            uarray<Dims> strides_;
            uarray<Dims> offsets_;
        };

      protected:
        // non-owning pointer to source data
        const ndarray_t<T, Dims>* source_;
        // view into data
        std::span<T> data_;
    };
}   // namespace simbi

#endif
