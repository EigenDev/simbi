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

#include "build_options.hpp"                // for DUAL, real
#include "core/managers/array_props.hpp"    // for array_properties
#include "core/traits.hpp"                  // for is_maybe_v, has_value_type
#include "util/parallel/exec_policy.hpp"    // for ExecutionPolicy
#include "util/parallel/parallel_for.hpp"   // for parallel_for
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
        DUAL array_view(
            const ndarray<T, Dims>& source,
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
            return access(data_[ii] + this->compute_offset(this->offsets_));
        }
        DUAL value_type& operator[](size_type ii) const
        {
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
            if constexpr (global::on_gpu) {
                // First prepare device arrays for views
                auto center_data    = this->data();
                auto center_shape   = this->shape();
                auto center_strides = this->strides();
                auto center_offsets = this->offsets();

                // Helper to extract properties from each array
                auto make_properties = [](const auto& arr) {
                    return std::make_tuple(
                        arr.data(),
                        arr.shape(),
                        arr.strides(),
                        arr.offsets()
                    );
                };

                // Create tuples of properties for dependent arrays
                auto dep_properties =
                    std::make_tuple(make_properties(arrays)...);

                // Launch kernel with device-friendly lambda
                parallel_for(policy, [=, this] DEV(size_type idx) {
                    stencil_view<T> center_view(
                        center_data,
                        get_local_coords(idx, center_shape),
                        center_shape,
                        center_strides,
                        center_offsets
                    );

                    // Create dependent views from properties tuple
                    auto dep_views = create_device_views<DependentViews...>(
                        idx,
                        dep_properties,
                        std::make_index_sequence<sizeof...(DependentViews)>{}
                    );

                    // Apply operation
                    center_view.value() =
                        call_device_op(op, center_view, dep_views);
                });
            }
            else {
                parallel_for(policy, [=, this](size_type idx) {
                    // Get local coordinates
                    auto pos = this->get_local_coords(idx);

                    // Create span from data pointer
                    std::span<T> data_span(this->data(), this->source_size());
                    stencil_view center_view(
                        data_span,
                        pos,
                        this->shape(),
                        this->strides(),
                        this->offsets()
                    );

                    auto all_views = std::make_tuple(
                        center_view,
                        stencil_view<
                            typename array_raw_type<DependentViews>::type>(
                            std::span<
                                typename array_raw_type<DependentViews>::type>(
                                arrays.data(),
                                arrays.source_size()
                            ),
                            arrays.get_local_coords(idx, this->shape()),
                            arrays.shape(),
                            arrays.strides(),
                            arrays.offsets()
                        )...
                    );

                    center_view.value() = std::apply(op, all_views);
                });
            }
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
            DUAL const auto position() const
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
            DUAL const auto global_position() const
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

        template <typename... Views, size_t... Is>
        DUAL auto
        create_device_views(const std::tuple<std::tuple<std::span<typename array_raw_type<Views>::type>, uarray<Dims>, uarray<Dims>, uarray<Dims>>...>& props, size_type idx, std::index_sequence<Is...>)
        {
            return std::make_tuple(
                stencil_view<typename array_raw_type<Views>::type>(
                    std::get<0>(std::get<Is>(props)),   // span
                    get_local_coords(
                        idx,
                        std::get<1>(std::get<Is>(props))
                    ),                                  // pos
                    std::get<1>(std::get<Is>(props)),   // shape
                    std::get<2>(std::get<Is>(props)),   // strides
                    std::get<3>(std::get<Is>(props))    // offsets
                )...
            );
        }

        template <typename F, typename View, typename Array, size_t N>
        DUAL auto call_device_op(
            F& op,
            View& center_view,
            const std::array<Array, N>& dep_views
        )
        {
            if constexpr (N == 1) {
                return op(center_view, dep_views[0]);
            }
            else if constexpr (N == 2) {
                return op(center_view, dep_views[0], dep_views[1]);
            }
            else if constexpr (N == 3) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2]
                );
            }
            else if constexpr (N == 4) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3]
                );
            }
            else if constexpr (N == 5) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3],
                    dep_views[4]
                );
            }
            else if constexpr (N == 6) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3],
                    dep_views[4],
                    dep_views[5]
                );
            }
            else if constexpr (N == 7) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3],
                    dep_views[4],
                    dep_views[5],
                    dep_views[6]
                );
            }
            else if constexpr (N == 8) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3],
                    dep_views[4],
                    dep_views[5],
                    dep_views[6],
                    dep_views[7]
                );
            }
            else if constexpr (N == 9) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3],
                    dep_views[4],
                    dep_views[5],
                    dep_views[6],
                    dep_views[7],
                    dep_views[8]
                );
            }
            else if constexpr (N == 10) {
                return op(
                    center_view,
                    dep_views[0],
                    dep_views[1],
                    dep_views[2],
                    dep_views[3],
                    dep_views[4],
                    dep_views[5],
                    dep_views[6],
                    dep_views[7],
                    dep_views[8],
                    dep_views[9]
                );
            }
            else {
                if constexpr (!global::on_gpu) {
                    throw std::runtime_error("Too many dependent views");
                }
            }
        }

      protected:
        // non-owning pointer to source data
        const ndarray<T, Dims>* source_;
        // view into data
        std::span<T> data_;
    };
}   // namespace simbi

#endif
