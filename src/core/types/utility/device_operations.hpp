/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            device_operations.hpp
 *  * @brief
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-03-12
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
 *  * 2025-03-12      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef DEVICE_OPERATORS_HPP
#define DEVICE_OPERATORS_HPP

#include "build_options.hpp"
#include "util/tools/helpers.hpp"   // for unravel_idx
#include <span>
#include <tuple>

namespace simbi {

    // Device-side operator for pointwise operations
    template <typename F, typename... Arrays>
    class DeviceOperator
    {
      public:
        DUAL DeviceOperator(F op, void* target, Arrays*... arrays)
            : op_(op),
              target_(static_cast<typename F::value_type*>(target)),
              arrays_(std::make_tuple(arrays...))
        {
        }

        template <size_t... Is>
        DUAL void apply_impl(size_type idx, std::index_sequence<Is...>)
        {
            target_[idx] = op_(target_[idx], std::get<Is>(arrays_)[idx]...);
        }

        DUAL void operator()(size_type idx)
        {
            apply_impl(idx, std::make_index_sequence<sizeof...(Arrays)>{});
        }

      private:
        F op_;
        typename F::value_type* target_;
        std::tuple<Arrays*...> arrays_;
    };

    // Device-side operator for stencil operations
    template <typename F, typename Shape, typename... Arrays>
    class DeviceStencil
    {
      public:
        DUAL
        DeviceStencil(F op, void* target, const Shape& shape, Arrays*... arrays)
            : op_(op),
              target_(static_cast<typename F::value_type*>(target)),
              shape_(shape),
              arrays_(std::make_tuple(arrays...))
        {
        }

        template <size_t... Is>
        DUAL void apply_impl(size_type idx, std::index_sequence<Is...>)
        {
            // Create views using array_view::stencil_view pattern
            using T     = typename F::value_type;
            auto coords = unravel_idx(idx, shape_);

            stencil_view<T> target_view(
                std::span<T>(target_, compute_size(shape_)),
                coords,
                shape_
            );

            auto dep_views = std::make_tuple(
                stencil_view<typename std::remove_pointer_t<Arrays>>(
                    std::span<typename std::remove_pointer_t<Arrays>>(
                        std::get<Is>(arrays_),
                        compute_size(shape_)
                    ),
                    coords,
                    shape_
                )...
            );

            // Apply operation using same pattern as array_view
            target_[idx] = std::apply(
                op_,
                std::tuple_cat(std::make_tuple(target_view), dep_views)
            );
        }

        DUAL void operator()(size_type idx)
        {
            apply_impl(idx, std::make_index_sequence<sizeof...(Arrays)>{});
        }

      private:
        F op_;
        typename F::value_type* target_;
        Shape shape_;
        std::tuple<Arrays*...> arrays_;

        // Use same stencil_view as array_view but device-friendly
        template <typename T>
        class stencil_view
        {
          public:
            DUAL
            stencil_view(std::span<T> data, const auto& pos, const Shape& shape)
                : data_(data), center_(pos), shape_(shape)
            {
            }

            DUAL T& at(int i, int j = 0, int k = 0) const
            {
                auto coords = center_;
                coords[0] += i;
                if constexpr (Shape::dims >= 2) {
                    coords[1] += j;
                }
                if constexpr (Shape::dims >= 3) {
                    coords[2] += k;
                }
                return data_[linear_index(coords, shape_)];
            }

            DUAL T& value() const { return at(0); }

          private:
            std::span<T> data_;
            decltype(unravel_idx(0, shape_)) center_;
            Shape shape_;
        };
    };

}   // namespace simbi

#endif