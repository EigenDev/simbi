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

#include "config.hpp"
#include "core/types/alias.hpp"
#include "core/utility/helpers.hpp"   // for unravel_idx
#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace simbi::helpers;
namespace simbi {

    // Device-side operator for pointwise operations
    template <typename F, typename T, typename... Arrays>
    class DeviceOperator
    {
      public:
        DeviceOperator(F op, T* target, Arrays*... arrays)
            : op_(op), target_(target), arrays_(std::make_tuple(arrays...))
        {
        }

        DEV void operator()(std::uint64_t idx)
        {
            apply_tuple(idx, std::make_index_sequence<sizeof...(Arrays)>{});
        }

      private:
        template <size_t... Is>
        DEV void apply_tuple(std::uint64_t idx, std::index_sequence<Is...>)
        {
            target_[idx] = op_(target_[idx], std::get<Is>(arrays_)[idx]...);
        }

        F op_;
        T* target_;
        std::tuple<Arrays*...> arrays_;
    };

    template <typename F, typename MainView, typename... DependentViews>
    class DeviceStencilOperator
    {
      public:
        using main_t = typename MainView::raw_type;

        DeviceStencilOperator(
            F op,
            MainView* main_view,
            DependentViews*... dependent_views
        )
            : op_(op),
              main_view_(main_view),
              main_view_data_(main_view->data()),
              main_view_shape_(main_view->shape()),
              main_view_strides_(main_view->strides()),
              main_view_offsets_(main_view->offsets()),
              dependent_views_(std::make_tuple(dependent_views...))
        {
            // Extract and store data for each dependent view
            extract_dependent_data(
                std::make_index_sequence<sizeof...(DependentViews)>{}
            );
        }

        DEV void operator()(std::uint64_t idx)
        {
            // Get local coordinates
            const auto pos = unravel_idx<MainView::dim>(idx, main_view_shape_);

            // Create device-side stencil view for center position
            typename MainView::template device_stencil_view<main_t> center_view(
                main_view_data_,
                pos,
                main_view_shape_,
                main_view_strides_,
                main_view_offsets_
            );

            // Apply operation with stencil views
            apply_stencil_op(
                center_view,
                idx,
                std::make_index_sequence<sizeof...(DependentViews)>{}
            );
        }

      private:
        // Extract data for all dependent views
        template <size_t... Is>
        void extract_dependent_data(std::index_sequence<Is...>)
        {
            // Using parameter pack expansion to initialize arrays
            (void) std::initializer_list<std::int64_t>{
              (dependent_data_[Is]   = std::get<Is>(dependent_views_)->data(),
               dependent_shapes_[Is] = std::get<Is>(dependent_views_)->shape(),
               dependent_strides_[Is] =
                   std::get<Is>(dependent_views_)->strides(),
               dependent_offsets_[Is] =
                   std::get<Is>(dependent_views_)->offsets(),
               0)...
            };
        }
        template <size_t... Is>
        DEV void apply_stencil_op(
            typename MainView::template device_stencil_view<main_t>&
                center_view,
            std::uint64_t idx,
            std::index_sequence<Is...>
        )
        {
            center_view.value() =
                op_(center_view, create_dependent_view<Is>(idx)...);
        }

        template <size_t I>
        DEV auto create_dependent_view(std::uint64_t idx)
        {
            using view_t =
                std::tuple_element_t<I, std::tuple<DependentViews*...>>;
            using actual_view_t = typename std::remove_pointer<view_t>::type;
            using raw_t         = typename actual_view_t::raw_type;

            // Manually unravel the index for this dependent view
            auto pos = unravel_idx<MainView::dim>(idx, main_view_shape_);

            return typename MainView::template device_stencil_view<raw_t>(
                static_cast<raw_t*>(dependent_data_[I]),
                pos,
                dependent_shapes_[I],
                dependent_strides_[I],
                dependent_offsets_[I]
            );
        }

        F op_;
        MainView* main_view_;
        main_t* main_view_data_;
        uarray<MainView::dim> main_view_shape_;
        uarray<MainView::dim> main_view_strides_;
        uarray<MainView::dim> main_view_offsets_;
        std::tuple<DependentViews*...> dependent_views_;

        std::array<void*, sizeof...(DependentViews)> dependent_data_;
        std::array<uarray<MainView::dim>, sizeof...(DependentViews)>
            dependent_shapes_;
        std::array<uarray<MainView::dim>, sizeof...(DependentViews)>
            dependent_strides_;
        std::array<uarray<MainView::dim>, sizeof...(DependentViews)>
            dependent_offsets_;
    };
}   // namespace simbi

#endif
