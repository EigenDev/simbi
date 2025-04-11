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
#include "core/traits.hpp"
#include "util/tools/helpers.hpp"   // for unravel_idx
#include <tuple>
#include <type_traits>

namespace simbi {

    // Device-side operator for pointwise operations
    template <typename F, typename T, typename... Arrays>
    class DeviceOperator
    {
      public:
        DUAL DeviceOperator(F op, T* target, Arrays*... arrays)
            : op_(op), target_(target), arrays_(std::make_tuple(arrays...))
        {
        }

        DUAL void operator()(size_type idx)
        {
            apply_tuple(idx, std::make_index_sequence<sizeof...(Arrays)>{});
        }

      private:
        template <size_t... Is>
        DUAL void apply_tuple(size_type idx, std::index_sequence<Is...>)
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

        DUAL DeviceStencilOperator(
            F op,
            MainView* main_view,
            DependentViews*... dependent_views
        )
            : op_(op),
              main_view_(main_view),
              dependent_views_(std::make_tuple(dependent_views...))
        {
        }

        DUAL void operator()(size_type idx)
        {
            // Get local coordinates
            auto pos = main_view_->get_local_coords(idx);

            // Create device-side stencil view for center position
            typename MainView::template device_stencil_view<main_t> center_view(
                main_view_->data(),
                pos,
                main_view_->shape(),
                main_view_->strides(),
                main_view_->offsets()
            );

            // Apply operation with stencil views
            apply_stencil_op(
                center_view,
                idx,
                std::make_index_sequence<sizeof...(DependentViews)>{}
            );
        }

      private:
        template <size_t... Is>
        DUAL void
        apply_stencil_op(typename MainView::template device_stencil_view<main_t>& center_view, size_type idx, std::index_sequence<Is...>)
        {
            center_view.value() =
                op_(center_view, create_dependent_view<Is>(idx)...);
        }

        template <size_t I>
        DUAL auto create_dependent_view(size_type idx)
        {
            using view_t =
                std::tuple_element_t<I, std::tuple<DependentViews*...>>;

            // Dereference the pointer type to get the actual view type
            using actual_view_t = typename std::remove_pointer<view_t>::type;

            // Now get raw_type from the actual view type
            using raw_t = typename actual_view_t::raw_type;

            return typename MainView::template device_stencil_view<raw_t>(
                std::get<I>(dependent_views_)->data(),
                std::get<I>(dependent_views_)
                    ->get_local_coords(idx, main_view_->shape()),
                std::get<I>(dependent_views_)->shape(),
                std::get<I>(dependent_views_)->strides(),
                std::get<I>(dependent_views_)->offsets()
            );
        }

        F op_;
        MainView* main_view_;
        std::tuple<DependentViews*...> dependent_views_;
    };
}   // namespace simbi

#endif
