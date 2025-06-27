/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            operation_traits.hpp
 *  * @brief           home for host and device-side operation traits
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

#ifndef OPERATION_TRAITS_HPP
#define OPERATION_TRAITS_HPP

#include "config.hpp"
#include "core/traits.hpp"
#include "core/types/alias.hpp"
#include "device_operations.hpp"
#include "util/parallel/parallel_for.hpp"
#include <span>

namespace simbi {

    // Base operation tag types
    struct PointwiseOp {
    };

    struct PointwiseOpIdx {
    };

    struct StencilOp {
    };

    struct ReduceOp {
    };

    // Operation traits interface
    template <typename OpType>
    struct OperationTraits {
        template <typename T, typename Policy, typename F, typename... Arrays>
        static void execute(
            T* data,
            std::uint64_t size,
            F op,
            const Policy& policy,
            Arrays*... arrays
        );
    };

    // Specialization for pointwise operations
    template <>
    struct OperationTraits<PointwiseOp> {
        template <typename T, typename Policy, typename F, typename... Arrays>
        static void
        execute(T* data, F op, const Policy& policy, Arrays*... arrays)
        {
            if constexpr (platform::is_gpu) {
                DeviceOperator<F, T, Arrays...> device_op(op, data, arrays...);
                parallel_for(policy, [=] DEV(std::uint64_t ii) mutable {
                    device_op(ii);
                });
            }
            else {
                parallel_for(policy, [&](std::uint64_t ii) {
                    data[ii] = op(data[ii], arrays[ii]...);
                });
            }
        }
    };

    // Specialization for pointwise operations with indices
    template <>
    struct OperationTraits<PointwiseOpIdx> {
        template <typename T, typename Policy, typename F, typename... Arrays>
        static void
        execute(T* data, F op, const Policy& policy, Arrays*... arrays)
        {
            if constexpr (platform::is_gpu) {
                DeviceOperator<F, T, Arrays...> device_op(op, data, arrays...);
                parallel_for(policy, [=] DEV(std::uint64_t ii) mutable {
                    device_op(ii);
                });
            }
            else {
                parallel_for(policy, [&](std::uint64_t ii) {
                    data[ii] = op(data[ii], ii, arrays[ii]...);
                });
            }
        }
    };

    // Specialization for stencil operations
    template <>
    struct OperationTraits<StencilOp> {
        template <
            typename MainView,
            typename F,
            typename Policy,
            typename... DependentViews>
        static void execute(
            MainView* main_view,
            F op,
            const Policy& policy,
            DependentViews&... dependent_views
        )
        {
            // get underlying raw type from main view
            using main_t = typename MainView::raw_type;
            if constexpr (platform::is_gpu) {
                // TODO: Implement GPU version
                DeviceStencilOperator<F, MainView, DependentViews...> device_op(
                    op,
                    main_view,
                    &dependent_views...
                );

                parallel_for(policy, [=] DEV(std::uint64_t idx) mutable {
                    device_op(idx);
                });
            }
            else {
                parallel_for(policy, [=](std::uint64_t idx) {
                    // Get local coordinates
                    auto pos = main_view->get_local_coords(idx);

                    // Create span from data pointer
                    std::span<main_t> data_span(
                        main_view->data(),
                        main_view->source_size()
                    );

                    // Create stencil view for center position
                    typename MainView::stencil_view center_view(
                        data_span,
                        pos,
                        main_view->shape(),
                        main_view->strides(),
                        main_view->offsets()
                    );

                    // Create tuple of dependent views
                    auto dependent_views_tuple = std::make_tuple(
                        typename MainView::template stencil_view_t<
                            typename array_raw_type<DependentViews>::type>(
                            std::span<
                                typename array_raw_type<DependentViews>::type>(
                                dependent_views.data(),
                                dependent_views.source_size()
                            ),
                            dependent_views
                                .get_local_coords(idx, main_view->shape()),
                            dependent_views.shape(),
                            dependent_views.strides(),
                            dependent_views.offsets()
                        )...
                    );

                    // Apply operation with all stencil views
                    center_view.value() = std::apply(
                        [&op, &center_view](auto&... views) {
                            return op(center_view, views...);
                        },
                        dependent_views_tuple
                    );
                });
            }
        }
    };

}   // namespace simbi

#endif
