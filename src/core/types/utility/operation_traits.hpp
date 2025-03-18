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

#include "build_options.hpp"
#include "device_operations.hpp"

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
            size_t size,
            F op,
            const Policy& policy,
            Arrays*... arrays
        );
    };

    // Specialization for pointwise operations
    template <>
    struct OperationTraits<PointwiseOp> {
        template <typename T, typename Policy, typename F, typename... Arrays>
        static void execute(
            T* data,
            size_t size,
            F op,
            const Policy& policy,
            Arrays*... arrays
        )
        {
            if constexpr (global::on_gpu) {
                DeviceOperator<F, Arrays...> device_op(op, data, arrays...);
                parallel_for(policy, [=] DEV(size_t ii) { device_op(ii); });
            }
            else {
                parallel_for(policy, [&](size_t ii) {
                    data[ii] = op(data[ii], arrays[ii]...);
                });
            }
        }
    };

    // Specialization for pointwise operations with indices
    template <>
    struct OperationTraits<PointwiseOpIdx> {
        template <typename T, typename Policy, typename F, typename... Arrays>
        static void execute(
            T* data,
            size_t size,
            F op,
            const Policy& policy,
            Arrays*... arrays
        )
        {
            if constexpr (global::on_gpu) {
                DeviceOperator<F, Arrays...> device_op(op, data, arrays...);
                parallel_for(policy, [=] DEV(size_t ii) { device_op(ii); });
            }
            else {
                parallel_for(policy, [&](size_t ii) {
                    data[ii] = op(data[ii], ii, arrays[ii]...);
                });
            }
        }
    };

}   // namespace simbi

#endif