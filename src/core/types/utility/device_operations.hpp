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
}   // namespace simbi

#endif
