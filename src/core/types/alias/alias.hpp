/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            alias.hpp
 *  * @brief           home for most type aliases
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

#ifndef USING_DECL_HPP
#define USING_DECL_HPP

#include "adapter/device_adapter_api.hpp"
#include "config.hpp"
#include "core/types/containers/array.hpp"
#include "core/types/utility/smart_ptr.hpp"

namespace simbi {
    // namespace types {
    template <size_type N>
    using uarray = array_t<size_type, N>;

    template <size_type N>
    using iarray = array_t<lint, N>;

    template <typename T>
    struct gpuDeleter {
        void operator()(T* ptr) { gpu::api::free(ptr); }
    };

    template <typename T, typename Deleter>
    using unique_ptr = util::smart_ptr<T[], Deleter>;

    using size_type = std::size_t;
    using luint     = std::size_t;
    using lint      = int64_t;
    // }   // namespace types

}   // namespace simbi

#endif
