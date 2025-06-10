/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            launch.hpp
 *  * @brief           Launch function object with no configuration
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
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
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef LAUNCH_HPP
#define LAUNCH_HPP

#include "adapter/device_types.hpp"
#include "kernel.hpp"                      // for kernel
#include "util/parallel/exec_policy.hpp"   // for ExecutionPolicy

namespace simbi {
    // Launch with explicit configuration
    template <typename Function, typename... Arguments>
    void launch(
        const ExecutionPolicy<>& policy,
        const int device,
        const adapter::stream_t<> stream,
        Function f,
        Arguments... args
    )
    {

#if GPU_ENABLED
        // If streams are specified, use them
        if (!policy.streams.empty()) {
            kernel<<<
                policy.get_device_grid_size(device),
                policy.block_size,
                policy.shared_mem_bytes,
                stream>>>(f, args...);
        }
        else {
            kernel<<<
                policy.get_device_grid_size(device),
                policy.block_size,
                policy.shared_mem_bytes>>>(f, args...);
        }
#else
        (void) device;   // Avoid unused parameter warning
        (void) stream;   // Avoid unused parameter warning
        (void) policy;   // Avoid unused parameter warning
        f(args...);
#endif
    }
}   // namespace simbi
#endif
