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

#include "util/parallel/exec_policy.hpp"   // for ExecutionPolicy

namespace simbi {
    // Launch function object with no configuration
    // Launch with no configuration
    template <typename Function, typename... Arguments>
    void launch(Function f, Arguments... args)
    {
        f(args...);
    }

    // Launch with explicit (or partial) configuration
    template <typename Function, typename... Arguments>
    void launch(
        const ExecutionPolicy<>& policy,
        const int device,
        Function f,
        Arguments... args
    )
    {

#if GPU_CODE
        // If streams are specified, use them
        if (!policy.streams.empty()) {
            Kernel<<<
                policy.get_device_grid_size(device),
                policy.block_size,
                policy.shared_mem_bytes,
                policy.streams[device % policy.streams.size()]>>>(f, args...);
        }
        else {
            Kernel<<<
                policy.get_device_grid_size(device),
                policy.block_size,
                policy.shared_mem_bytes>>>(f, args...);
        }
#else
        f(args...);
#endif
    }
}   // namespace simbi
#endif
