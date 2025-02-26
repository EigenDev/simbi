/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            parallel_for.hpp
 *  * @brief           general parallel_for loop that can be used on the any
 * architecture
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
#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include "build_options.hpp"   // for global::BuildPlatform, DEV, Platform ...
#include "core/types/utility/range.hpp"    // for range
#include "util/parallel/exec_policy.hpp"   // for ExecutionPolicy
#include "util/parallel/launch.hpp"        // for launch
#include "util/parallel/thread_pool.hpp"   // for (anonymous), ThreadPool, get_nthreads
#include "util/tools/device_api.hpp"   // for api::setDevice

namespace simbi {
    template <
        typename index_type,
        typename F,
        global::Platform P = global::BuildPlatform>
    void parallel_for(
        const ExecutionPolicy<>& policy,
        index_type first,
        index_type last,
        F function
    )
    {
        const auto total_work      = last - first;
        const auto num_devices     = policy.devices.size();
        const auto work_per_device = total_work / num_devices;
        if constexpr (global::on_gpu) {
            for (int dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
                const auto device_first = first + dev_idx * work_per_device;
                const auto device_last  = (dev_idx == num_devices - 1)
                                              ? last
                                              : device_first + work_per_device;

                policy.switch_to_device(policy.devices[dev_idx]);
                simbi::launch(policy, dev_idx, [=] DEV() {
                    for (auto idx : range(
                             device_first,
                             device_last,
                             globalThreadCount()
                         )) {
                        function(idx);
                    }
                });
            }
            // Synchronize streams if needed
            policy.synchronize();
        }
        else {
            simbi::pooling::getThreadPool().parallel_for(first, last, function);
        }
    }

    template <
        typename index_type,
        typename F,
        global::Platform P = global::BuildPlatform>
    void
    parallel_for(const ExecutionPolicy<>& policy, index_type last, F function)
    {
        const auto first = static_cast<index_type>(0);
        parallel_for(policy, first, last, function);
    }

    template <typename F, global::Platform P = global::BuildPlatform>
    void parallel_for(const ExecutionPolicy<>& policy, F function)
    {
        const auto last  = static_cast<luint>(policy.get_full_extent());
        const auto first = static_cast<luint>(0);
        parallel_for(policy, first, last, function);
    }
}   // namespace simbi

#endif