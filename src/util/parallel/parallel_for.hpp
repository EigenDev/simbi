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
        const auto total_work  = last - first;
        const auto num_devices = policy.devices.size();

        if constexpr (global::on_gpu) {
            // Enable peer access if configured
            if (policy.config.enable_peer_access) {
                for (int i = 0; i < num_devices; i++) {
                    for (int j = 0; j < num_devices; j++) {
                        if (i != j) {
                            gpu::api::enablePeerAccess(policy.devices[j]);
                        }
                    }
                }
            }

            // Create streams if needed
            std::vector<simbiStream_t> local_streams;
            if (policy.streams.empty()) {
                local_streams.resize(num_devices);
                for (auto& stream : local_streams) {
                    gpu::api::streamCreate(&stream);
                }
            }

            // Launch on each device
            for (int dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
                const auto device_first =
                    first + dev_idx * (total_work / num_devices);
                const auto device_last =
                    (dev_idx == num_devices - 1)
                        ? last
                        : device_first + (total_work / num_devices);

                policy.switch_to_device(policy.devices[dev_idx]);
                auto stream = policy.streams.empty() ? local_streams[dev_idx]
                                                     : policy.streams[dev_idx];

                // Launch kernel with appropriate stream
                launch(policy, dev_idx, stream, [=] DEV() {
                    for (auto idx : range(
                             device_first,
                             device_last,
                             globalThreadCount()
                         )) {
                        function(idx);
                    }
                });

                // Optional halo exchange between devices
                if (policy.config.halo_radius > 0 &&
                    dev_idx < num_devices - 1) {
                    const size_type halo_size =
                        policy.config.halo_radius *
                        sizeof(typename std::invoke_result_t<F, index_type>);

                    if (policy.config.halo_mode == HaloExchangeMode::ASYNC) {
                        // Async halo exchange
                        gpu::api::peerCopyAsync(
                            // Right halo of current device to left halo of next
                            // device
                            static_cast<void*>(
                                reinterpret_cast<char*>(device_first) +
                                device_last - halo_size
                            ),
                            policy.devices[dev_idx],
                            static_cast<void*>(reinterpret_cast<char*>(
                                device_first + total_work / num_devices
                            )),
                            policy.devices[dev_idx + 1],
                            halo_size,
                            stream
                        );
                    }
                }
            }

            // Cleanup streams
            if (policy.streams.empty()) {
                for (auto stream : local_streams) {
                    gpu::api::streamDestroy(stream);
                }
            }

            // Synchronize if using sync halo exchange
            if (policy.config.halo_mode == HaloExchangeMode::SYNC) {
                policy.synchronize();
            }
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
