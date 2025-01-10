#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include "build_options.hpp"     // for global::BuildPlatform, DEV, Platform ...
#include "thread_pool.hpp"       // for (anonymous), ThreadPool, get_nthreads
#include "util/device_api.hpp"   // for api::setDevice
#include "util/exec_policy.hpp"   // for ExecutionPolicy
#include "util/launch.hpp"        // for launch
#include "util/range.hpp"         // for range

namespace simbi {
    template <typename index_type, typename F>
    void parallel_for(index_type first, index_type last, F function)
    {
        ExecutionPolicy p(last - first);
        parallel_for(p, first, last, function);
    }

    template <
        typename index_type,
        typename F,
        global::Platform P = global::BuildPlatform>
    void parallel_for(
        const ExecutionPolicy<>& p,
        index_type first,
        index_type last,
        F function
    )
    {
        const auto total_work      = last - first;
        const auto num_devices     = p.devices.size();
        const auto work_per_device = total_work / num_devices;

        if constexpr (global::on_gpu) {
            for (int i = 0; i < num_devices; ++i) {
                const auto device_first = first + i * work_per_device;
                const auto device_last  = (i == num_devices - 1)
                                              ? last
                                              : device_first + work_per_device;

                gpu::api::setDevice(p.devices[i]);
                simbi::launch(p, [=] DEV() {
                    for (auto idx : range(device_first, device_last)) {
                        function(idx);
                    }
                });
            }
        }
        else {
            for (int i = 0; i < num_devices; ++i) {
                const auto device_first = first + i * work_per_device;
                const auto device_last  = (i == num_devices - 1)
                                              ? last
                                              : device_first + work_per_device;

                simbi::pooling::getThreadPool()
                    .parallel_for(device_first, device_last, function);
            }
        }

        //         for (int i = 0; i < num_devices; ++i) {
        //             const auto device_first = first + i * work_per_device;
        //             const auto device_last =
        //                 (i == num_devices - 1) ? last : device_first +
        //                 work_per_device;

        //             gpu::api::setDevice(p.devices[i]);
        //             simbi::launch(p, [=] DEV() {
        // #if GPU_CODE
        //                 for (auto idx :
        //                      range(device_first, device_last,
        //                      globalThreadCount())) {
        //                     function(idx);
        //                 }
        // #else
        //                 simbi::pooling::getThreadPool().parallel_for(device_first,
        //                 device_last, function);
        // #endif
        //             });
        //         }
    }

    template <
        typename index_type,
        typename F,
        global::Platform P = global::BuildPlatform>
    void parallel_for(
        const ExecutionPolicy<>& p,
        index_type first,
        index_type last,
        index_type step,
        F function
    )
    {
        const auto total_work      = last - first;
        const auto num_devices     = p.devices.size();
        const auto work_per_device = total_work / num_devices;

        if constexpr (global::on_gpu) {
            for (int i = 0; i < num_devices; ++i) {
                const auto device_first = first + i * work_per_device;
                const auto device_last  = (i == num_devices - 1)
                                              ? last
                                              : device_first + work_per_device;

                gpu::api::setDevice(p.devices[i]);
                simbi::launch(p, [=] DEV() {
                    for (auto idx : range(device_first, device_last, step)) {
                        function(idx);
                    }
                });
            }
        }
        else {
            for (int i = 0; i < num_devices; ++i) {
                const auto device_first = first + i * work_per_device;
                const auto device_last  = (i == num_devices - 1)
                                              ? last
                                              : device_first + work_per_device;

                simbi::pooling::getThreadPool()
                    .parallel_for(device_first, device_last, step, function);
            }
        }
        //         for (int i = 0; i < num_devices; ++i) {
        //             const auto device_first = first + i * work_per_device;
        //             const auto device_last =
        //                 (i == num_devices - 1) ? last : device_first +
        //                 work_per_device;

        //             gpu::api::setDevice(p.devices[i]);
        //             simbi::launch(p, [=] DEV() {
        // #if GPU_CODE
        //                 for (auto idx :
        //                      range(device_first, device_last,
        //                      globalThreadCount())) {
        //                     function(idx);
        //                 }
        // #else
        //                 simbi::pooling::getThreadPool().parallel_for(device_first,
        //                 device_last, step, function);
        // #endif
        //             });
        //         }
    }

    template <
        typename index_type,
        typename F,
        global::Platform P = global::BuildPlatform>
    void parallel_for(const ExecutionPolicy<>& p, index_type last, F function)
    {
        const auto first = static_cast<index_type>(0);
        parallel_for(p, first, last, function);
    }

    template <typename F, global::Platform P = global::BuildPlatform>
    void parallel_for(const ExecutionPolicy<>& p, F function)
    {
        const auto last  = static_cast<luint>(p.get_full_extent());
        const auto first = static_cast<luint>(0);
        parallel_for(p, first, last, function);
    }
}   // namespace simbi

#endif