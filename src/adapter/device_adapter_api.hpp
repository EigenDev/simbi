/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            device_adapter_api.hpp
 * @brief           API layer for device operations
 * @details         Provides a unified API that delegates to appropriate backend
 *
 * @version         0.8.0
 * @date            2025-06-09
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 */
#ifndef DEVICE_ADAPTER_API_HPP
#define DEVICE_ADAPTER_API_HPP

#include "config.hpp"
#include "cpu_backend.hpp"
#include "cuda_backend.hpp"
#include "device_backend.hpp"
#include "device_types.hpp"
#include "hip_backend.hpp"
#include <cstddef>
#include <cstdint>
#include <thread>

namespace simbi {
    namespace gpu {
        // re-export the error namespace from adapter
        namespace error = adapter::error;

        // main API that delegates to the appropriate backend
        namespace api {
            // memory operations
            void copy_host_to_device(void* to, const void* from, size_t bytes);

            void copy_device_to_host(void* to, const void* from, size_t bytes);

            void
            copy_device_to_device(void* to, const void* from, size_t bytes);

            void malloc(void** obj, size_t bytes);

            void malloc_managed(void** obj, size_t bytes);

            void free(void* obj);

            void memset(void* obj, std::int64_t val, size_t bytes);

            // event handling
            void event_create(adapter::event_t<>* event);

            void event_destroy(adapter::event_t<> event);

            void event_record(
                adapter::event_t<> event,
                adapter::stream_t<> stream = {}
            );

            void event_synchronize(adapter::event_t<> event);

            void event_elapsed_time(
                float* time,
                adapter::event_t<> start,
                adapter::event_t<> end
            );

            // device management
            void get_device_count(std::int64_t* count);

            void get_device_properties(
                adapter::device_properties_t<>* props,
                std::int64_t device
            );

            void set_device(std::int64_t device);

            void device_synch();

            // stream operations
            void stream_create(adapter::stream_t<>* stream);

            void stream_destroy(adapter::stream_t<> stream);

            void stream_synchronize(adapter::stream_t<> stream);

            void stream_wait_event(
                adapter::stream_t<> stream,
                adapter::event_t<> event,
                std::uint64_t flags = 0
            );

            void stream_query(adapter::stream_t<> stream, int* status);

            // asynchronous operations
            void async_copy_host_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            );

            void async_copy_device_to_host(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            );

            void async_copy_device_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            );

            void memcpy_async(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::memcpy_kind_t<> kind,
                adapter::stream_t<> stream
            );

            // peer operations
            void
            enable_peer_access(std::int64_t device, std::uint64_t flags = 0);

            void peer_copy_async(
                void* dst,
                std::int64_t dst_device,
                const void* src,
                std::int64_t src_device,
                size_t bytes,
                adapter::stream_t<> stream
            );

            // host memory management
            void host_register(void* ptr, size_t size, std::uint64_t flags);

            void host_unregister(void* ptr);

            void aligned_malloc(void** ptr, size_t size);

            // specialized operations
            void
            memcpy_from_symbol(void* dst, const void* symbol, size_t count);

            void prefetch_to_device(
                const void* obj,
                size_t bytes,
                std::int64_t device = 0
            );

            void launch_kernel(
                adapter::function_t<> function,
                adapter::types::dim3 grid,
                adapter::types::dim3 block,
                void** args,
                size_t shared_mem,
                adapter::stream_t<> stream
            );

            // atomic operations
            template <typename T>
            DEV T atomic_min(T* address, T val)
            {
                return adapter::get_device_backend().atomic_min(address, val);
            }

            template <typename T>
            DEV T atomic_add(T* address, T val)
            {
                return adapter::get_device_backend().atomic_add(address, val);
            }

            // thread synchronization
            DEV void synchronize();

        }   // namespace api
    }   // namespace gpu

    namespace grid {
        // return current execution index
        DUAL adapter::grid_index idx();

        // launch kernels with configuration
        template <typename Func, typename... Args>
        void launch(
            Func kernel,
            const adapter::grid::launch_config& config,
            Args&&... args
        )
        {
            adapter::launch_kernel(kernel, config, std::forward<Args>(args)...);
        }

        // launch single thread
        template <typename Func, typename... Args>
        void launch_single(Func kernel, Args&&... args)
        {
            adapter::launch_single_thread(kernel, std::forward<Args>(args)...);
        }

        // create configuration from block/thread counts
        adapter::grid::launch_config
        config(std::uint64_t blocks, std::uint64_t threads);

        // create configuration from dimensions
        adapter::grid::launch_config config(
            adapter::types::dim3 grid,
            adapter::types::dim3 block,
            size_t shared_memory = 0
        );

        // calculate grid size from total elements
        adapter::types::dim3 calculate_grid(
            size_t elements,
            adapter::types::dim3 threads_per_block = {256, 1, 1}
        );
    }   // namespace grid

    // Thread and block utilities
    DEV auto global_thread_idx();

    DEV auto global_thread_count();

    DEV auto get_thread_id();

    DEV auto get_block_id();

    DEV auto get_threads_per_block();
}   // namespace simbi
#endif   // DEVICE_ADAPTER_API_HPP
