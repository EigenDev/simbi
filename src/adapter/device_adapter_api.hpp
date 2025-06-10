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

#include "cpu_backend.hpp"
#include "cuda_backend.hpp"
#include "device_backend.hpp"
#include "device_types.hpp"
#include "hip_backend.hpp"
#include <cstddef>
#include <thread>

namespace simbi {
    namespace gpu {
        // re-export the error namespace from adapter
        namespace error = adapter::error;

        // main API that delegates to the appropriate backend
        namespace api {
            // memory operations
            inline void
            copy_host_to_device(void* to, const void* from, size_t bytes)
            {
                adapter::get_device_backend()
                    .copy_host_to_device(to, from, bytes);
            }

            inline void
            copy_device_to_host(void* to, const void* from, size_t bytes)
            {
                adapter::get_device_backend()
                    .copy_device_to_host(to, from, bytes);
            }

            inline void
            copy_device_to_device(void* to, const void* from, size_t bytes)
            {
                adapter::get_device_backend()
                    .copy_device_to_device(to, from, bytes);
            }

            inline void malloc(void** obj, size_t bytes)
            {
                adapter::get_device_backend().malloc(obj, bytes);
            }

            inline void malloc_managed(void** obj, size_t bytes)
            {
                adapter::get_device_backend().malloc_managed(obj, bytes);
            }

            inline void free(void* obj)
            {
                adapter::get_device_backend().free(obj);
            }

            inline void memset(void* obj, int val, size_t bytes)
            {
                adapter::get_device_backend().memset(obj, val, bytes);
            }

            // event handling
            inline void event_create(adapter::event_t<>* event)
            {
                adapter::get_device_backend().event_create(event);
            }

            inline void event_destroy(adapter::event_t<> event)
            {
                adapter::get_device_backend().event_destroy(event);
            }

            inline void event_record(adapter::event_t<> event)
            {
                adapter::get_device_backend().event_record(event);
            }

            inline void event_synchronize(adapter::event_t<> event)
            {
                adapter::get_device_backend().event_synchronize(event);
            }

            inline void event_elapsed_time(
                float* time,
                adapter::event_t<> start,
                adapter::event_t<> end
            )
            {
                adapter::get_device_backend()
                    .event_elapsed_time(time, start, end);
            }

            // device management
            inline void get_device_count(int* count)
            {
                adapter::get_device_backend().get_device_count(count);
            }

            inline void get_device_properties(
                adapter::device_properties_t<>* props,
                int device
            )
            {
                adapter::get_device_backend().get_device_properties(
                    props,
                    device
                );
            }

            inline void set_device(int device)
            {
                adapter::get_device_backend().set_device(device);
            }

            inline void device_synch()
            {
                adapter::get_device_backend().device_synchronize();
            }

            // stream operations
            inline void stream_create(adapter::stream_t<>* stream)
            {
                adapter::get_device_backend().stream_create(stream);
            }

            inline void stream_destroy(adapter::stream_t<> stream)
            {
                adapter::get_device_backend().stream_destroy(stream);
            }

            inline void stream_synchronize(adapter::stream_t<> stream)
            {
                adapter::get_device_backend().stream_synchronize(stream);
            }

            inline void stream_wait_event(
                adapter::stream_t<> stream,
                adapter::event_t<> event,
                unsigned int flags = 0
            )
            {
                adapter::get_device_backend()
                    .stream_wait_event(stream, event, flags);
            }

            inline void stream_query(adapter::stream_t<> stream, int* status)
            {
                adapter::get_device_backend().stream_query(stream, status);
            }

            // asynchronous operations
            inline void async_copy_host_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .async_copy_host_to_device(dst, src, bytes, stream);
            }

            inline void async_copy_device_to_host(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .async_copy_device_to_host(dst, src, bytes, stream);
            }

            inline void async_copy_device_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .async_copy_device_to_device(dst, src, bytes, stream);
            }

            inline void memcpy_async(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::memcpy_kind_t<> kind,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .memcpy_async(dst, src, bytes, kind, stream);
            }

            // peer operations
            inline void enable_peer_access(int device, unsigned int flags = 0)
            {
                adapter::get_device_backend().enable_peer_access(device, flags);
            }

            inline void peer_copy_async(
                void* dst,
                int dst_device,
                const void* src,
                int src_device,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend().peer_copy_async(
                    dst,
                    dst_device,
                    src,
                    src_device,
                    bytes,
                    stream
                );
            }

            // host memory management
            inline void
            host_register(void* ptr, size_t size, unsigned int flags)
            {
                adapter::get_device_backend().host_register(ptr, size, flags);
            }

            inline void host_unregister(void* ptr)
            {
                adapter::get_device_backend().host_unregister(ptr);
            }

            inline void aligned_malloc(void** ptr, size_t size)
            {
                adapter::get_device_backend().aligned_malloc(ptr, size);
            }

            // specialized operations
            inline void
            memcpy_from_symbol(void* dst, const void* symbol, size_t count)
            {
                adapter::get_device_backend()
                    .memcpy_from_symbol(dst, symbol, count);
            }

            inline void
            prefetch_to_device(const void* obj, size_t bytes, int device = 0)
            {
                adapter::get_device_backend()
                    .prefetch_to_device(obj, bytes, device);
            }

            inline void launch_kernel(
                adapter::function_t<> function,
                adapter::types::dim3 grid,
                adapter::types::dim3 block,
                void** args,
                size_t shared_mem,
                adapter::stream_t<> stream
            )
            {
                adapter::types::dim3 adapter_grid(grid.x, grid.y, grid.z);
                adapter::types::dim3 adapter_block(block.x, block.y, block.z);

                adapter::get_device_backend().launch_kernel(
                    function,
                    adapter_grid,
                    adapter_block,
                    args,
                    shared_mem,
                    stream
                );
            }

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
            DEV inline void synchronize()
            {
                adapter::get_device_backend().synchronize_threads();
            }

        }   // namespace api
    }   // namespace gpu

    namespace grid {
        // return current execution index
        DUAL inline adapter::grid_index idx()
        {
            return adapter::grid_index::current<>();
        }

        // launch kernels with configuration
        template <typename Func, typename... Args>
        inline void launch(
            Func kernel,
            const adapter::grid::launch_config& config,
            Args&&... args
        )
        {
            adapter::launch_kernel(kernel, config, std::forward<Args>(args)...);
        }

        // launch single thread
        template <typename Func, typename... Args>
        inline void launch_single(Func kernel, Args&&... args)
        {
            adapter::launch_single_thread(kernel, std::forward<Args>(args)...);
        }

        // create configuration from block/thread counts
        inline adapter::grid::launch_config
        config(unsigned int blocks, unsigned int threads)
        {
            return adapter::grid::launch_config(
                adapter::types::dim3(blocks),
                adapter::types::dim3(threads)
            );
        }

        // create configuration from dimensions
        inline adapter::grid::launch_config config(
            adapter::types::dim3 grid,
            adapter::types::dim3 block,
            size_t shared_memory = 0
        )
        {
            return adapter::grid::launch_config(grid, block, shared_memory);
        }

        // calculate grid size from total elements
        inline adapter::types::dim3
        calculate_grid(size_t elements, unsigned int threads_per_block = 256)
        {
            return adapter::grid::calculate_grid(
                elements,
                adapter::types::dim3(threads_per_block)
            );
        }
    }   // namespace grid

    // Thread and block utilities
    STATIC unsigned int global_thread_idx()
    {
        return adapter::grid_index::current<>().global_thread_id();
    }

    STATIC unsigned int global_thread_count()
    {
        return adapter::grid_index::current<>().total_threads();
    }

    STATIC unsigned int get_thread_id()
    {
#if GPU_ENABLED
        return adapter::grid_index::current<>().thread_id();
#else
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    }

    STATIC unsigned int get_block_id()
    {
        return adapter::grid_index::current<>().block_id();
    }

    STATIC unsigned int get_threads_per_block()
    {
        return adapter::grid_index::current<>().threads_per_block();
    }
}   // namespace simbi
#endif   // DEVICE_ADAPTER_API_HPP
