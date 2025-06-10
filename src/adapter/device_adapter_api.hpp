/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            device_api.hpp
 * @brief
 * @details
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
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * 2025-06-09      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */

#ifndef DEVICE_API_HPP
#define DEVICE_API_HPP

#include "cpu_backend.hpp"
#include "cuda_backend.hpp"
#include "device_backend.hpp"
#include "hip_backend.hpp"
#include <cstddef>

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

            inline void malloc(void* obj, size_t bytes)
            {
                adapter::get_device_backend().malloc(&obj, bytes);
            }

            inline void malloc_managed(void* obj, size_t bytes)
            {
                adapter::get_device_backend().malloc_managed(&obj, bytes);
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
            inline void event_create(devEvent_t* event)
            {
                adapter::get_device_backend().event_create(
                    reinterpret_cast<void**>(event)
                );
            }

            inline void event_destroy(devEvent_t event)
            {
                adapter::get_device_backend().event_destroy(
                    static_cast<void*>(event)
                );
            }

            inline void event_record(devEvent_t event)
            {
                adapter::get_device_backend().event_record(
                    static_cast<void*>(event)
                );
            }

            inline void event_synchronize(devEvent_t event)
            {
                adapter::get_device_backend().event_synchronize(
                    static_cast<void*>(event)
                );
            }

            inline void
            event_elapsed_time(float* time, devEvent_t start, devEvent_t end)
            {
                adapter::get_device_backend().event_elapsed_time(
                    time,
                    static_cast<void*>(start),
                    static_cast<void*>(end)
                );
            }

            // device management
            inline void get_device_count(int* count)
            {
                adapter::get_device_backend().get_device_count(count);
            }

            inline void get_device_properties(devProp_t* props, int device)
            {
                adapter::get_device_backend().get_device_properties(
                    static_cast<void*>(props),
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
            inline void stream_create(simbiStream_t* stream)
            {
                adapter::get_device_backend().stream_create(
                    reinterpret_cast<void**>(stream)
                );
            }

            inline void stream_destroy(simbiStream_t stream)
            {
                adapter::get_device_backend().stream_destroy(
                    static_cast<void*>(stream)
                );
            }

            inline void stream_synchronize(simbiStream_t stream)
            {
                adapter::get_device_backend().stream_synchronize(
                    static_cast<void*>(stream)
                );
            }

            inline void stream_wait_event(
                simbiStream_t stream,
                devEvent_t event,
                unsigned int flags = 0
            )
            {
                adapter::get_device_backend().stream_wait_event(
                    static_cast<void*>(stream),
                    static_cast<void*>(event),
                    flags
                );
            }

            inline void stream_query(simbiStream_t stream, int* status)
            {
                adapter::get_device_backend().stream_query(
                    static_cast<void*>(stream),
                    status
                );
            }

            // asynchronous operations
            inline void async_copy_host_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            )
            {
                adapter::get_device_backend().async_copy_host_to_device(
                    dst,
                    src,
                    bytes,
                    static_cast<void*>(stream)
                );
            }

            inline void async_copy_device_to_host(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            )
            {
                adapter::get_device_backend().async_copy_device_to_host(
                    dst,
                    src,
                    bytes,
                    static_cast<void*>(stream)
                );
            }

            inline void async_copy_device_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            )
            {
                adapter::get_device_backend().async_copy_device_to_device(
                    dst,
                    src,
                    bytes,
                    static_cast<void*>(stream)
                );
            }

            inline void memcpy_async(
                void* dst,
                const void* src,
                size_t bytes,
                simbiMemcpyKind kind,
                simbiStream_t stream
            )
            {
                adapter::get_device_backend().memcpy_async(
                    dst,
                    src,
                    bytes,
                    static_cast<int>(kind),
                    static_cast<void*>(stream)
                );
            }

            // peer operations
            inline void enable_peer_access(int device, unsigned int flags = 0)
            {
                adapter::get_device_backend().enable_peer_access(device, flags);
            }

            inline void peerCopyAsync(
                void* dst,
                int dst_device,
                const void* src,
                int src_device,
                size_t bytes,
                simbiStream_t stream
            )
            {
                adapter::get_device_backend().peer_copy_async(
                    dst,
                    dst_device,
                    src,
                    src_device,
                    bytes,
                    static_cast<void*>(stream)
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
                devFunction_t function,
                dim3 grid,
                dim3 block,
                void** args,
                size_t shared_mem,
                simbiStream_t stream
            )
            {
                adapter::types::dim3 adapter_grid(grid.x, grid.y, grid.z);
                adapter::types::dim3 adapter_block(block.x, block.y, block.z);

                adapter::get_device_backend().launch_kernel(
                    static_cast<void*>(function),
                    adapter_grid,
                    adapter_block,
                    args,
                    shared_mem,
                    static_cast<void*>(stream)
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
}   // namespace simbi
#endif
