/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            cpu_backend.hpp
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

#ifndef CPU_BACKEND_HPP
#define CPU_BACKEND_HPP

#include "device_backend.hpp"
#include <algorithm>   // for std::min
#include <chrono>      // for timing
#include <cstring>     // for memcpy, memset
#include <memory>      // for aligned_alloc

namespace simbi::adapter {

    // CPU backend specialization
    template <>
    class DeviceBackend<cpu_backend_tag>
    {
      public:
        // memory operations
        void copy_host_to_device(void* to, const void* from, std::size_t bytes)
        {
            std::memcpy(to, from, bytes);
        }

        void copy_device_to_host(void* to, const void* from, std::size_t bytes)
        {
            std::memcpy(to, from, bytes);
        }

        void
        copy_device_to_device(void* to, const void* from, std::size_t bytes)
        {
            std::memcpy(to, from, bytes);
        }

        void malloc(void** obj, std::size_t bytes)
        {
            *obj = std::malloc(bytes);
            if (*obj == nullptr) {
                throw error::runtime_error(
                    error::status_t::error,
                    "CPU malloc failed"
                );
            }
        }

        void malloc_managed(void** obj, std::size_t bytes)
        {
            // on CPU, regular malloc serves as managed memory
            malloc(obj, bytes);
        }

        void free(void* obj) { std::free(obj); }

        void memset(void* obj, int val, std::size_t bytes)
        {
            std::memset(obj, val, bytes);
        }

        // event handling (using std::chrono for timing)
        void event_create(void** event)
        {
            auto* timestamp =
                new std::chrono::high_resolution_clock::time_point();
            *event = static_cast<void*>(timestamp);
        }

        void event_destroy(void* event)
        {
            auto* timestamp =
                static_cast<std::chrono::high_resolution_clock::time_point*>(
                    event
                );
            delete timestamp;
        }

        void event_record(void* event)
        {
            auto* timestamp =
                static_cast<std::chrono::high_resolution_clock::time_point*>(
                    event
                );
            *timestamp = std::chrono::high_resolution_clock::now();
        }

        void event_synchronize(void* event)
        {
            // no-op on CPU, events are implicitly synchronized
            (void) event;
        }

        void event_elapsed_time(float* time, void* start, void* end)
        {
            auto* start_time =
                static_cast<std::chrono::high_resolution_clock::time_point*>(
                    start
                );
            auto* end_time =
                static_cast<std::chrono::high_resolution_clock::time_point*>(
                    end
                );

            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    *end_time - *start_time
                )
                    .count();
            // convert to milliseconds
            *time = static_cast<float>(duration) / 1000.0f;
        }

        // device management (CPU fallbacks)
        void get_device_count(int* count)
        {
            // cpu backend reports a single "device"
            *count = 1;
        }

        void get_device_properties(void* props, int device)
        {
            // no properties to report for CPU
            (void) props;
            (void) device;
        }

        void set_device(int device)
        {
            // calidate device ID (only 0 is valid for CPU)
            if (device != 0) {
                throw error::runtime_error(
                    error::status_t::error,
                    "Invalid device ID for CPU backend: " +
                        std::to_string(device)
                );
            }
        }

        void device_synchronize()
        {
            // no-op on CPU, all operations are synchronous
        }

        // Stream operations (CPU fallbacks)
        void stream_create(void** stream)
        {
            // dummy stream for compatibility
            *stream = nullptr;
        }

        void stream_destroy(void* stream)
        {
            // no-op for CPU
            (void) stream;
        }

        void stream_synchronize(void* stream)
        {
            // no-op for CPU
            (void) stream;
        }

        void
        stream_wait_event(void* stream, void* event, unsigned int flags = 0)
        {
            // no-op for CPU
            (void) stream;
            (void) event;
            (void) flags;
        }

        void stream_query(void* stream, int* status)
        {
            // always report streams as completed on CPU
            (void) stream;
            *status = 0;   // 0 typically means "completed" in GPU APIs
        }

        // asynchronous operations (synchronous on CPU)
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            (void) stream;   // ignore stream parameter on CPU
            copy_host_to_device(to, from, bytes);
        }

        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            (void) stream;   // ignore stream parameter on CPU
            copy_device_to_host(to, from, bytes);
        }

        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            (void) stream;   // ignore stream parameter on CPU
            copy_device_to_device(to, from, bytes);
        }

        void memcpy_async(
            void* to,
            const void* from,
            std::size_t bytes,
            int kind,
            void* stream
        )
        {
            (void) kind;     // kind is meaningless on CPU
            (void) stream;   // ignore stream parameter on CPU
            std::memcpy(to, from, bytes);
        }

        // peer operations (no-ops on CPU)
        void enable_peer_access(int device, unsigned int flags = 0)
        {
            // no-op for CPU
            (void) device;
            (void) flags;
        }

        void peer_copy_async(
            void* dst,
            int dst_device,
            const void* src,
            int src_device,
            std::size_t bytes,
            void* stream
        )
        {
            // validate device IDs (only 0 is valid for CPU)
            if (dst_device != 0 || src_device != 0) {
                throw error::runtime_error(
                    error::status_t::error,
                    "Invalid device IDs for CPU peer copy"
                );
            }

            (void) stream;   // ignore stream parameter on CPU
            std::memcpy(dst, src, bytes);
        }

        // Host memory management
        void host_register(void* ptr, std::size_t size, unsigned int flags)
        {
            // no-op for CPU
            (void) ptr;
            (void) size;
            (void) flags;
        }

        void host_unregister(void* ptr)
        {
            // no-op for CPU
            (void) ptr;
        }

        void aligned_malloc(void** ptr, std::size_t size)
        {
            // use aligned_alloc with 64-byte alignment (cache line size on most
            // CPUs)
            constexpr std::size_t alignment = 64;
            *ptr                            = std::aligned_alloc(
                alignment,
                (size + alignment - 1) & ~(alignment - 1)
            );

            if (*ptr == nullptr) {
                throw error::runtime_error(
                    error::status_t::error,
                    "CPU aligned_malloc failed"
                );
            }
        }

        // specialized operations
        void
        memcpy_from_symbol(void* dst, const void* symbol, std::size_t count)
        {
            // on CPU, symbols are just regular memory
            std::memcpy(dst, symbol, count);
        }

        void
        prefetch_to_device(const void* obj, std::size_t bytes, int device = 0)
        {
            // no-op for CPU
            (void) obj;
            (void) bytes;
            (void) device;
        }

        void launch_kernel(
            void* function,
            types::dim3 grid,
            types::dim3 block,
            void** args,
            std::size_t shared_mem,
            void* stream
        )
        {
            // cannot implement kernel launching for CPU without more context
            throw error::runtime_error(
                error::status_t::error,
                "Kernel launching not implemented for CPU backend"
            );
        }

        // atomic operations
        template <typename T>
        T atomic_min(T* address, T val)
        {
            T old    = *address;
            *address = std::min(*address, val);
            return old;
        }

        template <typename T>
        T atomic_add(T* address, T val)
        {
            T old = *address;
            *address += val;
            return old;
        }

        // thread synchronization
        void synchronize_threads()
        {
            // no-op on CPU
        }
    };

}   // namespace simbi::adapter
#endif
