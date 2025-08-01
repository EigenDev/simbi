/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            cpu_backend.hpp
 * @brief           CPU implementation of device backend
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
 * @depends         None
 * @platform        Linux, MacOS
 * @parallel        CPU (OpenMP)
 *
 *==============================================================================
 */

#ifndef CPU_BACKEND_HPP
#define CPU_BACKEND_HPP

#include "device_backend.hpp"
#include "device_types.hpp"
#include <algorithm>   // for std::min
#include <chrono>      // for timing
#include <cstdint>     // for std::int64_t, std::uint64_t
#include <cstdlib>
#include <cstring>   // for memcpy, memset
#include <string>    // for to_string
#include <thread>    // for std::thread::hardware_concurrency

namespace simbi::adapter {
    // Helper function to handle unused parameters
    template <typename T>
    void unused_param(const T& /*param*/)
    {
    }

    // CPU backend specialization
    template <>
    class DeviceBackend<cpu_backend_tag>
    {

      public:
        // Memory operations
        void copy_host_to_device(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/
        )
        {
            // no-op
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

        void malloc(void** /*obj*/, std::size_t /*bytes*/)
        {
            // no-op on CPU
        }

        void malloc_managed(void**, std::size_t)
        {
            // no-op on CPU
        }

        void free(void* obj) { std::free(obj); }

        void memset(void* obj, std::int64_t val, std::size_t bytes)
        {
            std::memset(obj, val, bytes);
        }

        void event_create(adapter::event_t<cpu_backend_tag>* event)
        {
            (*event).value = std::chrono::high_resolution_clock::time_point();
        }

        void event_destroy(adapter::event_t<cpu_backend_tag> /*event*/)
        {
            // no-op on CPU, events are just time points
        }

        void event_record(
            adapter::event_t<cpu_backend_tag>& event,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            event.value = std::chrono::high_resolution_clock::now();
        }

        void event_synchronize(adapter::event_t<cpu_backend_tag> /*event*/)
        {
            // No-op on CPU, events are implicitly synchronized
        }

        void event_elapsed_time(
            float* time,
            adapter::event_t<cpu_backend_tag> start,
            adapter::event_t<cpu_backend_tag> end
        )
        {
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end.value - start.value
                )
                    .count();

            // convert to milliseconds to match CUDA behavior
            *time = static_cast<float>(duration) / 1000.0f;
        }

        // Device management
        void get_device_count(int* count)
        {
            // CPU backend reports a single "device"
            // [TODO]: update this for MPI later
            *count = 1;
        }

        void get_device_properties(
            adapter::device_properties_t<cpu_backend_tag>* props,
            std::int64_t device
        )
        {
            if (device != 0) {
                throw error::runtime_error(
                    error::status_t::error,
                    "Invalid device ID for CPU backend: " +
                        std::to_string(device)
                );
            }

            std::strncpy(props->name, "CPU", sizeof(props->name) - 1);
            props->major               = 1;
            props->minor               = 0;
            props->totalGlobalMem      = 0;
            props->multiProcessorCount = std::thread::hardware_concurrency();
            props->maxThreadsPerBlock  = 1;
            props->maxThreadsPerMultiProcessor = 1;
            props->maxThreadsDim               = types::dim3(1, 1, 1);
            props->maxGridSize                 = types::dim3(1, 1, 1);
        }

        void set_device(std::int64_t device)
        {
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
            // No-op on CPU, all operations are synchronous
        }

        // stream operations (CPU fallbacks)
        void stream_create(adapter::stream_t<cpu_backend_tag>* stream)
        {
            // dummy stream for compatibility
            *stream = {};
        }

        void stream_destroy(adapter::stream_t<cpu_backend_tag> /*stream*/)
        {
            // No-op for CPU
        }

        void stream_synchronize(adapter::stream_t<cpu_backend_tag> /*stream*/)
        {
            // no-op for CPU
        }

        void stream_wait_event(
            adapter::stream_t<cpu_backend_tag> /*stream*/,
            adapter::event_t<cpu_backend_tag> /*event*/,
            std::uint64_t /*flags*/ = 0
        )
        {
            // no-op for CPU
        }

        void
        stream_query(adapter::stream_t<cpu_backend_tag> /*stream*/, int* status)
        {
            // always report streams as completed on CPU
            *status = 0;   // 0 typically means "completed" in GPU APIs
        }

        // asynchronous operations (synchronous on CPU)
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            // ignore stream parameter on CPU
            copy_host_to_device(to, from, bytes);
        }

        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            // ignore stream parameter on CPU
            copy_device_to_host(to, from, bytes);
        }

        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            // ignore stream parameter on CPU
            copy_device_to_device(to, from, bytes);
        }

        void memcpy_async(
            void* to,
            const void* from,
            std::size_t bytes,
            adapter::memcpy_kind_t<cpu_backend_tag> /*kind*/,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            // kind is meaningless on CPU
            // ignore stream parameter on CPU
            std::memcpy(to, from, bytes);
        }

        // Peer operations (no-ops on CPU)
        void
        enable_peer_access(std::int64_t /*device*/, std::uint64_t /*flags*/ = 0)
        {
            // No-op for CPU
        }

        void peer_copy_async(
            void* dst,
            std::int64_t dst_device,
            const void* src,
            std::int64_t src_device,
            std::size_t bytes,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            // validate device IDs (only 0 is valid for CPU)
            if (dst_device != 0 || src_device != 0) {
                throw error::runtime_error(
                    error::status_t::error,
                    "Invalid device IDs for CPU peer copy"
                );
            }

            // ignore stream parameter on CPU
            std::memcpy(dst, src, bytes);
        }

        void host_register(
            void* /*ptr*/,
            std::size_t /*size*/,
            std::uint64_t /*flags*/
        )
        {
            // No-op for CPU
        }

        void host_unregister(void* /*ptr*/)
        {
            // No-op for CPU
        }

        void aligned_malloc(void** ptr, std::size_t size)
        {
            // use aligned_alloc with 64-byte alignment (cache line size on most
            // CPUs)
            constexpr std::size_t alignment = 64;
            std::size_t aligned_size =
                (size + alignment - 1) & ~(alignment - 1);

            *ptr = std::aligned_alloc(alignment, aligned_size);

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
            // On CPU, symbols are just regular memory
            std::memcpy(dst, symbol, count);
        }

        void prefetch_to_device(
            const void* /*obj*/,
            std::size_t /*bytes*/,
            std::int64_t /*device*/ = 0
        )
        {
            // No-op for CPU
        }

        void launch_kernel(
            adapter::function_t<cpu_backend_tag> function,
            types::dim3 /*grid*/,
            types::dim3 /*block*/,
            void** /*args*/,
            std::size_t /*shared_mem*/,
            adapter::stream_t<cpu_backend_tag> /*stream*/
        )
        {
            // Simple implementation for CPU: just call the function directly
            // This assumes function is a standard function pointer
            // [TODO]: this should be a std::function to hold any callable
            // type, not just a function pointer. also, we should parallelize
            // this piece
            if (function) {
                function();
            }
            else {
                throw error::runtime_error(
                    error::status_t::error,
                    "CPU kernel launch failed: null function pointer"
                );
            }
        }

        // Atomic operations
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

        // Thread synchronization
        void synchronize_threads()
        {
            // No-op on CPU
        }
    };

}   // namespace simbi::adapter
#endif
