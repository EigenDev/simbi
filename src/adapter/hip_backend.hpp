/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            hip_backend.hpp
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

#ifndef HIP_BACKEND_HPP
#define HIP_BACKEND_HPP

#include "device_backend.hpp"

// only include HIP headers if HIP is enabled
#if defined(HIP_ENABLED) || (defined(GPU_CODE) && HIP_CODE)
#include <hip/hip_runtime.h>

namespace simbi::adapter {

    // error handling for HIP
    namespace hip_error {
        inline error::status_t check_hip_error(hipError_t code)
        {
            if (code != hipSuccess) {
                return error::status_t::error;
            }
            return error::status_t::success;
        }

        inline std::string get_hip_error_string(hipError_t code)
        {
            return hipGetErrorString(code);
        }
    }   // namespace hip_error

    // HIP backend specialization
    template <>
    class DeviceBackend<hip_backend_tag>
    {
      public:
        // memory operations
        void copy_host_to_device(void* to, const void* from, std::size_t bytes)
        {
            hipError_t status =
                hipMemcpy(to, from, bytes, hipMemcpyHostToDevice);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP copy from host to device failed"
            );
        }

        void copy_device_to_host(void* to, const void* from, std::size_t bytes)
        {
            hipError_t status =
                hipMemcpy(to, from, bytes, hipMemcpyDeviceToHost);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP copy from device to host failed"
            );
        }

        void
        copy_device_to_device(void* to, const void* from, std::size_t bytes)
        {
            hipError_t status =
                hipMemcpy(to, from, bytes, hipMemcpyDeviceToDevice);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP copy from device to device failed"
            );
        }

        void malloc(void** obj, std::size_t bytes)
        {
            hipError_t status = hipMalloc(obj, bytes);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP malloc failed"
            );
        }

        void malloc_managed(void** obj, std::size_t bytes)
        {
            hipError_t status = hipMallocManaged(obj, bytes);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP malloc managed failed"
            );
        }

        void free(void* obj)
        {
            hipError_t status = hipFree(obj);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP free failed"
            );
        }

        void memset(void* obj, int val, std::size_t bytes)
        {
            hipError_t status = hipMemset(obj, val, bytes);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP memset failed"
            );
        }

        // event handling
        void event_create(void** event)
        {
            hipEvent_t* hip_event = static_cast<hipEvent_t*>(event);
            hipError_t status     = hipEventCreate(hip_event);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP event creation failed"
            );
        }

        void event_destroy(void* event)
        {
            hipEvent_t hip_event = *static_cast<hipEvent_t*>(event);
            hipError_t status    = hipEventDestroy(hip_event);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP event destruction failed"
            );
        }

        void event_record(void* event)
        {
            hipEvent_t hip_event = *static_cast<hipEvent_t*>(event);
            hipError_t status    = hipEventRecord(hip_event);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP event recording failed"
            );
        }

        void event_synchronize(void* event)
        {
            hipEvent_t hip_event = *static_cast<hipEvent_t*>(event);
            hipError_t status    = hipEventSynchronize(hip_event);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP event synchronization failed"
            );
        }

        void event_elapsed_time(float* time, void* start, void* end)
        {
            hipEvent_t start_event = *static_cast<hipEvent_t*>(start);
            hipEvent_t end_event   = *static_cast<hipEvent_t*>(end);

            hipError_t status =
                hipEventElapsedTime(time, start_event, end_event);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP event elapsed time calculation failed"
            );
        }

        // device management
        void get_device_count(int* count)
        {
            hipError_t status = hipGetDeviceCount(count);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP get device count failed"
            );
        }

        void get_device_properties(void* props, int device)
        {
            hipDeviceProp_t* hip_props = static_cast<hipDeviceProp_t*>(props);
            hipError_t status = hipGetDeviceProperties(hip_props, device);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP get device properties failed"
            );
        }

        void set_device(int device)
        {
            hipError_t status = hipSetDevice(device);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP set device failed"
            );
        }

        void device_synchronize()
        {
            hipError_t status = hipDeviceSynchronize();
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP device synchronization failed"
            );
        }

        // stream operations
        void stream_create(void** stream)
        {
            hipStream_t* hip_stream = static_cast<hipStream_t*>(stream);
            hipError_t status       = hipStreamCreate(hip_stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP stream creation failed"
            );
        }

        void stream_destroy(void* stream)
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipError_t status      = hipStreamDestroy(hip_stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP stream destruction failed"
            );
        }

        void stream_synchronize(void* stream)
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipError_t status      = hipStreamSynchronize(hip_stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP stream synchronization failed"
            );
        }

        void
        stream_wait_event(void* stream, void* event, unsigned int flags = 0)
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipEvent_t hip_event   = *static_cast<hipEvent_t*>(event);

            hipError_t status =
                hipStreamWaitEvent(hip_stream, hip_event, flags);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP stream wait event failed"
            );
        }

        void stream_query(void* stream, int* status)
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipError_t hip_status  = hipStreamQuery(hip_stream);

            // convert HIP status to int
            *status = static_cast<int>(hip_status);
        }

        // asynchronous operations
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipError_t status      = hipMemcpyAsync(
                to,
                from,
                bytes,
                hipMemcpyHostToDevice,
                hip_stream
            );

            error::check_err(
                hip_error::check_hip_error(status),
                "HIP async copy host to device failed"
            );
        }

        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipError_t status      = hipMemcpyAsync(
                to,
                from,
                bytes,
                hipMemcpyDeviceToHost,
                hip_stream
            );

            error::check_err(
                hip_error::check_hip_error(status),
                "HIP async copy device to host failed"
            );
        }

        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipError_t status      = hipMemcpyAsync(
                to,
                from,
                bytes,
                hipMemcpyDeviceToDevice,
                hip_stream
            );

            error::check_err(
                hip_error::check_hip_error(status),
                "HIP async copy device to device failed"
            );
        }

        void memcpy_async(
            void* to,
            const void* from,
            std::size_t bytes,
            int kind,
            void* stream
        )
        {
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            hipMemcpyKind hip_kind = static_cast<hipMemcpyKind>(kind);

            hipError_t status =
                hipMemcpyAsync(to, from, bytes, hip_kind, hip_stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP memcpy async failed"
            );
        }

        // peer operations
        void enable_peer_access(int device, unsigned int flags = 0)
        {
            hipError_t status = hipDeviceEnablePeerAccess(device, flags);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP enable peer access failed"
            );
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
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);

            hipError_t status = hipMemcpyPeerAsync(
                dst,
                dst_device,
                src,
                src_device,
                bytes,
                hip_stream
            );
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP peer copy async failed"
            );
        }

        // host memory management
        void host_register(void* ptr, std::size_t size, unsigned int flags)
        {
            hipError_t status = hipHostRegister(ptr, size, flags);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP host register failed"
            );
        }

        void host_unregister(void* ptr)
        {
            hipError_t status = hipHostUnregister(ptr);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP host unregister failed"
            );
        }

        void aligned_malloc(void** ptr, std::size_t size)
        {
            hipError_t status = hipHostMalloc(ptr, size);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP aligned malloc failed"
            );
        }

        // specialized operations
        void
        memcpy_from_symbol(void* dst, const void* symbol, std::size_t count)
        {
            hipError_t status = hipMemcpyFromSymbol(dst, symbol, count);
            error::check_err(
                hip_error::check_hip_error(status),
                "HIP memcpy from symbol failed"
            );
        }

        void
        prefetch_to_device(const void* obj, std::size_t bytes, int device = 0)
        {
            hipError_t status = hipMemPrefetchAsync(obj, bytes, device);
            error::check_err(
                hip_error::check_cuda_error(status),
                "HIP prefetch to device failed"
            );
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
            hipStream_t hip_stream = *static_cast<hipStream_t*>(&stream);
            void* hip_function     = static_cast<void*>(function);

            // convert our dim3 to HIP dim3
            dim3 hip_grid(grid.x, grid.y, grid.z);
            dim3 hip_block(block.x, block.y, block.z);

            hipError_t status = hipLaunchKernel(
                hip_function,
                hip_grid,
                hip_block,
                args,
                shared_mem,
                hip_stream
            );

            error::check_err(
                hip_error::check_hip_error(status),
                "HIP launch kernel failed"
            );
        }

        // atomic operations
        template <typename T>
        __device__ T atomic_min(T* address, T val)
        {
            if constexpr (std::is_floating_point_v<T>) {
                // Use appropriate atomic implementation for HIP
                using IntType = typename std::conditional<
                    sizeof(T) == 4,
                    unsigned int,
                    unsigned long long>::type;
                IntType* address_as_int = reinterpret_cast<IntType*>(address);
                IntType old_val         = *address_as_int;
                IntType assumed;

                do {
                    assumed      = old_val;
                    T value_as_t = *reinterpret_cast<T*>(&assumed);
                    T min_val    = (val < value_as_t) ? val : value_as_t;
                    IntType min_val_as_int =
                        *reinterpret_cast<IntType*>(&min_val);

                    old_val =
                        atomicCAS(address_as_int, assumed, min_val_as_int);
                } while (assumed != old_val);

                return *reinterpret_cast<T*>(&old_val);
            }
            else {
                return atomicMin(address, val);
            }
        }

        template <typename T>
        __device__ T atomic_add(T* address, T val)
        {
            if constexpr (std::is_floating_point_v<T>) {
                return atomicAdd(address, val);
            }
            else {
                // tnteger types
                return atomicAdd(address, val);
            }
        }

        // thread synchronization
        __device__ void synchronize_threads() { __syncthreads(); }
    };

}   // namespace simbi::adapter

#else   // HIP not enabled

namespace simbi::adapter {

    // stub implementation when HIP is not available
    template <>
    class DeviceBackend<hip_backend_tag>
    {
      public:
        // all methods throw an error since HIP is not available
        template <typename... Args>
        void not_supported(const std::string& function_name, Args&&...)
        {
            throw error::runtime_error(
                error::status_t::error,
                "HIP backend not available: " + function_name + " called"
            );
        }

        // memory operations
        void copy_host_to_device(void* to, const void* from, std::size_t bytes)
        {
            not_supported("copy_host_to_device");
        }

        void copy_device_to_host(void* to, const void* from, std::size_t bytes)
        {
            not_supported("copy_device_to_host");
        }

        void
        copy_device_to_device(void* to, const void* from, std::size_t bytes)
        {
            not_supported("copy_device_to_device");
        }

        void malloc(void** obj, std::size_t bytes) { not_supported("malloc"); }

        void malloc_managed(void** obj, std::size_t bytes)
        {
            not_supported("malloc_managed");
        }

        void free(void* obj) { not_supported("free"); }

        void memset(void* obj, int val, std::size_t bytes)
        {
            not_supported("memset");
        }

        // event handling
        void event_create(void** event) { not_supported("event_create"); }

        void event_destroy(void* event) { not_supported("event_destroy"); }

        void event_record(void* event) { not_supported("event_record"); }

        void event_synchronize(void* event)
        {
            not_supported("event_synchronize");
        }

        void event_elapsed_time(float* time, void* start, void* end)
        {
            not_supported("event_elapsed_time");
        }

        // device management
        void get_device_count(int* count) { not_supported("get_device_count"); }

        void get_device_properties(void* props, int device)
        {
            not_supported("get_device_properties");
        }

        void set_device(int device) { not_supported("set_device"); }

        void device_synchronize() { not_supported("device_synchronize"); }

        // stream operations
        void stream_create(void** stream) { not_supported("stream_create"); }

        void stream_destroy(void* stream) { not_supported("stream_destroy"); }

        void stream_synchronize(void* stream)
        {
            not_supported("stream_synchronize");
        }

        void
        stream_wait_event(void* stream, void* event, unsigned int flags = 0)
        {
            not_supported("stream_wait_event");
        }

        void stream_query(void* stream, int* status)
        {
            not_supported("stream_query");
        }

        // asynchronous operations
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            not_supported("async_copy_host_to_device");
        }

        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            not_supported("async_copy_device_to_host");
        }

        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        )
        {
            not_supported("async_copy_device_to_device");
        }

        void memcpy_async(
            void* to,
            const void* from,
            std::size_t bytes,
            int kind,
            void* stream
        )
        {
            not_supported("memcpy_async");
        }

        // peer operations
        void enable_peer_access(int device, unsigned int flags = 0)
        {
            not_supported("enable_peer_access");
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
            not_supported("peer_copy_async");
        }

        // host memory management
        void host_register(void* ptr, std::size_t size, unsigned int flags)
        {
            not_supported("host_register");
        }

        void host_unregister(void* ptr) { not_supported("host_unregister"); }

        void aligned_malloc(void** ptr, std::size_t size)
        {
            not_supported("aligned_malloc");
        }

        // specialized operations
        void
        memcpy_from_symbol(void* dst, const void* symbol, std::size_t count)
        {
            not_supported("memcpy_from_symbol");
        }

        void
        prefetch_to_device(const void* obj, std::size_t bytes, int device = 0)
        {
            not_supported("prefetch_to_device");
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
            not_supported("launch_kernel");
        }

        // atomic operations
        template <typename T>
        T atomic_min(T* address, T val)
        {
            not_supported("atomic_min");
            return *address;
        }

        template <typename T>
        T atomic_add(T* address, T val)
        {
            not_supported("atomic_add");
            return *address;
        }

        // thread synchronization
        void synchronize_threads() { not_supported("synchronize_threads"); }
    };

}   // namespace simbi::adapter

#endif   // HIP enabled or not

#endif
