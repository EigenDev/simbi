/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            hip_backend.hpp
 * @brief           cuda implementation of device backend
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
 * @depends         cuda >= 11.0
 * @platform        Linux, MacOS
 * @parallel        GPU (cuda)
 *
 *==============================================================================
 */

#ifndef cuda_BACKEND_HPP
#define cuda_BACKEND_HPP

#include "device_backend.hpp"
#include "device_types.hpp"

// only include cuda headers if cuda is enabled
#if defined(cuda_ENABLED) || (defined(GPU_ENABLED) && cuda_CODE)
#include <hip/hip_runtime.h>

namespace simbi::adapter {

    // error handling for cuda
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

    // cuda backend specialization
    template <>
    class DeviceBackend<hip_backend_tag>
    {
      public:
        // Type aliases for this backend
        using event_t             = event_t<hip_backend_tag>;
        using stream_t            = stream_t<hip_backend_tag>;
        using device_properties_t = device_properties_t<hip_backend_tag>;
        using function_t          = function_t<hip_backend_tag>;
        using memcpy_kind_t       = memcpy_kind_t<hip_backend_tag>;

        // Memory operations
        void copy_host_to_device(void* to, const void* from, std::size_t bytes)
        {
            hipError_t status =
                hipMemcpy(to, from, bytes, hipMemcpyHostToDevice);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda copy from host to device failed"
            );
        }

        void copy_device_to_host(void* to, const void* from, std::size_t bytes)
        {
            hipError_t status =
                hipMemcpy(to, from, bytes, hipMemcpyDeviceToHost);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda copy from device to host failed"
            );
        }

        void
        copy_device_to_device(void* to, const void* from, std::size_t bytes)
        {
            hipError_t status =
                hipMemcpy(to, from, bytes, hipMemcpyDeviceToDevice);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda copy from device to device failed"
            );
        }

        void malloc(void** obj, std::size_t bytes)
        {
            hipError_t status = hipMalloc(obj, bytes);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda malloc failed"
            );
        }

        void malloc_managed(void** obj, std::size_t bytes)
        {
            hipError_t status = hipMallocManaged(obj, bytes);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda malloc managed failed"
            );
        }

        void free(void* obj)
        {
            hipError_t status = hipFree(obj);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda free failed"
            );
        }

        void memset(void* obj, std::int64_t val, std::size_t bytes)
        {
            hipError_t status = hipMemset(obj, val, bytes);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda memset failed"
            );
        }

        // Event handling
        void event_create(event_t* event)
        {
            hipError_t status = hipEventCreate(event);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda event creation failed"
            );
        }

        void event_destroy(event_t event)
        {
            hipError_t status = hipEventDestroy(event);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda event destruction failed"
            );
        }

        void event_record(event_t event, stream_t stream = 0)
        {
            hipError_t status = hipEventRecord(event, stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda event recording failed"
            );
        }

        void event_synchronize(event_t event)
        {
            hipError_t status = hipEventSynchronize(event);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda event synchronization failed"
            );
        }

        void event_elapsed_time(float* time, event_t start, event_t end)
        {
            hipError_t status = hipEventElapsedTime(time, start, end);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda event elapsed time calculation failed"
            );
        }

        // Device management
        void get_device_count(int* count)
        {
            hipError_t status = hipGetDeviceCount(count);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda get device count failed"
            );
        }

        void
        get_device_properties(device_properties_t* props, std::int64_t device)
        {
            hipError_t status = hipGetDeviceProperties(props, device);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda get device properties failed"
            );
        }

        void set_device(std::int64_t device)
        {
            hipError_t status = hipSetDevice(device);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda set device failed"
            );
        }

        void device_synchronize()
        {
            hipError_t status = hipDeviceSynchronize();
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda device synchronization failed"
            );
        }

        // Stream operations
        void stream_create(stream_t* stream)
        {
            hipError_t status = hipStreamCreate(stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda stream creation failed"
            );
        }

        void stream_destroy(stream_t stream)
        {
            hipError_t status = hipStreamDestroy(stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda stream destruction failed"
            );
        }

        void stream_synchronize(stream_t stream)
        {
            hipError_t status = hipStreamSynchronize(stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda stream synchronization failed"
            );
        }

        void stream_wait_event(
            stream_t stream,
            event_t event,
            std::uint64_t flags = 0
        )
        {
            hipError_t status = hipStreamWaitEvent(stream, event, flags);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda stream wait event failed"
            );
        }

        void stream_query(stream_t stream, int* status)
        {
            hipError_t hip_status = hipStreamQuery(stream);
            *status               = static_cast<std::int64_t>(hip_status);
        }

        // asynchronous operations
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            stream_t stream
        )
        {
            hipError_t status =
                hipMemcpyAsync(to, from, bytes, hipMemcpyHostToDevice, stream);

            error::check_err(
                hip_error::check_hip_error(status),
                "cuda async copy host to device failed"
            );
        }

        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            stream_t stream
        )
        {
            hipError_t status =
                hipMemcpyAsync(to, from, bytes, hipMemcpyDeviceToHost, stream);

            error::check_err(
                hip_error::check_hip_error(status),
                "cuda async copy device to host failed"
            );
        }

        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            stream_t stream
        )
        {
            hipError_t status = hipMemcpyAsync(
                to,
                from,
                bytes,
                hipMemcpyDeviceToDevice,
                stream
            );

            error::check_err(
                hip_error::check_hip_error(status),
                "cuda async copy device to device failed"
            );
        }

        void memcpy_async(
            void* to,
            const void* from,
            std::size_t bytes,
            memcpy_kind_t kind,
            stream_t stream
        )
        {
            hipError_t status = hipMemcpyAsync(to, from, bytes, kind, stream);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda memcpy async failed"
            );
        }

        // peer operations
        void enable_peer_access(std::int64_t device, std::uint64_t flags = 0)
        {
            hipError_t status = hipDeviceEnablePeerAccess(device, flags);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda enable peer access failed"
            );
        }

        void peer_copy_async(
            void* dst,
            std::int64_t dst_device,
            const void* src,
            std::int64_t src_device,
            std::size_t bytes,
            stream_t stream
        )
        {
            hipError_t status = hipMemcpyPeerAsync(
                dst,
                dst_device,
                src,
                src_device,
                bytes,
                stream
            );
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda peer copy async failed"
            );
        }

        // host memory management
        void host_register(void* ptr, std::size_t size, std::uint64_t flags)
        {
            hipError_t status = hipHostRegister(ptr, size, flags);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda host register failed"
            );
        }

        void host_unregister(void* ptr)
        {
            hipError_t status = hipHostUnregister(ptr);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda host unregister failed"
            );
        }

        void aligned_malloc(void** ptr, std::size_t size)
        {
            hipError_t status = hipMallocHost(ptr, size);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda aligned malloc failed"
            );
        }

        // specialized operations
        void
        memcpy_from_symbol(void* dst, const void* symbol, std::size_t count)
        {
            hipError_t status = hipMemcpyFromSymbol(dst, symbol, count);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda memcpy from symbol failed"
            );
        }

        void prefetch_to_device(
            const void* obj,
            std::size_t bytes,
            std::int64_t device = 0
        )
        {
            hipError_t status = hipMemPrefetchAsync(obj, bytes, device);
            error::check_err(
                hip_error::check_hip_error(status),
                "cuda prefetch to device failed"
            );
        }

        void launch_kernel(
            function_t function,
            types::dim3 grid,
            types::dim3 block,
            void** args,
            std::size_t shared_mem,
            stream_t stream
        )
        {
            // convert our dim3 to cuda dim3
            dim3 hip_grid(grid.x, grid.y, grid.z);
            dim3 hip_block(block.x, block.y, block.z);

            hipError_t status = hipLaunchKernel(
                static_cast<const void*>(function),
                hip_grid,
                hip_block,
                args,
                shared_mem,
                stream
            );

            error::check_err(
                hip_error::check_hip_error(status),
                "cuda launch kernel failed"
            );
        }

        // atomic operations
        template <typename T>
        __device__ T atomic_min(T* address, T val)
        {
            if constexpr (std::is_floating_point_v<T>) {
                // use type punning for floating postd::int64_t atomics
                using IntType = typename std::conditional<
                    sizeof(T) == 4,
                    std::uint64_t,
                    std::uint64_t>::type;
                IntType* address_as_std::int64_t =
                    reinterpret_cast<IntType*>(address);
                IntType old_val = *address_as_int;
                IntType assumed;

                do {
                    assumed      = old_val;
                    T value_as_t = *reinterpret_cast<T*>(&assumed);
                    T min_val    = (val < value_as_t) ? val : value_as_t;
                    IntType min_val_as_std::int64_t =
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
                if constexpr (sizeof(T) == 4) {   // float
                    return atomicAdd(address, val);
                }
                else {   // double
                    std::uint64_t int* address_as_ull =
                        reinterpret_cast<std::uint64_t int*>(address);
                    std::uint64_t old = *address_as_ull;
                    std::uint64_t assumed;

                    do {
                        assumed = old;
                        old     = atomicCAS(
                            address_as_ull,
                            assumed,
                            __double_as_longlong(
                                val + __longlong_as_double(assumed)
                            )
                        );

                    } while (assumed != old);
                    return __longlong_as_double(old);
                }
            }
            else {
                return atomicAdd(address, val);
            }
        }

        // Thread synchronization
        __device__ void synchronize_threads() { __syncthreads(); }
    };

}   // namespace simbi::adapter

#else   // cuda not enabled

namespace simbi::adapter {

    // stub implementation when cuda is not available
    template <>
    class DeviceBackend<hip_backend_tag>
    {
      public:   // Helper to throw not supported error
        template <typename... Args>
        void not_supported(const std::string& function_name, Args&&...)
        {
            throw error::runtime_error(
                error::status_t::error,
                "cuda backend not available: " + function_name + " called"
            );
        }

        // Memory operations
        void copy_host_to_device(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/
        )
        {
            not_supported("copy_host_to_device");
        }

        void copy_device_to_host(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/
        )
        {
            not_supported("copy_device_to_host");
        }

        void copy_device_to_device(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/
        )
        {
            not_supported("copy_device_to_device");
        }

        void malloc(void** /*obj*/, std::size_t /*bytes*/)
        {
            not_supported("malloc");
        }

        void malloc_managed(void** /*obj*/, std::size_t /*bytes*/)
        {
            not_supported("malloc_managed");
        }

        void free(void* /*obj*/) { not_supported("free"); }

        void memset(void* /*obj*/, std::int64_t /*val*/, std::size_t /*bytes*/)
        {
            not_supported("memset");
        }

        // Event handling
        void event_create(event_t<>* /*event*/)
        {
            not_supported("event_create");
        }

        void event_destroy(event_t<> /*event*/)
        {
            not_supported("event_destroy");
        }

        void event_record(event_t<> /*event*/, stream_t<> /*stream*/)
        {
            not_supported("event_record");
        }

        void event_synchronize(event_t<> /*event*/)
        {
            not_supported("event_synchronize");
        }

        void event_elapsed_time(
            float* /*time*/,
            event_t<> /*start*/,
            event_t<> /*end*/
        )
        {
            not_supported("event_elapsed_time");
        }

        // Device management
        void get_device_count(int* /*count*/)
        {
            not_supported("get_device_count");
        }

        void get_device_properties(
            device_properties_t<>* /*props*/,
            std::int64_t /*device*/
        )
        {
            not_supported("get_device_properties");
        }

        void set_device(std::int64_t /*device*/)
        {
            not_supported("set_device");
        }

        void device_synchronize() { not_supported("device_synchronize"); }

        // Stream operations
        void stream_create(stream_t<>* /*stream*/)
        {
            not_supported("stream_create");
        }

        void stream_destroy(stream_t<> /*stream*/)
        {
            not_supported("stream_destroy");
        }

        void stream_synchronize(stream_t<> /*stream*/)
        {
            not_supported("stream_synchronize");
        }

        void stream_wait_event(
            stream_t<> /*stream*/,
            event_t<> /*event*/,
            std::uint64_t /*flags*/ = 0
        )
        {
            not_supported("stream_wait_event");
        }

        void stream_query(stream_t<> /*stream*/, int* /*status*/)
        {
            not_supported("stream_query");
        }

        // Asynchronous operations
        void async_copy_host_to_device(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/,
            stream_t<> /*stream*/
        )
        {
            not_supported("async_copy_host_to_device");
        }

        void async_copy_device_to_host(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/,
            stream_t<> /*stream*/
        )
        {
            not_supported("async_copy_device_to_host");
        }

        void async_copy_device_to_device(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/,
            stream_t<> /*stream*/
        )
        {
            not_supported("async_copy_device_to_device");
        }

        void memcpy_async(
            void* /*to*/,
            const void* /*from*/,
            std::size_t /*bytes*/,
            memcpy_kind_t<> /*kind*/,
            stream_t<> /*stream*/
        )
        {
            not_supported("memcpy_async");
        }

        // Peer operations
        void
        enable_peer_access(std::int64_t /*device*/, std::uint64_t /*flags*/ = 0)
        {
            not_supported("enable_peer_access");
        }

        void peer_copy_async(
            void* /*dst*/,
            std::int64_t /*dst_device*/,
            const void* /*src*/,
            std::int64_t /*src_device*/,
            std::size_t /*bytes*/,
            stream_t<> /*stream*/
        )
        {
            not_supported("peer_copy_async");
        }

        // Host memory management
        void host_register(
            void* /*ptr*/,
            std::size_t /*size*/,
            std::uint64_t /*flags*/
        )
        {
            not_supported("host_register");
        }

        void host_unregister(void* /*ptr*/)
        {
            not_supported("host_unregister");
        }

        void aligned_malloc(void** /*ptr*/, std::size_t /*size*/)
        {
            not_supported("aligned_malloc");
        }

        // Specialized operations
        void memcpy_from_symbol(
            void* /*dst*/,
            const void* /*symbol*/,
            std::size_t /*count*/
        )
        {
            not_supported("memcpy_from_symbol");
        }

        void prefetch_to_device(
            const void* /*obj*/,
            std::size_t /*bytes*/,
            std::int64_t /*device*/ = 0
        )
        {
            not_supported("prefetch_to_device");
        }

        void launch_kernel(
            function_t<> /*function*/,
            types::dim3 /*grid*/,
            types::dim3 /*block*/,
            void** /*args*/,
            std::size_t /*shared_mem*/,
            stream_t<> /*stream*/
        )
        {
            not_supported("launch_kernel");
        }

        // Atomic operations
        template <typename T>
        T atomic_min(T* address, T /*val*/)
        {
            not_supported("atomic_min");
            return *address;   // Unreachable, but needed for compilation
        }

        template <typename T>
        T atomic_add(T* address, T /*val*/)
        {
            not_supported("atomic_add");
            return *address;   // Unreachable, but needed for compilation
        }

        // Thread synchronization
        void synchronize_threads() { not_supported("synchronize_threads"); }
    };

}   // namespace simbi::adapter

#endif   // cuda enabled or not

#endif
