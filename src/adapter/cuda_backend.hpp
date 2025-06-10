/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            cuda_backend.hpp
 * @brief           CUDA implementation of device backend
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
 * @depends         CUDA >= 11.0
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA)
 *
 *==============================================================================
 */

#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP

#include "device_backend.hpp"
#include "device_types.hpp"

// only include CUDA headers if CUDA is enabled
#if defined(CUDA_ENABLED) || (defined(GPU_CODE) && CUDA_CODE)
#include <cuda.h>
#include <cuda_runtime.h>

namespace simbi::adapter {

    // error handling for CUDA
    namespace cuda_error {
        inline error::status_t check_cuda_error(cudaError_t code)
        {
            if (code != cudaSuccess) {
                return error::status_t::error;
            }
            return error::status_t::success;
        }

        inline std::string get_cuda_error_string(cudaError_t code)
        {
            return cudaGetErrorString(code);
        }
    }   // namespace cuda_error

    // CUDA backend specialization
    template <>
    class DeviceBackend<cuda_backend_tag>
    {
      public:
        // Type aliases for this backend
        using event_t             = event_t<cuda_backend_tag>;
        using stream_t            = stream_t<cuda_backend_tag>;
        using device_properties_t = device_properties_t<cuda_backend_tag>;
        using function_t          = function_t<cuda_backend_tag>;
        using memcpy_kind_t       = memcpy_kind_t<cuda_backend_tag>;

        // Memory operations
        void copy_host_to_device(void* to, const void* from, std::size_t bytes)
        {
            cudaError_t status =
                cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA copy from host to device failed"
            );
        }

        void copy_device_to_host(void* to, const void* from, std::size_t bytes)
        {
            cudaError_t status =
                cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToHost);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA copy from device to host failed"
            );
        }

        void
        copy_device_to_device(void* to, const void* from, std::size_t bytes)
        {
            cudaError_t status =
                cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA copy from device to device failed"
            );
        }

        void malloc(void** obj, std::size_t bytes)
        {
            cudaError_t status = cudaMalloc(obj, bytes);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA malloc failed"
            );
        }

        void malloc_managed(void** obj, std::size_t bytes)
        {
            cudaError_t status = cudaMallocManaged(obj, bytes);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA malloc managed failed"
            );
        }

        void free(void* obj)
        {
            cudaError_t status = cudaFree(obj);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA free failed"
            );
        }

        void memset(void* obj, int val, std::size_t bytes)
        {
            cudaError_t status = cudaMemset(obj, val, bytes);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA memset failed"
            );
        }

        // Event handling
        void event_create(event_t* event)
        {
            cudaError_t status = cudaEventCreate(event);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA event creation failed"
            );
        }

        void event_destroy(event_t event)
        {
            cudaError_t status = cudaEventDestroy(event);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA event destruction failed"
            );
        }

        void event_record(event_t event)
        {
            cudaError_t status = cudaEventRecord(event);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA event recording failed"
            );
        }

        void event_synchronize(event_t event)
        {
            cudaError_t status = cudaEventSynchronize(event);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA event synchronization failed"
            );
        }

        void event_elapsed_time(float* time, event_t start, event_t end)
        {
            cudaError_t status = cudaEventElapsedTime(time, start, end);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA event elapsed time calculation failed"
            );
        }

        // Device management
        void get_device_count(int* count)
        {
            cudaError_t status = cudaGetDeviceCount(count);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA get device count failed"
            );
        }

        void get_device_properties(device_properties_t* props, int device)
        {
            cudaError_t status = cudaGetDeviceProperties(props, device);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA get device properties failed"
            );
        }

        void set_device(int device)
        {
            cudaError_t status = cudaSetDevice(device);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA set device failed"
            );
        }

        void device_synchronize()
        {
            cudaError_t status = cudaDeviceSynchronize();
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA device synchronization failed"
            );
        }

        // Stream operations
        void stream_create(stream_t* stream)
        {
            cudaError_t status = cudaStreamCreate(stream);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA stream creation failed"
            );
        }

        void stream_destroy(stream_t stream)
        {
            cudaError_t status = cudaStreamDestroy(stream);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA stream destruction failed"
            );
        }

        void stream_synchronize(stream_t stream)
        {
            cudaError_t status = cudaStreamSynchronize(stream);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA stream synchronization failed"
            );
        }

        void stream_wait_event(
            stream_t stream,
            event_t event,
            unsigned int flags = 0
        )
        {
            cudaError_t status = cudaStreamWaitEvent(stream, event, flags);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA stream wait event failed"
            );
        }

        void stream_query(stream_t stream, int* status)
        {
            cudaError_t cuda_status = cudaStreamQuery(stream);
            *status                 = static_cast<int>(cuda_status);
        }

        // asynchronous operations
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            stream_t stream
        )
        {
            cudaError_t status = cudaMemcpyAsync(
                to,
                from,
                bytes,
                cudaMemcpyHostToDevice,
                stream
            );

            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA async copy host to device failed"
            );
        }

        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            stream_t stream
        )
        {
            cudaError_t status = cudaMemcpyAsync(
                to,
                from,
                bytes,
                cudaMemcpyDeviceToHost,
                stream
            );

            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA async copy device to host failed"
            );
        }

        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            stream_t stream
        )
        {
            cudaError_t status = cudaMemcpyAsync(
                to,
                from,
                bytes,
                cudaMemcpyDeviceToDevice,
                stream
            );

            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA async copy device to device failed"
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
            cudaError_t status = cudaMemcpyAsync(to, from, bytes, kind, stream);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA memcpy async failed"
            );
        }

        // peer operations
        void enable_peer_access(int device, unsigned int flags = 0)
        {
            cudaError_t status = cudaDeviceEnablePeerAccess(device, flags);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA enable peer access failed"
            );
        }

        void peer_copy_async(
            void* dst,
            int dst_device,
            const void* src,
            int src_device,
            std::size_t bytes,
            stream_t stream
        )
        {
            cudaError_t status = cudaMemcpyPeerAsync(
                dst,
                dst_device,
                src,
                src_device,
                bytes,
                stream
            );
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA peer copy async failed"
            );
        }

        // host memory management
        void host_register(void* ptr, std::size_t size, unsigned int flags)
        {
            cudaError_t status = cudaHostRegister(ptr, size, flags);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA host register failed"
            );
        }

        void host_unregister(void* ptr)
        {
            cudaError_t status = cudaHostUnregister(ptr);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA host unregister failed"
            );
        }

        void aligned_malloc(void** ptr, std::size_t size)
        {
            cudaError_t status = cudaMallocHost(ptr, size);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA aligned malloc failed"
            );
        }

        // specialized operations
        void
        memcpy_from_symbol(void* dst, const void* symbol, std::size_t count)
        {
            cudaError_t status = cudaMemcpyFromSymbol(dst, symbol, count);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA memcpy from symbol failed"
            );
        }

        void
        prefetch_to_device(const void* obj, std::size_t bytes, int device = 0)
        {
            cudaError_t status = cudaMemPrefetchAsync(obj, bytes, device);
            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA prefetch to device failed"
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
            // convert our dim3 to CUDA dim3
            dim3 cuda_grid(grid.x, grid.y, grid.z);
            dim3 cuda_block(block.x, block.y, block.z);

            cudaError_t status = cudaLaunchKernel(
                static_cast<const void*>(function),
                cuda_grid,
                cuda_block,
                args,
                shared_mem,
                stream
            );

            error::check_err(
                cuda_error::check_cuda_error(status),
                "CUDA launch kernel failed"
            );
        }

        // atomic operations
        template <typename T>
        __device__ T atomic_min(T* address, T val)
        {
            if constexpr (std::is_floating_point_v<T>) {
                // use type punning for floating point atomics
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
                if constexpr (sizeof(T) == 4) {   // float
                    return atomicAdd(address, val);
                }
                else {   // double
                    unsigned long long int* address_as_ull =
                        reinterpret_cast<unsigned long long int*>(address);
                    unsigned long long int old = *address_as_ull;
                    unsigned long long int assumed;

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
            }
        }

        // Thread synchronization
        __device__ void synchronize_threads() { __syncthreads(); }
    };

}   // namespace simbi::adapter

#else   // CUDA not enabled

namespace simbi::adapter {

    // stub implementation when CUDA is not available
    template <>
    class DeviceBackend<cuda_backend_tag>
    {
      public:   // Helper to throw not supported error
        template <typename... Args>
        void not_supported(const std::string& function_name, Args&&...)
        {
            throw error::runtime_error(
                error::status_t::error,
                "CUDA backend not available: " + function_name + " called"
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

        void memset(void* /*obj*/, int /*val*/, std::size_t /*bytes*/)
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

        void event_record(event_t<> /*event*/)
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

        void
        get_device_properties(device_properties_t<>* /*props*/, int /*device*/)
        {
            not_supported("get_device_properties");
        }

        void set_device(int /*device*/) { not_supported("set_device"); }

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
            unsigned int /*flags*/ = 0
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
        void enable_peer_access(int /*device*/, unsigned int /*flags*/ = 0)
        {
            not_supported("enable_peer_access");
        }

        void peer_copy_async(
            void* /*dst*/,
            int /*dst_device*/,
            const void* /*src*/,
            int /*src_device*/,
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
            unsigned int /*flags*/
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
            int /*device*/ = 0
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

#endif   // CUDA enabled or not

#endif
