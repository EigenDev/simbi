// =============================================================================
// cuda_space.hpp
//
// cuda execution space implementation for heterogeneous computing.
// provides clean abstraction over cuda vendor device while exposing
// raw cuda types for test compatibility and performance.
//
// design principles:
//   - vendor device internally: uses cuda_device_t for implementation
//   - raw types publicly: exposes cudaStream_t/cudaEvent_t for compatibility
//   - concept compliant: satisfies execution_space concept requirements
//   - zero overhead: compile-time dispatch, no virtual calls
//
// usage:
//   parallel_for<cuda_space>(range, kernel);
//   auto stream = cuda_space::create_stream();
//   cuda_space::synchronize_stream(stream);
// =============================================================================

#pragma once

#include "xpu/core/device_concepts.hpp"
#include "xpu/core/execution_concepts.hpp"
#include "xpu/vendors/cpu/cpu_device.hpp"
#include "xpu/vendors/cuda/cuda_device.hpp"

#include <cstddef>

#ifdef XPU_CUDA_AVAILABLE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string_view>
#endif

namespace simbi::xpu::exec {

#ifdef XPU_CUDA_AVAILABLE

    struct cuda_space
    {
        // =============================================================================
        // execution space type system
        // =============================================================================

        // vendor device for internal implementation
        using device_type = vendors::cuda::cuda_device_t;

        // memory space type for concept requirements
        using memory_space_type = void; // placeholder for future memory spaces

        // raw cuda types for test compatibility and performance
        using stream_handle_type = cudaStream_t;
        using event_handle_type  = cudaEvent_t;

        // =============================================================================
        // space identification
        // =============================================================================

        static constexpr std::string_view space_name()
        {
            return "cuda";
        }

        static constexpr std::string_view vendor_name()
        {
            return "nvidia";
        }

        static constexpr int default_device_id()
        {
            return 0;
        }

        // execution_space concept requirements
        static constexpr bool is_host_space    = false;
        static constexpr bool is_device_space  = true;
        static constexpr bool supports_async   = true;
        static constexpr bool supports_kernels = true;

        // test compatibility aliases
        static constexpr bool is_gpu  = true;
        static constexpr bool is_host = false;

        // device properties
        static constexpr bool supports_shared_memory   = true;
        static constexpr bool supports_async_execution = true;
        static constexpr bool supports_unified_memory  = true;

        // test compatibility functions
        static constexpr std::string_view name()
        {
            return space_name();
        }

        // =============================================================================
        // execution context
        // =============================================================================

        struct execution_context
        {
            stream_handle_type stream;
            int                device_id;
            int                block_size;

            execution_context() : stream(create_stream()), device_id(0), block_size(256) {}

            execution_context(stream_handle_type s, int id = 0)
                : stream(s), device_id(id), block_size(256)
            {
            }

            execution_context(int id, int bs)
                : stream(create_stream()), device_id(id), block_size(bs)
            {
            }

            execution_context(stream_handle_type s, int id, int bs)
                : stream(s), device_id(id), block_size(bs)
            {
            }

            // copy semantics - streams are shared handles, safe to copy
            execution_context(const execution_context& other)
                : stream(other.stream), device_id(other.device_id), block_size(other.block_size)
            {
            }

            execution_context& operator=(const execution_context& other)
            {
                if (this != &other) {
                    stream     = other.stream;
                    device_id  = other.device_id;
                    block_size = other.block_size;
                }
                return *this;
            }

            // move semantics - clear moved-from stream handle
            execution_context(execution_context&& other) noexcept
                : stream(other.stream), device_id(other.device_id), block_size(other.block_size)
            {
                other.stream = nullptr;
            }

            execution_context& operator=(execution_context&& other) noexcept
            {
                if (this != &other) {
                    stream       = other.stream;
                    device_id    = other.device_id;
                    block_size   = other.block_size;
                    other.stream = nullptr;
                }
                return *this;
            }

            ~execution_context() = default;
        };

        // =============================================================================
        // device management
        // =============================================================================

        static void initialize()
        {
            // cuda runtime initialization handled by vendor device
        }

        static void finalize()
        {
            // cleanup handled by vendor device destructors
        }

        static int device_count()
        {
            int count = 0;
            cudaGetDeviceCount(&count);
            return count;
        }

        static void set_device(std::int64_t device_id)
        {
            cudaSetDevice(static_cast<int>(device_id));
        }

        static int get_device()
        {
            int device_id = 0;
            cudaGetDevice(&device_id);
            return device_id;
        }

        // =============================================================================
        // memory management
        // =============================================================================

        static void* allocate(std::size_t bytes)
        {
            device_type device;
            auto        handle = device.allocate(bytes);
            return handle.ptr;
        }

        static void deallocate(void* ptr)
        {
            device_type                         device;
            vendors::cuda::cuda_memory_handle_t handle{ptr};
            device.deallocate(handle);
        }

        // =============================================================================
        // synchronization
        // =============================================================================

        static void synchronize()
        {
            cudaDeviceSynchronize();
        }

        static void fence()
        {
            cudaDeviceSynchronize();
        }

        // =============================================================================
        // stream management
        // =============================================================================

        static stream_handle_type create_stream()
        {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            return stream;
        }

        static void destroy_stream(stream_handle_type stream)
        {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }

        static void synchronize_stream(stream_handle_type stream)
        {
            if (stream != nullptr) {
                cudaStreamSynchronize(stream);
            }
        }

        static bool is_stream_ready(stream_handle_type stream)
        {
            if (stream == nullptr) {
                return true;
            }
            cudaError_t status = cudaStreamQuery(stream);
            return status == cudaSuccess;
        }

        // =============================================================================
        // event management
        // =============================================================================

        static event_handle_type create_event()
        {
            cudaEvent_t event;
            cudaEventCreate(&event);
            return event;
        }

        static void destroy_event(event_handle_type event)
        {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }

        static void record_event(event_handle_type event, stream_handle_type stream)
        {
            if (event != nullptr && stream != nullptr) {
                cudaEventRecord(event, stream);
            }
        }

        static event_handle_type record_event(stream_handle_type stream)
        {
            auto event = create_event();
            record_event(event, stream);
            return event;
        }

        static bool is_event_ready(event_handle_type event)
        {
            if (event == nullptr) {
                return true;
            }
            cudaError_t status = cudaEventQuery(event);
            return status == cudaSuccess;
        }

        static void wait_for_event(event_handle_type event)
        {
            if (event != nullptr) {
                cudaEventSynchronize(event);
            }
        }

        static void synchronize_event(event_handle_type event)
        {
            wait_for_event(event);
        }

        static void stream_wait_event(stream_handle_type stream, event_handle_type event)
        {
            if (stream != nullptr && event != nullptr) {
                cudaStreamWaitEvent(stream, event, 0);
            }
        }

        // =============================================================================
        // kernel execution (placeholder)
        // =============================================================================

        template <typename Kernel, typename... Args>
        static void launch_kernel(
            Kernel             kernel,
            dim3               grid_size,
            dim3               block_size,
            std::size_t        shared_mem,
            stream_handle_type stream,
            Args&&... args
        )
        {
            // placeholder for actual kernel launch
            // would require nvcc compilation and proper kernel dispatch
            (void) kernel;
            (void) grid_size;
            (void) block_size;
            (void) shared_mem;
            (void) stream;
            ((void) args, ...);
        }

        // =============================================================================
        // parallel execution (fallback implementations)
        // =============================================================================

        template <typename Index, typename Functor>
        static void parallel_for(Index first, Index last, Functor&& func)
        {
            // fallback: sequential execution
            for (Index ii = first; ii < last; ++ii) {
                func(ii);
            }
        }

        template <typename Index, typename Functor>
        static void
        parallel_for(Index first, Index last, Functor&& func, const execution_context& ctx)
        {
            (void) ctx;
            parallel_for(first, last, std::forward<Functor>(func));
        }

        template <typename Index, typename Functor, typename T>
        static T reduce(Index first, Index last, T init, Functor&& func)
        {
            // fallback: sequential reduction
            T result = init;
            for (Index ii = first; ii < last; ++ii) {
                result += func(ii);
            }
            return result;
        }

        // =============================================================================
        // error handling
        // =============================================================================

        static std::string get_error_string(int error_code)
        {
            return std::string(cudaGetErrorString(static_cast<cudaError_t>(error_code)));
        }

        static bool has_error()
        {
            return cudaGetLastError() != cudaSuccess;
        }

        static void clear_error()
        {
            cudaGetLastError(); // clears last error
        }

        // =============================================================================
        // device properties
        // =============================================================================

        static std::size_t available_memory()
        {
            device_type device;
            return device.available_memory();
        }

        static std::size_t total_memory()
        {
            device_type device;
            return device.total_memory();
        }

        static int compute_capability_major()
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, 0);
            return props.major;
        }

        static int compute_capability_minor()
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, 0);
            return props.minor;
        }

        // =============================================================================
        // memory space queries
        // =============================================================================

        template <typename MemorySpace>
        static constexpr bool is_accessible_from()
        {
            // cuda device memory accessible from cuda space, not from cpu space
            if constexpr (std::is_same_v<MemorySpace, cuda_space>) {
                return true;
            }
            else {
                return false;
            }
        }

        static constexpr bool can_access_host_memory()
        {
            return false; // cuda space cannot directly access host memory
        }

        static constexpr bool can_access_device_memory()
        {
            return true; // cuda space can access device memory
        }

        static constexpr bool can_access_unified_memory()
        {
            return true; // cuda supports unified memory
        }

        // =============================================================================
        // execution characteristics (execution_space concept)
        // =============================================================================

        static std::size_t max_concurrency()
        {
            device_type device;
            return device.compute_units() * device.max_threads_per_block();
        }

        static std::size_t preferred_block_size()
        {
            return 256;
        }

        static double memory_bandwidth_gb_per_sec()
        {
            device_type device;
            return device.memory_bandwidth_gb_per_sec();
        }
    };

#else // XPU_CUDA_AVAILABLE

    // cpu fallback implementation when cuda is not available
    // degrades to host execution using STL
    struct cuda_space
    {
        using device_type        = vendors::cpu::cpu_device_t;
        using memory_space_type  = void;
        using stream_handle_type = void*;
        using event_handle_type  = void*;

        static constexpr std::string_view space_name()
        {
            return "cuda";
        }

        static constexpr std::string_view name()
        {
            return space_name();
        }

        // concept requirements
        static constexpr bool is_host_space    = true;  // fallback to host
        static constexpr bool is_device_space  = false; // no real device
        static constexpr bool supports_async   = false; // no async in fallback
        static constexpr bool supports_kernels = false; // no kernels in fallback

        // legacy compatibility
        static constexpr bool is_async = false;
        static constexpr bool is_gpu   = false;

        static constexpr bool supports_shared_memory   = false;
        static constexpr bool supports_async_execution = false;
        static constexpr bool supports_unified_memory  = false;

        static std::size_t max_concurrency()
        {
            return 1; // fallback is sequential
        }

        static constexpr std::size_t preferred_block_size()
        {
            return 1;
        }

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            return 0.0; // not applicable
        }

        struct execution_context
        {
            void* stream     = nullptr;
            int   device_id  = 0;
            int   block_size = 1;

            execution_context() = default;
        };

        static void initialize() {}

        static stream_handle_type create_stream()
        {
            return nullptr;
        }

        static void destroy_stream(stream_handle_type) {}

        static event_handle_type create_event()
        {
            return nullptr;
        }

        static void destroy_event(event_handle_type) {}

        static void record_event(event_handle_type, stream_handle_type) {}

        static bool is_event_ready(event_handle_type)
        {
            return true;
        }

        static void synchronize_event(event_handle_type) {}

        static void wait_for_event(event_handle_type) {}

        static void stream_wait_event(stream_handle_type, event_handle_type) {}

        static event_handle_type record_event(stream_handle_type)
        {
            return nullptr;
        }

        static void synchronize_stream(stream_handle_type) {}

        static bool is_stream_ready(stream_handle_type)
        {
            return true;
        }

        static void set_device(std::int64_t) {}

        static void* allocate(std::size_t)
        {
            return nullptr;
        }

        static void deallocate(void*) {}

        static void synchronize() {}

        template <typename MemorySpace>
        static constexpr bool is_accessible_from()
        {
            return false;
        }
    };

#endif // XPU_CUDA_AVAILABLE

    // note: static_assert(execution_space<cuda_space>) moved to xpu.hpp
    // cannot verify here due to incomplete types

} // namespace simbi::xpu::exec
