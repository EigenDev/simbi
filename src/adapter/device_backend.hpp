/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            device_backend.hpp
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
#ifndef DEVICE_BACKEND_HPP
#define DEVICE_BACKEND_HPP

#include "build_options.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace simbi::adapter {

    // backend tags for compile-time selection
    struct cpu_backend_tag {
    };
    struct cuda_backend_tag {
    };
    struct hip_backend_tag {
    };
    struct metal_backend_tag {
    };
    struct sycl_backend_tag {
    };

// default backend selection based on compile flags
#if defined(CUDA_ENABLED) || (defined(GPU_CODE) && CUDA_CODE)
    using default_backend_tag = cuda_backend_tag;
#elif defined(HIP_ENABLED) || (defined(GPU_CODE) && HIP_CODE)
    using default_backend_tag = hip_backend_tag;
#elif defined(METAL_ENABLED)
    using default_backend_tag = metal_backend_tag;
#elif defined(SYCL_ENABLED)
    using default_backend_tag = sycl_backend_tag;
#else
    using default_backend_tag = cpu_backend_tag;
#endif

    // error handling utilities (reusing from existing code)
    namespace error {
        enum class status_t {
            success = 0,
            error
        };

        class runtime_error : public ::std::runtime_error
        {
          public:
            runtime_error(status_t error_code)
                : ::std::runtime_error(
                      "Device error: " +
                      std::to_string(static_cast<int>(error_code)) +
                      " at " __FILE__ ":" + std::to_string(__LINE__)
                  ),
                  internal_code(error_code)
            {
            }

            runtime_error(status_t error_code, const ::std::string& what_arg)
                : ::std::runtime_error(
                      what_arg + ": Error code " +
                      std::to_string(static_cast<int>(error_code)) +
                      " at " __FILE__ ":" + std::to_string(__LINE__)
                  ),
                  internal_code(error_code)
            {
            }

            status_t code() const { return internal_code; }

          private:
            status_t internal_code;
        };

        constexpr bool is_err(status_t status)
        {
            return status != status_t::success;
        }

        inline void
        check_err(status_t status, const ::std::string& message = {})
        {
            if (is_err(status)) {
                throw runtime_error(status, message);
            }
        }
    }   // namespace error

    // common types used across all backends
    namespace types {
        // forward declare common types
        struct dim3 {
            unsigned int x, y, z;

            constexpr dim3(
                unsigned int x_ = 1,
                unsigned int y_ = 1,
                unsigned int z_ = 1
            )
                : x(x_), y(y_), z(z_)
            {
            }

            constexpr unsigned int volume() const { return x * y * z; }
        };
    }   // namespace types

    // base device backend interface - implemented by specializations
    template <typename BackendTag>
    class DeviceBackend
    {
      public:
        // memory operations
        void copy_host_to_device(void* to, const void* from, std::size_t bytes);
        void copy_device_to_host(void* to, const void* from, std::size_t bytes);
        void
        copy_device_to_device(void* to, const void* from, std::size_t bytes);
        void malloc(void** obj, std::size_t bytes);
        void malloc_managed(void** obj, std::size_t bytes);
        void free(void* obj);
        void memset(void* obj, int val, std::size_t bytes);

        // event handling
        void event_create(void** event);
        void event_destroy(void* event);
        void event_record(void* event);
        void event_synchronize(void* event);
        void event_elapsed_time(float* time, void* start, void* end);

        // device management
        void get_device_count(int* count);
        void get_device_properties(void* props, int device);
        void set_device(int device);
        void device_synchronize();

        // stream operations
        void stream_create(void** stream);
        void stream_destroy(void* stream);
        void stream_synchronize(void* stream);
        void
        stream_wait_event(void* stream, void* event, unsigned int flags = 0);
        void stream_query(void* stream, int* status);

        // asynchronous operations
        void async_copy_host_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        );
        void async_copy_device_to_host(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        );
        void async_copy_device_to_device(
            void* to,
            const void* from,
            std::size_t bytes,
            void* stream
        );
        void memcpy_async(
            void* to,
            const void* from,
            std::size_t bytes,
            int kind,
            void* stream
        );

        // peer operations
        void enable_peer_access(int device, unsigned int flags = 0);
        void peer_copy_async(
            void* dst,
            int dst_device,
            const void* src,
            int src_device,
            std::size_t bytes,
            void* stream
        );

        // host memory management
        void host_register(void* ptr, std::size_t size, unsigned int flags);
        void host_unregister(void* ptr);
        void aligned_malloc(void** ptr, std::size_t size);

        // specialized operations
        void
        memcpy_from_symbol(void* dst, const void* symbol, std::size_t count);
        void
        prefetch_to_device(const void* obj, std::size_t bytes, int device = 0);
        void launch_kernel(
            void* function,
            types::dim3 grid,
            types::dim3 block,
            void** args,
            std::size_t shared_mem,
            void* stream
        );

        // atomic operations
        template <typename T>
        DUAL T atomic_min(T* address, T val);

        template <typename T>
        DUAL T atomic_add(T* address, T val);

        // thread synchronization
        DEV void synchronize_threads();
    };

    // get the singleton instance of the appropriate backend
    template <typename BackendTag = default_backend_tag>
    DeviceBackend<BackendTag>& get_device_backend()
    {
        static DeviceBackend<BackendTag> instance;
        return instance;
    }

}   // namespace simbi::adapter

#endif   // DEVICE_BACKEND_HPP
