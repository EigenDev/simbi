/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            device_types.hpp
 * @brief           Type definitions for device backends
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
 */
#ifndef DEVICE_TYPES_HPP
#define DEVICE_TYPES_HPP

#include "config.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>

// Include backend-specific headers conditionally
#if defined(CUDA_ENABLED) || (defined(GPU_ENABLED) && CUDA_ENABLED)
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <driver_types.h>
#elif defined(HIP_ENABLED) || (defined(GPU_ENABLED) && HIP_ENABLED)
#include <hip/hip_runtime.h>
#endif

namespace simbi::adapter {
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
}   // namespace simbi::adapter

namespace simbi::adapter {

    namespace types {

        // common types used across all backends
#if !defined(CUDA_ENABLED) && !defined(HIP_ENABLED) &&                         \
    !defined(METAL_ENABLED) && !defined(SYCL_ENABLED)
        // 3D dimensions structure
        struct dim3 {
            std::uint64_t x, y, z;

            constexpr dim3(
                std::uint64_t x_ = 1,
                std::uint64_t y_ = 1,
                std::uint64_t z_ = 1
            )
                : x(x_), y(y_), z(z_)
            {
            }

            constexpr std::uint64_t volume() const { return x * y * z; }
        };
#else
        using dim3 = ::dim3;   // Use standard dim3 from CUDA/HIP
#endif

        // Memory copy directions
        enum class memcpy_kind {
            host_to_device,
            device_to_host,
            device_to_device,
            host_to_host,
        };
    }   // namespace types

// Default backend selection based on compile flags
#if defined(CUDA_ENABLED) || (defined(GPU_ENABLED) && CUDA_ENABLED)
    using default_backend_tag = cuda_backend_tag;
#elif defined(HIP_ENABLED) || (defined(GPU_ENABLED) && HIP_ENABLED)
    using default_backend_tag = hip_backend_tag;
#elif defined(METAL_ENABLED)
    using default_backend_tag = metal_backend_tag;
#elif defined(SYCL_ENABLED)
    using default_backend_tag = sycl_backend_tag;
#else
    using default_backend_tag = cpu_backend_tag;
#endif

    // CPU type definitions for when no GPU backend is available
    namespace cpu {
        // Simple stubs for CPU emulation
        struct event_t {
        };
        struct stream_t {
        };
        struct device_properties_t {
            char name[256];
            std::int64_t major;
            std::int64_t minor;
            size_t totalGlobalMem;
            std::int64_t multiProcessorCount;
            std::int64_t maxThreadsPerBlock;
            std::int64_t maxThreadsPerMultiProcessor;
            types::dim3 maxThreadsDim;
            types::dim3 maxGridSize;
        };
        using function_t = void (*)(void);
    }   // namespace cpu

    // Type traits for backend-specific types
    // This is the primary template that gets specialized for each backend
    template <typename BackendTag>
    struct backend_types {
    };

    // CPU backend specialization
    template <>
    struct backend_types<cpu_backend_tag> {
        using event_t  = std::chrono::high_resolution_clock::time_point;
        using stream_t = cpu::stream_t;
        using device_properties_t = cpu::device_properties_t;
        using function_t          = cpu::function_t;
        using memcpy_kind_t       = types::memcpy_kind;

        // Constants for this backend
        static constexpr memcpy_kind_t memcpy_host_to_device =
            types::memcpy_kind::host_to_device;
        static constexpr memcpy_kind_t memcpy_device_to_host =
            types::memcpy_kind::device_to_host;
        static constexpr memcpy_kind_t memcpy_device_to_device =
            types::memcpy_kind::device_to_device;
        static constexpr memcpy_kind_t memcpy_host_to_host =
            types::memcpy_kind::host_to_host;
    };

// CUDA backend specialization
#if defined(CUDA_ENABLED) || (defined(GPU_ENABLED) && CUDA_ENABLED)
    template <>
    struct backend_types<cuda_backend_tag> {
        using event_t             = cudaEvent_t;
        using stream_t            = cudaStream_t;
        using device_properties_t = cudaDeviceProp;
        using function_t          = CUfunction;
        using memcpy_kind_t       = cudaMemcpyKind;

        // Constants for this backend
        static constexpr memcpy_kind_t memcpy_host_to_device =
            cudaMemcpyHostToDevice;
        static constexpr memcpy_kind_t memcpy_device_to_host =
            cudaMemcpyDeviceToHost;
        static constexpr memcpy_kind_t memcpy_device_to_device =
            cudaMemcpyDeviceToDevice;
        static constexpr memcpy_kind_t memcpy_host_to_host =
            cudaMemcpyHostToHost;
    };
#endif

// HIP backend specialization
#if defined(HIP_ENABLED) || (defined(GPU_ENABLED) && HIP_ENABLED)
    template <>
    struct backend_types<hip_backend_tag> {
        using event_t             = hipEvent_t;
        using stream_t            = hipStream_t;
        using device_properties_t = hipDeviceProp_t;
        using function_t          = hipFunction_t;
        using memcpy_kind_t       = hipMemcpyKind;

        // Constants for this backend
        static constexpr memcpy_kind_t memcpy_host_to_device =
            hipMemcpyHostToDevice;
        static constexpr memcpy_kind_t memcpy_device_to_host =
            hipMemcpyDeviceToHost;
        static constexpr memcpy_kind_t memcpy_device_to_device =
            hipMemcpyDeviceToDevice;
        static constexpr memcpy_kind_t memcpy_host_to_host =
            hipMemcpyHostToHost;
    };
#endif

    // Type aliases for common use
    template <typename BackendTag = default_backend_tag>
    using event_t = typename backend_types<BackendTag>::event_t;

    template <typename BackendTag = default_backend_tag>
    using stream_t = typename backend_types<BackendTag>::stream_t;

    template <typename BackendTag = default_backend_tag>
    using device_properties_t =
        typename backend_types<BackendTag>::device_properties_t;

    template <typename BackendTag = default_backend_tag>
    using function_t = typename backend_types<BackendTag>::function_t;

    template <typename BackendTag = default_backend_tag>
    using memcpy_kind_t = typename backend_types<BackendTag>::memcpy_kind_t;

    // Compile-time constants for memcpy kinds
    template <typename BackendTag = default_backend_tag>
    inline constexpr memcpy_kind_t<BackendTag> memcpy_host_to_device =
        backend_types<BackendTag>::memcpy_host_to_device;

    template <typename BackendTag = default_backend_tag>
    inline constexpr memcpy_kind_t<BackendTag> memcpy_device_to_host =
        backend_types<BackendTag>::memcpy_device_to_host;

    template <typename BackendTag = default_backend_tag>
    inline constexpr memcpy_kind_t<BackendTag> memcpy_device_to_device =
        backend_types<BackendTag>::memcpy_device_to_device;

    template <typename BackendTag = default_backend_tag>
    inline constexpr memcpy_kind_t<BackendTag> memcpy_host_to_host =
        backend_types<BackendTag>::memcpy_host_to_host;

    // execution grid abstractions
    namespace grid {
        // execution index class to replace direct access to blockIdx, threadIdx
        // etc.
        class execution_index
        {
          private:
            // store indices internally as types::dim3
            types::dim3 block_idx_;
            types::dim3 thread_idx_;
            types::dim3 block_dim_;
            types::dim3 grid_dim_;

          public:
            // constructor that takes backend tag to enable specialization
            template <typename BackendTag = default_backend_tag>
            DUAL static execution_index current();

            // access methods that mimic the original .x, .y, .z syntax
            DUAL std::uint64_t block_x() const { return block_idx_.x; }
            DUAL std::uint64_t block_y() const { return block_idx_.y; }
            DUAL std::uint64_t block_z() const { return block_idx_.z; }

            DUAL std::uint64_t thread_x() const { return thread_idx_.x; }
            DUAL std::uint64_t thread_y() const { return thread_idx_.y; }
            DUAL std::uint64_t thread_z() const { return thread_idx_.z; }

            DUAL std::uint64_t block_dim_x() const { return block_dim_.x; }
            DUAL std::uint64_t block_dim_y() const { return block_dim_.y; }
            DUAL std::uint64_t block_dim_z() const { return block_dim_.z; }

            DUAL std::uint64_t grid_dim_x() const { return grid_dim_.x; }
            DUAL std::uint64_t grid_dim_y() const { return grid_dim_.y; }
            DUAL std::uint64_t grid_dim_z() const { return grid_dim_.z; }

            // access to the raw dim3 objects if needed
            DUAL const types::dim3& block() const { return block_idx_; }
            DUAL const types::dim3& thread() const { return thread_idx_; }
            DUAL const types::dim3& block_dims() const { return block_dim_; }
            DUAL const types::dim3& grid_dims() const { return grid_dim_; }

            // common utility functions
            DUAL std::uint64_t global_thread_id() const
            {
                return (block_z() * block_dim_z() + thread_z()) *
                           block_dim_x() * grid_dim_x() * block_dim_y() *
                           grid_dim_y() +
                       (block_y() * block_dim_y() + thread_y()) *
                           block_dim_x() * grid_dim_x() +
                       block_x() * block_dim_x() + thread_x();
            }

            DUAL std::uint64_t thread_id() const
            {
                return block_dim_x() * block_dim_y() * thread_z() +
                       block_dim_x() * thread_y() + thread_x();
            }

            DUAL std::uint64_t block_id() const
            {
                return block_x() + block_y() * grid_dim_x() +
                       block_z() * grid_dim_x() * grid_dim_y();
            }

            DUAL std::uint64_t threads_per_block() const
            {
                return block_dim_x() * block_dim_y() * block_dim_z();
            }

            DUAL std::uint64_t total_threads() const
            {
                return block_dim_x() * grid_dim_x() * block_dim_y() *
                       grid_dim_y() * block_dim_z() * grid_dim_z();
            }
        };

        // template specialization for CPU backend
        template <>
        DUAL inline execution_index execution_index::current<cpu_backend_tag>()
        {
            execution_index idx;
            // cpu uses single-threaded execution model with 1,1,1 dimensions
            idx.block_idx_  = types::dim3(0, 0, 0);
            idx.thread_idx_ = types::dim3(0, 0, 0);
            idx.block_dim_  = types::dim3(1, 1, 1);
            idx.grid_dim_   = types::dim3(1, 1, 1);
            return idx;
        }

#if defined(CUDA_ENABLED)
        // CUDA backend specialization
        template <>
        DUAL inline execution_index execution_index::current<cuda_backend_tag>()
        {
            execution_index idx;
#if defined(__CUDA_ARCH__)   // only valid in device code
            // direct access to CUDA built-in variables
            idx.block_idx_ = types::dim3(blockIdx.x, blockIdx.y, blockIdx.z);
            idx.thread_idx_ =
                types::dim3(threadIdx.x, threadIdx.y, threadIdx.z);
            idx.block_dim_ = types::dim3(blockDim.x, blockDim.y, blockDim.z);
            idx.grid_dim_  = types::dim3(gridDim.x, gridDim.y, gridDim.z);
#else
            // host-side fallback (these aren't usable on host normally)
            idx.block_idx_  = types::dim3(0, 0, 0);
            idx.thread_idx_ = types::dim3(0, 0, 0);
            idx.block_dim_  = types::dim3(1, 1, 1);
            idx.grid_dim_   = types::dim3(1, 1, 1);
#endif
            return idx;
        }
#endif

#if defined(HIP_ENABLED)
        // HIP backend specialization
        template <>
        DUAL inline execution_index execution_index::current<hip_backend_tag>()
        {
            execution_index idx;
#if defined(__HIP_DEVICE_COMPILE__)   // only valid in device code
            // direct access to HIP built-in variables
            idx.block_idx_ = types::dim3(blockIdx.x, blockIdx.y, blockIdx.z);
            idx.thread_idx_ =
                types::dim3(threadIdx.x, threadIdx.y, threadIdx.z);
            idx.block_dim_ = types::dim3(blockDim.x, blockDim.y, blockDim.z);
            idx.grid_dim_  = types::dim3(gridDim.x, gridDim.y, gridDim.z);
#else
            // host-side fallback
            idx.block_idx_  = types::dim3(0, 0, 0);
            idx.thread_idx_ = types::dim3(0, 0, 0);
            idx.block_dim_  = types::dim3(1, 1, 1);
            idx.grid_dim_   = types::dim3(1, 1, 1);
#endif
            return idx;
        }
#endif

        // kernel launch configuration abstraction
        class launch_config
        {
          private:
            types::dim3 grid_;
            types::dim3 block_;
            size_t shared_memory_;

          public:
            DUAL launch_config(
                const types::dim3& grid  = types::dim3(1, 1, 1),
                const types::dim3& block = types::dim3(1, 1, 1),
                size_t shared_memory     = 0
            )
                : grid_(grid), block_(block), shared_memory_(shared_memory)
            {
            }

            // getters
            DUAL const types::dim3& grid() const { return grid_; }
            DUAL const types::dim3& block() const { return block_; }
            DUAL size_t shared_memory() const { return shared_memory_; }
        };

        // helper function to calculate grid size from total elements
        inline types::dim3
        calculate_grid(size_t total_elements, const types::dim3& block)
        {
            const auto x = (total_elements + block.x - 1) / block.x;
            if (block.y > 1) {
                const auto y = (total_elements + block.y - 1) / block.y;
                if (block.z > 1) {
                    const auto z = (total_elements + block.z - 1) / block.z;
                    return types::dim3(x, y, z);
                }
                return types::dim3(x, y, 1);
            }
            return types::dim3(x, 1, 1);
        }

        // helper to convert between types::dim3 and ::dim3
        inline types::dim3 to_native_dim3(const types::dim3& dim)
        {
            return types::dim3(dim.x, dim.y, dim.z);
        }

        inline types::dim3 from_native_dim3(const types::dim3& dim)
        {
            return types::dim3(dim.x, dim.y, dim.z);
        }
    }   // namespace grid

    // for easy access in both the adapter namespace and globally
    using grid_index = grid::execution_index;

    // updated kernel launch templates (to replace macros)
    template <typename Func, typename... Args>
    inline void launch_kernel(
        Func kernel,
        const grid::launch_config& config,
        Args&&... args
    )
    {
        if constexpr (std::is_same_v<default_backend_tag, cpu_backend_tag>) {
            (void) config;   // suppress unused warning for CPU backend
            // cpu implementation just calls the function directly
            kernel(std::forward<Args>(args)...);
        }
        else {
            // gpu implementation uses appropriate launch syntax
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
            kernel<<<
                grid::to_native_dim3(config.grid()),
                grid::to_native_dim3(config.block()),
                config.shared_memory()>>>(std::forward<Args>(args)...);
#endif
        }
    }

    template <typename Func, typename... Args>
    inline void launch_single_thread(Func kernel, Args&&... args)
    {
        launch_kernel(
            kernel,
            grid::launch_config(),
            std::forward<Args>(args)...
        );
    }

    namespace types {
        // foward declaration
        template <typename BackendTag = default_backend_tag>
        class atomic_bool;

        // CPU specialization
        template <>
        class atomic_bool<cpu_backend_tag>
        {
          private:
            std::atomic<bool> value;

          public:
            atomic_bool() : value(false) {}
            explicit atomic_bool(bool initial_value) : value(initial_value) {}

            void store(bool new_value)
            {
                value.store(new_value, std::memory_order_relaxed);
            }

            bool load() const { return value.load(std::memory_order_relaxed); }

            bool exchange(bool new_value)
            {
                return value.exchange(new_value, std::memory_order_relaxed);
            }

            bool compare_exchange(bool expected, bool desired)
            {
                bool expected_copy = expected;
                return value.compare_exchange_strong(
                    expected_copy,
                    desired,
                    std::memory_order_relaxed
                );
            }

            explicit operator bool() const { return load(); }
        };

// CUDA specialization - only compiled when CUDA is enabled
#if defined(CUDA_ENABLED)
        template <>
        class atomic_bool<cuda_backend_tag>
        {
          private:
            cuda::atomic<bool, cuda::thread_scope_device> value;

          public:
            DUAL atomic_bool() : value(false) {}
            DUAL explicit atomic_bool(bool initial_value) : value(initial_value)
            {
            }

            DUAL void store(bool new_value)
            {
                value.store(new_value, cuda::memory_order_relaxed);
            }

            DUAL bool load() const
            {
                return value.load(cuda::memory_order_relaxed);
            }

            DUAL bool exchange(bool new_value)
            {
                return value.exchange(new_value, cuda::memory_order_relaxed);
            }

            DUAL bool compare_exchange(bool expected, bool desired)
            {
                bool expected_copy = expected;
                return value.compare_exchange_strong(
                    expected_copy,
                    desired,
                    cuda::memory_order_relaxed
                );
            }

            DUAL explicit operator bool() const { return load(); }
        };
#endif

// HIP specialization - only compiled when HIP is enabled
#if defined(HIP_ENABLED)
        template <>
        class atomic_bool<hip_backend_tag>
        {
          private:
            std::int64_t value;   // HIP uses int-based atomics

          public:
            DUAL atomic_bool() : value(0) {}
            DUAL explicit atomic_bool(bool initial_value)
                : value(initial_value ? 1 : 0)
            {
            }

            DUAL void store(bool new_value)
            {
#if defined(__HIP_DEVICE_COMPILE__)
                atomicExch(&value, new_value ? 1 : 0);
#else
                std::atomic<std::int64_t>* atomic_value =
                    reinterpret_cast<std::atomic<std::int64_t>*>(&value);
                atomic_value->store(
                    new_value ? 1 : 0,
                    std::memory_order_relaxed
                );
#endif
            }

            DUAL bool load() const
            {
#if defined(__HIP_DEVICE_COMPILE__)
                return atomicAdd(const_cast<int*>(&value), 0) != 0;
#else
                const std::atomic<std::int64_t>* atomic_value =
                    reinterpret_cast<const std::atomic<std::int64_t>*>(&value);
                return atomic_value->load(std::memory_order_relaxed) != 0;
#endif
            }

            DUAL bool exchange(bool new_value)
            {
#if defined(__HIP_DEVICE_COMPILE__)
                return atomicExch(&value, new_value ? 1 : 0) != 0;
#else
                std::atomic<std::int64_t>* atomic_value =
                    reinterpret_cast<std::atomic<std::int64_t>*>(&value);
                return atomic_value->exchange(
                           new_value ? 1 : 0,
                           std::memory_order_relaxed
                       ) != 0;
#endif
            }

            DUAL bool compare_exchange(bool expected, bool desired)
            {
#if defined(__HIP_DEVICE_COMPILE__)
                std::int64_t expected_std::int64_t = expected ? 1 : 0;
                std::int64_t desired_std::int64_t  = desired ? 1 : 0;
                return atomicCAS(&value, expected_int, desired_int) ==
                       expected_int;
#else
                std::atomic<std::int64_t>* atomic_value =
                    reinterpret_cast<std::atomic<std::int64_t>*>(&value);
                std::int64_t expected_std::int64_t = expected ? 1 : 0;
                return atomic_value->compare_exchange_strong(
                    expected_int,
                    desired ? 1 : 0,
                    std::memory_order_relaxed
                );
#endif
            }

            DUAL explicit operator bool() const { return load(); }
        };
#endif

    }   // namespace types
}   // namespace simbi::adapter

#endif   // DEVICE_TYPES_HPP
