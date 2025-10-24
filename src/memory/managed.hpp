/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            managed.hpp
 *  * @brief           a custom implementation of managed memory for GPU/CPU
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef MANAGED_HPP
#define MANAGED_HPP

#include "compat.hpp"   // for global::managed_memory
#include "hetero/adapter.hpp"
#include <cstddef>   // for size_t
#include <cstdint>   // for int64_t, uint64_t

namespace simbi {
    // ===============================================================================
    // This is adapted from:
    // https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
    // ==============================================================================
    /**
     * @brief
     * A custom implementation of managed memory that can be used
     * for both GPU and CPU memory management.
     * srp: provide a consistent interface for memory management
     * across CPU and GPU, with optional GPU managed memory.
     */
    template <bool gpu_managed = global::managed_memory>
    class managed_t
    {
      public:
        ~managed_t() = default;

        static void* operator new(std::size_t len)
        {
            if constexpr (gpu_managed) {
                void* ptr;
// Use raw API for custom allocators - bypass RAII wrapper
#ifdef CUDA_ENABLED
                cudaMallocManaged(&ptr, len);
                cudaDeviceSynchronize();
#elif defined(HIP_ENABLED)
                hipMallocManaged(&ptr, len);
                hipDeviceSynchronize();
#else
                ptr = ::operator new(len);
#endif
                return ptr;
            }
            return ::operator new(len);
        }

        static void operator delete(void* ptr) noexcept
        {
            if constexpr (gpu_managed) {
#ifdef CUDA_ENABLED
                cudaDeviceSynchronize();
                cudaFree(ptr);
#elif defined(HIP_ENABLED)
                hipDeviceSynchronize();
                hipFree(ptr);
#else
                ::operator delete(ptr);
#endif
            }
            else {
                ::operator delete(ptr);
            }
        }

        void prefetch_to_device(std::int64_t device = 0) const
        {
            if constexpr (gpu_managed) {
                hetero::device::prefetch_to_device(this, sizeof(*this), device);
            }
        }
    };
}   // namespace simbi
#endif
