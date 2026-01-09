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

#include "compat.hpp" // for global::managed_memory

#include <cstddef> // for size_t

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
    class managed_t
    {
      public:
        ~managed_t() = default;

        static void* operator new(std::size_t len)
        {
            if constexpr (platform::is_gpu) {
                void* ptr;
// use raw api for custom allocators - bypass raii wrapper
#ifdef CUDA_ENABLED
                cudaMallocManaged(&ptr, len);
#elif defined(HIP_ENABLED)
                hipMallocManaged(&ptr, len);
#else
                ptr = ::operator new(len);
#endif
                return ptr;
            }
            return ::operator new(len);
        }

        static void operator delete(void* ptr) noexcept
        {
            if constexpr (platform::is_gpu) {
#ifdef CUDA_ENABLED
                cudaFree(ptr);
#elif defined(HIP_ENABLED)
                hipFree(ptr);
#else
                ::operator delete(ptr);
#endif
            }
            else {
                ::operator delete(ptr);
            }
        }
    };
} // namespace simbi
#endif
