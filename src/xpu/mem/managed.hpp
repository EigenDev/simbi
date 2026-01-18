// =============================================================================
// managed.hpp
//
// [TODO: Add description]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "platform.hpp" // for

#include <cstddef> // for size_t

namespace simbi {
    
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
