#ifndef HET_ALLOCATOR_HPP
#define HET_ALLOCATOR_HPP

#include "compat.hpp"
#include "hesi/core/types.hpp"

#include <cstdlib>
#include <stdexcept>

namespace simbi::het::mem {

    struct allocator_t {
        // allocates raw bytes based on locality and policy
        // throws on failure to keep return type simple
        static void*
        allocate(std::size_t bytes, locality_t loc, memory_type_t type)
        {
            void* ptr = nullptr;

            if (loc.backend == backend_type_t::cpu) {
                // cpu allocation logic
                if (type == memory_type_t::pinned) {
#if defined(CUDA_ENABLED)
                    cudaMallocHost(&ptr, bytes);
#elif defined(HIP_ENABLED)
                    hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
#else
                    ptr = std::malloc(bytes);
#endif
                }
                else {
                    ptr = std::malloc(bytes);
                }
            }
            else if (loc.backend == backend_type_t::cuda ||
                     loc.backend == backend_type_t::hip) {
// gpu allocation logic
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
                if (loc.device_id != -1) {
#if defined(CUDA_ENABLED)
                    cudaSetDevice(loc.device_id);
#else
                    hipSetDevice(loc.device_id);
#endif
                }

                if (type == memory_type_t::managed) {
#if defined(CUDA_ENABLED)
                    cudaMallocManaged(&ptr, bytes);
#elif defined(HIP_ENABLED)
                    hipMallocManaged(&ptr, bytes);
#endif
                }
                else {
// device_local
#if defined(CUDA_ENABLED)
                    cudaMalloc(&ptr, bytes);
#elif defined(HIP_ENABLED)
                    hipMalloc(&ptr, bytes);
#endif
                }
#else
                throw std::runtime_error("gpu backend disabled");
#endif
            }

            if (!ptr) {
                throw std::runtime_error("allocation failed");
            }
            return ptr;
        }

        // frees memory based on locality
        static void deallocate(void* ptr, locality_t loc, memory_type_t type)
        {
            if (!ptr) {
                return;
            }

            if (loc.backend == backend_type_t::cpu) {
                if (type == memory_type_t::pinned) {
#if defined(CUDA_ENABLED)
                    cudaFreeHost(ptr);
#elif defined(HIP_ENABLED)
                    hipHostFree(ptr);
#else
                    std::free(ptr);
#endif
                }
                else {
                    std::free(ptr);
                }
            }
            else if (loc.backend == backend_type_t::cuda ||
                     loc.backend == backend_type_t::hip) {
#if defined(CUDA_ENABLED)
                cudaFree(ptr);
#elif defined(HIP_ENABLED)
                hipFree(ptr);
#endif
            }
        }
    };

}   // namespace simbi::het::mem

#endif   // HETERO_ALLOCATOR_HPP
