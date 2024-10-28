/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       managed.hpp
 * @brief    houses the gpu-Managed object for modified new and delete operators
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef MANAGED_HPP
#define MANAGED_HPP

#include "build_options.hpp"   // for global::managed_memory
#include "device_api.hpp"      // for deviceSynch, gpuFree, gpuMallocManaged
#include <cstddef>             // for size_t

namespace simbi {
    template <bool gpu_managed = global::managed_memory>
    class Managed
    {
      public:
        static constexpr void* operator new(std::size_t len)
        {
            if constexpr (gpu_managed) {
                void* ptr = nullptr;
                gpu::api::gpuMallocManaged(&ptr, len);
                gpu::api::deviceSynch();
                return ptr;
            }
            return ::operator new(len);
        }

        static constexpr void operator delete(void* ptr)
        {
            if constexpr (gpu_managed) {
                gpu::api::deviceSynch();
                gpu::api::gpuFree(ptr);
            }
            else {
                ::operator delete(ptr);
            }
        }
    };
}   // namespace simbi
#endif
