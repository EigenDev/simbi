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

#include "config.hpp"                              // for global::managed_memory
#include "system/adapter/device_adapter_api.hpp"   // for deviceSynch, gpuFree, gpuMallocManaged
#include "system/adapter/device_types.hpp"
#include <cstddef>   // for size_t
#include <new>       // for new, delete

namespace simbi {
    template <bool gpu_managed = global::managed_memory>
    class managed_t
    {
      public:
        // virt destructor for polymorphic use
        virtual ~managed_t() = default;

        // bare bones new and delete
        static constexpr void* operator new(std::size_t len)
        {
            if constexpr (gpu_managed) {
                void* ptr = nullptr;
                gpu::api::malloc_managed(&ptr, len);
                gpu::api::device_synch();
                return ptr;
            }
            return ::operator new(len);
        }

        static constexpr void operator delete(void* ptr) noexcept
        {
            if constexpr (gpu_managed) {
                gpu::api::device_synch();
                gpu::api::free(ptr);
            }
            else {
                ::operator delete(ptr);
            }
        }

        // array support bc why not?
        static constexpr void* operator new[](std::size_t len)
        {
            if constexpr (gpu_managed) {
                void* ptr = nullptr;
                gpu::api::malloc_managed(&ptr, len);
                gpu::api::device_synch();
                return ptr;
            }
            return ::operator new[](len);
        }

        static constexpr void operator delete[](void* ptr) noexcept
        {
            if constexpr (gpu_managed) {
                gpu::api::device_synch();
                gpu::api::free(ptr);
            }
            else {
                ::operator delete[](ptr);
            }
        }

        // sized delete as well
        static constexpr void
        operator delete(void* ptr, std::size_t size) noexcept
        {
            if constexpr (gpu_managed) {
                gpu::api::device_synch();
                gpu::api::free(ptr);
            }
            else {
                ::operator delete(ptr, size);
            }
        }

        // sized array delete
        static constexpr void
        operator delete[](void* ptr, std::size_t size) noexcept
        {
            if constexpr (gpu_managed) {
                gpu::api::device_synch();
                gpu::api::free(ptr);
            }
            else {
                ::operator delete[](ptr, size);
            }
        }

        // placement new
        static constexpr void* operator new(std::size_t, void* ptr) noexcept
        {
            return ptr;
        }

        static constexpr void operator delete(void*, void*) noexcept
        {
            // placement delete is a no-op ;^]
        }

        // memory prefetching methods (TODO: revisit this)
        void prefetch_to_device(std::int64_t device = 0) const
        {
            if constexpr (gpu_managed) {
                adapter::stream_t<> stream;
                gpu::api::stream_create(&stream);
                gpu::api::prefetch_to_device(this, sizeof(*this), device);
                gpu::api::stream_synchronize(stream);
                gpu::api::stream_destroy(stream);
            }
        }
    };
}   // namespace simbi
#endif
