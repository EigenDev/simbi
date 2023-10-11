#ifndef MANAGED_HPP
#define MANAGED_HPP

#include "device_api.hpp"
namespace simbi
{
    template<bool gpu_managed = managed_memory>
    class Managed{
        public: 
        static constexpr void* operator new(std::size_t len) {
            if constexpr(gpu_managed) {
                void *ptr = nullptr;
                gpu::api::gpuMallocManaged(&ptr, len);
                gpu::api::deviceSynch();
                return ptr;
            } 
            return ::operator new(len);
        }
        static constexpr void operator delete(void *ptr) {
            if constexpr(gpu_managed) {
                gpu::api::deviceSynch();
                gpu::api::gpuFree(ptr);
            } else {
                ::operator delete(ptr);
            }
        }
    };
} // namespace simbi
#endif

