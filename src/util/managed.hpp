#include "build_options.hpp"
#include "device_api.hpp"

namespace simbi
{
    template<Platform P = BuildPlatform>
    class Managed{
        public: 
        static constexpr void* operator new(std::size_t len) {
            if constexpr(P == Platform::GPU) {
                void *ptr;
                gpu::api::gpuMallocManaged(&ptr, len);
                gpu::api::deviceSynch();
                return ptr;
            } else {
                return ::operator new(len);
            }
        }
        static constexpr void operator delete(void *ptr) {
            if constexpr(P == Platform::GPU) {
                gpu::api::deviceSynch();
                gpu::api::gpuFree(ptr);
            } else {
                return ::operator delete(ptr);
            }
        }
    };
} // namespace simbi

