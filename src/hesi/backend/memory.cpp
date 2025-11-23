#include "hesi/backend/memory.hpp"
#include "hesi/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace simbi::het::backend {

    // forward declarations for backend-specific implementations
    void* allocate_cpu(std::size_t bytes, memory_type_t type);
    void deallocate_cpu(void* ptr, memory_type_t type);

#ifdef CUDA_ENABLED
    void* allocate_cuda(
        std::size_t bytes,
        memory_type_t type,
        std::int32_t device_id
    );
    void deallocate_cuda(void* ptr, memory_type_t type);
    pointer_info_t query_pointer_cuda(const void* ptr);
#endif

#ifdef HIP_ENABLED
    void*
    allocate_hip(std::size_t bytes, memory_type_t type, std::int32_t device_id);
    void deallocate_hip(void* ptr, memory_type_t type);
    pointer_info_t query_pointer_hip(const void* ptr);
#endif

    // public dispatcher
    void* allocate(
        backend_type_t backend,
        std::size_t bytes,
        memory_type_t type,
        std::int32_t device_id
    )
    {
        switch (backend) {
            case backend_type_t::cpu: {
                (void) device_id;   // unused
                return allocate_cpu(bytes, type);
            }

#ifdef CUDA_ENABLED
            case backend_type_t::cuda:
                return allocate_cuda(bytes, type, device_id);
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip:
                return allocate_hip(bytes, type, device_id);
#endif

            default:
                throw std::runtime_error("unsupported backend for allocation");
        }
    }

    void deallocate(backend_type_t backend, void* ptr, memory_type_t type)
    {
        if (!ptr) {
            return;
        }

        switch (backend) {
            case backend_type_t::cpu: deallocate_cpu(ptr, type); break;

#ifdef CUDA_ENABLED
            case backend_type_t::cuda: deallocate_cuda(ptr, type); break;
#endif

#ifdef HIP_ENABLED
            case backend_type_t::hip: deallocate_hip(ptr, type); break;
#endif

            default:
                // don't throw in destructor path
                break;
        }
    }

    pointer_info_t query_pointer(const void* ptr)
    {
        if (!ptr) {
            return {
              backend_type_t::cpu,
              memory_type_t::host_visible,
              -1,
              false
            };
        }

        // try cuda first (most common)
#ifdef CUDA_ENABLED
        auto info = query_pointer_cuda(ptr);
        if (info.is_valid) {
            return info;
        }
#endif

#ifdef HIP_ENABLED
        auto info = query_pointer_hip(ptr);
        if (info.is_valid) {
            return info;
        }
#endif

        // fallback: assume cpu
        return {backend_type_t::cpu, memory_type_t::host_visible, -1, true};
    }

}   // namespace simbi::het::backend
