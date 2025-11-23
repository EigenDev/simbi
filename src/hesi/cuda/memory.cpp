#include "hesi/backend/memory.hpp"

#ifdef CUDA_ENABLED
#include "hesi/core/error_handling.hpp"
#include <cuda_runtime.h>

namespace simbi::het::backend {

    void*
    allocate_cuda(std::size_t bytes, memory_type_t type, std::int32_t device_id)
    {
        if (bytes == 0) {
            return nullptr;
        }

        // set device if specified
        if (device_id >= 0) {
            check_error<cuda_backend_t>(
                cudaSetDevice(device_id),
                "set device for allocation"
            );
        }

        void* ptr = nullptr;

        switch (type) {
            case memory_type_t::device_local:
                check_error<cuda_backend_t>(
                    cudaMalloc(&ptr, bytes),
                    "device allocation"
                );
                break;

            case memory_type_t::managed:
                check_error<cuda_backend_t>(
                    cudaMallocManaged(&ptr, bytes),
                    "managed allocation"
                );
                break;

            case memory_type_t::pinned:
                check_error<cuda_backend_t>(
                    cudaMallocHost(&ptr, bytes),
                    "pinned allocation"
                );
                break;

            case memory_type_t::host_visible:
                // host_visible on cuda means pinned for performance
                check_error<cuda_backend_t>(
                    cudaMallocHost(&ptr, bytes),
                    "host allocation"
                );
                break;
        }

        if (!ptr) {
            throw std::runtime_error("cuda allocation returned null");
        }

        return ptr;
    }

    void deallocate_cuda(void* ptr, memory_type_t type)
    {
        if (!ptr) {
            return;
        }

        switch (type) {
            case memory_type_t::device_local:
            case memory_type_t::managed:
                check_error<cuda_backend_t>(cudaFree(ptr), "device free");
                break;

            case memory_type_t::pinned:
            case memory_type_t::host_visible:
                check_error<cuda_backend_t>(cudaFreeHost(ptr), "host free");
                break;
        }
    }

    pointer_info_t query_pointer_cuda(const void* ptr)
    {
        cudaPointerAttributes attrs;
        cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

        if (err != cudaSuccess) {
            cudaGetLastError();   // clear error
            return {
              backend_type_t::cpu,
              memory_type_t::host_visible,
              -1,
              false
            };
        }

        pointer_info_t info;
        info.is_valid  = true;
        info.device_id = attrs.device;

        switch (attrs.type) {
            case cudaMemoryTypeDevice:
                info.backend = backend_type_t::cuda;
                info.type    = memory_type_t::device_local;
                break;
            case cudaMemoryTypeManaged:
                info.backend = backend_type_t::cuda;
                info.type    = memory_type_t::managed;
                break;
            case cudaMemoryTypeHost:
                info.backend = backend_type_t::cpu;
                info.type    = memory_type_t::pinned;
                break;
            default:
                info.backend = backend_type_t::cpu;
                info.type    = memory_type_t::host_visible;
        }

        return info;
    }

}   // namespace simbi::het::backend

#endif   // CUDA_ENABLED
