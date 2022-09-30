#include "device_api.hpp"

namespace simbi
{
    namespace gpu
    {
        namespace api
        {
            void copyHostToDevice(void *to, const void *from, size_t bytes)
            {
                auto status = simbi::gpu::error::status_t(anyGpuMemcpy(to, from, bytes, anyGpuMemcpyHostToDevice));
                simbi::gpu::error::check_err(status, "Synchronous copy to dev failed");
            }

            void copyDevToHost(void *to, const void *from, size_t bytes)
            {
                auto status = simbi::gpu::error::status_t(anyGpuMemcpy(to, from, bytes, anyGpuMemcpyDeviceToHost));
                simbi::gpu::error::check_err(status, "Synchronous copy to host failed");
            }

            void copyDevToDev(void *to, const void *from, size_t bytes)
            {
                auto status = simbi::gpu::error::status_t(anyGpuMemcpy(to, from, bytes, anyGpuMemcpyDeviceToDevice));
                simbi::gpu::error::check_err(status, "Synchronous copy to dev2dev failed");
            }

            void gpuMalloc(void *obj, size_t elements)
            {
                auto status = simbi::gpu::error::status_t(anyGpuMalloc((void**)obj, elements));
                simbi::gpu::error::check_err(status, "Failed to allocate resources on device");
            }
            void gpuMallocManaged(void *obj, size_t elements)
            {
                auto status = simbi::gpu::error::status_t(anyGpuMallocManaged((void**)obj, elements));
                simbi::gpu::error::check_err(status, "Failed to allocate resources on device");
            }

            void gpuFree(void *obj)
            {
                auto status = simbi::gpu::error::status_t(anyGpuFree(obj));
                simbi::gpu::error::check_err(status, "Failed to free resources from device");
            }

            void gpuMemset(void *obj, int val, size_t bytes)
            {
                auto status = simbi::gpu::error::status_t(anyGpuMemset(obj, val, bytes));
                simbi::gpu::error::check_err(status, "Failed to memset");
            };
        } // namespace api
    
    } // namespace gpu
    
} // namespace simbi