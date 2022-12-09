#include "device_api.hpp"

namespace simbi
{
    namespace gpu
    {
        namespace api
        {
            void deviceSynch() {
                if constexpr(BuildPlatform == Platform::GPU)
                {
                    auto status = error::status_t(anyGpuDeviceSynchronize());
                    error::check_err(status, "Failed to synch device(s)");
                } else {
                    return;
                }
            }
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

            void gpuEventSynchronize(anyGpuEvent_t a)
            {
                auto status = simbi::gpu::error::status_t(anyGpuEventSynchronize(a));
                simbi::gpu::error::check_err(status, "Failed to synchronize event");
            };

            void gpuEventCreate(anyGpuEvent_t *a)
            {
                auto status = simbi::gpu::error::status_t(anyGpuEventCreate(a));
                simbi::gpu::error::check_err(status, "Failed to create event");
            };

            void gpuEventDestroy(anyGpuEvent_t a)
            {
                auto status = simbi::gpu::error::status_t(anyGpuEventDestroy(a));
                simbi::gpu::error::check_err(status, "Failed to create event");
            };

            void gpuEventRecord(anyGpuEvent_t a)
            {
                auto status = simbi::gpu::error::status_t(anyGpuEventRecord(a));
                simbi::gpu::error::check_err(status, "Failed to record event");
            };
            void gpuEventElapsedTime(float *time, anyGpuEvent_t a, anyGpuEvent_t b)
            {
                auto status = simbi::gpu::error::status_t(anyGpuEventElapsedTime(time, a, b));
                simbi::gpu::error::check_err(status, "Failed to get event elapsed time");
            };

            void getDeviceCount(int *devCount){
                auto status = simbi::gpu::error::status_t(anyGpuGetDeviceCount(devCount));
                simbi::gpu::error::check_err(status, "Failed to get device count");
            };

            void getDeviceProperties(anyGpuProp_t *props, int i){
                auto status = simbi::gpu::error::status_t(anyGpuGetDeviceProperties(props, i));
                simbi::gpu::error::check_err(status, "Failed to get device properties");
            };
        } // namespace api
    
    } // namespace gpu
    
} // namespace simbi