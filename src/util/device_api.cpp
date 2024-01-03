#include "device_api.hpp"
#include <memory>   // for allocator

namespace simbi {
    namespace gpu {
        namespace api {
            void copyHostToDevice(void* to, const void* from, size_t bytes)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuMemcpy(to, from, bytes, anyGpuMemcpyHostToDevice)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Synchronous copy from host to dev failed"
                );
#endif
            }

            void copyDevToHost(void* to, const void* from, size_t bytes)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuMemcpy(to, from, bytes, anyGpuMemcpyDeviceToHost)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Synchronous copy from dev to host failed"
                );
#endif
            }

            void copyDevToDev(void* to, const void* from, size_t bytes)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuMemcpy(to, from, bytes, anyGpuMemcpyDeviceToDevice)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Synchronous copy from dev to dev failed"
                );
#endif
            }

            void gpuMalloc(void* obj, size_t elements)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuMalloc((void**) obj, elements)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to allocate resources on device"
                );
#endif
            }

            void gpuMallocManaged(void* obj, size_t elements)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuMallocManaged((void**) obj, elements)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to allocate resources on device"
                );
#endif
            }

            void gpuFree(void* obj)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(anyGpuFree(obj));
                simbi::gpu::error::check_err(
                    status,
                    "Failed to free resources from device"
                );
#endif
            }

            void gpuMemset(void* obj, int val, size_t bytes)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(anyGpuMemset(obj, val, bytes));
                simbi::gpu::error::check_err(status, "Failed to memset");
#endif
            };

            void gpuEventSynchronize(anyGpuEvent_t a)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(anyGpuEventSynchronize(a));
                simbi::gpu::error::check_err(
                    status,
                    "Failed to synchronize event"
                );
#endif
            };

            void gpuEventCreate(anyGpuEvent_t* a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(anyGpuEventCreate(a));
                simbi::gpu::error::check_err(status, "Failed to create event");
#endif
            };

            void gpuEventDestroy(anyGpuEvent_t a)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(anyGpuEventDestroy(a));
                simbi::gpu::error::check_err(status, "Failed to destroy event");
#endif
            };

            void gpuEventRecord(anyGpuEvent_t a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(anyGpuEventRecord(a));
                simbi::gpu::error::check_err(status, "Failed to record event");
#endif
            };

            void
            gpuEventElapsedTime(float* time, anyGpuEvent_t a, anyGpuEvent_t b)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuEventElapsedTime(time, a, b)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to get event elapsed time"
                );
#endif
            };

            void getDeviceCount(int* devCount)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(anyGpuGetDeviceCount(devCount));
                simbi::gpu::error::check_err(
                    status,
                    "Failed to get device count"
                );
#endif
            };

            void getDeviceProperties(anyGpuProp_t* props, int i)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    anyGpuGetDeviceProperties(props, i)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to get device properties"
                );
#endif
            };

            GPU_DEV_INLINE
            void synchronize()
            {
#if GPU_CODE
                __syncthreads();
#endif
            };

            void deviceSynch()
            {
#if GPU_CODE
                auto status = error::status_t(anyGpuDeviceSynchronize());
                error::check_err(status, "Failed to synch device(s)");
#endif
            };
        }   // namespace api

    }   // namespace gpu

}   // namespace simbi