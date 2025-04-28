#include "util/tools/device_api.hpp"

namespace simbi {
    namespace gpu {
        namespace api {
            void copyHostToDevice(void* to, const void* from, size_t bytes)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    devMemcpy(to, from, bytes, devMemcpyHostToDevice)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Synchronous copy from host to dev failed"
                );
#endif
            }

            void copyDeviceToHost(void* to, const void* from, size_t bytes)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    devMemcpy(to, from, bytes, devMemcpyDeviceToHost)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Synchronous copy from dev to host failed"
                );
#endif
            }

            void copyDeviceToDevice(void* to, const void* from, size_t bytes)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    devMemcpy(to, from, bytes, devMemcpyDeviceToDevice)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Synchronous copy from dev to dev failed"
                );
#endif
            }

            void malloc(void* obj, size_t elements)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    devMalloc((void**) obj, elements)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to allocate resources on device"
                );
#endif
            }

            void mallocManaged(void* obj, size_t elements)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(
                    devMallocManaged((void**) obj, elements)
                );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to allocate resources on device"
                );
#endif
            }

            void free(void* obj)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devFree(obj));
                simbi::gpu::error::check_err(
                    status,
                    "Failed to free resources from device"
                );
#endif
            }

            void memset(void* obj, int val, size_t bytes)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(devMemset(obj, val, bytes));
                simbi::gpu::error::check_err(status, "Failed to memset");
#endif
            };

            void eventSynchronize(devEvent_t a)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(devEventSynchronize(a));
                simbi::gpu::error::check_err(
                    status,
                    "Failed to synchronize event"
                );
#endif
            };

            void eventCreate(devEvent_t* a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devEventCreate(a));
                simbi::gpu::error::check_err(status, "Failed to create event");
#endif
            };

            void eventDestroy(devEvent_t a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devEventDestroy(a));
                simbi::gpu::error::check_err(status, "Failed to destroy event");
#endif
            };

            void eventRecord(devEvent_t a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devEventRecord(a));
                simbi::gpu::error::check_err(status, "Failed to record event");
#endif
            };

            void eventElapsedTime(float* time, devEvent_t a, devEvent_t b)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(devEventElapsedTime(time, a, b)
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
                    simbi::gpu::error::status_t(devGetDeviceCount(devCount));
                simbi::gpu::error::check_err(
                    status,
                    "Failed to get device count"
                );
#endif
            };

            void getDeviceProperties(devProp_t* props, int i)
            {
#if GPU_CODE
                auto status =
                    simbi::gpu::error::status_t(devGetDeviceProperties(props, i)
                    );
                simbi::gpu::error::check_err(
                    status,
                    "Failed to get device properties"
                );
#endif
            };

            void deviceSynch()
            {
#if GPU_CODE
                auto status = error::status_t(devDeviceSynchronize());
                error::check_err(status, "Failed to synch device(s)");
#endif
            };

            void setDevice(int device)
            {
#if GPU_CODE
                auto status = error::status_t(devSetDevice(device));
                error::check_err(status, "Failed to set device");
#endif
            };

            void streamCreate(simbiStream_t* stream)
            {
#if GPU_CODE
                auto status = error::status_t(devStreamCreate(stream));
                error::check_err(status, "Failed to create stream");
#endif
            };

            void streamDestroy(simbiStream_t stream)
            {
#if GPU_CODE
                auto status = error::status_t(devStreamDestroy(stream));
                error::check_err(status, "Failed to destroy stream");
#endif
            };

            void streamSynchronize(simbiStream_t stream)
            {
#if GPU_CODE
                auto status = error::status_t(devStreamSynchronize(stream));
                error::check_err(status, "Failed to synchronize stream");
#endif
            };

            void streamWaitEvent(
                simbiStream_t stream,
                devEvent_t event,
                unsigned int flags
            )
            {
#if GPU_CODE
                auto status =
                    error::status_t(devStreamWaitEvent(stream, event, flags));
                error::check_err(status, "Failed to wait for event");
#endif
            };

            void streamQuery(simbiStream_t stream, int* status)
            {
#if GPU_CODE
                auto stat = error::status_t(devStreamQuery(stream));
                // *status   = stat;
                error::check_err(stat, "Failed to query stream");
#endif
            };

            void memcpyAsync(
                void* dst,
                const void* src,
                size_t bytes,
                simbiMemcpyKind kind,
                simbiStream_t stream
            )
            {
#if GPU_CODE
                auto status = error::status_t(
                    devMemcpyAsync(dst, src, bytes, kind, stream)
                );
                error::check_err(status, "Failed to copy asynchronously");
#endif
            };

            void enablePeerAccess(int device, unsigned int flags)
            {
#if GPU_CODE
                auto status =
                    error::status_t(devEnablePeerAccess(device, flags));
                error::check_err(status, "Failed to enable peer access");
#endif
            };

            void peerCopyAsync(
                void* dst,
                int dst_device,
                const void* src,
                int src_device,
                size_t bytes,
                simbiStream_t stream
            )
            {
#if GPU_CODE
                auto status = error::status_t(devMemcpyPeerAsync(
                    dst,
                    dst_device,
                    src,
                    src_device,
                    bytes,
                    stream
                ));
                error::check_err(status, "Failed to copy peer asynchronously");
#endif
            };

            void memcpyFromSymbol(void* dst, const void* symbol, size_t count)
            {
#if GPU_CODE
                auto status =
                    error::status_t(devMemcpyFromSymbol(dst, symbol, count));
                error::check_err(status, "Failed to copy from symbol");
#endif
            }

            void hostRegister(void* ptr, size_t size, unsigned int flags)
            {
#if GPU_CODE
                auto status =
                    error::status_t(devHostRegister(ptr, size, flags));
                error::check_err(status, "Failed to register host memory");
#endif
            }

            void hostUnregister(void* ptr)
            {
#if GPU_CODE
                auto status = error::status_t(devHostUnregister(ptr));
                error::check_err(status, "Failed to unregister host memory");
#endif
            }

            void asyncCopyHostToDevice(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            )
            {
#if GPU_CODE
                auto status = error::status_t(devMemcpyAsync(
                    dst,
                    src,
                    bytes,
                    devMemcpyHostToDevice,
                    stream
                ));
                error::check_err(status, "Failed to copy host to device");
#endif
            }

            void asyncCopyDeviceToHost(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            )
            {
#if GPU_CODE
                auto status = error::status_t(devMemcpyAsync(
                    dst,
                    src,
                    bytes,
                    devMemcpyDeviceToHost,
                    stream
                ));
                error::check_err(status, "Failed to copy device to host");
#endif
            }

            void alignedMalloc(void** ptr, size_t size)
            {
#if GPU_CODE
                auto status = error::status_t(devMallocHost(ptr, size));
                error::check_err(status, "Failed to allocate aligned memory");
#endif
            }

            void launchKernel(
                devFunction_t function,
                dim3 grid,
                dim3 block,
                void** args,
                size_t shared_mem,
                simbiStream_t stream
            )
            {
#if GPU_CODE
                auto status = error::status_t(devLaunchKernel(
                    function,
                    grid,
                    block,
                    args,
                    shared_mem,
                    stream
                ));
                error::check_err(status, "Failed to launch kernel");
#endif
            }

        }   // namespace api

    }   // namespace gpu

}   // namespace simbi
