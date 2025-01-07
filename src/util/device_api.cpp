#include "device_api.hpp"
#include <memory>   // for allocator

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

            void copyDevToHost(void* to, const void* from, size_t bytes)
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

            void copyDevToDev(void* to, const void* from, size_t bytes)
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

            void gpuMalloc(void* obj, size_t elements)
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

            void gpuMallocManaged(void* obj, size_t elements)
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

            void gpuFree(void* obj)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devFree(obj));
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
                    simbi::gpu::error::status_t(devMemset(obj, val, bytes));
                simbi::gpu::error::check_err(status, "Failed to memset");
#endif
            };

            void gpuEventSynchronize(devEvent_t a)
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

            void gpuEventCreate(devEvent_t* a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devEventCreate(a));
                simbi::gpu::error::check_err(status, "Failed to create event");
#endif
            };

            void gpuEventDestroy(devEvent_t a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devEventDestroy(a));
                simbi::gpu::error::check_err(status, "Failed to destroy event");
#endif
            };

            void gpuEventRecord(devEvent_t a)
            {
#if GPU_CODE
                auto status = simbi::gpu::error::status_t(devEventRecord(a));
                simbi::gpu::error::check_err(status, "Failed to record event");
#endif
            };

            void gpuEventElapsedTime(float* time, devEvent_t a, devEvent_t b)
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

            void gpuMcFromSymbol(void* dst, const void* symbol, size_t count)
            {
#if GPU_CODE
                auto status =
                    error::status_t(devMemcpyFromSymbol(dst, symbol, count));
                error::check_err(status, "Failed to copy from symbol");
#endif
            }

            void gpuCreateProgram(
                devProgram_t* program,
                const char* source,
                const char* prog_name,
                int num_options,
                const char** options,
                const char** option_vals
            )
            {
#if GPU_CODE
                auto status = error::status_t(devCreateProgram(
                    program,
                    source,
                    prog_name,
                    num_options,
                    options,
                    option_vals
                ));
                error::check_err(status, "Failed to create program");
#endif
            }

            int gpuProgram(
                devProgram_t program,
                int num_options,
                const char** options
            )
            {
#if GPU_CODE
                auto status = error::status_t(
                    devCompileProgram(program, num_options, options)
                );
                error::check_err(status, "Failed to program");
#endif
                return 0;
            }

            void gpuGetProgramLogSize(devProgram_t program, size_t* size)
            {
#if GPU_CODE
                auto status =
                    error::status_t(devGetProgramLogSize(program, size));
                error::check_err(status, "Failed to get program log size");
#endif
            }

            void gpuGetProgramLog(devProgram_t program, char* log)
            {
#if GPU_CODE
                auto status = error::status_t(devGetProgramLog(program, log));
                error::check_err(status, "Failed to get program log");
#endif
            }

            void gpuGetProgramIRSize(devProgram_t program, size_t* size)
            {
#if GPU_CODE
                auto status = error::status_t(devGetIRSize(program, size));
                error::check_err(status, "Failed to get program IR size");
#endif
            }

            void gpuGetProgramIR(devProgram_t program, char* ir)
            {
#if GPU_CODE
                auto status = error::status_t(devGetIR(program, ir));
                error::check_err(status, "Failed to get program IR");
#endif
            }

            void gpuModuleLoadData(devModule_t* module, const char* ir)
            {
#if GPU_CODE
                auto status = error::status_t(devLoadModule(module, ir));
                error::check_err(status, "Failed to load module");
#endif
            }

            void gpuModuleUnload(devModule_t module)
            {
#if GPU_CODE
                auto status = error::status_t(devUnloadModule(module));
                error::check_err(status, "Failed to unload module");
#endif
            }

            void gpuLoadModule(devModule_t* module, const char* ir)
            {
#if GPU_CODE
                auto status = error::status_t(devLoadModule(module, ir));
                error::check_err(status, "Failed to load module");
#endif
            }

            void gpuGetFunction(
                devFunction_t* function,
                devModule_t module,
                const char* name
            )
            {
#if GPU_CODE
                auto status =
                    error::status_t(devGetFunction(function, module, name));
                error::check_err(status, "Failed to get function");
#endif
            }

        }   // namespace api

    }   // namespace gpu

}   // namespace simbi