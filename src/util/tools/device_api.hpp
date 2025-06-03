/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            device_api.hpp
 *  * @brief           API for device-side operations
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
#ifndef DEVICE_API_HPP
#define DEVICE_API_HPP

#include "build_options.hpp"   // for blockDim, threadIdx, STATIC
#include <cstddef>             // for size_t
#include <stdexcept>           // for runtime_error
#include <string>              // for allocator, operator+, char_traits, to_s...
#include <thread>              // for get_id, hash, thread

namespace simbi {
    namespace gpu {
        //===============================
        // Some Error Handling Utilities
        //================================
        namespace error {
            enum class status_t {
                success = 0,
                gpuError
            };

            inline ::std::string describe(status_t status)
            {
#if GPU_CODE
                return devGetErrorString(devError_t(status));
#else
                return "Simulated GPU Error: " +
                       std::to_string(static_cast<int>(status));
#endif
            }

            class runtime_error : public ::std::runtime_error
            {
              public:
                ///@cond
                // Just the error code? Okay, no problem
                runtime_error(status_t error_code)
                    : ::std::runtime_error(
                          describe(error_code) + " at  " __FILE__ ":" +
                          std::to_string(__LINE__)
                      ),
                      internal_code(error_code)
                {
                }

                // Human-readable error logic
                runtime_error(
                    status_t error_code,
                    const ::std::string& what_arg
                )
                    : ::std::runtime_error(
                          what_arg + ": " + describe(error_code) +
                          " at  " __FILE__ ":" + std::to_string(__LINE__)
                      ),
                      internal_code(error_code)
                {
                }

                ///@endcond

                /**
                 * Obtain the GPU status code which resulted in this error being
                 * thrown.
                 */
                status_t code() const { return internal_code; }

              private:
                status_t internal_code;
            };

            constexpr bool is_err(status_t status)
            {
                return status != status_t::success;
            }

            constexpr void check_err(
                status_t status,
                const ::std::string& message
            ) noexcept(false)
            {
                if (is_err(status)) {
                    throw runtime_error(status, message);
                }
            }

        }   // namespace error

        namespace api {
            void copyHostToDevice(void* to, const void* from, size_t bytes);
            void copyDeviceToHost(void* to, const void* from, size_t bytes);
            void copyDeviceToDevice(void* to, const void* from, size_t bytes);
            void malloc(void* obj, size_t bytes);
            void mallocManaged(void* obj, size_t bytes);
            void free(void* obj);
            void memcpyFromSymbol(void* dst, const void* symbol, size_t count);
            void eventSynchronize(devEvent_t a);
            void eventCreate(devEvent_t* a);
            void eventDestroy(devEvent_t a);
            void eventRecord(devEvent_t a);
            void eventElapsedTime(float* time, devEvent_t a, devEvent_t b);
            void getDeviceCount(int* devCount);
            void getDeviceProperties(devProp_t* props, int i);
            void memset(void* obj, int val, size_t bytes);
            void deviceSynch();
            void setDevice(int device);
            void streamCreate(simbiStream_t* stream);
            void streamDestroy(simbiStream_t stream);
            void streamSynchronize(simbiStream_t stream);
            void streamWaitEvent(
                simbiStream_t stream,
                devEvent_t event,
                unsigned int flags = 0
            );
            void
            prefetchToDevice(const void* obj, size_t bytes, int device = 0);
            void streamQuery(simbiStream_t stream, int* status);
            void peerCopyAsync(
                void* dst,
                int dst_device,
                const void* src,
                int src_device,
                size_t bytes,
                simbiStream_t stream
            );
            void memcpyAsync(
                void* dst,
                const void* src,
                size_t bytes,
                simbiMemcpyKind kind,
                simbiStream_t stream
            );
            void enablePeerAccess(int device, unsigned int flags = 0);
            void asyncCopyHostToDevice(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            );
            void asyncCopyDeviceToHost(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            );
            void asyncCopyDeviceToDevice(
                void* dst,
                const void* src,
                size_t bytes,
                simbiStream_t stream
            );
            void hostRegister(void* ptr, size_t size, unsigned int flags);
            void hostUnregister(void* ptr);

            void alignedMalloc(void** ptr, size_t size);
            void launchKernel(
                devFunction_t function,
                dim3 grid,
                dim3 block,
                void** args,
                size_t shared_mem,
                simbiStream_t stream
            );

            template <typename T>
            DEV T atomicMin(T* address, T val)
            {
#if GPU_CODE
                // if floating type, use custom atomic min
                if constexpr (std::is_floating_point_v<T>) {
                    return devAtomicMinReal(address, val);
                }
                else {
                    return devAtomicMin(address, val);
                }
#else
                // Fallback for non-GPU code
                return *address = std::min(*address, val);
#endif
            };

            template <typename T>
            DEV T atomicAdd(T* address, T val)
            {
#if GPU_CODE
                // if integral type, use standard atomicAdd
                if constexpr (std::is_integral_v<T>) {
                    return devAtomicAddInt(address, val);
                }
                else {
                    return devAtomicAddReal(address, val);
                }
#else
                return *address += val;   // Fallback for non-GPU code
#endif
            };

            DEV inline void synchronize()
            {
#if GPU_CODE
                __syncthreads();
#endif
            };

        }   // namespace api

    }   // namespace gpu

    STATIC
    unsigned int global_thread_idx()
    {
        if constexpr (global::on_gpu) {
            return (
                (blockIdx.z * blockDim.z + threadIdx.z) * blockDim.x *
                    gridDim.x * blockDim.y * gridDim.y +
                (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x *
                    gridDim.x +
                blockIdx.x * blockDim.x + threadIdx.x
            );
        }
        else {
            return 0;
        }
    }

    STATIC
    unsigned int globalThreadCount()
    {
        return blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z *
               gridDim.z;
    }

    STATIC unsigned int get_thread_id()
    {
#if GPU_CODE
        return blockDim.x * blockDim.y * threadIdx.z +
               blockDim.x * threadIdx.y + threadIdx.x;
#else
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    }

    STATIC unsigned int get_block_id()
    {
        if constexpr (global::on_gpu) {
            return blockIdx.x + blockIdx.y * gridDim.x +
                   blockIdx.z * gridDim.x * gridDim.y;
        }
        else {
            return 0;
        }
    }

    STATIC unsigned int get_threads_per_block()
    {
        if constexpr (global::on_gpu) {
            return blockDim.x * blockDim.y * blockDim.z;
        }
        else {
            return 1;
        }
    }
}   // namespace simbi

#endif
