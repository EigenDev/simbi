/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 * @file       device_api.hpp
 * @brief      houses the gpu device-specific api calls
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */
#ifndef DEVICE_API_HPP
#define DEVICE_API_HPP

#include "build_options.hpp"   // for blockDim, threadIdx, GPU_CALLABLE_INLINE
#include <cstddef>             // for size_t
#include <omp.h>               // for omp_get_thread_num
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
                return anyGpuGetErrorString(anyGpuError_t(status));
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

            constexpr inline bool is_err(status_t status)
            {
                return status != status_t::success;
            }

            inline void check_err(
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
            void copyDevToHost(void* to, const void* from, size_t bytes);
            void copyDevToDev(void* to, const void* from, size_t bytes);
            void gpuMalloc(void* obj, size_t bytes);
            void gpuMallocManaged(void* obj, size_t bytes);
            void gpuFree(void* obj);
            void gpuEventSynchronize(anyGpuEvent_t a);
            void gpuEventCreate(anyGpuEvent_t* a);
            void gpuEventDestroy(anyGpuEvent_t a);
            void gpuEventRecord(anyGpuEvent_t a);
            void
            gpuEventElapsedTime(float* time, anyGpuEvent_t a, anyGpuEvent_t b);
            void getDeviceCount(int* devCount);
            void getDeviceProperties(anyGpuProp_t* props, int i);
            void gpuMemset(void* obj, int val, size_t bytes);

            template <global::Platform P = global::BuildPlatform>
            inline void deviceSynch()
            {
                if constexpr (P == global::Platform::GPU) {
                    auto status = error::status_t(anyGpuDeviceSynchronize());
                    error::check_err(status, "Failed to synch device(s)");
                }
                else {
                    return;
                }
            }

            template <global::Platform P = global::BuildPlatform>
            GPU_DEV_INLINE void synchronize()
            {
                if constexpr (P == global::Platform::GPU) {
                    __syncthreads();
                }
                else {
                    return;
                }
            }
        }   // namespace api

    }   // namespace gpu

    GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx()
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

    GPU_CALLABLE_INLINE
    unsigned int globalThreadCount()
    {
        return blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z *
               gridDim.z;
    }

    GPU_CALLABLE_INLINE
    unsigned int get_ii_in2D()
    {
        if constexpr (global::col_maj) {
            return blockDim.y * blockIdx.y + threadIdx.y;
        }
        return blockDim.x * blockIdx.x + threadIdx.x;
    }

    GPU_CALLABLE_INLINE
    unsigned int get_jj_in2D()
    {
        if constexpr (global::col_maj) {
            return blockDim.x * blockIdx.x + threadIdx.x;
        }
        return blockDim.y * blockIdx.y + threadIdx.y;
    }

    template <global::Platform P = global::BuildPlatform>
    GPU_CALLABLE_INLINE unsigned int get_tx()
    {
        if constexpr (P == global::Platform::GPU) {
            if constexpr (global::col_maj) {
                return threadIdx.y;
            }
            return threadIdx.x;
        }
        else {
            return 0;
        }
    }

    template <global::Platform P = global::BuildPlatform>
    GPU_CALLABLE_INLINE unsigned int get_ty()
    {
        if constexpr (P == global::Platform::GPU) {
            if constexpr (global::col_maj) {
                return threadIdx.x;
            }
            return threadIdx.y;
        }
        else {
            return 0;
        }
    }

    template <global::Platform P = global::BuildPlatform>
    GPU_CALLABLE_INLINE unsigned int get_threadId()
    {
#if GPU_CODE
        return blockDim.x * blockDim.y * threadIdx.z +
               blockDim.x * threadIdx.y + threadIdx.x;
#else
        if (global::use_omp) {
            return omp_get_thread_num();
        }
        else {
            return std::hash<std::thread::id>{}(std::this_thread::get_id());
        }
#endif
    }

    GPU_DEV_INLINE
    void synchronize()
    {
#if GPU_CODE
        __syncthreads();
#endif
    }
}   // namespace simbi

#endif