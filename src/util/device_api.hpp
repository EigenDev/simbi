#ifndef DEVICE_API_HPP
#define DEVICE_API_HPP

#include "build_options.hpp"
#include <stdexcept>
#include <thread>

namespace simbi
{   
    namespace detail
    {
        template<typename index_type, typename T>
        GPU_CALLABLE_INLINE
        index_type get_column(index_type idx, T width, T length = 1, index_type k = 0)
        {
            idx -= (k * width * length);
            return idx % width;
        }

        template<typename index_type, typename T>
        GPU_CALLABLE_INLINE
        index_type get_row(index_type idx, T width, T length = 1, index_type k = 0)
        {
            idx -= (k * width * length);
            return idx / width;
        }

        template<typename index_type, typename T>
        GPU_CALLABLE_INLINE
        index_type get_height(index_type idx, T width, T length)
        {
            return idx / width / length;
        }
    } // namespace detail

    namespace gpu
    {
        //===============================
        // Some Error Handling Utilities
        //================================
        namespace error
        {
            enum class status_t {
                success  = 0,
                gpuError
            };

            inline ::std::string  describe(status_t status) { 
                return anyGpuGetErrorString(anyGpuError_t(status)); 
            }
            
            class runtime_error : public ::std::runtime_error {
            public:
                ///@cond
                // Just the error code? Okay, no problem
                runtime_error(status_t error_code) :
                    ::std::runtime_error(describe(error_code) + " at  " __FILE__ ":"  + std::to_string(__LINE__)),
                    internal_code(error_code)
                { }
                // Human-readable error logic
                runtime_error(status_t error_code, const ::std::string& what_arg) :
                    ::std::runtime_error(what_arg + ": " + describe(error_code) + " at  " __FILE__ ":" + std::to_string(__LINE__)),
                    internal_code(error_code)
                { }
                ///@endcond

                /**
                 * Obtain the GPU status code which resulted in this error being thrown.
                 */
                status_t code() const { return internal_code; }

            private:
                status_t internal_code;
            };
            

            constexpr inline bool is_err(status_t status)  { return status != status_t::success; }
            

            inline void check_err(status_t status, const ::std::string& message) noexcept(false)
            {
                if (is_err(status)) { throw runtime_error(status, message); }
            }

        } // namespace error

        namespace api{
            void copyHostToDevice(void *to, const void *from, size_t bytes);
            void copyDevToHost(void *to, const void *from, size_t bytes);
            void copyDevToDev(void *to, const void *from, size_t bytes);
            void gpuMalloc(void *obj, size_t bytes);
            void gpuMallocManaged(void *obj, size_t bytes);
            void gpuFree(void *obj);
            void gpuEventSynchronize(anyGpuEvent_t a);
            void gpuEventCreate(anyGpuEvent_t *a);
            void gpuEventDestroy(anyGpuEvent_t  a);
            void gpuEventRecord(anyGpuEvent_t a);
            void gpuEventElapsedTime(float *time, anyGpuEvent_t a, anyGpuEvent_t b);
            void getDeviceCount(int *devCount);
            void getDeviceProperties(anyGpuProp_t *props, int i);
            void gpuMemset(void *obj, int val, size_t bytes);
            // void deviceSynch();

            template<Platform P = BuildPlatform>
            inline void deviceSynch() {
                if constexpr(P == Platform::GPU)
                {
                    auto status = error::status_t(anyGpuDeviceSynchronize());
                    error::check_err(status, "Failed to synch device(s)");
                } else {
                    return;
                }
            }
            template<Platform P = BuildPlatform>
            GPU_DEV_INLINE void synchronize() {
                if constexpr(P == Platform::GPU)
                {
                    __syncthreads();
                } else {
                    return;
                }
            }
        } // namepsace api

    } // namespace gpu
    
    GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return ( 
                  (blockIdx.z * blockDim.z + threadIdx.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y
                + (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x 
                +  blockIdx.x * blockDim.x + threadIdx.x
            );
    else
    	return 0;
    
    }
    GPU_CALLABLE_INLINE
    unsigned int globalThreadCount() {
        return blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z * gridDim.z;
    }

    GPU_CALLABLE_INLINE
    unsigned int get_ii_in2D(){
        if constexpr(col_maj) {
            return blockDim.y * blockIdx.y + threadIdx.y;
        }
        return blockDim.x * blockIdx.x + threadIdx.x;
    }

    GPU_CALLABLE_INLINE
    unsigned int get_jj_in2D(){
        if constexpr(col_maj) {
            return blockDim.x * blockIdx.x + threadIdx.x;
        }
        return blockDim.y * blockIdx.y + threadIdx.y;
    }

    template<Platform P = BuildPlatform>
    GPU_CALLABLE_INLINE 
    unsigned int get_tx() {
        if constexpr(P == Platform::GPU) {
            if constexpr(col_maj) {
                return threadIdx.y;
            }
            return threadIdx.x;
        } else {
            return 0;
        }
    }

    template<Platform P = BuildPlatform>
    GPU_CALLABLE_INLINE 
    unsigned int get_ty() {
        if constexpr(P == Platform::GPU) {
            if constexpr(col_maj) {
                return threadIdx.x;
            }
            return threadIdx.y;
        } else {
            return 0;
        }
    }

    template<Platform P = BuildPlatform>
    GPU_CALLABLE_INLINE
    unsigned int get_threadId() {
        #if GPU_CODE
            return blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
        #else
            if (use_omp) {
                return omp_get_thread_num();
            } else {
                return std::hash<std::thread::id>{}(std::this_thread::get_id());
            }
        #endif
    }


    GPU_DEV_INLINE
    void synchronize() {
    #if GPU_CODE
    __syncthreads();
    #endif
    }
} // namespace simbi

#endif