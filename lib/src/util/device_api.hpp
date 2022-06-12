#ifndef DEVICE_API_HPP
#define DEVOCE_API_HPP


#ifndef DEVICE_API_HPP
#define DEVICE_API_HPP

#include "build_options.hpp"
#include <string>
#include <stdexcept>
#define NAME_OF(variable) ((decltype(&variable))nullptr, #variable)

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
                    ::std::runtime_error(describe(error_code)),
                    code_(error_code)
                { }
                // Human-readable error logic
                runtime_error(status_t error_code, const ::std::string& what_arg) :
                    ::std::runtime_error(what_arg + ": " + describe(error_code)),
                    code_(error_code)
                { }
                ///@endcond

                /**
                 * Obtain the GPU status code which resulted in this error being thrown.
                 */
                status_t code() const { return code_; }

            private:
                status_t code_;
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
            void gpuFree(void *obj);

            void gpuMemset(void *obj, int val, size_t bytes);

            inline GPU_DEV void synchronize() {
                if constexpr(BuildPlatform == Platform::GPU)
                {
                    __syncthreads();
                } else {
                    return;
                }
            }


            inline void deviceSynch() {
                if constexpr(BuildPlatform == Platform::GPU)
                {
                    auto status = error::status_t(anyGpuDeviceSynchronize());
                    error::check_err(status, "Failed to synch device(s)");
                } else {
                    return;
                }
            }
        } // namepsace api

    } // namespace gpu
    
    GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx_x() {
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
    unsigned int globalThreadIdx_y() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return threadIdx.y + blockIdx.y * blockDim.y;
    else
    	return 0;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx_z() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return threadIdx.z + blockIdx.z * blockDim.z;
    else
    	return 0;
    
    }


    GPU_CALLABLE_INLINE
    unsigned int globalThreadXCount() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
    else
    	return 1;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadYCount() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockDim.y * gridDim.y;
    else
    	return 1;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadZCount() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockDim.z * gridDim.z;
    else
    	return 1;
    
    }


    GPU_CALLABLE_INLINE
    unsigned int globalBlockXCount() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return gridDim.x;
    else
    	return 1;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockYCount() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return gridDim.y;
    else
    	return 1;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockZCount() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return gridDim.z;
    else
    	return 1;
    
    }


    GPU_CALLABLE_INLINE
    unsigned int localThreadIdx_x() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return threadIdx.x;
    else
    	return 0;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadIdx_y() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return threadIdx.y;
    else
    	return 0;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadIdx_z() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return threadIdx.z;
    else
    	return 0;
    
    }


    GPU_CALLABLE_INLINE
    unsigned int localThreadCount_x() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockDim.x;
    else
    	return 1;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadCount_y() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockDim.y;
    else
    	return 1;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadCount_z() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockDim.z;
    else
    	return 1;
    
    }


    GPU_CALLABLE_INLINE
    unsigned int globalBlockIdx_x() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockIdx.x;
    else
    	return 0;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockIdx_y() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockIdx.y;
    else
    	return 0;
    
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockIdx_z() {
    if constexpr(BuildPlatform == Platform::GPU)
    	return blockIdx.z;
    else
    	return 0;
    
    }


    GPU_DEV_INLINE
    void synchronize() {
    #if GPU_CODE
    __syncthreads();
    #endif
    
    }
} // namespace simbi

#endif


#endif