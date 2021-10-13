#ifndef BUILD_OPTIONS_HPP
#define BUILD_OPTIONS_HPP

#define GPU_CODE 0
enum class Platform: int {CPU = 0, GPU = 1};
#if GPU_CODE
#include "hip/hip_runtime.h"
constexpr Platform BuildPlatform = Platform::GPU;
// typedef float real;
typedef hipStream_t simbiStream_t;
#define GPU_DEV                        __device__
#define GPU_DEV_INLINE                 __device__ inline
#define GPU_LAUNCHABLE                 __global__
#define GPU_LAMBDA                     __device__
#define GPU_CALLABLE                   __host__ __device__
#define GPU_CALLABLE_INLINE            __host__ __device__ inline
#define GPU_CALLABLE_MEMBER            __host__ __device__
#define GPU_CALLABLE_INLINE_MEMBER     __host__ __device__ inline
#define GPU_SHARED_DYN                 extern __shared__
#define GPU_CAPTURE                    [=, *this]
//  declaration of a function 
#define GPU_KERNEL( name , ... ) __global__ void name ( __VA_ARGS__ )
//  call of a function 
#define GPU_KERNEL_LAUNCH( name , gridDim , blockDim , sharedBytes, streamId,  ... ) \
    hipLaunchKernelGGL(name, gridDim, blockDim, sharedBytes, streamId,  __VA_ARGS__ )
#else
#include <cstring>
#include <string>
#include <stdint.h>
#include <stdexcept>
constexpr Platform BuildPlatform = Platform::CPU;
// typedef double real;

#define GPU_DEV                         
#define GPU_LAUNCHABLE                 
#define GPU_LAMBDA                     
#define GPU_CALLABLE               
#define GPU_CALLABLE_INLINE          inline
#define GPU_CALLABLE_MEMBER        
#define GPU_CALLABLE_INLINE_MEMBER   inline
#define GPU_DEV_INLINE               inline 
#define __shared__ static
#define GPU_SHARED_DYN                 extern __shared__
#define GPU_CAPTURE                    [this]
// Define a bunch of ghost functions to make the compiler
// happy and avoid a bunch if #ifdefs throughout code
// Don't get too madd at me... I'm a physicist, not a programmer :-)
namespace simbi {
    // struct BuildOptions
    // {
    //     constexpr static bool gpu_on = static_cast<bool>(GPU_CODE);
    // };

    struct u3
    {
        ~u3() = default;
        uint32_t x;
        uint32_t y;
        uint32_t z;

        u3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1)
        :
        x(x),
        y(y),
        z(z)
        {
            
        }

    };

    typedef int simbiStream_t;
    typedef int hipError_t; 
    enum hipMemcpyType {
        hipMemcpyHostToDevice,
        hipMemcpyDeviceToDevice,
        hipMemcpyDeviceToHost
    };
    typedef u3 dim3;
    inline dim3 blockIdx(1, 1, 1); 
    inline dim3 blockDim(1, 1, 1); 
    inline dim3 gridDim(1, 1, 1); 
    inline dim3 threadIdx(1, 1, 1);
    constexpr Platform pyBuild = BuildPlatform;

    inline int hipMalloc(void** ptr , size_t size) {
        return 0;
    }
    template <class T> inline int hipFree(T * ptr) {return 0;}
    inline int hipMemcpy( void* dest , const void* src , size_t size, hipMemcpyType tt) {
        return 0;
    }

    inline void __syncthreads() { return;}
    inline hipError_t hipDeviceSynchronize() { return 0;}

    template <typename T>
    inline std::string hipGetErrorString(T) { return std::string("nothing odd here.");}
} // end simbi

using namespace simbi;

#endif





#endif
