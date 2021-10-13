#ifndef CALLABLES_H
#define CALLABLES_H

#ifdef __HIPCC__
#define GPU_DEV_CODE 1
#else
#define GPU_DEV_CODE 
#endif

#ifndef GPU_DEV_CODE

#define GPU_LAUNCHABLE                 
#define GPU_LAMBDA                     
#define GPU_CALLABLE               
#define GPU_CALLABLE_INLINE        inline
#define GPU_CALLABLE_MEMBER        
#define GPU_CALLABLE_INLINE_MEMBER inline
//  OMP part
//  declaration of a function
typedef unsigned int uint;
typedef int loupStream_t;
struct uint3
{
    uint3()  = default;
    ~uint3() = default;
    uint x;
    uint y;
    uint z;

    uint3(uint x = 1, uint y = 1, uint z = 1)
    :
    x(x),
    y(y),
    z(z)
    {
        
    }

};

typedef uint3 dim3;
#define __global__
#define __host__
#define __device__
#define __shared__ static
inline void hipMalloc(void** ptr , size_t size) {
(*ptr) = (void *) new char[size];
}
template <class T> inline void hipFree(T * ptr) {delete [] ptr; }
inline void* hipMemcpy( void* dest , void* src , size_t size, hipMemcpyType tt) {
return (memcpy(dest , src , size ));
}

#define GPU_KERNEL( name , ... ) void name##_OMP ( dim3 threadIdx , __VA_ARGS__ )
//  call of a function 
#define GPU_KERNEL_LAUNCH( name , gridDim , blockDim , sharedBytes, streamId, ... ) \
_Pragma ("omp parallel for") \
for (unsigned int omp=0; omp <blockDim.x; omp++) {\
dim3 threadIdx(omp ,0,0); \
name##_OMP ( threadIdx , __VA_ARGS__ ); \
}
#else
#include "hip/hip_runtime.h"
typedef hipStream_t loupStream_t;
#define GPU_LAUNCHABLE                 __global__
#define GPU_LAMBDA                     __device__
#define GPU_CALLABLE               __host__ __device__
#define GPU_CALLABLE_INLINE        __host__ __device__ inline
#define GPU_CALLABLE_MEMBER        __host__ __device__
#define GPU_CALLABLE_INLINE_MEMBER __host__ __device__ inline
//  GPU part
//  declaration of a function 
#define GPU_KERNEL( name , ... ) __global__ void name ( __VA_ARGS__ )
//  call of a function 
#define GPU_KERNEL_LAUNCH( name , gridDim , blockDim , sharedBytes, streamId,  ... ) \
    hipLaunchKernelGGL(name, gridDim, blockDim, sharedBytes, streamId,  __VA_ARGS__ )
#endif



#endif