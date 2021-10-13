#ifndef DEVICE_API_H
#define DEVICE_API_H

#include "gpu_error_check.h"
#include "shapeshift.h"
#include "launch_error.h"
namespace loup
{
	GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx_x() {
    #ifdef GPU_DEV_CODE
    	return threadIdx.x + blockIdx.x * blockDim.x;
    #else
    	return 0;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx_y() {
    #ifdef GPU_DEV_CODE
    	return threadIdx.y + blockIdx.y * blockDim.y;
    #else
    	return 0;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadIdx_z() {
    #ifdef GPU_DEV_CODE
    	return threadIdx.z + blockIdx.z * blockDim.z;
    #else
    	return 0;
    #endif
    }


    GPU_CALLABLE_INLINE
    unsigned int globalThreadXCount() {
    #ifdef GPU_DEV_CODE
    	return blockDim.x * gridDim.x;
    #else
    	return 1;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadYCount() {
    #ifdef GPU_DEV_CODE
    	return blockDim.y * gridDim.y;
    #else
    	return 1;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalThreadZCount() {
    #ifdef GPU_DEV_CODE
    	return blockDim.z * gridDim.z;
    #else
    	return 1;
    #endif
    }


    GPU_CALLABLE_INLINE
    unsigned int globalBlockXCount() {
    #ifdef GPU_DEV_CODE
    	return gridDim.x;
    #else
    	return 1;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockYCount() {
    #ifdef GPU_DEV_CODE
    	return gridDim.y;
    #else
    	return 1;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockZCount() {
    #ifdef GPU_DEV_CODE
    	return gridDim.z;
    #else
    	return 1;
    #endif
    }


    GPU_CALLABLE_INLINE
    unsigned int localThreadIdx_x() {
    #ifdef GPU_DEV_CODE
    	return threadIdx.x;
    #else
    	return 0;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadIdx_y() {
    #ifdef GPU_DEV_CODE
    	return threadIdx.y;
    #else
    	return 0;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadIdx_z() {
    #ifdef GPU_DEV_CODE
    	return threadIdx.z;
    #else
    	return 0;
    #endif
    }


    GPU_CALLABLE_INLINE
    unsigned int localThreadCount_x() {
    #ifdef GPU_DEV_CODE
    	return blockDim.x;
    #else
    	return 1;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadCount_y() {
    #ifdef GPU_DEV_CODE
    	return blockDim.y;
    #else
    	return 1;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int localThreadCount_z() {
    #ifdef GPU_DEV_CODE
    	return blockDim.z;
    #else
    	return 1;
    #endif
    }


    GPU_CALLABLE_INLINE
    unsigned int globalBlockIdx_x() {
    #ifdef GPU_DEV_CODE
    	return blockIdx.x;
    #else
    	return 0;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockIdx_y() {
    #ifdef GPU_DEV_CODE
    	return blockIdx.y;
    #else
    	return 0;
    #endif
    }

    GPU_CALLABLE_INLINE
    unsigned int globalBlockIdx_z() {
    #ifdef GPU_DEV_CODE
    	return blockIdx.z;
    #else
    	return 0;
    #endif
    }


    GPU_LAMBDA
    void synchronize() {
    #ifdef GPU_DEV_CODE
        __syncthreads();
    #endif
    }
}

#endif