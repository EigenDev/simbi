#ifndef GPU_ERROR_H
#define GPU_ERROR_H

#ifdef GPU_DEV_CODE
#define hipCheckErrors(msg) \
do { \
    hipError_t __err = hipGetLastError(); \
    if (__err != hipSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, hipGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
} while (0)
#else
#define hipCheckErrors(msg) 0
#endif


namespace loup{
    
    enum Error_t {
        success = 0,
        hipError = 1
    };

    inline Error_t deviceSynchronize() 
    {
    #ifdef HEMI_CUDA_COMPILER
            hipDeviceSynchronize();
            hipCheckErrors("Failed at synchronizing device.")
                return hipError; 
    #endif
            return success;
    }
}
#endif