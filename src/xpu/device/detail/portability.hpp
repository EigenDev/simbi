// =============================================================================
// portability.hpp
//
// platform portability macros for xpu.
// defines decorators that work across cpu/cuda/hip/sycl without
// coupling to simbi's legacy compat.hpp.
//
// usage:
//   XPU_HOST_DEVICE int add(int a, int b) { return a + b; }
//   XPU_DEVICE void kernel_helper() { /* device only */ }
// =============================================================================
#pragma once

// =============================================================================
// detect cuda compilation
// =============================================================================

#if defined(__CUDACC__) || defined(__NVCC__)
#define XPU_COMPILING_CUDA 1
#else
#define XPU_COMPILING_CUDA 0
#endif

// =============================================================================
// detect hip compilation
// =============================================================================

#if defined(__HIP__) || defined(__HIPCC__)
#define XPU_COMPILING_HIP 1
#else
#define XPU_COMPILING_HIP 0
#endif

// =============================================================================
// detect sycl compilation
// =============================================================================

#if defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_LANGUAGE_VERSION)
#define XPU_COMPILING_SYCL 1
#else
#define XPU_COMPILING_SYCL 0
#endif

// =============================================================================
// device function decorators
// =============================================================================

#if XPU_COMPILING_CUDA || XPU_COMPILING_HIP

// cuda/hip: use native decorators
#define XPU_HOST        __host__
#define XPU_DEVICE      __device__
#define XPU_HOST_DEVICE __host__ __device__
#define XPU_GLOBAL      __global__
#define XPU_SHARED      __shared__
#define XPU_CONSTANT    __constant__
#define XPU_FORCEINLINE __forceinline__

#elif XPU_COMPILING_SYCL

// sycl: no decorators needed (handled by sycl::kernel mechanism)
#define XPU_HOST
#define XPU_DEVICE
#define XPU_HOST_DEVICE
#define XPU_GLOBAL
#define XPU_SHARED
#define XPU_CONSTANT
#define XPU_FORCEINLINE [[gnu::always_inline]]

#else

// cpu: no decorators
#define XPU_HOST
#define XPU_DEVICE
#define XPU_HOST_DEVICE
#define XPU_GLOBAL
#define XPU_SHARED
#define XPU_CONSTANT
#define XPU_FORCEINLINE inline

#endif

// =============================================================================
// runtime detection (inside device code)
// =============================================================================

#if defined(__CUDA_ARCH__)
#define XPU_IS_DEVICE_CODE 1
#elif defined(__HIP_DEVICE_COMPILE__)
#define XPU_IS_DEVICE_CODE 1
#elif defined(__SYCL_DEVICE_ONLY__)
#define XPU_IS_DEVICE_CODE 1
#else
#define XPU_IS_DEVICE_CODE 0
#endif

// =============================================================================
// convenience aliases (match common usage patterns)
// =============================================================================

// common abbreviation
#define XPU_D  XPU_DEVICE
#define XPU_H  XPU_HOST
#define XPU_HD XPU_HOST_DEVICE
#define XPU_G  XPU_GLOBAL
