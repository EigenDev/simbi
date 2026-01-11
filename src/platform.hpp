// =============================================================================
// platform.hpp
//
// compile-time platform detection and hardware constants.
//
// usage:
//   if constexpr (platform::is_cuda) { /* cuda-specific */ }
//   constexpr auto ws = platform::warp_size;
// =============================================================================

#ifndef SIMBI_PLATFORM_HPP
#define SIMBI_PLATFORM_HPP

#include <cstdint>

namespace simbi::platform {

    inline constexpr bool is_cuda =
#if defined(__CUDACC__) || defined(__NVCC__)
        true;
#else
        false;
#endif

    inline constexpr bool is_hip =
#if defined(__HIP__) || defined(__HIPCC__)
        true;
#else
        false;
#endif

    inline constexpr bool is_sycl =
#if defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_LANGUAGE_VERSION)
        true;
#else
        false;
#endif

    inline constexpr bool is_gpu = is_cuda || is_hip || is_sycl;
    inline constexpr bool is_cpu = !is_gpu;

    inline constexpr bool is_device_code =
#if defined(__CUDA_ARCH__)
        true;
#elif defined(__HIP_DEVICE_COMPILE__)
        true;
#elif defined(__SYCL_DEVICE_ONLY__)
        true;
#else
        false;
#endif

    inline constexpr std::uint64_t warp_size = is_cuda ? 32 : is_hip ? 64 : 1;

    enum class type {
        cpu,
        cuda,
        hip,
        sycl,
    };

    inline constexpr type current = is_cuda   ? type::cuda
                                    : is_hip  ? type::hip
                                    : is_sycl ? type::sycl
                                              : type::cpu;

} // namespace simbi::platform

#endif
