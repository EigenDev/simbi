#ifndef HET_TRAITS_HPP
#define HET_TRAITS_HPP

#include "types.hpp"

#include <type_traits>

namespace simbi::het {

    // -------------------------------------------------------------------------
    // backend tag types
    // -------------------------------------------------------------------------
    struct cpu_backend_t {
    };
    struct cuda_backend_t {
    };
    struct hip_backend_t {
    };
    struct sycl_backend_t {
    };

    // -------------------------------------------------------------------------
    // trait queries
    // -------------------------------------------------------------------------

    template <typename Backend>
    struct backend_traits {
        static constexpr bool is_valid = false;
    };

    template <>
    struct backend_traits<cpu_backend_t> {
        static constexpr bool is_valid             = true;
        static constexpr backend_type_t type       = backend_type_t::cpu;
        static constexpr const char* name          = "cpu";
        static constexpr bool supports_async       = false;
        static constexpr bool supports_peer_access = false;
        static constexpr int warp_size             = 1;
    };

#ifdef CUDA_ENABLED
    template <>
    struct backend_traits<cuda_backend_t> {
        static constexpr bool is_valid             = true;
        static constexpr backend_type_t type       = backend_type_t::cuda;
        static constexpr const char* name          = "cuda";
        static constexpr bool supports_async       = true;
        static constexpr bool supports_peer_access = true;
        static constexpr int warp_size             = 32;
    };
#endif

#ifdef HIP_ENABLED
    template <>
    struct backend_traits<hip_backend_t> {
        static constexpr bool is_valid             = true;
        static constexpr backend_type_t type       = backend_type_t::hip;
        static constexpr const char* name          = "hip";
        static constexpr bool supports_async       = true;
        static constexpr bool supports_peer_access = true;
        static constexpr int warp_size = 64;   // usually, check arch
    };
#endif

    // -------------------------------------------------------------------------
    // concept helpers
    // -------------------------------------------------------------------------

    template <typename T>
    constexpr bool is_gpu_backend_v =
        std::is_same_v<T, cuda_backend_t> || std::is_same_v<T, hip_backend_t>;

}   // namespace simbi::het

#endif   // HETERO_TRAITS_HPP
