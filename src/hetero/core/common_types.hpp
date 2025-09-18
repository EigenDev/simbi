#ifndef HETERO_CORE_COMMON_TYPES_HPP
#define HETERO_CORE_COMMON_TYPES_HPP

#include "backend_traits.hpp"

#include <cstdint>

namespace simbi::hetero {

    struct dim3_t {
        std::uint32_t x, y, z;

        constexpr dim3_t(
            std::uint32_t x_ = 1,
            std::uint32_t y_ = 1,
            std::uint32_t z_ = 1
        )
            : x(x_), y(y_), z(z_)
        {
        }

        constexpr std::uint64_t volume() const noexcept
        {
            return static_cast<std::uint64_t>(x) * y * z;
        }
    };

    enum class memory_kind_t {
        host_to_device,
        device_to_host,
        device_to_device,
        host_to_host
    };

    enum class device_type_t {
        cpu,
        cuda_gpu,
        hip_gpu,
        sycl_device,
        metal_gpu
    };

    template <typename backend_t>
    struct backend_info_t {
        static_assert(False<backend_t>{}, "backend info not specialized");
    };

    template <>
    struct backend_info_t<cpu_backend_t> {
        static constexpr device_type_t device_type = device_type_t::cpu;
        static constexpr const char* name          = "cpu";
        static constexpr bool supports_async       = false;
        static constexpr bool supports_peer_access = false;
    };

#ifdef CUDA_ENABLED
    template <>
    struct backend_info_t<cuda_backend_t> {
        static constexpr device_type_t device_type = device_type_t::cuda_gpu;
        static constexpr const char* name          = "cuda";
        static constexpr bool supports_async       = true;
        static constexpr bool supports_peer_access = true;
    };
#endif

#ifdef HIP_ENABLED
    template <>
    struct backend_info_t<hip_backend_t> {
        static constexpr device_type_t device_type = device_type_t::hip_gpu;
        static constexpr const char* name          = "hip";
        static constexpr bool supports_async       = true;
        static constexpr bool supports_peer_access = true;
    };
#endif

    template <typename backend_t>
    constexpr auto backend_name = backend_info_t<backend_t>::name;

    template <typename backend_t>
    constexpr auto supports_async = backend_info_t<backend_t>::supports_async;

    template <typename backend_t>
    constexpr auto supports_peer_access =
        backend_info_t<backend_t>::supports_peer_access;

}   // namespace simbi::hetero

#endif   // HETERO_CORE_COMMON_TYPES_HPP
