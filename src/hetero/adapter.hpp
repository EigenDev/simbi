#ifndef HETERO_ADAPTER_HPP
#define HETERO_ADAPTER_HPP

#include "compat.hpp"
#include "core/resource_types.hpp"
#include "hetero/config.hpp"

#ifdef CUDA_ENABLED
#include "detail/cuda_impl.hpp"
#elif defined(HIP_ENABLED)
#include "detail/hip_impl.hpp"
#elif defined(SYCL_ENABLED)
#include "detail/sycl_impl.hpp"
#elif defined(METAL_ENABLED)
#include "detail/metal_impl.hpp"
#else
#include "detail/cpu_impl.hpp"
#endif

namespace simbi::hetero {

    using device        = device_adapter_t<default_backend_t>;
    using device_memory = device_memory_t<default_backend_t>;
    using stream        = stream_t<default_backend_t>;
    using event         = event_t<default_backend_t>;
    using memory        = device_memory_t<default_backend_t>;
    template <typename T>
    using managed_vector = device_vector_t<default_backend_t, T>;

    template <typename T>
    using vector = device_vector_t<default_backend_t, T>;

    namespace info {
        constexpr auto backend_name   = device::backend_name();
        constexpr auto supports_async = device::supports_async_operations();
        constexpr auto supports_peer_access = device::supports_peer_access();
    }   // namespace info

}   // namespace simbi::hetero

#endif
