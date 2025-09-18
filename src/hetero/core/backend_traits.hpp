#ifndef HETERO_BACKEND_TRAITS_HPP
#define HETERO_BACKEND_TRAITS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

// include backend-specific headers conditionally
#ifdef CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef HIP_ENABLED
#include <hip/hip_runtime.h>
#endif

// [TODO] re-add SYCL, Metal headers later

template <class...>
struct False : std::bool_constant<false> {
};
namespace simbi::hetero {
    // backend tag types - empty structs for compile-time dispatch
    // I think this is called tag dispatching?
    struct cpu_backend_t {
    };
    struct cuda_backend_t {
    };
    struct hip_backend_t {
    };
    struct sycl_backend_t {
    };
    struct metal_backend_t {
    };

    template <typename backend_t>
    struct backend_traits_t {
        static_assert(False<backend_t>{}, "unsupported backend type");
    };

    // cpu backend traits
    template <>
    struct backend_traits_t<cpu_backend_t> {
        // cpu uses dummy types since there's no real GPU concepts
        struct stream_handle_t {
        };
        struct event_handle_t {
        };
        struct device_props_t {
            std::int64_t core_count;
            std::size_t memory_bytes;
            // properties that act as stubs for compatibility
        };

        using stream_t            = stream_handle_t;
        using event_t             = event_handle_t;
        using device_properties_t = device_props_t;
    };

#ifdef CUDA_ENABLED
    // cuda backend traits
    template <>
    struct backend_traits_t<cuda_backend_t> {
        using stream_t            = cudaStream_t;
        using event_t             = cudaEvent_t;
        using device_properties_t = cudaDeviceProp;
        using error_t             = cudaError_t;

        // cuda-specific constants
        static constexpr auto success          = cudaSuccess;
        static constexpr auto host_to_device   = cudaMemcpyHostToDevice;
        static constexpr auto device_to_host   = cudaMemcpyDeviceToHost;
        static constexpr auto device_to_device = cudaMemcpyDeviceToDevice;
    };
#endif

#ifdef HIP_ENABLED
    // hip backend traits
    template <>
    struct backend_traits_t<hip_backend_t> {
        using stream_t            = hipStream_t;
        using event_t             = hipEvent_t;
        using device_properties_t = hipDeviceProp_t;
        using error_t             = hipError_t;

        // hip-specific constants
        static constexpr auto success          = hipSuccess;
        static constexpr auto host_to_device   = hipMemcpyHostToDevice;
        static constexpr auto device_to_host   = hipMemcpyDeviceToHost;
        static constexpr auto device_to_device = hipMemcpyDeviceToDevice;
    };
#endif

    // helper aliases
    template <typename backend_t>
    using stream_handle = typename backend_traits_t<backend_t>::stream_t;

    template <typename backend_t>
    using event_handle = typename backend_traits_t<backend_t>::event_t;

    template <typename backend_t>
    using device_props =
        typename backend_traits_t<backend_t>::device_properties_t;

}   // namespace simbi::hetero

#endif
