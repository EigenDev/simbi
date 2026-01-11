// =============================================================================
// execution_space.hpp
//
// c++20 execution space concepts and vendor-agnostic interface for
// heterogeneous computing. provides unified abstraction for cpu and gpu
// execution across different vendors (cuda, rocm, oneapi).
//
// design principles:
//   - concept-driven: compile-time requirements checking
//   - vendor-agnostic: works with any hetero_device implementation
//   - zero-overhead: compile-time dispatch and optimization
//   - extensible: adding new vendors is trivial
//
// usage:
//   template<execution_space Space>
//   auto parallel_algorithm(Space space) { /* works with any space */ };
// =============================================================================

#pragma once

#include "xpu/core/device_concepts.hpp"
#include "xpu/core/execution_concepts.hpp"
#include "xpu/core/memory_concepts.hpp"
#include "xpu/vendors/cuda/cuda_device.hpp"
#include "xpu/vendors/oneapi/oneapi_device.hpp"
#include "xpu/vendors/rocm/rocm_device.hpp"

#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace simbi::xpu::exec {

    // =============================================================================
    // re-export core concepts for convenience
    // =============================================================================

    using core::execution_space_c;
    using core::hetero_device_c;
    using core::memory_space_c;

    // =============================================================================
    // execution space concept (enhanced from core)
    // =============================================================================

    template <typename Space>
    concept xpu_execution_space_c = core::execution_space_c<Space> && requires {
        typename Space::device_type;
        requires hetero_device_c<typename Space::device_type>;

        // xpu-specific requirements
        { Space::vendor_name() } -> std::convertible_to<std::string_view>;
        { Space::default_device_id() } -> std::convertible_to<int>;
    };

    // =============================================================================
    // execution space traits (vendor-aware)
    // =============================================================================

    template <xpu_execution_space_c Space>
    struct execution_space_traits
    {
        using space_type         = Space;
        using device_type        = typename Space::device_type;
        using memory_space_type  = typename Space::memory_space_type;
        using stream_handle_type = typename Space::stream_handle_type;
        using event_handle_type  = typename Space::event_handle_type;

        static constexpr std::string_view name()
        {
            return Space::space_name();
        }

        static constexpr std::string_view vendor()
        {
            return Space::vendor_name();
        }

        static constexpr bool is_gpu           = Space::is_device_space;
        static constexpr bool is_host          = Space::is_host_space;
        static constexpr bool supports_async   = Space::supports_async;
        static constexpr bool supports_kernels = Space::supports_kernels;

        // performance characteristics
        static constexpr std::size_t max_concurrency()
        {
            return Space::max_concurrency();
        }

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            return Space::memory_bandwidth_gb_per_sec();
        }
    };

    // =============================================================================
    // vendor-agnostic space selection
    // =============================================================================

    // compile-time device availability detection
    template <typename Device>
    constexpr bool is_device_available()
    {
        if constexpr (std::same_as<Device, vendors::cuda::cuda_device_t>) {
#ifdef XPU_CUDA_AVAILABLE
            return true;
#else
            return false;
#endif
        }
        else if constexpr (std::same_as<Device, vendors::rocm::rocm_device_t>) {
#ifdef XPU_ROCM_AVAILABLE
            return true;
#else
            return false;
#endif
        }
        else if constexpr (std::same_as<Device, vendors::oneapi::oneapi_device_t>) {
#ifdef XPU_ONEAPI_AVAILABLE
            return true;
#else
            return false;
#endif
        }
        else {
            return false;
        }
    }

    // automatic vendor selection based on availability
    template <bool prefer_gpu = true>
    struct default_space_selector
    {
        // prefer cuda > rocm > oneapi > cpu when prefer_gpu=true
        // force cpu when prefer_gpu=false
        using type = std::conditional_t<
            prefer_gpu && is_device_available<vendors::cuda::cuda_device_t>(),
            /* cuda available */ class cuda_space,
            std::conditional_t<
                prefer_gpu && is_device_available<vendors::rocm::rocm_device_t>(),
                /* rocm available */ class rocm_space,
                std::conditional_t<
                    prefer_gpu && is_device_available<vendors::oneapi::oneapi_device_t>(),
                    /* oneapi available */ class oneapi_space,
                    /* fallback to cpu */ class cpu_space>>>;
    };

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using default_space = default_space_selector<>::type;

} // namespace simbi::xpu::exec
