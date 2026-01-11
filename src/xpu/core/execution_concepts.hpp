// =============================================================================
// execution_concepts.hpp
//
// c++20 concepts for execution spaces - compile-time vendor abstraction.
// defines requirements for cpu/cuda/hip/sycl execution environments.
//
// design: zero-overhead compile-time dispatch, no runtime polymorphism.
//
// usage:
//   template<execution_space Space>
//   void parallel_algorithm(domain_t<3> domain) { /* dispatch at compile time */ }
// =============================================================================

#pragma once

#include <concepts>
#include <cstddef>
#include <string_view>

namespace simbi::xpu::core {

    // =============================================================================
    // execution space concept - what can execute parallel work
    // =============================================================================

    template <typename Space>
    concept execution_space_c = requires {
        // space identification
        { Space::space_name() } -> std::convertible_to<std::string_view>;

        // space capabilities
        { Space::is_host_space } -> std::convertible_to<bool>;
        { Space::is_device_space } -> std::convertible_to<bool>;
        { Space::supports_async } -> std::convertible_to<bool>;
        { Space::supports_kernels } -> std::convertible_to<bool>;

        // performance characteristics
        { Space::max_concurrency() } -> std::convertible_to<std::size_t>;
        { Space::preferred_block_size() } -> std::convertible_to<std::size_t>;
        { Space::memory_bandwidth_gb_per_sec() } -> std::convertible_to<double>;
    };

    // convenience concepts for space categories
    template <typename Space>
    concept host_execution_space_c = execution_space_c<Space> && Space::is_host_space;

    template <typename Space>
    concept device_execution_space_c = execution_space_c<Space> && Space::is_device_space;

    template <typename Space>
    concept async_execution_space_c = execution_space_c<Space> && Space::supports_async;

} // namespace simbi::xpu::core
