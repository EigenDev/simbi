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
#include <cstdint>
#include <string_view>
#include <utility>

namespace simbi::xpu::core {

    

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

        // stream and event handle types
        typename Space::stream_handle_type;
        typename Space::event_handle_type;

        // stream management (required)
        { Space::create_stream() } -> std::same_as<typename Space::stream_handle_type>;
        {
            Space::destroy_stream(std::declval<typename Space::stream_handle_type>())
        } -> std::same_as<void>;
        {
            Space::synchronize_stream(std::declval<typename Space::stream_handle_type>())
        } -> std::same_as<void>;
        {
            Space::is_stream_ready(std::declval<typename Space::stream_handle_type>())
        } -> std::convertible_to<bool>;

        // event management (required)
        { Space::create_event() } -> std::same_as<typename Space::event_handle_type>;
        {
            Space::destroy_event(std::declval<typename Space::event_handle_type>())
        } -> std::same_as<void>;
        {
            Space::record_event(
                std::declval<typename Space::event_handle_type>(),
                std::declval<typename Space::stream_handle_type>()
            )
        } -> std::same_as<void>;
        {
            Space::is_event_ready(std::declval<typename Space::event_handle_type>())
        } -> std::convertible_to<bool>;
        {
            Space::synchronize_event(std::declval<typename Space::event_handle_type>())
        } -> std::same_as<void>;
        {
            Space::stream_wait_event(
                std::declval<typename Space::stream_handle_type>(),
                std::declval<typename Space::event_handle_type>()
            )
        } -> std::same_as<void>;

        // device management (required)
        { Space::set_device(std::declval<std::int64_t>()) } -> std::same_as<void>;
    };

    // convenience concepts for space categories
    template <typename Space>
    concept host_execution_space_c = execution_space_c<Space> && Space::is_host_space;

    template <typename Space>
    concept device_execution_space_c = execution_space_c<Space> && Space::is_device_space;

    template <typename Space>
    concept async_execution_space_c = execution_space_c<Space> && Space::supports_async;

} // namespace simbi::xpu::core
