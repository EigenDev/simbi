// =============================================================================
// device_concepts.hpp
//
// c++20 concepts defining requirements for heterogeneous device abstraction.
// any vendor (cuda, rocm, oneapi, opencl) must satisfy these concepts to
// integrate with xpu. provides zero-overhead compile-time dispatch while
// maintaining vendor abstraction for massive simulation workloads.
//
// design principles:
//   - concept-driven: compile-time requirements checking
//   - zero-overhead: no virtual dispatch or runtime polymorphism
//   - extensible: new vendors just implement concepts
//   - type-safe: strong typing with template metaprogramming
//
// usage:
//   template<hetero_device Device>
//   class my_algorithm { /* works with any vendor */ };
// =============================================================================

#pragma once

#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace simbi::xpu::core {

    // =============================================================================
    // fundamental device handle types
    // =============================================================================

    template <typename Handle>
    concept device_handle = std::is_trivially_copyable_v<Handle> &&
                            std::is_trivially_destructible_v<Handle> && !std::is_void_v<Handle>;

    template <typename Handle>
    concept memory_handle =
        device_handle<Handle> && std::equality_comparable<Handle> && requires(Handle h) {
            { static_cast<bool>(h) } -> std::convertible_to<bool>;
        }; // null check via explicit bool conversion

    template <typename Handle>
    concept stream_handle = device_handle<Handle> && std::equality_comparable<Handle>;

    template <typename Handle>
    concept event_handle = device_handle<Handle> && std::equality_comparable<Handle>;

    // =============================================================================
    // memory allocation requirements
    // =============================================================================

    template <typename Device>
    concept device_memory_allocator = requires(Device device, std::size_t bytes) {
        typename Device::memory_handle_type;
        requires memory_handle<typename Device::memory_handle_type>;

        // synchronous allocation
        { device.allocate(bytes) } -> std::convertible_to<typename Device::memory_handle_type>;
        {
            device.deallocate(std::declval<typename Device::memory_handle_type>())
        } -> std::same_as<void>;

        // memory properties
        { device.memory_alignment() } -> std::convertible_to<std::size_t>;
        { device.max_allocation_size() } -> std::convertible_to<std::size_t>;
        {
            device.is_accessible_from_host(std::declval<typename Device::memory_handle_type>())
        } -> std::convertible_to<bool>;
    };

    template <typename Device>
    concept async_memory_allocator =
        device_memory_allocator<Device> && requires(Device device, std::size_t bytes) {
            typename Device::stream_handle_type;

            // asynchronous allocation (cuda 11.2+, rocm 5.0+)
            {
                device.allocate_async(bytes, std::declval<typename Device::stream_handle_type>())
            } -> std::convertible_to<typename Device::memory_handle_type>;
            {
                device.deallocate_async(
                    std::declval<typename Device::memory_handle_type>(),
                    std::declval<typename Device::stream_handle_type>()
                )
            } -> std::same_as<void>;
        };

    // =============================================================================
    // execution stream requirements
    // =============================================================================

    template <typename Device>
    concept device_stream_manager = requires(Device device) {
        typename Device::stream_handle_type;
        requires stream_handle<typename Device::stream_handle_type>;

        // stream creation/destruction
        { device.create_stream() } -> std::convertible_to<typename Device::stream_handle_type>;
        {
            device.destroy_stream(std::declval<typename Device::stream_handle_type>())
        } -> std::same_as<void>;

        // synchronization
        {
            device.synchronize_stream(std::declval<typename Device::stream_handle_type>())
        } -> std::same_as<void>;
        {
            device.is_stream_ready(std::declval<typename Device::stream_handle_type>())
        } -> std::convertible_to<bool>;

        // default stream access
        { device.default_stream() } -> std::convertible_to<typename Device::stream_handle_type>;
    };

    // =============================================================================
    // event and synchronization requirements
    // =============================================================================

    template <typename Device>
    concept device_event_manager = requires(Device device) {
        typename Device::event_handle_type;
        requires event_handle<typename Device::event_handle_type>;

        // event creation/destruction
        { device.create_event() } -> std::convertible_to<typename Device::event_handle_type>;
        {
            device.destroy_event(std::declval<typename Device::event_handle_type>())
        } -> std::same_as<void>;

        // event recording and querying
        {
            device.record_event(
                std::declval<typename Device::event_handle_type>(),
                std::declval<typename Device::stream_handle_type>()
            )
        } -> std::same_as<void>;
        {
            device.is_event_ready(std::declval<typename Device::event_handle_type>())
        } -> std::convertible_to<bool>;
        {
            device.synchronize_event(std::declval<typename Device::event_handle_type>())
        } -> std::same_as<void>;

        // stream dependencies
        {
            device.stream_wait_event(
                std::declval<typename Device::stream_handle_type>(),
                std::declval<typename Device::event_handle_type>()
            )
        } -> std::same_as<void>;
    };

    // =============================================================================
    // memory transfer requirements
    // =============================================================================

    template <typename Device>
    concept device_memory_transfer = requires(Device device) {
        typename Device::memory_handle_type;
        typename Device::stream_handle_type;

        // synchronous transfers
        {
            device.copy_host_to_device(
                std::declval<const void*>(),
                std::declval<typename Device::memory_handle_type>(),
                std::declval<std::size_t>()
            )
        } -> std::same_as<void>;

        {
            device.copy_device_to_host(
                std::declval<typename Device::memory_handle_type>(),
                std::declval<void*>(),
                std::declval<std::size_t>()
            )
        } -> std::same_as<void>;

        {
            device.copy_device_to_device(
                std::declval<typename Device::memory_handle_type>(),
                std::declval<typename Device::memory_handle_type>(),
                std::declval<std::size_t>()
            )
        } -> std::same_as<void>;

        // asynchronous transfers
        {
            device.copy_host_to_device_async(
                std::declval<const void*>(),
                std::declval<typename Device::memory_handle_type>(),
                std::declval<std::size_t>(),
                std::declval<typename Device::stream_handle_type>()
            )
        } -> std::same_as<void>;

        {
            device.copy_device_to_host_async(
                std::declval<typename Device::memory_handle_type>(),
                std::declval<void*>(),
                std::declval<std::size_t>(),
                std::declval<typename Device::stream_handle_type>()
            )
        } -> std::same_as<void>;
    };

    // =============================================================================
    // device information and capabilities
    // =============================================================================

    template <typename Device>
    concept device_properties = requires(Device device) {
        // device identification
        { device.device_id() } -> std::convertible_to<int>;
        { device.device_name() } -> std::convertible_to<std::string_view>;
        { device.vendor_name() } -> std::convertible_to<std::string_view>;

        // memory properties
        { device.total_memory() } -> std::convertible_to<std::size_t>;
        { device.available_memory() } -> std::convertible_to<std::size_t>;
        { device.memory_bandwidth_gb_per_sec() } -> std::convertible_to<double>;

        // compute properties
        { device.compute_units() } -> std::convertible_to<std::size_t>;
        { device.max_threads_per_block() } -> std::convertible_to<std::size_t>;
        { device.warp_size() } -> std::convertible_to<std::size_t>;

        // capability queries
        { device.supports_unified_memory() } -> std::convertible_to<bool>;
        { device.supports_peer_to_peer() } -> std::convertible_to<bool>;
        { device.supports_async_memory_ops() } -> std::convertible_to<bool>;
    };

    // =============================================================================
    // kernel execution requirements
    // =============================================================================

    template <typename Device>
    concept device_kernel_executor = requires(Device device) {
        typename Device::kernel_handle_type;
        typename Device::stream_handle_type;
        typename Device::event_handle_type;

        // kernel launching - vendor-specific implementation details abstracted
        {
            device.launch_kernel(
                std::declval<typename Device::kernel_handle_type>(),
                std::declval<std::size_t>(), // grid size
                std::declval<std::size_t>(), // block size
                std::declval<typename Device::stream_handle_type>(),
                std::declval<int>() // kernel arguments placeholder
            )
        } -> std::convertible_to<typename Device::event_handle_type>;
    };

    // =============================================================================
    // composite device concept - the main requirement
    // =============================================================================

    template <typename Device>
    concept hetero_device = device_memory_allocator<Device> && device_stream_manager<Device> &&
                            device_event_manager<Device> && device_memory_transfer<Device> &&
                            device_properties<Device> &&
                            // kernel execution is optional for some device types
                            std::copyable<Device> && std::destructible<Device>;

    // optional advanced features
    template <typename Device>
    concept advanced_hetero_device =
        hetero_device<Device> && async_memory_allocator<Device> && device_kernel_executor<Device>;

    // =============================================================================
    // device type traits for compile-time optimization
    // =============================================================================

    template <hetero_device Device>
    struct device_traits
    {
        using device_type        = Device;
        using memory_handle_type = typename Device::memory_handle_type;
        using stream_handle_type = typename Device::stream_handle_type;
        using event_handle_type  = typename Device::event_handle_type;

        static constexpr bool is_gpu_device         = Device::is_gpu_device;
        static constexpr bool is_cpu_device         = Device::is_cpu_device;
        static constexpr bool supports_async_memory = async_memory_allocator<Device>;
        static constexpr bool supports_kernels      = device_kernel_executor<Device>;

        // vendor identification for optimizations
        static constexpr std::string_view vendor = Device::vendor_name();
    };

    // =============================================================================
    // device type categories for algorithm dispatch
    // =============================================================================

    template <typename Device>
    concept gpu_device = hetero_device<Device> && Device::is_gpu_device;

    template <typename Device>
    concept cpu_device = hetero_device<Device> && Device::is_cpu_device;

    template <typename Device>
    concept unified_memory_device = hetero_device<Device> && requires(Device device) {
        { device.supports_unified_memory() } -> std::same_as<bool>;
    } && Device{}.supports_unified_memory();

    template <typename Device>
    concept high_bandwidth_device = hetero_device<Device> && requires(Device device) {
        { device.memory_bandwidth_gb_per_sec() } -> std::convertible_to<double>;
    };

    // =============================================================================
    // compile-time device selection utilities
    // =============================================================================

    template <hetero_device... Devices>
    struct device_pack
    {
        static constexpr std::size_t count = sizeof...(Devices);

        template <std::size_t N>
        using nth_device = std::tuple_element_t<N, std::tuple<Devices...>>;

        // find first device satisfying concept
        template <template <typename> typename Concept>
        static constexpr std::size_t find_first()
        {
            std::size_t index = 0;
            ((Concept<Devices>::value ? true : (++index, false)) || ...);
            return index < count ? index : count;
        }
    };

    // =============================================================================
    // vendor-agnostic algorithm dispatch helpers
    // =============================================================================

    template <hetero_device Device>
    constexpr auto dispatch_by_vendor(Device /*device*/)
    {
        if constexpr (device_traits<Device>::vendor == "nvidia") {
            return []<typename... Args>(Args&&... args) {
                // cuda-specific optimizations
                return cuda_optimized_path(std::forward<Args>(args)...);
            };
        }
        else if constexpr (device_traits<Device>::vendor == "amd") {
            return []<typename... Args>(Args&&... args) {
                // rocm-specific optimizations
                return rocm_optimized_path(std::forward<Args>(args)...);
            };
        }
        else {
            return []<typename... Args>(Args&&... args) {
                // generic fallback
                return generic_path(std::forward<Args>(args)...);
            };
        }
    }

} // namespace simbi::xpu::core
