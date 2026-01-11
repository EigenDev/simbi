// =============================================================================
// memory_concepts.hpp
//
// c++20 concepts for high-performance heterogeneous memory management.
// defines requirements for memory spaces, allocators, and transfer engines
// across different vendors (cuda, rocm, oneapi). optimized for massive
// simulation workloads with zero-overhead abstractions.
//
// design principles:
//   - numa-aware: memory locality and cpu affinity
//   - bandwidth-optimal: minimize transfers, maximize throughput
//   - cache-friendly: alignment and prefetching strategies
//   - vendor-agnostic: works with any hetero_device implementation
//
// usage:
//   template<memory_space Space>
//   class arena_allocator { /* high-perf allocation */ };
// =============================================================================

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace simbi::xpu::core {

    // =============================================================================
    // memory alignment constants
    // =============================================================================

    static constexpr std::size_t cache_line_size      = 64;
    static constexpr std::size_t gpu_memory_alignment = 256;
    static constexpr std::size_t simd_alignment       = 32;

    // trivially transferable concept for safe memory operations
    template <typename T>
    concept trivially_transferable_c = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T> &&
                                       !std::has_virtual_destructor_v<T>;

    // =============================================================================
    // memory space concept
    // =============================================================================

    template <typename Space>
    concept memory_space_c = requires {
        typename Space::pointer_type;
        typename Space::const_pointer_type;
        typename Space::size_type;

        // space properties
        { Space::is_host_accessible } -> std::convertible_to<bool>;
        { Space::is_device_accessible } -> std::convertible_to<bool>;
        { Space::is_unified } -> std::convertible_to<bool>;
        { Space::preferred_alignment } -> std::convertible_to<std::size_t>;

        // space identification
        { Space::space_name() } -> std::convertible_to<std::string_view>;
        { Space::memory_bandwidth_gb_per_sec() } -> std::convertible_to<double>;
    };

    template <typename Space>
    concept host_memory_space_c = memory_space_c<Space> && Space::is_host_accessible;

    template <typename Space>
    concept device_memory_space =
        memory_space_c<Space> && Space::is_device_accessible && !Space::is_host_accessible;

    template <typename Space>
    concept unified_memory_space_c = memory_space_c<Space> && Space::is_unified &&
                                     Space::is_host_accessible && Space::is_device_accessible;

    template <typename Space>
    concept high_bandwidth_space_c = memory_space_c<Space> && requires {
        { Space::memory_bandwidth_gb_per_sec() } -> std::convertible_to<double>;
    };

    // =============================================================================
    // memory allocator concepts
    // =============================================================================

    template <typename Allocator, typename T>
    concept memory_allocator = requires(Allocator alloc, std::size_t n, T* ptr) {
        typename Allocator::value_type;
        typename Allocator::pointer_type;
        typename Allocator::space_type;
        requires memory_space_c<typename Allocator::space_type>;
        requires std::same_as<T, typename Allocator::value_type>;

        // allocation/deallocation
        { alloc.allocate(n) } -> std::convertible_to<typename Allocator::pointer_type>;
        { alloc.deallocate(ptr, n) } -> std::same_as<void>;

        // alignment requirements
        { alloc.alignment() } -> std::convertible_to<std::size_t>;
        { alloc.max_allocation_size() } -> std::convertible_to<std::size_t>;

        // memory properties
        { alloc.bytes_allocated() } -> std::convertible_to<std::size_t>;
        { alloc.bytes_available() } -> std::convertible_to<std::size_t>;
    };

    template <typename Allocator, typename T>
    concept async_allocator =
        memory_allocator<Allocator, T> && requires(Allocator alloc, std::size_t n) {
            typename Allocator::async_token_type;

            // asynchronous operations
            {
                alloc.allocate_async(n)
            } -> std::convertible_to<typename Allocator::async_token_type>;
            {
                alloc.allocate_async(n, std::declval<typename Allocator::stream_type>())
            } -> std::convertible_to<typename Allocator::async_token_type>;
        };

    template <typename Allocator, typename T>
    concept pool_allocator = memory_allocator<Allocator, T> && requires(Allocator alloc) {
        // pool management
        { alloc.reset_pool() } -> std::same_as<void>;
        { alloc.shrink_to_fit() } -> std::same_as<void>;
        { alloc.pool_utilization() } -> std::convertible_to<double>;

        // bucket information (for bucket allocators)
        { alloc.bucket_count() } -> std::convertible_to<std::size_t>;
        { alloc.bucket_size(std::declval<std::size_t>()) } -> std::convertible_to<std::size_t>;
    };

    // =============================================================================
    // arena allocator concepts for massive simulations
    // =============================================================================

    template <typename Arena>
    concept memory_arena = requires(Arena arena, std::size_t bytes) {
        typename Arena::space_type;
        requires memory_space_c<typename Arena::space_type>;

        // bulk allocation from arena
        { arena.allocate(bytes) } -> std::convertible_to<void*>;
        {
            arena.allocate_aligned(bytes, std::declval<std::size_t>())
        } -> std::convertible_to<void*>;

        // arena management
        { arena.reset() } -> std::same_as<void>;
        { arena.capacity() } -> std::convertible_to<std::size_t>;
        { arena.bytes_used() } -> std::convertible_to<std::size_t>;
        { arena.utilization() } -> std::convertible_to<double>;

        // numa/locality hints
        { arena.preferred_numa_node() } -> std::convertible_to<int>;
    };

    template <typename Arena>
    concept numa_aware_arena = memory_arena<Arena> && requires(Arena arena, int node) {
        { arena.bind_to_numa_node(node) } -> std::same_as<void>;
        { arena.current_numa_node() } -> std::convertible_to<int>;
        { arena.numa_distance(node) } -> std::convertible_to<double>;
    };

    template <typename Arena>
    concept multi_device_arena =
        memory_arena<Arena> && requires(Arena arena, std::int64_t device_id) {
            { arena.device_count() } -> std::convertible_to<std::size_t>;
            {
                arena.allocate_on_device(std::declval<std::size_t>(), device_id)
            } -> std::convertible_to<void*>;
            { arena.current_device() } -> std::convertible_to<int>;
        };

    // =============================================================================
    // memory transfer concepts
    // =============================================================================

    template <typename Transfer>
    concept memory_transfer_engine = requires(Transfer engine) {
        typename Transfer::source_space_type;
        typename Transfer::destination_space_type;
        requires memory_space_c<typename Transfer::source_space_type>;
        requires memory_space_c<typename Transfer::destination_space_type>;

        // synchronous transfers
        {
            engine.copy(
                std::declval<const void*>(),
                std::declval<void*>(),
                std::declval<std::size_t>()
            )
        } -> std::same_as<void>;

        // transfer properties
        { engine.bandwidth_gb_per_sec() } -> std::convertible_to<double>;
        { engine.latency_microseconds() } -> std::convertible_to<double>;
        { engine.preferred_chunk_size() } -> std::convertible_to<std::size_t>;
    };

    template <typename Transfer>
    concept async_transfer_engine = memory_transfer_engine<Transfer> && requires(Transfer engine) {
        typename Transfer::async_token_type;
        typename Transfer::stream_type;

        // asynchronous transfers
        {
            engine.copy_async(
                std::declval<const void*>(),
                std::declval<void*>(),
                std::declval<std::size_t>(),
                std::declval<typename Transfer::stream_type>()
            )
        } -> std::convertible_to<typename Transfer::async_token_type>;

        // pipelined transfers
        {
            engine.copy_pipelined(
                std::declval<std::span<const void* const>>(),
                std::declval<std::span<void* const>>(),
                std::declval<std::span<const std::size_t>>(),
                std::declval<typename Transfer::stream_type>()
            )
        } -> std::convertible_to<typename Transfer::async_token_type>;
    };

    template <typename Transfer>
    concept peer_transfer_engine =
        async_transfer_engine<Transfer> && requires(Transfer engine, int src_dev, int dst_dev) {
            // peer-to-peer transfers between devices
            { engine.supports_peer_access(src_dev, dst_dev) } -> std::convertible_to<bool>;
            { engine.enable_peer_access(src_dev, dst_dev) } -> std::same_as<void>;
            { engine.peer_bandwidth_gb_per_sec(src_dev, dst_dev) } -> std::convertible_to<double>;
        };

    // =============================================================================
    // memory pattern concepts for optimization
    // =============================================================================

    template <typename Pattern>
    concept memory_access_pattern = requires {
        typename Pattern::access_type;

        // access pattern properties
        { Pattern::is_sequential } -> std::convertible_to<bool>;
        { Pattern::is_random } -> std::convertible_to<bool>;
        { Pattern::is_strided } -> std::convertible_to<bool>;
        { Pattern::stride_bytes() } -> std::convertible_to<std::size_t>;
        { Pattern::prefetch_distance() } -> std::convertible_to<std::size_t>;
    };

    struct sequential_access_t
    {
        using access_type                          = sequential_access_t;
        static constexpr bool        is_sequential = true;
        static constexpr bool        is_random     = false;
        static constexpr bool        is_strided    = false;
        static constexpr std::size_t stride_bytes()
        {
            return sizeof(void*);
        }
        static constexpr std::size_t prefetch_distance()
        {
            return cache_line_size * 4;
        }
    };

    struct random_access_t
    {
        using access_type                          = random_access_t;
        static constexpr bool        is_sequential = false;
        static constexpr bool        is_random     = true;
        static constexpr bool        is_strided    = false;
        static constexpr std::size_t stride_bytes()
        {
            return 0;
        }
        static constexpr std::size_t prefetch_distance()
        {
            return cache_line_size;
        }
    };

    template <std::size_t Stride>
    struct strided_access_t
    {
        using access_type                          = strided_access_t<Stride>;
        static constexpr bool        is_sequential = false;
        static constexpr bool        is_random     = false;
        static constexpr bool        is_strided    = true;
        static constexpr std::size_t stride_bytes()
        {
            return Stride;
        }
        static constexpr std::size_t prefetch_distance()
        {
            return Stride * 2;
        }
    };

    // =============================================================================
    // memory layout concepts for data structures
    // =============================================================================

    template <typename Layout>
    concept memory_layout = requires {
        typename Layout::element_type;

        // layout properties
        { Layout::alignment } -> std::convertible_to<std::size_t>;
        { Layout::is_contiguous } -> std::convertible_to<bool>;
        { Layout::is_vectorizable } -> std::convertible_to<bool>;
    };

    // array-of-structures (aos) layout
    template <typename T>
    struct aos_layout_t
    {
        using element_type                           = T;
        static constexpr std::size_t alignment       = alignof(T);
        static constexpr bool        is_contiguous   = true;
        static constexpr bool        is_vectorizable = std::is_arithmetic_v<T>;
    };

    // structure-of-arrays (soa) layout - better for simd
    template <typename... Fields>
    struct soa_layout_t
    {
        using element_type                           = std::tuple<Fields...>;
        static constexpr std::size_t alignment       = std::max({alignof(Fields)...});
        static constexpr bool        is_contiguous   = false;
        static constexpr bool        is_vectorizable = (std::is_arithmetic_v<Fields> && ...);
    };

    // =============================================================================
    // cache-aware concepts for massive simulations
    // =============================================================================

    template <typename CachePolicy>
    concept cache_policy = requires {
        // cache hint types
        { CachePolicy::temporal_locality } -> std::convertible_to<bool>;
        { CachePolicy::prefetch_enabled } -> std::convertible_to<bool>;
        { CachePolicy::cache_level } -> std::convertible_to<int>; // l1, l2, l3
    };

    struct cache_friendly_t
    {
        static constexpr bool temporal_locality = true;
        static constexpr bool prefetch_enabled  = true;
        static constexpr int  cache_level       = 1; // target l1 cache
    };

    struct cache_bypass_t
    {
        static constexpr bool temporal_locality = false;
        static constexpr bool prefetch_enabled  = false;
        static constexpr int  cache_level       = 3; // bypass to l3/memory
    };

    // =============================================================================
    // composite memory management concept
    // =============================================================================

    template <typename Manager>
    concept hetero_memory_manager = requires(Manager mgr) {
        typename Manager::host_allocator_type;
        typename Manager::device_allocator_type;
        typename Manager::transfer_engine_type;

        requires memory_allocator<typename Manager::host_allocator_type, std::byte>;
        requires memory_allocator<typename Manager::device_allocator_type, std::byte>;
        requires memory_transfer_engine<typename Manager::transfer_engine_type>;

        // unified interface
        { mgr.host_allocator() } -> std::convertible_to<typename Manager::host_allocator_type&>;
        {
            mgr.device_allocator(std::declval<int>())
        } -> std::convertible_to<typename Manager::device_allocator_type&>;
        { mgr.transfer_engine() } -> std::convertible_to<typename Manager::transfer_engine_type&>;

        // memory topology
        { mgr.device_count() } -> std::convertible_to<std::size_t>;
        { mgr.numa_node_count() } -> std::convertible_to<std::size_t>;
        { mgr.memory_topology() } -> std::convertible_to<std::string_view>;
    };

    // =============================================================================
    // compile-time memory optimization utilities
    // =============================================================================

    template <memory_space_c Source, memory_space_c Destination>
    constexpr bool requires_transfer = !std::same_as<Source, Destination>;

    template <memory_space_c Space>
    constexpr std::size_t optimal_alignment()
    {
        if constexpr (device_memory_space<Space>) {
            return gpu_memory_alignment;
        }
        else if constexpr (high_bandwidth_space_c<Space>) {
            return simd_alignment;
        }
        else {
            return cache_line_size;
        }
    }

    template <trivially_transferable_c T>
    constexpr std::size_t optimal_chunk_size()
    {
        constexpr std::size_t element_size = sizeof(T);
        constexpr std::size_t target_chunk = 64 * 1024; // 64kb chunks
        return (target_chunk / element_size) * element_size;
    }

    // memory bandwidth-aware algorithm selection
    template <memory_space_c Source, memory_space_c Destination>
    constexpr bool use_async_transfer()
    {
        return requires_transfer<Source, Destination> &&
               (high_bandwidth_space_c<Source> || high_bandwidth_space_c<Destination>);
    }

    // =============================================================================
    // memory safety concepts for production use
    // =============================================================================

    template <typename Ptr>
    concept safe_memory_pointer = std::is_pointer_v<Ptr> && requires(Ptr ptr) {
        { ptr.is_valid() } -> std::convertible_to<bool>;
        { ptr.size() } -> std::convertible_to<std::size_t>;
        { ptr.space_id() } -> std::convertible_to<int>;
    };

    template <typename Guard>
    concept memory_guard = requires(Guard guard) {
        typename Guard::pointer_type;
        requires safe_memory_pointer<typename Guard::pointer_type>;

        // raii memory management
        { guard.get() } -> std::convertible_to<typename Guard::pointer_type>;
        { guard.release() } -> std::convertible_to<typename Guard::pointer_type>;
        { guard.reset() } -> std::same_as<void>;
    };

} // namespace simbi::xpu::core
