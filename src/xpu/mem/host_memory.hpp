// =============================================================================
// host_memory.hpp
//
// host memory space implementation using standard allocators.
// provides cpu-accessible memory allocation and management.
// implements memory_space concept for host-side data storage.
//
// usage:
//   auto block = allocate<host_memory, float>(1000);
//   shared_buffer_t<int, host_memory> buffer(n);
//   bool accessible = host_memory::is_accessible_from<device_memory>();
// =============================================================================

#pragma once

#include "memory_space.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string_view>

namespace simbi::xpu {

    // =============================================================================
    // host memory space
    // =============================================================================

    struct host_memory_t
    {
        // =============================================================================
        // type requirements for memory_space concept
        // =============================================================================

        using pointer_type       = void*;
        using const_pointer_type = const void*;
        using size_type          = std::size_t;

        // =============================================================================
        // compile-time properties
        // =============================================================================

        static constexpr std::string_view space_name()
        {
            return "host";
        }

        static constexpr std::string_view name() // legacy
        {
            return "host";
        }

        static constexpr bool        is_device_accessible = false;
        static constexpr bool        is_host_accessible   = true;
        static constexpr bool        is_unified           = false;
        static constexpr std::size_t preferred_alignment  = 64; // cache line aligned

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            return 30.0; // typical ddr4 bandwidth
        }

        // =============================================================================
        // allocation interface
        // =============================================================================

        static void* allocate(std::size_t size)
        {
            if (size == 0) {
                return nullptr;
            }

            // use aligned allocation for better performance
            constexpr std::size_t alignment = 64; // cache line aligned
            void* ptr = std::aligned_alloc(alignment, (size + alignment - 1) & ~(alignment - 1));
            if (!ptr) {
                throw std::bad_alloc{};
            }
            return ptr;
        }

        static void deallocate(void* ptr, std::size_t /* size */)
        {
            if (ptr) {
                std::free(ptr);
            }
        }

        // =============================================================================
        // accessibility queries
        // =============================================================================

        template <memory_space OtherSpace>
        static constexpr bool is_accessible_from()
        {
            // host memory is accessible from host spaces only
            return OtherSpace::is_host_accessible;
        }

        // specialized accessibility queries for known spaces
        static constexpr bool is_accessible_from_host()
        {
            return true;
        }
        static constexpr bool is_accessible_from_device()
        {
            return false;
        }
        static constexpr bool is_accessible_from_unified()
        {
            return true; // unified memory can access host
        }

        // =============================================================================
        // memory operations
        // =============================================================================

        static void memset(void* ptr, int value, std::size_t size)
        {
            std::memset(ptr, value, size);
        }

        static void memcpy(void* dest, const void* src, std::size_t size)
        {
            std::memcpy(dest, src, size);
        }

        // =============================================================================
        // performance hints
        // =============================================================================

        struct allocation_hints
        {
            static constexpr bool        supports_concurrent_access = true;
            static constexpr bool        supports_cache_coherency   = true;
            static constexpr bool        requires_explicit_sync     = false;
            static constexpr std::size_t preferred_alignment        = 64;
        };

        // =============================================================================
        // debug and introspection
        // =============================================================================

        static bool is_valid_pointer(const void* ptr)
        {
            return ptr != nullptr;
        }

        static std::size_t get_alignment()
        {
            return allocation_hints::preferred_alignment;
        }

        // =============================================================================
        // statistics (optional, for debugging)
        // =============================================================================

        struct stats
        {
            static std::size_t total_allocated;
            static std::size_t total_deallocated;
            static std::size_t current_usage;

            static void record_allocation(std::size_t size)
            {
                total_allocated += size;
                current_usage += size;
            }

            static void record_deallocation(std::size_t size)
            {
                total_deallocated += size;
                current_usage = (current_usage >= size) ? current_usage - size : 0;
            }

            static void reset()
            {
                total_allocated   = 0;
                total_deallocated = 0;
                current_usage     = 0;
            }
        };
    };

    // =============================================================================
    // static member definitions
    // =============================================================================

    inline std::size_t host_memory_t::stats::total_allocated   = 0;
    inline std::size_t host_memory_t::stats::total_deallocated = 0;
    inline std::size_t host_memory_t::stats::current_usage     = 0;

    // =============================================================================
    // default memory space specialization when cuda not available
    // =============================================================================

    template <>
    struct default_memory_space_selector<false>
    {
        using type = host_memory;
    };

    // static assertion to verify concept compliance
    static_assert(memory_space<host_memory_t>);

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using host_block_t = block_t<host_memory_t>;

    template <typename T>
    using host_buffer_t = block_t<host_memory_t>;

} // namespace simbi::xpu
