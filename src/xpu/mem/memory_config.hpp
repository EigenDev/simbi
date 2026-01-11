// =============================================================================
// memory_config.hpp
//
// compile-time memory space selection for xpu framework.
// chooses between device memory and unified memory based on build configuration.
// provides type aliases for consistent memory space usage throughout simbi.
//
// usage:
//   #include <xpu/mem/memory_config.hpp>
//   using namespace simbi::xpu;
//
//   field_t<float, sim_memory_space> field(1000);
//   shared_handle_t<data_t, sim_memory_space> handle;
//
// build configuration:
//   ./dev.py build --gpu --unified-memory  # uses unified_memory
//   ./dev.py build --gpu                   # uses device_memory (default)
// =============================================================================

#pragma once

#include "build_options.hpp"
#include "device_memory.hpp"
#include "unified_memory.hpp"

namespace simbi::xpu::mem {

    // =============================================================================
    // memory space selection based on build configuration
    // =============================================================================

#ifdef UNIFIED_MEMORY
    // development/debugging builds: unified memory for simplicity
    using sim_memory_space                    = unified_memory_t;
    constexpr bool        uses_unified_memory = true;
    constexpr const char* memory_space_name   = "unified_memory";
#else
    // production builds: device memory for performance
    using sim_memory_space                    = device_memory_t;
    constexpr bool        uses_unified_memory = false;
    constexpr const char* memory_space_name   = "device_memory";
#endif

    // =============================================================================
    // convenience aliases
    // =============================================================================

    // primary memory space for simulation data
    using simulation_memory = sim_memory_space;

    // explicit aliases for clarity
    using production_memory  = device_memory_t;
    using development_memory = unified_memory_t;

    // =============================================================================
    // compile-time feature detection
    // =============================================================================

    constexpr bool is_unified_memory_build()
    {
        return uses_unified_memory;
    }

    constexpr bool is_device_memory_build()
    {
        return !uses_unified_memory;
    }

    // =============================================================================
    // conditional dirty tracking for memory coherency
    // =============================================================================

    // mark data as dirty on host (no-op for unified memory)
    template <typename Handle>
    constexpr void mark_host_dirty_if_needed(Handle& handle) noexcept
    {
        if constexpr (is_device_memory_build()) {
            handle.mark_host_dirty();
        }
        // unified memory: no-op, data is always coherent
    }

    // mark data as dirty on device (no-op for unified memory)
    template <typename Handle>
    constexpr void mark_device_dirty_if_needed(Handle& handle) noexcept
    {
        if constexpr (is_device_memory_build()) {
            handle.mark_device_dirty();
        }
        // unified memory: no-op, data is always coherent
    }

    // check if synchronization is needed (always false for unified memory)
    template <typename Handle>
    constexpr bool needs_sync_check(const Handle& handle) noexcept
    {
        if constexpr (is_device_memory_build()) {
            return handle.needs_host_sync() || handle.needs_device_sync();
        }
        return false; // unified memory: never needs sync
    }

} // namespace simbi::xpu::mem
