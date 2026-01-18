// =============================================================================
// mem.hpp
//
// main header for xpu memory management
// includes all memory components as separate, focused modules.
// each module has one job and does it well.
//
// usage:
//   #include <xpu/mem/mem.hpp>
//   using namespace simbi::xpu::mem;
//
//   auto block = memory_block_t<device_memory>(1000);
//   auto handle = shared_handle_t<float>::make(42.0f);
//   sync_host_to_device<float>(host_block, device_block, count);
// =============================================================================

#pragma once

// memory space definitions
#include "device_memory.hpp"
#include "host_memory.hpp"
#include "memory_config.hpp"
#include "memory_space.hpp"
#include "unified_memory.hpp"

// core memory components -
#include "block.hpp"    // memory_block_t - simple raii ownership
#include "handle.hpp"   // shared_handle_t - reference counting + coherency
#include "ops.hpp"      // fill, zero, copy operations
#include "transfer.hpp" // explicit sync between memory spaces
#include "view.hpp"     // non-owning views

namespace simbi::xpu::mem {

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using host_memory_block_t    = memory_block_t<host_memory>;
    using device_memory_block_t  = memory_block_t<device_memory>;
    using unified_memory_block_t = memory_block_t<unified_memory>;
    using sim_memory_block_t     = memory_block_t<sim_memory_space>;

} // namespace simbi::xpu::mem
