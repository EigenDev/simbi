// =============================================================================
// mem.hpp
//
// convenience header for the xpu memory management library.
// includes all the core components of the memory library, such as memory
// space definitions, allocation blocks, shared handles, views, and memory
// operations, providing a single point of inclusion.
//
// usage:
//   #include "xpu/mem/mem.hpp"
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

    using host_memory_block_t    = memory_block_t<host_memory>;
    using device_memory_block_t  = memory_block_t<device_memory>;
    using unified_memory_block_t = memory_block_t<unified_memory>;
    using sim_memory_block_t     = memory_block_t<sim_memory_space>;

} // namespace simbi::xpu::mem
