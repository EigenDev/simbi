// =============================================================================
// comm.hpp
//
// convenience header for xpu communication layer
// includes all comm types and transfer operations
//
// usage:
//   #include <xpu/comm/comm.hpp>
//   using namespace simbi::xpu::comm;
//
//   rank_id_t src{0, 0};
//   rank_id_t dst{0, 1};
//   transfer_sync(src, src_ptr, dst, dst_ptr, bytes);
// =============================================================================
#pragma once

#include "transfer.hpp"
#include "types.hpp"

namespace simbi::xpu::comm {
    // all types and functions are already in the namespace
    // this header just provides convenient single-include access
}


