// =============================================================================
// decorators.hpp
//
// unified function/variable decorators for cross-platform compilation.
// provides single set of macros that work for simbi's use cases.
//
// wraps xpu/device/detail/portability.hpp with simbi-specific names.
// use this file instead of portability.hpp in simbi code.
//
// decorators:
//  DUAL   : function can be called from host and device
//  DEV    : function can be called from device only
//  KERNEL : function is a kernel entry point
//
// usage:
//   DUAL real add(real a, real b) { return a + b; }
//   DEV void helper() { /* device only */ }
//   KERNEL void kernel() { /* kernel entry point */ }
//
// rationale:
//   - single source of truth for decorators
//   - can swap underlying implementation without changing user code
// =============================================================================

#ifndef SIMBI_DECORATORS_HPP
#define SIMBI_DECORATORS_HPP

#include "xpu/device/detail/portability.hpp"

// =============================================================================
// object decorators for xpu compatibility
// =============================================================================

#define DUAL   XPU_HOST_DEVICE
#define DEV    XPU_DEVICE
#define KERNEL XPU_GLOBAL

#endif
