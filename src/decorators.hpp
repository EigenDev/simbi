// =============================================================================
// decorators.hpp
//
// function decorators for cross-platform compilation.
//
// usage:
//   DUAL real add(real a, real b) { return a + b; }
//   DEV void helper() { /* device only */ }
//   KERNEL void kernel() { /* kernel entry point */ }
// =============================================================================

#ifndef SIMBI_DECORATORS_HPP
#define SIMBI_DECORATORS_HPP

#include "xpu/device/detail/portability.hpp"

#define DUAL   XPU_HOST_DEVICE
#define DEV    XPU_DEVICE
#define KERNEL XPU_GLOBAL

#endif
