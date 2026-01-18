// =============================================================================
// build_config.hpp
//
// compile-time build configuration from meson.
//
// usage:
//   using simbi::real;
//   if constexpr (build::column_major) { /* ... */ }
//   constexpr auto eps = build::epsilon;
// =============================================================================
#pragma once

#include "build_options.hpp"

#include <type_traits>

namespace simbi::build {

#if FLOAT_PRECISION
    using real = float;
#else
    using real = double;
#endif

    inline constexpr real epsilon = std::is_same_v<real, float> ? 1e-6f : 1e-12;

#if COLUMN_MAJOR
    inline constexpr bool column_major = true;
#else
    inline constexpr bool column_major = false;
#endif

    inline constexpr bool row_major = !column_major;

#if FOUR_VELOCITY
    inline constexpr bool use_four_velocity = true;
#else
    inline constexpr bool use_four_velocity = false;
#endif

    inline constexpr bool use_beta = !use_four_velocity;

#if UNIFIED_MEMORY
    inline constexpr bool unified_memory = true;
#else
    inline constexpr bool unified_memory = false;
#endif

#if DEBUG_MODE
    inline constexpr bool debug_mode = true;
#else
    inline constexpr bool debug_mode = false;
#endif

    inline constexpr int max_iterations = 1000;

} // namespace simbi::build

namespace simbi {
    using real = build::real;
}
