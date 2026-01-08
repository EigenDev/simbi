// =============================================================================
// simbi_config.hpp
//
// master configuration header - single include for all configuration needs.
// aggregates platform detection, build config, runtime config, and decorators.
//
// include this instead of individual headers unless you specifically need
// only one aspect (e.g., only platform detection).
//
// usage:
//   #include "simbi_config.hpp"
//
//   using simbi::real;
//   if constexpr (simbi::platform::is_cuda) { /* ... */ }
//   if (simbi::runtime_config::use_omp) { /* ... */ }
//
//   DUAL real compute(real x) { return x * x; }
//
// components:
//   - platform.hpp       : compile-time platform detection
//   - build_config.hpp   : compile-time build configuration
//   - runtime_config.hpp : runtime mutable state
//   - decorators.hpp     : function/variable decorators
//
// =============================================================================

#ifndef SIMBI_CONFIG_HPP
#define SIMBI_CONFIG_HPP

// order matters: build_config depends on build_options,
// platform is independent, runtime_config is independent,
// decorators depends on platform (via portability.hpp)
#include "build_config.hpp"
#include "decorators.hpp"
#include "platform.hpp"
#include "runtime_config.hpp"

namespace simbi {

    // =========================================================================
    // convenience namespace imports (opt-in)
    // =========================================================================

    // users can do: using namespace simbi::config;
    // to get all constants in scope without qualification
    namespace config {
        // platform constants
        using namespace platform;

        // build configuration
        using build_config::column_major;
        using build_config::epsilon;
        using build_config::four_velocity;
        using build_config::real;
        using build_config::row_major;
        using build_config::unified_memory;
        using build_config::use_beta;

        // runtime config imported as references
        inline bool& use_omp = runtime_config::use_omp;
    } // namespace config

    // =========================================================================
    // backward compatibility layer for legacy code
    // =========================================================================

    namespace global {
        // platform detection (old enum style)
        enum class Platform : int {
            CPU = 0,
            GPU = 1
        };

        enum class Runtime : int {
            CUDA = 0,
            ROCM = 1,
            CPU  = 2
        };

        constexpr Platform BuildPlatform = platform::is_gpu ? Platform::GPU : Platform::CPU;

        constexpr Runtime BuildRuntime = platform::is_cuda  ? Runtime::CUDA
                                         : platform::is_hip ? Runtime::ROCM
                                                            : Runtime::CPU;

        // feature flags
        constexpr bool col_major           = build_config::column_major;
        constexpr bool using_four_velocity = build_config::four_velocity;
        constexpr bool on_gpu              = platform::is_gpu;

        // constants
        constexpr std::uint64_t WARP_SIZE = platform::warp_size;
        constexpr auto          epsilon   = build_config::epsilon;

        // runtime state (reference to allow mutation)
        inline bool& use_omp = runtime_config::use_omp;
    } // namespace global

} // namespace simbi

// =============================================================================
// global using declarations for convenience
// =============================================================================

// make 'real' available without qualification
using simbi::real;

#endif
