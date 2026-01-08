// =============================================================================
// runtime_config.hpp
//
// runtime configuration state.
// mutable flags set by environment variables, user input, or runtime detection.
//
// contrast with build_config.hpp (compile-time) and platform.hpp (compile-time).
// everything here is non-const and can change during program execution.
//
// usage:
//   runtime_config::use_omp = detect_openmp_from_env();
//   if (runtime_config::use_omp) { /* enable openmp */ }
//
// initialization:
//   call runtime_config::initialize() at program startup to read environment
//   variables and set defaults.
// =============================================================================

#ifndef SIMBI_RUNTIME_CONFIG_HPP
#define SIMBI_RUNTIME_CONFIG_HPP

#include <cstdlib>
#include <string>

namespace simbi::runtime_config {

    // =========================================================================
    // openmp configuration
    // =========================================================================

    // enable/disable openmp at runtime
    // set by USE_OMP environment variable or programmatically
    inline bool use_omp = false;

    // =========================================================================
    // initialization from environment
    // =========================================================================

    // read environment variables and set runtime config
    // call this once at program startup (e.g., in driver::run_simulation)
    inline void initialize()
    {
        // read USE_OMP environment variable
        if (const char* env = std::getenv("USE_OMP")) {
            std::string val(env);
            use_omp = (val == "1" || val == "true" || val == "TRUE");
        }

        // future: add other runtime config here
        // - thread counts
        // - device selection
        // - debug flags
    }

} // namespace simbi::runtime_config

#endif
