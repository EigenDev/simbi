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
    // block/tile dimensions
    // =========================================================================

    struct block_dims_t
    {
        int x = 256;
        int y = 1;
        int z = 1;

        int total_threads() const
        {
            return x * y * z;
        }
    };

    // block dimensions for gpu kernels and cpu tiling
    // set by BLOCK_X, BLOCK_Y, BLOCK_Z environment variables
    // defaults chosen based on typical workload dimensionality
    inline block_dims_t block_dims;

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

        // read block dimension environment variables
        if (const char* bx = std::getenv("BLOCK_X")) {
            block_dims.x = std::atoi(bx);
        }
        if (const char* by = std::getenv("BLOCK_Y")) {
            block_dims.y = std::atoi(by);
        }
        if (const char* bz = std::getenv("BLOCK_Z")) {
            block_dims.z = std::atoi(bz);
        }
    }

} // namespace simbi::runtime_config

#endif
