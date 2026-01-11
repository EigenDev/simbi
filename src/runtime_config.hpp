// =============================================================================
// runtime_config.hpp
//
// runtime mutable configuration state.
//
// usage:
//   runtime::use_omp = true;
//   if (runtime::use_omp) { /* enable openmp */ }
//
// initialization:
//   runtime::initialize(); // reads environment variables
// =============================================================================

#ifndef SIMBI_RUNTIME_CONFIG_HPP
#define SIMBI_RUNTIME_CONFIG_HPP

#include <cstdint>
#include <cstdlib>
#include <string>

namespace simbi::runtime {

    inline bool use_omp = false;

    struct block_dims_t
    {
        std::uint32_t x = 256;
        std::uint32_t y = 1;
        std::uint32_t z = 1;

        std::uint64_t total_threads() const
        {
            return x * y * z;
        }

        auto to_dim3() const
        {
#if defined(__CUDACC__) || defined(__HIPCC__)
            return dim3(x, y, z);
#endif
        }
    };

    inline block_dims_t block_dims;

    inline void initialize()
    {
        if (const char* env = std::getenv("USE_OMP")) {
            std::string val(env);
            use_omp = (val == "1" || val == "true" || val == "TRUE");
        }

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

} // namespace simbi::runtime

#endif
