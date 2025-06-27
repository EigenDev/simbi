/**
 * adaptive_tiling.hpp
 * hardware-adaptive tiling strategies
 */

#ifndef SIMBI_CORE_PARALLEL_ADAPTIVE_TILING_HPP
#define SIMBI_CORE_PARALLEL_ADAPTIVE_TILING_HPP

#include "config.hpp"
#include "core/types/alias.hpp"
#include "system/parallel/par_config.hpp"
#include "system/parallel/tiling/block_tiling.hpp"
#include <array>
#include <memory>
#include <type_traits>

namespace simbi::parallel {

    /**
     * cpu_cache_tiling_t - optimizes tile sizes for CPU cache
     */
    template <std::uint64_t Dims>
    class cpu_cache_tiling_t : public block_tiling_t<Dims>
    {
      public:
        // construct with element size for cache calculations
        explicit cpu_cache_tiling_t(
            std::uint64_t bytes_per_element = sizeof(real)
        );

        // create tiles optimized for CPU cache
        static std::shared_ptr<cpu_cache_tiling_t<Dims>> create();

      private:
        // calculate cache-optimized tile sizes
        std::array<std::uint64_t, Dims> calculate_tile_sizes() const;

        std::uint64_t bytes_per_element_;
    };

    /**
     * gpu_block_tiling_t - optimizes tile sizes for GPU thread blocks
     */
    template <std::uint64_t Dims>
    class gpu_block_tiling_t : public block_tiling_t<Dims>
    {
      public:
        // construct with device id for device-specific optimizations
        explicit gpu_block_tiling_t(std::int64_t device_id = 0);

        // create tiles optimized for GPU execution
        static std::shared_ptr<gpu_block_tiling_t<Dims>> create();

      private:
        // calculate GPU-optimized tile sizes
        std::array<std::uint64_t, Dims> calculate_tile_sizes() const;

        std::int64_t device_id_;
    };

    // convenience type aliases
    template <std::uint64_t Dims>
    using adaptive_tiling_t = std::conditional_t<
        platform::is_gpu,
        gpu_block_tiling_t<Dims>,
        cpu_cache_tiling_t<Dims>>;

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_ADAPTIVE_TILING_HPP
