#ifndef HET_POLICY_HPP
#define HET_POLICY_HPP

#include "hesi/core/types.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace simbi::het::exec {
    struct block_config_t {
        static constexpr dim3_t default_block_3d{16, 8, 4};
        static constexpr dim3_t default_block_2d{16, 16, 1};
        static constexpr std::uint64_t default_block_1d = 256;
        static constexpr std::uint64_t max_grid_dim     = 65535;
    };

    // defines the shape of the computation grid
    struct launch_policy_t {
        dim3_t grid;
        dim3_t block;
        std::size_t shared_mem_bytes = 0;

        // manual configuration
        constexpr launch_policy_t(dim3_t g, dim3_t b, std::size_t smem = 0)
            : grid(g), block(b), shared_mem_bytes(smem)
        {
        }

        // helper for 1d linear arrays
        template <std::integral T>
        static launch_policy_t
        linear(T total_threads, T block_size = block_config_t::default_block_1d)
        {
            std::uint32_t b = block_size;
            std::uint32_t g = std::min<std::uint64_t>(
                (total_threads + b - 1) / b,
                (T) block_config_t::max_grid_dim
            );
            return launch_policy_t({g, 1, 1}, {b, 1, 1});
        }

        // helper for 2d surfaces
        template <std::integral T>
        static launch_policy_t surface(
            T width,
            T height,
            dim3_t block_dim = block_config_t::default_block_2d
        )
        {
            auto gx = static_cast<std::uint32_t>(
                (width + block_dim.x - 1) / block_dim.x
            );
            auto gy = static_cast<std::uint32_t>(
                (height + block_dim.y - 1) / block_dim.x
            );
            return launch_policy_t({gx, gy, 1}, block_dim);
        }

        template <std::integral T>
        static launch_policy_t
        volume(T width, T height, T depth, dim3_t block_dim = {8, 8, 8})
        {
            auto gx = static_cast<std::uint32_t>(
                (width + block_dim.x - 1) / block_dim.x
            );
            auto gy = static_cast<std::uint32_t>(
                (height + block_dim.y - 1) / block_dim.y
            );
            auto gz = static_cast<std::uint32_t>(
                (depth + block_dim.z - 1) / block_dim.z
            );
            return launch_policy_t({gx, gy, gz}, block_dim);
        }
    };

}   // namespace simbi::het::exec

#endif   // HETERO_POLICY_HPP
