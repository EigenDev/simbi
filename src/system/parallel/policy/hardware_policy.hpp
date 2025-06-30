/**
 * hardware_policy.hpp
 * hardware-specific execution policies
 */

#ifndef SIMBI_CORE_PARALLEL_HARDWARE_POLICY_HPP
#define SIMBI_CORE_PARALLEL_HARDWARE_POLICY_HPP

#include "config.hpp"
#include "core/base/coordinate.hpp"
#include "core/mesh/grid_topology.hpp"
#include "core/types/alias.hpp"
#include "system/adapter/device_adapter_api.hpp"
#include "system/adapter/device_types.hpp"
#include "system/parallel/par_config.hpp"
#include "system/parallel/policy/policy.hpp"
#include "system/parallel/tiling/tiling_strategy.hpp"
#include "util/parallel/thread_pool.hpp"
#include <cstddef>
#include <functional>
#include <memory>

namespace simbi::parallel {

    /**
     * cpu_policy_t - optimized for CPU execution
     */
    template <std::uint64_t Dims>
    class cpu_policy_t : public execution_policy_t<Dims>
    {
      public:
        explicit cpu_policy_t(policy_config_t<Dims> config = {})
            : config_(config)
        {
        }

      protected:
        void execute_domain_impl(
            const mesh::grid_topology_t<Dims>& topology,
            std::shared_ptr<tiling_strategy_t<Dims>> tiling,
            std::function<void(const base::coordinate_t<Dims>&)> kernel
        ) const override
        {
            // create CPU-optimized tiles
            auto tiles = tiling->create_tiles(topology, config_.ghost_zones);

            // use thread pool for tile-level parallelism
            pooling::get_thread_pool().parallel_for(
                tiles.size(),
                [&](std::uint64_t tile_idx) {
                    const auto& tile = tiles[tile_idx];
                    auto coordinates = tile.get_active_cell_coordinates();

                    // Execute kernel on each coordinate in this tile
                    for (const auto& coord : coordinates) {
                        kernel(coord);
                    }
                }
            );
        }

        void execute_range_impl(
            std::uint64_t begin,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override
        {
            pooling::get_thread_pool().parallel_for(begin, end, func);
        }

      private:
        policy_config_t<Dims> config_;
    };

    /**
     * gpu_policy_t - optimized for GPU execution
     */
    template <std::uint64_t Dims>
    class gpu_policy_t : public execution_policy_t<Dims>
    {
      public:
        explicit gpu_policy_t(policy_config_t<Dims> config = {})
            : config_(config)
        {
        }

      protected:
        void execute_domain_impl(
            const mesh::grid_topology_t<Dims>& topology,
            std::shared_ptr<tiling_strategy_t<Dims>> tiling,
            std::function<void(const base::coordinate_t<Dims>&)> kernel
        ) const override
        {
            // Create GPU-optimized tiles (shared memory sized)
            auto tiles = tiling->create_tiles(topology, config_.ghost_zones);

            for (const auto& tile : tiles) {
                auto coordinates = tile.get_active_cell_coordinates();

                // Calculate grid/block dimensions
                auto grid_dim = grid::calculate_grid(
                    coordinates.size(),
                    config_.threads_per_block
                );
                auto block_dim = config_.threads_per_block;

                // ;aunch using adapter
                auto launch_config = grid::config(
                    grid_dim,
                    block_dim,
                    config_.shared_memory_bytes
                );

                grid::launch(
                    [=] DEV(std::uint64_t idx) {
                        if (idx < coordinates.size()) {
                            kernel(coordinates[idx]);
                        }
                    },
                    launch_config
                );
            }

            // Handle multi-device if configured
            if (config_.device_ids.size() > 1) {
                // Synchronize all devices
                for (std::int64_t device_id : config_.device_ids) {
                    gpu::api::set_device(device_id);
                    gpu::api::device_synch();
                }
            }
        }

        void execute_range_impl(
            std::uint64_t begin,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override
        {
            auto grid_dim =
                grid::calculate_grid(end - begin, config_.threads_per_block);
            auto block_dim     = config_.threads_per_block;
            auto launch_config = grid::config(grid_dim, block_dim);

            grid::launch(
                [=] DEV(std::uint64_t idx) {
                    if (idx < end - begin) {
                        func(begin + idx);
                    }
                },
                launch_config
            );
        }

      private:
        policy_config_t<Dims> config_;
    };

    template <std::uint64_t Dims>
    auto create_policy()
    {
        if constexpr (platform::is_gpu) {
            return gpu_policy_t<Dims>{};
        }
        else {
            return cpu_policy_t<Dims>{};
        }
    }

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_HARDWARE_POLICY_HPP
