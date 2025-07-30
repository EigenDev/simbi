/**
 * policy.hpp
 * execution policy abstraction
 */

#ifndef SIMBI_CORE_PARALLEL_POLICY_HPP
#define SIMBI_CORE_PARALLEL_POLICY_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "base/coordinate.hpp"
#include "mesh/grid_topology.hpp"
#include "parallel/par_config.hpp"
#include "parallel/tiling/tiling_strategy.hpp"
#include "types/alias.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace simbi::parallel {

    /**
     * execution_policy_t - base class for execution strategies
     */

    template <std::uint64_t Dims>
    class execution_policy_t
    {
      public:
        virtual ~execution_policy_t() = default;

        // Core interface - execute kernel over domain with tiling
        template <typename Kernel>
        void execute_domain(
            const mesh::grid_topology_t<Dims>& topology,
            std::shared_ptr<tiling_strategy_t<Dims>> tiling,
            Kernel&& kernel
        ) const
        {
            execute_domain_impl(topology, tiling, std::forward<Kernel>(kernel));
        }

        // Traditional range-based execution
        template <typename Func>
        void
        execute_range(std::uint64_t begin, std::uint64_t end, Func&& func) const
        {
            execute_range_impl(begin, end, std::forward<Func>(func));
        }

      protected:
        virtual void execute_domain_impl(
            const mesh::grid_topology_t<Dims>& topology,
            std::shared_ptr<tiling_strategy_t<Dims>> tiling,
            std::function<void(const base::coordinate_t<Dims>&)> kernel
        ) const = 0;

        virtual void execute_range_impl(
            std::uint64_t begin,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const = 0;
    };

    /**
     * configuration for execution policies
     */
    template <std::uint64_t Dims>
    struct policy_config_t {
        std::uint64_t effective_dim = 1;   // effective dimension for tiling
        base::coordinate_t<Dims> ghost_zones = {0};   // ghost zones for tiling
        // hardware settings
        adapter::types::dim3 threads_per_block =
            parallel::get_default_block_size(effective_dim);
        std::uint64_t shared_memory_bytes = 0;

        // device selection
        std::vector<std::int64_t> device_ids = {0};

        // execution settings
        schedule_type_t schedule_type = schedule_type_t::automatic;
        std::uint64_t chunk_size      = 0;   // 0 means auto-determine

        // memory settings
        memory_type_t memory_type = memory_type_t::device;

        // synchronization
        execution_mode_t execution_mode = execution_mode_t::sync;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_POLICY_HPP
