/**
 * executor.hpp
 * base executor interface for parallel operations
 */

#ifndef SIMBI_CORE_PARALLEL_EXECUTOR_HPP
#define SIMBI_CORE_PARALLEL_EXECUTOR_HPP

#include "core/graph/graph_concept.hpp"
#include "core/mesh/grid_topology.hpp"
#include "core/types/alias.hpp"
#include "system/parallel/par_config.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace simbi::parallel {

    // forward declarations
    template <std::uint64_t Dims>
        requires graph::valid_dimension<Dims>
    class tiling_strategy_t;
    class execution_policy_t;

    /**
     * base class for all executors
     * handles executing operations on various hardware
     */
    class executor_t
    {
      public:
        virtual ~executor_t() = default;

        // execute a function over a range
        template <typename Func>
        void
        execute_range(std::uint64_t start, std::uint64_t end, Func&& func) const
        {
            execute_range_impl(start, end, std::forward<Func>(func));
        }

        // asynchronous execution
        template <typename Func>
        void
        execute_async(std::uint64_t start, std::uint64_t end, Func&& func) const
        {
            execute_async_impl(start, end, std::forward<Func>(func));
        }

        // wait for all operations to complete
        virtual void synchronize() const = 0;

        // factory method to get appropriate executor for current hardware
        static std::shared_ptr<executor_t> get_default();

      protected:
        // implementation-specific methods
        virtual void execute_range_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const = 0;

        virtual void execute_async_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const = 0;
    };

    /**
     * tiled_executor_t - executes operations in hardware-optimized tiles
     */
    template <std::uint64_t Dims>
        requires graph::valid_dimension<Dims>
    class tiled_executor_t
    {
      public:
        // create with specified strategy and executor
        tiled_executor_t(
            std::shared_ptr<tiling_strategy_t<Dims>> strategy,
            std::shared_ptr<executor_t> executor
        )
            : strategy_(strategy), executor_(executor)
        {
        }

        // create with default strategy and executor
        tiled_executor_t();

        // execute operation on a topology with tiling
        template <typename Func>
        void
        execute(const mesh::grid_topology_t<Dims>& topology, Func&& func) const;

        // synchronize all operations
        void synchronize() const { executor_->synchronize(); }

      private:
        std::shared_ptr<tiling_strategy_t<Dims>> strategy_;
        std::shared_ptr<executor_t> executor_;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_EXECUTOR_HPP
