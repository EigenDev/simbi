/**
 * tiling_strategy.hpp
 * abstract base class for domain decomposition strategies
 */

#ifndef SIMBI_CORE_PARALLEL_TILING_STRATEGY_HPP
#define SIMBI_CORE_PARALLEL_TILING_STRATEGY_HPP

#include "core/base/coordinate.hpp"   // for coordinate_t
#include "mesh/grid_topology.hpp"
// #include "system/parallel/par_config.hpp"
#include <cstddef>
#include <memory>
#include <vector>
namespace simbi::parallel {

    /**
     * abstract base class for tiling strategies
     * decomposes a computational domain into hardware-optimized tiles
     */
    template <std::uint64_t Dims>
        requires concepts::valid_dimension<Dims>
    class tiling_strategy_t
    {
      public:
        virtual ~tiling_strategy_t() = default;

        // create tiles from a graph topology
        virtual std::vector<mesh::grid_topology_t<Dims>> create_tiles(
            const mesh::grid_topology_t<Dims>& topology,
            const base::coordinate_t<Dims>& ghost_zones = {}
        ) const = 0;

        // factory method to get the optimal strategy for the current hardware
        static std::shared_ptr<tiling_strategy_t<Dims>> get_optimal();
    };

    // convenience aliases for common dimensions
    using tiling_strategy_1d_t = tiling_strategy_t<1>;
    using tiling_strategy_2d_t = tiling_strategy_t<2>;
    using tiling_strategy_3d_t = tiling_strategy_t<3>;

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_TILING_STRATEGY_HPP
