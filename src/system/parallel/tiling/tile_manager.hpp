/**
 * tile_manager.hpp
 * manages tile execution and coordination
 */

#ifndef SIMBI_CORE_PARALLEL_TILE_MANAGER_HPP
#define SIMBI_CORE_PARALLEL_TILE_MANAGER_HPP

#include "core/types/alias.hpp"
#include "system/parallel/par_config.hpp"
#include "system/parallel/tiling/tiling_strategy.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace simbi::parallel {

    /**
     * tile_manager_t - coordinates tile execution across hardware
     */
    template <std::uint64_t Dims>
    class tile_manager_t
    {
      public:
        // construct with strategy and optional explicit domain
        explicit tile_manager_t(
            std::shared_ptr<tiling_strategy_t<Dims>> strategy
        );

        // process a domain using the provided function
        template <typename Domain, typename Func>
        void process(
            const Domain& domain,
            Func&& func,
            const std::array<std::uint64_t, Dims>& halo_sizes = {}
        );

        // process with multiple functions (pipeline)
        template <typename Domain, typename... Funcs>
        void pipeline(
            const Domain& domain,
            const std::array<std::uint64_t, Dims>& halo_sizes,
            Funcs&&... funcs
        );

      private:
        std::shared_ptr<tiling_strategy_t<Dims>> strategy_;

        // execute a function on each tile
        template <typename TileType, typename Func>
        void execute_on_tiles(const std::vector<TileType>& tiles, Func&& func);
    };

    // convenience type aliases
    using tile_manager_1d_t = tile_manager_t<1>;
    using tile_manager_2d_t = tile_manager_t<2>;
    using tile_manager_3d_t = tile_manager_t<3>;

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_TILE_MANAGER_HPP
