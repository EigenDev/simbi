/**
 * executor.hpp
 * main execution engine for parallel operations
 */
#ifndef SIMBI_PARALLEL_EXECUTOR_HPP
#define SIMBI_PARALLEL_EXECUTOR_HPP

#include "core/containers/array.hpp"
#include "core/types/alias/alias.hpp"
#include "domain.hpp"
#include "pattern.hpp"
#include "policy.hpp"
#include "tiling.hpp"
#include "view.hpp"
#include <memory>

namespace simbi::parallel {

    /**
     * main executor class that applies operations to data
     */
    template <size_type Dims>
    class executor_t
    {
      public:
        // create executor with specific strategy and policy
        explicit executor_t(
            std::shared_ptr<tiling_strategy_t<Dims>> strategy = nullptr,
            std::shared_ptr<execution_policy_t> policy        = nullptr
        )
            : strategy_(strategy ? strategy : get_optimal_strategy<Dims>()),
              policy_(policy ? policy : get_default_policy())
        {
        }

        // apply an operation to a single view
        template <typename T, typename F>
        void apply(
            data_view_t<T, Dims>& view,
            F&& op,
            const pattern_t<Dims>& pat = pattern_t<Dims>()
        )
        {
            // get halo sizes needed by this pattern
            auto halo_sizes = pat.halo_size();

            // create tiles
            auto tiles = strategy_->create_tiles(view.domain(), halo_sizes);

            // apply operation to each tile
            for (const auto& tile : tiles) {
                data_view_t<T, Dims> tile_view(view.data(), tile);

                // exec in parallel across the tile's domain
                policy_->execute_range(
                    0,
                    tile.size(),
                    [&tile_view, &tile, op](size_type idx) {
                        // convert linear index to position
                        array_t<size_type, Dims> pos{};
                        size_type remaining = idx;

                        // convert linear index to multi-dimensional index
                        for (size_type i = Dims - 1; i > 0; --i) {
                            pos[i] = remaining % tile.shape()[i];
                            remaining /= tile.shape()[i];
                        }
                        pos[0] = remaining;

                        // adjust for tile offset
                        for (size_type i = 0; i < Dims; ++i) {
                            pos[i] += tile.offset()[i];
                        }

                        // apply operation at this position
                        op(tile_view, pos);
                    }
                );
            }
        }

        // apply an operation to two views
        template <typename T1, typename T2, typename F>
        void apply(
            data_view_t<T1, Dims>& view1,
            data_view_t<T2, Dims>& view2,
            F&& op,
            const pattern_t<Dims>& pat = pattern_t<Dims>()
        )
        {
            // get halo sizes needed by this pattern
            auto halo_sizes = pat.halo_size();

            // create tiles based on the first view
            auto tiles = strategy_->create_tiles(view1.domain(), halo_sizes);

            // apply operation to each tile
            for (const auto& tile : tiles) {
                data_view_t<T1, Dims> tile_view1(view1.data(), tile);
                data_view_t<T2, Dims> tile_view2(view2.data(), tile);

                // exec in parallel across the tile's domain
                policy_->execute_range(
                    0,
                    tile.size(),
                    [&tile_view1, &tile_view2, &tile, op](size_type idx) {
                        // convert linear index to position
                        array_t<size_type, Dims> pos{};
                        size_type remaining = idx;

                        // convert linear index to multi-dimensional index
                        for (size_type i = Dims - 1; i > 0; --i) {
                            pos[i] = remaining % tile.shape()[i];
                            remaining /= tile.shape()[i];
                        }
                        pos[0] = remaining;

                        // adjust for tile offset
                        for (size_type i = 0; i < Dims; ++i) {
                            pos[i] += tile.offset()[i];
                        }

                        // apply operation at this position
                        op(tile_view1, tile_view2, pos);
                    }
                );
            }
        }

        // version that works with raw containers by making a view
        template <typename Container, typename F>
        void apply_to_container(
            Container& container,
            const domain_t<Dims>& dom,
            F&& op,
            const pattern_t<Dims>& pat = pattern_t<Dims>()
        )
        {
            auto view = make_view(container, dom);
            apply(view, std::forward<F>(op), pat);
        }

      private:
        std::shared_ptr<tiling_strategy_t<Dims>> strategy_;
        std::shared_ptr<execution_policy_t> policy_;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_PARALLEL_EXECUTOR_HPP
