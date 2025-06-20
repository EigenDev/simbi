// =============================================================================
// index/grid_policies.hpp
// =============================================================================

#ifndef GRID_POLICIES_HPP
#define GRID_POLICIES_HPP

#include "config.hpp"
#include "global_index.hpp"
#include <concepts>
#include <cstdint>

namespace simbi::index {

    // policy concept - what grid policies must provide
    template <typename T>
    concept grid_policy_c = requires(T policy, cell_index_t idx) {
        // get reference to data at index
        { policy.local_data(idx) } -> std::convertible_to<real&>;

        // check if index is valid/in-bounds
        { policy.is_valid(idx) } -> std::convertible_to<bool>;

        // get grid bounds
        { policy.bounds() };
    };

    // simple cartesian policy for uniform structured grids
    template <typename data_t>
    struct simple_cartesian_t {
        data_t& data_;
        int64_t n1_, n2_, n3_;

        constexpr simple_cartesian_t(
            data_t& data,
            int64_t n1,
            int64_t n2,
            int64_t n3
        )
            : data_(data), n1_(n1), n2_(n2), n3_(n3)
        {
        }

        // direct array access - fastest path for uniform grids
        constexpr real& local_data(cell_index_t idx) const
        {
            return data_[idx.x1 * n2_ * n3_ + idx.x2 * n3_ + idx.x3];
        }

        constexpr bool is_valid(cell_index_t idx) const
        {
            return idx.x1 >= 0 && idx.x1 < n1_ && idx.x2 >= 0 && idx.x2 < n2_ &&
                   idx.x3 >= 0 && idx.x3 < n3_ &&
                   idx.level == 0;   // only level 0 for uniform grids
        }

        struct bounds_t {
            int64_t n1, n2, n3;
            constexpr cell_index_t start() const { return {0, 0, 0, 0}; }
            constexpr cell_index_t end() const { return {n1, n2, n3, 0}; }
        };

        constexpr bounds_t bounds() const { return {n1_, n2_, n3_}; }
    };

    // fmr policy for static multi-resolution (future implementation)
    template <typename level_manager_t>
    struct fmr_cartesian_t {
        level_manager_t& levels_;

        // will handle static level hierarchy, inter-level operations
        real& local_data(cell_index_t idx) const
        {
            // auto level_data = levels_.get_level(idx.level);
            // auto local_idx = level_data.global_to_local(idx);
            // return level_data[local_idx];
            static real dummy = 0.0;
            return dummy;   // placeholder for now
        }

        bool is_valid(cell_index_t idx) const
        {
            return levels_.has_level(idx.level) &&
                   levels_.get_level(idx.level).contains(idx);
        }

        auto bounds() const { return levels_.global_bounds(); }
    };

}   // namespace simbi::index
#endif
