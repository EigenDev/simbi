#ifndef SIMBI_VIEW_STATE_TILE_HPP
#define SIMBI_VIEW_STATE_TILE_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/containers/collapsable.hpp"
#include "core/containers/vector.hpp"
#include "core/index/global_index.hpp"
#include "core/index/stencil_patterns.hpp"
#include "core/memory/values/state_value.hpp"
#include "core/parallel/view.hpp"
#include "core/types/alias/alias.hpp"
#include "core/utility/enums.hpp"
#include <cstddef>
#include <cstdint>

namespace simbi::views {
    using namespace simbi::parallel;
    using namespace simbi::index;

    template <Regime R, size_type Dims>
    class state_accessor_t
    {
      private:
        const data_view_t<real, Dims>& view_;
        size_type nzones_;

      public:
        state_accessor_t(const data_view_t<real, Dims>& view, size_type nzones)
            : view_(view), nzones_(nzones)
        {
        }

        // access specific variable at a spatial position
        template <typename... Indices>
        const real& variable(size_type var_idx, Indices... indices) const
        {
            size_type spatial_idx =
                view_.at({static_cast<size_type>(indices)...});
            return view_[var_idx * nzones_ + spatial_idx];
        }

        // access variables at specific position with proper offsets
        template <typename... Indices>
        auto primitive_at(Indices... indices) const
        {
            // create position array
            array_t<int64_t, Dims> pos =
                collapsable_t<int64_t, Dims>{static_cast<int64_t>(indices)...};

            // // calculate flat index in first variable array (rho)
            size_type base_idx = view_.linear_index(pos);

            // // access density directly
            const real& rho = view_[base_idx];

            // create velocity vector (access each component)
            spatial_vector_t<real, Dims> vel;
            for (size_type d = 0; d < Dims; ++d) {
                vel[d] = view_[(1 + d) * nzones_ + base_idx];
            }

            // access pressure
            const real& pre = view_[(1 + Dims) * nzones_ + base_idx];

            // access tracer (if present)
            const real& chi = view_[(2 + Dims) * nzones_ + base_idx];

            // return primitive state value
            return values::primitive_value_t<R, Dims>{rho, vel, pre, chi};
            // return values::primitive_value_t<R, Dims>{0.0};
        }
    };

    template <Regime R, size_type Dims, direction_t Dir>
    auto primitive_stencil(
        const data_view_t<real, Dims>& tile,
        const index::cell_index_t& idx,
        size_type domain_size
    )
    {
        // Create state accessor
        state_accessor_t<R, Dims> accessor(tile, domain_size);

        // Get three-point stencil for the specified direction
        auto stencil = index::three_point<Dir>(idx);

        // Access left and right states directly using the accessor
        auto left_state = accessor.primitive_at(
            stencil[0].x1,
            (Dims > 1) ? stencil[0].x2 : 0,
            (Dims > 2) ? stencil[0].x3 : 0
        );

        auto right_state = accessor.primitive_at(
            stencil[2].x1,
            (Dims > 1) ? stencil[2].x2 : 0,
            (Dims > 2) ? stencil[2].x3 : 0
        );

        return std::make_pair(left_state, right_state);
    }

}   // namespace simbi::views

#endif
