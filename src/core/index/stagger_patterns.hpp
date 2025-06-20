// =============================================================================
// index/stagger_patterns.hpp
// =============================================================================
#ifndef STAGGER_PATTERNS_HPP
#define STAGGER_PATTERNS_HPP

#include "global_index.hpp"
#include "stencil_patterns.hpp"
#include <cstddef>

namespace simbi::index {
    // inter-stagger conversion patterns for finite volume operations

    // cell-to-face gradient stencils (for momentum equations)
    struct cell_to_face_x1_gradient_t {
        static constexpr auto pattern = stencil_pattern_t<2>{{
          {{-1, 0, 0}, {0, 0, 0}}   // cell(i-1) and cell(i) for face(i-1/2)
        }};
    };

    struct cell_to_face_x2_gradient_t {
        static constexpr auto pattern = stencil_pattern_t<2>{{
          {{0, -1, 0}, {0, 0, 0}}   // cell(j-1) and cell(j) for face(j-1/2)
        }};
    };

    struct cell_to_face_x3_gradient_t {
        static constexpr auto pattern = stencil_pattern_t<2>{{
          {{0, 0, -1}, {0, 0, 0}}   // cell(k-1) and cell(k) for face(k-1/2)
        }};
    };

    // face-to-cell divergence stencils (for continuity equation)
    struct face_to_cell_divergence_x1_t {
        static constexpr auto pattern = stencil_pattern_t<2>{{
          {{0, 0, 0}, {1, 0, 0}}   // face(i-1/2) and face(i+1/2) for cell(i)
        }};
    };

    struct face_to_cell_divergence_x2_t {
        static constexpr auto pattern = stencil_pattern_t<2>{{
          {{0, 0, 0}, {0, 1, 0}}   // face(j-1/2) and face(j+1/2) for cell(j)
        }};
    };

    struct face_to_cell_divergence_x3_t {
        static constexpr auto pattern = stencil_pattern_t<2>{{
          {{0, 0, 0}, {0, 0, 1}}   // face(k-1/2) and face(k+1/2) for cell(k)
        }};
    };

    // inter-stagger stencil view (different input/output stagger types)
    template <typename Pattern, stagger_t InputStagger, stagger_t OutputStagger>
    class inter_stagger_view_t
    {
        staggered_index_t<OutputStagger> center_;

      public:
        constexpr explicit inter_stagger_view_t(
            staggered_index_t<OutputStagger> center
        )
            : center_(center)
        {
        }

        constexpr staggered_index_t<InputStagger> operator[](size_t i) const
        {
            auto offset = Pattern::pattern.offsets[i];
            return staggered_index_t<InputStagger>{
              center_.x1 + offset[0],
              center_.x2 + offset[1],
              center_.x3 + offset[2],
              center_.level
            };
        }

        static constexpr size_t size() { return Pattern::pattern.size; }
    };

    // factory functions for inter-stagger operations
    template <direction_t Dir>
    constexpr auto cell_to_face_gradient(face_x1_index_t face_idx)
        requires(Dir == direction_t::x1)
    {
        return inter_stagger_view_t<
            cell_to_face_x1_gradient_t,
            stagger_t::cell,
            stagger_t::face_x1>{face_idx};
    }

    template <direction_t Dir>
    constexpr auto cell_to_face_gradient(face_x2_index_t face_idx)
        requires(Dir == direction_t::x2)
    {
        return inter_stagger_view_t<
            cell_to_face_x2_gradient_t,
            stagger_t::cell,
            stagger_t::face_x2>{face_idx};
    }

    template <direction_t Dir>
    constexpr auto cell_to_face_gradient(face_x3_index_t face_idx)
        requires(Dir == direction_t::x3)
    {
        return inter_stagger_view_t<
            cell_to_face_x3_gradient_t,
            stagger_t::cell,
            stagger_t::face_x3>{face_idx};
    }

    template <direction_t Dir>
    constexpr auto face_to_cell_divergence(cell_index_t cell_idx)
    {
        if constexpr (Dir == direction_t::x1) {
            return inter_stagger_view_t<
                face_to_cell_divergence_x1_t,
                stagger_t::face_x1,
                stagger_t::cell>{cell_idx};
        }
        else if constexpr (Dir == direction_t::x2) {
            return inter_stagger_view_t<
                face_to_cell_divergence_x2_t,
                stagger_t::face_x2,
                stagger_t::cell>{cell_idx};
        }
        else if constexpr (Dir == direction_t::x3) {
            return inter_stagger_view_t<
                face_to_cell_divergence_x3_t,
                stagger_t::face_x3,
                stagger_t::cell>{cell_idx};
        }
    }

}   // namespace simbi::index
#endif
