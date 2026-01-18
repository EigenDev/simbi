// =============================================================================
// stencil_view.hpp
//
// provides a view into a data field for stencil-based operations.
// the `stencil_view_t` class template allows for the extraction of neighbor
// cell data ("stencils") at a given face, which is essential for higher-order
// reconstruction methods. it also includes the reconstruction implementation.
//
// usage:
//   auto stencil = stencils::make_stencil<rec_t>(field, coord, dir);
//   auto [left_vals, right_vals] = stencil.neighbor_values();
//   auto q_l = stencils::reconstruct_left<rec_t>(left_vals);
//   auto q_r = stencils::reconstruct_right<rec_t>(right_vals);
// =============================================================================
#pragma once

#include "base/stencil.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp" // for DEV
#include "utility/enums.hpp"
#include "utility/helpers.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace simbi::base::stencils {
    template <reconstruction_t Rec, typename field_type, std::uint64_t Rank = field_type::rank>
    struct stencil_view_t
    {
        using value_type                   = std::remove_cvref_t<typename field_type::value_type>;
        static constexpr auto stencil_size = base::stencil_size<Rec>();
        using stencil_values_t             = vector_t<value_type, stencil_size>;

        const field_type& field_;
        iarray<Rank>      face_coord_;
        std::uint64_t     direction_;

        stencil_values_t DEV left_values() const
        {
            auto pattern = base::stencil_t<Rank, Rec>::left_pattern(direction_);
            return gather_pattern(pattern);
        }

        stencil_values_t DEV right_values() const
        {
            auto pattern = base::stencil_t<Rank, Rec>::right_pattern(direction_);
            return gather_pattern(pattern);
        }

        // both at once for reconstruction
        std::pair<stencil_values_t, stencil_values_t> DEV neighbor_values() const
        {
            return {left_values(), right_values()};
        }

      private:
        stencil_values_t DEV gather_pattern(const auto& pattern) const
        {
            stencil_values_t values;
            for (std::uint64_t ii = 0; ii < stencil_size; ++ii) {
                iarray<Rank> cell_coord = face_coord_;
                for (std::uint64_t d = 0; d < Rank; ++d) {
                    cell_coord[d] += pattern[ii][d];
                }
                values[ii] = field_(cell_coord);
            }
            return values;
        }
    };

    template <reconstruction_t Rec, typename field_type, std::uint64_t Rank = field_type::rank>
    DEV auto make_stencil(const field_type& field, const iarray<Rank>& coord, std::uint64_t dir)
    {
        return stencil_view_t<Rec, field_type, Rank>{field, coord, dir};
    }

    // === RECONSTRUCTION INTERFACE ===
    template <reconstruction_t Rec, typename T>
    DEV T reconstruct_left(const vector_t<T, base::stencil_size<Rec>()>& values, double theta = 1.5)
    {
        if constexpr (Rec == reconstruction_t::PCM) {
            return values[0];
        }
        else if constexpr (Rec == reconstruction_t::PLM) {
            const auto gradient = helpers::plm_gradient(values[0], values[1], values[2], theta);
            return values[1] + gradient * 0.5;
        }
        else {
            // lambda trick
            []<bool flag = false>() {
                static_assert(flag, "reconstruction_t method not implemented");
            }();
        }
    }

    template <reconstruction_t Rec, typename T>
    DEV T
    reconstruct_right(const vector_t<T, base::stencil_size<Rec>()>& values, double theta = 1.5)
    {
        if constexpr (Rec == reconstruction_t::PCM) {
            return values[0];
        }
        else if constexpr (Rec == reconstruction_t::PLM) {
            auto gradient = helpers::plm_gradient(values[0], values[1], values[2], theta);
            return values[1] - 0.5 * gradient;
        }
        else {
            // lambda trick to satisfy nvcc
            []<bool flag = false>() {
                static_assert(flag, "reconstruction_t method not implemented");
            }();
        }
    }
} // namespace simbi::base::stencils
