#ifndef SIMBI_STENCIL_VIEW_HPP
#define SIMBI_STENCIL_VIEW_HPP

#include "containers/vector.hpp"
#include "core/base/stencil.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

namespace simbi::base::stencils {
    template <
        Reconstruction Rec,
        typename field_type,
        std::uint64_t Dims = field_type::dimensions>
    struct stencil_view_t {
        using value_type = std::remove_cvref_t<typename field_type::value_type>;
        static constexpr auto stencil_size = base::stencil_size<Rec>();
        using stencil_values_t             = vector_t<value_type, stencil_size>;

        const field_type& field_;
        iarray<Dims> base_coord_;
        std::uint64_t direction_;

        // direct stencil gathering - no intermediate fields!
        stencil_values_t left_values() const
        {
            auto pattern = base::stencil_t<Dims, Rec>::left_pattern(direction_);
            return gather_pattern(pattern);
        }

        stencil_values_t right_values() const
        {
            auto pattern =
                base::stencil_t<Dims, Rec>::right_pattern(direction_);
            return gather_pattern(pattern);
        }

        // both at once for reconstruction
        std::pair<stencil_values_t, stencil_values_t> neighbor_values() const
        {
            return {left_values(), right_values()};
        }

      private:
        stencil_values_t gather_pattern(const auto& pattern) const
        {
            stencil_values_t values;
            for (std::uint64_t ii = 0; ii < stencil_size; ++ii) {
                iarray<Dims> coord = base_coord_;
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    coord[d] += pattern[ii][d];
                }
                values[ii] = field_[coord];
            }
            return values;
        }
    };

    // factory function for clean stencil creation
    template <
        Reconstruction Rec,
        typename field_type,
        std::uint64_t Dims = field_type::dimensions>
    auto make_stencil(
        const field_type& field,
        const iarray<Dims>& coord,
        std::uint64_t dir
    )
    {
        return stencil_view_t<Rec, field_type, Dims>{field, coord, dir};
    }

    // === RECONSTRUCTION INTERFACE ===
    // reconstruction that works directly with stencil values

    template <Reconstruction Rec, typename T>
    T reconstruct_left(
        const vector_t<T, base::stencil_size<Rec>()>& values,
        double theta = 1.5
    )
    {
        if constexpr (Rec == Reconstruction::PCM) {
            return values[0];
        }
        else if constexpr (Rec == Reconstruction::PLM) {
            const auto gradient =
                helpers::plm_gradient(values[0], values[1], values[2], theta);
            return values[1] + gradient * 0.5;
        }
        else {
            static_assert(false, "Reconstruction method not implemented");
        }
    }

    template <Reconstruction Rec, typename T>
    T reconstruct_right(
        const vector_t<T, base::stencil_size<Rec>()>& values,
        double theta = 1.5
    )
    {
        if constexpr (Rec == Reconstruction::PCM) {
            return values[0];
        }
        else if constexpr (Rec == Reconstruction::PLM) {
            auto gradient =
                helpers::plm_gradient(values[0], values[1], values[2], theta);
            return values[1] - 0.5 * gradient;
        }
        else {
            static_assert(false, "Reconstruction method not implemented");
        }
    }
}   // namespace simbi::base::stencils

#endif   // SIMBI_STENCIL_VIEW_HPP
