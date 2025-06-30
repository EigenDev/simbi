#ifndef SIMBI_STENCIL_VIEW_HPP
#define SIMBI_STENCIL_VIEW_HPP

#include "compute/math/field.hpp"
#include "compute/math/index_space.hpp"
#include "core/base/stencil.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"
#include "data/containers/vector.hpp"
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace simbi::base::stencils {
    // === CLEAN STENCIL VIEW ===
    // first-class stencil object that knows how to gather efficiently

    template <typename T, std::uint64_t Dims, Reconstruction Rec>
    struct stencil_view_t {
        const field_t<T, Dims>& field_;
        uarray<Dims> base_coord_;
        std::uint64_t direction_;

        static constexpr auto stencil_size = base::stencil_size<Rec>();
        using stencil_values_t             = vector_t<T, stencil_size>;

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
            for (std::uint64_t i = 0; i < stencil_size; ++i) {
                uarray<Dims> coord = base_coord_;
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    coord[d] += pattern[i][d];
                }
                values[i] = field_(coord);
            }
            return values;
        }
    };

    // factory function for clean stencil creation
    template <Reconstruction Rec, typename T, std::uint64_t Dims>
    auto make_stencil(
        const field_t<T, Dims>& field,
        const uarray<Dims>& coord,
        std::uint64_t dir
    )
    {
        return stencil_view_t<T, Dims, Rec>{field, coord, dir};
    }

    // === RECONSTRUCTION INTERFACE ===
    // clean reconstruction that works directly with stencil values

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
        else if constexpr (Rec == Reconstruction::PPM) {
            throw std::runtime_error(
                "PPM reconstruction not implemented in this context"
            );
            // return helpers::ppm_reconstruction_left(
            //     values[0],
            //     values[1],
            //     values[2],
            //     values[3]
            // );
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
        else if constexpr (Rec == Reconstruction::PPM) {
            throw std::runtime_error(
                "PPM reconstruction not implemented in this context"
            );
            // return helpers::ppm_reconstruction_right(
            //     values[0],
            //     values[1],
            //     values[2],
            //     values[3]
            // );
        }
        else {
            static_assert(false, "Reconstruction method not implemented");
        }
    }

    // === BONUS: VECTORIZED STENCIL OPERATIONS ===
    // for when you need many stencils at once

    template <Reconstruction Rec, typename T, std::uint64_t Dims>
    class batch_stencil_extractor_t
    {
        const field_t<T, Dims>& field_;
        std::uint64_t direction_;

      public:
        batch_stencil_extractor_t(
            const field_t<T, Dims>& field,
            std::uint64_t dir
        )
            : field_(field), direction_(dir)
        {
        }

        // extract stencils for an entire domain at once
        auto extract_all(const index_space_t<Dims>& domain) const
        {
            using stencil_t = vector_t<T, base::stencil_size<Rec>()>;

            auto left_stencils  = make_field<stencil_t, Dims>(domain.shape());
            auto right_stencils = make_field<stencil_t, Dims>(domain.shape());

            // Fill stencil arrays
            for (auto coord : domain) {   // Assuming domain iteration
                auto stencil = make_stencil<Rec>(field_, coord, direction_);
                auto [left_vals, right_vals] = stencil.neighbor_values();

                left_stencils.at(coord)  = left_vals;
                right_stencils.at(coord) = right_vals;
            }

            return std::make_pair(left_stencils, right_stencils);
        }
    };

    // === COMPILE-TIME STENCIL VALIDATION ===
    // ensure stencil fits within field bounds

    template <Reconstruction Rec, std::uint64_t Dims>
    constexpr bool stencil_fits(
        const index_space_t<Dims>& field_domain,
        const uarray<Dims>& coord,
        std::uint64_t direction
    )
    {
        constexpr auto left_pattern =
            base::stencil_t<Dims, Rec>::left_pattern(direction);
        constexpr auto right_pattern =
            base::stencil_t<Dims, Rec>::right_pattern(direction);

        // Check if all stencil points are within bounds
        for (std::uint64_t i = 0; i < base::stencil_size<Rec>(); ++i) {
            uarray<Dims> left_coord = coord, right_coord = coord;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                left_coord[d] += left_pattern[i][d];
                right_coord[d] += right_pattern[i][d];
            }

            if (!field_domain.contains(left_coord) ||
                !field_domain.contains(right_coord)) {
                return false;
            }
        }
        return true;
    }

    // === SAFE STENCIL ACCESS ===
    // wWith automatic bounds checking

    template <Reconstruction Rec, typename T, std::uint64_t Dims>
    auto make_safe_stencil(
        const field_t<T, Dims>& field,
        const uarray<Dims>& coord,
        std::uint64_t dir
    )
    {
        static_assert(
            stencil_fits<Rec, Dims>(field.domain(), coord, dir),
            "Stencil extends beyond field bounds"
        );
        return make_stencil<Rec>(field, coord, dir);
    }

}   // namespace simbi::base::stencils

#endif   // SIMBI_STENCIL_VIEW_HPP
