#ifndef SIMBI_CORE_GRAPH_STENCIL_HPP
#define SIMBI_CORE_GRAPH_STENCIL_HPP

#include "core/base/concepts.hpp"
#include "core/utility/enums.hpp"
#include "data/containers/vector.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace simbi::base {
    // compile-time stencil size calculation
    template <Reconstruction Rec>
    constexpr std::uint64_t stencil_size()
    {
        if constexpr (Rec == Reconstruction::PCM) {
            return 1;
        }
        else if constexpr (Rec == Reconstruction::PLM) {
            return 3;
        }
        else if constexpr (Rec == Reconstruction::PPM) {
            return 4;
        }
        else if constexpr (Rec == Reconstruction::WENO3) {
            return 3;
        }
        else if constexpr (Rec == Reconstruction::WENO5) {
            return 5;
        }
        else if constexpr (Rec == Reconstruction::WENO7) {
            return 7;
        }
        else if constexpr (Rec == Reconstruction::WENO9) {
            return 9;
        }
        else {
            static_assert(false, "Unsupported reconstruction order");
        }
    }

    // compile-time stencil pattern generation
    template <std::uint64_t Dims, Reconstruction Rec>
    struct stencil_t {
        static constexpr std::uint64_t size = stencil_size<Rec>();
        using coord_array_t                 = vector_t<uarray<Dims>, size>;

        // generate left reconstruction pattern
        static constexpr coord_array_t left_pattern(std::uint64_t direction)
        {
            coord_array_t pattern{};

            if constexpr (Rec == Reconstruction::PCM) {
                pattern[0][direction] = -1;   // Use left cell
            }
            else if constexpr (Rec == Reconstruction::PLM) {
                // PLM: i-1, i, i+1
                pattern[0][direction] = -2;
                pattern[1][direction] = -1;
                pattern[2][direction] = 0;
            }
            else if constexpr (Rec == Reconstruction::PPM) {
                // PPM: i-2, i-1, i, i+1
                pattern[0][direction] = -3;
                pattern[1][direction] = -2;
                pattern[2][direction] = -1;
                pattern[3][direction] = 0;
            }

            return pattern;
        }

        // generate right reconstruction pattern
        static constexpr coord_array_t right_pattern(std::uint64_t direction)
        {
            coord_array_t pattern{};

            if constexpr (Rec == Reconstruction::PCM) {
                pattern[0][direction] = 0;   // Use right cell
            }
            else if constexpr (Rec == Reconstruction::PLM) {
                // PLM: i, i+1, i+2
                pattern[0][direction] = -1;
                pattern[1][direction] = 0;
                pattern[2][direction] = 1;
            }
            else if constexpr (Rec == Reconstruction::PPM) {
                // PPM: i-1, i, i+1, i+2
                pattern[0][direction] = -2;
                pattern[1][direction] = -1;
                pattern[2][direction] = 0;
                pattern[3][direction] = 1;
            }

            return pattern;
        }
    };

    // factory functions for common stencil patterns
    namespace stencils {
        // helper to create symmetric stencil in given direction
        template <std::uint64_t Dims, Reconstruction Rec>
            requires valid_dimension<Dims>
        auto make_symmetric_stencil(std::uint64_t /*direction*/)
        {
            return stencil_t<Dims, Rec>{};
        }

        // convenient functions for common patterns
        template <std::uint64_t Dims>
            requires valid_dimension<Dims>
        auto one_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Dims, 1>(direction);
        }
        template <std::uint64_t Dims>
            requires valid_dimension<Dims>
        auto three_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Dims, 3>(direction);
        }

        template <std::uint64_t Dims>
            requires valid_dimension<Dims>
        auto five_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Dims, 5>(direction);
        }

        template <std::uint64_t Dims>
            requires valid_dimension<Dims>
        auto seven_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Dims, 7>(direction);
        }
    }   // namespace stencils
}   // namespace simbi::base
#endif
