#pragma once

#include "base/concepts.hpp"
#include "containers/vector.hpp"
#include "utility/enums.hpp"

#include <cassert>
#include <cstdint>

namespace simbi::base {
    // compile-time stencil size calculation
    template <reconstruction_t Rec>
    constexpr std::uint64_t stencil_size()
    {
        if constexpr (Rec == reconstruction_t::PCM) {
            return 1;
        }
        else if constexpr (Rec == reconstruction_t::PLM) {
            return 3;
        }
        else {
            // nvcc doesn't like false static_assert, so we switch to the
            // lambda trick
            []<bool flag = false>() { static_assert(flag, "Unsupported reconstruction order"); }();
        }
    }

    // compile-time stencil pattern generation
    template <std::uint64_t Rank, reconstruction_t Rec>
    struct stencil_t
    {
        static constexpr std::uint64_t size = stencil_size<Rec>();
        using coord_array_t                 = vector_t<iarray<Rank>, size>;

        // generate left reconstruction pattern
        static constexpr coord_array_t left_pattern(std::uint64_t direction)
        {
            coord_array_t pattern{};

            if constexpr (Rec == reconstruction_t::PCM) {
                pattern[0][direction] = -1; // use left cell
            }
            else if constexpr (Rec == reconstruction_t::PLM) {
                // PLM: i-1, i, i+1
                pattern[0][direction] = -2;
                pattern[1][direction] = -1;
                pattern[2][direction] = 0;
            }

            return pattern;
        }

        // generate right reconstruction pattern
        static constexpr coord_array_t right_pattern(std::uint64_t direction)
        {
            coord_array_t pattern{};

            if constexpr (Rec == reconstruction_t::PCM) {
                pattern[0][direction] = 0; // use right cell
            }
            else if constexpr (Rec == reconstruction_t::PLM) {
                // PLM: i, i+1, i+2
                pattern[0][direction] = -1;
                pattern[1][direction] = 0;
                pattern[2][direction] = 1;
            }

            return pattern;
        }
    };

    // factory functions for common stencil patterns
    namespace stencils {
        // helper to create symmetric stencil in given direction
        template <std::uint64_t Rank, reconstruction_t Rec>
            requires valid_dimension<Rank>
        auto make_symmetric_stencil(std::uint64_t /*direction*/)
        {
            return stencil_t<Rank, Rec>{};
        }

        // convenient functions for common patterns
        template <std::uint64_t Rank>
            requires valid_dimension<Rank>
        auto one_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Rank, 1>(direction);
        }
        template <std::uint64_t Rank>
            requires valid_dimension<Rank>
        auto three_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Rank, 3>(direction);
        }

        template <std::uint64_t Rank>
            requires valid_dimension<Rank>
        auto five_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Rank, 5>(direction);
        }

        template <std::uint64_t Rank>
            requires valid_dimension<Rank>
        auto seven_point(std::uint64_t direction)
        {
            return make_symmetric_stencil<Rank, 7>(direction);
        }
    } // namespace stencils
} // namespace simbi::base
