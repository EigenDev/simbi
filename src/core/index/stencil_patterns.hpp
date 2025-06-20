// =============================================================================
// index/stencil_patterns.hpp
// =============================================================================
#ifndef STENCIL_PATTERNS_HPP
#define STENCIL_PATTERNS_HPP

#include "core/containers/array.hpp"    // for array_t
#include "core/types/alias/alias.hpp"   // for size_type, luint
#include "global_index.hpp"             // for cell_index_t
#include <concepts>                     // for std::convertible_to, requires
#include <cstddef>                      // for size_t
#include <cstdint>                      // for int64_t, int32_t
#include <type_traits>

namespace simbi::index {
    template <typename T>
    concept stencil_view_c = requires(T t, size_t i) {
        t.begin();
        t.end();
        { t[i] } -> std::convertible_to<cell_index_t>;
        { T::size() } -> std::convertible_to<size_t>;
    };

    // compile-time stencil pattern definition
    template <size_t N>
    struct stencil_pattern_t {
        array_t<array_t<int64_t, 3>, N> offsets;
        static constexpr size_t size = N;

        constexpr stencil_pattern_t(const array_t<array_t<int64_t, 3>, N>& offs)
            : offsets(offs)
        {
        }
    };

    // zero-allocation stencil iterator
    template <typename Pattern>
    class stencil_iterator_t
    {
        cell_index_t center_;
        size_t index_;

      public:
        constexpr stencil_iterator_t(cell_index_t center, size_t index)
            : center_(center), index_(index)
        {
        }

        constexpr cell_index_t operator*() const
        {
            return center_ + Pattern::pattern.offsets[index_];
        }

        constexpr stencil_iterator_t& operator++()
        {
            ++index_;
            return *this;
        }

        constexpr bool operator!=(const stencil_iterator_t& other) const
        {
            return index_ != other.index_;
        }

        constexpr bool operator==(const stencil_iterator_t& other) const
        {
            return index_ == other.index_;
        }
    };

    // zero-allocation stencil view
    template <typename Pattern>
    class stencil_view_t
    {
        cell_index_t center_;

      public:
        constexpr explicit stencil_view_t(cell_index_t center) : center_(center)
        {
        }

        constexpr auto begin() const
        {
            return stencil_iterator_t<Pattern>{center_, 0};
        }

        constexpr auto end() const
        {
            return stencil_iterator_t<Pattern>{center_, Pattern::pattern.size};
        }

        constexpr cell_index_t operator[](size_t i) const
        {
            return center_ + Pattern::pattern.offsets[i];
        }

        static constexpr size_t size() { return Pattern::pattern.size; }
    };

    // compile-time stencil pattern definitions
    struct face_neighbors_t {
        static constexpr auto pattern = stencil_pattern_t<6>{{
          {{-1, 0, 0},
           {1, 0, 0},   // x1-direction faces
           {0, -1, 0},
           {0, 1, 0},   // x2-direction faces
           {0, 0, -1},
           {0, 0, 1}}   // x3-direction faces
        }};
    };

    // directional five-point stencils for second-order schemes
    struct five_point_x1_t {
        static constexpr auto pattern = stencil_pattern_t<5>{
          {{{-2, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {2, 0, 0}}}
        };
    };

    struct five_point_x2_t {
        static constexpr auto pattern = stencil_pattern_t<5>{
          {{{0, -2, 0}, {0, -1, 0}, {0, 0, 0}, {0, 1, 0}, {0, 2, 0}}}
        };
    };

    struct five_point_x3_t {
        static constexpr auto pattern = stencil_pattern_t<5>{
          {{{0, 0, -2}, {0, 0, -1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 2}}}
        };
    };

    // three-point stencils for gradients
    struct three_point_x1_t {
        static constexpr auto pattern =
            stencil_pattern_t<3>{{{{-1, 0, 0}, {0, 0, 0}, {1, 0, 0}}}};
    };

    struct three_point_x2_t {
        static constexpr auto pattern =
            stencil_pattern_t<3>{{{{0, -1, 0}, {0, 0, 0}, {0, 1, 0}}}};
    };

    struct three_point_x3_t {
        static constexpr auto pattern =
            stencil_pattern_t<3>{{{{0, 0, -1}, {0, 0, 0}, {0, 0, 1}}}};
    };

    // convenient factory functions
    constexpr auto face_neighbors(cell_index_t center)
    {
        return stencil_view_t<face_neighbors_t>{center};
    }

    template <direction_t Dir>
    constexpr auto five_point(cell_index_t center)
    {
        if constexpr (Dir == direction_t::x1) {
            return stencil_view_t<five_point_x1_t>{center};
        }
        else if constexpr (Dir == direction_t::x2) {
            return stencil_view_t<five_point_x2_t>{center};
        }
        else if constexpr (Dir == direction_t::x3) {
            return stencil_view_t<five_point_x3_t>{center};
        }
    }

    template <direction_t Dir>
    constexpr auto three_point(cell_index_t center)
    {
        if constexpr (Dir == direction_t::x1) {
            return stencil_view_t<three_point_x1_t>{center};
        }
        else if constexpr (Dir == direction_t::x2) {
            return stencil_view_t<three_point_x2_t>{center};
        }
        else if constexpr (Dir == direction_t::x3) {
            return stencil_view_t<three_point_x3_t>{center};
        }
        else {
            static_assert(
                false,
                "Invalid direction for three-point stencil: must be x1, x2, or "
                "x3"
            );
        }
    }

    // zero-allocation interior sweep via callback
    template <typename Func>
    constexpr void
    interior_sweep(int64_t n1, int64_t n2, int64_t n3, Func&& callback)
    {
        for (int64_t i = 1; i < n1 - 1; ++i) {
            for (int64_t j = 1; j < n2 - 1; ++j) {
                for (int64_t k = 1; k < n3 - 1; ++k) {
                    callback(cell_index_t{i, j, k, 0});
                }
            }
        }
    }

    // zero-allocation boundary sweep via callback
    template <typename Func>
    constexpr void
    boundary_sweep(int64_t n1, int64_t n2, int64_t n3, Func&& callback)
    {
        for (int64_t i = 0; i < n1; ++i) {
            for (int64_t j = 0; j < n2; ++j) {
                for (int64_t k = 0; k < n3; ++k) {
                    if (i == 0 || i == n1 - 1 || j == 0 || j == n2 - 1 ||
                        k == 0 || k == n3 - 1) {
                        callback(cell_index_t{i, j, k, 0});
                    }
                }
            }
        }
    }

    template <direction_t Dir, typename ViewType>
    auto get_interface_states(const auto& data, const cell_index_t& idx)
    {
        auto stencil = three_point<Dir>(idx);

        // for three-point stencils:
        // stencil[0] is left neighbor
        // stencil[1] is center cell
        // stencil[2] is right neighbor

        // note to self: the difference from getting points directly is that
        // this generically works regardless of the direction specified in Dir
        return std::make_pair(
            data.at(stencil[0].x1, stencil[0].x2, stencil[0].x3),
            data.at(stencil[2].x1, stencil[2].x2, stencil[2].x3)
        );
    }

    template <size_type Dims, typename Func>
    auto dispatch_direction(size_type dir, Func&& func)
    {
        if constexpr (Dims >= 1) {
            if (dir == 0) {
                return func(
                    std::integral_constant<direction_t, direction_t::x1>{}
                );
            }
        }
        if constexpr (Dims >= 2) {
            if (dir == 1) {
                return func(
                    std::integral_constant<direction_t, direction_t::x2>{}
                );
            }
        }
        if constexpr (Dims >= 3) {
            if (dir == 2) {
                return func(
                    std::integral_constant<direction_t, direction_t::x3>{}
                );
            }
        }
        // Default case (should not happen with proper bounds checking)
        return func(std::integral_constant<direction_t, direction_t::x1>{});
    }

}   // namespace simbi::index

#endif
