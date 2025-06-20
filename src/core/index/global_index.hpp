// =============================================================================
// index/global_index.hpp
// =============================================================================

#ifndef GLOBAL_INDEX_HPP
#define GLOBAL_INDEX_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include <cstddef>
#include <cstdint>

namespace simbi::index {

    // stagger location enumeration
    enum class stagger_t {
        cell,        // cell-centered (i,j,k)
        face_x1,     // x1-face centered (i+1/2,j,k)
        face_x2,     // x2-face centered (i,j+1/2,k)
        face_x3,     // x3-face centered (i,j,k+1/2)
        edge_x1x2,   // edge centered (i+1/2,j+1/2,k)
        edge_x1x3,   // edge centered (i+1/2,j,k+1/2)
        edge_x2x3,   // edge centered (i,j+1/2,k+1/2)
        node         // node centered (i+1/2,j+1/2,k+1/2)
    };

    // staggered index with compile-time stagger location
    template <stagger_t Location>
    struct staggered_index_t {
        int64_t x1, x2, x3;
        int32_t level                               = 0;
        static constexpr stagger_t stagger_location = Location;

        // ctors
        constexpr staggered_index_t() = default;
        constexpr staggered_index_t(
            int64_t x1,
            int64_t x2,
            int64_t x3,
            int32_t level = 0
        )
            : x1(x1), x2(x2), x3(x3), level(level)
        {
        }

        template <typename IndexType>
        constexpr staggered_index_t(
            const array_t<IndexType, 1>& coords,
            int32_t level = 0
        )
            : x1(static_cast<int64_t>(coords[0])), x2(0), x3(0), level(level)
        {
        }

        template <typename IndexType>
        constexpr staggered_index_t(
            const array_t<IndexType, 2>& coords,
            int32_t level = 0
        )
            : x1(static_cast<int64_t>(coords[0])),
              x2(static_cast<int64_t>(coords[1])),
              x3(0),
              level(level)
        {
        }

        template <typename IndexType>
        constexpr staggered_index_t(
            const array_t<IndexType, 3>& coords,
            int32_t level = 0
        )
            : x1(static_cast<int64_t>(coords[0])),
              x2(static_cast<int64_t>(coords[1])),
              x3(static_cast<int64_t>(coords[2])),
              level(level)
        {
        }

        ~staggered_index_t() = default;

        // arithmetic operations preserve stagger type
        constexpr staggered_index_t
        operator+(const array_t<int64_t, 3>& offset) const
        {
            return {x1 + offset[0], x2 + offset[1], x3 + offset[2], level};
        }

        constexpr staggered_index_t
        shift(int64_t dx1, int64_t dx2, int64_t dx3) const
        {
            return {x1 + dx1, x2 + dx2, x3 + dx3, level};
        }

        // coordinate access by direction
        constexpr int64_t coord(size_t dir) const
        {
            switch (dir) {
                case 0: return x1;
                case 1: return x2;
                case 2: return x3;
                default: return x1;
            }
        }

        // comparison for ordering/hashing
        constexpr bool operator==(const staggered_index_t&) const  = default;
        constexpr auto operator<=>(const staggered_index_t&) const = default;
    };

    // type aliases for common stagger locations
    using cell_index_t      = staggered_index_t<stagger_t::cell>;
    using face_x1_index_t   = staggered_index_t<stagger_t::face_x1>;
    using face_x2_index_t   = staggered_index_t<stagger_t::face_x2>;
    using face_x3_index_t   = staggered_index_t<stagger_t::face_x3>;
    using edge_x1x2_index_t = staggered_index_t<stagger_t::edge_x1x2>;
    using edge_x1x3_index_t = staggered_index_t<stagger_t::edge_x1x3>;
    using edge_x2x3_index_t = staggered_index_t<stagger_t::edge_x2x3>;
    using node_index_t      = staggered_index_t<stagger_t::node>;

    // backward compatibility
    // using global_index_t = cell_index_t;

    // direction enum for stencil operations
    enum class direction_t : size_t {
        x1 = 0,
        x2 = 1,
        x3 = 2
    };

    // compile-time stagger array size calculation
    template <stagger_t Stagger>
    constexpr auto stagger_array_size(int64_t n1, int64_t n2, int64_t n3)
    {
        if constexpr (Stagger == stagger_t::cell) {
            return n1 * n2 * n3;
        }
        else if constexpr (Stagger == stagger_t::face_x1) {
            return (n1 + 1) * n2 * n3;
        }
        else if constexpr (Stagger == stagger_t::face_x2) {
            return n1 * (n2 + 1) * n3;
        }
        else if constexpr (Stagger == stagger_t::face_x3) {
            return n1 * n2 * (n3 + 1);
        }
        else if constexpr (Stagger == stagger_t::edge_x1x2) {
            return (n1 + 1) * (n2 + 1) * n3;
        }
        else if constexpr (Stagger == stagger_t::edge_x1x3) {
            return (n1 + 1) * n2 * (n3 + 1);
        }
        else if constexpr (Stagger == stagger_t::edge_x2x3) {
            return n1 * (n2 + 1) * (n3 + 1);
        }
        else if constexpr (Stagger == stagger_t::node) {
            return (n1 + 1) * (n2 + 1) * (n3 + 1);
        }
    }

    // compile-time stagger array indexing
    template <stagger_t Stagger>
    constexpr int64_t stagger_array_index(
        staggered_index_t<Stagger> idx,
        int64_t n1,
        int64_t n2,
        int64_t n3
    )
    {
        if constexpr (Stagger == stagger_t::cell) {
            if constexpr (global::col_major) {
                return idx.x1 * n2 * n3 + idx.x2 * n3 + idx.x3;
            }
            return idx.x1 + idx.x2 * n1 + idx.x3 * n1 * n2;
        }
        else if constexpr (Stagger == stagger_t::face_x1) {
            if constexpr (global::col_major) {
                return idx.x1 * (n2 + 1) * n3 + idx.x2 * n3 + idx.x3;
            }
            return idx.x1 + (n1 + 1) * idx.x2 + (n1 + 1) * n2 * idx.x3;
        }
        else if constexpr (Stagger == stagger_t::face_x2) {
            if constexpr (global::col_major) {
                return idx.x1 * n2 * (n3 + 1) + idx.x2 * (n3 + 1) + idx.x3;
            }
            return idx.x1 + n1 * idx.x2 + n1 * (n2 + 1) * idx.x3;
        }
        else if constexpr (Stagger == stagger_t::face_x3) {
            return idx.x1 + idx.x2 * n1 + idx.x3 * n1 * n2;
        }
        else if constexpr (Stagger == stagger_t::edge_x1x2) {
            return idx.x1 * (n2 + 1) * n3 + idx.x2 * n3 + idx.x3;
        }
        else if constexpr (Stagger == stagger_t::edge_x1x3) {
            return idx.x1 * n2 * (n3 + 1) + idx.x2 * (n3 + 1) + idx.x3;
        }
        else if constexpr (Stagger == stagger_t::edge_x2x3) {
            return idx.x1 * (n2 + 1) * (n3 + 1) + idx.x2 * (n3 + 1) + idx.x3;
        }
        else if constexpr (Stagger == stagger_t::node) {
            return idx.x1 * (n2 + 1) * (n3 + 1) + idx.x2 * (n3 + 1) + idx.x3;
        }
    }

}   // namespace simbi::index

#endif
