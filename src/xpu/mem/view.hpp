#ifndef XPU_VIEW_HPP
#define XPU_VIEW_HPP

#include "containers/vector.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi::xpu {

    // non-owning, lightweight window into a memory block
    // uses explicit strides for correct multi-dimensional access
    template <typename T, std::uint64_t Rank = 1>
    struct view_t
    {
        static_assert(std::is_trivially_copyable_v<T>, "view types must be trivial");
        static_assert(Rank >= 1 && Rank <= 3, "view rank must be 1, 2, or 3");

        T*           data_;
        iarray<Rank> shape_; // logical dimensions (interior size)
        iarray<Rank> start_; // physical start (for safety)
        // stride ordering:
        // strides_[0] = slowest dimension (z/outer)
        // strides_[1] = middle dimension (y)
        // strides_[2] = fastest dimension (x/inner)
        // for row-major 3D: strides = [ny*nx, nx, 1]
        iarray<Rank> strides_;

        // the functor interface required by the computation graph
        // allows view to be treated as f(x)
        DUAL T& operator()(iarray<Rank> coord) const
        {
            return data_[vecops::dot(coord - start_, strides_)];
        }
        DUAL T& operator[](iarray<Rank> coord)
        {
            return data_[vecops::dot(coord - start_, strides_)];
        }

        // helper to get raw pointer
        DUAL T* data() const
        {
            return data_;
        }

        // helper for bounds
        DUAL std::uint64_t size() const
        {
            std::uint64_t result = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                result *= shape_[ii];
            }
            return result;
        }
        DUAL iarray<Rank> shape() const
        {
            return shape_;
        }
    };

} // namespace simbi::xpu

#endif // XPU_VIEW_HPP
