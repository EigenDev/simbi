// =============================================================================
// view.hpp
//
// multi-dimensional view implementation for xpu shared buffer integration.
// provides non-owning view of contiguous memory with stride-based indexing.
// preserves hesi view semantics while integrating with xpu memory spaces.
//
// design principles:
//   - non-owning view of memory (like std::span)
//   - multi-dimensional indexing with strides
//   - zero-overhead element access
//   - hesi-compatible api preservation
//
// usage:
//   view_t<float, 2> view{data, {100, 50}, {0, 0}, {50, 1}};
//   auto element = view(10, 20);
//   auto subview = view.subview({5, 10}, {20, 30});
// =============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace xpu {

    // =============================================================================
    // multi-dimensional view implementation
    // =============================================================================

    template <typename T, std::size_t Rank>
    class view_t
    {
      public:
        using value_type      = T;
        using pointer         = T*;
        using const_pointer   = const T*;
        using reference       = T&;
        using const_reference = const T&;
        using size_type       = std::size_t;
        using difference_type = std::ptrdiff_t;

        static constexpr std::size_t rank = Rank;

        using extent_array = std::array<size_type, Rank>;
        using offset_array = std::array<size_type, Rank>;
        using stride_array = std::array<size_type, Rank>;
        using index_array  = std::array<size_type, Rank>;

      private:
        pointer      data_;
        extent_array extents_;
        offset_array offsets_;
        stride_array strides_;

      public:
        // =============================================================================
        // construction
        // =============================================================================

        // default constructor (empty view)
        view_t() : data_(nullptr), extents_{}, offsets_{}, strides_{}
        {
            strides_.fill(1);
        }

        // construct from data pointer and layout
        view_t(
            pointer             data,
            const extent_array& extents,
            const offset_array& offsets,
            const stride_array& strides
        )
            : data_(data), extents_(extents), offsets_(offsets), strides_(strides)
        {
        }

        // construct from data pointer and extents (zero offsets, unit strides)
        view_t(pointer data, const extent_array& extents)
            : data_(data), extents_(extents), offsets_{}, strides_{}
        {
            // compute row-major strides
            if (Rank > 0) {
                strides_[Rank - 1] = 1;
                for (int ii = static_cast<int>(Rank) - 2; ii >= 0; --ii) {
                    strides_[ii] = strides_[ii + 1] * extents_[ii + 1];
                }
            }
        }

        // copy semantics (views are copyable)
        view_t(const view_t&)            = default;
        view_t& operator=(const view_t&) = default;

        // =============================================================================
        // element access
        // =============================================================================

        // multi-dimensional indexing
        template <typename... Indices>
        reference operator()(Indices... indices)
        {
            static_assert(sizeof...(Indices) == Rank, "index count must match view rank");
            return data_[linear_index(static_cast<size_type>(indices)...)];
        }

        template <typename... Indices>
        const_reference operator()(Indices... indices) const
        {
            static_assert(sizeof...(Indices) == Rank, "index count must match view rank");
            return data_[linear_index(static_cast<size_type>(indices)...)];
        }

        // array-based indexing
        reference operator[](const index_array& indices)
        {
            return data_[linear_index(indices)];
        }

        const_reference operator[](const index_array& indices) const
        {
            return data_[linear_index(indices)];
        }

        // linear indexing (for rank-1 views)
        template <std::size_t R = Rank>
        std::enable_if_t<R == 1, reference> operator[](size_type index)
        {
            return data_[offsets_[0] + index * strides_[0]];
        }

        template <std::size_t R = Rank>
        std::enable_if_t<R == 1, const_reference> operator[](size_type index) const
        {
            return data_[offsets_[0] + index * strides_[0]];
        }

        // bounds-checked access
        template <typename... Indices>
        reference at(Indices... indices)
        {
            check_bounds(static_cast<size_type>(indices)...);
            return (*this)(indices...);
        }

        template <typename... Indices>
        const_reference at(Indices... indices) const
        {
            check_bounds(static_cast<size_type>(indices)...);
            return (*this)(indices...);
        }

        // =============================================================================
        // properties
        // =============================================================================

        pointer data() const noexcept
        {
            return data_;
        }

        const extent_array& extents() const noexcept
        {
            return extents_;
        }

        const offset_array& offsets() const noexcept
        {
            return offsets_;
        }

        const stride_array& strides() const noexcept
        {
            return strides_;
        }

        size_type extent(size_type dim) const
        {
            if (dim >= Rank) {
                throw std::out_of_range("dimension index out of range");
            }
            return extents_[dim];
        }

        size_type stride(size_type dim) const
        {
            if (dim >= Rank) {
                throw std::out_of_range("dimension index out of range");
            }
            return strides_[dim];
        }

        size_type size() const noexcept
        {
            size_type total = 1;
            for (size_type extent : extents_) {
                total *= extent;
            }
            return total;
        }

        bool empty() const noexcept
        {
            for (size_type extent : extents_) {
                if (extent == 0) {
                    return true;
                }
            }
            return false;
        }

        explicit operator bool() const noexcept
        {
            return data_ != nullptr && !empty();
        }

        // =============================================================================
        // view operations
        // =============================================================================

        // create subview with different extents and offsets
        view_t subview(const offset_array& new_offsets, const extent_array& new_extents) const
        {
            // validate bounds
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                if (new_offsets[ii] + new_extents[ii] > extents_[ii]) {
                    throw std::out_of_range("subview extends beyond parent view bounds");
                }
            }

            // compute new absolute offsets
            offset_array absolute_offsets;
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                absolute_offsets[ii] = offsets_[ii] + new_offsets[ii];
            }

            return view_t{data_, new_extents, absolute_offsets, strides_};
        }

        // slice along a dimension (reduce rank by 1)
        template <std::size_t SliceDim>
        auto slice(size_type index) const
        {
            static_assert(SliceDim < Rank, "slice dimension out of range");
            static_assert(Rank > 1, "cannot slice rank-1 view");

            if (index >= extents_[SliceDim]) {
                throw std::out_of_range("slice index out of range");
            }

            constexpr std::size_t           new_rank = Rank - 1;
            std::array<size_type, new_rank> new_extents;
            std::array<size_type, new_rank> new_offsets;
            std::array<size_type, new_rank> new_strides;

            // copy extents, offsets, and strides, skipping the sliced dimension
            std::size_t new_idx = 0;
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                if (ii != SliceDim) {
                    new_extents[new_idx] = extents_[ii];
                    new_offsets[new_idx] = offsets_[ii];
                    new_strides[new_idx] = strides_[ii];
                    ++new_idx;
                }
            }

            // adjust data pointer for slice
            pointer slice_data = data_ + offsets_[SliceDim] + index * strides_[SliceDim];

            return view_t<T, new_rank>{slice_data, new_extents, new_offsets, new_strides};
        }

        // transpose view (swap dimensions)
        view_t transpose(size_type dim1, size_type dim2) const
        {
            if (dim1 >= Rank || dim2 >= Rank) {
                throw std::out_of_range("transpose dimension out of range");
            }

            if (dim1 == dim2) {
                return *this; // no-op
            }

            extent_array new_extents = extents_;
            offset_array new_offsets = offsets_;
            stride_array new_strides = strides_;

            std::swap(new_extents[dim1], new_extents[dim2]);
            std::swap(new_offsets[dim1], new_offsets[dim2]);
            std::swap(new_strides[dim1], new_strides[dim2]);

            return view_t{data_, new_extents, new_offsets, new_strides};
        }

        // flatten to 1D view
        view_t<T, 1> flatten() const
        {
            // ensure contiguous layout for flattening
            if (!is_contiguous()) {
                throw std::runtime_error("can only flatten contiguous views");
            }

            std::array<size_type, 1> flat_extents = {size()};
            std::array<size_type, 1> flat_offsets = {0};
            std::array<size_type, 1> flat_strides = {1};

            return view_t<T, 1>{data_ + linear_offset(), flat_extents, flat_offsets, flat_strides};
        }

        // =============================================================================
        // iterator support (rank-1 only)
        // =============================================================================

        template <std::size_t R = Rank>
        std::enable_if_t<R == 1, pointer> begin() const
        {
            return data_ + offsets_[0];
        }

        template <std::size_t R = Rank>
        std::enable_if_t<R == 1, pointer> end() const
        {
            return data_ + offsets_[0] + extents_[0] * strides_[0];
        }

        // =============================================================================
        // utility
        // =============================================================================

        bool is_contiguous() const noexcept
        {
            // check if strides match row-major layout
            if (Rank == 0) {
                return true;
            }

            size_type expected_stride = 1;
            for (int ii = static_cast<int>(Rank) - 1; ii >= 0; --ii) {
                if (strides_[ii] != expected_stride) {
                    return false;
                }
                expected_stride *= extents_[ii];
            }
            return true;
        }

        bool is_same_layout(const view_t& other) const noexcept
        {
            return extents_ == other.extents_ && offsets_ == other.offsets_ &&
                   strides_ == other.strides_;
        }

      private:
        // compute linear index from multi-dimensional indices
        template <typename... Indices>
        size_type linear_index(Indices... indices) const
        {
            static_assert(sizeof...(Indices) == Rank, "index count must match view rank");

            index_array idx_array = {static_cast<size_type>(indices)...};
            return linear_index(idx_array);
        }

        size_type linear_index(const index_array& indices) const
        {
            size_type linear_idx = 0;
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                linear_idx += (offsets_[ii] + indices[ii]) * strides_[ii];
            }
            return linear_idx;
        }

        // compute linear offset for base of view
        size_type linear_offset() const
        {
            size_type offset = 0;
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                offset += offsets_[ii] * strides_[ii];
            }
            return offset;
        }

        // bounds checking
        template <typename... Indices>
        void check_bounds(Indices... indices) const
        {
            static_assert(sizeof...(Indices) == Rank, "index count must match view rank");

            index_array idx_array = {static_cast<size_type>(indices)...};
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                if (idx_array[ii] >= extents_[ii]) {
                    throw std::out_of_range("view index out of bounds");
                }
            }
        }
    };

    // =============================================================================
    // convenience aliases
    // =============================================================================

    template <typename T>
    using view_1d_t = view_t<T, 1>;

    template <typename T>
    using view_2d_t = view_t<T, 2>;

    template <typename T>
    using view_3d_t = view_t<T, 3>;

    // =============================================================================
    // factory functions
    // =============================================================================

    template <typename T>
    view_1d_t<T> make_view(T* data, std::size_t size)
    {
        return view_1d_t<T>{data, {size}};
    }

    template <typename T>
    view_2d_t<T> make_view(T* data, std::size_t rows, std::size_t cols)
    {
        return view_2d_t<T>{data, {rows, cols}};
    }

    template <typename T>
    view_3d_t<T> make_view(T* data, std::size_t depth, std::size_t rows, std::size_t cols)
    {
        return view_3d_t<T>{data, {depth, rows, cols}};
    }

    // =============================================================================
    // comparison operators
    // =============================================================================

    template <typename T, std::size_t Rank>
    bool operator==(const view_t<T, Rank>& lhs, const view_t<T, Rank>& rhs)
    {
        if (lhs.extents() != rhs.extents()) {
            return false;
        }

        // compare element-wise (slow for non-contiguous views)
        if (lhs.size() != rhs.size()) {
            return false;
        }

        // for rank-1, use iterators if possible
        if constexpr (Rank == 1) {
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }
        else {
            // multi-dimensional comparison would need recursive indexing
            // simplified implementation: assume equal if same data and layout
            return lhs.data() == rhs.data() && lhs.is_same_layout(rhs);
        }
    }

    template <typename T, std::size_t Rank>
    bool operator!=(const view_t<T, Rank>& lhs, const view_t<T, Rank>& rhs)
    {
        return !(lhs == rhs);
    }

} // namespace xpu
