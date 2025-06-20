/**
 * view.hpp
 * container-agnostic view concept for data access
 */

#ifndef SIMBI_PARALLEL_VIEW_HPP
#define SIMBI_PARALLEL_VIEW_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/containers/collapsable.hpp"
#include "core/types/alias/alias.hpp"
#include "domain.hpp"
#include "pattern.hpp"
#include <cassert>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace simbi::parallel {

    // concept for any type that can be accessed with operator[]
    template <typename T>
    concept Indexable = requires(T t, size_type idx) {
        { t[idx] } -> std::convertible_to<typename T::value_type>;
    };

    // concept for N-dimensional containers
    template <typename T>
    concept NDIndexable =
        requires(T t, const array_t<size_type, T::dimensions>& idx) {
            { t.at(idx) } -> std::convertible_to<typename T::value_type>;
            { T::dimensions } -> std::convertible_to<size_type>;
        };

    /**
     * generic view over data - doesn't own the data, just provides access
     * this is what stencil operators work with
     */
    template <typename T, size_type Dims>
    class data_view_t
    {
      public:
        using value_type                      = std::remove_cvref_t<T>;
        static constexpr size_type dimensions = Dims;

        constexpr data_view_t() = default;

        // create from a pointer and shape
        DUAL constexpr data_view_t(T* data, const domain_t<Dims>& dom)
            : data_(data), domain_(dom), strides_(compute_strides(dom.shape()))
        {
        }

        // access methods
        DUAL constexpr T& operator[](size_type idx)
        {
            // assert(idx < domain_.size() && "Index out of bounds");
            return data_[idx];
        }

        DUAL constexpr const T& operator[](size_type idx) const
        {
            // assert(idx < domain_.size() && "Index out of bounds");
            return data_[idx];
        }

        // multi-dimensional access
        template <std::integral IndexType>
        DUAL constexpr T& at(const array_t<IndexType, Dims>& idx)
        {
            return data_[compute_index(idx)];
        }

        template <std::integral IndexType>
        DUAL constexpr const T& at(const array_t<IndexType, Dims>& idx) const
        {
            return data_[compute_index(idx)];
        }

        template <typename... Args>
        DUAL constexpr T& at(Args... args)
        {
            // static_assert(sizeof...(args) == Dims, "Invalid number of
            // indices");
            collapsable_t<int64_t, Dims> idx{static_cast<int64_t>(args)...};
            return at(idx.vals);
        }

        template <typename... Args>
        DUAL constexpr const T& at(Args... args) const
        {
            // static_assert(sizeof...(args) == Dims, "Invalid number of
            // indices");
            collapsable_t<int64_t, Dims> idx{static_cast<int64_t>(args)...};
            return at(idx.vals);
        }

        // access with relative offset
        DUAL constexpr const T& at_offset(
            const array_t<int64_t, Dims>& base,
            const offset_t<Dims>& off
        ) const
        {
            array_t<size_type, Dims> idx;
            for (size_type ii = 0; ii < Dims; ++ii) {
                // Check bounds
                int target = static_cast<int>(base[ii]) + off.indices[ii];
                assert(target >= 0 && "Negative index");
                idx[ii] = static_cast<size_type>(target);
            }
            return at(idx);
        }

        // support for std::begin/end
        DUAL constexpr T* begin() { return data_; }
        DUAL constexpr T* end() { return data_ + domain_.size(); }
        DUAL constexpr const T* begin() const { return data_; }
        DUAL constexpr const T* end() const { return data_ + domain_.size(); }

        // get domain information
        DUAL constexpr const domain_t<Dims>& domain() const { return domain_; }
        DUAL constexpr const array_t<size_type, Dims>& shape() const
        {
            return domain_.shape();
        }
        DUAL constexpr size_type size() const { return domain_.size(); }

        // return raw pointer to data
        DUAL constexpr T* data() { return data_; }
        DUAL constexpr const T* data() const { return data_; }

        // create a subview for a subdomain
        DUAL constexpr data_view_t<T, Dims> subview(
            const array_t<size_type, Dims>& start,
            const array_t<size_type, Dims>& end
        ) const
        {
            auto subdom      = domain_.subregion(start, end);
            size_type offset = compute_index(start);
            return data_view_t<T, Dims>(data_ + offset, subdom);
        }

        // create a stencil view centered at a specific position
        template <typename F>
        DUAL constexpr auto stencil_apply(
            const array_t<size_type, Dims>& center,
            const pattern_t<Dims>& pat,
            F&& func
        ) const
        {
            // Apply function to each point in the pattern
            return std::invoke(
                std::forward<F>(func),
                [this, &center, &pat](const offset_t<Dims>& off) -> T& {
                    array_t<size_type, Dims> pos;
                    for (size_type i = 0; i < Dims; ++i) {
                        int target =
                            static_cast<int>(center[i]) + off.indices[i];
                        assert(target >= 0 && "Negative index");
                        pos[i] = static_cast<size_type>(target);
                    }
                    return this->at(pos);
                }
            );
        }

        template <std::integral IndexType>
        DUAL auto linear_index(const array_t<IndexType, Dims>& idx) const
        {
            return compute_index(idx);
        }

      private:
        // compute linear index from multi-dimensional indices
        template <std::integral IndexType>
        DUAL constexpr size_type
        compute_index(const array_t<IndexType, Dims>& idx) const
        {
            size_type index = 0;
            for (size_type ii = 0; ii < Dims; ++ii) {
                size_type adjusted_idx = idx[ii] + domain_.offset()[ii];
                index += adjusted_idx * strides_[ii];
            }
            return index;
        }

        // compute strides from shape
        DUAL constexpr static array_t<size_type, Dims>
        compute_strides(const array_t<size_type, Dims>& shape)
        {
            array_t<size_type, Dims> strides;
            if constexpr (global::col_major) {
                // Column-major (Fortran order)
                strides[0] = 1;
                for (size_type i = 1; i < Dims; ++i) {
                    strides[i] = strides[i - 1] * shape[i - 1];
                }
            }
            else {
                // Row-major (C order)
                strides[Dims - 1] = 1;
                for (size_type i = Dims - 1; i > 0; --i) {
                    strides[i - 1] = strides[i] * shape[i];
                }
            }
            return strides;
        }

        T* data_;                 // pointer to data (non-owning)
        domain_t<Dims> domain_;   // domain information
        array_t<size_type, Dims>
            strides_;   // strides for multi-dimensional indexing
    };

    /**
     * Special view type for stencil operations
     * Provides relative access around a center point
     */
    template <typename T, size_type Dims>
    class stencil_view_t
    {
      public:
        using value_type                      = std::remove_cvref_t<T>;
        static constexpr size_type dimensions = Dims;

        // Create from a data_view_t and center position
        DUAL constexpr stencil_view_t(
            const data_view_t<T, Dims>& view,
            const array_t<size_type, Dims>& center
        )
            : view_(view), center_(center)
        {
        }

        // Access relative to center
        DUAL constexpr T& operator()(const offset_t<Dims>& off) const
        {
            array_t<size_type, Dims> pos;
            for (size_type i = 0; i < Dims; ++i) {
                int target = static_cast<int>(center_[i]) + off.indices[i];
                assert(target >= 0 && "Negative index");
                pos[i] = static_cast<size_type>(target);
            }
            return view_.at(pos);
        }

        // Direct access to center value
        DUAL constexpr T& center() const { return view_.at(center_); }

        // Get center position
        DUAL constexpr const auto& position() const { return center_; }

        // Common access patterns
        DUAL constexpr T& east() const
        {
            return (*this)(direction::east<Dims>());
        }
        DUAL constexpr T& west() const
        {
            return (*this)(direction::west<Dims>());
        }
        DUAL constexpr T& north() const
        {
            return (*this)(direction::north<Dims>());
        }
        DUAL constexpr T& south() const
        {
            return (*this)(direction::south<Dims>());
        }

      private:
        data_view_t<T, Dims> view_;
        array_t<size_type, Dims> center_;
    };

    // concept for types that can be adapted to a view
    template <typename T>
    concept Viewable = Indexable<T> || NDIndexable<T>;

    // adapter function to create view from viewable type
    template <typename T, size_type Dims>
    DUAL auto make_view(T& container, const domain_t<Dims>& dom)
    {
        if constexpr (std::is_pointer_v<T>) {
            return data_view_t<std::remove_pointer_t<T>, Dims>(container, dom);
        }
        else {
            return data_view_t<typename T::value_type, Dims>(
                container.data(),
                dom
            );
        }
    }

}   // namespace simbi::parallel

#endif   // SIMBI_PARALLEL_VIEW_HPP
