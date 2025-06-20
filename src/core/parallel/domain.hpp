
/**
 * domain.hpp
 * hardware-agnostic representation of computational domains
 */

#ifndef SIMBI_PARALLEL_DOMAIN_HPP
#define SIMBI_PARALLEL_DOMAIN_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/types/alias/alias.hpp"
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>

namespace simbi::parallel {

    /**
     * a multi-dimensional domain abstraction that's container-independent
     */
    template <size_type Dims>
    class domain_t
    {
      public:
        // constructors
        domain_t() = default;

        // create from shape
        DUAL constexpr explicit domain_t(const array_t<size_type, Dims>& shape)
            : shape_(shape)
        {
            size_ = std::accumulate(
                shape.begin(),
                shape.end(),
                size_type{1},
                std::multiplies<size_type>()
            );
        }

        // create from extents
        template <typename... Sizes>
        DUAL constexpr explicit domain_t(Sizes... sizes)
            requires(sizeof...(Sizes) == Dims)
            : shape_{static_cast<size_type>(sizes)...}
        {
            size_ = std::accumulate(
                shape_.begin(),
                shape_.end(),
                size_type{1},
                std::multiplies<size_type>()
            );
        }

        // accessors
        DUAL constexpr const auto& shape() const { return shape_; }
        DUAL constexpr size_type size() const { return size_; }
        DUAL constexpr size_type extent(size_type dim) const
        {
            return shape_[dim];
        }

        // create a subdomain
        DUAL constexpr domain_t subregion(
            const array_t<size_type, Dims>& start,
            const array_t<size_type, Dims>& end
        ) const
        {

            array_t<size_type, Dims> new_shape;
            for (size_type i = 0; i < Dims; ++i) {
                new_shape[i] = end[i] - start[i];
            }

            domain_t result(new_shape);
            result.offset_ = start;
            return result;
        }

        // check if a point is within the domain
        DUAL constexpr bool
        contains(const array_t<size_type, Dims>& point) const
        {
            for (size_type i = 0; i < Dims; ++i) {
                if (point[i] < offset_[i] ||
                    point[i] >= offset_[i] + shape_[i]) {
                    return false;
                }
            }
            return true;
        }

        // offset from global coordinates
        DUAL constexpr const auto& offset() const { return offset_; }

        // global to local coordinate conversion
        DUAL constexpr array_t<size_type, Dims>
        to_local(const array_t<size_type, Dims>& global_pos) const
        {
            array_t<size_type, Dims> local;
            for (size_type i = 0; i < Dims; ++i) {
                local[i] = global_pos[i] - offset_[i];
            }
            return local;
        }

        // local to global coordinate conversion
        DUAL constexpr array_t<size_type, Dims>
        to_global(const array_t<size_type, Dims>& local_pos) const
        {
            array_t<size_type, Dims> global;
            for (size_type i = 0; i < Dims; ++i) {
                global[i] = local_pos[i] + offset_[i];
            }
            return global;
        }

        // expand domain by halo in all directions
        DUAL constexpr domain_t<Dims> expand(size_type halo) const
        {
            array_t<size_type, Dims> new_shape;
            array_t<size_type, Dims> new_offset;

            for (size_type i = 0; i < Dims; ++i) {
                new_shape[i]  = shape_[i] + 2 * halo;
                new_offset[i] = offset_[i] - halo;
            }

            domain_t result(new_shape);
            result.offset_ = new_offset;
            return result;
        }

        // expand domain with different halo sizes per dimension
        DUAL constexpr domain_t<Dims>
        expand(const array_t<size_type, Dims>& halo) const
        {
            array_t<size_type, Dims> new_shape;
            array_t<size_type, Dims> new_offset;

            for (size_type i = 0; i < Dims; ++i) {
                new_shape[i]  = shape_[i] + 2 * halo[i];
                new_offset[i] = offset_[i] - halo[i];
            }

            domain_t result(new_shape);
            result.offset_ = new_offset;
            return result;
        }

        // iterator support for traversal
        class iterator
        {
          public:
            using value_type        = array_t<size_type, Dims>;
            using reference         = const value_type&;
            using pointer           = const value_type*;
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;

            DUAL constexpr iterator(const domain_t& dom, bool is_end = false)
                : domain_(dom), current_pos_{}, is_end_(is_end)
            {
                if (is_end) {
                    // Set to end position
                    current_pos_[0] = domain_.shape()[0];
                }
                else {
                    // Initialize all positions to zeros
                    for (size_type i = 0; i < Dims; ++i) {
                        current_pos_[i] = 0;
                    }
                }
            }

            DUAL constexpr reference operator*() const { return current_pos_; }

            DUAL constexpr pointer operator->() const { return &current_pos_; }

            // prefix increment
            DUAL constexpr iterator& operator++()
            {
                advance();
                return *this;
            }

            // postfix increment
            DUAL constexpr iterator operator++(int)
            {
                iterator tmp = *this;
                advance();
                return tmp;
            }

            DUAL constexpr bool operator==(const iterator& other) const
            {
                if (is_end_ && other.is_end_) {
                    return true;
                }
                if (is_end_ != other.is_end_) {
                    return false;
                }
                return current_pos_ == other.current_pos_;
            }

            DUAL constexpr bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }

            // get global position (with offset)
            DUAL constexpr array_t<size_type, Dims> global_position() const
            {
                return domain_.to_global(current_pos_);
            }

          private:
            // advance to next position
            DUAL constexpr void advance()
            {
                // Multi-dimensional increment with carry
                for (size_type dim = Dims - 1; dim < Dims; --dim) {
                    current_pos_[dim]++;
                    if (current_pos_[dim] < domain_.shape()[dim]) {
                        return;
                    }
                    current_pos_[dim] = 0;
                }

                // If we've wrapped around completely, mark as end
                is_end_         = true;
                current_pos_[0] = domain_.shape()[0];
            }

            const domain_t& domain_;
            array_t<size_type, Dims> current_pos_;
            bool is_end_;
        };

        // begin/end iterators for range-based for loop
        DUAL constexpr iterator begin() const { return iterator(*this); }
        DUAL constexpr iterator end() const { return iterator(*this, true); }

      private:
        array_t<size_type, Dims> shape_  = {};   // extent in each dimension
        array_t<size_type, Dims> offset_ = {};   // offset from global origin
        size_type size_                  = 0;    // total number of elements
    };

    // helper function to create domains
    template <typename... Sizes>
    DUAL constexpr auto make_domain(Sizes... sizes)
    {
        constexpr size_type Dims = sizeof...(Sizes);
        return domain<Dims>(sizes...);
    }

}   // namespace simbi::parallel

#endif   // SIMBI_PARALLEL_DOMAIN_HPP
