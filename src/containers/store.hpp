// =============================================================================
// store.hpp
//
// dynamic array for heterogeneous memory backed by xpu shared_buffer_t
// thin wrapper around shared_buffer_t<T, unified_memory> that adds:
//   - add() for dynamic growth during parsing
//   - constructor from std::vector
//   - compatibility with existing simbi store_t api
//
// usage:
//   store_t<ExprNode> nodes;
//   nodes.reserve(50);
//   nodes.add(node);  // grows dynamically
//   // then use as const ref with unified memory access
// =============================================================================

#ifndef CONTAINERS_STORE_HPP
#define CONTAINERS_STORE_HPP

#include "compat.hpp"
#include "xpu/xpu.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace simbi {

    // -------------------------------------------------------------------------
    // dynamic array backed by xpu unified memory
    // -------------------------------------------------------------------------
    template <typename T>
    class store_t
    {
      private:
        xpu::shared_buffer_t<T, xpu::unified_memory> buffer_;
        std::uint64_t                                size_ = 0;

      public:
        // ---------------------------------------------------------------------
        // construction
        // ---------------------------------------------------------------------

        // default: empty
        store_t() = default;

        // sized with optional fill value
        explicit store_t(std::uint64_t size, const T& val = T()) : buffer_(size), size_(size)
        {
            if (size > 0) {
                for (std::uint64_t ii = 0; ii < size; ++ii) {
                    buffer_.data()[ii] = val;
                }
            }
        }

        // from host vector
        explicit store_t(const std::vector<T>& values)
            : buffer_(values.size()), size_(values.size())
        {
            if (!values.empty()) {
                std::copy(values.begin(), values.end(), buffer_.data());
            }
        }

        // move semantics (shared_buffer_t handles this)
        store_t(store_t&&) noexcept            = default;
        store_t& operator=(store_t&&) noexcept = default;

        // shallow copy semantics (reference counting, preserves hesi behavior)
        store_t(const store_t&)            = default;
        store_t& operator=(const store_t&) = default;

        // ---------------------------------------------------------------------
        // accessors
        // ---------------------------------------------------------------------

        T& operator[](std::uint64_t idx)
        {
            return buffer_.data()[idx];
        }
        const T& operator[](std::uint64_t idx) const
        {
            return buffer_.data()[idx];
        }

        T* data()
        {
            return buffer_.data();
        }
        DUAL const T* data() const
        {
            return buffer_.data();
        }

        DUAL std::uint64_t size() const
        {
            return size_;
        }
        DUAL std::uint64_t capacity() const
        {
            return buffer_.capacity();
        }
        DUAL bool empty() const
        {
            return size_ == 0;
        }

        // ---------------------------------------------------------------------
        // mutation
        // ---------------------------------------------------------------------

        // add element with dynamic growth
        void add(const T& value)
        {
            if (size_ >= buffer_.capacity()) {
                // geometric growth: 2x or minimum 16
                std::uint64_t new_capacity = buffer_.capacity() == 0 ? 16 : buffer_.capacity() * 2;
                buffer_.reserve(new_capacity);
            }
            buffer_.resize(size_ + 1);
            buffer_.data()[size_] = value;
            ++size_;
        }

        // reserve capacity
        void reserve(std::uint64_t new_capacity)
        {
            if (new_capacity > buffer_.capacity()) {
                buffer_.reserve(new_capacity);
            }
        }

        // clear (keep capacity)
        void clear()
        {
            size_ = 0;
        }

        // resize with optional fill value
        void resize(std::uint64_t new_size, const T& value = T())
        {
            buffer_.resize(new_size);

            // fill new elements if growing
            if (new_size > size_) {
                for (std::uint64_t ii = size_; ii < new_size; ++ii) {
                    buffer_.data()[ii] = value;
                }
            }

            size_ = new_size;
        }

        // ---------------------------------------------------------------------
        // unified memory optimization (no-ops for now)
        // ---------------------------------------------------------------------

        // unified memory driver handles migration automatically
        // explicit prefetch can be added later if profiling shows benefit
        void sync_to_all_devices() {}

        // placeholder for explicit prefetch
        template <typename ExecutorType>
        void prefetch(ExecutorType& /*exec*/)
        {
        }
    };

} // namespace simbi

#endif // CONTAINERS_STORE_HPP
