// =============================================================================
// store.hpp
//
// dynamic array for heterogeneous memory backed by xpu mem system
// thin wrapper around shared_handle_t<block_t> that adds:
//   - add() for dynamic growth during parsing
//   - constructor from std::vector
//   - compatibility with existing simbi store_t api
//
// usage:
//   store_t<ExprNode> nodes;
//   nodes.reserve(50);
//   nodes.add(node);  // grows dynamically
//   // then use as const ref with configured memory access
// =============================================================================

#ifndef CONTAINERS_STORE_HPP
#define CONTAINERS_STORE_HPP

#include "decorators.hpp"
#include "xpu/mem/memory_config.hpp"
#include "xpu/xpu.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace simbi {

    // -------------------------------------------------------------------------
    // dynamic array backed by xpu new memory system
    // -------------------------------------------------------------------------
    template <typename T>
    class store_t
    {
      private:
        xpu::shared_handle_t<xpu::sim_block_t> buffer_;
        T*                                     data_ptr_ = nullptr;
        std::uint64_t                          size_     = 0;
        std::uint64_t                          capacity_ = 0;

      public:
        // ---------------------------------------------------------------------
        // construction
        // ---------------------------------------------------------------------

        // default: empty
        store_t() = default;

        // sized with optional fill value
        explicit store_t(std::uint64_t size, const T& val = T()) : size_(size), capacity_(size)
        {
            if (size > 0) {
                auto block = xpu::make_memory_block<T>(size);
                buffer_    = xpu::make_shared_handle<xpu::sim_block_t>(std::move(block));
                data_ptr_  = buffer_->template as<T>();

                for (std::uint64_t ii = 0; ii < size; ++ii) {
                    data_ptr_[ii] = val;
                }
                xpu::mem::mark_host_dirty_if_needed(buffer_);
            }
        }

        // from host vector
        explicit store_t(const std::vector<T>& values)
            : size_(values.size()), capacity_(values.size())
        {
            if (!values.empty()) {
                auto block = xpu::make_memory_block<T>(values.size());
                buffer_    = simbi::xpu::make_shared_handle<xpu::sim_block_t>(std::move(block));
                data_ptr_  = buffer_->template as<T>();

                std::copy(values.begin(), values.end(), data_ptr_);
                xpu::mem::mark_host_dirty_if_needed(buffer_);
            }
        }

        // move semantics (shared_handle_t handles this)
        store_t(store_t&&) noexcept            = default;
        store_t& operator=(store_t&&) noexcept = default;

        // shallow copy semantics (reference counting)
        store_t(const store_t&)            = default;
        store_t& operator=(const store_t&) = default;

        // ---------------------------------------------------------------------
        // accessors
        // ---------------------------------------------------------------------

        // mutable indexing (host-only)
        T& operator[](std::uint64_t idx)
        {
            xpu::mem::mark_host_dirty_if_needed(buffer_);
            return data_ptr_[idx];
        }

        // read-only indexing (device-safe via cached pointer)
        DUAL const T& operator[](std::uint64_t idx) const
        {
            return data_ptr_[idx];
        }

        // mutable access (host-only, with coherency tracking)
        T* data()
        {
            if (buffer_) {
                xpu::mem::mark_host_dirty_if_needed(buffer_);
            }
            return data_ptr_;
        }

        // read-only access (device-safe via cached pointer)
        DUAL const T* data() const
        {
            return data_ptr_;
        }

        DUAL std::uint64_t size() const
        {
            return size_;
        }
        std::uint64_t capacity() const
        {
            return capacity_;
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
            if (size_ >= capacity_) {
                // geometric growth: 2x or minimum 16
                std::uint64_t new_capacity = capacity_ == 0 ? 16 : capacity_ * 2;
                reserve(new_capacity);
            }

            data_ptr_[size_] = value;
            xpu::mem::mark_host_dirty_if_needed(buffer_);
            ++size_;
        }

        // reserve capacity
        void reserve(std::uint64_t new_capacity)
        {
            if (new_capacity > capacity_) {
                // create new larger buffer
                auto new_block  = xpu::make_memory_block<T>(new_capacity);
                auto new_buffer = xpu::make_shared_handle<xpu::sim_block_t>(std::move(new_block));
                T*   new_ptr    = new_buffer->template as<T>();

                // copy existing data if any
                if (buffer_ && size_ > 0) {
                    std::copy(data_ptr_, data_ptr_ + size_, new_ptr);
                    xpu::mem::mark_host_dirty_if_needed(new_buffer);
                }

                buffer_   = std::move(new_buffer);
                data_ptr_ = new_ptr;
                capacity_ = new_capacity;
            }
        }

        // clear (keep capacity)
        void clear()
        {
            size_ = 0;
        }
    };

} // namespace simbi

#endif // CONTAINERS_STORE_HPP
