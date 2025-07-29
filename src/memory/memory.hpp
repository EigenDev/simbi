#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "span.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace simbi {

    /**
     * owned_span_t - RAII wrapper for owned memory
     *
     * SRP: manage lifetime of allocated memory
     * provides span_t view of owned data
     */
    template <typename T>
    class owned_span_t
    {
        std::unique_ptr<T[]> ptr_;
        std::size_t size_;

      public:
        // construction
        owned_span_t() noexcept : ptr_(nullptr), size_(0) {}

        explicit owned_span_t(std::size_t count)
            : ptr_(std::make_unique<T[]>(count)), size_(count)
        {
        }

        owned_span_t(std::unique_ptr<T[]> ptr, std::size_t count) noexcept
            : ptr_(std::move(ptr)), size_(count)
        {
        }

        // move only
        owned_span_t(const owned_span_t&)            = delete;
        owned_span_t& operator=(const owned_span_t&) = delete;

        owned_span_t(owned_span_t&&) noexcept            = default;
        owned_span_t& operator=(owned_span_t&&) noexcept = default;

        // span access
        span_t<T> span() noexcept { return span_t<T>{ptr_.get(), size_}; }

        span_t<const T> span() const noexcept
        {
            return span_t<const T>{ptr_.get(), size_};
        }

        // implicit conversion to span
        operator span_t<T>() noexcept { return span(); }
        operator span_t<const T>() const noexcept { return span(); }

        // direct access (for convenience)
        T* data() noexcept { return ptr_.get(); }
        const T* data() const noexcept { return ptr_.get(); }

        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }

        T& operator[](std::size_t idx) noexcept { return ptr_[idx]; }
        const T& operator[](std::size_t idx) const noexcept
        {
            return ptr_[idx];
        }

        // iterator support
        T* begin() noexcept { return ptr_.get(); }
        T* end() noexcept { return ptr_.get() + size_; }

        const T* begin() const noexcept { return ptr_.get(); }
        const T* end() const noexcept { return ptr_.get() + size_; }

        // release ownership
        std::unique_ptr<T[]> release() noexcept
        {
            size_ = 0;
            return std::move(ptr_);
        }

        // reset
        void reset() noexcept
        {
            ptr_.reset();
            size_ = 0;
        }

        void reset(std::unique_ptr<T[]> new_ptr, std::size_t new_size) noexcept
        {
            ptr_  = std::move(new_ptr);
            size_ = new_size;
        }
    };

    /**
     * borrowed_span_t - non-owning wrapper that tracks original owner
     *
     * SRP: provide safe access to externally managed memory
     * useful for numpy arrays, GPU buffers, etc.
     */
    template <typename T>
    class borrowed_span_t
    {
        T* data_;
        std::size_t size_;
        std::shared_ptr<void> owner_;   // keeps original owner alive

      public:
        // construction
        borrowed_span_t() noexcept : data_(nullptr), size_(0) {}

        borrowed_span_t(
            T* ptr,
            std::size_t count,
            std::shared_ptr<void> owner = nullptr
        ) noexcept
            : data_(ptr), size_(count), owner_(std::move(owner))
        {
        }

        // span access
        span_t<T> span() noexcept { return span_t<T>{data_, size_}; }

        span_t<const T> span() const noexcept
        {
            return span_t<const T>{data_, size_};
        }

        // implicit conversion to span
        operator span_t<T>() noexcept { return span(); }
        operator span_t<const T>() const noexcept { return span(); }

        // direct access
        T* data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }

        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }

        T& operator[](std::size_t idx) noexcept { return data_[idx]; }
        const T& operator[](std::size_t idx) const noexcept
        {
            return data_[idx];
        }

        // iterator support
        T* begin() noexcept { return data_; }
        T* end() noexcept { return data_ + size_; }

        const T* begin() const noexcept { return data_; }
        const T* end() const noexcept { return data_ + size_; }

        // check if owner is still alive
        bool owner_valid() const noexcept { return owner_ != nullptr; }
        std::size_t owner_use_count() const noexcept
        {
            return owner_ ? owner_.use_count() : 0;
        }
    };

    // factory functions for clean construction
    template <typename T>
    auto make_owned_span(std::size_t count)
    {
        return owned_span_t<T>{count};
    }

    template <typename T>
    auto make_borrowed_span(
        T* ptr,
        std::size_t count,
        std::shared_ptr<void> owner = nullptr
    )
    {
        return borrowed_span_t<T>{ptr, count, std::move(owner)};
    }

    // zero-initialized owned span
    template <typename T>
    auto make_zeroed_span(std::size_t count)
    {
        auto result = owned_span_t<T>{count};
        std::fill(result.begin(), result.end(), T{});
        return result;
    }

    // numpy integration helpers
    template <typename T>
    auto wrap_numpy_array(
        void* numpy_ptr,
        std::size_t count,
        std::shared_ptr<void> python_obj = nullptr
    )
    {
        return borrowed_span_t<T>{
          static_cast<T*>(numpy_ptr),
          count,
          std::move(python_obj)
        };
    }

    template <typename T>
    auto copy_from_numpy(void* numpy_ptr, std::size_t count)
    {
        auto result = owned_span_t<T>{count};
        std::copy_n(static_cast<const T*>(numpy_ptr), count, result.data());
        return result;
    }

}   // namespace simbi

#endif
