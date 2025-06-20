// =============================================================================
// containers/ndarray.hpp - Clean std::vector-like container with GPU support
// =============================================================================
#ifndef CONTAINERS_NDARRAY_HPP
#define CONTAINERS_NDARRAY_HPP

#include "config.hpp"                       // for platform::is_gpu, DUAL, etc.
#include "core/memory/memory_manager.hpp"   // for memory_manager
#include <algorithm>                        // for std::copy, std::fill
#include <cstddef>                          // for size_t
#include <cstdio>                           // for printf
#include <initializer_list>                 // for std::initializer_list
#include <iterator>   // for std::random_access_iterator_tag, std::forward_iterator_tag
#include <stdexcept>   // for std::out_of_range
#include <vector>

namespace simbi::containers {
    template <typename T>
    class ndarray_t
    {
      public:
        // Type aliases (std::vector compatibility)
        using value_type      = T;
        using size_type       = size_t;
        using difference_type = std::ptrdiff_t;
        using reference       = T&;
        using const_reference = const T&;
        using pointer         = T*;
        using const_pointer   = const T*;

        // Construction
        ndarray_t() = default;

        explicit ndarray_t(size_type count) { mem_.allocate(count); }

        ndarray_t(size_type count, const T& value)
        {
            mem_.allocate(count);
            fill(value);
        }

        ndarray_t(std::initializer_list<T> init)
        {
            mem_.allocate(init.size());
            std::copy(init.begin(), init.end(), mem_.host_data());
            mem_.mark_host_dirty();
        }

        // Copy and move semantics (delegated to memory_manager)
        ndarray_t(const ndarray_t&)                = default;
        ndarray_t& operator=(const ndarray_t&)     = default;
        ndarray_t(ndarray_t&&) noexcept            = default;
        ndarray_t& operator=(ndarray_t&&) noexcept = default;
        ~ndarray_t()                               = default;

        // construct from std::vector
        ndarray_t(std::vector<T>&& vec)
        {
            mem_.allocate(vec.size());
            std::copy(vec.begin(), vec.end(), mem_.host_data());
            mem_.mark_host_dirty();
        }

        // construct from raw pointer
        ndarray_t(T* data, size_type count, bool take_ownership = false)
        {
            if (take_ownership) {
                mem_.set_data(data, count);
            }
            else {
                mem_.allocate(count);
                std::copy(data, data + count, mem_.host_data());
                mem_.mark_host_dirty();
            }
        }

        // element access
        DUAL reference operator[](size_type index) { return mem_[index]; }

        DUAL const_reference operator[](size_type index) const
        {
            return mem_[index];
        }

        DUAL reference at(size_type index)
        {
            if (index >= size()) {
                if constexpr (platform::is_cpu) {
                    throw std::out_of_range(
                        "ndarray_t::at: index out of range"
                    );
                }
                else {
                    printf("ndaraay::at index out of range\n");
                }
            }
            return mem_[index];
        }

        DUAL const_reference at(size_type index) const
        {
            if (index >= size()) {
                throw std::out_of_range("ndarray_t::at: index out of range");
            }
            return mem_[index];
        }

        reference front() { return mem_[0]; }
        const_reference front() const { return mem_[0]; }
        reference back() { return mem_[size() - 1]; }
        const_reference back() const { return mem_[size() - 1]; }

        // Raw data access
        DUAL pointer data() { return mem_.data(); }
        DUAL const_pointer data() const { return mem_.data(); }

        pointer host_data() { return mem_.host_data(); }
        const_pointer host_data() const { return mem_.host_data(); }

        DUAL pointer device_data() { return mem_.device_data(); }
        DUAL const_pointer device_data() const { return mem_.device_data(); }

        // Iterators (host-side only for now)
        class iterator
        {
          public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type        = T;
            using difference_type   = std::ptrdiff_t;
            using pointer           = T*;
            using reference         = T&;

            DUAL iterator(T* ptr) : ptr_(ptr) {}

            DUAL reference operator*() const { return *ptr_; }
            DUAL pointer operator->() const { return ptr_; }

            DUAL iterator& operator++()
            {
                ++ptr_;
                return *this;
            }
            DUAL iterator operator++(int)
            {
                iterator tmp = *this;
                ++ptr_;
                return tmp;
            }

            DUAL iterator& operator--()
            {
                --ptr_;
                return *this;
            }
            DUAL iterator operator--(int)
            {
                iterator tmp = *this;
                --ptr_;
                return tmp;
            }

            DUAL iterator& operator+=(difference_type n)
            {
                ptr_ += n;
                return *this;
            }

            DUAL iterator& operator-=(difference_type n)
            {
                ptr_ -= n;
                return *this;
            }

            DUAL iterator operator+(difference_type n) const
            {
                return iterator(ptr_ + n);
            }

            DUAL iterator operator-(difference_type n) const
            {
                return iterator(ptr_ - n);
            }

            DUAL difference_type operator-(const iterator& other) const
            {
                return ptr_ - other.ptr_;
            }

            DUAL reference operator[](difference_type n) const
            {
                return ptr_[n];
            }

            DUAL bool operator==(const iterator& other) const
            {
                return ptr_ == other.ptr_;
            }
            DUAL bool operator!=(const iterator& other) const
            {
                return ptr_ != other.ptr_;
            }
            DUAL bool operator<(const iterator& other) const
            {
                return ptr_ < other.ptr_;
            }
            DUAL bool operator>(const iterator& other) const
            {
                return ptr_ > other.ptr_;
            }
            DUAL bool operator<=(const iterator& other) const
            {
                return ptr_ <= other.ptr_;
            }
            DUAL bool operator>=(const iterator& other) const
            {
                return ptr_ >= other.ptr_;
            }

          private:
            T* ptr_;
        };

        using const_iterator = iterator;   // Simplified for now

        iterator begin()
        {
            mem_.ensure_host_synced();
            return iterator(mem_.host_data());
        }

        iterator end()
        {
            mem_.ensure_host_synced();
            return iterator(mem_.host_data() + size());
        }

        const_iterator begin() const
        {
            return const_iterator(mem_.host_data());
        }

        const_iterator end() const
        {
            return const_iterator(mem_.host_data() + size());
        }

        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }

        // Capacity
        bool empty() const { return mem_.empty(); }
        size_type size() const { return mem_.size(); }
        size_type capacity() const { return mem_.capacity(); }

        void reserve(size_type new_capacity) { mem_.reserve(new_capacity); }

        // Modifiers
        void clear() { mem_.deallocate(); }

        void resize(size_type new_size) { mem_.resize(new_size); }

        void resize(size_type new_size, const T& value)
        {
            size_type old_size = size();
            mem_.resize(new_size);

            // Fill new elements with value
            if (new_size > old_size) {
                std::fill(
                    mem_.host_data() + old_size,
                    mem_.host_data() + new_size,
                    value
                );
                mem_.mark_host_dirty();
            }
        }

        void push_back(const T& value)
        {
            if (size() >= capacity()) {
                size_type new_capacity = capacity() == 0 ? 1 : capacity() * 2;
                reserve(new_capacity);
            }

            // Construct new element at end
            new (mem_.host_data() + mem_.size()) T(value);
            mem_.resize(mem_.size() + 1);
            mem_.mark_host_dirty();
        }

        void push_back(T&& value)
        {
            if (size() >= capacity()) {
                size_type new_capacity = capacity() == 0 ? 1 : capacity() * 2;
                reserve(new_capacity);
            }

            new (mem_.host_data() + mem_.size()) T(std::move(value));
            mem_.resize(mem_.size() + 1);
            mem_.mark_host_dirty();
        }

        void push_back_with_sync(const T& value)
        {
            if (size() >= capacity()) {
                size_type new_capacity = capacity() == 0 ? 1 : capacity() * 2;
                reserve(new_capacity);
            }

            new (mem_.host_data() + mem_.size()) T(value);
            mem_.resize(mem_.size() + 1);
            mem_.mark_host_dirty();
            mem_.sync_to_device();
        }

        void push_back_with_sync(T&& value)
        {
            if (size() >= capacity()) {
                size_type new_capacity = capacity() == 0 ? 1 : capacity() * 2;
                reserve(new_capacity);
            }

            new (mem_.host_data() + mem_.size()) T(std::move(value));
            mem_.resize(mem_.size() + 1);
            mem_.mark_host_dirty();
            mem_.sync_to_device();
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            if (size() >= capacity()) {
                size_type new_capacity = capacity() == 0 ? 1 : capacity() * 2;
                reserve(new_capacity);
            }

            new (mem_.host_data() + mem_.size()) T(std::forward<Args>(args)...);
            mem_.resize(mem_.size() + 1);
            mem_.mark_host_dirty();
        }

        void pop_back()
        {
            if (!empty()) {
                mem_.host_data()[size() - 1].~T();
                mem_.resize(size() - 1);
            }
        }

        // GPU-specific operations
        void sync_to_device() { mem_.sync_to_device(); }

        void sync_to_host() { mem_.sync_to_host(); }

        bool is_device_synced() const { return mem_.is_device_synced(); }

        bool is_host_synced() const { return mem_.is_host_synced(); }

        // Bulk operations
        void fill(const T& value)
        {
            std::fill(mem_.host_data(), mem_.host_data() + size(), value);
            mem_.mark_host_dirty();
        }

        // Copy from another ndarray_t
        void copy_from(const ndarray_t& other)
        {
            if (this != &other) {
                resize(other.size());
                std::copy(
                    other.mem_.host_data(),
                    other.mem_.host_data() + other.size(),
                    mem_.host_data()
                );
                mem_.mark_host_dirty();
            }
        }

      private:
        memory::memory_manager_t<T> mem_;
    };

}   // namespace simbi::containers

#endif   // CONTAINERS_NDARRAY_HPP
