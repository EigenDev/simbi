#ifndef STORE_HPP
#define STORE_HPP

#include "adapter/device_adapter_api.hpp"
#include "config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace simbi {
    template <typename T>
    class store_t
    {
      private:
        T* data_                = nullptr;
        std::uint64_t size_     = 0;
        std::uint64_t capacity_ = 0;

        void cleanup()
        {
            if (data_) {
                gpu::api::free(data_);
                data_     = nullptr;
                size_     = 0;
                capacity_ = 0;
            }
        }

        void ensure_capacity(std::uint64_t required)
        {
            if (required <= capacity_) {
                return;
            }

            std::uint64_t new_capacity = std::max(required, capacity_ * 2);
            if (capacity_ == 0) {
                new_capacity = std::max(required, std::uint64_t(16));
            }

            T* new_data = nullptr;
            gpu::api::malloc_managed(
                &reinterpret_cast<void*&>(new_data),
                new_capacity * sizeof(T)
            );

            if (size_ > 0) {
                gpu::api::copy_device_to_device(
                    new_data,
                    data_,
                    size_ * sizeof(T)
                );
            }

            if (data_) {
                gpu::api::free(data_);
            }

            data_     = new_data;
            capacity_ = new_capacity;
        }

      public:
        store_t() = default;

        explicit store_t(std::uint64_t size, const T& initial_value = T())
            : size_(size), capacity_(size)
        {
            if (size_ > 0) {
                gpu::api::malloc_managed(
                    &reinterpret_cast<void*&>(data_),
                    size_ * sizeof(T)
                );

                // initialize with the given value
                for (std::uint64_t ii = 0; ii < size_; ++ii) {
                    data_[ii] = initial_value;
                }
            }
        }

        explicit store_t(const std::vector<T>& values)
            : size_(values.size()), capacity_(values.size())
        {
            if (size_ > 0) {
                gpu::api::malloc_managed(
                    &reinterpret_cast<void*&>(data_),
                    size_ * sizeof(T)
                );
                gpu::api::copy_host_to_device(
                    data_,
                    values.data(),
                    size_ * sizeof(T)
                );
            }
        }

        ~store_t() { cleanup(); }

        // move semantics
        store_t(store_t&& other) noexcept
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
        {
            other.data_     = nullptr;
            other.size_     = 0;
            other.capacity_ = 0;
        }

        store_t& operator=(store_t&& other) noexcept
        {
            if (this != &other) {
                cleanup();
                data_           = other.data_;
                size_           = other.size_;
                capacity_       = other.capacity_;
                other.data_     = nullptr;
                other.size_     = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        // copy semantics
        store_t(const store_t& other)
            : size_(other.size_), capacity_(other.size_)
        {
            if (size_ > 0) {
                gpu::api::malloc_managed(
                    &reinterpret_cast<void*&>(data_),
                    size_ * sizeof(T)
                );
                gpu::api::copy_device_to_device(
                    data_,
                    other.data_,
                    size_ * sizeof(T)
                );
            }
        }

        store_t& operator=(const store_t& other)
        {
            if (this != &other) {
                cleanup();
                size_     = other.size_;
                capacity_ = other.size_;

                if (size_ > 0) {
                    gpu::api::malloc_managed(
                        &reinterpret_cast<void*&>(data_),
                        size_ * sizeof(T)
                    );
                    gpu::api::copy_device_to_device(
                        data_,
                        other.data_,
                        size_ * sizeof(T)
                    );
                }
            }
            return *this;
        }

        DUAL T& operator[](std::uint64_t idx) { return data_[idx]; }
        DUAL const T& operator[](std::uint64_t idx) const { return data_[idx]; }

        DUAL T* data() { return data_; }
        DUAL const T* data() const { return data_; }

        // properties
        std::uint64_t size() const { return size_; }
        std::uint64_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }

        void add(const T& value)
        {
            ensure_capacity(size_ + 1);
            data_[size_++] = value;
        }

        void reserve(std::uint64_t new_capacity)
        {
            if (new_capacity > capacity_) {
                ensure_capacity(new_capacity);
            }
        }

        void clear()
        {
            size_ = 0;
            // Note: This keeps the allocated memory for reuse
        }

        void resize(std::uint64_t new_size, const T& value = T())
        {
            if (new_size > size_) {
                ensure_capacity(new_size);

                for (std::uint64_t ii = size_; ii < new_size; ++ii) {
                    data_[ii] = value;
                }
            }

            size_ = new_size;
        }

        void sync_to_all_devices()
        {
            if (size_ > 0) {
                std::int64_t device_count = 0;
                gpu::api::get_device_count(&device_count);

                for (int ii = 0; ii < device_count; ii++) {
                    gpu::api::prefetch_to_device(data_, size_ * sizeof(T), ii);
                }
            }
        }
    };

}   // namespace simbi
#endif
