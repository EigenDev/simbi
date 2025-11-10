#ifndef STORE_HPP
#define STORE_HPP

#include "compat.hpp"
#include "hetero/adapter.hpp"
#include "hetero/core/common_types.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace simbi {
    template <typename T>
    class store_t
    {
      private:
        hetero::managed_vector<T> data_;
        std::uint64_t size_ = 0;

        void ensure_capacity(std::uint64_t required)
        {
            if (required <= data_.size()) {
                return;
            }

            std::uint64_t new_capacity =
                std::max<std::uint64_t>(required, data_.size() * 2);
            if (data_.size() == 0) {
                new_capacity = std::max(required, std::uint64_t(16));
            }

            auto new_data =
                hetero::device::allocate_managed_vector<T>(new_capacity);

            if (size_ > 0) {
                hetero::device::copy(
                    new_data.data(),
                    data_.data(),
                    size_ * sizeof(T),
                    hetero::memory_direction_t::device_to_device
                );
            }

            data_ = std::move(new_data);
        }

      public:
        store_t()
            : data_(hetero::device::allocate_managed_vector<T>(0)), size_(0)
        {
        }

        explicit store_t(std::uint64_t size, const T& initial_value = T())
            : data_(hetero::device::allocate_managed_vector<T>(size)),
              size_(size)
        {
            for (std::uint64_t ii = 0; ii < size_; ++ii) {
                data_[ii] = initial_value;
            }
        }

        explicit store_t(const std::vector<T>& values)
            : data_(hetero::device::allocate_managed_vector<T>(values.size())),
              size_(values.size())
        {
            hetero::device::copy_vector_from_host(data_, values.data());
        }

        // Rule of 5 automatically handled by managed_vector RAII

        DUAL T& operator[](std::uint64_t idx) { return data_[idx]; }
        DUAL const T& operator[](std::uint64_t idx) const { return data_[idx]; }

        DUAL T* data() { return data_.typed_data(); }
        DUAL const T* data() const { return data_.typed_data(); }

        DUAL std::uint64_t size() const { return size_; }
        DUAL std::uint64_t capacity() const { return data_.size(); }
        DUAL bool empty() const { return size_ == 0; }

        void add(const T& value)
        {
            ensure_capacity(size_ + 1);
            data_[size_++] = value;
        }

        void reserve(std::uint64_t new_capacity)
        {
            if (new_capacity > data_.size()) {
                ensure_capacity(new_capacity);
            }
        }

        void clear() { size_ = 0; }

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
            auto device_count = hetero::device::get_device_count();
            for (int ii = 0; ii < device_count; ii++) {
                hetero::device::prefetch_to_device(
                    data_.data(),
                    size_ * sizeof(T),
                    ii
                );
            }
        }
    };
}   // namespace simbi

#endif
