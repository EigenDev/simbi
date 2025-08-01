#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include "base/concepts.hpp"
#include "config.hpp"
#include "memory/device.hpp"
#include "memory/span.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace simbi::nd {
    using namespace simbi::concepts;
    using namespace simbi::mem;

    template <typename T, std::uint64_t Dims = 1>
    class ndarray_t
    {
      private:
        device_owned_span_t<T> memory_;
        std::uint64_t size_ = 0;

        void ensure_cpu_accessible(const char* operation) const
        {
            if (is_on_gpu()) {
                throw std::runtime_error(
                    std::string(operation) +
                    " requires cpu memory - call move_to_cpu() first"
                );
            }
        }

      public:
        using value_type                    = T;
        static constexpr std::uint64_t rank = Dims;

        // construction
        ndarray_t() = default;

        explicit ndarray_t(std::uint64_t size, bool use_gpu = global::on_gpu)
            : memory_(size, use_gpu), size_(size)
        {
        }

        // move semantics
        ndarray_t(ndarray_t&&) noexcept            = default;
        ndarray_t& operator=(ndarray_t&&) noexcept = default;

        // copy semantics (explicit deep copy)
        ndarray_t(const ndarray_t& other)
            : memory_(other.size_, other.is_on_gpu()), size_(other.size_)
        {
            if (size_ > 0) {
                if (other.is_on_gpu()) {
                    memory_.copy_from_device(other.memory_, size_);
                }
                else {
                    memory_.copy_from_host(other.memory_.data(), size_);
                }
            }
        }

        ndarray_t& operator=(const ndarray_t& other)
        {
            if (this != &other) {
                memory_ =
                    device_owned_span_t<T>(other.size_, other.is_on_gpu());
                size_ = other.size_;
                if (size_ > 0) {
                    if (other.is_on_gpu()) {
                        memory_.copy_from_device(other.memory_, size_);
                    }
                    else {
                        memory_.copy_from_host(other.memory_.data(), size_);
                    }
                }
            }
            return *this;
        }

        // from vector (always starts on cpu)
        explicit ndarray_t(const std::vector<T>& vec)
            : memory_(vec.size(), false), size_(vec.size())
        {
            if (!vec.empty()) {
                memory_.copy_from_host(vec.data(), vec.size());
            }
        }

        // element access (cpu only, bounds checked in debug)
        DUAL T& operator[](std::uint64_t idx)
        {
            if constexpr (global::bounds_checking) {
                if constexpr (platform::is_cpu) {
                    if (idx >= size_) {
                        throw std::out_of_range("ndarray index out of bounds");
                    }
                }
                else {
                    // GPU bounds checking is not supported
                    printf("ndarray index out of bounds: %lu", idx);
                }
            }
            return memory_.data()[idx];
        }

        DUAL const T& operator[](std::uint64_t idx) const
        {
            if constexpr (global::bounds_checking) {
                // bounds checking only in debug mode
                if constexpr (platform::is_cpu) {
                    if (idx >= size_) {
                        throw std::out_of_range("ndarray index out of bounds");
                    }
                }
                else {
                    // GPU bounds checking is not supported
                    printf("ndarray index out of bounds: %lu", idx);
                }
            }
            return memory_.data()[idx];
        }

        // device transfers - copy operations (immutable)
        [[nodiscard]] ndarray_t copy_to_gpu() const
        {
            if constexpr (platform::is_cpu) {
                return *this;
            }
            if (is_on_gpu()) {
                return *this;   // copy constructor handles deep copy
            }

            ndarray_t gpu_copy(size_, true);
            if (size_ > 0) {
                gpu_copy.memory_.copy_from_host(memory_.data(), size_);
            }
            return gpu_copy;
        }

        [[nodiscard]] ndarray_t copy_to_cpu() const
        {
            if constexpr (platform::is_cpu) {
                return *this;
            }
            if (is_on_cpu()) {
                return *this;   // copy constructor handles deep copy
            }

            ndarray_t cpu_copy(size_, false);
            if (size_ > 0) {
                memory_.copy_to_host(cpu_copy.memory_.data(), size_);
            }
            return cpu_copy;
        }

        // device transfers - move operations
        void move_to_gpu()
        {
            if constexpr (platform::is_cpu) {
                return;
            }
            if (is_on_gpu()) {
                return;
            }

            device_owned_span_t<T> gpu_memory(size_, true);
            if (size_ > 0) {
                gpu_memory.copy_from_host(memory_.data(), size_);
            }
            memory_ = std::move(gpu_memory);
        }

        void move_to_cpu()
        {
            if constexpr (platform::is_cpu) {
                return;
            }
            if (is_on_cpu()) {
                return;
            }

            device_owned_span_t<T> cpu_memory(size_, false);
            if (size_ > 0) {
                memory_.copy_to_host(cpu_memory.data(), size_);
            }
            memory_ = std::move(cpu_memory);
        }

        // dynamic operations (cpu only)
        void reserve(std::uint64_t new_capacity)
        {
            if (new_capacity <= capacity()) {
                return;
            }

            device_owned_span_t<T> new_memory(new_capacity, is_on_gpu());
            if (size_ > 0) {
                if (is_on_gpu()) {
                    new_memory.copy_from_device(memory_, size_);
                }
                else {
                    new_memory.copy_from_host(memory_.data(), size_);
                }
            }
            memory_ = std::move(new_memory);
        }

        void push_back(const T& value)
        {
            ensure_cpu_accessible("push_back");

            if (size_ >= capacity()) {
                std::uint64_t new_cap = capacity() == 0 ? 1 : capacity() * 2;
                reserve(new_cap);
            }

            memory_.data()[size_] = value;
            ++size_;
        }

        void resize(std::uint64_t new_size)
        {
            if (new_size > capacity()) {
                reserve(new_size);
            }
            size_ = new_size;
        }

        // data access
        T* data() { return memory_.data(); }
        const T* data() const { return memory_.data(); }

        // properties
        std::uint64_t size() const { return size_; }
        std::uint64_t capacity() const { return memory_.size(); }
        bool empty() const { return size_ == 0; }

        bool is_on_gpu() const { return memory_.is_device_memory(); }
        bool is_on_cpu() const { return !memory_.is_device_memory(); }

        // utilities
        void fill(const T& value)
        {
            if (size_ > 0) {
                memory_.fill(value);
            }
        }

        template <ArrayFunction<Dims> Func>
        void initialize_from(const Func& func)
        {
            ensure_cpu_accessible("initialize_from");

            for (std::uint64_t i = 0; i < size_; ++i) {
                memory_.data()[i] = func(i);
            }
        }

        // conversion to vector (always creates cpu copy)
        [[nodiscard]] std::vector<T> to_vector() const
        {
            if (empty()) {
                return {};
            }

            std::vector<T> result(size_);
            if (is_on_gpu()) {
                memory_.copy_to_host(result.data(), size_);
            }
            else {
                std::copy_n(memory_.data(), size_, result.data());
            }
            return result;
        }

        // span access (cpu only)
        span_t<T> span()
        {
            ensure_cpu_accessible("span access");
            return span_t<T>{memory_.data(), size_};
        }

        span_t<const T> span() const
        {
            ensure_cpu_accessible("span access");
            return span_t<const T>{memory_.data(), size_};
        }
    };

}   // namespace simbi::nd

#endif   // NDARRAY_SYSTEM_HPP
