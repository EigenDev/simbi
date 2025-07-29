// =============================================================================
// Complete ndarray_t System
// =============================================================================
#ifndef SIMBI_NDARRAY_SYSTEM_HPP
#define SIMBI_NDARRAY_SYSTEM_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/base/memory.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace simbi::nd {
    using namespace simbi::concepts;
    using namespace simbi::base;
    // =============================================================================
    // Memory-Backed ndarray_t
    // =============================================================================
    template <typename T, std::uint64_t Dims = 1>
    class ndarray_t
    {
      private:
        std::unique_ptr<unified_memory_t<T>> memory_;
        std::uint64_t size_ = 0;

      public:
        using value_type                    = T;
        static constexpr std::uint64_t rank = Dims;

        ndarray_t() = default;

        // construction
        ndarray_t(const std::uint64_t& size)
            : memory_(std::make_unique<unified_memory_t<T>>(size)), size_(size)
        {
        }

        // move constructor
        ndarray_t(ndarray_t&& other) noexcept
            : memory_(std::move(other.memory_)), size_(other.size_)
        {
            other.size_ = 0;   // reset other's size
        }

        // move assignment
        ndarray_t& operator=(ndarray_t&& other) noexcept
        {
            if (this != &other) {
                memory_     = std::move(other.memory_);
                size_       = other.size_;
                other.size_ = 0;   // reset other's size
            }
            return *this;
        }

        // from vector
        ndarray_t(const std::vector<T>& vec)
            : memory_(std::make_unique<unified_memory_t<T>>(vec.size())),
              size_(vec.size())
        {
            std::copy(vec.begin(), vec.end(), memory_->data());
        }

        // element access (immutable)
        DUAL T& operator[](const std::uint64_t idx)
        {
            return memory_->data()[idx];
        }

        DUAL const T& operator[](const std::uint64_t idx) const
        {
            return memory_->data()[idx];
        }

        void reserve(std::uint64_t new_size)
        {
            if (!memory_) {
                memory_ = std::make_unique<unified_memory_t<T>>(new_size);
                return;
            }
            if (new_size > capacity()) {
                memory_->reserve(new_size);
            }
        }

        void push_back(const T& value)
        {
            memory_->data()[size_] = value;
            size_++;
        }

        void push_back_with_sync(const T& value)
        {
            push_back(value);
            ensure_host_synced();   // sync after push_back
        }

        // Enhanced memory management methods
        void to_gpu() { memory_->to_gpu(); }

        void to_cpu() { memory_->to_cpu(); }

        void ensure_device_synced() { memory_->ensure_gpu_synced(); }

        void ensure_host_synced() { memory_->ensure_cpu_synced(); }

        void mark_host_dirty() { memory_->mark_cpu_dirty(); }

        void mark_device_dirty() { memory_->mark_gpu_dirty(); }

        bool is_host_synced() const { return memory_->is_cpu_valid(); }

        bool is_device_synced() const { return memory_->is_gpu_valid(); }

        T* cpu_data() const { return memory_->cpu_data(); }
        T* gpu_data() const { return memory_->gpu_data(); }
        T* data() const { return memory_->data(); }

        // properties
        std::uint64_t size() const { return size_; }
        std::uint64_t capacity() const
        {
            return memory_ ? memory_->capacity() : 0;
        }

        // utility
        void fill(const T& value)
        {
            T* ptr = memory_->data();
            for (std::uint64_t i = 0; i < size(); ++i) {
                ptr[i] = value;
            }
        }

        // initialization from function
        template <ArrayFunction<Dims> Func>
        void initialize_from(const Func& func)
        {
            if (!memory_) {
                throw std::runtime_error("ndarray_t not initialized");
            }
            for (std::uint64_t ii = 0; ii < size(); ++ii) {
                memory_->data()[ii] = func(ii);
            }
        }
    };

}   // namespace simbi::nd

#endif   // NDARRAY_SYSTEM_HPP
