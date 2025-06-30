#ifndef SIMBI_MEMORY_HPP
#define SIMBI_MEMORY_HPP

#include "config.hpp"
#include "core/types/alias.hpp"
#include "system/adapter/device_adapter_api.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>

namespace simbi::base {
    // =============================================================================
    // Memory Management
    // =============================================================================

    enum class memory_location {
        cpu,
        gpu
    };

    template <typename T>
    struct configurable_deleter {
        bool should_delete = true;

        void operator()(T* ptr)
        {
            if (should_delete && ptr) {
                delete[] ptr;
            }
        }
    };

    template <typename T>
    class unified_memory_t
    {
      private:
        std::unique_ptr<T[], configurable_deleter<T>> cpu_ptr_;
        std::unique_ptr<T[], gpuDeleter<T>> gpu_ptr_;
        std::uint64_t size_              = 0;
        std::uint64_t capacity_          = 0;
        memory_location active_location_ = memory_location::cpu;
        bool cpu_valid_                  = false;   // cpu data current?
        bool gpu_valid_                  = false;   // gpu data current?
        bool owns_cpu_memory_            = true;    // do we own the CPU memory?

      public:
        using value_type   = T;
        unified_memory_t() = default;

        unified_memory_t(std::uint64_t size)
            : size_(size),
              capacity_(size),
              active_location_(memory_location::cpu),
              cpu_valid_(true),
              gpu_valid_(false)
        {
            auto deleter = configurable_deleter<T>{true};   // owns memory
            cpu_ptr_     = std::unique_ptr<T[], configurable_deleter<T>>(
                new T[size],
                deleter
            );
            if constexpr (platform::is_gpu) {
                T* device_ptr;
                gpu::api::malloc(
                    reinterpret_cast<void**>(&device_ptr),
                    this->size_ * sizeof(T)
                );
                gpu_ptr_ = std::unique_ptr<T[], gpuDeleter<T>>(device_ptr);
            }
        }

        ~unified_memory_t() = default;

        // copy constructor - deep copy both CPU and GPU data
        unified_memory_t(const unified_memory_t& other)
            : size_(other.size_),
              capacity_(other.capacity_),
              active_location_(other.active_location_),
              cpu_valid_(other.cpu_valid_),
              gpu_valid_(false)   // we'll copy GPU data only if needed
        {
            if (this != &other) {
                size_      = other.size_;
                gpu_valid_ = other.gpu_valid_;

                // allocate and copy host data
                cpu_ptr_ = std::make_unique<T[]>(size_);
                if (size_ > 0 && other.cpu_ptr_) {
                    std::copy(
                        other.cpu_ptr_.get(),
                        other.cpu_ptr_.get() + size_,
                        cpu_ptr_.get()
                    );
                }

                // allocate and copy device data if applicable
                if constexpr (platform::is_gpu) {
                    if (size_ > 0) {
                        T* device_ptr;
                        gpu::api::malloc(
                            reinterpret_cast<void**>(&device_ptr),
                            size_ * sizeof(T)
                        );
                        gpu_ptr_ =
                            std::unique_ptr<T[], gpuDeleter<T>>(device_ptr);

                        // copy data from other's device memory if available
                        if (other.gpu_ptr_) {
                            gpu::api::copy_device_to_device(
                                gpu_ptr_.get(),
                                other.gpu_ptr_.get(),
                                size_ * sizeof(T)
                            );
                        }
                        // otherwise copy from host data
                        else if (!gpu_valid_) {
                            to_gpu();
                        }
                    }
                }
            }
        }

        // copy assignment
        unified_memory_t& operator=(const unified_memory_t& other)
        {
            if (this != &other) {
                size_      = other.size_;
                gpu_valid_ = other.gpu_valid_;

                // Allocate and copy host data
                cpu_ptr_ = std::make_unique<T[]>(size_);
                if (size_ > 0 && other.cpu_ptr_) {
                    std::copy(
                        other.cpu_ptr_.get(),
                        other.cpu_ptr_.get() + size_,
                        cpu_ptr_.get()
                    );
                }

                // allocate and copy device data if applicable
                if constexpr (platform::is_gpu) {
                    if (size_ > 0) {
                        T* device_ptr;
                        gpu::api::malloc(
                            reinterpret_cast<void**>(&device_ptr),
                            size_ * sizeof(T)
                        );
                        gpu_ptr_ =
                            std::unique_ptr<T, gpuDeleter<T>>(device_ptr);

                        // copy data from other's device memory if available
                        if (other.gpu_ptr_) {
                            gpu::api::copy_device_to_device(
                                gpu_ptr_.get(),
                                other.gpu_ptr_.get(),
                                size_ * sizeof(T)
                            );
                        }
                        // otherwise copy from host data
                        else if (!gpu_valid_) {
                            to_gpu();
                        }
                    }
                }
            }
            return *this;
        }

        // movable
        unified_memory_t(unified_memory_t&& other) noexcept
            : size_(other.size_),
              capacity_(other.capacity_),
              active_location_(other.active_location_),
              cpu_valid_(other.cpu_valid_),
              gpu_valid_(other.gpu_valid_)
        {
            cpu_ptr_        = std::move(other.cpu_ptr_);
            gpu_ptr_        = std::move(other.gpu_ptr_);
            other.size_     = 0;
            other.capacity_ = 0;
        }

        // move assignment
        unified_memory_t& operator=(unified_memory_t&& other) noexcept
        {
            if (this != &other) {
                // move from other
                cpu_ptr_         = std::move(other.cpu_ptr_);
                gpu_ptr_         = std::move(other.gpu_ptr_);
                size_            = other.size_;
                capacity_        = other.capacity_;
                active_location_ = other.active_location_;
                cpu_valid_       = other.cpu_valid_;
                gpu_valid_       = other.gpu_valid_;

                // reset other
                other.size_     = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        void resize(std::uint64_t new_size)
        {
            if (new_size > capacity_) {
                // allocate new memory
                auto deleter = configurable_deleter<T>{true};   // owns memory
                cpu_ptr_     = std::unique_ptr<T[], configurable_deleter<T>>(
                    new T[new_size],
                    deleter
                );
                capacity_ = new_size;

                if (gpu_ptr_) {
                    gpu_ptr_.reset(new T[new_size]);
                    to_gpu();
                }

                gpu_valid_       = false;
                cpu_valid_       = true;
                active_location_ = memory_location::cpu;
            }
            size_ = new_size;
        }

        void wrap_external_memory(T* external_ptr, std::uint64_t size)
        {
            cpu_ptr_.reset();
            if (gpu_ptr_) {
                gpu_ptr_.reset();
            }

            // wrap the external pointer
            auto no_delete = configurable_deleter<T>{false};   // don't delete
            cpu_ptr_       = std::unique_ptr<T[], configurable_deleter<T>>(
                external_ptr,
                no_delete
            );
            size_            = size;
            capacity_        = size;
            owns_cpu_memory_ = false;   // don't own this memory
            cpu_valid_       = true;
            gpu_valid_       = false;
            active_location_ = memory_location::cpu;
        }

        void reserve(std::uint64_t new_capacity)
        {
            if (new_capacity > capacity_) {
                // allocate new memory
                cpu_ptr_.reset(new T[new_capacity]);
                capacity_ = new_capacity;

                if (gpu_ptr_) {
                    gpu_ptr_.reset(new T[new_capacity]);
                }

                gpu_valid_       = false;
                cpu_valid_       = true;
                active_location_ = memory_location::cpu;
            }
        }

        // smart synchronization methods
        void ensure_cpu_synced()
        {
            if (!cpu_valid_ && gpu_valid_) {
                to_cpu();
            }
        }

        void ensure_gpu_synced()
        {
            if (!gpu_valid_ && cpu_valid_) {
                to_gpu();
            }
        }

        void to_cpu()
        {
            if (!cpu_valid_ && gpu_valid_) {
                gpu::api::copy_device_to_host(
                    cpu_ptr_.get(),
                    gpu_ptr_.get(),
                    size_ * sizeof(T)
                );
                cpu_valid_ = true;
            }
            active_location_ = memory_location::cpu;
        }

        void to_gpu()
        {
            if constexpr (platform::is_gpu) {
                if (!gpu_ptr_) {
                    T* device_ptr;
                    // lazy GPU allocation
                    gpu::api::malloc(
                        reinterpret_cast<void**>(&device_ptr),
                        size_ * sizeof(T)
                    );
                    gpu_ptr_ = std::unique_ptr<T[], gpuDeleter<T>>(device_ptr);
                }

                if (!gpu_valid_ && cpu_valid_) {
                    gpu::api::copy_host_to_device(
                        gpu_ptr_.get(),
                        cpu_ptr_.get(),
                        size_ * sizeof(T)
                    );
                    gpu_valid_ = true;
                }
                active_location_ = memory_location::gpu;
            }
        }

        // marking methods for external modifications
        void mark_cpu_dirty()
        {
            cpu_valid_ = true;
            gpu_valid_ = false;
        }

        void mark_gpu_dirty()
        {
            gpu_valid_ = true;
            cpu_valid_ = false;
        }

        // external data management
        void set_data(T* data, std::uint64_t count, bool take_ownership = false)
        {
            if (data == nullptr || count == 0) {
                size_     = 0;
                capacity_ = 0;
                return;
            }

            if (take_ownership) {
                // take ownership of the provided memory
                auto deleter = configurable_deleter<T>{true};
                cpu_ptr_     = std::unique_ptr<T[], configurable_deleter<T>>(
                    new T[count],
                    deleter
                );
                std::copy(data, data + count, cpu_ptr_.get());
            }
            else {
                // just reference the data - don't delete
                auto no_delete = configurable_deleter<T>{false};
                cpu_ptr_       = std::unique_ptr<T[], configurable_deleter<T>>(
                    data,
                    no_delete
                );
            }

            size_            = count;
            cpu_valid_       = true;
            gpu_valid_       = false;
            active_location_ = memory_location::cpu;

            // reset device data and sync state
            if constexpr (platform::is_gpu) {
                // we need to allocate device memory before it can be used
                T* device_ptr;
                gpu::api::malloc(
                    reinterpret_cast<void**>(&device_ptr),
                    this->size_ * sizeof(T)
                );
                gpu_ptr_   = std::unique_ptr<T[], gpuDeleter<T>>(device_ptr);
                gpu_valid_ = false;
            }
        }

        T* data()
        {
            if constexpr (platform::is_gpu) {
                if (!gpu_valid_) {
                    to_gpu();
                }
                return gpu_ptr_.get();
            }
            return cpu_ptr_.get();
        }

        DUAL const T* data() const
        {
            if constexpr (platform::is_gpu) {
                return gpu_ptr_.get();
            }
            return cpu_ptr_.get();
        }

        // explicit accessors that ensure synchronization
        T* cpu_data()
        {
            ensure_cpu_synced();
            return cpu_ptr_.get();
        }

        DUAL T* gpu_data()
        {
            ensure_gpu_synced();
            return gpu_ptr_.get();
        }

        std::uint64_t size() const { return size_; }
        std::uint64_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }

        memory_location location() const { return active_location_; }

        void invalidate_cpu() { cpu_valid_ = false; }
        void invalidate_gpu() { gpu_valid_ = false; }

        // iterator enable
        auto begin() { return cpu_ptr_.get(); }
        auto end() { return cpu_ptr_.get() + size_; }
        auto begin() const { return cpu_ptr_.get(); }
        auto end() const { return cpu_ptr_.get() + size_; }
    };
}   // namespace simbi::base

#endif   // SIMBI_MEMORY_HPP
