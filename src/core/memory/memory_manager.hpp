// =============================================================================
// memory/memory_manager.hpp - simple memory manager for ndarray
// =============================================================================
#ifndef MEMORY_MEMORY_MANAGER_HPP
#define MEMORY_MEMORY_MANAGER_HPP

#include "adapter/device_adapter_api.hpp"   // for gpu::api::malloc, gpu::api::copy_host_to_device, etc.
#include "config.hpp"                   // for platform::is_gpu
#include "core/types/alias/alias.hpp"   // for gpuDeleter
#include "core/utility/smart_ptr.hpp"   // for unique_ptr, smart_ptr
#include <cstddef>                      // for size_t

namespace simbi::memory {

    template <typename T>
    class memory_manager_t
    {
      public:
        // ctors and dtors
        memory_manager_t()  = default;
        ~memory_manager_t() = default;

        // copy semantics - deep copy both host and device
        memory_manager_t(const memory_manager_t& other)
        {
            if (other.size_ > 0) {
                allocate(other.size_);

                // Copy host data
                std::copy(
                    other.host_data_.get(),
                    other.host_data_.get() + size_,
                    host_data_.get()
                );

                // Copy device data if it exists
                if constexpr (platform::is_gpu) {
                    if (other.device_synced_) {
                        sync_to_device();
                    }
                }
            }
        }

        // copy assignment operator
        memory_manager_t& operator=(const memory_manager_t& other)
        {
            if (this != &other) {
                if (other.size_ > 0) {
                    allocate(other.size_);

                    std::copy(
                        other.host_data_.get(),
                        other.host_data_.get() + size_,
                        host_data_.get()
                    );

                    if constexpr (platform::is_gpu) {
                        if (other.device_synced_) {
                            sync_to_device();
                        }
                    }
                }
                else {
                    deallocate();
                }
            }
            return *this;
        }

        // Move semantics - transfer ownership
        memory_manager_t(memory_manager_t&& other) noexcept
            : host_data_(std::move(other.host_data_)),
              device_data_(std::move(other.device_data_)),
              size_(other.size_),
              capacity_(other.capacity_),
              host_synced_(other.host_synced_),
              device_synced_(other.device_synced_)
        {

            other.size_          = 0;
            other.capacity_      = 0;
            other.host_synced_   = true;
            other.device_synced_ = true;
        }

        // move assignment operator
        memory_manager_t& operator=(memory_manager_t&& other) noexcept
        {
            if (this != &other) {
                host_data_     = std::move(other.host_data_);
                device_data_   = std::move(other.device_data_);
                size_          = other.size_;
                capacity_      = other.capacity_;
                host_synced_   = other.host_synced_;
                device_synced_ = other.device_synced_;

                other.size_          = 0;
                other.capacity_      = 0;
                other.host_synced_   = true;
                other.device_synced_ = true;
            }
            return *this;
        }

        // mem allocation
        void allocate(size_t size)
        {
            if (size == 0) {
                deallocate();
                return;
            }

            size_     = size;
            capacity_ = size;

            // allocate host memory
            host_data_     = util::make_unique_array<T[]>(size);
            host_synced_   = true;
            device_synced_ = false;

            // allocate device memory if GPU enabled
            if constexpr (platform::is_gpu) {
                T* device_ptr;
                gpu::api::malloc(
                    reinterpret_cast<void**>(&device_ptr),
                    size * sizeof(T)
                );
                device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);
            }
        }

        void deallocate()
        {
            host_data_.reset();
            if constexpr (platform::is_gpu) {
                device_data_.reset();
            }
            size_          = 0;
            capacity_      = 0;
            host_synced_   = true;
            device_synced_ = true;
        }

        // resize with capacity management (like std::vector)
        void resize(size_t new_size)
        {
            if (new_size <= capacity_) {
                size_ = new_size;
                return;
            }

            // need to grow - allocate new memory
            memory_manager_t<T> new_mem;
            new_mem.allocate(new_size);

            // copy existing data
            if (size_ > 0) {
                std::copy(
                    host_data_.get(),
                    host_data_.get() + size_,
                    new_mem.host_data_.get()
                );
            }

            // move the new memory into this object
            *this = std::move(new_mem);
            size_ = new_size;
        }

        void reserve(size_t new_capacity)
        {
            if (new_capacity <= capacity_) {
                return;
            }

            memory_manager_t<T> new_mem;
            new_mem.allocate(new_capacity);

            // copy existing data
            if (size_ > 0) {
                std::copy(
                    host_data_.get(),
                    host_data_.get() + size_,
                    new_mem.host_data_.get()
                );
            }

            // move and update capacity
            *this     = std::move(new_mem);
            capacity_ = new_capacity;
        }

        // data access
        T* host_data() { return host_data_.get(); }
        const T* host_data() const { return host_data_.get(); }

        T* device_data()
        {
            if constexpr (platform::is_gpu) {
                return device_data_.get();
            }
            return nullptr;
        }

        const T* device_data() const
        {
            if constexpr (platform::is_gpu) {
                return device_data_.get();
            }
            return nullptr;
        }

        // smart data access - returns appropriate pointer for context
        T* data()
        {
            if constexpr (platform::is_gpu) {
                ensure_device_synced();
                return device_data_.get();
            }
            else {
                return host_data_.get();
            }
        }

        const T* data() const
        {
            if constexpr (platform::is_gpu) {
                return device_data_.get();
            }
            else {
                return host_data_.get();
            }
        }

        void set_data(T* data, size_t count)
        {
            if (data == nullptr || count == 0) {
                deallocate();
                return;
            }

            // ensure we have enough capacity
            if (count > capacity_) {
                reserve(count);
            }

            // copy data into host memory
            std::copy(data, data + count, host_data_.get());
            size_ = count;

            // mark as dirty
            host_synced_   = false;
            device_synced_ = false;
        }

        // element access (context-aware)
        DUAL T& operator[](size_t index)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            return device_data_.get()[index];
#else
            return host_data_.get()[index];
#endif
        }

        DUAL const T& operator[](size_t index) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            return device_data_.get()[index];
#else
            return host_data_.get()[index];
#endif
        }

        // synchronize
        void sync_to_device()
        {
            if constexpr (platform::is_gpu) {
                if (!device_data_) {
                    T* device_ptr;
                    gpu::api::malloc(
                        reinterpret_cast<void**>(&device_ptr),
                        capacity_ * sizeof(T)
                    );
                    device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);
                }

                gpu::api::copy_host_to_device(
                    device_data_.get(),
                    host_data_.get(),
                    size_ * sizeof(T)
                );
                device_synced_ = true;
            }
        }

        void sync_to_host()
        {
            if constexpr (platform::is_gpu) {
                gpu::api::copy_device_to_host(
                    host_data_.get(),
                    device_data_.get(),
                    size_ * sizeof(T)
                );
                host_synced_ = true;
            }
        }

        void ensure_device_synced()
        {
            if constexpr (platform::is_gpu) {
                if (!device_synced_) {
                    sync_to_device();
                }
            }
        }

        void ensure_host_synced()
        {
            if constexpr (platform::is_gpu) {
                if (!host_synced_) {
                    sync_to_host();
                }
            }
        }

        // status queries
        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }
        bool is_device_synced() const { return device_synced_; }
        bool is_host_synced() const { return host_synced_; }

        // mark sync state dirty (for external modifications)
        void mark_host_dirty() { device_synced_ = false; }

        void mark_device_dirty() { host_synced_ = false; }

      private:
        util::smart_ptr<T[]> host_data_;
        unique_ptr<T, gpuDeleter<T>> device_data_;
        size_t size_        = 0;
        size_t capacity_    = 0;
        bool host_synced_   = true;
        bool device_synced_ = true;
    };

}   // namespace simbi::memory
#endif
