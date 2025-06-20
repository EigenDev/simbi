/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            memory_manager.hpp
 *  * @brief           a helper struct to manage memory on the GPU and CPU
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-21
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-21      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include "core/types/alias/alias.hpp"
#include "core/types/monad/maybe.hpp"
#include "core/types/utility/smart_ptr.hpp"

namespace simbi {
    template <typename T>
    class memory_manager

    {
      public:
        // custom constructors

        // default
        memory_manager() = default;

        // copy
        memory_manager(const memory_manager& other)
        {
            if (this != &other) {
                size_      = other.size_;
                is_synced_ = other.is_synced_;

                // Allocate and copy host data
                host_data_ = util::make_unique_array<T[]>(size_);
                if (size_ > 0 && other.host_data_) {
                    std::copy(
                        other.host_data_.get(),
                        other.host_data_.get() + size_,
                        host_data_.get()
                    );
                }

                // Allocate and copy device data if applicable
                if constexpr (platform::is_gpu) {
                    if (size_ > 0) {
                        T* device_ptr;
                        gpu::api::malloc(
                            reinterpret_cast<void**>(&device_ptr),
                            size_ * sizeof(T)
                        );
                        device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);

                        // Copy data from other's device memory if available
                        if (other.device_data_) {
                            gpu::api::copy_device_to_device(
                                device_data_.get(),
                                other.device_data_.get(),
                                size_ * sizeof(T)
                            );
                        }
                        // Otherwise copy from host data
                        else if (is_synced_) {
                            sync_to_device();
                        }
                    }
                }
            }
        }

        // move
        memory_manager(memory_manager&& other) noexcept
        {
            host_data_   = std::move(other.host_data_);
            size_        = other.size_;
            is_synced_   = other.is_synced_;
            device_data_ = std::move(other.device_data_);
            other.size_  = 0;
        }

        // copy assignment
        memory_manager& operator=(const memory_manager& other)
        {
            if (this != &other) {
                size_      = other.size_;
                is_synced_ = other.is_synced_;

                // Allocate and copy host data
                host_data_ = util::make_unique_array<T[]>(size_);
                if (size_ > 0 && other.host_data_) {
                    std::copy(
                        other.host_data_.get(),
                        other.host_data_.get() + size_,
                        host_data_.get()
                    );
                }

                // Allocate and copy device data if applicable
                if constexpr (platform::is_gpu) {
                    if (size_ > 0) {
                        T* device_ptr;
                        gpu::api::malloc(
                            reinterpret_cast<void**>(&device_ptr),
                            size_ * sizeof(T)
                        );
                        device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);

                        // Copy data from other's device memory if available
                        if (other.device_data_) {
                            gpu::api::copy_device_to_device(
                                device_data_.get(),
                                other.device_data_.get(),
                                size_ * sizeof(T)
                            );
                        }
                        // Otherwise copy from host data
                        else if (is_synced_) {
                            sync_to_device();
                        }
                    }
                }
            }
            return *this;
        }

        // move assignment
        memory_manager& operator=(memory_manager&& other) noexcept
        {
            if (this != &other) {
                host_data_   = std::move(other.host_data_);
                device_data_ = std::move(other.device_data_);
                size_        = other.size_;
                is_synced_   = other.is_synced_;

                // clear the other
                other.size_      = 0;
                other.is_synced_ = true;
            }
            return *this;
        }

        // destructor (not default so GPU doesn't complain)
        ~memory_manager() {};

        void allocate(size_type size)
        {
            this->size_ = size;
            host_data_  = util::make_unique_array<T[]>(size);
            if constexpr (platform::is_gpu) {
                T* device_ptr;
                gpu::api::malloc(
                    reinterpret_cast<void**>(&device_ptr),
                    this->size_ * sizeof(T)
                );
                device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);
            }
        }

        void reference_external(
            T* data_ptr,
            size_type size,
            bool take_ownership = false
        )
        {
            // clean up existing data if needed
            this->size_ = size;

            if (take_ownership) {
                // take ownership - copy the data
                host_data_ = util::make_unique_array<T[]>(size);
                // std::memcpy(host_data_.get(), data_ptr, size * sizeof(T));
            }
            else {
                // just reference the data without copying or owning it
                // we use a custom deleter that does nothing when the pointer is
                // released
                host_data_ = util::smart_ptr<T[]>(data_ptr, [](T*) {});
            }

            // reset device data and sync state
            if constexpr (platform::is_gpu) {
                // we need to allocate device memory before it can be used
                T* device_ptr;
                gpu::api::malloc(
                    reinterpret_cast<void**>(&device_ptr),
                    this->size_ * sizeof(T)
                );
                device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);
                is_synced_   = false;
            }
        }

        template <typename V, typename U>
        void construct_maybe_wrapper(const U* raw_data, size_type size)
        {
            // allocate storage for Maybe<V> objects
            this->allocate(size);

            // construct Maybe<V> objects directly in the allocated storage
            for (size_type i = 0; i < size; ++i) {
                new (&(host_data()[i])) Maybe<V>(raw_data[i]);
            }

            is_synced_ = false;
        }

        T* data()
        {
            if constexpr (platform::is_gpu) {
                if (!is_synced_) {
                    sync_to_device();
                }
                return device_data_.get();
            }
            return host_data_.get();
        }
        DUAL T* data() const
        {
            if constexpr (platform::is_gpu) {
                return device_data_.get();
            }
            return host_data_.get();
        }

        void sync_to_device()
        {
            if constexpr (platform::is_gpu) {
                // check if device memory is allocated
                if (!device_data_) {
                    T* device_ptr;
                    gpu::api::malloc(
                        reinterpret_cast<void**>(&device_ptr),
                        this->size_ * sizeof(T)
                    );
                    device_data_ = unique_ptr<T, gpuDeleter<T>>(device_ptr);
                }

                gpu::api::copy_host_to_device(
                    device_data_.get(),
                    host_data_.get(),
                    this->size_ * sizeof(T)
                );
                is_synced_ = true;
            }
        }
        void sync_to_host()
        {
            if constexpr (platform::is_gpu) {
                gpu::api::copy_device_to_host(
                    host_data_.get(),
                    device_data_.get(),
                    this->size_ * sizeof(T)
                );
                is_synced_ = true;
            }
        }
        void ensure_device_synced()
        {
            if constexpr (platform::is_gpu) {
                if (!is_synced_) {
                    sync_to_device();
                }
            }
        }

        // access operators
        DUAL T& operator[](size_type ii)
        {
// if accessing from the device, get the device data
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            return device_data()[ii];
#else
            return host_data()[ii];
#endif
        }

        DUAL T& operator[](size_type ii) const
        {
// if accessing from the device, get the device data
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            return device_data()[ii];
#else
            return host_data()[ii];
#endif
        }

        // host data accessors
        DUAL T* host_data() { return host_data_.get(); }

        DUAL T* host_data() const { return host_data_.get(); }

        // device data accessors
        DUAL T* device_data() { return device_data_.get(); }
        DUAL T* device_data() const { return device_data_.get(); }

      private:
        util::smart_ptr<T[]> host_data_;
        unique_ptr<T, gpuDeleter<T>> device_data_;
        bool is_synced_{false};
        size_type size_{0};
    };
}   // namespace simbi

#endif
