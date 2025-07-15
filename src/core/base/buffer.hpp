#ifndef BUFFER_HPP
#define BUFFER_HPP

#include "system/adapter/device_adapter_api.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace simbi {

    enum class device_type_t : std::uint8_t {
        cpu,
        gpu
    };

    struct device_id_t {
        int node_id;          // which physical node/machine
        int device_id;        // which gpu/cpu on that node
        device_type_t type;   // cpu or gpu

        // convenience constructors
        static constexpr device_id_t cpu_device(int node = 0)
        {
            return {node, 0, device_type_t::cpu};
        }

        static constexpr device_id_t gpu_device(int device, int node = 0)
        {
            return {node, device, device_type_t::gpu};
        }

        // comparison for containers/maps
        auto operator<=>(const device_id_t&) const = default;
    };

    template <typename T>
    struct buffer_t {
        T* data_;
        std::size_t size_;
        device_id_t device_;
        bool owns_memory_;

        // construction
        buffer_t()
            : data_(nullptr),
              size_(0),
              device_(device_id_t::cpu_device()),
              owns_memory_(true)
        {
            // empty buffer, no allocation needed
        }

        buffer_t(
            std::size_t size,
            device_id_t device = device_id_t::cpu_device()
        )
            : data_(nullptr), size_(size), device_(device), owns_memory_(true)
        {

            if (device.type == device_type_t::cpu) {
                data_ = new T[size];
            }
            else {
                gpu::api::set_device(device.device_id);
                gpu::api::malloc(
                    reinterpret_cast<void**>(&data_),
                    size * sizeof(T)
                );
            }
        }

        // non-allocating constructor
        buffer_t(
            T* data,
            std::size_t size,
            device_id_t device = device_id_t::cpu_device(),
            bool owns_memory   = true
        )
            : data_(data),
              size_(size),
              device_(device),
              owns_memory_(owns_memory)
        {
            // if we don't own memory, we assume data is already allocated
            if (owns_memory_ && !data_) {
                throw std::runtime_error(
                    "Cannot create buffer with null data when owning memory"
                );
            }
        }

        ~buffer_t()
        {
            if (data_ && owns_memory_) {
                if (device_.type == device_type_t::cpu) {
                    delete[] data_;
                }
                else {
                    gpu::api::set_device(device_.device_id);
                    gpu::api::free(data_);
                }
            }
        }

        // move only
        buffer_t(buffer_t&& other) noexcept
            : data_(other.data_),
              size_(other.size_),
              device_(other.device_),
              owns_memory_(other.owns_memory_)
        {
            // Explicitly nullify source to prevent double deletion
            other.data_        = nullptr;
            other.size_        = 0;
            other.owns_memory_ = false;
        }

        buffer_t& operator=(buffer_t&& other) noexcept
        {
            if (this != &other) {
                // Clean up our existing resources first
                if (data_ && owns_memory_) {
                    if (device_.type == device_type_t::cpu) {
                        delete[] data_;
                    }
                    else {
                        gpu::api::set_device(device_.device_id);
                        gpu::api::free(data_);
                    }
                }

                // Transfer ownership
                data_        = other.data_;
                size_        = other.size_;
                device_      = other.device_;
                owns_memory_ = other.owns_memory_;

                // Nullify source
                other.data_        = nullptr;
                other.size_        = 0;
                other.owns_memory_ = false;
            }
            return *this;
        }
        buffer_t(const buffer_t&)            = delete;
        buffer_t& operator=(const buffer_t&) = delete;

        // zero-copy numpy wrapping
        static buffer_t wrap_numpy(
            void* array,
            std::size_t size,
            device_id_t target_device = device_id_t::cpu_device()
        )
        {
            return buffer_t{static_cast<T*>(array), size, target_device, false};
        };

        static buffer_t copy_from_numpy(
            void* numpy_array,
            std::size_t size,
            device_id_t target_device = device_id_t::cpu_device()
        )
        {
            // allocate our own memory
            buffer_t buf(size, target_device);

            // copy from numpy array to our buffer
            if (target_device.type == device_type_t::cpu) {
                std::memcpy(buf.data_, numpy_array, size * sizeof(T));
            }
            else {
                gpu::api::set_device(target_device.device_id);
                // copy from host numpy to our gpu buffer
                gpu::api::copy_host_to_device(
                    buf.data_,
                    numpy_array,
                    size * sizeof(T)
                );
            }

            return buf;
        }

        // simple accessors
        T* data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }
        std::size_t size() const noexcept { return size_; }
        device_id_t device() const noexcept { return device_; }
        bool empty() const noexcept { return size_ == 0; }
    };
}   // namespace simbi

#endif
