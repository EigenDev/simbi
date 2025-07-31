#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "adapter/device_adapter_api.hpp"
#include "config.hpp"
#include "span.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace simbi::mem {
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

    /**
     * device_owned_span_t - RAII wrapper for device memory
     *
     * SRP: manage lifetime of GPU/CPU allocated memory
     * uses device_adapter_api underneath
     */
    template <typename T>
    class device_owned_span_t
    {
        T* data_;
        std::size_t size_;
        bool is_device_memory_;

      public:
        // construction
        device_owned_span_t() noexcept
            : data_(nullptr), size_(0), is_device_memory_(false)
        {
        }

        explicit device_owned_span_t(
            std::size_t count,
            bool use_device = global::on_gpu
        )
            : size_(count), is_device_memory_(use_device)
        {

            if (is_device_memory_) {
                gpu::api::malloc(
                    reinterpret_cast<void**>(&data_),
                    count * sizeof(T)
                );
            }
            else {
                data_ = new T[count];
            }
        }

        // move only
        device_owned_span_t(const device_owned_span_t&)            = delete;
        device_owned_span_t& operator=(const device_owned_span_t&) = delete;

        device_owned_span_t(device_owned_span_t&& other) noexcept
            : data_(other.data_),
              size_(other.size_),
              is_device_memory_(other.is_device_memory_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        device_owned_span_t& operator=(device_owned_span_t&& other) noexcept
        {
            if (this != &other) {
                cleanup();
                data_             = other.data_;
                size_             = other.size_;
                is_device_memory_ = other.is_device_memory_;
                other.data_       = nullptr;
                other.size_       = 0;
            }
            return *this;
        }

        ~device_owned_span_t() { cleanup(); }

        // span access - only safe for CPU memory
        span_t<T> span() noexcept
        {
            // Note: This is only safe for CPU memory
            // GPU memory access requires explicit copy operations
            return span_t<T>{data_, size_};
        }

        span_t<const T> span() const noexcept
        {
            return span_t<const T>{data_, size_};
        }

        // resize support
        void resize(std::size_t new_size, bool use_device = global::on_gpu)
        {
            if (new_size == size_) {
                return;   // no change
            }

            cleanup();   // free old memory

            size_             = new_size;
            is_device_memory_ = use_device;

            if (is_device_memory_) {
                gpu::api::malloc(
                    reinterpret_cast<void**>(&data_),
                    new_size * sizeof(T)
                );
            }
            else {
                data_ = new T[new_size];
            }
        }

        // direct access (use with caution for GPU memory)
        DUAL T* data() noexcept { return data_; }
        DUAL const T* data() const noexcept { return data_; }

        std::size_t size() const noexcept { return size_; }
        std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }
        bool empty() const noexcept { return size_ == 0; }
        bool is_device_memory() const noexcept { return is_device_memory_; }

        void
        copy_from_host(const T* host_data, std::size_t count = std::size_t(-1))
        {
            if (count == std::size_t(-1)) {
                count = size_;
            }

            if (is_device_memory_) {
                gpu::api::copy_host_to_device(
                    data_,
                    host_data,
                    count * sizeof(T)
                );
            }
            else {
                std::copy_n(host_data, count, data_);
            }
        }

        void
        copy_to_host(T* host_data, std::size_t count = std::size_t(-1)) const
        {
            if (count == std::size_t(-1)) {
                count = size_;
            }

            if (is_device_memory_) {
                gpu::api::copy_device_to_host(
                    host_data,
                    data_,
                    count * sizeof(T)
                );
            }
            else {
                std::copy_n(data_, count, host_data);
            }
        }

        void copy_from_device(
            const device_owned_span_t<T>& other,
            std::size_t count = std::size_t(-1)
        )
        {
            if (count == std::size_t(-1)) {
                count = std::min(size_, other.size_);
            }

            if (is_device_memory_ && other.is_device_memory_) {
                gpu::api::copy_device_to_device(
                    data_,
                    other.data_,
                    count * sizeof(T)
                );
            }
            else if (!is_device_memory_ && other.is_device_memory_) {
                gpu::api::copy_device_to_host(
                    data_,
                    other.data_,
                    count * sizeof(T)
                );
            }
            else if (is_device_memory_ && !other.is_device_memory_) {
                gpu::api::copy_host_to_device(
                    data_,
                    other.data_,
                    count * sizeof(T)
                );
            }
            else {
                std::copy_n(other.data_, count, data_);
            }
        }

        void fill(const T& value)
        {
            if (is_device_memory_) {
                // Note: memset only works for byte values
                // For general T, would need a kernel launch
                if constexpr (std::is_trivially_copyable_v<T> &&
                              sizeof(T) == 1) {
                    gpu::api::memset(data_, static_cast<int>(value), size_);
                }
                else {
                    // Would need to implement a GPU kernel for general fill
                    throw std::runtime_error(
                        "Device fill not implemented for this type"
                    );
                }
            }
            else {
                std::fill_n(data_, size_, value);
            }
        }

      private:
        void cleanup() noexcept
        {
            if (data_) {
                try {
                    if (is_device_memory_) {
                        gpu::api::free(data_);
                    }
                    else {
                        delete[] data_;
                    }
                }
                catch (...) {
                    // destructor must not throw
                }
            }
        }
    };

    // factory functions
    template <typename T>
    auto make_device_span(std::size_t count, bool use_device = global::on_gpu)
    {
        return device_owned_span_t<T>{count, use_device};
    }

    template <typename T>
    auto make_cpu_span(std::size_t count)
    {
        return device_owned_span_t<T>{count, false};
    }

    template <typename T>
    auto make_gpu_span(std::size_t count)
    {
        return device_owned_span_t<T>{count, true};
    }

    template <typename T>
    auto
    make_zeroed_device_span(std::size_t count, bool use_device = global::on_gpu)
    {
        auto result = device_owned_span_t<T>{count, use_device};
        result.fill(T{});
        return result;
    }

    // transfer operations
    template <typename T>
    auto copy_to_device(const span_t<const T>& cpu_data)
    {
        auto device_buffer = make_gpu_span<T>(cpu_data.size());
        device_buffer.copy_from_host(cpu_data.data, cpu_data.size());
        return device_buffer;
    }

    template <typename T>
    auto copy_to_host(const device_owned_span_t<T>& device_data)
    {
        auto cpu_buffer = make_cpu_span<T>(device_data.size());
        device_data.copy_to_host(cpu_buffer.data(), device_data.size());
        return cpu_buffer;
    }

    // numpy integration with device awareness
    template <typename T>
    auto copy_numpy_to_device(void* numpy_ptr, std::size_t count)
    {
        auto device_buffer = make_gpu_span<T>(count);
        device_buffer.copy_from_host(static_cast<const T*>(numpy_ptr), count);
        return device_buffer;
    }

    template <typename T>
    auto copy_device_to_numpy(
        const device_owned_span_t<T>& device_data,
        void* numpy_ptr
    )
    {
        device_data.copy_to_host(
            static_cast<T*>(numpy_ptr),
            device_data.size()
        );
    }

}   // namespace simbi::mem

#endif
