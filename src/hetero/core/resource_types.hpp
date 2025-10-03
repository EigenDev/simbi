#ifndef HETERO_CORE_RESOURCE_TYPES_HPP
#define HETERO_CORE_RESOURCE_TYPES_HPP

#include "backend_traits.hpp"
#include "compat.hpp"

#include <cstddef>

namespace simbi::hetero {

    template <typename backend_t>
    class stream_t
    {
        using native_stream = stream_handle<backend_t>;
        native_stream handle_;
        bool owns_resource_;

      public:
        stream_t();
        ~stream_t();

        stream_t(const stream_t&)            = delete;
        stream_t& operator=(const stream_t&) = delete;

        stream_t(stream_t&& other) noexcept
            : handle_(other.handle_), owns_resource_(other.owns_resource_)
        {
            other.owns_resource_ = false;
        }

        stream_t& operator=(stream_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                handle_              = other.handle_;
                owns_resource_       = other.owns_resource_;
                other.owns_resource_ = false;
            }
            return *this;
        }

        native_stream native_handle() const noexcept { return handle_; }

        void synchronize();
        bool query_complete();
        operator bool() const noexcept;

      private:
        void destroy();
    };

    template <typename backend_t>
    class event_t
    {
        using native_event = event_handle<backend_t>;
        native_event handle_;
        bool owns_resource_;

      public:
        event_t();
        ~event_t();

        event_t(const event_t&)            = delete;
        event_t& operator=(const event_t&) = delete;

        event_t(event_t&& other) noexcept
            : handle_(other.handle_), owns_resource_(other.owns_resource_)
        {
            other.owns_resource_ = false;
        }

        event_t& operator=(event_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                handle_              = other.handle_;
                owns_resource_       = other.owns_resource_;
                other.owns_resource_ = false;
            }
            return *this;
        }

        native_event native_handle() const noexcept { return handle_; }

        void record(const stream_t<backend_t>& stream);
        void synchronize();
        float elapsed_time_ms(const event_t& start_event);

      private:
        void destroy();
    };

    template <typename backend_t>
    class device_memory_t
    {
        void* ptr_;
        std::size_t size_;
        bool owns_resource_;
        bool is_managed_;
        bool is_host_memory_ = false;

      public:
        enum class alloc_type {
            device,
            host,
        };
        device_memory_t(std::size_t bytes);
        device_memory_t(std::size_t bytes, bool managed);
        device_memory_t(std::size_t bytes, alloc_type type);
        device_memory_t(std::size_t bytes, bool managed, alloc_type type);
        ~device_memory_t();

        device_memory_t(const device_memory_t&)            = delete;
        device_memory_t& operator=(const device_memory_t&) = delete;

        device_memory_t(device_memory_t&& other) noexcept
            : ptr_(other.ptr_),
              size_(other.size_),
              owns_resource_(other.owns_resource_),
              is_managed_(other.is_managed_)
        {
            other.ptr_           = nullptr;
            other.owns_resource_ = false;
        }

        device_memory_t& operator=(device_memory_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                ptr_                 = other.ptr_;
                size_                = other.size_;
                owns_resource_       = other.owns_resource_;
                is_managed_          = other.is_managed_;
                other.ptr_           = nullptr;
                other.owns_resource_ = false;
            }
            return *this;
        }

        DUAL void* data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return size_; }
        bool is_managed() const noexcept { return is_managed_; }
        bool is_host_memory() const noexcept { return is_host_memory_; }

        template <typename T>
        DUAL T* as() const noexcept
        {
            return static_cast<T*>(ptr_);
        }

      private:
        void destroy();
    };

    template <typename backend_t, typename T>
    class device_vector_t
    {
        device_memory_t<backend_t> memory_;
        std::size_t count_;

      public:
        explicit device_vector_t(std::size_t count)
            : memory_(count * sizeof(T)), count_(count)
        {
        }

        explicit device_vector_t(std::size_t count, bool managed)
            : memory_(count * sizeof(T), managed), count_(count)
        {
        }

        void* data() const noexcept { return memory_.data(); }
        DUAL T* typed_data() const noexcept { return memory_.template as<T>(); }
        std::size_t size() const noexcept { return count_; }
        std::size_t size_bytes() const noexcept { return count_ * sizeof(T); }
        bool is_managed() const noexcept { return memory_.is_managed(); }

        DUAL T& operator[](std::size_t index) noexcept
        {
            return typed_data()[index];
        }

        DUAL const T& operator[](std::size_t index) const noexcept
        {
            return typed_data()[index];
        }
    };

}   // namespace simbi::hetero

#endif   // HETERO_CORE_RESOURCE_TYPES_HPP
