#ifndef HETERO_DETAIL_CPU_IMPL_HPP
#define HETERO_DETAIL_CPU_IMPL_HPP

#include "../core/backend_traits.hpp"
#include "../core/common_types.hpp"
#include "../core/error_handling.hpp"
#include "../core/resource_types.hpp"
#include "../device/execution_context.hpp"
#include "adapter_impl.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ratio>

namespace simbi::hetero {

    template <>
    class stream_t<cpu_backend_t>
    {
        using native_stream = stream_handle<cpu_backend_t>;
        native_stream handle_;
        bool owns_resource_;

      public:
        stream_t() : owns_resource_(true) {}

        ~stream_t() { destroy(); }

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

        void synchronize() {}

        bool query_complete() { return true; }

      private:
        void destroy() { owns_resource_ = false; }
    };

    template <>
    class event_t<cpu_backend_t>
    {
        using native_event = event_handle<cpu_backend_t>;
        std::chrono::high_resolution_clock::time_point timestamp_;
        bool owns_resource_;

      public:
        event_t() : owns_resource_(true) {}

        ~event_t() { destroy(); }

        event_t(const event_t&)            = delete;
        event_t& operator=(const event_t&) = delete;

        event_t(event_t&& other) noexcept
            : timestamp_(other.timestamp_), owns_resource_(other.owns_resource_)
        {
            other.owns_resource_ = false;
        }

        event_t& operator=(event_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                timestamp_           = other.timestamp_;
                owns_resource_       = other.owns_resource_;
                other.owns_resource_ = false;
            }
            return *this;
        }

        native_event native_handle() const noexcept
        {
            return native_event{};   // dummy handle for CPU
        }

        void record(const stream_t<cpu_backend_t>&)
        {
            timestamp_ = std::chrono::high_resolution_clock::now();
        }

        void synchronize()
        {
            // no-op for CPU since operations are synchronous
        }

        double elapsed_time_ms(const event_t<cpu_backend_t>& start_event)
        {
            auto duration = timestamp_ - start_event.timestamp_;
            return std::chrono::duration<double, std::milli>(duration).count();
        }

      private:
        void destroy() { owns_resource_ = false; }
    };

    template <>
    class device_memory_t<cpu_backend_t>
    {
        void* ptr_;
        size_t size_;
        bool owns_resource_;
        bool is_managed_ = false;

      public:
        device_memory_t(size_t bytes) : size_(bytes), owns_resource_(true)
        {
            ptr_ = std::malloc(bytes);
            if (!ptr_ && bytes > 0) {
                throw compute_error(
                    status_t::out_of_memory,
                    "cpu malloc failed"
                );
            }
        }
        device_memory_t(size_t bytes, bool) : device_memory_t(bytes) {}

        ~device_memory_t() { destroy(); }

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
                other.ptr_           = nullptr;
                other.owns_resource_ = false;
                other.is_managed_    = false;
            }
            return *this;
        }

        void* data() const noexcept { return ptr_; }
        size_t size() const noexcept { return size_; }
        bool is_managed() const noexcept { return is_managed_; }

        template <typename T>
        T* as() const noexcept
        {
            return static_cast<T*>(ptr_);
        }

      private:
        void destroy()
        {
            if (owns_resource_ && ptr_) {
                std::free(ptr_);
                ptr_ = nullptr;
            }
            owns_resource_ = false;
        }
    };

    template <>
    class device_adapter_t<cpu_backend_t>
    {
      public:
        using stream_type = stream_t<cpu_backend_t>;
        using event_type  = event_t<cpu_backend_t>;
        using memory_type = device_memory_t<cpu_backend_t>;

        template <typename T>
        using vector_type = device_vector_t<cpu_backend_t, T>;

        static void
        copy(void* dst, const void* src, size_t bytes, memory_kind_t)
        {
            std::memcpy(dst, src, bytes);
        }

        static void copy_async(
            void* dst,
            const void* src,
            size_t bytes,
            memory_kind_t kind,
            const stream_type&
        )
        {
            copy(dst, src, bytes, kind);
        }

        static memory_type allocate(size_t bytes) { return memory_type(bytes); }

        template <typename T>
        static vector_type<T> allocate_vector(size_t count)
        {
            return vector_type<T>(count);
        }

        template <typename T>
        static vector_type<T> allocate_managed_vector(size_t count)
        {
            return vector_type<T>(count, true);
        }

        static void prefetch_to_device(const void*, size_t, int) {}

        static stream_type create_stream() { return stream_type(); }

        static event_type create_event() { return event_type(); }

        static void synchronize_device() {}

        static std::int64_t get_device_count() { return 1; }

        static memory_type allocate_managed(std::size_t bytes)
        {
            return memory_type(bytes);
        }

        static bool
        can_access_peer(std::int64_t device_id, std::int64_t peer_device_id)
        {
            if (device_id != 0 || peer_device_id != 0) {
                throw compute_error(
                    status_t::invalid_argument,
                    "cpu backend only supports device 0"
                );
            }
            return true;
        }

        static void enable_peer_access(std::int64_t peer_device_id)
        {
            if (peer_device_id != 0) {
                throw compute_error(
                    status_t::invalid_argument,
                    "cpu backend only supports device 0"
                );
            }
        }

        static void set_device(std::int64_t device_id)
        {
            if (device_id != 0) {
                throw compute_error(
                    status_t::invalid_argument,
                    "cpu backend only supports device 0"
                );
            }
        }

        static std::int64_t get_current_device() { return 0; }

        template <typename kernel_t, typename... args_t>
        static void launch_kernel(
            kernel_t kernel,
            dim3_t grid,
            dim3_t block,
            args_t... args
        )
        {
            for (std::uint32_t z = 0; z < grid.z; ++z) {
                for (std::uint32_t y = 0; y < grid.y; ++y) {
                    for (std::uint32_t x = 0; x < grid.x; ++x) {
                        for (std::uint32_t bz = 0; bz < block.z; ++bz) {
                            for (std::uint32_t by = 0; by < block.y; ++by) {
                                for (std::uint32_t bx = 0; bx < block.x; ++bx) {
                                    kernel(args...);
                                }
                            }
                        }
                    }
                }
            }
        }

        template <typename kernel_t, typename... args_t>
        static void launch_kernel_async(
            kernel_t kernel,
            dim3_t grid,
            dim3_t block,
            const stream_type&,
            args_t... args
        )
        {
            launch_kernel(kernel, grid, block, args...);
        }

        template <typename kernel_t, typename... args_t>
        static void launch(
            kernel_t kernel,
            grid::launch_config_t& launch_config,
            args_t... args
        )
        {
            launch_kernel(
                kernel,
                launch_config.grid(),
                launch_config.block(),
                args...
            );
        }

        static void memset(void* ptr, std::int64_t value, size_t bytes)
        {
            std::memset(ptr, value, bytes);
        }

        static void memset_async(
            void* ptr,
            std::int64_t value,
            size_t bytes,
            const stream_type&
        )
        {
            memset(ptr, value, bytes);
        }

        template <typename T>
        static void
        copy_vector_to_host(T* host_ptr, const vector_type<T>& device_vec)
        {
            std::memcpy(
                host_ptr,
                device_vec.typed_data(),
                device_vec.size_bytes()
            );
        }

        template <typename T>
        static void
        copy_vector_from_host(vector_type<T>& device_vec, const T* host_ptr)
        {
            std::memcpy(
                device_vec.typed_data(),
                host_ptr,
                device_vec.size_bytes()
            );
        }

        template <typename T>
        static void copy_vector_to_host_async(
            T* host_ptr,
            const vector_type<T>& device_vec,
            const stream_type&
        )
        {
            copy_vector_to_host(host_ptr, device_vec);
        }

        template <typename T>
        static void copy_vector_from_host_async(
            vector_type<T>& device_vec,
            const T* host_ptr,
            const stream_type&
        )
        {
            copy_vector_from_host(device_vec, host_ptr);
        }

        static constexpr const char* backend_name() { return "cpu"; }

        static constexpr bool supports_async_operations() { return false; }

        static constexpr bool supports_peer_access() { return false; }
    };

}   // namespace simbi::hetero

#endif   // SIMBI_HETERO_DETAIL_CPU_IMPL_HPP
