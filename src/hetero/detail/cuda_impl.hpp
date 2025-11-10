#ifndef HETERO_DETAIL_CUDA_IMPL_HPP
#define HETERO_DETAIL_CUDA_IMPL_HPP

#ifdef CUDA_ENABLED

#include "../core/backend_traits.hpp"
#include "../core/common_types.hpp"
#include "../core/error_handling.hpp"
#include "../core/resource_types.hpp"
#include "../device/execution_context.hpp"
#include "adapter_impl.hpp"
#include "compat.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace simbi::hetero {

    template <typename Func, typename... Args>
    KERNEL void generic_kernel(Func func, Args... args)
    {
        func(args...);
    }

    template <>
    class stream_t<cuda_backend_t>
    {
        using native_stream = stream_handle<cuda_backend_t>;
        native_stream handle_;
        bool owns_resource_;

      public:
        stream_t() : owns_resource_(true)
        {
            check_error<cuda_backend_t>(
                cudaStreamCreate(&handle_),
                "stream creation"
            );
        }

        ~stream_t() { destroy(); }

        stream_t(const stream_t&)            = delete;
        stream_t& operator=(const stream_t&) = delete;

        stream_t(stream_t&& other) noexcept
            : handle_(other.handle_), owns_resource_(other.owns_resource_)
        {
            other.owns_resource_ = false;
            other.handle_        = nullptr;
        }

        stream_t& operator=(stream_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                handle_              = other.handle_;
                owns_resource_       = other.owns_resource_;
                other.owns_resource_ = false;
                other.handle_        = nullptr;
            }
            return *this;
        }

        native_stream native_handle() const noexcept { return handle_; }

        void synchronize()
        {
            check_error<cuda_backend_t>(
                cudaStreamSynchronize(handle_),
                "stream synchronize"
            );
        }

        bool query_complete()
        {
            auto result = cudaStreamQuery(handle_);
            if (result == cudaSuccess) {
                return true;
            }
            if (result == cudaErrorNotReady) {
                return false;
            }
            check_error<cuda_backend_t>(result, "stream query");
            return false;
        }

        operator bool() const noexcept { return handle_ != nullptr; }

      private:
        void destroy()
        {
            if (owns_resource_ && handle_) {
                cudaStreamDestroy(handle_);
            }
            handle_        = nullptr;
            owns_resource_ = false;
        }
    };

    template <>
    class event_t<cuda_backend_t>
    {
        using native_event = event_handle<cuda_backend_t>;
        native_event handle_;
        bool owns_resource_;

      public:
        event_t() : owns_resource_(true)
        {
            check_error<cuda_backend_t>(
                cudaEventCreate(&handle_),
                "event creation"
            );
        }

        ~event_t() { destroy(); }

        event_t(const event_t&)            = delete;
        event_t& operator=(const event_t&) = delete;

        event_t(event_t&& other) noexcept
            : handle_(other.handle_), owns_resource_(other.owns_resource_)
        {
            other.owns_resource_ = false;
            other.handle_        = nullptr;
        }

        event_t& operator=(event_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                handle_              = other.handle_;
                owns_resource_       = other.owns_resource_;
                other.owns_resource_ = false;
                other.handle_        = nullptr;
            }
            return *this;
        }

        native_event native_handle() const noexcept { return handle_; }

        void record(const stream_t<cuda_backend_t>& stream)
        {
            check_error<cuda_backend_t>(
                cudaEventRecord(handle_, stream.native_handle()),
                "event record"
            );
        }

        void synchronize()
        {
            check_error<cuda_backend_t>(
                cudaEventSynchronize(handle_),
                "event synchronize"
            );
        }

        float elapsed_time_ms(const event_t<cuda_backend_t>& start_event)
        {
            float ms;
            check_error<cuda_backend_t>(
                cudaEventElapsedTime(&ms, start_event.handle_, handle_),
                "event elapsed time"
            );
            return ms;
        }

      private:
        void destroy()
        {
            if (owns_resource_) {
                cudaEventDestroy(handle_);
            }
            owns_resource_ = false;
        }
    };

    template <>
    class device_memory_t<cuda_backend_t>
    {
        void* ptr_;
        std::size_t size_;
        bool owns_resource_;
        bool is_managed_;
        bool is_host_memory_ = false;

      public:
        enum class alloc_type {
            device,
            host
        };

        device_memory_t(std::size_t bytes)
            : device_memory_t(bytes, alloc_type::device)
        {
        }
        device_memory_t(std::size_t bytes, bool managed)
            : device_memory_t(bytes, managed, alloc_type::device)
        {
        }

        device_memory_t(std::size_t bytes, alloc_type type)
            : size_(bytes), owns_resource_(true), is_managed_(false)
        {
            is_host_memory_ = (type == alloc_type::host);

            if (type == alloc_type::host) {
                // use pinned host memory for better transfer performance
                check_error<cuda_backend_t>(
                    cudaMallocHost(&ptr_, bytes),
                    "pinned host malloc"
                );
            }
            else {
                check_error<cuda_backend_t>(
                    cudaMalloc(&ptr_, bytes),
                    "device malloc"
                );
            }
        }

        device_memory_t(std::size_t bytes, bool managed, alloc_type type)
            : size_(bytes), owns_resource_(true), is_managed_(managed)
        {
            is_host_memory_ = (type == alloc_type::host);

            if (type == alloc_type::host) {
                if (managed) {
                    // managed memory is accessible from both host and device
                    check_error<cuda_backend_t>(
                        cudaMallocManaged(&ptr_, bytes),
                        "managed malloc"
                    );
                }
                else {
                    check_error<cuda_backend_t>(
                        cudaMallocHost(&ptr_, bytes),
                        "pinned host malloc"
                    );
                }
            }
            else {
                // Device allocation
                if (managed) {
                    check_error<cuda_backend_t>(
                        cudaMallocManaged(&ptr_, bytes),
                        "managed malloc"
                    );
                }
                else {
                    check_error<cuda_backend_t>(
                        cudaMalloc(&ptr_, bytes),
                        "device malloc"
                    );
                }
            }
        }

        ~device_memory_t() { destroy(); }
        device_memory_t(const device_memory_t&)            = delete;
        device_memory_t& operator=(const device_memory_t&) = delete;

        device_memory_t(device_memory_t&& other) noexcept
            : ptr_(other.ptr_),
              size_(other.size_),
              owns_resource_(other.owns_resource_),
              is_managed_(other.is_managed_),
              is_host_memory_(other.is_host_memory_)
        {
            other.ptr_           = nullptr;
            other.owns_resource_ = false;
        }

        device_memory_t& operator=(device_memory_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                ptr_            = other.ptr_;
                size_           = other.size_;
                owns_resource_  = other.owns_resource_;
                is_managed_     = other.is_managed_;
                is_host_memory_ = other.is_host_memory_;

                other.ptr_           = nullptr;
                other.owns_resource_ = false;
            }
            return *this;
        }

        void* data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return size_; }
        bool is_managed() const noexcept { return is_managed_; }
        bool is_host_memory() const noexcept { return is_host_memory_; }

        template <typename T>
        DUAL T* as() const noexcept
        {
            return static_cast<T*>(ptr_);
        }

      private:
        void destroy()
        {
            if (owns_resource_ && ptr_) {
                if (is_host_memory_ && !is_managed_) {
                    cudaFreeHost(ptr_);
                }
                else {
                    // works for both device and managed memory
                    cudaFree(ptr_);
                }
                ptr_ = nullptr;
            }
            owns_resource_ = false;
        }
    };

    template <>
    class device_adapter_t<cuda_backend_t>
    {
      public:
        using stream_type = stream_t<cuda_backend_t>;
        using event_type  = event_t<cuda_backend_t>;
        using memory_type = device_memory_t<cuda_backend_t>;

        template <typename T>
        using vector_type = device_vector_t<cuda_backend_t, T>;

        static void copy(
            void* dst,
            const void* src,
            std::size_t bytes,
            memory_direction_t kind
        )
        {
            cudaMemcpyKind cuda_kind;
            switch (kind) {
                case memory_direction_t::host_to_device:
                    cuda_kind = cudaMemcpyHostToDevice;
                    break;
                case memory_direction_t::device_to_host:
                    cuda_kind = cudaMemcpyDeviceToHost;
                    break;
                case memory_direction_t::device_to_device:
                    cuda_kind = cudaMemcpyDeviceToDevice;
                    break;
                case memory_direction_t::host_to_host:
                    cuda_kind = cudaMemcpyHostToHost;
                    break;
            }
            check_error<cuda_backend_t>(
                cudaMemcpy(dst, src, bytes, cuda_kind),
                "memory copy"
            );
        }

        static void copy_async(
            void* dst,
            const void* src,
            std::size_t bytes,
            memory_direction_t kind,
            const stream_type& stream
        )
        {
            cudaMemcpyKind cuda_kind;
            switch (kind) {
                case memory_direction_t::host_to_device:
                    cuda_kind = cudaMemcpyHostToDevice;
                    break;
                case memory_direction_t::device_to_host:
                    cuda_kind = cudaMemcpyDeviceToHost;
                    break;
                case memory_direction_t::device_to_device:
                    cuda_kind = cudaMemcpyDeviceToDevice;
                    break;
                case memory_direction_t::host_to_host:
                    cuda_kind = cudaMemcpyHostToHost;
                    break;
            }
            check_error<cuda_backend_t>(
                cudaMemcpyAsync(
                    dst,
                    src,
                    bytes,
                    cuda_kind,
                    stream.native_handle()
                ),
                "async memory copy"
            );
        }

        static void peer_copy_async(
            void* dst,
            int dst_device_id,
            const void* src,
            int src_device_id,
            std::size_t bytes,
            const stream_type& stream
        )
        {
            check_error<cuda_backend_t>(
                cudaMemcpyPeerAsync(
                    dst,
                    dst_device_id,
                    src,
                    src_device_id,
                    bytes,
                    stream.native_handle()
                ),
                "async peer memory copy"
            );
        }

        static void peer_copy(
            void* dst,
            int dst_device_id,
            const void* src,
            int src_device_id,
            std::size_t bytes
        )
        {
            check_error<cuda_backend_t>(
                cudaMemcpyPeer(dst, dst_device_id, src, src_device_id, bytes),
                "peer memory copy"
            );
        }

        static memory_type allocate(std::size_t bytes)
        {
            return memory_type(bytes);
        }

        template <typename T>
        static vector_type<T> allocate_vector(std::size_t count)
        {
            return vector_type<T>(count);
        }

        template <typename T>
        static vector_type<T> allocate_managed_vector(std::size_t count)
        {
            return vector_type<T>(count, true);
        }

        static void prefetch_to_device(
            const void* ptr,
            std::size_t bytes,
            std::int64_t device_id = 0
        )
        {
            int device       = static_cast<int>(device_id);
            auto cuda_device = cudaMemLocation{
              .type = cudaMemLocationTypeDevice,
              .id   = device
            };
            check_error<cuda_backend_t>(
                cudaGetDevice(&device),
                "get current device for prefetch"
            );
            check_error<cuda_backend_t>(
                cudaMemPrefetchAsync(ptr, bytes, cuda_device, 0),
                "prefetch to device"
            );
        }

        static stream_type create_stream() { return stream_type(); }

        static event_type create_event() { return event_type(); }

        static void synchronize_device()
        {
            check_error<cuda_backend_t>(
                cudaDeviceSynchronize(),
                "device synchronize"
            );
        }

        static std::int64_t get_device_count()
        {
            int count;
            check_error<cuda_backend_t>(
                cudaGetDeviceCount(&count),
                "get device count"
            );
            return static_cast<std::int64_t>(count);
        }

        static memory_type allocate_managed(std::size_t bytes)
        {
            void* ptr;
            check_error<cuda_backend_t>(
                cudaMallocManaged(&ptr, bytes),
                "managed memory allocation"
            );
            return memory_type(bytes, true);
        }

        static bool
        can_access_peer(std::int64_t device_id, std::int64_t peer_device_id)
        {
            int can_access;
            check_error<cuda_backend_t>(
                cudaDeviceCanAccessPeer(
                    &can_access,
                    static_cast<int>(device_id),
                    static_cast<int>(peer_device_id)
                ),
                "check peer access"
            );
            return can_access != 0;
        }

        static void enable_peer_access(std::int64_t peer_device_id)
        {
            check_error<cuda_backend_t>(
                cudaDeviceEnablePeerAccess(static_cast<int>(peer_device_id), 0),
                "enable peer access"
            );
        }

        static void set_device(std::int64_t device_id)
        {
            check_error<cuda_backend_t>(
                cudaSetDevice(static_cast<int>(device_id)),
                "set device"
            );
        }

        static std::int64_t get_current_device()
        {
            int device;
            check_error<cuda_backend_t>(
                cudaGetDevice(&device),
                "get current device"
            );
            return static_cast<std::int64_t>(device);
        }

        template <typename kernel_t, typename... args_t>
        static void launch_kernel(
            kernel_t kernel,
            dim3_t grid,
            dim3_t block,
            args_t... args
        )
        {
            dim3 cuda_grid(grid.x, grid.y, grid.z);
            dim3 cuda_block(block.x, block.y, block.z);
            generic_kernel<<<cuda_grid, cuda_block>>>(
                kernel,
                std::forward<args_t>(args)...
            );
            check_error<cuda_backend_t>(cudaGetLastError(), "kernel launch");
        }

        template <typename kernel_t, typename... args_t>
        static void launch_kernel_async(
            kernel_t kernel,
            dim3_t grid,
            dim3_t block,
            const stream_type& stream,
            args_t... args
        )
        {
            dim3 cuda_grid(grid.x, grid.y, grid.z);
            dim3 cuda_block(block.x, block.y, block.z);
            generic_kernel<<<
                cuda_grid,
                cuda_block,
                0,
                stream.native_handle()>>>(
                kernel,
                std::forward<args_t>(args)...
            );
            check_error<cuda_backend_t>(
                cudaGetLastError(),
                "async kernel launch"
            );
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
                std::forward<args_t>(args)...
            );
        }

        template <typename kernel_t, typename... args_t>
        static void launch_async(
            kernel_t kernel,
            grid::launch_config_t& launch_config,
            const stream_type& stream,
            args_t... args
        )
        {
            launch_kernel_async(
                kernel,
                launch_config.grid(),
                launch_config.block(),
                stream,
                std::forward<args_t>(args)...
            );
        }

        static void memset(void* ptr, std::int64_t value, std::size_t bytes)
        {
            check_error<cuda_backend_t>(
                cudaMemset(ptr, value, bytes),
                "memset"
            );
        }

        static void memset_async(
            void* ptr,
            std::int64_t value,
            std::size_t bytes,
            const stream_type& stream
        )
        {
            check_error<cuda_backend_t>(
                cudaMemsetAsync(ptr, value, bytes, stream.native_handle()),
                "async memset"
            );
        }

        template <typename T>
        static void
        copy_vector_to_host(T* host_ptr, const vector_type<T>& device_vec)
        {
            copy(
                host_ptr,
                device_vec.data(),
                device_vec.size_bytes(),
                memory_direction_t::device_to_host
            );
        }

        template <typename T>
        static void
        copy_vector_from_host(vector_type<T>& device_vec, const T* host_ptr)
        {
            copy(
                device_vec.data(),
                host_ptr,
                device_vec.size_bytes(),
                memory_direction_t::host_to_device
            );
        }

        template <typename T>
        static void copy_vector_to_host_async(
            T* host_ptr,
            const vector_type<T>& device_vec,
            const stream_type& stream
        )
        {
            copy_async(
                host_ptr,
                device_vec.data(),
                device_vec.size_bytes(),
                memory_direction_t::device_to_host,
                stream
            );
        }

        template <typename T>
        static void copy_vector_from_host_async(
            vector_type<T>& device_vec,
            const T* host_ptr,
            const stream_type& stream
        )
        {
            copy_async(
                device_vec.data(),
                host_ptr,
                device_vec.size_bytes(),
                memory_direction_t::host_to_device,
                stream
            );
        }

        static constexpr const char* backend_name() { return "cuda"; }

        static constexpr bool supports_async_operations() { return true; }

        static constexpr bool supports_peer_access() { return true; }

        static device_props<cuda_backend_t>
        get_device_properties(std::int64_t device_id)
        {
            cudaDeviceProp props;
            check_error<cuda_backend_t>(
                cudaGetDeviceProperties(&props, static_cast<int>(device_id)),
                "get device properties"
            );
            return props;
        }
    };

}   // namespace simbi::hetero

#endif   // CUDA_ENABLED
#endif   // HETERO_DETAIL_CUDA_IMPL_HPP
