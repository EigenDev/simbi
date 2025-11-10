#ifndef HETERO_DETAIL_ADAPTER_IMPL_HPP
#define HETERO_DETAIL_ADAPTER_IMPL_HPP

#include "../core/common_types.hpp"
#include "../core/resource_types.hpp"
#include "hetero/core/backend_traits.hpp"
#include "hetero/device/execution_context.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::hetero {

    template <typename backend_t>
    class device_adapter_t;

    namespace detail {
        template <typename backend_t>
        struct adapter_traits_t {
            using stream_type = stream_t<backend_t>;
            using event_type  = event_t<backend_t>;
            using memory_type = device_memory_t<backend_t>;

            template <typename T>
            using vector_type = device_vector_t<backend_t, T>;
        };
    }   // namespace detail

    template <typename backend_t>
    class device_adapter_t
    {
      public:
        using stream_type =
            typename detail::adapter_traits_t<backend_t>::stream_type;
        using event_type =
            typename detail::adapter_traits_t<backend_t>::event_type;
        using memory_type =
            typename detail::adapter_traits_t<backend_t>::memory_type;

        template <typename T>
        using vector_type = typename detail::adapter_traits_t<
            backend_t>::template vector_type<T>;

        static void copy(
            void* dst,
            const void* src,
            std::size_t bytes,
            memory_direction_t kind
        );

        static void copy_async(
            void* dst,
            const void* src,
            std::size_t bytes,
            memory_direction_t kind,
            const stream_type& stream
        );

        static void peer_copy_async(
            void* dst,
            int dst_device_id,
            const void* src,
            int src_device_id,
            std::size_t bytes,
            const stream_type& stream
        );

        static void peer_copy(
            void* dst,
            int dst_device_id,
            const void* src,
            int src_device_id,
            std::size_t bytes
        );

        static memory_type allocate(std::size_t bytes);

        template <typename T>
        static vector_type<T> allocate_vector(std::size_t count);

        template <typename T>
        static vector_type<T> allocate_managed_vector(std::size_t count);

        static void
        prefetch_to_device(const void* ptr, std::size_t bytes, int device_id);

        static stream_type create_stream();

        static event_type create_event();

        static void synchronize_device();

        static std::int64_t get_device_count();

        static memory_type allocate_managed(std::size_t bytes);
        static bool
        can_access_peer(std::int64_t device_id, std::int64_t peer_device_id);
        static void enable_peer_access(std::int64_t peer_device_id);

        static void set_device(std::int64_t device_id);

        static std::int64_t get_current_device();

        template <typename kernel_t, typename... args_t>
        static void launch_kernel(
            kernel_t kernel,
            dim3_t grid,
            dim3_t block,
            args_t... args
        );

        template <typename kernel_t, typename... args_t>
        static void launch_kernel_async(
            kernel_t kernel,
            dim3_t grid,
            dim3_t block,
            const stream_type& stream,
            args_t... args
        );

        template <typename kernel_t>
        static void
        launch(kernel_t kernel, grid::launch_config_t& launch_config);

        template <typename kernel_t>
        static void launch_async(
            kernel_t kernel,
            grid::launch_config_t& launch_config,
            const stream_type& stream
        );

        static void memset(void* ptr, int value, std::size_t bytes);

        static void memset_async(
            void* ptr,
            std::int64_t value,
            std::size_t bytes,
            const stream_type& stream
        );

        template <typename T>
        static void
        copy_vector_to_host(T* host_ptr, const vector_type<T>& device_vec);

        template <typename T>
        static void
        copy_vector_from_host(vector_type<T>& device_vec, const T* host_ptr);

        template <typename T>
        static void copy_vector_to_host_async(
            T* host_ptr,
            const vector_type<T>& device_vec,
            const stream_type& stream
        );

        template <typename T>
        static void copy_vector_from_host_async(
            vector_type<T>& device_vec,
            const T* host_ptr,
            const stream_type& stream
        );

        static constexpr const char* backend_name()
        {
            return backend_info_t<backend_t>::name;
        }

        static constexpr bool supports_async_operations()
        {
            return backend_info_t<backend_t>::supports_async;
        }

        static constexpr bool supports_peer_access()
        {
            return backend_info_t<backend_t>::supports_peer_access;
        }

        static device_props<backend_t>
        get_device_properties(std::int64_t device_id);
    };

}   // namespace simbi::hetero

#endif   // HETERO_DETAIL_ADAPTER_IMPL_HPP
