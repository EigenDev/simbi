#include "device_adapter_api.hpp"
#include "config.hpp"
#include "device_backend.hpp"
#include "device_types.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>

namespace simbi {
    namespace gpu {
        // main API that delegates to the appropriate backend
        namespace api {
            // memory operations
            void copy_host_to_device(void* to, const void* from, size_t bytes)
            {
                adapter::get_device_backend()
                    .copy_host_to_device(to, from, bytes);
            }

            void copy_device_to_host(void* to, const void* from, size_t bytes)
            {
                adapter::get_device_backend()
                    .copy_device_to_host(to, from, bytes);
            }

            void copy_device_to_device(void* to, const void* from, size_t bytes)
            {
                adapter::get_device_backend()
                    .copy_device_to_device(to, from, bytes);
            }

            void malloc(void** obj, size_t bytes)
            {
                adapter::get_device_backend().malloc(obj, bytes);
            }

            void malloc_managed(void** obj, size_t bytes)
            {
                adapter::get_device_backend().malloc_managed(obj, bytes);
            }

            void free(void* obj) { adapter::get_device_backend().free(obj); }

            void memset(void* obj, std::int64_t val, size_t bytes)
            {
                adapter::get_device_backend().memset(obj, val, bytes);
            }

            // event handling
            void event_create(adapter::event_t<>* event)
            {
                adapter::get_device_backend().event_create(event);
            }

            void event_destroy(adapter::event_t<> event)
            {
                adapter::get_device_backend().event_destroy(event);
            }

            void
            event_record(adapter::event_t<> event, adapter::stream_t<> stream)
            {
                adapter::get_device_backend().event_record(event, stream);
            }

            void event_synchronize(adapter::event_t<> event)
            {
                adapter::get_device_backend().event_synchronize(event);
            }

            void event_elapsed_time(
                float* time,
                adapter::event_t<> start,
                adapter::event_t<> end
            )
            {
                adapter::get_device_backend()
                    .event_elapsed_time(time, start, end);
            }

            // device management
            void get_device_count(std::int64_t* count)
            {
                adapter::get_device_backend().get_device_count(count);
            }

            void get_device_properties(
                adapter::device_properties_t<>* props,
                std::int64_t device
            )
            {
                adapter::get_device_backend().get_device_properties(
                    props,
                    device
                );
            }

            void set_device(std::int64_t device)
            {
                adapter::get_device_backend().set_device(device);
            }

            void device_synch()
            {
                adapter::get_device_backend().device_synchronize();
            }

            // stream operations
            void stream_create(adapter::stream_t<>* stream)
            {
                adapter::get_device_backend().stream_create(stream);
            }

            void stream_destroy(adapter::stream_t<> stream)
            {
                adapter::get_device_backend().stream_destroy(stream);
            }

            void stream_synchronize(adapter::stream_t<> stream)
            {
                adapter::get_device_backend().stream_synchronize(stream);
            }

            void stream_wait_event(
                adapter::stream_t<> stream,
                adapter::event_t<> event,
                std::uint64_t flags
            )
            {
                adapter::get_device_backend()
                    .stream_wait_event(stream, event, flags);
            }

            void stream_query(adapter::stream_t<> stream, int* status)
            {
                adapter::get_device_backend().stream_query(stream, status);
            }

            // asynchronous operations
            void async_copy_host_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .async_copy_host_to_device(dst, src, bytes, stream);
            }

            void async_copy_device_to_host(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .async_copy_device_to_host(dst, src, bytes, stream);
            }

            void async_copy_device_to_device(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .async_copy_device_to_device(dst, src, bytes, stream);
            }

            void memcpy_async(
                void* dst,
                const void* src,
                size_t bytes,
                adapter::memcpy_kind_t<> kind,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend()
                    .memcpy_async(dst, src, bytes, kind, stream);
            }

            // peer operations
            void enable_peer_access(std::int64_t device, std::uint64_t flags)
            {
                adapter::get_device_backend().enable_peer_access(device, flags);
            }

            void peer_copy_async(
                void* dst,
                std::int64_t dst_device,
                const void* src,
                std::int64_t src_device,
                size_t bytes,
                adapter::stream_t<> stream
            )
            {
                adapter::get_device_backend().peer_copy_async(
                    dst,
                    dst_device,
                    src,
                    src_device,
                    bytes,
                    stream
                );
            }

            // host memory management
            void host_register(void* ptr, size_t size, std::uint64_t flags)
            {
                adapter::get_device_backend().host_register(ptr, size, flags);
            }

            void host_unregister(void* ptr)
            {
                adapter::get_device_backend().host_unregister(ptr);
            }

            void aligned_malloc(void** ptr, size_t size)
            {
                adapter::get_device_backend().aligned_malloc(ptr, size);
            }

            // specialized operations
            void memcpy_from_symbol(void* dst, const void* symbol, size_t count)
            {
                adapter::get_device_backend()
                    .memcpy_from_symbol(dst, symbol, count);
            }

            void prefetch_to_device(
                const void* obj,
                size_t bytes,
                std::int64_t device
            )
            {
                adapter::get_device_backend()
                    .prefetch_to_device(obj, bytes, device);
            }

            void launch_kernel(
                adapter::function_t<> function,
                adapter::types::dim3 grid,
                adapter::types::dim3 block,
                void** args,
                size_t shared_mem,
                adapter::stream_t<> stream
            )
            {
                adapter::types::dim3 adapter_grid(grid.x, grid.y, grid.z);
                adapter::types::dim3 adapter_block(block.x, block.y, block.z);

                adapter::get_device_backend().launch_kernel(
                    function,
                    adapter_grid,
                    adapter_block,
                    args,
                    shared_mem,
                    stream
                );
            }

            // thread synchronization
            DEV void synchronize()
            {
                adapter::get_device_backend().synchronize_threads();
            }

        }   // namespace api
    }   // namespace gpu

    namespace grid {
        // return current execution index
        DUAL adapter::grid_index idx()
        {
            return adapter::grid_index::current<>();
        }

        // create configuration from block/thread counts
        adapter::grid::launch_config
        config(std::uint64_t blocks, std::uint64_t threads)
        {
            return adapter::grid::launch_config(
                adapter::types::dim3(blocks),
                adapter::types::dim3(threads)
            );
        }

        // create configuration from dimensions
        adapter::grid::launch_config config(
            adapter::types::dim3 grid,
            adapter::types::dim3 block,
            size_t shared_memory
        )
        {
            return adapter::grid::launch_config(grid, block, shared_memory);
        }

        // calculate grid size from total elements
        adapter::types::dim3
        calculate_grid(size_t elements, adapter::types::dim3 threads_per_block)
        {
            return adapter::grid::calculate_grid(elements, threads_per_block);
        }
    }   // namespace grid

    // Thread and block utilities
    DEV auto global_thread_idx()
    {
        return adapter::grid_index::current<>().global_thread_id();
    }

    DEV auto global_thread_count()
    {
        return adapter::grid_index::current<>().total_threads();
    }

    DEV auto get_thread_id()
    {
#if GPU_ENABLED
        return adapter::grid_index::current<>().thread_id();
#else
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    }

    DEV auto get_block_id()
    {
        return adapter::grid_index::current<>().block_id();
    }

    DEV auto get_threads_per_block()
    {
        return adapter::grid_index::current<>().threads_per_block();
    }
}   // namespace simbi
