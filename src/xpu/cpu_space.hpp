// =============================================================================
// cpu_space.hpp
//
// cpu execution space implementation using openmp for parallel execution.
// provides compile-time dispatch for cpu-based parallel algorithms.
// implements execution_space concept with openmp backend.
//
// usage:
//   parallel_for<cpu_space>(range, kernel);
//   executor_t<cpu_space> exec;
//   auto token = exec.submit([]() { /* work */ });
// =============================================================================

#pragma once

#include "execution_space.hpp"
#include "vendors/cpu/cpu_device.hpp"

#include <atomic>
#include <cstring>
#include <future>
#include <omp.h>
#include <string_view>
#include <thread>

namespace xpu {

    // =============================================================================
    // cpu execution space
    // =============================================================================

    struct cpu_space
    {
        // device type for concept requirements
        using device_type = vendors::cpu::cpu_device_t;

        // memory space type for concept requirements
        using memory_space_type = void; // placeholder

        // stream and event handle types - use vendor abstractions
        using stream_handle_type = vendors::cpu::cpu_stream_handle_t;
        using event_handle_type  = vendors::cpu::cpu_event_handle_t;

        // =============================================================================
        // compile-time properties
        // =============================================================================

        static constexpr std::string_view space_name()
        {
            return "cpu";
        }

        static constexpr std::string_view vendor_name()
        {
            return "host";
        }

        static constexpr int default_device_id()
        {
            return 0;
        }

        // execution_space concept requirements
        static constexpr bool is_host_space    = true;
        static constexpr bool is_device_space  = false;
        static constexpr bool supports_async   = true;
        static constexpr bool supports_kernels = false;

        // legacy aliases (required by tests)
        static constexpr bool is_gpu  = false;
        static constexpr bool is_host = true;

        // test compatibility functions
        static constexpr std::string_view name()
        {
            return space_name();
        }

        // =============================================================================
        // execution context
        // =============================================================================

        struct execution_context
        {
            stream_handle_type stream_id;
            int                thread_count;

            execution_context()
                : stream_id(std::this_thread::get_id()), thread_count(omp_get_max_threads())
            {
            }

            execution_context(int threads)
                : stream_id(std::this_thread::get_id()), thread_count(threads)
            {
            }
        };

        // =============================================================================
        // parallel execution primitives
        // =============================================================================

        template <typename Index, typename Functor>
        static void parallel_for(Index first, Index last, Functor&& func)
        {
#pragma omp parallel for
            for (Index ii = first; ii < last; ++ii) {
                func(ii);
            }
        }

        template <typename Index, typename Functor>
        static void
        parallel_for(Index first, Index last, Functor&& func, const execution_context& ctx)
        {
            const int old_threads = omp_get_max_threads();
            omp_set_num_threads(ctx.thread_count);

#pragma omp parallel for
            for (Index ii = first; ii < last; ++ii) {
                func(ii);
            }

            omp_set_num_threads(old_threads);
        }

        template <typename Index, typename Functor, typename T>
        static T reduce(Index first, Index last, T init, Functor&& func)
        {
            T result = init;
#pragma omp parallel for reduction(+ : result)
            for (Index ii = first; ii < last; ++ii) {
                result += func(ii);
            }
            return result;
        }

        template <typename Index, typename Functor, typename T>
        static T
        reduce(Index first, Index last, T init, Functor&& func, const execution_context& ctx)
        {
            const int old_threads = omp_get_max_threads();
            omp_set_num_threads(ctx.thread_count);

            T result = init;
#pragma omp parallel for reduction(+ : result)
            for (Index ii = first; ii < last; ++ii) {
                result += func(ii);
            }

            omp_set_num_threads(old_threads);
            return result;
        }

        // =============================================================================
        // stream management
        // =============================================================================

        static stream_handle_type create_stream()
        {
            return stream_handle_type{std::this_thread::get_id()};
        }

        static void destroy_stream(stream_handle_type /* stream */)
        {
            // no-op for cpu
        }

        static void synchronize_stream(stream_handle_type /* stream */)
        {
            // no-op for cpu - all work is synchronous within thread
        }

        // =============================================================================
        // event management
        // =============================================================================

        static event_handle_type create_event()
        {
            std::promise<void> promise;
            auto               future = promise.get_future();
            promise.set_value(); // immediately ready
            return event_handle_type{std::move(future)};
        }

        static void wait_for_event(const event_handle_type& event)
        {
            // delegate to vendor device implementation
            device_type device;
            device.synchronize_event(event);
        }

        static void stream_wait_event(stream_handle_type stream, const event_handle_type& event)
        {
            // delegate to vendor device implementation
            device_type device;
            device.stream_wait_event(stream, event);
        }

        static event_handle_type record_event(stream_handle_type /* stream */)
        {
            return create_event(); // immediately ready
        }

        // =============================================================================
        // execution characteristics (required by execution_space concept)
        // =============================================================================

        static std::size_t max_concurrency()
        {
            return static_cast<std::size_t>(omp_get_max_threads());
        }

        static constexpr std::size_t preferred_block_size()
        {
            return 256; // not applicable for cpu but required by concept
        }

        static constexpr double memory_bandwidth_gb_per_sec()
        {
            // approximate ddr4 bandwidth: ~25-50 GB/s, use conservative estimate
            return 30.0;
        }
    };

    // note: static_assert(execution_space<cpu_space>) moved to xpu.hpp
    // cannot verify here due to incomplete types (executor_t, token_t, host_memory)

} // namespace xpu
