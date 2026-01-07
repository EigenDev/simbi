// =============================================================================
// buffer_ops.hpp
//
// high-level buffer operations for shared_buffer_t implementation.
// provides copy, fill, transform, and staging operations across memory spaces.
// preserves hesi semantics while using clean xpu abstractions.
//
// design principles:
//   - memory space agnostic operations
//   - async operation support with tokens
//   - zero-overhead space dispatch
//   - hesi-compatible semantics
//
// usage:
//   auto token = copy_async(src_buffer, dst_buffer, executor);
//   fill_buffer(buffer, value);
//   auto staged = stage_buffer_to<device_memory>(buffer, executor);
// =============================================================================

#pragma once

#include "device_memory.hpp"
#include "execution_space.hpp"
#include "host_memory.hpp"
#include "memory_space.hpp"
#include "unified_memory.hpp"

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace simbi::xpu {

    // forward declarations
    template <typename T, memory_space MemorySpace>
    class shared_buffer_t;

    template <execution_space ExecutionSpace>
    class executor_t;

    template <execution_space ExecutionSpace>
    class token_t;

    namespace buffer_ops {

        // =============================================================================
        // synchronous buffer operations
        // =============================================================================

        template <typename T, memory_space SrcSpace, memory_space DstSpace>
        void copy_buffer(
            const shared_buffer_t<T, SrcSpace>& src,
            shared_buffer_t<T, DstSpace>&       dst,
            std::size_t                         count = 0
        )
        {
            const std::size_t copy_count = (count == 0) ? std::min(src.size(), dst.size()) : count;

            if (copy_count == 0) {
                return;
            }

            // space-specific copy dispatch
            if constexpr (std::is_same_v<SrcSpace, DstSpace>) {
                // same space: direct copy
                std::copy_n(src.data(), copy_count, dst.data());
            }
            else if constexpr (std::is_same_v<SrcSpace, host_memory> &&
                               std::is_same_v<DstSpace, device_memory>) {
                // host to device copy
                device_memory::memcpy_from_host(dst.data(), src.data(), copy_count * sizeof(T));
            }
            else if constexpr (std::is_same_v<SrcSpace, device_memory> &&
                               std::is_same_v<DstSpace, host_memory>) {
                // device to host copy
                device_memory::memcpy_to_host(dst.data(), src.data(), copy_count * sizeof(T));
            }
            else if constexpr (std::is_same_v<SrcSpace, unified_memory> ||
                               std::is_same_v<DstSpace, unified_memory>) {
                // unified memory: can use direct copy
                std::copy_n(src.data(), copy_count, dst.data());
            }
            else {
                // fallback: copy through host
                auto temp = shared_buffer_t<T, host_memory>(copy_count);
                copy_buffer(src, temp);
                copy_buffer(temp, dst);
            }
        }

        template <typename T, memory_space MemorySpace>
        void fill_buffer(shared_buffer_t<T, MemorySpace>& buffer, const T& value)
        {
            if constexpr (std::is_same_v<MemorySpace, host_memory> ||
                          std::is_same_v<MemorySpace, unified_memory>) {
                // host-accessible memory: direct fill
                std::fill_n(buffer.data(), buffer.size(), value);
            }
            else {
                // device memory: stage through host
                auto temp = shared_buffer_t<T, host_memory>(buffer.size());
                std::fill_n(temp.data(), temp.size(), value);
                copy_buffer(temp, buffer);
            }
        }

        template <typename T, memory_space MemorySpace>
        void zero_buffer(shared_buffer_t<T, MemorySpace>& buffer)
        {
            if constexpr (std::is_trivially_constructible_v<T>) {
                if constexpr (std::is_same_v<MemorySpace, host_memory> ||
                              std::is_same_v<MemorySpace, unified_memory>) {
                    std::memset(buffer.data(), 0, buffer.size() * sizeof(T));
                }
                else if constexpr (std::is_same_v<MemorySpace, device_memory>) {
                    // device memory: use device memset
                    device_memory::memset(buffer.data(), 0, buffer.size() * sizeof(T));
                }
                else {
                    fill_buffer(buffer, T{});
                }
            }
            else {
                fill_buffer(buffer, T{});
            }
        }

        // =============================================================================
        // asynchronous buffer operations
        // =============================================================================

        template <
            execution_space ExecutionSpace,
            typename T,
            memory_space SrcSpace,
            memory_space DstSpace>
        token_t<ExecutionSpace> copy_async(
            const shared_buffer_t<T, SrcSpace>& src,
            shared_buffer_t<T, DstSpace>&       dst,
            executor_t<ExecutionSpace>&         exec,
            std::size_t                         count = 0
        )
        {
            return exec.submit([src, dst, count]() mutable { copy_buffer(src, dst, count); });
        }

        template <execution_space ExecutionSpace, typename T, memory_space MemorySpace>
        token_t<ExecutionSpace> fill_async(
            shared_buffer_t<T, MemorySpace>& buffer,
            const T&                         value,
            executor_t<ExecutionSpace>&      exec
        )
        {
            return exec.submit([buffer, value]() mutable { fill_buffer(buffer, value); });
        }

        template <execution_space ExecutionSpace, typename T, memory_space MemorySpace>
        token_t<ExecutionSpace>
        zero_async(shared_buffer_t<T, MemorySpace>& buffer, executor_t<ExecutionSpace>& exec)
        {
            return exec.submit([buffer]() mutable { zero_buffer(buffer); });
        }

        // =============================================================================
        // buffer transformation operations
        // =============================================================================

        template <typename T, memory_space MemorySpace, typename Func>
        void transform_buffer(shared_buffer_t<T, MemorySpace>& buffer, Func&& func)
        {
            if constexpr (std::is_same_v<MemorySpace, host_memory> ||
                          std::is_same_v<MemorySpace, unified_memory>) {
                // host-accessible: direct transform
                std::transform(
                    buffer.data(),
                    buffer.data() + buffer.size(),
                    buffer.data(),
                    std::forward<Func>(func)
                );
            }
            else {
                // device memory: stage through host
                auto temp = shared_buffer_t<T, host_memory>(buffer.size());
                copy_buffer(buffer, temp);
                std::transform(
                    temp.data(),
                    temp.data() + temp.size(),
                    temp.data(),
                    std::forward<Func>(func)
                );
                copy_buffer(temp, buffer);
            }
        }

        template <
            execution_space ExecutionSpace,
            typename T,
            memory_space MemorySpace,
            typename Func>
        token_t<ExecutionSpace> transform_async(
            shared_buffer_t<T, MemorySpace>& buffer,
            Func&&                           func,
            executor_t<ExecutionSpace>&      exec
        )
        {
            return exec.submit([buffer, f = std::forward<Func>(func)]() mutable {
                transform_buffer(buffer, std::move(f));
            });
        }

        // =============================================================================
        // buffer staging operations (cross-space transfers)
        // =============================================================================

        template <
            memory_space    DstSpace,
            execution_space ExecutionSpace,
            typename T,
            memory_space SrcSpace>
        auto
        stage_buffer_to(const shared_buffer_t<T, SrcSpace>& src, executor_t<ExecutionSpace>& exec)
            -> std::pair<shared_buffer_t<T, DstSpace>, token_t<ExecutionSpace>>
        {
            auto dst_buffer = shared_buffer_t<T, DstSpace>(src.size());
            auto token      = copy_async(src, dst_buffer, exec);
            return std::make_pair(std::move(dst_buffer), std::move(token));
        }

        template <memory_space DstSpace, typename T, memory_space SrcSpace>
        shared_buffer_t<T, DstSpace> stage_buffer_to_sync(const shared_buffer_t<T, SrcSpace>& src)
        {
            auto dst_buffer = shared_buffer_t<T, DstSpace>(src.size());
            copy_buffer(src, dst_buffer);
            return dst_buffer;
        }

        // =============================================================================
        // buffer reduction operations
        // =============================================================================

        template <typename T, memory_space MemorySpace, typename BinaryOp>
        T reduce_buffer(const shared_buffer_t<T, MemorySpace>& buffer, T init_value, BinaryOp&& op)
        {
            if (buffer.empty()) {
                return init_value;
            }

            if constexpr (std::is_same_v<MemorySpace, host_memory> ||
                          std::is_same_v<MemorySpace, unified_memory>) {
                // host-accessible: direct reduction
                return std::reduce(
                    buffer.data(),
                    buffer.data() + buffer.size(),
                    init_value,
                    std::forward<BinaryOp>(op)
                );
            }
            else {
                // device memory: stage to host first
                auto temp = stage_buffer_to_sync<host_memory>(buffer);
                return std::reduce(
                    temp.data(),
                    temp.data() + temp.size(),
                    init_value,
                    std::forward<BinaryOp>(op)
                );
            }
        }

        template <typename T, memory_space MemorySpace>
        T sum_buffer(const shared_buffer_t<T, MemorySpace>& buffer)
        {
            return reduce_buffer(buffer, T{}, std::plus<T>{});
        }

        template <typename T, memory_space MemorySpace>
        T max_buffer(const shared_buffer_t<T, MemorySpace>& buffer)
        {
            if (buffer.empty()) {
                return T{};
            }
            return reduce_buffer(buffer, *buffer.data(), [](const T& a, const T& b) {
                return std::max(a, b);
            });
        }

        template <typename T, memory_space MemorySpace>
        T min_buffer(const shared_buffer_t<T, MemorySpace>& buffer)
        {
            if (buffer.empty()) {
                return T{};
            }
            return reduce_buffer(buffer, *buffer.data(), [](const T& a, const T& b) {
                return std::min(a, b);
            });
        }

        // =============================================================================
        // async reduction operations
        // =============================================================================

        template <
            execution_space ExecutionSpace,
            typename T,
            memory_space MemorySpace,
            typename BinaryOp>
        auto reduce_async(
            const shared_buffer_t<T, MemorySpace>& buffer,
            T                                      init_value,
            BinaryOp&&                             op,
            executor_t<ExecutionSpace>&            exec
        ) -> std::pair<std::shared_ptr<T>, token_t<ExecutionSpace>>
        {
            auto result = std::make_shared<T>();
            auto token  = exec.submit(
                [buffer, init_value, result, op = std::forward<BinaryOp>(op)]() mutable {
                    *result = reduce_buffer(buffer, init_value, std::move(op));
                }
            );
            return std::make_pair(result, std::move(token));
        }

        // =============================================================================
        // buffer validation and comparison
        // =============================================================================

        template <typename T, memory_space SrcSpace, memory_space DstSpace>
        bool buffers_equal(
            const shared_buffer_t<T, SrcSpace>& lhs,
            const shared_buffer_t<T, DstSpace>& rhs,
            std::size_t                         count = 0
        )
        {
            const std::size_t compare_count =
                (count == 0) ? std::min(lhs.size(), rhs.size()) : count;

            if (compare_count == 0) {
                return lhs.size() == rhs.size();
            }

            // stage both to host for comparison if needed
            auto lhs_host = [&]() {
                if constexpr (std::is_same_v<SrcSpace, host_memory> ||
                              std::is_same_v<SrcSpace, unified_memory>) {
                    return lhs; // already host-accessible
                }
                else {
                    return stage_buffer_to_sync<host_memory>(lhs);
                }
            }();

            auto rhs_host = [&]() {
                if constexpr (std::is_same_v<DstSpace, host_memory> ||
                              std::is_same_v<DstSpace, unified_memory>) {
                    return rhs; // already host-accessible
                }
                else {
                    return stage_buffer_to_sync<host_memory>(rhs);
                }
            }();

            return std::equal(lhs_host.data(), lhs_host.data() + compare_count, rhs_host.data());
        }

        // =============================================================================
        // buffer statistics and information
        // =============================================================================

        template <typename T, memory_space MemorySpace>
        struct buffer_stats_t
        {
            std::size_t element_count;
            std::size_t byte_size;
            T           min_value;
            T           max_value;
            T           sum_value;
            double      mean_value;
        };

        template <typename T, memory_space MemorySpace>
        buffer_stats_t<T, MemorySpace>
        compute_buffer_stats(const shared_buffer_t<T, MemorySpace>& buffer)
        {
            if (buffer.empty()) {
                return {0, 0, T{}, T{}, T{}, 0.0};
            }

            buffer_stats_t<T, MemorySpace> stats;
            stats.element_count = buffer.size();
            stats.byte_size     = buffer.size() * sizeof(T);
            stats.min_value     = min_buffer(buffer);
            stats.max_value     = max_buffer(buffer);
            stats.sum_value     = sum_buffer(buffer);
            stats.mean_value =
                static_cast<double>(stats.sum_value) / static_cast<double>(stats.element_count);

            return stats;
        }

    } // namespace buffer_ops

    // =============================================================================
    // convenience free functions
    // =============================================================================

    using buffer_ops::buffers_equal;
    using buffer_ops::compute_buffer_stats;
    using buffer_ops::copy_async;
    using buffer_ops::copy_buffer;
    using buffer_ops::fill_async;
    using buffer_ops::fill_buffer;
    using buffer_ops::max_buffer;
    using buffer_ops::min_buffer;
    using buffer_ops::reduce_buffer;
    using buffer_ops::stage_buffer_to;
    using buffer_ops::stage_buffer_to_sync;
    using buffer_ops::sum_buffer;
    using buffer_ops::transform_async;
    using buffer_ops::transform_buffer;
    using buffer_ops::zero_async;
    using buffer_ops::zero_buffer;

} // namespace simbi::xpu
