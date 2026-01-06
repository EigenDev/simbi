// =============================================================================
// executor.hpp
//
// phase 2/3 spec-compliant executor implementation for xpu framework.
// provides type-safe executor with execution space parameter, preserving
// hesi async semantics while using clean xpu abstractions.
//
// design principles:
//   - template on execution space for compile-time dispatch
//   - direct stream management (no arena internally)
//   - hesi-compatible api with submit/then/sync semantics
//   - phase 3: domain-based dispatch for parallel algorithms
//   - raii stream lifetime
//
// usage:
//   executor_t<cuda_space> exec(device_id);
//   auto token = exec.submit(kernel, args...);
//   auto chained = exec.then(token, next_kernel);
//
//   // phase 3: domain dispatch
//   auto domain = xpu::extents<3>({100, 200, 50});
//   exec.dispatch(domain, [=](auto idx) { /* work */ });
//   exec.sync();
// =============================================================================

#pragma once

#include "detail/cpu_dispatch.hpp"
#include "detail/event_wrapper.hpp"
#include "detail/stream_wrapper.hpp"
#include "domain.hpp"
#include "execution_space.hpp"

#ifdef XPU_CUDA_AVAILABLE
#include "detail/cuda_dispatch.hpp"
#endif

#include <functional>
#include <type_traits>
#include <utility>

namespace xpu {

    // =============================================================================
    // reduction operators (global scope for cuda compatibility)
    // =============================================================================

    template <typename T>
    struct max_op_t
    {
        __host__ __device__ constexpr T operator()(const T& a, const T& b) const
        {
            return a > b ? a : b;
        }
    };

    template <typename T>
    struct min_op_t
    {
        __host__ __device__ constexpr T operator()(const T& a, const T& b) const
        {
            return a < b ? a : b;
        }
    };

    // forward declaration
    template <execution_space ExecutionSpace>
    class token_t;

    // =============================================================================
    // executor implementation
    // =============================================================================

    template <execution_space ExecutionSpace>
    class executor_t
    {
      public:
        using execution_space_type = ExecutionSpace;
        using stream_handle_type   = typename ExecutionSpace::stream_handle_type;
        using token_type           = token_t<ExecutionSpace>;

      private:
        detail::stream_wrapper_t<ExecutionSpace> stream_;
        int                                      device_id_;

      public:
        // =============================================================================
        // construction and destruction
        // =============================================================================

        explicit executor_t(int device_id = 0) : stream_(device_id), device_id_(device_id) {}

        ~executor_t()
        {
            // raii: stream wrapper handles cleanup
        }

        // move-only (preserves hesi semantics)
        executor_t(executor_t&& other) noexcept
            : stream_(std::move(other.stream_)), device_id_(other.device_id_)
        {
        }

        executor_t& operator=(executor_t&& other) noexcept
        {
            if (this != &other) {
                stream_    = std::move(other.stream_);
                device_id_ = other.device_id_;
            }
            return *this;
        }

        // no copy (hesi semantics)
        executor_t(const executor_t&)            = delete;
        executor_t& operator=(const executor_t&) = delete;

        // =============================================================================
        // async execution api (preserving hesi semantics)
        // =============================================================================

        template <typename Kernel, typename... Args>
        token_type submit(Kernel&& kernel, Args&&... args)
        {
            // create token for this operation
            auto token = token_type::create();

            // execute kernel based on execution space
            if constexpr (std::is_same_v<ExecutionSpace, cpu_space>) {
                // cpu execution: run on thread pool or directly
                execute_cpu_kernel(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
                // cpu operations complete immediately, mark token ready
                token.mark_ready();
            }
            else if constexpr (std::is_same_v<ExecutionSpace, cuda_space>) {
                // cuda execution: launch on stream
                execute_cuda_kernel(
                    std::forward<Kernel>(kernel),
                    stream_.native_handle(),
                    std::forward<Args>(args)...
                );
                // record event on stream for async tracking
                token.record(*this);
            }
            else {
                // generic execution space
                execute_generic_kernel(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
                token.mark_ready();
            }

            return token;
        }

        template <typename Kernel, typename... Args>
        token_type then(const token_type& dependency, Kernel&& kernel, Args&&... args)
        {
            // wait for dependency on this stream
            dependency.wait_on(*this);

            // submit dependent work
            return submit(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
        }

        // =============================================================================
        // phase 3: domain-based dispatch
        // =============================================================================

        // dispatch work over n-dimensional domain
        // returns token for async composition
        template <std::uint64_t Rank, typename Func>
        token_type dispatch(const domain_t<Rank>& domain, Func&& kernel)
        {
            if (domain.empty()) {
                // empty domain: return ready token
                auto token = token_type::create();
                token.mark_ready();
                return token;
            }

            // forward to space-specific dispatch implementation
            return dispatch_impl(domain, std::forward<Func>(kernel));
        }

        // =============================================================================
        // phase 3: domain-based reduction (map-reduce pattern)
        // =============================================================================

        // reduce over domain with map-reduce pattern
        // map_func: domain index -> value
        // reduce_op: binary operator to combine values
        template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
        T
        reduce(const domain_t<Rank>& domain, T init_value, MapFunc&& map_func, ReduceOp&& reduce_op)
        {
            if (domain.empty()) {
                return init_value;
            }

            // forward to space-specific reduce implementation
            return reduce_impl(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op)
            );
        }

        // convenience: sum reduction
        template <std::uint64_t Rank, typename T, typename MapFunc>
        T sum(const domain_t<Rank>& domain, MapFunc&& map_func)
        {
            return reduce(domain, T{}, std::forward<MapFunc>(map_func), std::plus<T>{});
        }

        // convenience: max reduction
        template <std::uint64_t Rank, typename T, typename MapFunc>
        T max(const domain_t<Rank>& domain, T init_value, MapFunc&& map_func)
        {
            return reduce(domain, init_value, std::forward<MapFunc>(map_func), max_op_t<T>{});
        }

        // convenience: min reduction
        template <std::uint64_t Rank, typename T, typename MapFunc>
        T min(const domain_t<Rank>& domain, T init_value, MapFunc&& map_func)
        {
            return reduce(domain, init_value, std::forward<MapFunc>(map_func), min_op_t<T>{});
        }

        // =============================================================================
        // synchronization
        // =============================================================================

        void sync()
        {
            stream_.sync();
        }

        bool ready() const
        {
            return stream_.ready();
        }

        void wait()
        {
            sync();
        }

        // =============================================================================
        // resource access
        // =============================================================================

        stream_handle_type stream() const
        {
            return stream_.native_handle();
        }

        constexpr execution_space_type execution_space() const
        {
            return {};
        }

        int device_id() const noexcept
        {
            return device_id_;
        }

        // =============================================================================
        // space-specific accessors
        // =============================================================================

        // cuda-specific stream access
        template <typename Space = ExecutionSpace>
        auto cuda_stream() const noexcept
            -> std::enable_if_t<std::is_same_v<Space, cuda_space>, cudaStream_t>
        {
            return stream_.template cuda_stream<Space>();
        }

        // cpu-specific thread access
        template <typename Space = ExecutionSpace>
        auto thread_id() const noexcept
            -> std::enable_if_t<std::is_same_v<Space, cpu_space>, std::thread::id>
        {
            return stream_.template thread_id<Space>();
        }

      private:
        // =============================================================================
        // space-specific kernel execution
        // =============================================================================

        template <typename Kernel, typename... Args>
        void execute_cpu_kernel(Kernel&& kernel, Args&&... args)
        {
            if constexpr (std::is_invocable_v<Kernel, Args...>) {
                // direct function call
                std::invoke(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
            }
            else {
                // assume parallel algorithm
                static_assert(
                    std::is_invocable_v<Kernel, Args...>,
                    "kernel must be invocable with provided arguments"
                );
            }
        }

        template <typename Kernel, typename... Args>
        void execute_cuda_kernel(Kernel&& kernel, stream_handle_type stream, Args&&... args)
        {
            (void) stream; // unused in placeholder implementation
            // placeholder for actual cuda kernel launch
            // this would require nvcc compilation and proper kernel dispatch
            // for now, execute as host function
            if constexpr (std::is_invocable_v<Kernel, Args...>) {
                std::invoke(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
            }
            else {
                // cuda kernel launch would go here
                // launch_cuda_kernel<<<grid, block, shared, stream>>>(kernel, args...);
                static_assert(
                    std::is_invocable_v<Kernel, Args...>,
                    "cuda kernel launch not yet implemented - use host functions"
                );
            }
        }

        template <typename Kernel, typename... Args>
        void execute_generic_kernel(Kernel&& kernel, Args&&... args)
        {
            // fallback: direct execution
            if constexpr (std::is_invocable_v<Kernel, Args...>) {
                std::invoke(std::forward<Kernel>(kernel), std::forward<Args>(args)...);
            }
        }

        // =============================================================================
        // phase 3: domain dispatch implementation
        // =============================================================================

        // cpu dispatch - iterate domain with openmp
        template <std::uint64_t Rank, typename Func>
        token_type dispatch_impl(const domain_t<Rank>& domain, Func&& kernel)
            requires(std::is_same_v<ExecutionSpace, cpu_space>)
        {
            // dispatch to cpu parallel implementation
            cpu_dispatch(domain, std::forward<Func>(kernel));

            // cpu operations complete synchronously
            auto token = token_type::create();
            token.mark_ready();
            return token;
        }

        // cuda dispatch - launch grid-stride kernel
        template <std::uint64_t Rank, typename Func>
        token_type dispatch_impl(const domain_t<Rank>& domain, Func&& kernel)
            requires(std::is_same_v<ExecutionSpace, cuda_space>)
        {
            // dispatch to cuda kernel implementation
            auto token = token_type::create();
            cuda_dispatch(domain, std::forward<Func>(kernel), stream_.native_handle());
            token.record(*this);
            return token;
        }

        // cpu reduce - openmp parallel reduction
        template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
        T reduce_impl(
            const domain_t<Rank>& domain,
            T                     init_value,
            MapFunc&&             map_func,
            ReduceOp&&            reduce_op
        )
            requires(std::is_same_v<ExecutionSpace, cpu_space>)
        {
            return cpu_reduce(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op)
            );
        }

        // cuda reduce - two-phase gpu reduction
        template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
        T reduce_impl(
            const domain_t<Rank>& domain,
            T                     init_value,
            MapFunc&&             map_func,
            ReduceOp&&            reduce_op
        )
            requires(std::is_same_v<ExecutionSpace, cuda_space>)
        {
            return cuda_reduce(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op),
                stream_.native_handle()
            );
        }

        // cpu dispatch implementation - inline wrapper to detail implementation
        template <std::uint64_t Rank, typename Func>
        void cpu_dispatch(const domain_t<Rank>& domain, Func&& kernel)
        {
            xpu::cpu_dispatch(domain, std::forward<Func>(kernel));
        }

        // cuda dispatch implementation - inline wrapper to detail implementation
        template <std::uint64_t Rank, typename Func>
        void cuda_dispatch(const domain_t<Rank>& domain, Func&& kernel, stream_handle_type stream)
        {
#ifdef XPU_CUDA_AVAILABLE
            xpu::cuda_dispatch(domain, std::forward<Func>(kernel), stream);
#else
            (void) domain;
            (void) kernel;
            (void) stream;
#endif
        }

        // cpu reduce implementation - inline wrapper to detail implementation
        template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
        T cpu_reduce(
            const domain_t<Rank>& domain,
            T                     init_value,
            MapFunc&&             map_func,
            ReduceOp&&            reduce_op
        )
        {
            return xpu::cpu_reduce(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op)
            );
        }

        // cuda reduce implementation - inline wrapper to detail implementation
        template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
        T cuda_reduce(
            const domain_t<Rank>& domain,
            T                     init_value,
            MapFunc&&             map_func,
            ReduceOp&&            reduce_op,
            stream_handle_type    stream
        )
        {
#ifdef XPU_CUDA_AVAILABLE
            return xpu::cuda_reduce(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op),
                stream
            );
#else
            (void) domain;
            (void) init_value;
            (void) map_func;
            (void) reduce_op;
            (void) stream;
            return init_value;
#endif
        }
    };

    // =============================================================================
    // factory functions
    // =============================================================================

    template <execution_space ExecutionSpace>
    executor_t<ExecutionSpace> make_executor(int device_id = 0)
    {
        return executor_t<ExecutionSpace>{device_id};
    }

    // create executor on specific cuda stream
    template <execution_space ExecutionSpace = cuda_space>
    auto make_stream_executor(typename ExecutionSpace::stream_handle_type stream, int device_id = 0)
        -> std::enable_if_t<std::is_same_v<ExecutionSpace, cuda_space>, executor_t<ExecutionSpace>>
    {
        // note: would need constructor that accepts existing stream
        // for now, create new executor and rely on stream management
        return executor_t<ExecutionSpace>{device_id};
    }

    // =============================================================================
    // convenience aliases
    // =============================================================================

    using cpu_executor  = executor_t<cpu_space>;
    using cuda_executor = executor_t<cuda_space>;

} // namespace xpu
