// =============================================================================
// execution_concepts.hpp
//
// c++20 concepts for high-performance heterogeneous task execution across
// cpu and gpu devices. defines requirements for executors, kernels, and
// async operations that work with any vendor (cuda, rocm, oneapi). optimized
// for massive simulation workloads with zero-overhead vendor abstraction.
//
// design principles:
//   - coroutine-ready: native c++20 co_await support
//   - simd-aware: vectorization and parallel execution
//   - numa-conscious: cpu affinity and memory locality
//   - vendor-agnostic: unified interface for all devices
//
// usage:
//   template<hetero_executor Executor>
//   auto async_algorithm(Executor& exec) -> task<result_t>;
// =============================================================================

#pragma once

#include "device_concepts.hpp"
#include "memory_concepts.hpp"

#include <concepts>
#include <coroutine>
#include <functional>
#include <future>
#include <ranges>
#include <type_traits>
#include <utility>

namespace xpu::core {

    // =============================================================================
    // execution token concepts for async operations
    // =============================================================================

    template <typename Token>
    concept execution_token = requires(Token token) {
        typename Token::result_type;

        // token state queries
        { token.is_ready() } -> std::convertible_to<bool>;
        { token.wait() } -> std::same_as<void>;
        { token.get() } -> std::convertible_to<typename Token::result_type>;

        // async composition
        { token.valid() } -> std::convertible_to<bool>;
    };

    template <typename Token>
    concept async_execution_token = execution_token<Token> && requires(Token token) {
        typename Token::duration_type;

        // timeout support
        {
            token.wait_for(std::declval<typename Token::duration_type>())
        } -> std::convertible_to<std::future_status>;

        // cancellation support
        { token.request_stop() } -> std::same_as<void>;
        { token.is_cancelled() } -> std::convertible_to<bool>;
    };

    template <typename Token>
    concept coroutine_token = execution_token<Token> && requires(Token token) {
        // c++20 coroutine support
        { token.operator co_await() };
    };

    // =============================================================================
    // kernel and function concepts
    // =============================================================================

    template <typename Kernel>
    concept device_kernel = requires {
        typename Kernel::argument_tuple_type;
        typename Kernel::return_type;

        // kernel properties
        { Kernel::grid_size_hint() } -> std::convertible_to<std::size_t>;
        { Kernel::block_size_hint() } -> std::convertible_to<std::size_t>;
        { Kernel::shared_memory_bytes() } -> std::convertible_to<std::size_t>;
        { Kernel::registers_per_thread() } -> std::convertible_to<std::size_t>;
    };

    template <typename Func, typename... Args>
    concept cpu_callable =
        std::invocable<Func, Args...> && (!std::is_void_v<std::invoke_result_t<Func, Args...>> ||
                                          std::is_void_v<std::invoke_result_t<Func, Args...>>);

    template <typename Func>
    concept vectorizable_function = cpu_callable<Func> && requires {
        // simd-friendly properties
        { Func::is_vectorizable } -> std::convertible_to<bool>;
        { Func::vector_width() } -> std::convertible_to<std::size_t>;
        { Func::alignment_requirement() } -> std::convertible_to<std::size_t>;
    } && Func::is_vectorizable;

    template <typename Range>
    concept execution_range = std::ranges::sized_range<Range> &&
                              std::ranges::random_access_range<Range> && requires(Range range) {
                                  // parallel execution hints
                                  { std::ranges::size(range) } -> std::convertible_to<std::size_t>;
                                  { range.chunk_size_hint() } -> std::convertible_to<std::size_t>;
                                  { range.parallel_depth() } -> std::convertible_to<std::size_t>;
                              };

    // =============================================================================
    // execution space concepts
    // =============================================================================

    template <typename Space>
    concept execution_space = requires {
        // note: executor_type, memory_space_type, token_type removed to avoid
        // circular dependencies with template parameter constraints
        // these types are accessed via execution_space_traits<Space> instead

        // space properties
        { Space::is_host_space } -> std::convertible_to<bool>;
        { Space::is_device_space } -> std::convertible_to<bool>;
        { Space::supports_async } -> std::convertible_to<bool>;
        { Space::supports_kernels } -> std::convertible_to<bool>;

        // execution characteristics
        { Space::max_concurrency() } -> std::convertible_to<std::size_t>;
        { Space::preferred_block_size() } -> std::convertible_to<std::size_t>;
        { Space::memory_bandwidth_gb_per_sec() } -> std::convertible_to<double>;
    };

    template <typename Space>
    concept host_execution_space = execution_space<Space> && Space::is_host_space;

    template <typename Space>
    concept device_execution_space =
        execution_space<Space> && Space::is_device_space && !Space::is_host_space;

    template <typename Space>
    concept unified_execution_space =
        execution_space<Space> && Space::is_host_space && Space::is_device_space;

    template <typename Space>
    concept async_execution_space = execution_space<Space> && Space::supports_async;

    // =============================================================================
    // executor concepts for task submission
    // =============================================================================

    template <typename Executor>
    concept basic_executor = requires(Executor exec) {
        typename Executor::execution_space_type;
        typename Executor::token_type;

        requires execution_space<typename Executor::execution_space_type>;
        requires execution_token<typename Executor::token_type>;

        // synchronization
        { exec.sync() } -> std::same_as<void>;
    } && requires(Executor exec) {
        // basic submission - templated check in separate requires clause
        {
            exec.submit(std::declval<std::function<void()>>())
        } -> std::convertible_to<typename Executor::token_type>;
    };

    template <typename Executor>
    concept async_executor = basic_executor<Executor> && requires(Executor exec) {
        typename Executor::stream_type;

        // stream-based execution
        { exec.default_stream() } -> std::convertible_to<typename Executor::stream_type>;
        { exec.create_stream() } -> std::convertible_to<typename Executor::stream_type>;
    } && requires(Executor exec) {
        // async submission with dependencies - templated check in separate requires clause
        {
            exec.submit_async(
                std::declval<std::function<void()>>(),
                std::declval<typename Executor::stream_type>()
            )
        } -> std::convertible_to<typename Executor::token_type>;
    };

    template <typename Executor>
    concept parallel_executor = async_executor<Executor> && requires(Executor exec) {
        // work distribution
        { exec.hardware_concurrency() } -> std::convertible_to<std::size_t>;
        {
            exec.optimal_block_size(std::declval<std::size_t>())
        } -> std::convertible_to<std::size_t>;
    };

    template <typename Executor>
    concept kernel_executor = parallel_executor<Executor> && requires(Executor exec) {
        typename Executor::kernel_type;

        // resource queries
        { exec.max_threads_per_block() } -> std::convertible_to<std::size_t>;
        { exec.max_blocks_per_grid() } -> std::convertible_to<std::size_t>;
        { exec.shared_memory_per_block() } -> std::convertible_to<std::size_t>;
    };

    // composite executor concept
    template <typename Executor>
    concept hetero_executor = async_executor<Executor> && requires(Executor exec) {
        typename Executor::device_type;
        requires hetero_device<typename Executor::device_type>;

        // device access
        { exec.device() } -> std::convertible_to<typename Executor::device_type&>;
        { exec.device_id() } -> std::convertible_to<int>;

        // execution context
        { exec.is_initialized() } -> std::convertible_to<bool>;
        { exec.initialize() } -> std::same_as<void>;
        { exec.shutdown() } -> std::same_as<void>;
    };

    // =============================================================================
    // coroutine support for async execution
    // =============================================================================

    template <typename T>
    class task
    {
      public:
        struct promise_type
        {
            task get_return_object()
            {
                return task{std::coroutine_handle<promise_type>::from_promise(*this)};
            }

            std::suspend_never initial_suspend() noexcept
            {
                return {};
            }

            std::suspend_never final_suspend() noexcept
            {
                return {};
            }

            void unhandled_exception()
            {
                exception_ = std::current_exception();
            }

            void return_value(T&& value)
            {
                result_ = std::forward<T>(value);
            }

            void return_value(const T& value)
            {
                result_ = value;
            }

            T result()
            {
                if (exception_) {
                    std::rethrow_exception(exception_);
                }
                return std::move(result_.value());
            }

          private:
            std::optional<T>   result_;
            std::exception_ptr exception_;
        };

        task(std::coroutine_handle<promise_type> handle) : handle_(handle) {}

        ~task()
        {
            if (handle_) {
                handle_.destroy();
            }
        }

        task(const task&)            = delete;
        task& operator=(const task&) = delete;

        task(task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}

        task& operator=(task&& other) noexcept
        {
            if (this != &other) {
                if (handle_) {
                    handle_.destroy();
                }
                handle_ = std::exchange(other.handle_, {});
            }
            return *this;
        }

        T get()
        {
            if (!handle_) {
                throw std::runtime_error("task is empty");
            }
            return handle_.promise().result();
        }

        bool is_ready() const noexcept
        {
            return handle_ && handle_.done();
        }

        // coroutine awaiter interface
        bool await_ready() const noexcept
        {
            return is_ready();
        }

        void await_suspend(std::coroutine_handle<> continuation) noexcept
        {
            continuation_ = continuation;
        }

        T await_resume()
        {
            return get();
        }

      private:
        std::coroutine_handle<promise_type> handle_;
        std::coroutine_handle<>             continuation_;
    };

    // specialization for void
    template <>
    class task<void>
    {
      public:
        struct promise_type
        {
            task get_return_object()
            {
                return task{std::coroutine_handle<promise_type>::from_promise(*this)};
            }

            std::suspend_never initial_suspend() noexcept
            {
                return {};
            }

            std::suspend_never final_suspend() noexcept
            {
                return {};
            }

            void unhandled_exception()
            {
                exception_ = std::current_exception();
            }

            void return_void() noexcept {}

            void result()
            {
                if (exception_) {
                    std::rethrow_exception(exception_);
                }
            }

          private:
            std::exception_ptr exception_;
        };

        task(std::coroutine_handle<promise_type> handle) : handle_(handle) {}

        ~task()
        {
            if (handle_) {
                handle_.destroy();
            }
        }

        task(const task&)            = delete;
        task& operator=(const task&) = delete;

        task(task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}

        task& operator=(task&& other) noexcept
        {
            if (this != &other) {
                if (handle_) {
                    handle_.destroy();
                }
                handle_ = std::exchange(other.handle_, {});
            }
            return *this;
        }

        void get()
        {
            if (!handle_) {
                throw std::runtime_error("task is empty");
            }
            handle_.promise().result();
        }

        bool is_ready() const noexcept
        {
            return handle_ && handle_.done();
        }

        // coroutine awaiter interface
        bool await_ready() const noexcept
        {
            return is_ready();
        }

        void await_suspend(std::coroutine_handle<> continuation) noexcept
        {
            continuation_ = continuation;
        }

        void await_resume()
        {
            get();
        }

      private:
        std::coroutine_handle<promise_type> handle_;
        std::coroutine_handle<>             continuation_;
    };

    // =============================================================================
    // execution policy concepts
    // =============================================================================

    template <typename Policy>
    concept execution_policy = requires {
        // execution characteristics
        { Policy::is_parallel } -> std::convertible_to<bool>;
        { Policy::is_vectorized } -> std::convertible_to<bool>;
        { Policy::is_unsequenced } -> std::convertible_to<bool>;

        // performance hints
        { Policy::chunk_size() } -> std::convertible_to<std::size_t>;
        { Policy::max_concurrency() } -> std::convertible_to<std::size_t>;
    };

    struct sequential_policy_t
    {
        static constexpr bool        is_parallel    = false;
        static constexpr bool        is_vectorized  = false;
        static constexpr bool        is_unsequenced = false;
        static constexpr std::size_t chunk_size()
        {
            return 1;
        }
        static constexpr std::size_t max_concurrency()
        {
            return 1;
        }
    };

    struct parallel_policy_t
    {
        static constexpr bool        is_parallel    = true;
        static constexpr bool        is_vectorized  = false;
        static constexpr bool        is_unsequenced = false;
        static constexpr std::size_t chunk_size()
        {
            return 1000;
        }
        static inline std::size_t max_concurrency()
        {
            return std::thread::hardware_concurrency();
        }
    };

    struct vectorized_policy_t
    {
        static constexpr bool        is_parallel    = true;
        static constexpr bool        is_vectorized  = true;
        static constexpr bool        is_unsequenced = true;
        static constexpr std::size_t chunk_size()
        {
            return 1024;
        }
        static inline std::size_t max_concurrency()
        {
            return std::thread::hardware_concurrency();
        }
    };

    template <hetero_device Device>
    struct device_policy_t
    {
        static constexpr bool        is_parallel    = true;
        static constexpr bool        is_vectorized  = Device::supports_simd;
        static constexpr bool        is_unsequenced = true;
        static constexpr std::size_t chunk_size()
        {
            return Device::preferred_block_size;
        }
        static constexpr std::size_t max_concurrency()
        {
            return Device::compute_units() * Device::max_threads_per_block();
        }
    };

    // standard policy objects
    inline constexpr sequential_policy_t seq{};
    inline constexpr parallel_policy_t   par{};
    inline constexpr vectorized_policy_t par_unseq{};

    // =============================================================================
    // algorithm dispatch utilities
    // =============================================================================

    template <execution_policy Policy, hetero_executor Executor>
    constexpr auto select_executor(Executor& exec)
    {
        if constexpr (Policy::is_parallel && Policy::is_vectorized) {
            return exec.vectorized_executor();
        }
        else if constexpr (Policy::is_parallel) {
            return exec.parallel_executor();
        }
        else {
            return exec.sequential_executor();
        }
    }

    template <execution_policy Policy, execution_range Range>
    constexpr std::size_t calculate_chunk_size(const Range& range)
    {
        std::size_t size        = std::ranges::size(range);
        std::size_t hint        = Policy::chunk_size();
        std::size_t concurrency = Policy::max_concurrency();

        if constexpr (Policy::is_parallel) {
            return std::max(hint, size / concurrency);
        }
        else {
            return size;
        }
    }

    // =============================================================================
    // execution graph concepts for complex workflows
    // =============================================================================

    template <typename Graph>
    concept execution_graph = requires(Graph graph) {
        typename Graph::node_type;
        typename Graph::edge_type;

        // graph structure
        {
            graph.add_node(std::declval<typename Graph::node_type>())
        } -> std::convertible_to<std::size_t>;
        {
            graph.add_edge(std::declval<std::size_t>(), std::declval<std::size_t>())
        } -> std::same_as<void>;

        // execution
        { graph.execute() } -> execution_token;
        { graph.is_dag() } -> std::convertible_to<bool>;
        { graph.topological_sort() } -> std::ranges::range;
    };

    template <typename Node>
    concept graph_node = requires(Node node) {
        typename Node::result_type;
        typename Node::dependency_list_type;

        // node execution
        { node.execute() } -> std::convertible_to<typename Node::result_type>;
        { node.dependencies() } -> std::convertible_to<typename Node::dependency_list_type>;
        { node.is_ready() } -> std::convertible_to<bool>;
    };

    // =============================================================================
    // compile-time execution optimization
    // =============================================================================

    template <typename Func, typename Range>
    constexpr bool is_vectorizable_v =
        vectorizable_function<Func> && std::is_arithmetic_v<std::ranges::range_value_t<Range>>;

    template <execution_space Space, typename Func>
    constexpr bool prefer_device_execution_v = device_execution_space<Space> && device_kernel<Func>;

    template <execution_space Source, execution_space Target>
    constexpr bool requires_synchronization_v = !std::same_as<Source, Target>;

    // optimal_block_size helper function (can't be constexpr variable template with Range)
    template <hetero_executor Executor, typename Range>
    inline std::size_t optimal_block_size(const Range& range)
    {
        return std::min(
            Executor::execution_space_type::preferred_block_size(),
            std::ranges::size(range) / Executor::execution_space_type::max_concurrency()
        );
    }

} // namespace xpu::core
