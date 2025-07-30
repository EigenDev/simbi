#ifndef EXECUTOR_TRAITS_HPP
#define EXECUTOR_TRAITS_HPP

#include "executor.hpp"

namespace simbi::execution {
    enum class strategy_type {
        sequential,   // single-threaded execution
        parallel,     // multi-threaded execution
        gpu           // gpu execution
    };

    // minimal traits - just identify executor type
    template <typename Executor>
    struct executor_traits {
        static constexpr bool is_gpu      = false;
        static constexpr bool is_parallel = false;
        static constexpr strategy_type default_strategy =
            strategy_type::sequential;
    };

    // specializations for each executor type
    template <>
    struct executor_traits<async::cpu_executor_t> {
        static constexpr bool is_gpu      = false;
        static constexpr bool is_parallel = false;
        static constexpr strategy_type default_strategy =
            strategy_type::sequential;
    };

    template <>
    struct executor_traits<async::par_cpu_executor_t> {
        static constexpr bool is_gpu      = false;
        static constexpr bool is_parallel = true;
        static constexpr strategy_type default_strategy =
            strategy_type::parallel;
    };

    template <>
    struct executor_traits<async::gpu_executor_t> {
        static constexpr bool is_gpu                    = true;
        static constexpr bool is_parallel               = true;
        static constexpr strategy_type default_strategy = strategy_type::gpu;
    };

    // simple helper to get default strategy
    template <typename Executor>
    constexpr strategy_type get_default_strategy(const Executor&)
    {
        return executor_traits<Executor>::default_strategy;
    }

}   // namespace simbi::execution

#endif   // SIMBI_EXECUTION_EXECUTOR_TRAITS_HPP
