#ifndef SIMBI_TASK_COMPUTATION_HPP
#define SIMBI_TASK_COMPUTATION_HPP

#include "base/buffer.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace simbi {

    // pure description of WHAT to compute
    template <std::uint64_t Dims, typename ResultType>
    struct computational_task_t {
        using result_type = ResultType;
        // what coordinates to iterate over
        domain_t<Dims> domain_;
        // how to compute at each coord
        std::function<result_type(const iarray<Dims>&)> evaluator_;

        // metadata for optimization
        std::string task_name_;                   // for debugging/profiling
        std::size_t estimated_flops_per_coord_;   // for load balancing

        template <typename Expr>
        static computational_task_t from_expression(Expr&& expr)
        {
            return {
              .domain_    = expr.domain(),
              .evaluator_ = [expr = std::forward<Expr>(expr)](
                                const auto& coord
                            ) { return expr.eval(coord); },
              .task_name_                 = "expression_eval",
              .estimated_flops_per_coord_ = 1   // could be smarter
            };
        }
    };

    // execution plan - HOW to execute the task
    struct execution_plan_t {
        enum class strategy_t {
            auto_detect,     // use platform::is_gpu + env vars
            cpu_parallel,    // your thread pool
            single_gpu,      // single device
            multi_gpu,       // domain decomposition
            specific_cpus,   // explicit cpu count
            specific_gpus    // explicit gpu list
        };

        strategy_t strategy = strategy_t::auto_detect;
        std::vector<device_id_t> target_devices;
        std::size_t num_threads = 0;   // for cpu execution

        static execution_plan_t auto_detect()
        {
            return {.strategy = strategy_t::auto_detect};
        }

        static execution_plan_t cpu_parallel(std::size_t threads = 0)
        {
            return {
              .strategy    = strategy_t::cpu_parallel,
              .num_threads = threads
            };
        }

        static execution_plan_t single_gpu(int device_id = 0)
        {
            return {
              .strategy       = strategy_t::single_gpu,
              .target_devices = {device_id_t::gpu_device(device_id)}
            };
        }

        static execution_plan_t multi_gpu(const std::vector<int>& device_ids)
        {
            std::vector<device_id_t> devices;
            for (int id : device_ids) {
                devices.push_back(device_id_t::gpu_device(id));
            }
            return {
              .strategy       = strategy_t::multi_gpu,
              .target_devices = std::move(devices)
            };
        }
    };

    // the computation monad
    template <typename Task>
    struct computation_t {
        using task_type   = Task;
        using result_type = typename Task::result_type;

        Task task_;
        execution_plan_t plan_;

        // monadic interface
        static computation_t
        pure(Task task, execution_plan_t plan = execution_plan_t::auto_detect())
        {
            return {std::move(task), std::move(plan)};
        }

        // change execution strategy
        computation_t with_plan(execution_plan_t new_plan) &&
        {
            plan_ = std::move(new_plan);
            return std::move(*this);
        }

        // pipeline syntax
        template <typename Op>
        auto operator|(Op&& op) &&
        {
            return std::forward<Op>(op)(std::move(*this));
        }

        // access for executors
        const Task& task() const { return task_; }
        const execution_plan_t& plan() const { return plan_; }
    };

    // deduction guide
    template <typename T>
    computation_t(T) -> computation_t<T>;

    // factory function - converts expressions to computational tasks
    template <typename Expr>
    auto compute(Expr&& expr)
    {
        constexpr auto dims = Expr::dimensions;   // or however you expose this
        using coord_type    = iarray<dims>;
        using result_type   = decltype(expr.eval(coord_type{}));

        auto task = computational_task_t<dims, result_type>::from_expression(
            std::forward<Expr>(expr)
        );
        return computation_t<decltype(task)>::pure(std::move(task));
    }

    // execution strategy modifiers
    auto execute();

    auto execute_on_cpu(std::size_t num_threads = 0);

    auto execute_on_gpu(int device_id = 0);

    auto execute_on_gpus(const std::vector<int>& device_ids);

    // forward declaration for actual executor
    template <typename Computation>
    auto execute_computation(Computation&& comp);

}   // namespace simbi

#endif   // SIMBI_TASK_COMPUTATION_HPP
