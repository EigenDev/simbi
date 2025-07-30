#include "task.hpp"
#include <cstddef>
#include <vector>

namespace simbi {
    // execution strategy modifiers
    auto execute()
    {
        return [](auto&& comp) {
            // this will call the actual executor with auto-detected plan
            return execute_computation(std::move(comp));
        };
    }

    auto execute_on_cpu(std::size_t num_threads)
    {
        return [num_threads](auto&& comp) {
            return std::move(comp).with_plan(
                       execution_plan_t::cpu_parallel(num_threads)
                   ) |
                   execute();
        };
    }

    auto execute_on_gpu(int device_id)
    {
        return [device_id](auto&& comp) {
            return std::move(comp).with_plan(
                       execution_plan_t::single_gpu(device_id)
                   ) |
                   execute();
        };
    }

    auto execute_on_gpus(const std::vector<int>& device_ids)
    {
        return [device_ids](auto&& comp) {
            return std::move(comp).with_plan(
                       execution_plan_t::multi_gpu(device_ids)
                   ) |
                   execute();
        };
    }

}   // namespace simbi
