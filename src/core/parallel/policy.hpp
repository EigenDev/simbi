/**
 * policy.hpp
 * execution policies that guide parallel execution
 */

#ifndef SIMBI_PARALLEL_POLICY_HPP
#define SIMBI_PARALLEL_POLICY_HPP

#include "adapter/device_adapter_api.hpp"
#include "config.hpp"
#include "core/types/alias/alias.hpp"
#include "util/parallel/exec_policy.hpp"
#include "util/parallel/parallel_for.hpp"
#include <algorithm>
#include <functional>
#include <memory>
#include <thread>
#include <utility>

namespace simbi::parallel {

    /**
     * configuration for parallel execution
     */
    struct policy_condig_t {
        // hardware configuration
        size_type threads_per_block   = 256;
        size_type blocks_per_grid     = 0;   // 0 means auto-configure
        size_type shared_memory_bytes = 0;

        // optional device selection
        int device_id = 0;

        // scheduling parameters
        enum class schedule_type {
            automatic,   // let the system decide
            static_,     // equal chunks
            dynamic,     // work stealing
            guided       // decreasing chunk sizes
        };

        schedule_type schedule = schedule_type::automatic;
        size_type chunk_size   = 0;   // 0 means auto-configure

        // memory options
        bool use_shared_memory  = true;
        bool use_managed_memory = false;

        // synchronization options
        bool synchronize_after = true;
    };

    /**
     * abstract base class for execution policies
     */
    class execution_policy_t
    {
      public:
        virtual ~execution_policy_t() = default;

        // execute a kernel over a range
        template <typename F>
        void execute_range(size_type begin, size_type end, F&& func) const
        {
            execute_range_impl(begin, end, std::forward<F>(func));
        }

        // get configuration
        const policy_condig_t& config() const { return config_; }

      protected:
        explicit execution_policy_t(policy_condig_t config) : config_(config) {}

        // implementation specific execution
        virtual void execute_range_impl(
            size_type begin,
            size_type end,
            std::function<void(size_type)> func
        ) const = 0;

        policy_condig_t config_;
    };

    /**
     * CPU-specific execution policy
     */
    class cpu_policy : public execution_policy_t
    {
      public:
        explicit cpu_policy(policy_condig_t config = {})
            : execution_policy_t(config)
        {
            // Set reasonable defaults for CPU
            if (config_.threads_per_block == 0) {
                config_.threads_per_block = std::thread::hardware_concurrency();
            }
        }

      protected:
        void execute_range_impl(
            size_type begin,
            size_type end,
            std::function<void(size_type)> func
        ) const override
        {
            // Use the existing parallel_for implementation
            simbi::ExecutionPolicy<> exec_policy;
            simbi::parallel_for(exec_policy, begin, end, func);
        }
    };

    /**
     * GPU-specific execution policy
     */
    class gpu_policy : public execution_policy_t
    {
      public:
        explicit gpu_policy(policy_condig_t config = {})
            : execution_policy_t(config)
        {
            // Set reasonable defaults for GPU
            if (config_.threads_per_block == 0) {
                config_.threads_per_block = 256;
            }

            // Select device
            if (config_.device_id >= 0) {
                gpu::api::set_device(config_.device_id);
            }
        }

      protected:
        void execute_range_impl(
            size_type begin,
            size_type end,
            std::function<void(size_type)> func
        ) const override
        {
            // Calculate grid dimensions
            const size_type total_elements    = end - begin;
            const size_type threads_per_block = config_.threads_per_block;
            const size_type blocks_needed =
                (total_elements + threads_per_block - 1) / threads_per_block;

            const size_type blocks_per_grid =
                config_.blocks_per_grid > 0
                    ? std::min(config_.blocks_per_grid, blocks_needed)
                    : blocks_needed;

            // Create execution policy
            simbi::ExecutionPolicy<> exec_policy(
                {blocks_per_grid},
                {threads_per_block}
            );

            // Launch kernel
            simbi::parallel_for(exec_policy, begin, end, func);

            // Synchronize if needed
            if (config_.synchronize_after) {
                exec_policy.synchronize();
            }
        }
    };

    /**
     * get default policy based on available hardware
     */
    inline std::shared_ptr<execution_policy_t> get_default_policy()
    {
        policy_condig_t config;

        if constexpr (platform::is_gpu) {
            return std::make_shared<gpu_policy>(config);
        }
        else {
            return std::make_shared<cpu_policy>(config);
        }
    }

}   // namespace simbi::parallel

#endif   // SIMBI_PARALLEL_POLICY_HPP
