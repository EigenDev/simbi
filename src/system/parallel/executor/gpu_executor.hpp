/**
 * gpu_executor.hpp
 * gpu-specific executor implementation
 */

#ifndef SIMBI_CORE_PARALLEL_GPU_EXECUTOR_HPP
#define SIMBI_CORE_PARALLEL_GPU_EXECUTOR_HPP

#include "config.hpp"
#include "core/types/alias.hpp"
#include "system/adapter/device_adapter_api.hpp"
#include "system/adapter/device_types.hpp"
#include "system/parallel/executor/executor.hpp"
#include "system/parallel/par_config.hpp"
#include <functional>
#include <vector>

namespace simbi::parallel {

    /**
     * gpu_executor_t - handles execution on gpu
     */
    class gpu_executor_t : public executor_t
    {
      public:
        // construct with specific device
        explicit gpu_executor_t(std::int64_t device_id = 0);

        // construct with multiple devices
        explicit gpu_executor_t(const std::vector<std::int64_t>& device_ids);

        // wait for all operations to complete
        void synchronize() const override;

      protected:
        // implementations of the base class methods
        void execute_range_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override
        {
            // launch kernel using the adapter interface
            auto config = grid::config(
                grid::calculate_grid(end - start, default_threads_per_block),
                {default_threads_per_block, 1, 1}
            );

            grid::launch(
                [=] DEV(std::uint64_t idx) {
                    if (idx < end - start) {
                        func(start + idx);
                    }
                },
                config
            );
        };

        void execute_async_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override;

      private:
        std::vector<std::int64_t> device_ids_;
        std::vector<adapter::stream_t<>> streams_;

        // calculate grid and block dimensions
        adapter::types::dim3 calculate_grid_size(std::uint64_t elements) const;

        // launch kernel on specific device
        template <typename Func>
        void launch_on_device(
            std::int64_t device_index,
            std::uint64_t start,
            std::uint64_t end,
            Func&& func
        ) const;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_GPU_EXECUTOR_HPP
