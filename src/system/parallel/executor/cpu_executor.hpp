/**
 * cpu_executor.hpp
 * cpu-specific executor implementation
 */

#ifndef SIMBI_CORE_PARALLEL_CPU_EXECUTOR_HPP
#define SIMBI_CORE_PARALLEL_CPU_EXECUTOR_HPP

#include "core/types/alias.hpp"
#include "system/parallel/executor/executor.hpp"
#include "system/parallel/par_config.hpp"
#include <functional>

namespace simbi::parallel {

    /**
     * cpu_executor_t - handles execution on cpu
     */
    class cpu_executor_t : public executor_t
    {
      public:
        // construct with specific thread count
        explicit cpu_executor_t(
            std::uint64_t num_threads = get_optimal_thread_count()
        );

        // destructor ensures threads are cleaned up
        ~cpu_executor_t() override;

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
            if constexpr (platform::is_cpu) {
#pragma omp parallel for
                for (std::uint64_t ii = start; ii < end; ++ii) {
                    func(ii);
                }
            }
        };

        void execute_async_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override;

      private:
        std::uint64_t num_threads_;
        mutable bool has_async_work_ = false;

        // thread pool implementation detail
        // would defer to the thread_pool_t in the final implementation
    };

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_CPU_EXECUTOR_HPP
