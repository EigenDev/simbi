/**
 * multi_device.hpp
 * multi-device orchestration
 */

#ifndef SIMBI_CORE_PARALLEL_MULTI_DEVICE_HPP
#define SIMBI_CORE_PARALLEL_MULTI_DEVICE_HPP

#include "system/parallel/executor/executor.hpp"
#include "types/alias.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace simbi::parallel {

    /**
     * multi_device_executor_t - handles execution across multiple devices
     */
    class multi_device_executor_t : public executor_t
    {
      public:
        // construct with specific executors
        explicit multi_device_executor_t(
            std::vector<std::shared_ptr<executor_t>> executors
        );

        // construct with all available devices
        static std::shared_ptr<multi_device_executor_t>
        create_with_all_devices();

        // wait for all operations to complete
        void synchronize() const override;

      protected:
        // implementations of the base class methods
        void execute_range_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override;

        void execute_async_impl(
            std::uint64_t start,
            std::uint64_t end,
            std::function<void(std::uint64_t)> func
        ) const override;

      private:
        std::vector<std::shared_ptr<executor_t>> executors_;

        // divide work among executors
        struct work_division_t {
            std::uint64_t start;
            std::uint64_t end;
            std::uint64_t executor_index;
        };

        std::vector<work_division_t>
        divide_work(std::uint64_t start, std::uint64_t end) const;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_CORE_PARALLEL_MULTI_DEVICE_HPP
