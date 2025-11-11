#ifndef EXECUTION_CONTEXT_HPP
#define EXECUTION_CONTEXT_HPP

#include "execution/executor.hpp"
#include "hetero/adapter.hpp"
#include "memory/device.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace simbi {

    // forward decls
    class executor_error_t;
    class execution_context_t;

    // exception hierarchy
    class executor_error_t : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    class resource_limit_error : public executor_error_t
    {
        using executor_error_t::executor_error_t;
    };

    // performance and resource tracking
    struct executor_stats_t {
        std::atomic<std::size_t> tasks_submitted{0};
        std::atomic<std::size_t> tasks_completed{0};
        std::atomic<std::size_t> active_kernels{0};
        std::atomic<std::size_t> peak_memory_used{0};
        std::atomic<std::uint64_t> total_compute_time_ns{0};

        // last error, if any
        std::string last_error;
        std::chrono::system_clock::time_point last_error_time;
    };

    struct executor_limits_t {
        std::size_t max_concurrent_kernels{0};   // 0 = unlimited
        std::size_t max_memory_usage{0};         // 0 = unlimited
        bool allow_peer_access{true};            // for GPU executors

        // opt: scheduling priority
        int priority{0};   // Higher = more priority
    };

    class device_pool_t
    {
      private:
        std::vector<mem::device_t> devices_;
        std::atomic<std::size_t> next_device_{0};

      public:
        // default ctor for an empty/invalid pool
        device_pool_t() = default;

        // main constructor
        explicit device_pool_t(std::vector<mem::device_t> devices)
            : devices_(std::move(devices))
        {
            if (devices_.empty()) {
                throw executor_error_t("cannot create empty device pool");
            }
        }

        mem::device_t get_next_device()
        {
            if (devices_.empty()) {
                throw executor_error_t("device pool is empty");
            }
            // this is thread-safe
            std::size_t idx = next_device_++ % devices_.size();
            return devices_[idx];
        }

        std::size_t size() const { return devices_.size(); }

        const std::vector<mem::device_t>& devices() const { return devices_; }
    };

    // exec handle with metadata and resource management
    class executor_handle_t
    {
        friend class execution_context_t;

      public:
        // exec storage - matches all possible executor types
        using executor_variant = std::variant<
            exec::cpu_executor_t,
            // exec::par_cpu_executor_t,
            exec::omp_executor_t,
            exec::gpu_executor_t>;

      private:
        executor_variant executor_;
        executor_stats_t stats_;
        executor_limits_t limits_;
        mem::device_t device_;
        // for thread-safe stats/limits access
        mutable std::shared_mutex mutex_;

      public:
        template <typename T>
        explicit executor_handle_t(T&& exec, mem::device_t dev)
            : executor_(std::forward<T>(exec)), device_(dev)
        {
        }

        // thread-safe accessors
        template <typename T>
        T& get()
        {
            std::shared_lock lock(mutex_);
            try {
                return std::get<T>(executor_);
            }
            catch (const std::bad_variant_access&) {
                throw executor_error_t(
                    "Incorrect executor type requested for device " +
                    std::to_string(device_.device_id)
                );
            }
        }

        const executor_stats_t& stats() const
        {
            std::shared_lock lock(mutex_);
            return stats_;
        }

        const executor_limits_t& limits() const
        {
            std::shared_lock lock(mutex_);
            return limits_;
        }

        mem::device_t device() const { return device_; }

        // resource management
        bool check_and_update_limits(std::size_t memory_required)
        {
            std::unique_lock lock(mutex_);

            if (limits_.max_memory_usage > 0) {
                auto new_usage = stats_.peak_memory_used + memory_required;
                if (new_usage > limits_.max_memory_usage) {
                    return false;
                }
                stats_.peak_memory_used = new_usage;
            }

            if (limits_.max_concurrent_kernels > 0) {
                auto new_kernels = ++stats_.active_kernels;
                if (new_kernels > limits_.max_concurrent_kernels) {
                    --stats_.active_kernels;
                    return false;
                }
            }

            ++stats_.tasks_submitted;
            return true;
        }

        void record_task_completion(std::uint64_t compute_time_ns)
        {
            std::unique_lock lock(mutex_);
            ++stats_.tasks_completed;
            --stats_.active_kernels;
            stats_.total_compute_time_ns += compute_time_ns;
        }

        void set_limits(executor_limits_t limits)
        {
            std::unique_lock lock(mutex_);
            limits_ = limits;
        }

        void record_error(std::string error)
        {
            std::unique_lock lock(mutex_);
            stats_.last_error      = std::move(error);
            stats_.last_error_time = std::chrono::system_clock::now();
        }
    };

    // main execution context
    class execution_context_t
    {
      private:
        // thread-safe containers
        mutable std::shared_mutex executors_mutex_;
        mutable std::shared_mutex pools_mutex_;

        std::unordered_map<mem::device_t, executor_handle_t> executors_;

        // dev validation
        bool validate_device(mem::device_t dev) const
        {
            if (dev.is_gpu) {
                return dev.device_id >= 0 &&
                       dev.device_id < hetero::device::get_device_count();
            }
            return true;   // cpu devices always valid
        }

      public:
        // dtor ensures proper cleanup
        ~execution_context_t() { shutdown(); }

        // get or create executor for device
        template <typename ExecutorType = exec::default_executor_t>
        ExecutorType& get_executor(mem::device_t dev)
        {
            if (!validate_device(dev)) {
                throw executor_error_t(
                    "Invalid device requested: " + std::to_string(dev.device_id)
                );
            }

            std::unique_lock lock(executors_mutex_);

            auto it = executors_.find(dev);
            if (it == executors_.end()) {
                try {
                    auto [inserted, success] = [this, dev]() {
                        if constexpr (std::is_same_v<
                                          ExecutorType,
                                          exec::gpu_executor_t>) {
                            return executors_.try_emplace(
                                dev,
                                exec::gpu_executor_t(dev.device_id),
                                dev
                            );
                        }
                        else {
                            return executors_
                                .try_emplace(dev, ExecutorType{}, dev);
                        }
                    }();
                    if (!success) {
                        throw executor_error_t(
                            "Failed to create executor for device " +
                            std::to_string(dev.device_id)
                        );
                    }
                    it = inserted;
                }
                catch (const std::exception& e) {
                    throw executor_error_t(
                        std::string("Executor creation failed: ") + e.what()
                    );
                }
            }

            return it->second.template get<ExecutorType>();
        }

        // device pool management
        device_pool_t create_device_pool(std::vector<mem::device_t> devices)
        {
            for (const auto& dev : devices) {
                if (!validate_device(dev)) {
                    throw executor_error_t(
                        "invalid device in pool creation: " +
                        std::to_string(dev.device_id)
                    );
                }
            }
            return device_pool_t(std::move(devices));
        }

        device_pool_t create_gpu_pool()
        {
            std::vector<mem::device_t> devices;
            auto n_gpus = hetero::device::get_device_count();
            if (n_gpus == 0) {
                // it's valid to have a pool of 0 gpus,
                // but the pool ctor will throw if we try to use it.
                // or, we could throw here. [TODO]: decide
                throw executor_error_t("no gpus found to create gpu pool");
            }
            devices.reserve(n_gpus);
            for (std::int64_t ii = 0; ii < n_gpus; ++ii) {
                devices.push_back(mem::device_t::gpu(ii));
            }
            return create_device_pool(std::move(devices));
        }

        // resource management
        void set_executor_limits(mem::device_t dev, executor_limits_t limits)
        {
            std::shared_lock lock(executors_mutex_);

            auto it = executors_.find(dev);
            if (it != executors_.end()) {
                it->second.set_limits(std::move(limits));
            }
        }

        // cleanup
        void shutdown()
        {
            std::unique_lock exec_lock(executors_mutex_);
            std::unique_lock pool_lock(pools_mutex_);

            for (auto& [dev, handle] : executors_) {
                std::visit(
                    [](auto& exec) {
                        if constexpr (requires { exec.synchronize(); }) {
                            exec.synchronize();
                        }
                    },
                    handle.executor_
                );
            }

            executors_.clear();
        }
    };

    // thread-local context accessor
    inline execution_context_t& current_context()
    {
        thread_local execution_context_t ctx;
        return ctx;
    }

}   // namespace simbi

#endif   // EXECUTION_CONTEXT_HPP
