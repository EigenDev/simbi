#ifndef DEVICE_CALLABLE_HPP
#define DEVICE_CALLABLE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory
#include "core/types/alias/function.hpp"
#include <functional>   // for std::function

namespace simbi::jit {
    template <size_type Dims>
    class DeviceCallable
    {
      private:
        // cpu functions
        std::function<user_function_t<Dims>> cpu_function_;

        // gpu functions
        void* device_function_ptr_ = nullptr;
        bool has_dev_function_     = false;

        // function name for diagnostics
        std::string function_name_;

      public:
        // Default constructor
        DeviceCallable() = default;

        // CPU constructor
        DeviceCallable(
            std::string name,
            std::function<user_function_t<Dims>> func
        )
            : cpu_function_(std::move(func)), function_name_(std::move(name))
        {
        }

        // GPU constructor
        DeviceCallable(std::string name, void* func)
            : device_function_ptr_(func),
              has_dev_function_(func != nullptr),
              function_name_(std::move(name))
        {
        }

        // Call operator with dimension-aware dispatch
        template <typename... Args>
            requires(sizeof...(Args) == Dims + 2)
        DEV void operator()(Args&&... args) const
        {
            if constexpr (global::on_gpu) {
                if (!has_dev_function_) {
                    printf(
                        "Device function '%s' not set\n",
                        function_name_.c_str()
                    );
                    return;
                }
                auto func_ptr = reinterpret_cast<user_function_ptr_t<Dims>>(
                    device_function_ptr_
                );
                func_ptr(std::forward<Args>(args)...);
            }
            else {
                if (!cpu_function_) {
                    // In CPU context, we can throw
                    throw std::runtime_error(
                        "CPU function '" + function_name_ + "' not set"
                    );
                }
                cpu_function_(std::forward<Args>(args)...);
            }
        }

        bool is_callable() const
        {
            return (global::on_gpu && has_dev_function_) ||
                   (!global::on_gpu && static_cast<bool>(cpu_function_));
        }

        const std::string& name() const { return function_name_; }
    };
}   // namespace simbi::jit

#endif   // DEVICE_CALLABLE_HPP
