#ifndef DEVICE_CALLABLE_HPP
#define DEVICE_CALLABLE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory
#include <functional>          // for std::function

namespace simbi::jit {
    template <typename Signature>
    class DeviceCallable
    {
      private:
        using FunctionType = Signature;

        // cpu function pointer
        std::function<FunctionType> cpu_function_;

        // gpu function pointer
        void* device_function_ptr_;
        bool has_dev_function_{false};

      public:
        // cpu ctor
        DeviceCallable(std::function<Signature> func)
            : cpu_function_(std::move(func)), device_function_ptr_(nullptr)
        {
        }

        // gpu ctor
        DeviceCallable(devFunction_t func)
            : device_function_ptr_(func), has_dev_function_(true)
        {
        }

        // call operator for cpu
        template <typename... Args>
        auto operator()(Args&&... args) const -> std::enable_if_t<
            !global::on_gpu,
            std::invoke_result_t<std::function<Signature>, Args...>>
        {
            return cpu_func_(std::forward<Args>(args)...);
        }

        // call operator for gpu
        template <typename... Args>
        DEV auto operator()(dim3 grid, dim3 block, Args&&... args) const
            -> std::enable_if_t<
                global::on_gpu,
                std::invoke_result_t<devFunction_t, Args...>>
        {
            if (!has_dev_function_) {
                printf("Device function not set.");
            }

            void* kernel_args[] = {
              &args...,
            };

            // cast to the appropriate function pointer type and call
            using DeviceFuncType = Signature*;
            auto func_ptr =
                reinterpret_cast<DeviceFuncType>(device_function_ptr_);
            func_ptr(std::forward<Args>(args)...);
        }

        bool is_callable() const
        {
            return has_dev_function_ || static_cast<bool>(cpu_function_);
        }
    };
}   // namespace simbi::jit

#endif   // DEVICE_CALLABLE_HPP
