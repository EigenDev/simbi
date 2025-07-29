#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP

#include "core/base/buffer.hpp"
#include "system/execution.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace simbi {
    template <typename T>
    struct computation_t {
        T value_;
        execution_context_t context_;

        // monadic interface
        static computation_t pure(T value, execution_context_t ctx = {})
        {
            return {std::move(value), std::move(ctx)};
        }

        template <typename F>
        auto then(F&& operation) &&
        {
            auto result = operation(std::move(value_));
            return computation_t<decltype(result)>::pure(
                std::move(result),
                std::move(context_)
            );
        }

        T unwrap() && { return std::move(value_); }

        // pipeline syntax
        template <typename Op>
        auto operator|(Op&& op) &&
        {
            return std::forward<Op>(op)(std::move(*this));
        }
    };

    // free functions that work with the pipeline
    auto on_device(device_id_t target)
    {
        return [target](auto&& comp) {
            comp.context_.device_ = target;
            return std::move(comp);
        };
    }

    auto with_error_handling()
    {
        return [](auto&& comp) {
            return comp.then([](auto value) -> maybe_t<decltype(value)> {
                // error handling logic
                return maybe_t<decltype(value)>::some(std::move(value));
            });
        };
    }
}   // namespace simbi
#endif
