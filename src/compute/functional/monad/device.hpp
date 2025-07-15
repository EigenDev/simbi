#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "core/base/buffer.hpp"
#include "system/execution.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace simbi {
    template <typename T, std::uint64_t Dims>
    struct field_t;

    template <typename T>
    struct device_computation_t {
        T value_;
        execution_context_t context_;

        static device_computation_t
        pure(T value, device_id_t device = device_id_t::cpu_device())
        {
            return {std::move(value), execution_context_t{device}};
        }

        template <typename F>
        auto then(F&& operation)
            -> device_computation_t<decltype(operation(std::move(value_)))>
        {
            // move instead of copy
            auto result = operation(std::move(value_));
            return {std::move(result), std::move(context_)};
        }

        template <typename F>
        auto try_then(F&& operation) -> device_computation_t<
            std::optional<decltype(operation(std::move(value_)))>>
        {
            try {
                auto result = operation(std::move(value_));
                return {std::optional{std::move(result)}, std::move(context_)};
            }
            catch (...) {
                using result_type = decltype(operation(std::move(value_)));
                return {std::optional<result_type>{}, std::move(context_)};
            }
        }

        template <typename Default>
        auto unwrap_or(Default&& default_value) &&
        {
            if constexpr (requires { value_.has_value(); }) {
                return value_.has_value()
                           ? std::move(*value_)
                           : std::forward<Default>(default_value);
            }
            else {
                return std::move(value_);
            }
        }

        T unwrap() && { return std::move(value_); }

        template <typename F>
        auto flat_then(F&& operation) -> decltype(operation(std::move(*this)))
        {
            return operation(std::move(*this));
        }

        template <typename DT, std::uint64_t Dims>
        auto
        migrate_field_to(field_t<DT, Dims>& field, device_id_t target) const
        {
            if (field.device() != target) {
                field.buffer_ = context_.to_device(field.buffer_);
            }
        }

        auto migrate_state_to(device_id_t target) &&
            requires requires {
                value_.cons;
                value_.prim;
            }
        {
            if (context_.device_ != target) {
                migrate_field_to(value_.cons, target);
                migrate_field_to(value_.prim, target);

                // handle flux arrays
                for (std::uint64_t dir = 0; dir < T::dimensions; ++dir) {
                    migrate_field_to(value_.flux[dir], target);
                }

                // handle MHD fields if present
                if constexpr (T::is_mhd) {
                    for (std::uint64_t dir = 0; dir < T::dimensions; ++dir) {
                        migrate_field_to(value_.bstaggs[dir], target);
                    }
                }

                context_ = execution_context_t{target};
            }
            return std::move(*this);
        }

        // convenience methods
        auto to_gpu(int device_id = 0) &&
        {
            return std::move(*this).migrate_state_to(
                device_id_t::gpu_device(device_id)
            );
        }

        auto to_cpu() &&
        {
            return std::move(*this).migrate_state_to(device_id_t::cpu_device());
        }
    };

    template <typename T>
    device_computation_t(T) -> device_computation_t<T>;

    template <typename T>
    auto make_device_computation(
        T&& value,
        device_id_t device = device_id_t::cpu_device()
    )
    {
        return device_computation_t<std::decay_t<T>>::pure(
            std::forward<T>(value),
            device
        );
    }

    template <typename T>
    auto pure(T&& value, device_id_t device = device_id_t::cpu_device())
    {
        return device_computation_t<std::decay_t<T>>::pure(
            std::forward<T>(value),
            device
        );
    }
}   // namespace simbi
#endif
