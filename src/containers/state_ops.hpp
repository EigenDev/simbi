#ifndef STATE_OPS_HPP
#define STATE_OPS_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "physics/eos/isothermal.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace simbi::structs {
    using namespace simbi::concepts;

    // ---- primary template for trait-based field access ----
    template <typename T>
    struct state_traits;

    // specialization for primitive hydro values
    template <typename T>
        requires is_hydro_primitive_c<T> && (!is_mhd_primitive_c<T>)
    struct state_traits<T> {
        static constexpr auto& density(const T& v) { return v.rho; }
        static constexpr auto& momentum_or_velocity(const T& v)
        {
            return v.vel;
        }
        static constexpr auto& energy_or_pressure(const T& v) { return v.pre; }
        static constexpr auto& passive_scalar(const T& v) { return v.chi; }

        static constexpr auto& density(T& v) { return v.rho; }
        static constexpr auto& momentum_or_velocity(T& v) { return v.vel; }
        static constexpr auto& energy_or_pressure(T& v) { return v.pre; }
        static constexpr auto& passive_scalar(T& v) { return v.chi; }

        static constexpr bool has_magnetic_field = false;
        static constexpr bool is_isothermal =
            std::same_as<typename T::eos_t, eos::isothermal_gas_eos_t>;
    };

    // specialization for conserved hydro values
    template <typename T>
        requires is_hydro_conserved_c<T> && (!is_mhd_conserved_c<T>)
    struct state_traits<T> {
        static constexpr auto& density(const T& v) { return v.den; }
        static constexpr auto& momentum_or_velocity(const T& v)
        {
            return v.mom;
        }
        static constexpr auto& energy_or_pressure(const T& v) { return v.nrg; }
        static constexpr auto& passive_scalar(const T& v) { return v.chi; }

        static constexpr auto& density(T& v) { return v.den; }
        static constexpr auto& momentum_or_velocity(T& v) { return v.mom; }
        static constexpr auto& energy_or_pressure(T& v) { return v.nrg; }
        static constexpr auto& passive_scalar(T& v) { return v.chi; }

        static constexpr bool has_magnetic_field = false;
        static constexpr bool is_isothermal =
            std::same_as<typename T::eos_t, eos::isothermal_gas_eos_t>;
    };

    // specialization for primitive MHD values
    template <typename T>
        requires is_mhd_primitive_c<T>
    struct state_traits<T> {
        static constexpr auto& density(const T& v) { return v.rho; }
        static constexpr auto& momentum_or_velocity(const T& v)
        {
            return v.vel;
        }
        static constexpr auto& energy_or_pressure(const T& v) { return v.pre; }
        static constexpr auto& magnetic_field(const T& v) { return v.mag; }
        static constexpr auto& passive_scalar(const T& v) { return v.chi; }

        static constexpr auto& density(T& v) { return v.rho; }
        static constexpr auto& momentum_or_velocity(T& v) { return v.vel; }
        static constexpr auto& energy_or_pressure(T& v) { return v.pre; }
        static constexpr auto& magnetic_field(T& v) { return v.mag; }
        static constexpr auto& passive_scalar(T& v) { return v.chi; }

        static constexpr bool has_magnetic_field = true;
        static constexpr bool is_isothermal =
            std::same_as<typename T::eos_t, eos::isothermal_gas_eos_t>;
    };

    // specialization for conserved MHD values
    template <typename T>
        requires is_mhd_conserved_c<T>
    struct state_traits<T> {
        static constexpr auto& density(const T& v) { return v.den; }
        static constexpr auto& momentum_or_velocity(const T& v)
        {
            return v.mom;
        }
        static constexpr auto& energy_or_pressure(const T& v) { return v.nrg; }
        static constexpr auto& magnetic_field(const T& v) { return v.mag; }
        static constexpr auto& passive_scalar(const T& v) { return v.chi; }

        static constexpr auto& density(T& v) { return v.den; }
        static constexpr auto& momentum_or_velocity(T& v) { return v.mom; }
        static constexpr auto& energy_or_pressure(T& v) { return v.nrg; }
        static constexpr auto& magnetic_field(T& v) { return v.mag; }
        static constexpr auto& passive_scalar(T& v) { return v.chi; }

        static constexpr bool has_magnetic_field = true;
        static constexpr bool is_isothermal =
            std::same_as<typename T::eos_t, eos::isothermal_gas_eos_t>;
    };

    // ---- Gas-only operation pipeline components ----

    // gas variable transformation functors
    struct scale_gas_t {
        real factor;
        constexpr explicit scale_gas_t(real f) : factor(f) {}
    };

    template <typename T>
    struct add_gas_t {
        T other;
        constexpr explicit add_gas_t(T&& o) : other(std::move(o)) {}
        constexpr explicit add_gas_t(const T& o) : other(o) {}
    };

    template <typename BinaryOp>
    struct combine_gas_t {
        BinaryOp op;
        constexpr explicit combine_gas_t(BinaryOp&& o) : op(std::move(o)) {}
    };

    // pipeline factory functions
    constexpr auto scale_gas(real factor) { return scale_gas_t{factor}; }

    template <typename T>
    constexpr auto add_gas(T&& other)
    {
        return add_gas_t<std::decay_t<T>>{std::forward<T>(other)};
    }

    template <typename BinaryOp>
    constexpr auto combine_gas(BinaryOp&& op)
    {
        return combine_gas_t<std::decay_t<BinaryOp>>{
          std::forward<BinaryOp>(op)
        };
    }

    // ---- helper functions for gas variable manipulation ----

    template <typename StateT, typename F>
    constexpr auto map_gas_vars(const StateT& state, F&& func)
    {
        using traits_t = state_traits<StateT>;
        StateT result  = state;   // copy preserves magnetic fields

        // apply function only to gas variables
        if constexpr (!traits_t::is_isothermal) {
            func(
                traits_t::density(result),
                traits_t::momentum_or_velocity(result),
                traits_t::energy_or_pressure(result),
                traits_t::passive_scalar(result)
            );
        }
        else {
            func(
                traits_t::density(result),
                traits_t::momentum_or_velocity(result),
                traits_t::passive_scalar(result)
            );
        }

        return result;
    }

    // ---- pipeline operator overloads ----

    // scale gas variables
    template <is_any_state_variable_c StateT>
    constexpr auto operator|(const StateT& state, scale_gas_t op)
    {
        return map_gas_vars(state, [factor = op.factor](auto&... vars) {
            ((vars *= factor), ...);   // fold expression
        });
    }

    // add to gas variables
    template <is_any_state_variable_c StateT>
    constexpr auto operator|(const StateT& state, add_gas_t<StateT> op)
    {
        using traits_t = state_traits<StateT>;
        return map_gas_vars(state, [&op](auto&... vars) {
            if constexpr (!traits_t::is_isothermal) {
                // for non-isothermal: density, momentum, energy, passive scalar
                auto var_tuple = std::tie(vars...);
                std::get<0>(var_tuple) += traits_t::density(op.other);
                std::get<1>(var_tuple) +=
                    traits_t::momentum_or_velocity(op.other);
                std::get<2>(var_tuple) +=
                    traits_t::energy_or_pressure(op.other);
                std::get<3>(var_tuple) += traits_t::passive_scalar(op.other);
            }
            else {
                // for isothermal: density, momentum, passive scalar (skip
                // energy)
                auto var_tuple = std::tie(vars...);
                std::get<0>(var_tuple) += traits_t::density(op.other);
                std::get<1>(var_tuple) +=
                    traits_t::momentum_or_velocity(op.other);
                std::get<2>(var_tuple) += traits_t::passive_scalar(op.other);
            }
        });
    }

    // combine gas variables with binary operation
    template <is_any_state_variable_c StateT, typename BinaryOp>
    constexpr auto operator|(const StateT& state, combine_gas_t<BinaryOp> op)
    {
        return map_gas_vars(state, [&op](auto&... vars) {
            ((vars = op.op(vars)), ...);
        });
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator+(const StateT& lhs, const StateT& rhs)
    {
        using traits_t = state_traits<StateT>;
        StateT result;

        traits_t::density(result) =
            traits_t::density(lhs) + traits_t::density(rhs);
        traits_t::momentum_or_velocity(result) =
            traits_t::momentum_or_velocity(lhs) +
            traits_t::momentum_or_velocity(rhs);

        if constexpr (!traits_t::is_isothermal) {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs) +
                traits_t::energy_or_pressure(rhs);
        }
        else {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs);
        }

        traits_t::passive_scalar(result) =
            traits_t::passive_scalar(lhs) + traits_t::passive_scalar(rhs);

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(result) =
                traits_t::magnetic_field(lhs) + traits_t::magnetic_field(rhs);
        }

        return result;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator-(const StateT& lhs, const StateT& rhs)
    {
        using traits_t = state_traits<StateT>;
        StateT result;

        traits_t::density(result) =
            traits_t::density(lhs) - traits_t::density(rhs);
        traits_t::momentum_or_velocity(result) =
            traits_t::momentum_or_velocity(lhs) -
            traits_t::momentum_or_velocity(rhs);

        if constexpr (!traits_t::is_isothermal) {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs) -
                traits_t::energy_or_pressure(rhs);
        }
        else {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs);
        }

        traits_t::passive_scalar(result) =
            traits_t::passive_scalar(lhs) - traits_t::passive_scalar(rhs);

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(result) =
                traits_t::magnetic_field(lhs) - traits_t::magnetic_field(rhs);
        }

        return result;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator*(const StateT& lhs, const real rhs)
    {
        using traits_t = state_traits<StateT>;
        StateT result;

        traits_t::density(result) = traits_t::density(lhs) * rhs;
        traits_t::momentum_or_velocity(result) =
            traits_t::momentum_or_velocity(lhs) * rhs;

        if constexpr (!traits_t::is_isothermal) {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs) * rhs;
        }
        else {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs);
        }

        traits_t::passive_scalar(result) = traits_t::passive_scalar(lhs) * rhs;

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(result) =
                traits_t::magnetic_field(lhs) * rhs;
        }

        return result;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator*(const real lhs, const StateT& rhs)
    {
        return rhs * lhs;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator/(const StateT& lhs, const real rhs)
    {
        return lhs * (1.0 / rhs);
    }

    template <is_any_state_variable_c StateT>
    constexpr auto& operator+=(StateT& lhs, const StateT& rhs)
    {
        using traits_t = state_traits<StateT>;

        traits_t::density(lhs) += traits_t::density(rhs);
        traits_t::momentum_or_velocity(lhs) +=
            traits_t::momentum_or_velocity(rhs);

        if constexpr (!traits_t::is_isothermal) {
            traits_t::energy_or_pressure(lhs) +=
                traits_t::energy_or_pressure(rhs);
        }

        traits_t::passive_scalar(lhs) += traits_t::passive_scalar(rhs);

        return lhs;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto& operator-=(StateT& lhs, const StateT& rhs)
    {
        using traits_t = state_traits<StateT>;

        traits_t::density(lhs) -= traits_t::density(rhs);
        traits_t::momentum_or_velocity(lhs) -=
            traits_t::momentum_or_velocity(rhs);

        if constexpr (!traits_t::is_isothermal) {
            traits_t::energy_or_pressure(lhs) -=
                traits_t::energy_or_pressure(rhs);
        }

        traits_t::passive_scalar(lhs) -= traits_t::passive_scalar(rhs);

        return lhs;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator==(const StateT& lhs, const StateT& rhs)
    {
        using traits_t = state_traits<StateT>;
        bool result =
            (traits_t::density(lhs) == traits_t::density(rhs)) &&
            (traits_t::momentum_or_velocity(lhs) ==
             traits_t::momentum_or_velocity(rhs)) &&
            (traits_t::energy_or_pressure(lhs) ==
             traits_t::energy_or_pressure(rhs)) &&
            (traits_t::passive_scalar(lhs) == traits_t::passive_scalar(rhs));

        if constexpr (traits_t::has_magnetic_field) {
            result = result && (traits_t::magnetic_field(lhs) ==
                                traits_t::magnetic_field(rhs));
        }

        return result;
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator!=(const StateT& lhs, const StateT& rhs)
    {
        return !(lhs == rhs);
    }

    template <is_any_state_variable_c StateT>
    constexpr auto operator-(const StateT& v)
    {
        using traits_t = state_traits<StateT>;
        StateT result;

        traits_t::density(result) = -traits_t::density(v);
        traits_t::momentum_or_velocity(result) =
            -traits_t::momentum_or_velocity(v);
        traits_t::energy_or_pressure(result) = -traits_t::energy_or_pressure(v);
        traits_t::passive_scalar(result)     = -traits_t::passive_scalar(v);

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(result) = -traits_t::magnetic_field(v);
        }

        return result;
    }

}   // namespace simbi::structs

#endif   // STATE_OPS_HPP
