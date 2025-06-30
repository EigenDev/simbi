#ifndef STATE_OPS_HPP
#define STATE_OPS_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "physics/eos/isothermal.hpp"
#include <concepts>

namespace simbi::structs {
    using namespace simbi::concepts;
    // ---- primary template for trait-based field access ----
    // this handles the different member names between primitive and
    // conserved types

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

    // ---- generic operator implementations using traits ----

    // addition operator
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
        // isothermal gas doesn't have energy, so we ingnore this
        // for isothermal gas states
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

    // subtraction operator
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

    // multiplication by scalar
    template <is_any_state_variable_c StateT>
    constexpr auto operator*(const StateT& lhs, const real rhs)
    {
        using traits_t = state_traits<StateT>;
        StateT result;

        traits_t::density(result) = traits_t::density(lhs) * rhs;
        traits_t::momentum_or_velocity(result) =
            traits_t::momentum_or_velocity(lhs) * rhs;
        if constexpr (!traits_t::is_isothermal) {
            // isothermal gas doesn't have energy, so we ignore this
            // for isothermal gas states
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

    // scalar multiplication (commutative)
    template <is_any_state_variable_c StateT>
    constexpr auto operator*(const real lhs, const StateT& rhs)
    {
        return rhs * lhs;   // leverage the previous overload
    }

    // division by scalar
    template <is_any_state_variable_c StateT>
    constexpr auto operator/(const StateT& lhs, const real rhs)
    {
        using traits_t = state_traits<StateT>;
        StateT result;

        traits_t::density(result) = traits_t::density(lhs) / rhs;
        traits_t::momentum_or_velocity(result) =
            traits_t::momentum_or_velocity(lhs) / rhs;
        if constexpr (!traits_t::is_isothermal) {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs) / rhs;
        }
        else {
            traits_t::energy_or_pressure(result) =
                traits_t::energy_or_pressure(lhs);
        }
        traits_t::passive_scalar(result) = traits_t::passive_scalar(lhs) / rhs;

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(result) =
                traits_t::magnetic_field(lhs) / rhs;
        }

        return result;
    }

    // equality comparison
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

    // inequality comparison
    template <is_any_state_variable_c StateT>
    constexpr auto operator!=(const StateT& lhs, const StateT& rhs)
    {
        return !(lhs == rhs);
    }

    // unary minus operator
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

    // increment operator
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

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(lhs) += traits_t::magnetic_field(rhs);
        }

        return lhs;
    }

    // decrement operator
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

        if constexpr (traits_t::has_magnetic_field) {
            traits_t::magnetic_field(lhs) -= traits_t::magnetic_field(rhs);
        }

        return lhs;
    }

}   // namespace simbi::structs
#endif   // STATE_OPS_HPP
