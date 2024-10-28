/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       state.hpp
 * @brief      key namespace for context switching to correct sim state for run
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef STATE_HPP
#define STATE_HPP

#include "build_options.hpp"     // for real, DUAL, lint, luint
#include "util/functional.hpp"   // for simbi::function
#include <functional>            // for std::function
#include <optional>              // for optional
#include <string>                // for string
#include <vector>                // for vector

struct InitialConditions;

namespace simbi {
    namespace hydrostate {
        template <int dim>
        struct func_t {
            using type = int;
        };

        template <>
        struct func_t<1> {
            using type = simbi::function<real(real, real)>;
        };

        template <>
        struct func_t<2> {
            using type = simbi::function<real(real, real, real)>;
        };

        template <>
        struct func_t<3> {
            using type = simbi::function<real(real, real, real, real)>;
        };

        template <int dim>
        using fopt = std::optional<typename func_t<dim>::type>;

        enum class HydroRegime {
            Newtonian,
            SRHD,
            RMHD
        };

        template <int D, HydroRegime R>
        void simulate(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<D>> const& bsources,
            std::vector<fopt<D>> const& hsources,
            std::vector<fopt<D>> const& gsources
        ) = delete;

        template <>
        void simulate<1, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        );

        template <>
        void simulate<1, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        );

        template <>
        void simulate<1, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        );

        template <>
        void simulate<2, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        );

        template <>
        void simulate<2, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        );

        template <>
        void simulate<2, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        );

        template <>
        void simulate<3, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        );

        template <>
        void simulate<3, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        );

        template <>
        void simulate<3, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        );
    }   // namespace hydrostate
}   // namespace simbi

#endif