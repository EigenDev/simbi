/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       state.hpp
 * @brief      key namespace for context switching to correct sim state for run
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
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

#include "build_options.hpp"           // for real, DUAL, lint, luint
#include "core/types/functional.hpp"   // for simbi::function
#include <functional>                  // for std::function
#include <optional>                    // for optional
#include <string>                      // for string
#include <vector>                      // for vector

struct InitialConditions;

namespace simbi {
    namespace hydrostate {
        enum class HydroRegime {
            Newtonian,
            SRHD,
            RMHD
        };

        template <int D, HydroRegime R>
        void simulate(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        ) = delete;

        template <>
        void simulate<1, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<1, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<1, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<2, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<2, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<2, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<3, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<3, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

        template <>
        void simulate<3, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );
    }   // namespace hydrostate
}   // namespace simbi

#endif