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

#include "build_options.hpp"   // for real
#include "newt.hpp"            // for Newtonian
#include "rmhd.hpp"            // for RMHD
#include "srhd.hpp"            // for SRHD
#include <functional>          // for function
#include <optional>            // for optional
#include <string>              // for string
#include <vector>              // for vector

using namespace simbi::helpers;
struct InitialConditions;

namespace simbi {
    namespace hydrostate {
        template <int dim>
        using fopt = std::optional<typename real_func<dim>::type>;

        template <int D>
        void simulate(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<D>> const& bsources,
            std::vector<fopt<D>> const& hsources,
            std::vector<fopt<D>> const& gsources
        ) = delete;

        template <>
        void simulate<1>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        );

        template <>
        void simulate<2>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        );

        template <>
        void simulate<3>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        );
    }   // namespace hydrostate
}   // namespace simbi

#endif