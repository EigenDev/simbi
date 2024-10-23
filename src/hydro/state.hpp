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

struct InitialConditions;

namespace simbi {
    namespace hydrostate {
        template <int dim>
        struct func_type {
            using type = int;
        };

        template <int dim>
        using fopt = std::optional<typename func_type<dim>::type>;

        template <>
        struct func_type<1> {
            using type = std::function<real(real, real)>;
        };

        template <>
        struct func_type<2> {
            using type = std::function<real(real, real, real)>;
        };

        template <>
        struct func_type<3> {
            using type = std::function<real(real, real, real, real)>;
        };

        // Make use of the default template a compilation error
        // template <int dim, typename F>
        // void simulate(
        //     std::vector<std::vector<real>>& state,
        //     const InitialConditions& init_cond,
        //     const std::string regime,
        //     std::function<real(real)> const& scale_factor,
        //     std::function<real(real)> const& scale_factor_derivative,
        //     const std::vector<F>& bsources,
        //     const std::vector<F>& hsources,
        //     const std::vector<F>& gsources
        // );

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