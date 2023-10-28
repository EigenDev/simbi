#ifndef STATE_HPP
#define STATE_HPP

#include <functional>         // for function
#include <optional>           // for optional
#include <string>             // for string
#include <vector>             // for vector
#include "build_options.hpp"  // for real

struct InitialConditions;
namespace simbi
{
    namespace hydrostate
    {   
        template<int dim>
        struct func_type {
            using type = int;
        };

        template<>
        struct func_type<1>{
            using type = std::function<real(real)>;
        };

        template<>
        struct func_type<2>{
            using type = std::function<real(real, real)>;
        };

        template<>
        struct func_type<3>{
            using type = std::function<real(real, real, real)>;
        };

        // Make use of the default template a compilation error
        template<int dim>
        void simulate(
            std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string regime,
            std::function<real(real)> const &scale_factor,
            std::function<real(real)> const &scale_factor_derivative,
            std::optional<typename func_type<dim>::type> const &density_lambda,
            std::optional<typename func_type<dim>::type> const &mom1_lambda,
            std::optional<typename func_type<dim>::type> const &mom2_lambda,
            std::optional<typename func_type<dim>::type> const &mom3_lambda,
            std::optional<typename func_type<dim>::type> const &enrg_lambda
        ) = delete;

        template<>
        void simulate<1>(
            std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string regime,
            std::function<real(real)> const &scale_factor,
            std::function<real(real)> const &scale_factor_derivative,
            std::optional<typename func_type<1>::type> const &density_lambda,
            std::optional<typename func_type<1>::type> const &mom1_lambda,
            std::optional<typename func_type<1>::type> const &mom2_lambda,
            std::optional<typename func_type<1>::type> const &mom3_lambda,
            std::optional<typename func_type<1>::type> const &enrg_lambda
        );

        template<>
        void simulate<2>(
            std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string regime,
            std::function<real(real)> const &scale_factor,
            std::function<real(real)> const &scale_factor_derivative,
            std::optional<typename func_type<2>::type> const &density_lambda,
            std::optional<typename func_type<2>::type> const &mom1_lambda,
            std::optional<typename func_type<2>::type> const &mom2_lambda,
            std::optional<typename func_type<2>::type> const &mom3_lambda,
            std::optional<typename func_type<2>::type> const &enrg_lambda
        );

        template<>
        void simulate<3>(
            std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string regime,
            std::function<real(real)> const &scale_factor,
            std::function<real(real)> const &scale_factor_derivative,
            std::optional<typename func_type<3>::type> const &density_lambda,
            std::optional<typename func_type<3>::type> const &mom1_lambda,
            std::optional<typename func_type<3>::type> const &mom2_lambda,
            std::optional<typename func_type<3>::type> const &mom3_lambda,
            std::optional<typename func_type<3>::type> const &enrg_lambda
        );
    } // namespace hydrostate    
} // namespace simbi

#endif