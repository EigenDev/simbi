/**
 * the srhd and newtonian headers can only be included here since
 * they are templated and cython cannot switch to the gpu compiler
 * when externing from state.hpp
*/
#include <variant>
#include "state.hpp"
#include "srhd.hpp"
#include "newt.hpp"

namespace simbi
{
    namespace hydrostate
    {
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
        ){
            using sr_or_nt = std::variant<std::unique_ptr<Newtonian<1>>, std::unique_ptr<SRHD<1>>>;
            auto self = [&]() -> sr_or_nt {
                if (regime == "relativistic") {
                    return std::make_unique<SRHD<1>>(state, init_cond);
                } else {
                    return std::make_unique<Newtonian<1>>(state, init_cond);
                } 
            }();
            
            std::visit([=](auto &&arg){
                arg->simulate(
                    scale_factor, 
                    scale_factor_derivative,
                    density_lambda,
                    mom1_lambda,
                    nullptr,
                    nullptr,
                    enrg_lambda
                );
            }, self);
        };

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
        ){
            using sr_or_nt = std::variant<std::unique_ptr<Newtonian<2>>, std::unique_ptr<SRHD<2>>>;
            auto self = [&]() -> sr_or_nt {
                if (regime == "relativistic") {
                    return std::make_unique<SRHD<2>>(state, init_cond);
                } else {
                    return std::make_unique<Newtonian<2>>(state, init_cond);
                } 
            }();
            std::visit([=](auto &&arg){
                arg->simulate(
                    scale_factor, 
                    scale_factor_derivative,
                    density_lambda,
                    mom1_lambda,
                    mom2_lambda,
                    mom3_lambda,
                    enrg_lambda
                );
            }, self);
        };

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
        ){
            using sr_or_nt = std::variant<std::unique_ptr<Newtonian<3>>, std::unique_ptr<SRHD<3>>>;
            auto self = [&]() -> sr_or_nt {
                if (regime == "relativistic") {
                    return std::make_unique<SRHD<3>>(state, init_cond);
                } else {
                    return std::make_unique<Newtonian<3>>(state, init_cond);
                } 
            }();
            std::visit([=](auto &&arg){
                arg->simulate(
                    scale_factor, 
                    scale_factor_derivative,
                    density_lambda,
                    mom1_lambda,
                    mom2_lambda,
                    mom3_lambda,
                    enrg_lambda
                );
            }, self);
        };
    } // namespace hydrostate

} // namespace simbi
