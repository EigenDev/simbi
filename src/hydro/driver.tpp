#include <optional>
#include "state.hpp"

using namespace simbi;

template<typename F>
std::optional<F> optional_wrapper(F func){
    if (func){
        return func;
    }
    return {};
}

Driver::Driver(){

}

Driver::~Driver(){

}

template<typename Func>
void Driver::run(
    std::vector<std::vector<real>> state,
    const int dim,
    const std::string regime,
    const InitialConditions &init_cond,
    std::function<real(real)> const &scale_factor,
    std::function<real(real)> const &scale_factor_derivative,
    Func const &density_lambda,
    Func const &mom1_lambda,
    Func const &mom2_lambda,
    Func const &mom3_lambda,
    Func const &enrg_lambda
) {
    if (dim == 1) {
        hydrostate::simulate<1>(
            state, 
            init_cond, 
            regime,scale_factor, 
            scale_factor_derivative,
            optional_wrapper(density_lambda),
            optional_wrapper(mom1_lambda),
            std::nullopt,
            std::nullopt,
            optional_wrapper(enrg_lambda)
        );
    } else if (dim == 2) {
        hydrostate::simulate<2>(
            state, 
            init_cond, 
            regime,scale_factor, 
            scale_factor_derivative,
            optional_wrapper(density_lambda),
            optional_wrapper(mom1_lambda),
            optional_wrapper(mom2_lambda),
            std::nullopt,
            optional_wrapper(enrg_lambda)
        );
    } else {
        hydrostate::simulate<3>(
            state, 
            init_cond, 
            regime,scale_factor, 
            scale_factor_derivative,
            optional_wrapper(density_lambda),
            optional_wrapper(mom1_lambda),
            optional_wrapper(mom2_lambda),
            optional_wrapper(mom3_lambda),
            optional_wrapper(enrg_lambda)
        );
    }
}

