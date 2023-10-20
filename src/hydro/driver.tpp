#include "state.hpp"

using namespace simbi;

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
    if (density_lambda) {
        std::cout << "nullptr not working!" << "\n";
        std::cin.get();
    }

    auto self = hydrostate::create(state, init_cond, regime, dim);
    self->simulate(
        scale_factor, 
        scale_factor_derivative,
        density_lambda,
        mom1_lambda,
        mom2_lambda,
        mom3_lambda,
        enrg_lambda
    );
    // std::visit([=](auto &&arg){
    //     arg->simulate(
    //         scale_factor, 
    //         scale_factor_derivative,
    //         density_lambda,
    //         mom1_lambda,
    //         mom2_lambda,
    //         mom3_lambda,
    //         enrg_lambda
    //     );
    // }, self);
}

