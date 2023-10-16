#include "driver.hpp"
#include "srhd.hpp"
#include "newt.hpp"
using namespace simbi;
using namespace std::placeholders;

using func_one_var = std::function<real(real)>;
using func_two_var = std::function<real(real, real)>;
using func_thr_var = std::function<real(real, real, real)>;

Driver::Driver(){

}

Driver::~Driver(){

}

void Driver::run(
    std::vector<std::vector<real>> state,
    const int dim,
    const std::string regime,
    const InitialConditions &init_cond,
    std::function<real(real)> const &scale_factor,
    std::function<real(real)> const &scale_factor_derivative,
    Driver::func_t const &density_lambda,
    Driver::func_t const &mom1_lambda,
    Driver::func_t const &mom2_lambda,
    Driver::func_t const &mom3_lambda,
    Driver::func_t const &enrg_lambda
) {
    // if (density_lambda == std::nullptr_t) {
    //     std::cout << "nullptr not working!" << "\n";
    //     std::cin.get();
    // }
    if (regime == "relativistic") {
        if (dim == 1) {
            auto self = std::make_unique<SRHD<1>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        } else if (dim == 2) {
            auto self = std::make_unique<SRHD<2>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        } else {
            auto self = std::make_unique<SRHD<3>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }
    } else {
        if (dim == 1) {
            auto self = std::make_unique<Newtonian<1>>(state, init_cond);
            self->simulate(
                scale_factor, 
                scale_factor_derivative);
        } else if (dim == 2) {
            auto self = std::make_unique<Newtonian<2>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        } else {
            auto self = std::make_unique<Newtonian<3>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }
    }
}

