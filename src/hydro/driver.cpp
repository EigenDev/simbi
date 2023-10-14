#include "driver.hpp"
#include "srhd.hpp"
#include "newt.hpp"
using namespace simbi;

Driver::Driver(){

}

Driver::~Driver(){

}


void Driver::run(
    std::vector<std::vector<real>> state,
    const int dim,
    const std::string regime,
    const InitialConditions &init_cond
) {
    if (regime == "relativistic") {
        if (dim == 1) {
            auto self = std::make_unique<SRHD<1>>(state, init_cond);
            self->simulate([](real i){ return 1;}, [](real i){return 0;});
        } else if (dim == 2) {
            auto self = std::make_unique<SRHD<2>>(state, init_cond);
            self->simulate([](real i){ return 1;}, [](real i){return 0;});
        } else {
            auto self = std::make_unique<SRHD<3>>(state, init_cond);
            self->simulate([](real i){ return 1;}, [](real i){return 0;});
        }
    } else {
        if (dim == 1) {
            auto self = std::make_unique<Newtonian<1>>(state, init_cond);
            self->simulate([](real i){ return 1;}, [](real i){return 0;});
        } else if (dim == 2) {
            auto self = std::make_unique<Newtonian<2>>(state, init_cond);
            self->simulate([](real i){ return 1;}, [](real i){return 0;});
        } else {
            auto self = std::make_unique<Newtonian<3>>(state, init_cond);
            self->simulate([](real i){ return 1;}, [](real i){return 0;});
        }
    }
}

