#include "driver.hpp"
#include "srhd.hpp"

using namespace simbi;

Driver::Driver(){

}

Driver::~Driver(){

}


std::vector<std::vector<real>> Driver::run(
    std::vector<std::vector<real>> state,
    const int dim,
    const std::string regime,
    const InitialConditions &init_cond
) {

    if (regime == "relativistic") {
        if (dim == 1) {
            auto self = std::make_unique<SRHD<1>>(state, init_cond);
            // auto self = new SRHD<1>(state, init_cond);
            auto res = self->simulate([](real i){ return 0;}, [](real i){return 1;});
            return res;
        } else if (dim == 2) {
            // auto self = new SRHD<2>(state, init_cond);
            // auto res = self.simulate([](real i){ return 0;}, [](real i){return 1;});
            // delete self;
            // return res;
        } else {
            // auto self = new SRHD<3>(state, init_cond);
            // auto res = self.simulate([](real i){ return 0;}, [](real i){return 1;});
            // delete self;
            // return res;
        }
    }

    return std::vector<std::vector<real>> {{1, 1, 1}, {1,1,1}};
}

