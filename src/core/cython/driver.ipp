#include "state.hpp"
#include <optional>

using namespace simbi;
using namespace simbi::hydrostate;

Driver::Driver() = default;

Driver::~Driver() = default;

void Driver::run(
    std::vector<std::vector<real>> state,
    const int dim,
    const std::string regime,
    InitialConditions& init_cond,
    std::function<real(real)> const& scale_factor,
    std::function<real(real)> const& scale_factor_derivative
)
{
    if (dim == 1) {
        if (regime == "classical") {
            hydrostate::simulate<1, HydroRegime::Newtonian>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
        else if (regime == "srhd") {
            hydrostate::simulate<1, HydroRegime::SRHD>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
        else {
            hydrostate::simulate<1, HydroRegime::RMHD>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
    }
    else if (dim == 2) {
        if (regime == "classical") {
            hydrostate::simulate<2, HydroRegime::Newtonian>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
        else if (regime == "srhd") {
            hydrostate::simulate<2, HydroRegime::SRHD>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
        else {
            hydrostate::simulate<2, HydroRegime::RMHD>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
    }
    else {
        if (regime == "classical") {
            hydrostate::simulate<3, HydroRegime::Newtonian>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
        else if (regime == "srhd") {
            hydrostate::simulate<3, HydroRegime::SRHD>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
        else {
            hydrostate::simulate<3, HydroRegime::RMHD>(
                state,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
    }
}
