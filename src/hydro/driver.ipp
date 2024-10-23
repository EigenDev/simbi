#include "state.hpp"
#include <optional>

using namespace simbi;
using namespace simbi::hydrostate;

template <typename F>
std::optional<F> optional_wrapper(F func)
{
    if (func) {
        return func;
    }
    return {};
}

template <int dim, typename F>
auto optional_vec(std::vector<F> const& vfunc)
{
    // vector of optional functions
    std::vector<fopt<dim>> res;
    for (auto&& i : vfunc) {
        res.push_back(optional_wrapper(i));
    }
    return res;
}

Driver::Driver() = default;

Driver::~Driver() = default;

template <typename Func>
void Driver::run(
    std::vector<std::vector<real>> state,
    const int dim,
    const std::string regime,
    const InitialConditions& init_cond,
    std::function<real(real)> const& scale_factor,
    std::function<real(real)> const& scale_factor_derivative,
    Func const& bsources,
    Func const& hsources,
    Func const& gsources
)
{
    if (dim == 1) {
        hydrostate::simulate<1>(
            state,
            init_cond,
            regime,
            scale_factor,
            scale_factor_derivative,
            optional_vec<1>(bsources),
            optional_vec<1>(hsources),
            optional_vec<1>(gsources)
        );
    }
    else if (dim == 2) {
        hydrostate::simulate<2>(
            state,
            init_cond,
            regime,
            scale_factor,
            scale_factor_derivative,
            optional_vec<2>(bsources),
            optional_vec<2>(hsources),
            optional_vec<2>(gsources)
        );
    }
    else {
        hydrostate::simulate<3>(
            state,
            init_cond,
            regime,
            scale_factor,
            scale_factor_derivative,
            optional_vec<3>(bsources),
            optional_vec<3>(hsources),
            optional_vec<3>(gsources)
        );
    }
}
