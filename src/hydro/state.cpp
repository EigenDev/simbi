#include "state.hpp"
#include "srhd.hpp"
#include "newt.hpp"
namespace simbi
{
    namespace hydrostate
    {
        hydro_t create(
            std::vector<std::vector<real>> &state,
            const std::string regime, 
            const int dim,
            const InitialConditions &init_cond)
        {
            if (regime == "relativistic") {
                if (dim == 1) {
                    return std::make_unique<SRHD<1>>(state, init_cond);
                } else if (dim == 2) {
                    return std::make_unique<SRHD<2>>(state, init_cond);
                } else {
                    return std::make_unique<SRHD<3>>(state, init_cond);
                }
            } else {
                if (dim == 1) {
                    return std::make_unique<Newtonian<1>>(state, init_cond);
                } else if (dim == 2) {
                    return std::make_unique<Newtonian<2>>(state, init_cond);
                } else {
                    return std::make_unique<Newtonian<3>>(state, init_cond);
                }
            }
        };
    } // namespace hydrostate
    
} // namespace simbi
