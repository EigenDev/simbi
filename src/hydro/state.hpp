#ifndef STATE_HPP
#define STATE_HPP
#include "base.hpp"

namespace simbi
{
    namespace hydrostate
    {
        std::unique_ptr<HydroBase> create(
            std::vector<std::vector<real>> &state,
            const std::string regime, 
            const int dim,
            const InitialConditions &init_cond);
    } // namespace hydrostate
    
} // namespace simbi



#endif