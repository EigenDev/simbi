#ifndef STATE_HPP
#define STATE_HPP
#include <variant>
#include <any>
#include "base.hpp"
#include "common/enums.hpp"

namespace simbi
{
    namespace hydrostate
    {   
        // template<typename T, HydroRegime regime, int dim>
        std::unique_ptr<HydroBase> create(
            const std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string &regime,
            const int dim
        );
    } // namespace hydrostate    
} // namespace simbi

#endif