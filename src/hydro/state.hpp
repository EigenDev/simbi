#ifndef STATE_HPP
#define STATE_HPP
#include <variant>
#include "base.hpp"

namespace simbi
{
    namespace hydrostate
    {
        using hydro_t = std::variant<
            std::unique_ptr<Newtonian<1>>,
            std::unique_ptr<Newtonian<2>>,
            std::unique_ptr<Newtonian<3>>,
            std::unique_ptr<SRHD<1>>,
            std::unique_ptr<SRHD<2>>,
            std::unique_ptr<SRHD<3>>
        >;
        
        hydro_t create(
            std::vector<std::vector<real>> &state,
            const std::string regime, 
            const int dim,
            const InitialConditions &init_cond
        );
    } // namespace hydrostate    
} // namespace simbi

#endif