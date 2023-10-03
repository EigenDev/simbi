#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "common/hydro_structs.hpp"

namespace simbi
{
    struct Driver
    {
        Driver();
        ~Driver();

        std::vector<std::vector<real>> run(
            std::vector<std::vector<real>> state,
            int dim, 
            std::string regime, 
            InitialConditions &init_cond
        );
    };

} // namespace simbi


#endif