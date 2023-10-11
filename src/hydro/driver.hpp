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
            const int dim, 
            const std::string regime, 
            const InitialConditions &init_cond
        );
    };

} // namespace simbi


#endif