#ifndef DRIVER_HPP
#define DRIVER_HPP
#include "common/hydro_structs.hpp"

namespace simbi
{
    struct Driver
    {    
        Driver();
        ~Driver();

        template<typename Func>
        void run(
            std::vector<std::vector<real>> state,
            const int dim, 
            const std::string regime, 
            const InitialConditions &init_cond,
            std::function<real(real)> const &scale_factor,
            std::function<real(real)> const &scale_factor_derivative,
            Func const &density_lambda = nullptr,
            Func const &mom1_lambda = nullptr,
            Func const &mom2_lambda = nullptr,
            Func const &mom3_lambda = nullptr,
            Func const &enrg_lambda = nullptr
        );
    };

} // namespace simbi

#include "driver.tpp"
#endif