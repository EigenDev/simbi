#ifndef DRIVER_HPP
#define DRIVER_HPP
#include <variant>
#include "common/hydro_structs.hpp"

using namespace std::placeholders;

namespace simbi
{
    struct Driver
    {
        // using func_t = std::variant<
        //     std::function<real(real)>,
        //     std::function<real(real, real)>,
        //     std::function<real(real, real, real)>,
        //     std::nullptr_t
        // >;

        using func_t = std::function<real(real)>;
        
        Driver();
        ~Driver();
        
        void run(
            std::vector<std::vector<real>> state,
            const int dim, 
            const std::string regime, 
            const InitialConditions &init_cond,
            std::function<real(real)> const &scale_factor,
            std::function<real(real)> const &scale_factor_derivative,
            func_t const &density_lambda = nullptr,
            func_t const &mom1_lambda = nullptr,
            func_t const &mom2_lambda = nullptr,
            func_t const &mom3_lambda = nullptr,
            func_t const &enrg_lambda = nullptr
        );
    };

} // namespace simbi

#endif