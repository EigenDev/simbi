/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       driver.hpp
 * @brief      the key driver for any simulation in any regime, dimensions
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "build_options.hpp"          // for real
#include "common/hydro_structs.hpp"   // for InitialConditions
#include <functional>                 // for function
#include <string>                     // for string
#include <vector>                     // for vector

namespace simbi {
    struct Driver {
        Driver();
        ~Driver();

        void
        run(std::vector<std::vector<real>> state,
            const int dim,
            const std::string regime,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative);
    };

}   // namespace simbi

#include "driver.ipp"
#endif