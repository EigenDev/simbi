/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            driver.hpp
 *  * @brief           driver for all simulations
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "build_options.hpp"                        // for real
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "state.hpp"
#include <functional>   // for function
#include <string>       // for string
#include <vector>       // for vector

using namespace simbi::hydrostate;

namespace simbi {
    struct Driver {
        Driver()  = default;
        ~Driver() = default;

        void
        run(std::vector<std::vector<real>> state,
            const int dim,
            const std::string regime,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative)
        {
            if (dim == 1) {
                if (regime == "classical") {
                    simulate<1, HydroRegime::Newtonian>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else if (regime == "srhd") {
                    simulate<1, HydroRegime::SRHD>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else {
                    simulate<1, HydroRegime::RMHD>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
            }
            else if (dim == 2) {
                if (regime == "classical") {
                    simulate<2, HydroRegime::Newtonian>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else if (regime == "srhd") {
                    simulate<2, HydroRegime::SRHD>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else {
                    simulate<2, HydroRegime::RMHD>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
            }
            else {
                if (regime == "classical") {
                    simulate<3, HydroRegime::Newtonian>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else if (regime == "srhd") {
                    simulate<3, HydroRegime::SRHD>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else {
                    simulate<3, HydroRegime::RMHD>(
                        state,
                        init_cond,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
            }
        };
    };
}   // namespace simbi
#endif