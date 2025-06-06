#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "build_options.hpp"
#include "config_converter.hpp"
#include "core/types/utility/init_conditions.hpp"
#include "state.hpp"
#include <functional>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// namespace py = pybind11;

namespace simbi {
    namespace driver {
        // main driver function that takes NumPy array directly
        inline void run_simulation(
            py::array_t<real, py::array::c_style> cons_array,
            py::array_t<real, py::array::c_style> prim_array,
            py::dict sim_info,
            py::function a_func,
            py::function adot_func
        )
        {
            // extract parameters from sim_info
            int dims               = py::cast<int>(sim_info["dimensionality"]);
            std::string regime_str = py::cast<std::string>(sim_info["regime"]);
            // convert Python dict to ConfigDict
            ConfigDict config_dict = dict_to_config(sim_info);
            InitialConditions init_cond =
                InitialConditions::create(config_dict);

            // create C++ function wrappers for callbacks
            auto scale_factor = [a_func](real t) -> real {
                py::gil_scoped_acquire gil;
                return a_func(t).cast<real>();
            };

            auto scale_factor_derivative = [adot_func](real t) -> real {
                py::gil_scoped_acquire gil;
                return adot_func(t).cast<real>();
            };

            // dispatch to appropriate simulation
            hydrostate::dispatch_simulation(
                cons_array,
                prim_array,
                dims,
                regime_str,
                init_cond,
                scale_factor,
                scale_factor_derivative
            );
        }
    }   // namespace driver
}   // namespace simbi
#endif
