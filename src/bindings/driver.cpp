#include "driver.hpp"
#include "compat.hpp"
#include "config_converter.hpp"
#include "evolver.hpp"

#include <cassert>
#include <cstdlib>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>

namespace simbi::driver {
    void run_simulation(
        py::iterator prim_gen,
        py::list staggered_bfields,
        py::dict sim_info,
        py::function a_func,
        py::function adot_func
    )
    {
        // read runtime hints from environment variable
        bool omp_flag_set = []() {
            if (const char* env_p = std::getenv("USE_OMP")) {
                std::string val(env_p);
                if (val == "1" || val == "true" || val == "TRUE") {
                    return true;
                }
            }
            return false;
        }();
        simbi::global::use_omp = omp_flag_set;

        // convert Python dict to config_dict_t
        auto config_dict = dict_to_config(sim_info);

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
            config_dict,
            prim_gen,
            staggered_bfields,
            scale_factor,
            scale_factor_derivative
        );
    }
}   // namespace simbi::driver
