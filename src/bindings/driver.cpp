#include "driver.hpp"
#include "compat.hpp"
#include "config_converter.hpp"
#include "evolver.hpp"
#include "utility/init_conditions.hpp"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace simbi::driver {
    void run_simulation(
        py::iterator prim_gen,
        py::list staggered_bfields,
        py::dict sim_info,
        py::function a_func,
        py::function adot_func
    )
    {
        // convert Python dict to config_dict_t
        auto config_dict = dict_to_config(sim_info);
        auto init_cond   = initial_conditions_t::create(config_dict);

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
            prim_gen,
            staggered_bfields,
            init_cond,
            scale_factor,
            scale_factor_derivative
        );
    }
}   // namespace simbi::driver
