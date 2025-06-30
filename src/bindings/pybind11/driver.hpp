#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "config.hpp"
#include "state.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace simbi {
    namespace driver {
        // main driver function that takes NumPy array directly
        void run_simulation(
            py::array_t<real, py::array::c_style> cons_array,
            py::array_t<real, py::array::c_style> prim_array,
            py::list staggered_bfields,
            py::dict sim_info,
            py::function a_func,
            py::function adot_func
        );
    }   // namespace driver
}   // namespace simbi
#endif
