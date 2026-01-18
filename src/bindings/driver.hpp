// =============================================================================
// driver.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace simbi::driver {
    void run_simulation(
        py::iterator prim_gen,
        py::list staggered_bfields,
        py::dict sim_info,
        py::function a_func,
        py::function adot_func
    );
}   // namespace simbi::driver

