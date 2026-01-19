// =============================================================================
// driver.hpp
//
// main entry point for running a simulation from python.
// the `run_simulation` function takes python objects (generators, dicts,
// functions) and orchestrates the c++ simulation lifecycle.
//
// usage (from python):
//   import simbi.cpu_ext as simbi_cpu
//   simbi_cpu.run_simulation(...)
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
        py::list     staggered_bfields,
        py::dict     sim_info,
        py::function a_func,
        py::function adot_func
    );
} // namespace simbi::driver
