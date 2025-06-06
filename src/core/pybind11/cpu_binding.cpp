#include "driver.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cpu_ext, m)
{
    m.doc() =
        "SIMBI - Special Relativistic Magnetohydrodynamics Simulation Code";

    // expose cpu-specific simulation driver
    m.def(
        "run_simulation",
        &simbi::driver::run_simulation,
        py::arg("cons_array"),
        py::arg("prim_array"),
        py::arg("staggered_bfields"),
        py::arg("sim_info"),
        py::arg("a"),
        py::arg("adot"),
        "Run a SIMBI simulation with the provided state and parameters"
    );
}
