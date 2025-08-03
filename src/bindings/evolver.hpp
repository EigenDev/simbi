#ifndef BINDINGS_STATE_HPP
#define BINDINGS_STATE_HPP

#include "config.hpp"

#include <functional>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace simbi {
    struct initial_conditions_t;
};

namespace simbi::hydrostate {
    // convenience dispatcher based on runtime parameters
    void dispatch_simulation(
        py::array_t<real, py::array::c_style> cons_array,
        py::array_t<real, py::array::c_style> prim_array,
        py::list staggered_bfields,
        initial_conditions_t& init,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    );
}   // namespace simbi::hydrostate

#endif
