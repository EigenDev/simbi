// =============================================================================
// evolver.hpp
//
// high-level simulation dispatcher.
// this file contains the `dispatch_simulation` function, which takes a
// runtime configuration and calls the main simulation driver templated on the
// appropriate physics and geometry.
//
// usage:
//   // called internally by driver::run_simulation
//   hydrostate::dispatch_simulation(config, ...);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "utility/config_dict.hpp"

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
    void dispatch_simulation(
        config_dict_t&                   init,
        py::iterator                     prim_gen,
        py::list                         staggered_bfields,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    );
} // namespace simbi::hydrostate
