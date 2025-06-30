#include "state.hpp"
#include "config.hpp"
#include "core/utility/init_conditions.hpp"
#include <cstdint>
#include <functional>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace simbi::hydrostate {
    // convenience dispatcher based on runtime parameters
    void dispatch_simulation(
        py::array_t<real, py::array::c_style> cons_array,
        py::array_t<real, py::array::c_style> prim_array,
        py::list staggered_bfields,
        InitialConditions& init,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {
        // dispatch based on dimension and regime
        if (init.dimensionality == 1) {
            simulate_from_numpy<1>(
                cons_array,
                prim_array,
                staggered_bfields,
                init,
                scale_factor,
                scale_factor_derivative
            );
        }
        else if (init.dimensionality == 2) {
            simulate_from_numpy<2>(
                cons_array,
                prim_array,
                staggered_bfields,
                init,
                scale_factor,
                scale_factor_derivative
            );
        }
        else {   // dims == 3
            simulate_from_numpy<3>(
                cons_array,
                prim_array,
                staggered_bfields,
                init,
                scale_factor,
                scale_factor_derivative
            );
        }
    }
}   // namespace simbi::hydrostate
