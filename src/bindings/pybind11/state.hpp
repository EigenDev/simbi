#ifndef STATE_HPP
#define STATE_HPP

#include "bindings/pybind11/dispatch.hpp"
#include "config.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "physics/update.hpp"
#include <cstdint>
#include <functional>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace simbi {
    struct InitialConditions;
};

namespace simbi::hydrostate {
    // primary entry pointfor simulation with NumPy array
    template <std::int64_t Dims>
    void simulate_from_numpy(
        py::array_t<real, py::array::c_style> cons_array,
        py::array_t<real, py::array::c_style> prim_array,
        py::list staggered_bfields,
        InitialConditions& init,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {

        // Get buffer info for conserved and primitive arrays
        py::buffer_info cons_buffer = cons_array.request();
        py::buffer_info prim_buffer = prim_array.request();

        // Prepare bfield pointers
        vector_t<void*, 3> bfield_ptrs = {};
        if (init.is_mhd) {
            for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                if (dir < staggered_bfields.size()) {
                    auto bfield_array =
                        staggered_bfields[dir].cast<py::array_t<real>>();
                    py::buffer_info bfield_buffer = bfield_array.request();
                    bfield_ptrs[dir]              = bfield_buffer.ptr;
                }
                else {
                    bfield_ptrs[dir] = nullptr;
                }
            }
        }

        dispatch::with_hydro_state(
            cons_buffer.ptr,
            prim_buffer.ptr,
            bfield_ptrs,
            scale_factor,
            scale_factor_derivative,
            init,
            [=](auto& state) {
                // Update the state based on the initial conditions
                hydro::run_simulation(state);
            }
        );
    }

    // convenience dispatcher based on runtime parameters
    void dispatch_simulation(
        py::array_t<real, py::array::c_style> cons_array,
        py::array_t<real, py::array::c_style> prim_array,
        py::list staggered_bfields,
        InitialConditions& init,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    );
}   // namespace simbi::hydrostate

#endif
