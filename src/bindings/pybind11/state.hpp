#ifndef STATE_HPP
#define STATE_HPP

#include "bindings/pybind11/dispatch.hpp"
#include "config.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "physics/update.hpp"
#include <functional>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace simbi {

    struct InitialConditions;
    namespace hydrostate {

        // primary entry postd::int64_t for simulation with NumPy array
        template <std::int64_t Dims, Regime R, Geometry G = Geometry::CARTESIAN>
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
                init,
                [=](auto& state) {
                    // Update the state based on the initial conditions
                    hydro::compute_fluxes(state);
                    hydro::update_state(state);
                }
            );
        }

        // convenience dispatcher based on runtime parameters
        inline void dispatch_simulation(
            py::array_t<real, py::array::c_style> cons_array,
            py::array_t<real, py::array::c_style> prim_array,
            py::list staggered_bfields,
            const std::int64_t dims,
            const std::string& regime_str,
            InitialConditions& init,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            // dispatch based on dimension and regime
            if (dims == 1) {
                if (regime_str == "classical") {
                    simulate_from_numpy<1, Regime::NEWTONIAN>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else {
                    simulate_from_numpy<1, Regime::SRHD>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
            }
            else if (dims == 2) {
                if (regime_str == "classical") {
                    simulate_from_numpy<2, Regime::NEWTONIAN>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else {
                    simulate_from_numpy<2, Regime::SRHD>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
            }
            else {   // dims == 3
                if (regime_str == "classical") {
                    simulate_from_numpy<3, Regime::NEWTONIAN>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else if (regime_str == "srhd") {
                    simulate_from_numpy<3, Regime::SRHD>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
                else {
                    simulate_from_numpy<3, Regime::RMHD>(
                        cons_array,
                        prim_array,
                        staggered_bfields,
                        init,
                        scale_factor,
                        scale_factor_derivative
                    );
                }
            }
        }
    }   // namespace hydrostate
}   // namespace simbi

#endif
