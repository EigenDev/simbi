#include "evolver.hpp"
#include "compute/context.hpp"
#include "compute/functional/monad/computation.hpp"
#include "compute/math/cfd.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "dispatch.hpp"
#include "update/adaptive_timestep.hpp"

#include <cstdint>
#include <functional>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace simbi::hydrostate {
    // convenience dispatcher based on runtime parameters
    void dispatch_simulation(
        py::array_t<real, py::array::c_style> cons_array,
        py::array_t<real, py::array::c_style> prim_array,
        py::list staggered_bfields,
        initial_conditions_t& init,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {
        // get buffer info for conserved and primitive arrays
        py::buffer_info cons_buffer = cons_array.request();
        py::buffer_info prim_buffer = prim_array.request();

        const auto dims = init.dimensionality;

        // prepare bfield pointers
        vector_t<void*, 3> bfield_ptrs = {};
        if (init.is_mhd) {
            for (std::uint64_t dir = 0; dir < dims; ++dir) {
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
            [](auto& state, auto& ops, auto& mesh) {
                // initialize timestep
                boundary::apply_boundary_conditions(state, mesh);
                hydro::recover_primitives(state);
                update_timestep(state, mesh);

                // rev up those engines
                with_simulation(state, mesh, [&](auto& sim) {
                    sim.evolve([&](auto& ctx) {
                        compute(ctx)
                            .then([&](auto& s) { cfd::step(s, mesh, ops); })
                            .run();
                    });
                });
            }
        );
    }
}   // namespace simbi::hydrostate
