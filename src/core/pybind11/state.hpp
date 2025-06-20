#ifndef STATE_HPP
#define STATE_HPP

#include "config.hpp"
#include "core/containers/collapsable.hpp"
#include "core/containers/ndarray.hpp"
#include "core/state/hydro_state.hpp"
#include "core/types/alias/alias.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "physics/hydro/update.hpp"
#include <functional>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace simbi {
    using namespace containers;

    struct InitialConditions;
    namespace hydrostate {

        // primary entry point for simulation with NumPy array
        template <int Dims, Regime R>
        void simulate_from_numpy(
            py::array_t<real, py::array::c_style> cons_array,
            py::array_t<real, py::array::c_style> prim_array,
            py::list staggered_bfields,
            InitialConditions& init,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            // get buffer info
            py::buffer_info cons_buffer = cons_array.request();
            py::buffer_info prim_buffer = prim_array.request();

            collapsable_t<size_type, Dims> ushape = {init.nz, init.ny, init.nx};

            // create the ndarray wrapping the numpy data
            ndarray_t<real> conserved_array(
                static_cast<real*>(cons_buffer.ptr),
                cons_array.size(),
                false   // don't take ownership - numpy owns the memory
            );
            ndarray_t<real> primitives(
                static_cast<real*>(prim_buffer.ptr),
                prim_array.size(),
                false
            );

            auto state = [&]() {
                if constexpr (R == Regime::RMHD) {
                    // handle staggered B-fields
                    array_t<ndarray_t<real>, Dims> bfields;
                    for (size_type ii = 0; ii < Dims; ++ii) {
                        auto bfield_array =
                            staggered_bfields[ii].cast<py::array_t<real>>();
                        py::buffer_info bfield_buffer = bfield_array.request();
                        bfields[ii]                   = ndarray_t<real>(
                            static_cast<real*>(bfield_buffer.ptr),
                            bfield_array.size(),
                            false
                        );
                    }
                    return state::hydro_state_t<R, Dims>::from_init(
                        std::move(conserved_array),
                        std::move(primitives),
                        std::move(bfields),
                        std::move(init)
                    );
                }
                else {
                    return state::hydro_state_t<R, Dims>::from_init(
                        std::move(conserved_array),
                        std::move(primitives),
                        {},
                        std::move(init)
                    );
                }
            }();

            hydro::advance_state(state, 0.01);
        }

        // convenience dispatcher based on runtime parameters
        inline void dispatch_simulation(
            py::array_t<real, py::array::c_style> cons_array,
            py::array_t<real, py::array::c_style> prim_array,
            py::list staggered_bfields,
            const int dims,
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
