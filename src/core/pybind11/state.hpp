#ifndef STATE_HPP
#define STATE_HPP

#include "build_options.hpp"
#include "core/types/alias/alias.hpp"
#include "core/types/containers/collapsable.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/monad/maybe.hpp"
#include "core/types/utility/init_conditions.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include <functional>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace simbi {
    struct InitialConditions;
    namespace hydrostate {

        // primary template for simulation with ndarray
        template <size_type Dims, Regime R>
        void simulate_pure_hydro(
            ndarray<anyConserved<Dims, R>, Dims>&& cons,
            ndarray<Maybe<anyPrimitive<Dims, R>>, Dims>&& prim,
            InitialConditions& init,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );
        template <size_type Dims, Regime R>
        void simulate_mhd(
            ndarray<anyConserved<Dims, R>, Dims>&& cons,
            ndarray<Maybe<anyPrimitive<Dims, R>>, Dims>&& prim,
            std::vector<ndarray<real, Dims>>&& staggered_bfields,
            InitialConditions& init,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        );

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

            collapsable<Dims> ushape = {init.nz, init.ny, init.nx};

            // create ndarray view of the data
            using conserved_t = anyConserved<Dims, R>;
            using primitive_t = anyPrimitive<Dims, R>;

            // create the ndarray wrapping the numpy data
            ndarray<conserved_t, Dims> conserved_array(
                reinterpret_cast<conserved_t*>(cons_buffer.ptr),
                ushape,
                false   // don't take ownership - numpy owns the memory
            );
            ndarray<Maybe<primitive_t>, Dims> maybe_primitives(
                reinterpret_cast<primitive_t*>(prim_buffer.ptr),
                ushape
            );

            // we now release the primitive array
            // since we had to make copies of it
            // to wrap the prims in the maybe monad
            prim_array = py::array_t<real>();

            if constexpr (R == Regime::RMHD) {
                std::vector<ndarray<real, Dims>> stag_fields_list;
                size_type ii            = 0;
                const auto [xa, ya, za] = init.active_zones();
                for (const auto& stag_field : staggered_bfields) {
                    // convert each staggered field to ndarray
                    py::buffer_info stag_buffer =
                        stag_field.cast<py::array_t<real, py::array::c_style>>()
                            .request();
                    collapsable<Dims> ushape_stag = {
                      za + 1 * (ii == 2) + 2 * (ii != 2),
                      ya + 1 * (ii == 1) + 2 * (ii != 1),
                      xa + 1 * (ii == 0) + 2 * (ii != 0)
                    };
                    stag_fields_list.emplace_back(
                        ndarray<real, Dims>(
                            reinterpret_cast<real*>(stag_buffer.ptr),
                            ushape_stag,
                            false   // don't take ownership
                        )
                    );
                    ii++;
                }
                // mhd simulation
                simulate_mhd<Dims, R>(
                    std::move(conserved_array),
                    std::move(maybe_primitives),
                    std::move(stag_fields_list),
                    init,
                    scale_factor,
                    scale_factor_derivative
                );
            }
            else {
                // pure hydro simulation
                simulate_pure_hydro<Dims, R>(
                    std::move(conserved_array),
                    std::move(maybe_primitives),
                    init,
                    scale_factor,
                    scale_factor_derivative
                );
            }
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
