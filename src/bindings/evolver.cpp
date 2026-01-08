#include "evolver.hpp"

#include "compat.hpp"
#include "containers/vector.hpp"
#include "context/evolution.hpp"
#include "dispatch.hpp"
#include "ecs/systems.hpp"
#include "utility/config_dict.hpp"

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
        config_dict_t&                   init,
        py::iterator                     prim_gen,
        py::list                         bstagg,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {
        const auto dims   = init.at("dimensionality").get<std::uint64_t>();
        const bool is_mhd = init.at("is_mhd").get<bool>();

        // prepare bfield iters
        vector_t<py::iterator, 3> bfield_gens = {};
        if (is_mhd) {
            for (std::uint64_t idx = 0; idx < dims; ++idx) {
                if (idx < bstagg.size()) {
                    // since we are doing array index, we need the logical
                    // offset index. i.e., idx=0 -> x-dir -> dir=2
                    auto bn_gen      = bstagg[idx].cast<py::iterator>();
                    auto dir         = dims - idx - 1;
                    bfield_gens[dir] = bn_gen;
                }
            }
        }

        dispatch::with_hydro_state(
            init,
            prim_gen,
            bfield_gens,
            scale_factor,
            scale_factor_derivative,
            [](auto& sim, const auto& ops) {
                // rev up those fryers
                auto evo_state = evolution::initialize(sim, "Cool Simulation [TM]");
                auto pipeline  = evolution::hydro_pipeline_t{sim, ops};
                pipeline.configure();

                evolution::run(sim, [&](auto&) { pipeline.step_all(); }, evo_state);
            }
        );
    }
} // namespace simbi::hydrostate
