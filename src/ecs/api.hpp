#ifndef ECS_INITIALIZATION_API_HPP
#define ECS_INITIALIZATION_API_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "compute/numerics.hpp"
#include "containers/vector.hpp"
#include "grid/amr/prolongation.hpp"
#include "physics/hydro/conversion.hpp"
#include "physics/hydro/physics.hpp"

#include <cstdint>

namespace simbi::ecs::initialization {
    using namespace simbi::compute;
    using namespace simbi::grid;

    // -------------------------------------------------------------------------
    // core initialization driver
    // -------------------------------------------------------------------------
    template <typename Sim, typename Exec, typename InitFunc>
    void initialize_state(Sim& sim, Exec& exec, InitFunc&& func)
    {
        // initialize level 0 (the root)
        auto&       l0_hydro = sim.hydro(0);
        const auto& l0_mesh  = sim.mesh(0);

        // apply functor to primitive variables
        // func signature: coord_t -> primitive_t
        l0_hydro.prim = computation(l0_mesh.geometry, func).with(exec);

        // convert to conserved variables
        const real gamma = sim.metadata().gamma;
        l0_hydro.cons    = l0_hydro.prim.map(numerics::to_conserved_t{gamma}).with(exec);

        // prolongate to fine levels (if amr enabled)
        if (sim.has_refinement()) {
            for (std::uint64_t lvl = 1; lvl < sim.num_levels(); ++lvl) {
                auto& coarse = sim.hydro(lvl - 1);
                auto& fine   = sim.hydro(lvl);
                auto& info   = sim.level_info(lvl);

                // construct refinement ratio vector
                iarray<Sim::rank> ratio;
                ratio.fill(info.refinement_ratio);

                // use amr api to create prolongation computation
                // we prolongate conserved variables to maintain conservation
                // laws
                auto prolong_op = amr::prolong(coarse.cons.view(), ratio);

                // execute fill on the fine domain
                // the computation engine handles the coordinate mapping (fine
                // -> coarse)
                fine.cons = compute::computation(fine.cons.domain(), prolong_op).with(exec);

                // recover primitives on fine level
                fine.prim = fine.cons.enum_map(numerics::to_primitive_t{gamma}).with(exec);
            }
        }

        // mhd handling (optional)
        if constexpr (Sim::is_mhd) {
            // [future]: initialize staggered magnetic fields here
            // typically involves computing vector potential A on edges
            // then taking curl to get B on faces.
        }
    }

} // namespace simbi::ecs::initialization

#endif // ECS_INITIALIZATION_API_HPP
