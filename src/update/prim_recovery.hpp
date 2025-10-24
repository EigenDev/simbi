#ifndef PRIM_RECOVERY_HPP
#define PRIM_RECOVERY_HPP

#include "compat.hpp"
#include "physics/hydro/conversion.hpp"

#include <cstdint>
namespace simbi::hydro {
    /**
     *
     */
    template <typename HydroState>
    void recover_primitives(HydroState& state)
    {
        const auto gamma = state.metadata.gamma;

        state.prim = state.cons.map([gamma] DEV(const auto& cons) {
            return to_primitive(cons, gamma);
        });
    }

    template <typename SimState>
    void recover_primitives(SimState& sim, std::uint64_t level_id)
    {
        auto& prim       = sim.hydro(level_id).prim;
        const auto& cons = sim.hydro(level_id).cons;
        const auto gamma = sim.metadata().gamma;
        prim             = cons.map([gamma] DEV(const auto& cons) {
            return to_primitive(cons, gamma);
        });
    }
}   // namespace simbi::hydro
#endif
