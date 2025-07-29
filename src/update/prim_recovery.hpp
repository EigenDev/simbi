#ifndef PRIM_RECOVERY_HPP
#define PRIM_RECOVERY_HPP

#include "config.hpp"
#include "physics/hydro/conversion.hpp"

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
}   // namespace simbi::hydro
#endif
