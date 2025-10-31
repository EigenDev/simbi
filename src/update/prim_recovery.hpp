#ifndef PRIM_RECOVERY_HPP
#define PRIM_RECOVERY_HPP

#include "compat.hpp"
#include "physics/hydro/conversion.hpp"

namespace simbi::hydro {
    /**
     *
     */
    template <typename ConsField, typename PrimField>
    void recover_primitives(PrimField& prim, const ConsField& cons, real gamma)
    {
        prim = cons.map([gamma] DEV(const auto& c) {
            return to_primitive(c, gamma);
        });
    }
}   // namespace simbi::hydro
#endif
