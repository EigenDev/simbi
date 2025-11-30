#ifndef PRIM_RECOVERY_HPP
#define PRIM_RECOVERY_HPP

#include "compat.hpp"
#include "physics/hydro/conversion.hpp"

#include <iostream>
namespace simbi::hydro {
    /**
     *
     */
    template <typename Executor, typename ConsField, typename PrimField>
    void recover_primitives(
        Executor& exec,
        PrimField& prim,
        const ConsField& cons,
        real gamma
    )
    {
        prim = cons.map(
                       [gamma] DEV(const auto& c) {
                           return to_primitive(c, gamma);
                       }
        ).with(exec);
    }
}   // namespace simbi::hydro
#endif
