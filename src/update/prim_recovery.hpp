#ifndef PRIM_RECOVERY_HPP
#define PRIM_RECOVERY_HPP

#include "compat.hpp"
#include "compute/numerics.hpp"
#include "functional/monad/maybe.hpp"
#include "physics/hydro/conversion.hpp"

namespace simbi::hydro {

    /**
     *
     */
    template <typename Executor, typename ConsField, typename PrimField>
    void recover_primitives(Executor& exec, PrimField& prim, const ConsField& cons, real gamma)
    {
        prim = cons.map(numerics::to_primitive_t{gamma}).with(exec);
    }
} // namespace simbi::hydro
#endif
