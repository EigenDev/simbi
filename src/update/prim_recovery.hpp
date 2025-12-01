#ifndef PRIM_RECOVERY_HPP
#define PRIM_RECOVERY_HPP

#include "compat.hpp"
#include "functional/monad/maybe.hpp"
#include "physics/hydro/conversion.hpp"

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
        using prim_t = PrimField::value_type;
        using cons_t = typename prim_t::counterpart_t;
        prim         = cons.map(
                       [gamma] DEV(const cons_t& c) -> maybe_t<prim_t> {
                           return to_primitive(c, gamma);
                       }
        ).with(exec);
    }
}   // namespace simbi::hydro
#endif
