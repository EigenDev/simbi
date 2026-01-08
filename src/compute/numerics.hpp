#ifndef COMPUTE_NUMERICS_HPP
#define COMPUTE_NUMERICS_HPP

// =============================================================================
// numerics.hpp
//
// physics and numerics-specific functors for use with computation_t.
// these functors encode domain knowledge (hydro, mhd, eos) unlike the
// pure mathematical operations in functional/fp.hpp.
//
// usage:
//   cons_field = prim_field.map(to_conserved_t{gamma}).with(exec);
//   prim_field = cons_field.map(to_primitive_t{gamma}).with(exec);
// =============================================================================

#include "../compat.hpp"
#include "../containers/state_ops.hpp"
#include "../physics/hydro/conversion.hpp"
#include "../physics/hydro/physics.hpp"

namespace simbi::numerics {

    // =========================================================================
    // conserved <-> primitive conversion functors
    // =========================================================================

    struct to_conserved_t
    {
        real gamma;

        template <typename Prim>
        constexpr DEV typename Prim::counterpart_t operator()(const Prim& prim) const
        {
            return hydro::to_conserved(prim, gamma);
        }
    };

    struct to_primitive_t
    {
        real gamma;

        template <typename Cons>
        constexpr DEV typename Cons::counterpart_t operator()(const Cons& cons) const
        {
            return hydro::to_primitive(cons, gamma);
        }
    };

    // =========================================================================
    // time-stepping functors
    // =========================================================================

    // forward euler step: u^{n+1} = u^n + dt * L(u^n)
    struct euler_step_t
    {
        real dt;

        template <typename State>
        constexpr DEV State operator()(const State& u, const State& dudt) const
        {
            using namespace simbi::structs;
            return u | add_gas(dudt * dt);
        }
    };

    // linear time interpolation: u = (1-alpha)*u_n + alpha*u_curr
    // used for temporal refinement in amr ghost cell filling
    struct time_interpolate_t
    {
        real alpha;

        template <typename State>
        constexpr DEV State operator()(const State& u_n, const State& u_curr) const
        {
            using namespace simbi::structs;
            return u_n | scale_gas(1.0 - alpha) | add_gas(u_curr | scale_gas(alpha));
        }
    };

} // namespace simbi::numerics

#endif // COMPUTE_NUMERICS_HPP
