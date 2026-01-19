// =============================================================================
// hlle.hpp
//
// the hlle approximate riemann solver.
// provides a generic implementation of the hlle (harten-lax-van leer-eymard)
// approximate riemann solver, which is applicable across all supported physics
// regimes (newtonian, srhd, rmhd).
//
// usage:
//   auto flux = hydro::hlle_flux(primL, primR, nhat, vface, gamma);
// =============================================================================
#pragma once

// HLLE is the same for all regimes, so it can be defined here
#include "base/concepts.hpp"
#include "build_config.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "physics/hydro/physics.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "utility/enums.hpp"

#include <iostream>

namespace simbi::hydro {
    using namespace simbi::em;
    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto hlle_flux(
        const primitive_t&                      primL,
        const primitive_t&                      primR,
        const unit_vector_t<primitive_t::rank>& nhat,
        real                                    vface,
        real                                    gamma,
        shockwave_limiter_t = shockwave_limiter_t::NONE
    )
    {
        const auto uL       = to_conserved(primL, gamma);
        const auto uR       = to_conserved(primR, gamma);
        const auto fL       = to_flux(primL, nhat, gamma);
        const auto fR       = to_flux(primR, nhat, gamma);
        const auto [sL, sR] = extremal_speeds(primL, primR, nhat, gamma);

        auto net_flux = [&]() {
            if (sL >= vface) {
                // left state is supersonic
                return fL - uL * vface;
            }
            else if (sR <= vface) {
                // right state is supersonic
                return fR - uR * vface;
            }
            else {
                // intermediate state
                auto f_hll = (fL * sR - fR * sL + (uR - uL) * sR * sL) / (sR - sL);
                auto u_hll = (uR * sR - uL * sL - fR + fL) / (sR - sL);
                return f_hll - u_hll * vface;
            }
        }();

        // upwind the scalar concentration
        if (net_flux.den < 0.0) {
            net_flux.chi = primR.chi * net_flux.den;
        }
        else {
            net_flux.chi = primL.chi * net_flux.den;
        }

        return net_flux;
    }
} // namespace simbi::hydro
