#ifndef SIMBI_HYDRO_HLLE_HPP
#define SIMBI_HYDRO_HLLE_HPP

// HLLE is the same for all regimes, so it can be defined here
#include "config.hpp"
#include "core/containers/vector.hpp"
#include "core/memory/values/value_concepts.hpp"
#include "core/utility/enums.hpp"
#include "physics/em/electromagnetism.hpp"
#include "physics/hydro/physics.hpp"
#include "physics/hydro/wave_speeds.hpp"

namespace simbi::hydro {
    using namespace simbi::em;
    template <is_hydro_primitive_c primitive_t>
    DEV auto hlle_flux(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real vface,
        real gamma,
        ShockWaveLimiter = ShockWaveLimiter::NONE
    )
    {
        const auto uR       = to_conserved(primL, gamma);
        const auto uL       = to_conserved(primR, gamma);
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
                auto wsfac = 1.0 / (sR - sL);
                auto f_hll = (fL * sR - fR * sL + (uR - uL) * sR * sL) * wsfac;
                auto u_hll = (uR * sR - uL * sL - fR + fL) * wsfac;
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

        if constexpr (is_mhd_primitive_c<primitive_t>) {
            // move the electric fields in the flux variable
            // this is a hack to avoid changing the flux type.
            // the electric field is just -nhat x F_B
            net_flux = shift_electric_field(std::move(net_flux), nhat);
        }

        return net_flux;
    }
}   // namespace simbi::hydro

#endif
