#ifndef ZERO_HPP
#define ZERO_HPP

#include "build_options.hpp"   // for DUAL

namespace simbi {
    namespace ct {
        // the Constrained Transport "Zero" scheme
        // described in section 3.2, Eqn. (40)
        // of Gardiner & Stone (2005)
        struct CTZero {
            template <typename Flux>
            static DUAL real compute_emf(
                const Flux& fw,
                const Flux& fe,
                const Flux& fs,
                const Flux& fn,
                const real esw,
                const real ese,
                const real enw,
                const real ene,
                const luint nhat
            )
            {
                // south, north, east, west electric fields
                // from Riemann fluxes
                const auto es = fs.ecomponent(nhat);
                const auto en = fn.ecomponent(nhat);
                const auto ew = fw.ecomponent(nhat);
                const auto ee = fe.ecomponent(nhat);
                return static_cast<real>(0.50) * (es + en + ew + ee) -
                       static_cast<real>(0.25) * (esw + enw + ese + ene);
            }
        };
    }   // namespace ct

}   // namespace simbi

#endif