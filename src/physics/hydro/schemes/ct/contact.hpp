#ifndef CONTACT_HPP
#define CONTACT_HPP
#include "build_options.hpp"

namespace simbi {
    namespace ct {
        // the Constrained Transport "Contact" scheme
        // described in section 3.2, Eqn. (51)
        // of Gardiner & Stone (2005)
        struct CTContact {
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
                // j + 1/4
                const real de_dqjL = [&] {
                    if (fw.dens() > 0.0) {
                        return static_cast<real>(2.0) * (es - esw);
                    }
                    else if (fw.dens() < 0.0) {
                        return static_cast<real>(2.0) * (en - enw);
                    }
                    return es - esw + en - enw;
                }();

                // j + 3/4
                const real de_dqjR = [&] {
                    if (fe.dens() > 0.0) {
                        return static_cast<real>(2.0) * (ese - es);
                    }
                    else if (fe.dens() < 0.0) {
                        return static_cast<real>(2.0) * (ene - en);
                    }
                    return ese - es + ene - en;
                }();

                // k + 1/4
                const real de_dqkL = [&] {
                    if (fs.dens() > 0.0) {
                        return static_cast<real>(2.0) * (ew - esw);
                    }
                    else if (fs.dens() < 0.0) {
                        return static_cast<real>(2.0) * (ee - ese);
                    }
                    return ew - esw + ee - ese;
                }();

                // k + 3/4
                const real de_dqkR = [&] {
                    if (fn.dens() > 0.0) {
                        return static_cast<real>(2.0) * (enw - ew);
                    }
                    else if (fn.dens() < 0.0) {
                        return static_cast<real>(2.0) * (ene - ee);
                    }
                    return enw - ew + ene - ee;
                }();

                return (
                    eavg + one_eighth * (de_dqjL - de_dqjR + de_dqkL - de_dqkR)
                );
            }
        };
    }   // namespace ct

}   // namespace simbi

#endif