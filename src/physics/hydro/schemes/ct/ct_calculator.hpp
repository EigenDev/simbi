#ifndef CT_CALCULATOR_HPP
#define CT_CALCULATOR_HPP

#include "build_options.hpp"   // for DUAL
#include "contact.hpp"         // for CTContact
#include "uct_ct.hpp"          // for CTMdZ
#include "zero.hpp"            // for CTZero

namespace simbi {
    namespace scheme {
        template <typename CTScheme>
        class EMFCalculator
        {
          public:
            template <Plane P, Corner C>
            static DUAL real calc_edge_emf(
                const auto& fw,
                const auto& fe,
                const auto& fs,
                const auto& fn,
                const auto* prims,
                const luint nhat,
                const real bw = 0.0,
                const real be = 0.0,
                const real bs = 0.0,
                const real bn = 0.0
            )
            {
                if constexpr (std::is_same_v<CTScheme, CTMdZ>) {
                    return CTScheme::compute_emf(
                        fw,
                        fe,
                        fs,
                        fn,
                        bw,
                        be,
                        bs,
                        bn,
                        nhat
                    );
                }
                else {
                    const auto [esw, ese, enw, ene] =
                        compute_mean_efields(prims, fw, fe, fs, fn, nhat);
                    return CTScheme::compute_emf(
                        fw,
                        fe,
                        fs,
                        fn,
                        esw,
                        ese,
                        enw,
                        ene,
                        nhat
                    );
                }
            }

          private:
            static DUAL auto compute_mean_efields(...);
        };

    }   // namespace scheme

}   // namespace simbi

#endif