#ifndef CT_CALCULATOR_HPP
#define CT_CALCULATOR_HPP

#include "build_options.hpp"   // for DUAL
#include "contact.hpp"         // for CTContact
#include "stencil.hpp"         // for StencilView
#include "uct_ct.hpp"          // for CTMdZ
#include "zero.hpp"            // for CTZero

namespace simbi {
    namespace scheme {
        template <typename CTScheme>
        class EMFCalculator
        {
          public:
            template <BlockAx B, Plane P, Corner C>
            static DUAL real calc_edge_emf(
                const auto& vertical_flux,
                const auto& horizontal_flux,
                const auto& prims,
                const luint nhat
            )
            {
                using flux_t = decltype(vertical_flux);
                using prim_t = decltype(prims);
                // Create plane-aware stencil
                ct::StencilView<B, P, C, flux_t, prim_t> stencil(
                    vertical_flux,
                    horizontal_flux,
                    prims
                );

                // Get fluxes using plane-specific directions
                const auto fn = stencil.horizontal_flux(Dir::N);
                const auto fs = stencil.horizontal_flux(Dir::S);
                const auto fe = stencil.vertical_flux(Dir::E);
                const auto fw = stencil.vertical_flux(Dir::W);

                // Get corner primitives
                const auto ene = stencil.prim(Dir::NE).ecomponent(nhat);
                const auto enw = stencil.prim(Dir::NW).ecomponent(nhat);
                const auto ese = stencil.prim(Dir::SE).ecomponent(nhat);
                const auto esw = stencil.prim(Dir::SW).ecomponent(nhat);

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
        };

    }   // namespace scheme

}   // namespace simbi

#endif