/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            ct_calculator.hpp
 *  * @brief           CT Calculator for MHD Schemes
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef CT_CALCULATOR_HPP
#define CT_CALCULATOR_HPP

#include "config.hpp"               // for DUAL
#include "core/utility/enums.hpp"   // for Dir, BlockAx, Plane, Corner
#include "stencil.hpp"              // for StencilView

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
                const auto& vertical_bfield,
                const auto& horizontal_bfield,
                const auto& prims,
                const luint nhat
            )
            {
                using flux_t = decltype(vertical_flux);
                using prim_t = decltype(prims);
                // Create plane-aware stencil
                ct::StencilView<B, P, C, flux_t, prim_t> flux_stencil(
                    vertical_flux,
                    horizontal_flux,
                    prims
                );

                // Get fluxes using plane-specific directions
                const auto fn = flux_stencil.horizontal_field(Dir::N);
                const auto fs = flux_stencil.horizontal_field(Dir::S);
                const auto fe = flux_stencil.vertical_field(Dir::E);
                const auto fw = flux_stencil.vertical_field(Dir::W);

                if constexpr (comp_ct_type == CTAlgo::MdZ) {
                    using bfield_t = decltype(vertical_bfield);
                    ct::StencilView<B, P, C, bfield_t, prim_t> bfield_stencil(
                        vertical_bfield,
                        horizontal_bfield,
                        prims
                    );

                    // Get corner primitives
                    const auto bn = bfield_stencil.horizontal_field(Dir::N);
                    const auto bs = bfield_stencil.horizontal_field(Dir::S);
                    const auto be = bfield_stencil.vertical_field(Dir::E);
                    const auto bw = bfield_stencil.vertical_field(Dir::W);

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
                    // Get corner primitives
                    auto ene = flux_stencil.prim(Dir::NE).ecomponent(nhat);
                    auto enw = flux_stencil.prim(Dir::NW).ecomponent(nhat);
                    auto ese = flux_stencil.prim(Dir::SE).ecomponent(nhat);
                    auto esw = flux_stencil.prim(Dir::SW).ecomponent(nhat);

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
        };

    }   // namespace scheme

}   // namespace simbi

#endif
