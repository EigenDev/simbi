/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            zero.hpp
 *  * @brief           CT Zero Scheme from Gardiner & Stone (2005)
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
#ifndef ZERO_HPP
#define ZERO_HPP

#include "config.hpp"   // for DUAL

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
                const std::uint64_t nhat
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
