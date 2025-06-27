/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            contact.hpp
 *  * @brief           CT Contact Scheme from Gardiner & Stone (2005)
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
#ifndef CONTACT_HPP
#define CONTACT_HPP
#include "config.hpp"

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
                const std::uint64_t nhat
            )
            {
                // south, north, east, west electric fields
                // from Riemann fluxes
                const auto es   = fs.ecomponent(nhat);
                const auto en   = fn.ecomponent(nhat);
                const auto ew   = fw.ecomponent(nhat);
                const auto ee   = fe.ecomponent(nhat);
                const auto eavg = static_cast<real>(0.25) * (es + en + ew + ee);
                constexpr auto one_eighth = static_cast<real>(0.125);

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

                // Eqn. (42)
                return (
                    eavg + one_eighth * (de_dqjL - de_dqjR + de_dqkL - de_dqkR)
                );
            }
        };
    }   // namespace ct

}   // namespace simbi

#endif
