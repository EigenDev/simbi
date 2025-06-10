/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            alpha.hpp
 *  * @brief           CT Alpha Scheme from Gardiner & Stone (2005)
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
#ifndef ALPHA_HPP
#define ALPHA_HPP

#include "config.hpp"

namespace simbi {
    namespace ct {
        struct CTAlpha {
            template <typename Flux>
            template <typename Flux>
            static DUAL real compute_emf(
                const Flux& fw,
                const Flux& fe,
                const Flux& fs,
                const Flux& fn,
                const real bw,
                const real be,
                const real bs,
                const real bn,
                const luint nhat
            )
            {
                constexpr real alpha = 0.1;
                // compute permutation indices
                const auto np1 = (P == Plane::JK) ? 2 : 1;
                const auto np2 = (P == Plane::IJ) ? 2 : 3;

                // face-center magnetic field indices
                const auto [nx1, ny1, nz1] = [&] {
                    if constexpr (P == Plane::JK) {
                        return std::make_tuple(xag + 2, nyv, zag + 2);   //
                        B2
                    }
                    return std::make_tuple(nxv, yag + 2, zag + 2);   // B1
                }();
                const auto sidx = cidx<P, C, Dir::S>(ii, jj, kk, nx1, ny1, nz1);
                const auto nidx = cidx<P, C, Dir::N>(ii, jj, kk, nx1, ny1, nz1);

                const auto [nx2, ny2, nz2] = [&] {
                    if constexpr (P == Plane::IJ) {   // B2
                        return std::make_tuple(xag + 2, nyv, zag + 2);
                    }
                    return std::make_tuple(xag + 2, yag + 2, nzv);   // B3
                }();
                const auto eidx = cidx<P, C, Dir::E>(ii, jj, kk, nx2, ny2, nz2);
                const auto widx = cidx<P, C, Dir::W>(ii, jj, kk, nx2, ny2, nz2);

                // perpendicular mean field 1
                const auto bp1sw = swp.bcomponent(np1);
                const auto bp1nw = nwp.bcomponent(np1);
                const auto bp1se = sep.bcomponent(np1);
                const auto bp1ne = nep.bcomponent(np1);

                // perpendicular mean field 2
                const auto bp2sw = swp.bcomponent(np2);
                const auto bp2nw = nwp.bcomponent(np2);
                const auto bp2se = sep.bcomponent(np2);
                const auto bp2ne = nep.bcomponent(np2);

                const auto de_dq2L =
                    (ew - esw + ee - ese) + alpha * (be - bp2se - bw + bp2sw);
                const auto de_dq2R =
                    (enw - ew + ene - ee) + alpha * (bp2ne - be - bp2nw + bw);
                const auto de_dq1L =
                    (es - esw + en - enw) + alpha * (bs - bp1sw - bn + bp1nw);
                const auto de_dq1R =
                    (ese - es + ene - en) + alpha * (bp1se - bs - bp1ne + bn);

                return (
                    eavg + one_eighth * (de_dq2L - de_dq2R + de_dq1L - de_dq1R)
                );
            }
        }
    }   // namespace ct

}   // namespace simbi

#endif
