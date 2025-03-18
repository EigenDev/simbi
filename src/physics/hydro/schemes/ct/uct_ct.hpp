/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            uct_ct.hpp
 *  * @brief           Upwind Corner Transport - Constrained Transport Scheme
 * from Mignone and delZanna (2021)
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
#ifndef UCT_CT_HPP
#define UCT_CT_HPP
#include "build_options.hpp"   // for DUAL

namespace simbi {
    namespace ct {
        // the Constrained Transport scheme
        // encompassing Upwind Corner Transport - Constrained Transpoty
        // described in Mignone and DelZanna 2021.
        struct CTMdZ {
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
                // MdZ coefficients calculation
                const auto [dw, de, ds, dn] =
                    compute_d_coefficients(fw, fe, fs, fn);
                const auto [aw, ae, as, an] =
                    compute_a_coefficients(fw, fe, fs, fn);
                const auto [vw, ve, vs, vn] =
                    compute_transverse_velocities(fw, fe, fs, fn, nhat);

                // Compute fluxes and dissipation
                const auto f_we   = compute_emf_flux(aw, ae, vw, ve, bw, be);
                const auto f_ns   = compute_emf_flux(an, as, vn, vs, bn, bs);
                const auto phi_we = compute_dissipation(dw, de, bw, be);
                const auto phi_ns = compute_dissipation(dn, ds, bn, bs);

                const auto sign = nhat == 2 ? -1.0 : 1.0;
                return sign * ((f_ns - phi_ns) - (f_we + phi_we));
            }

          private:
            template <typename Flux>
            static DUAL auto compute_d_coefficients(
                const Flux& fw,
                const Flux& fe,
                const Flux& fs,
                const Flux& fn
            ) -> std::tuple<real, real, real, real>
            {
                // d-coefficients from MdZ (2021), Eqns. (34 & 35)a
                return std::make_tuple(
                    static_cast<real>(0.5) * (fs.dL + fn.dL),
                    static_cast<real>(0.5) * (fs.dR + fn.dR),
                    static_cast<real>(0.5) * (fw.dL + fe.dL),
                    static_cast<real>(0.5) * (fw.dR + fe.dR)
                );
            };

            template <typename Flux>
            static DUAL auto compute_a_coefficients(
                const Flux& fw,
                const Flux& fe,
                const Flux& fs,
                const Flux& fn
            ) -> std::tuple<real, real, real, real>
            {
                // a-coefficients from MdZ (2021), Eqns. (34 & 35)b
                return std::make_tuple(
                    static_cast<real>(0.5) * (fs.aL + fn.aL),
                    static_cast<real>(0.5) * (fs.aR + fn.aR),
                    static_cast<real>(0.5) * (fw.aL + fe.aL),
                    static_cast<real>(0.5) * (fw.aR + fe.aR)
                );
            };

            template <typename Flux>
            static DUAL auto compute_transverse_velocities(
                const Flux& fw,
                const Flux& fe,
                const Flux& fs,
                const Flux& fn,
                const luint nhat
            ) -> std::tuple<real, real, real, real>
            {
                // transverse velocities from MdZ (2021), Eqns. (34 & 35)c
                // average velocity coefficients, just after Eq. (27)
                const auto lPv = my_max(fn.lamR, fs.lamR);
                const auto lMv = my_min(fn.lamL, fs.lamL);
                const auto lPh = my_max(fe.lamR, fw.lamR);
                const auto lMh = my_min(fe.lamL, fw.lamL);

                // check for stationary flow
                if (lPh == lMh || lPv == lMv) {
                    return std::make_tuple(
                        fw.vcomponent(nhat),
                        fe.vcomponent(nhat),
                        fs.vcomponent(nhat),
                        fn.vcomponent(nhat)
                    );
                }

                // compute transverse velocities according to Eq. (29)
                const auto nj  = nhat % 2 == 0 ? 1 : 2;
                const auto nk  = nhat % 2 == 0 ? 2 : 1;
                const auto aph = lPh / (lPh - lMh);
                const auto amh = lMh / (lPh - lMh);
                const auto apv = lPv / (lPv - lMv);
                const auto amv = lMv / (lPv - lMv);
                const auto vw  = aph * fw.vLtrans(nj) - amh * fw.vRtrans(nj);
                const auto ve  = aph * fe.vLtrans(nj) - amh * fe.vRtrans(nj);
                const auto vn  = apv * fn.vLtrans(nk) - amv * fn.vRtrans(nk);
                const auto vs  = apv * fs.vLtrans(nk) - amv * fs.vRtrans(nk);

                return std::make_tuple(vw, ve, vs, vn);
            }

            static DUAL auto compute_emf_flux(
                real aL,
                real vL,
                real bL,
                real aR,
                real vR,
                real bR
            ) -> real
            {
                return (aL * vL * bL) + (aR * vR * bR);
            }

            static DUAL auto
            compute_disspipation(real dL, real bL, real dR, real bR) -> real
            {
                return (dL * bL) + (dR * bR);
            }
        };
    }   // namespace ct

}   // namespace simbi

#endif
