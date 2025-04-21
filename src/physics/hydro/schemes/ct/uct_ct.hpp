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
        // encompassing Upwind Corner Transport - Constrained Transport
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
                const auto f_we   = compute_emf_flux(aw, vw, bw, ae, ve, be);
                const auto f_ns   = compute_emf_flux(an, vn, bn, as, vs, bs);
                const auto phi_we = compute_dissipation(dw, bw, de, be);
                const auto phi_ns = compute_dissipation(dn, bn, ds, bs);

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
                const auto& fn_mdz = fn.mdz_vars().value();
                const auto& fs_mdz = fs.mdz_vars().value();
                const auto& fe_mdz = fe.mdz_vars().value();
                const auto& fw_mdz = fw.mdz_vars().value();
                return std::make_tuple(
                    static_cast<real>(0.5) * (fs_mdz.dL + fn_mdz.dL),
                    static_cast<real>(0.5) * (fs_mdz.dR + fn_mdz.dR),
                    static_cast<real>(0.5) * (fw_mdz.dL + fe_mdz.dL),
                    static_cast<real>(0.5) * (fw_mdz.dR + fe_mdz.dR)
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
                const auto& fn_mdz = fn.mdz_vars().value();
                const auto& fs_mdz = fs.mdz_vars().value();
                const auto& fe_mdz = fe.mdz_vars().value();
                const auto& fw_mdz = fw.mdz_vars().value();
                return std::make_tuple(
                    static_cast<real>(0.5) * (fs_mdz.aL + fn_mdz.aL),
                    static_cast<real>(0.5) * (fs_mdz.aR + fn_mdz.aR),
                    static_cast<real>(0.5) * (fw_mdz.aL + fe_mdz.aL),
                    static_cast<real>(0.5) * (fw_mdz.aR + fe_mdz.aR)
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
                const auto& fn_mdz = fn.mdz_vars().value();
                const auto& fs_mdz = fs.mdz_vars().value();
                const auto& fe_mdz = fe.mdz_vars().value();
                const auto& fw_mdz = fw.mdz_vars().value();
                const auto lPv     = my_max(fn_mdz.lamR, fs_mdz.lamR);
                const auto lMv     = my_min(fn_mdz.lamL, fs_mdz.lamL);
                const auto lPh     = my_max(fe_mdz.lamR, fw_mdz.lamR);
                const auto lMh     = my_min(fe_mdz.lamL, fw_mdz.lamL);

                // check for stationary flow
                if (lPh == lMh || lPv == lMv) {
                    return std::make_tuple(
                        fw_mdz.vnorm,
                        fe_mdz.vnorm,
                        fs_mdz.vnorm,
                        fn_mdz.vnorm
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
            compute_dissipation(real dL, real bL, real dR, real bR) -> real
            {
                return (dL * bL) - (dR * bR);
            }
        };
    }   // namespace ct

}   // namespace simbi

#endif
