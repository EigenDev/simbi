/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       rmhd.hpp
 * @brief      Single header for 1, 2, and 3D RMHD calculations
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */

#ifndef RMHD_HPP
#define RMHD_HPP

#include "base.hpp"                   // for HydroBase
#include "build_options.hpp"          // for real, HD, lint, luint
#include "common/enums.hpp"           // for TIMESTEP_TYPE
#include "common/helpers.hpp"         // for my_min, my_max, ...
#include "common/hydro_structs.hpp"   // for Conserved, Primitive
#include "util/exec_policy.hpp"       // for ExecutionPolicy
#include "util/ndarray.hpp"           // for ndarray
#include <functional>                 // for function
#include <optional>                   // for optional
#include <type_traits>                // for conditional_t
#include <vector>                     // for vector

namespace simbi {
    template <int dim>
    struct RMHD : public HydroBase {

        // set the primitive and conservative types at compile time
        using primitive_t     = rmhd::AnyPrimitive<dim>;
        using conserved_t     = rmhd::AnyConserved<dim>;
        using primitive_soa_t = rmhd::PrimitiveSOA;
        using eigenvals_t     = rmhd::Eigenvals;
        using mag_fourvec_t   = rmhd::mag_four_vec<dim>;
        using function_t      = typename std::conditional_t<
                 dim == 1,
                 std::function<real(real)>,
                 std::conditional_t<
                     dim == 2,
                     std::function<real(real, real)>,
                     std::function<real(real, real, real)>>>;

        function_t dens_outer;
        function_t mom1_outer;
        function_t mom2_outer;
        function_t mom3_outer;
        function_t enrg_outer;
        function_t mag1_outer;
        function_t mag2_outer;
        function_t mag3_outer;

        const static int dimensions = dim;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> edens_guess, dt_min, bstag1, bstag2, bstag3;
        bool scalar_all_zeros;
        luint nzone_edges;

        /* Methods */
        RMHD();
        RMHD(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );
        ~RMHD();

        void cons2prim(const ExecutionPolicy<>& p);

        /**
         * Return the primitive
         * variables density , three-velocity, pressure
         *
         * @param  con conserved array at index
         * @param gid  current global index
         * @return none
         */
        HD primitive_t cons2prim(const conserved_t& cons) const;

        void advance(const ExecutionPolicy<>& p);

        HD void calc_max_wave_speeds(
            const primitive_t& prims,
            const luint nhat,
            real speeds[],
            real& cs2
        ) const;

        HD eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        HD conserved_t prims2cons(const primitive_t& prims) const;

        HD conserved_t calc_hll_flux(
            primitive_t& prL,
            primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        HD conserved_t calc_hllc_flux(
            primitive_t& prL,
            primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        HD conserved_t calc_hlld_flux(
            primitive_t& prL,
            primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        HD conserved_t (RMHD<dim>::* riemann_solve)(
            primitive_t& prL,
            primitive_t& prR,
            const luint nhat,
            const real vface
        ) const;

        DEV conserved_t (RMHD<dim>::* d_riemann_solve)(
            primitive_t& prL,
            primitive_t& prR,
            const luint nhat,
            const real vface
        ) const;

        conserved_t (RMHD<dim>::* h_riemann_solve)(
            primitive_t& prL,
            primitive_t& prR,
            const luint nhat,
            const real vface
        ) const;

        HD void set_riemann_solver()
        {
            switch (sim_solver) {
                case Solver::HLLE:
                    this->h_riemann_solve = &RMHD<dim>::calc_hll_flux;
                    break;
                case Solver::HLLC:
                    this->h_riemann_solve = &RMHD<dim>::calc_hllc_flux;
                    break;
                default:
                    this->h_riemann_solve = &RMHD<dim>::calc_hlld_flux;
                    break;
            }
            if constexpr (global::BuildPlatform == global::Platform::GPU) {
                gpu::api::gpuMcFromSymbol(
                    &this->d_riemann_solve,
                    &this->h_riemann_solve,
                    sizeof(void*)
                );
                this->riemann_solve = this->d_riemann_solve;
            }
            else {
                this->riemann_solve = this->h_riemann_solve;
            }
        }

        HD conserved_t
        prims2flux(const primitive_t& prims, const luint nhat) const;

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<>& p);

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot,
            std::optional<function_t> const& d_outer  = nullptr,
            std::optional<function_t> const& s1_outer = nullptr,
            std::optional<function_t> const& s2_outer = nullptr,
            std::optional<function_t> const& s3_outer = nullptr,
            std::optional<function_t> const& e_outer  = nullptr
        );

        HD constexpr real get_x1face(const lint ii, const int side) const;

        HD constexpr real get_x2face(const lint ii, const int side) const;

        HD constexpr real get_x3face(const lint ii, const int side) const;

        HD constexpr real get_x1_differential(const lint ii) const;

        HD constexpr real get_x2_differential(const lint ii) const;

        HD constexpr real get_x3_differential(const lint ii) const;

        HD real get_cell_volume(
            const lint ii,
            const lint jj = 0,
            const lint kk = 0
        ) const;

        HD real curl_e(
            const luint nhat,
            const real ejl,
            const real ejr,
            const real ekl,
            const real ekr
        ) const;

        /**
         * @brief
         * @retval
         */
        template <Plane P, Corner C>
        HD real calc_edge_emf(
            const conserved_t& fw,
            const conserved_t& fe,
            const conserved_t& fs,
            const conserved_t& fn,
            const ndarray<real>& bstagp1,
            const ndarray<real>& bstagp2,
            const primitive_t* prims,
            const luint ii,
            const luint jj,
            const luint kk,
            const luint ia,
            const luint ja,
            const luint ka,
            const luint nhat
        ) const;

        void emit_troubled_cells() const;

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            edens_guess.copyToGpu();
            dt_min.copyToGpu();
            density_source.copyToGpu();
            m1_source.copyToGpu();
            m2_source.copyToGpu();
            m3_source.copyToGpu();
            object_pos.copyToGpu();
            energy_source.copyToGpu();
            inflow_zones.copyToGpu();
            bcs.copyToGpu();
            troubled_cells.copyToGpu();
            sourceG1.copyToGpu();
            sourceG2.copyToGpu();
            sourceG3.copyToGpu();
            sourceB1.copyToGpu();
            sourceB2.copyToGpu();
            sourceB3.copyToGpu();
            bstag1.copyToGpu();
            bstag2.copyToGpu();
            bstag3.copyToGpu();
        }

        HD std::tuple<real, primitive_t, primitive_t, primitive_t> hlld_vdiff(
            const real p,
            const conserved_t r[2],
            const real lam[2],
            const real bn,
            const luint nhat
        ) const

        {
            static real eta[2];
            static real kv[2][3], bv[2][3], vv[2][3];

            // compute "sign" of the normal bfield
            // we do it this way to avoid exploding
            // terms in the Alfven speed
            const auto sgnBn = sgn(bn);
            const auto bfn   = limit_zero(bn);

            // store the left and right prims (rotational)
            // and the contact prims
            primitive_t prims[3];
            const auto np1 = next_perm(nhat, 1);
            const auto np2 = next_perm(nhat, 2);
            // compute Alfven terms
            for (int ii = 0; ii < 2; ii++) {
                const auto aS   = lam[ii];
                const auto rS   = r[ii];
                const auto rmn  = rS.momentum(nhat);
                const auto rmp1 = rS.momentum(np1);
                const auto rmp2 = rS.momentum(np2);
                const auto rbn  = rS.bcomponent(nhat);
                const auto rbp1 = rS.bcomponent(np1);
                const auto rbp2 = rS.bcomponent(np2);
                const auto ret  = rS.total_energy();

                // Eqs (26) - (30)
                const real a  = rmn - aS * ret + p * (1.0 - aS * aS);
                const real g  = rbp1 * rbp1 + rbp2 * rbp2;
                const real ag = (a + g);
                const real c  = rbp1 * rmp1 + rbp2 * rmp2;
                const real q  = -ag + bn * bn * (1.0 - aS * aS);
                const real x  = bn * (a * aS * bn + c) - ag * (aS * p + ret);

                // Eqs (23) - (25)
                const real term = (c + bn * (aS * rmn - ret));
                const real vn   = (bn * (a * bn + aS * c) - ag * (p + rmn)) / x;
                const real vp1  = (q * rmp1 + rbp1 * term) / x;
                const real vp2  = (q * rmp2 + rbp2 * term) / x;

                // Equation (21)
                const real var1 = 1.0 / (aS - vn);
                const real bp1  = (rbp1 - bn * vp1) * var1;
                const real bp2  = (rbp2 - bn * vp2) * var1;

                // Equation (31)
                const real rdv = (vn * rmn + vp1 * rmp1 + vp2 * rmp2);
                const real wt  = p + (ret - rdv) * var1;

                // Equation (35) & (43)
                eta[ii] = (ii < 1 ? -1.0 : 1.0) * sgnBn * std::sqrt(wt);
                // h[ii]           = wt;
                const auto etaS = eta[ii];
                const real var2 = 1.0 / (aS * p + ret + bn * etaS);
                const real kn   = (rmn + p + rbn * etaS) * var2;
                const real kp1  = (rmp1 + rbp1 * etaS) * var2;
                const real kp2  = (rmp2 + rbp2 * etaS) * var2;

                vv[ii][0] = vn;
                vv[ii][1] = vp1;
                vv[ii][2] = vp2;

                // the normal component of the k-vector is the Alfven speed
                kv[ii][0] = kn;
                kv[ii][1] = kp1;
                kv[ii][2] = kp2;

                bv[ii][0] = bn;
                bv[ii][1] = bp1;
                bv[ii][2] = bp2;
            }

            // Load left and right vars
            const auto kL   = kv[LF];
            const auto kR   = kv[RF];
            const auto bL   = bv[LF];
            const auto bR   = bv[RF];
            const auto vL   = vv[LF];
            const auto vR   = vv[RF];
            const auto etaL = eta[LF];
            const auto etaR = eta[RF];

            auto bterm = [bn](real b, real lam, real vn, real v) {
                return b * (lam - vn) + bn * v;
            };

            // Compute contact terms
            // Equation (45)
            const real dkn  = (kR[0] - kL[0] + global::tol_scale);
            const real var3 = 1.0 / dkn;
            const real bcn  = bn;
            const real bcp1 = (bterm(bR[1], kR[0], vR[0], vR[1]) -
                               bterm(bL[1], kL[0], vL[0], vL[1])) *
                              var3;
            const real bcp2 = (bterm(bR[2], kR[0], vR[0], vR[2]) -
                               bterm(bL[2], kL[0], vL[0], vL[2])) *
                              var3;

            // Left side Eq.(49)
            real kcn      = kL[0];
            real kcp1     = kL[1];
            real kcp2     = kL[2];
            auto ksq      = kcn * kcn + kcp1 * kcp1 + kcp2 * kcp2;
            auto kdb      = kcn * bcn + kcp1 * bcp1 + kcp2 * bcp2;
            auto bhc      = kdb * dkn;
            auto reg      = bfn ? 0.0 : (1.0 - ksq) / (etaL - kdb);
            const real yL = (1.0 - ksq) / (etaL * dkn - bhc);

            const real vncL  = kcn - bcn * reg;
            const real vpc1L = kcp1 - bcp1 * reg;
            const real vpc2L = kcp2 - bcp2 * reg;

            // Right side Eq. (49)
            kcn           = kR[0];
            kcp1          = kR[1];
            kcp2          = kR[2];
            ksq           = kcn * kcn + kcp1 * kcp1 + kcp2 * kcp2;
            kdb           = kcn * bcn + kcp1 * bcp1 + kcp2 * bcp2;
            bhc           = kdb * dkn;
            reg           = bfn ? 0.0 : (1.0 - ksq) / (etaR - kdb);
            const real yR = (1.0 - ksq) / (etaR * dkn - bhc);

            const real vncR  = kcn - bcn * reg;
            const real vpc1R = kcp1 - bcp1 * reg;
            const real vpc2R = kcp2 - bcp2 * reg;

            // Equation (48)
            const real f = dkn * (1.0 - bn * (yR - yL));

            // printf("vncL: %.2e, vncR: %.2e\n", vncL, vncR);
            // printf("f: %.2e\n", f);
            // printf("dkn: %.2e\n", dkn);
            // Return prims for later computation
            prims[0].vcomponent(nhat) = vL[0];
            prims[0].vcomponent(np1)  = vL[1];
            prims[0].vcomponent(np2)  = vL[2];
            prims[0].bcomponent(nhat) = bL[0];
            prims[0].bcomponent(np1)  = bL[1];
            prims[0].bcomponent(np2)  = bL[2];
            prims[0].p                = kL[0];   // store the Alfven speed

            prims[1].vcomponent(nhat) = vR[0];
            prims[1].vcomponent(np1)  = vR[1];
            prims[1].vcomponent(np2)  = vR[2];
            prims[1].bcomponent(nhat) = bR[0];
            prims[1].bcomponent(np1)  = bR[1];
            prims[1].bcomponent(np2)  = bR[2];
            prims[1].p                = kR[0];   // store the Alfven speed

            prims[2].vcomponent(nhat) = 0.5 * (vncR + vncL);
            prims[2].vcomponent(np1)  = 0.5 * (vpc1R + vpc1L);
            prims[2].vcomponent(np2)  = 0.5 * (vpc2R + vpc2L);
            prims[2].bcomponent(nhat) = bcn;
            prims[2].bcomponent(np1)  = bcp1;
            prims[2].bcomponent(np2)  = bcp2;

            /* -- check if sweep makes physical sense -- */

            // auto success = (vncL - kL[0]) > -1.e-6;
            // success *= (kR[0] - vncR) > -1.e-6;

            // success *= (lam[0] - vL[0]) < 0.0;
            // success *= (lam[1] - vR[0]) > 0.0;

            // success *= (h[1] - p) > 0.0;
            // success *= (h[0] - p) > 0.0;
            // success *= (kL[0] - lam[0]) > -1.e-6;
            // success *= (lam[1] - kR[0]) > -1.e-6;

            // if (!success) {
            //     printf("Solution not physical!\n");
            //     printf("bn: %.5e\n", bn);
            //     std::cout << bfn << "\n";
            //     std::cout << kL[0] << "\n";
            //     std::cout << "Check 1: " << (vncL - kL[0]) << "\n";
            //     std::cout << "Check 2: " << (kR[0] - vncR) << "\n";
            //     std::cout << "Check 3: " << (lam[0] - vL[0]) << "\n";
            //     std::cout << "Check 4: " << (lam[1] - vR[0]) << "\n";
            //     std::cout << "Check 5: " << (h[1] - p) << "\n";
            //     std::cout << "Check 6: " << (h[0] - p) << "\n";
            //     std::cout << "Check 7: " << (kL[0] - lam[0]) << "\n";
            //     std::cout << "Check 8: " << (lam[1] - kR[0]) << "\n";
            // }

            return {f, prims[0], prims[1], prims[2]};
        }
    };
}   // namespace simbi

template <>
struct is_relativistic_mhd<simbi::RMHD<1>> {
    static constexpr bool value = true;
};

template <>
struct is_relativistic_mhd<simbi::RMHD<2>> {
    static constexpr bool value = true;
};

template <>
struct is_relativistic_mhd<simbi::RMHD<3>> {
    static constexpr bool value = true;
};

#include "rmhd.tpp"
#endif