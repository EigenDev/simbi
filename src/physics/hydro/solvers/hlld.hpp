#ifndef SIMBI_HYDRO_HLLD_HPP
#define SIMBI_HYDRO_HLLD_HPP

#include "config.hpp"               // for global::epsilon
#include "core/base/concepts.hpp"   // for is_hydro_primitive_c, is_mhd_primitive_c
#include "core/utility/enums.hpp"   // for Regime
#include "core/utility/helpers.hpp"   // for goes_to_zero, sgn, safe_less_than, safe_greater_than
#include "data/containers/vector.hpp"   // for vector_t
#include "physics/em/electromagnetism.hpp"   // for to_flux, to_conserved, to_primitive
#include "physics/hydro/conversion.hpp"
#include "physics/hydro/physics.hpp"   // for to_flux, to_conserved, to_primitive
#include "physics/hydro/wave_speeds.hpp"   // for wave_speeds
#include <algorithm>                       // for min, max
#include <cmath>                           // for abs, sqrt
#include <cstdint>                         // for int64_t
#include <limits>                          // for numeric_limits

namespace simbi::hydro::rmhd {
    using namespace simbi::concepts;
    using namespace simbi::em;
    using namespace simbi::vecops;
    using namespace simbi::unit_vectors;

    template <is_hydro_primitive_c primitive_t>
    DUAL real hlld_vdiff(
        const real p,
        const vector_t<typename primitive_t::counterpart_t, 2> r,
        const vector_t<real, 2> lam,
        const real bn,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        primitive_t& praL,
        primitive_t& praR,
        primitive_t& prC
    )

    {
        vector_t<real, 2> eta, enthalpy;
        vector_t<vector_t<real, 3>, 2> kv, bv, vv;
        const auto sgnBn = sgn(bn) + global::epsilon;

        // compute Alfven terms
        for (std::int64_t ii = 0; ii < 2; ii++) {
            const auto aS      = lam[ii];
            const auto rS      = r[ii];
            const auto& rmn    = dot(rS.mom, nhat);
            const auto rmtrans = rS.mom - rmn * nhat;
            const auto rbn     = dot(rS.mag, nhat);
            const auto rbtrans = rS.mag - rbn * nhat;
            const auto ret     = rS.nrg + rS.den;

            // Eqs (26) - (30)
            const real a  = rmn - aS * ret + p * (1.0 - aS * aS);
            const real g  = dot(rbtrans, rbtrans);
            const real ag = (a + g);
            const real c  = dot(rbtrans, rmtrans);
            const real q  = -ag + bn * bn * (1.0 - aS * aS);
            const real x  = bn * (a * aS * bn + c) - ag * (aS * p + ret);

            // Eqs (23) - (25)
            const real term   = (c + bn * (aS * rmn - ret));
            const real vn     = (bn * (a * bn + aS * c) - ag * (p + rmn)) / x;
            const auto vtrans = (q * rmtrans + rbtrans * term) / x;

            // Equation (21)
            const real var1   = 1.0 / (aS - vn);
            const auto btrans = (rbtrans - bn * vtrans) * var1;

            // Equation (31)
            const real rdv = (vn * rmn + dot(vtrans, rmtrans));
            const real wt  = p + (ret - rdv) * var1;

            enthalpy[ii] = wt;

            // Equation (35) & (43)
            eta[ii]           = (ii == 0 ? -1.0 : 1.0) * sgnBn * std::sqrt(wt);
            const auto etaS   = eta[ii];
            const real var2   = 1.0 / (aS * p + ret + bn * etaS);
            const real kn     = (rmn + p + rbn * etaS) * var2;
            const auto ktrans = (rmtrans + rbtrans * etaS) * var2;

            bv[ii] = bn * nhat + btrans;
            vv[ii] = vn * nhat + vtrans;
            kv[ii] = kn * nhat + ktrans;
        }

        // Load left and right vars
        const auto& kL   = kv[LF];
        const auto& kR   = kv[RF];
        const auto& bL   = bv[LF];
        const auto& bR   = bv[RF];
        const auto& vL   = vv[LF];
        const auto& vR   = vv[RF];
        const auto& etaL = eta[LF];
        const auto& etaR = eta[RF];

        // the normal component of the k-vector is the Alfven speed
        const auto& alfL = kL[index(nhat)];
        const auto& alfR = kR[index(nhat)];
        const auto& vnL  = vL[index(nhat)];
        const auto& vnR  = vR[index(nhat)];

        // Compute contact terms
        // Equation (45)
        const auto dkn  = (alfR - alfL) + global::epsilon;
        const auto var3 = 1.0 / dkn;
        const auto bc =
            ((bR * (alfR - vnR) + bn * vR) - (bL * (alfL - vnL) + bn * vL)) *
            var3;

        // Left side Eq.(49)
        real ksq      = dot(kL, kL);
        real kdb      = dot(kL, bc);
        const auto yL = (1.0 - ksq) / (etaL * dkn - kdb);
        // Left side Eq.(47)
        const auto vcL = kL - ((bc * (1.0 - ksq) / (etaL - kdb)));

        // Right side Eq. (49)
        ksq           = dot(kR, kR);
        kdb           = dot(kR, bc);
        const auto yR = (1.0 - ksq) / (etaR * dkn - kdb);
        // Right side Eq.(47)
        const auto vcR = kR - ((bc * (1.0 - ksq) / (etaR - kdb)));

        // Equation (48)
        const auto f = [=] {
            if (goes_to_zero(dkn)) {
                return 0.0;
            }
            else if (goes_to_zero(bn)) {
                return dkn;
            }
            else {
                return dkn * (1.0 - bn * (yR - yL));
            }
        }();

        // check if solution is physically consistent, Eq. (54)
        auto eqn54ok = (vL[index(nhat)] - kL[index(nhat)]) > -global::epsilon;
        eqn54ok &= (kR[index(nhat)] - vR[index(nhat)]) > -global::epsilon;
        eqn54ok &= (lam[0] - vL[index(nhat)]) < 0.0;
        eqn54ok &= (lam[1] - vR[index(nhat)]) > 0.0;
        eqn54ok &= (enthalpy[1] - p) > 0.0;
        eqn54ok &= (enthalpy[0] - p) > 0.0;
        eqn54ok &= (kL[index(nhat)] - lam[index(nhat)]) > -global::epsilon;
        eqn54ok &= (lam[1] - kR[index(nhat)]) > -global::epsilon;

        if (!eqn54ok) {
            return std::numeric_limits<real>::infinity();
        }

        // Fill in the Alfven (L / R) and Contact Prims
        praL.vel      = vL;
        praL.mag      = bL;
        praL.alfven() = kL[0];

        praR.vel      = vR;
        praR.mag      = bR;
        praR.alfven() = kR[0];

        prC.vel = 0.5 * (vcL + vcR);
        prC.mag = bc;

        return f;
    };

    template <is_mhd_primitive_c primitive_t>
    DUAL auto hlld_flux(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real vface,
        real gamma,
        ShockWaveLimiter shock_smoother
    )
    {
        using conserved_t = typename primitive_t::counterpart_t;
        if constexpr (primitive_t::dimensions > 1) {
            if (shock_smoother == ShockWaveLimiter::QUIRK &&
                quirk_strong_shock(primL.pre, primR.pre)) {
                return hlle_flux(primL, primR, nhat, vface, gamma);
            }
        }

        const auto uL = to_conserved(primL, gamma);
        const auto uR = to_conserved(primR, gamma);
        const auto fL = to_flux(primL, nhat, gamma);
        const auto fR = to_flux(primR, nhat, gamma);

        auto [aL, aR]         = extremal_speeds(primL, primR, nhat, gamma);
        aL                    = std::min(aL, 0.0);
        aR                    = std::max(aR, 0.0);
        const auto stationary = aL == aR;

        auto net_flux = [&]() {
            //---- Check Wave Speeds before wasting computations
            if (stationary) {
                return (fL + fR) * 0.5 - (uR + uL) * 0.5 * vface;
            }
            else if (vface <= aL) {
                return fL - uL * vface;
            }
            else if (vface >= aR) {
                return fR - uR * vface;
            }

            const real afac = 1.0 / (aR - aL);

            //-------------------Calculate the HLL Intermediate State
            const auto hll_state = (uR * aR - uL * aL - fR + fL) * afac;

            //------------------Calculate the RHLLE Flux---------------
            const auto hll_flux =
                (fL * aR - fR * aL + (uR - uL) * aR * aL) * afac;

            // the normal component of the magnetic field is assumed to
            // be continuous across the interface, so bnL = bnR = bn
            // const auto bn = bface;
            const real& bn = dot(primL.mag, nhat);

            // Eq. (12)
            const vector_t<conserved_t, 2> r{uL * aL - fL, uR * aR - fR};
            const vector_t<real, 2> lam{aL, aR};

            //------------------------------------
            // Iteratively solve for the pressure
            //------------------------------------
            //------------ initial pressure guess
            const auto maybe_prim = to_primitive(hll_state, gamma);
            if (!maybe_prim.has_value()) {
                return hll_flux - hll_state * vface;
            }
            auto p0 = total_pressure(maybe_prim.value());

            // params to smoothen secant method if HLLD fails
            constexpr real feps          = global::epsilon;
            constexpr real peps          = global::epsilon;
            constexpr real prat_lim      = 0.01;    // pressure ratio limit
            constexpr real pguess_offset = 1.e-6;   // pressure guess offset
            constexpr std::int64_t num_tries =
                15;   // secant tries before giving up
            bool hlld_success = true;

            // L / R Alfven prims and Contact prims
            primitive_t prAL, prAR, prC;
            const auto p = [&] {
                if (bn * bn / p0 < prat_lim) {   // Eq.(53)
                    // in this limit, the pressure is found through Eq. (55)
                    const real et_hll  = hll_state.total_energy();
                    const real fet_hll = hll_flux.total_energy();
                    const real mn_hll  = dot(hll_state.mom, nhat);
                    const real fmn_hll = dot(hll_flux.mom, nhat);

                    const real b    = et_hll - fmn_hll;
                    const real c    = fet_hll * mn_hll - et_hll * fmn_hll;
                    const real quad = std::max(0.0, b * b - 4.0 * c);
                    p0              = 0.5 * (-b + std::sqrt(quad));
                }

                auto f0 = hlld_vdiff(p0, r, lam, bn, nhat, prAL, prAR, prC);
                if (std::abs(f0) < feps) {
                    return p0;
                }

                const real ptol = p0 * peps;
                real p1         = p0 * (1.0 + pguess_offset);
                auto iter       = 0;
                real dp;
                // Use the secant method to solve for the pressure
                do {
                    auto f_lower =
                        hlld_vdiff(p1, r, lam, bn, nhat, prAL, prAR, prC);

                    dp = (p1 - p0) / (f_lower - f0) * f_lower;
                    p0 = p1;
                    f0 = f_lower;
                    p1 -= dp;

                    if (iter++ > num_tries || !std::isfinite(f_lower)) {
                        hlld_success = false;
                        break;
                    }

                } while (std::abs(dp) > ptol || std::abs(f0) > feps);

                return p1;
            }();
            // speed of the contact wave
            const auto vnc = dot(prC.vel, nhat);

            // Alfven speeds
            const auto laL = prAL.alfven();
            const auto laR = prAR.alfven();

            if (!hlld_success) {
                return hll_flux - hll_state * vface;
            }

            // do compound inequalities in two steps
            const auto on_left =
                (safe_less_than(vface, vnc) && safe_greater_than(vface, laL)) ||
                (safe_less_than(vface, laL) && safe_greater_than(vface, aL));
            const auto at_contact =
                (safe_less_than(vface, laR) && safe_greater_than(vface, vnc)) ||
                (safe_less_than(vface, vnc) && safe_greater_than(vface, laL));

            const auto uc = on_left ? uL : uR;
            const auto pa = on_left ? prAL : prAR;
            const auto fc = on_left ? fL : fR;
            const auto rc = on_left ? r[LF] : r[RF];
            const auto lc = on_left ? aL : aR;
            const auto la = on_left ? laL : laR;

            // compute intermediate state across fast waves (Section 3.1)
            // === Fast / Slow Waves ===
            const auto& va  = pa.vel;
            const auto& ba  = pa.mag;
            const auto vdba = dot(va, ba);
            const auto vna  = dot(va, nhat);

            const auto fac = 1.0 / (lc - vna);
            const auto da  = rc.den * fac;
            const auto ea  = (rc.total_energy() + p * vna - vdba * bn) * fac;
            const auto ma  = (ea + p) * va - vdba * ba;

            conserved_t ua;
            ua.den = da;
            ua.mom = ma;
            ua.nrg = ea - da;
            ua.mag = ba;

            const auto fa = fc + (ua - uc) * lc;

            if (!at_contact) {
                return fa - ua * vface;
            }

            // === Contact Wave ===
            // compute jump conditions across alfven waves (Section 3.3)
            const auto vdbC = dot(prC.vel, prC.mag);
            const auto& bc  = prC.mag;
            const auto& vc  = prC.vel;
            const auto fac2 = 1.0 / (la - vnc);
            const auto dc   = da * (la - vna) * fac2;
            const auto man  = dot(ua.mom, nhat);
            const auto ec   = (ea * la - man + p * vnc - vdbC * bn) * fac2;
            const auto mc   = (ec + p) * vc - vdbC * bc;

            conserved_t ut;
            ut.den = dc;
            ut.mom = mc;
            ut.nrg = ec - dc;
            ut.mag = bc;

            return fa + (ut - ua) * la - ut * vface;
        }();

        // upwind the concentration
        if (net_flux.den < 0.0) {
            net_flux.chi = primR.chi * net_flux.den;
        }
        else {
            net_flux.chi = primL.chi * net_flux.den;
        }

        net_flux = shift_electric_field(net_flux, nhat);
        return net_flux;
    };
}   // namespace simbi::hydro::rmhd

#endif   // SIMBI_HYDRO_HLLD_HPP
