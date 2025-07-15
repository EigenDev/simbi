#ifndef SIMBI_HYDRO_HLLC_HPP
#define SIMBI_HYDRO_HLLC_HPP

#include "config.hpp"               // for DEV macro
#include "core/base/concepts.hpp"   // for is_hydro_primitive_c
#include "core/utility/enums.hpp"   // for ShockWaveLimiter
#include "core/utility/helpers.hpp"   // for goes_to_zero, sgn, vecops::dot, vecops::norm
#include "data/containers/vector.hpp"        // for VectorLike
#include "physics/em/electromagnetism.hpp"   // for shift_electric_field
#include "physics/hydro/solvers/hlle.hpp"    // for hlle_flux
#include "physics/hydro/wave_speeds.hpp"     // for extremal_speeds
#include <algorithm>                         // for std::max, std::min
#include <cmath>                             // for std::abs, std::log
#include <numbers>                           // for std::numbers::pi

namespace simbi::hydro {
    using namespace simbi::helpers;
    struct EntropyDetector {
        template <is_hydro_primitive_c primitive_t>
        DEV static real compute_local_mach(
            const primitive_t& primL,
            const primitive_t& primR,
            real gamma
        )
        {
            const auto velL = primL.vel;
            const auto velR = primR.vel;
            const auto csL  = sound_speed(primL, gamma);
            const auto csR  = sound_speed(primR, gamma);

            // use the maximum of the Mach numbers from both sides
            // as suggested by Fleischmann et al. (2020)
            const real mach_L = vecops::norm(velL) / csL;
            const real mach_R = vecops::norm(velR) / csR;

            return std::max(mach_L, mach_R);
        }

        template <is_hydro_primitive_c primitive_t>
        DEV static real detect_interface_correction(
            const primitive_t& primL,
            const primitive_t& primR
        )
        {
            // detect material interfaces (RT-style contact discontinuities)
            const real rho_jump = std::abs(primL.rho - primR.rho) /
                                  (0.5 * (primL.rho + primR.rho));
            const real pressure_jump = std::abs(primL.pre - primR.pre) /
                                       (0.5 * (primL.pre + primR.pre));

            // interface = large density jump, small pressure jump (nearly
            // isentropic)
            const bool is_contact = (rho_jump > 0.1) && (pressure_jump < 0.05);

            if (is_contact) {
                // need moderate dissipation to prevent interface oscillations
                return 0.4;   // fixed moderate phi for interfaces
            }
            else {
                return 0.0;   // no correction
            }
        }

        template <is_hydro_primitive_c primitive_t, VectorLike UnitVector>
        DEV static real detect_shock_correction(
            const primitive_t& primL,
            const primitive_t& primR,
            const UnitVector& nhat,
            real gamma
        )
        {
            // entropy production (shocks increase entropy)
            const real sL = std::log(primL.pre) - gamma * std::log(primL.rho);
            const real sR = std::log(primR.pre) - gamma * std::log(primR.rho);
            const real entropy_production = sR - sL;

            // velocity convergence (shocks compress flow)
            const real vL_normal = vecops::dot(primL.vel, nhat);
            const real vR_normal = vecops::dot(primR.vel, nhat);
            const real velocity_convergence =
                vL_normal - vR_normal;   // > 0 for compression

            // combined shock indicator
            const bool is_shock =
                (entropy_production > 0.01) && (velocity_convergence > 0.0);

            if (is_shock) {
                return 1.0;   // force standard HLLC for shocks
            }
            else {
                return 0.0;   // no correction
            }
        }

        template <is_hydro_primitive_c primitive_t>
        DEV static real detect_stagnation_correction(
            const primitive_t& primL,
            const primitive_t& primR,
            real gamma
        )
        {
            // detect very low velocity regions (want maximum LM treatment)
            const real cL     = sound_speed(primL, gamma);
            const real cR     = sound_speed(primR, gamma);
            const real vL_mag = vecops::norm(primL.vel);
            const real vR_mag = vecops::norm(primR.vel);

            const real mach_L   = vL_mag / cL;
            const real mach_R   = vR_mag / cR;
            const real max_mach = std::max(mach_L, mach_R);

            if (max_mach < 0.01) {
                // nearly stagnant → force low dissipation
                return -0.5;   // negative correction to reduce phi
            }
            else {
                return 0.0;   // no correction
            }
        }

        template <is_hydro_primitive_c primitive_t, VectorLike UnitVector>
        DEV static real detect_alignment_correction(
            const primitive_t& primL,
            const primitive_t& primR,
            const UnitVector& nhat,
            real gamma
        )
        {
            // check if flow is aligned with interface (carbuncle risk)
            const real vL_normal = vecops::dot(primL.vel, nhat);
            const real vR_normal = vecops::dot(primR.vel, nhat);
            const real vL_mag    = vecops::norm(primL.vel);
            const real vR_mag    = vecops::norm(primR.vel);

            if (vL_mag > 1e-10 && vR_mag > 1e-10) {
                const real alignment_L   = std::abs(vL_normal) / vL_mag;
                const real alignment_R   = std::abs(vR_normal) / vR_mag;
                const real max_alignment = std::max(alignment_L, alignment_R);

                // high speed + high alignment = carbuncle risk
                const real avg_mach =
                    0.5 * (vL_mag / sound_speed(primL, gamma) +
                           vR_mag / sound_speed(primR, gamma));

                if ((max_alignment > 0.8) && (avg_mach > 0.5)) {
                    return 1.0;   // force standard HLLC to prevent carbuncle
                }
            }

            return 0.0;   // no correction
        }

        template <is_hydro_primitive_c primitive_t, VectorLike UnitVector>
        DEV static real compute_adaptive_phi(
            const primitive_t& primL,
            const primitive_t& primR,
            const UnitVector& nhat,
            real gamma,
            bool use_fleischmann
        )
        {
            if (!use_fleischmann) {
                return 1.0;   // no adaptive phi, use standard HLLC
            }

            // base Mach number criterion
            // This number is found in Fleischamnn et al. (2020)
            // A shock-stable modification of the HLLC Riemann solver with
            // reduced numerical dissipation
            constexpr real mach_lim = 0.1;
            const real ma_local     = compute_local_mach(primL, primR, gamma);
            real phi                = std::sin(
                std::min(1.0, ma_local / mach_lim) * std::numbers::pi * 0.5
            );

            // physics-based corrections
            const real correction_factors[] = {
              detect_interface_correction(
                  primL,
                  primR
              ),   // boost phi for interfaces
              detect_shock_correction(
                  primL,
                  primR,
                  nhat,
                  gamma
              ),   // force phi=1 for shocks
              detect_stagnation_correction(
                  primL,
                  primR,
                  gamma
              ),   // reduce phi for stagnant regions
              detect_alignment_correction(
                  primL,
                  primR,
                  nhat,
                  gamma
              )   // boost phi for aligned flows
            };

            // apply the strongest correction
            for (auto factor : correction_factors) {
                phi = std::max(phi, factor);
            }

            return std::min(1.0, phi);
        }
    };

}   // namespace simbi::hydro

// ==========================================================================
// NEWTONIAN HLLC FLUX
// ==========================================================================
namespace simbi::hydro::newtonian {
    using namespace simbi::concepts;
    template <is_hydro_primitive_c primitive_t>
    DEV auto hllc_flux(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real vface,
        real gamma,
        ShockWaveLimiter shock_smoother
    )
    {
        if constexpr (primitive_t::dimensions > 1) {
            if (shock_smoother == ShockWaveLimiter::QUIRK &&
                quirk_strong_shock(primL.pre, primR.pre)) {
                return hlle_flux(primL, primR, nhat, vface, gamma);
            }
        }

        using conserved_t = typename primitive_t::counterpart_t;
        // convert to conserved variables
        const auto uL = to_conserved(primL, gamma);
        const auto uR = to_conserved(primR, gamma);
        const auto fL = to_flux(primL, nhat, gamma);
        const auto fR = to_flux(primR, nhat, gamma);

        // calculate wave speeds
        const auto wave_info = wave_properties(primL, primR, nhat, gamma);
        const auto& ws       = wave_info.speeds;
        const auto aL        = ws.min();
        const auto aR        = ws.max();

        const auto& contact = wave_info.contact;
        const real a_star   = contact.speed;
        const real p_star   = contact.pressure;

        // --------------Compute the L Star State----------
        real pre = primL.pre;
        real rho = uL.den;
        auto mom = uL.mom;
        real nrg = uL.nrg;
        real fac = 1.0 / (aL - a_star);

        const real vnL = vecops::dot(primL.vel, nhat);
        const real vnR = vecops::dot(primR.vel, nhat);

        // Left Star State in x-direction of coordinate lattice
        real rhostar = fac * (aL - vnL) * rho;
        auto mstar   = fac * (mom * (aL - vnL) + nhat * (p_star - pre));
        real estar   = fac * (nrg * (aL - vnL) + (p_star * a_star - pre * vnL));
        const auto starStateL =
            conserved_t{rhostar, mstar, estar, rhostar * primL.chi};

        pre = primR.pre;
        rho = uR.den;
        mom = uR.mom;
        nrg = uR.nrg;
        fac = 1.0 / (aR - a_star);

        rhostar = fac * (aR - vnR) * rho;
        mstar   = fac * (mom * (aR - vnR) + nhat * (-pre + p_star));
        estar   = fac * (nrg * (aR - vnR) + (p_star * a_star - pre * vnR));
        const auto starStateR =
            conserved_t{rhostar, mstar, estar, rhostar * primR.chi};

        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020:
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        const real phi = EntropyDetector::compute_adaptive_phi(
            primL,
            primR,
            nhat,
            gamma,
            // flag for Fleischmann et al. (2020) low-Mach fix
            shock_smoother == ShockWaveLimiter::FLEISCHMANN
        );
        const real aL_lm          = phi * aL;
        const real aR_lm          = phi * aR;
        const auto face_starState = (a_star <= 0) ? starStateR : starStateL;
        auto net_flux             = (fL + fR) * 0.5 +
                        ((starStateL - uL) * aL_lm +
                         (starStateL - starStateR) * std::abs(a_star) +
                         (starStateR - uR) * aR_lm) *
                            0.5 -
                        face_starState * vface;

        // upwind the concentration
        if (net_flux.den < 0.0) {
            net_flux.chi = primR.chi * net_flux.den;
        }
        else {
            net_flux.chi = primL.chi * net_flux.den;
        }

        return net_flux;
    }
}   // namespace simbi::hydro::newtonian

// ==========================================================================
// SRHD HLLC FLUX
// =========================================================================
namespace simbi::hydro::srhd {
    using namespace simbi::concepts;

    template <is_hydro_primitive_c primitive_t>
    DUAL auto hllc_flux(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real vface,
        real gamma,
        ShockWaveLimiter shock_smoother
    )
    {
        if constexpr (primitive_t::dimensions > 1) {
            if (shock_smoother == ShockWaveLimiter::QUIRK) {
                if (quirk_strong_shock(primL.pre, primR.pre)) {
                    return hlle_flux(primL, primR, nhat, vface, gamma);
                }
            }
        }

        const auto uL       = to_conserved(primL, gamma);
        const auto uR       = to_conserved(primR, gamma);
        const auto fL       = to_flux(primL, nhat, gamma);
        const auto fR       = to_flux(primR, nhat, gamma);
        const auto [aL, aR] = extremal_speeds(primL, primR, nhat, gamma);

        auto net_flux = [&]() {
            // quick returns for supersonic states
            if (aL >= vface) {
                return fL - uL * vface;   // left state is supersonic
            }
            if (aR <= vface) {
                return fR - uR * vface;   // right state is supersonic
            }

            // calculate intermediate state
            const auto [a_star, p_star] =
                contact_props(uL, uR, fL, fR, nhat, aL, aR);
            const bool on_left = (vface <= a_star);

            const auto& prim = on_left ? primL : primR;
            const auto& u    = on_left ? uL : uR;
            const auto& a    = on_left ? aL : aR;
            const auto& f    = on_left ? fL : fR;
            const auto us    = star_state(prim, u, a, a_star, p_star, nhat);

            const auto& pf = on_left ? primR : primL;
            const auto& uf = on_left ? uR : uL;
            // const auto& ff = on_left ? fR : fL;
            const auto af = on_left ? aR : aL;
            const auto un = star_state(pf, uf, af, a_star, p_star, nhat);

            return f + (us - u) * a - un * vface;
        }();

        // upwind the scalar concentration
        if (net_flux.den < 0.0) {
            net_flux.chi = primR.chi * net_flux.den;
        }
        else {
            net_flux.chi = primL.chi * net_flux.den;
        }

        return net_flux;
    }
}   // namespace simbi::hydro::srhd

namespace simbi::hydro::rmhd {
    using namespace simbi::concepts;
    using namespace simbi::vecops;
    using namespace simbi::em;

    template <is_hydro_primitive_c primitive_t>
    DUAL auto hllc_flux(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        // real bface,
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

        const auto uL       = to_conserved(primL, gamma);
        const auto uR       = to_conserved(primR, gamma);
        const auto fL       = to_flux(primL, nhat, gamma);
        const auto fR       = to_flux(primR, nhat, gamma);
        const auto [aL, aR] = extremal_speeds(primL, primR, nhat, gamma);

        const auto stationary = aL == aR;
        auto net_flux         = [&]() {
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

            //-------------------Calculate the HLL Intermediate State
            const auto hll_state = (uR * aR - uL * aL - fR + fL) / (aR - aL);

            //------------------Calculate the RHLLE Flux---------------
            const auto hll_flux =
                (fL * aR - fR * aL + (uR - uL) * aR * aL) / (aR - aL);

            // get the perpendicular directional unit vectors
            // the normal component of the magnetic field is assumed to
            // be continuous across the interface, so bnL = bnR = bn
            const auto bn = dot(hll_state.mag, nhat);
            // const auto bn = bface;
            // const real bn  = hll_state.bcomponent(nhat);
            const auto& b_hll = hll_state.mag;
            const auto bt_hll = b_hll - dot(b_hll, nhat) * nhat;

            // check if normal magnetic field is approaching zero
            const auto null_normal_field = goes_to_zero(bn);

            const auto uhlld = hll_state.den;
            const auto uhllm = dot(hll_state.mom, nhat);
            const auto uhlle = hll_state.nrg + uhlld;

            const auto fhlld   = hll_flux.den;
            const auto fhllm   = dot(hll_flux.mom, nhat);
            const auto fhlle   = hll_flux.nrg + fhlld;
            const auto& fb_hll = hll_flux.mag;
            const auto ft_hll  = fb_hll - dot(fb_hll, nhat) * nhat;

            // //------Calculate the contact wave velocity and pressure
            real a, b, c;
            if (null_normal_field) {
                a = fhlle;
                b = -(fhllm + uhlle);
                c = uhllm;
            }
            else {
                const auto fdb   = dot(ft_hll, bt_hll);
                const auto bpsq  = dot(bt_hll, bt_hll);
                const auto fbpsq = dot(ft_hll, ft_hll);
                a                = fhlle - fdb;
                b                = -(fhllm + uhlle) + bpsq + fbpsq;
                c                = uhllm - fdb;
            }

            const auto disc   = b * b - 4.0 * a * c;
            const auto quad   = -0.5 * (b + helpers::sgn(b) * std::sqrt(disc));
            const auto a_star = c / quad;

            const auto on_left = vface < a_star;
            const auto u       = on_left ? uL : uR;
            const auto f       = on_left ? fL : fR;
            const auto pr      = on_left ? primL : primR;
            const auto ws      = on_left ? aL : aR;

            const auto den     = u.den;
            const auto mn      = dot(u.mom, nhat);
            const auto umtrans = u.mom - nhat * mn;
            const auto fmtrans = f.mom - nhat * dot(f.mom, nhat);
            const auto etot    = u.nrg + den;
            const auto cfac    = 1.0 / (ws - a_star);

            const auto v  = dot(pr.vel, nhat);
            const auto vs = cfac * (ws - v);
            const auto ds = vs * den;
            // star state
            conserved_t us;
            if (null_normal_field) {
                const auto p_star = -a_star * fhlle + fhllm;
                const auto es     = cfac * (ws * etot - mn + p_star * a_star);
                const auto mn     = (es + p_star) * a_star;
                auto btrans       = pr.mag - nhat * dot(pr.mag, nhat);
                us.den            = ds;
                us.mom            = mn * nhat + vs * umtrans;
                us.nrg            = es - ds;
                us.mag            = bn * nhat + vs * btrans;
            }
            else {
                const auto vtrans = (bt_hll * a_star - ft_hll) / bn;
                const auto invg2 =
                    (1.0 - (a_star * a_star + dot(vtrans, vtrans)));
                const auto vsdB = (a_star * bn + dot(bt_hll, vtrans));
                const auto p_star =
                    -a_star * (fhlle - bn * vsdB) + fhllm + bn * bn * invg2;
                const auto es =
                    cfac * (ws * etot - mn + p_star * a_star - vsdB * bn);
                const auto mn = (es + p_star) * a_star - vsdB * bn;
                const auto mtrans =
                    cfac * (-bn * (bt_hll * invg2 + vsdB * vtrans) +
                            ws * umtrans - fmtrans);

                us.den = ds;
                us.mom = mn * nhat + mtrans;
                us.nrg = es - ds;
                us.mag = bn * nhat + bt_hll;
            }

            //------Return the HLLC flux
            return f + (us - u) * ws - us * vface;
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
    }
}   // namespace simbi::hydro::rmhd

namespace simbi::hydro {
    // HLLC flux function
    template <is_hydro_primitive_c primitive_t>
    DUAL auto hllc_flux(
        const primitive_t& primL,
        const primitive_t& primR,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real vface,
        real gamma,
        ShockWaveLimiter shock_smoother
    )
    {
        if constexpr (primitive_t::regime == Regime::NEWTONIAN) {
            return newtonian::hllc_flux(
                primL,
                primR,
                nhat,
                vface,
                gamma,
                shock_smoother
            );
        }
        else if constexpr (primitive_t::regime == Regime::SRHD) {
            return srhd::hllc_flux(
                primL,
                primR,
                nhat,
                vface,
                gamma,
                shock_smoother
            );
        }
        else if constexpr (primitive_t::regime == Regime::RMHD) {
            return rmhd::hllc_flux(
                primL,
                primR,
                nhat,
                vface,
                gamma,
                shock_smoother
            );
        }
    }
}   // namespace simbi::hydro

#endif
