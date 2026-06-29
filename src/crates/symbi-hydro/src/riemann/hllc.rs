// =============================================================================
// riemann/hllc.rs
//
// the HLLC three-wave riemann solvers — one function per regime, all
// rotationally and dimensionally invariant (nhat-parametrized, generic over
// `S: Scalar` and `const D: usize`). a `ShockwaveLimiter` parameter
// selects the variant (Standard / Fleischmann LM / Quirk-fallback); the
// relativistic regimes ignore it.
//
//   newtonian  `hllc`       — toro eq 10.37-10.39 star state, +/- fleischmann LM.
//   srhd       `hllc_srhd`  — mignone & bodo (2005) star state.
//   rmhd       `hllc_rmhd`  — mignone & bodo (2006), null/non-null-B branch.
//
// every solver is GPU-traceable (`S::branch` / `S::select` on the
// carrier-generic mask) and `vface`-aware (the ALE grid velocity is
// subtracted from the conserved flux post-star).
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::{Scalar, Selectable};
use crate::eos::Eos;
use crate::state::{Prim, Cons};
use crate::regime::Regime;
use crate::newtonian::Newtonian;
use crate::newtonian_mhd::NewtonianMhd;
use crate::mhd_state::{MhdPrim, MhdCons};
use crate::rmhd::Rmhd;
use crate::dissipation::{adaptive_phi, quirk_strong_shock, ShockwaveLimiter};
use super::hlle::hlle;

use super::{DIVZERO_GUARD, NULL_FIELD_THRESHOLD};

/// the Quirk fallback gate shared by every regime — `D > 1` AND
/// `shock_smoother == Quirk`. the `D > 1` half is a compile-time check
/// (`const D: usize` is fixed per monomorphization, so the dead branch
/// drops at codegen for D = 1), the `if constexpr (rank > 1)` guard.
/// the `Quirk` half is runtime, but at S = Gv trace time `shock_smoother`
/// is fixed by the host that built the trace, so the match below
/// monomorphizes too — no per-cell smoother branch leaks into the kernel.
#[inline]
fn quirk_gate_active<const D: usize>(shock_smoother: ShockwaveLimiter) -> bool {
    D > 1 && matches!(shock_smoother, ShockwaveLimiter::Quirk)
}

// =============================================================================
// newtonian HLLC — toro section 9.5.2 adaptive estimates + fleischmann LM.
// =============================================================================

/// wave properties for newtonian HLLC: signal speeds + contact speed.
/// implements toro section 9.5.2 adaptive estimates (PVRS / two-rarefaction
/// / two-shock). returns `(s_l, s_r, s_star)`.
#[inline]
fn wave_properties<S: Scalar>(
    rho_l: S, rho_r: S,
    pre_l: S, pre_r: S,
    vn_l: S, vn_r: S,
    cs_l: S, cs_r: S,
    gamma: S,
) -> (S, S, S) {
    let half = S::from_f64(0.5);
    let one = S::ONE;
    let two = S::from_f64(2.0);

    // pvrs estimate
    let rho_bar = half * (rho_l + rho_r);
    let c_bar = half * (cs_l + cs_r);
    let pvrs = half * (pre_l + pre_r) - half * (vn_r - vn_l) * rho_bar * c_bar;
    let p_min = pre_l.min(pre_r);
    let p_max = pre_l.max(pre_r);

    let q_user = S::from_f64(2.0);

    // compute all three estimates unconditionally, select via mask.
    let p_pvrs = S::ZERO.max(pvrs);

    // two-rarefaction case
    let gf = (gamma - one) / (two * gamma);
    let pl_pow = pre_l.powf(gf);
    let pr_pow = pre_r.powf(gf);
    let num = cs_l + cs_r - half * (gamma - one) * (vn_r - vn_l);
    let den = cs_l / pl_pow + cs_r / pr_pow;
    let arg = num / den;
    let p_rarefaction = S::select(arg.cmp_gt(S::ZERO), arg.powf(one / gf), S::ZERO);

    // two-shock case
    let gp1 = gamma + one;
    let gm1 = gamma - one;
    let alpha_l = two / (gp1 * rho_l);
    let alpha_r = two / (gp1 * rho_r);
    let beta_l = gm1 / gp1 * pre_l;
    let beta_r = gm1 / gp1 * pre_r;
    let p0 = S::ZERO.max(pvrs);
    let g_l = (alpha_l / (p0 + beta_l)).sqrt();
    let g_r = (alpha_r / (p0 + beta_r)).sqrt();
    let p_shock = S::ZERO.max((g_l * pre_l + g_r * pre_r - (vn_r - vn_l)) / (g_l + g_r));

    // pvrs if the pressure ratio is mild AND pvrs is bounded, else rarefaction
    // (if pvrs <= p_min) or shock. mask AND uses `&` on `S::Mask` (the carrier's
    // bitwise BitAnd; not native `&&`, which would lock to a host carrier).
    let cond_pvrs = (p_max / p_min).cmp_le(q_user)
        & p_min.cmp_le(pvrs) & pvrs.cmp_le(p_max);
    let cond_rarefaction = pvrs.cmp_le(p_min);
    let p_else = S::select(cond_rarefaction, p_rarefaction, p_shock);
    let p_star = S::select(cond_pvrs, p_pvrs, p_else);

    // q factors (toro eq 9.43). carrier gate: both select arms trace at S = Gv,
    // so clamp the radicand to >= 0 BEFORE the sqrt (matches the RMHD HLLC disc
    // clamp). on the shock arm (p_star > pre_k) the radicand is already >= 1, so
    // the clamp is identity there and only neutralizes the discarded arm.
    let gp1_2g = (gamma + one) / (two * gamma);
    let q_l_alt = (one + gp1_2g * (p_star / pre_l - one)).max(S::ZERO).sqrt();
    let q_l = S::select(p_star.cmp_le(pre_l), one, q_l_alt);
    let q_r_alt = (one + gp1_2g * (p_star / pre_r - one)).max(S::ZERO).sqrt();
    let q_r = S::select(p_star.cmp_le(pre_r), one, q_r_alt);

    // signal speeds
    let s_l = vn_l - cs_l * q_l;
    let s_r = vn_r + cs_r * q_r;

    // contact wave speed (toro eq 10.37 — robust form)
    let s_star = (pre_r - pre_l + rho_l * vn_l * (s_l - vn_l) - rho_r * vn_r * (s_r - vn_r))
        / (rho_l * (s_l - vn_l) - rho_r * (s_r - vn_r));

    (s_l, s_r, s_star)
}

/// the SINGLE newtonian HLLC star state. given one side `(prim, u_k, s_k)`
/// and the contact `(s_star, chi_k = rho * (s_k - vn))`, build the
/// intermediate conserved state per toro eq 10.39. nhat-parametrized: the
/// normal momentum component swaps `s_star * nhat`; transverse components
/// flow through unchanged.
#[inline]
fn star_state<S: Scalar, const D: usize>(
    prim: &Prim<S, D>,
    u_k: &Cons<S, D>,
    s_k: S,
    s_star: S,
    chi_k: S,
    nhat: &Tensor<S, D>,
) -> Cons<S, D> {
    let vn = prim.vel.dot(nhat);
    let omega = (s_k - vn) / (s_k - s_star);
    let den_star = prim.rho * omega;
    // mom_star = den_star * (vel - vn * nhat + s_star * nhat)
    //          = den_star * vel + den_star * (s_star - vn) * nhat
    let mom_star = prim.vel.scale(den_star) + nhat.scale(den_star * (s_star - vn));
    let nrg_star = den_star
        * (u_k.nrg / prim.rho + (s_star - vn) * (s_star + prim.pre / chi_k));
    Cons { den: den_star, mom: mom_star, nrg: nrg_star }
}

/// HLLC for newtonian (compressible Euler) — toro eq 10.37-10.39. ONE function
/// for all dimensions / directions / shock-limiter modes.
///
/// `shock_smoother`:
///   - `Standard`     — plain HLLC.
///   - `Fleischmann`  — symmetric flux (fleischmann eq 11) with adaptive phi.
///   - `Quirk`        — in `D > 1`, falls back to HLLE per-cell when
///                      `quirk_strong_shock` fires (relative pressure jump
///                      exceeds `QUIRK_THRESHOLD`).
pub fn hllc<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> Cons<S, D> {
    let regime = Newtonian;

    // Quirk fallback to HLLE — `D > 1` is a compile-time const guard, the
    // `Quirk` arm matches at host build time; per-cell mask via
    // `quirk_strong_shock` picks HLLE for shocked faces, HLLC for everything
    // else. the early-return lives in the `constexpr if (rank > 1)` guard.
    if quirk_gate_active::<D>(shock_smoother) {
        let mask = quirk_strong_shock(prim_l.pre, prim_r.pre);
        return S::branch(mask,
            || hlle(&regime, eos, prim_l, prim_r, nhat, vface),
            || hllc_newtonian_body(eos, prim_l, prim_r, nhat, vface, ShockwaveLimiter::Standard),
        );
    }

    hllc_newtonian_body(eos, prim_l, prim_r, nhat, vface, shock_smoother)
}

/// the newtonian HLLC body — Standard / Fleischmann star-state dispatch,
/// no Quirk handling (the outer `hllc` already routed Quirk + strong-shock
/// cells to HLLE before reaching this point). callable directly for
/// regression diff harnesses that want to bypass the Quirk gate.
#[inline]
fn hllc_newtonian_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> Cons<S, D> {
    let regime = Newtonian;
    let u_l = prim_l.to_conserved(eos);
    let u_r = prim_r.to_conserved(eos);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);

    let cs_l = eos.sound_speed(prim_l.rho, prim_l.pre);
    let cs_r = eos.sound_speed(prim_r.rho, prim_r.pre);

    let vn_l = prim_l.vel.dot(nhat);
    let vn_r = prim_r.vel.dot(nhat);

    let gamma = eos.gamma();
    let (s_l, s_r, s_star) = wave_properties(
        prim_l.rho, prim_r.rho, prim_l.pre, prim_r.pre,
        vn_l, vn_r, cs_l, cs_r, gamma,
    );

    let chi_l = prim_l.rho * (s_l - vn_l);
    let chi_r = prim_r.rho * (s_r - vn_r);

    match shock_smoother {
        // standard HLLC: branchless three-way dispatch on the signal speeds.
        // the upwind side uses its OWN star state (toro 10.21); supersonic
        // states pass through with the ALE `vface` correction.
        ShockwaveLimiter::Standard | ShockwaveLimiter::Quirk => {
            S::branch(s_l.cmp_ge(vface),
                || f_l - u_l * vface,
                || S::branch(s_r.cmp_le(vface),
                    || f_r - u_r * vface,
                    || S::branch(s_star.cmp_ge(vface),
                        || {
                            let us = star_state(prim_l, &u_l, s_l, s_star, chi_l, nhat);
                            f_l + (us - u_l) * s_l - us * vface
                        },
                        || {
                            let us = star_state(prim_r, &u_r, s_r, s_star, chi_r, nhat);
                            f_r + (us - u_r) * s_r - us * vface
                        },
                    )
                )
            )
        }
        // fleischmann et al. (2020) eq 11: symmetric central+star flux with an
        // adaptive dissipation factor `phi`. recovers standard HLLC at phi=1
        // (supersonic) and central differencing at phi=0 (zero-mach).
        ShockwaveLimiter::Fleischmann => {
            let u_star_l = star_state(prim_l, &u_l, s_l, s_star, chi_l, nhat);
            let u_star_r = star_state(prim_r, &u_r, s_r, s_star, chi_r, nhat);

            let phi = adaptive_phi(prim_l, prim_r, nhat, gamma);
            let s_l_lm = phi * s_l;
            let s_r_lm = phi * s_r;

            let face_star = <Cons<S, D> as Selectable<S>>::select(
                s_star.cmp_ge(vface), u_star_l, u_star_r,
            );
            let half = S::from_f64(0.5);
            (f_l + f_r) * half
                + ((u_star_l - u_l) * s_l_lm
                    + (u_star_l - u_star_r) * s_star.abs()
                    + (u_star_r - u_r) * s_r_lm)
                    * half
                - face_star * vface
        }
    }
}

// =============================================================================
// SRHD HLLC (mignone-bodo 2005) — relativistic; no Fleischmann LM correction.
// =============================================================================

/// contact properties for SRHD: solve quadratic on HLL intermediate state.
/// returns `(a_star, p_star)` — contact wave speed and pressure.
#[inline]
fn srhd_contact_props<S: Scalar, const D: usize>(
    u_l: &Cons<S, D>,
    u_r: &Cons<S, D>,
    f_l: &Cons<S, D>,
    f_r: &Cons<S, D>,
    nhat: &Tensor<S, D>,
    a_l: S,
    a_r: S,
) -> (S, S) {
    let inv = S::ONE / (a_r - a_l);
    let hll_den = (*u_r * a_r - *u_l * a_l - *f_r + *f_l) * inv;
    let hll_flux = (*f_l * a_r - *f_r * a_l + (*u_r - *u_l) * (a_r * a_l)) * inv;

    // srhd total energy: e = tau + D (nrg + den).
    let ee = hll_den.nrg + hll_den.den;
    let s_norm = hll_den.mom.dot(nhat);
    let fe = hll_flux.nrg + hll_flux.den;
    let fs_norm = hll_flux.mom.dot(nhat);

    // quadratic: a x^2 + b x + c = 0 with numerically-stable sign-of-b form.
    let aa = fe;
    let bb = -(ee + fs_norm);
    let cc = s_norm;
    let disc = bb * bb - S::from_f64(4.0) * aa * cc;
    let disc_sqrt = disc.abs().sqrt();
    let sgn_b = S::select(bb.cmp_ge(S::ZERO), S::ONE, -S::ONE);
    let half = S::from_f64(0.5);
    let quad = -half * (bb + sgn_b * disc_sqrt);
    // guard the contact-speed divide against the degenerate `quad -> 0` root (bb == 0 with fe or
    // s_norm == 0): unguarded this returns NaN/Inf and poisons the flux. mirrors the proven RMHD
    // guard (the `a_star` select above). Gv evaluates both arms, but the Inf is selected away, never
    // combined into an output — the same carrier-safe pattern the RMHD path uses.
    let a_star = S::select(quad.abs().cmp_gt(S::from_f64(DIVZERO_GUARD)), cc / quad, S::ZERO);
    let p_star = -a_star * fe + fs_norm;
    (a_star, p_star)
}

/// SRHD star state: intermediate state between contact and signal wave.
/// uses `(a, a_star, p_star)` from mignone-bodo (2005).
#[inline]
fn srhd_star_state<S: Scalar, const D: usize>(
    prim: &Prim<S, D>,
    cons: &Cons<S, D>,
    a: S,
    a_star: S,
    p_star: S,
    nhat: &Tensor<S, D>,
) -> Cons<S, D> {
    let vn = prim.vel.dot(nhat);
    let ee = cons.nrg + cons.den;
    let fac = S::ONE / (a - a_star);
    let ds = fac * (a - vn) * cons.den;
    let ms = cons.mom.scale(a - vn).scale(fac) + nhat.scale((p_star - prim.pre) * fac);
    let es = fac * (ee * (a - vn) + p_star * a_star - prim.pre * vn);
    // srhd convention: nrg = tau = e - D.
    Cons { den: ds, mom: ms, nrg: es - ds }
}

/// HLLC for special-relativistic hydrodynamics (mignone-bodo 2005). ONE
/// function for all dimensions and directions. honors the `Quirk` fallback
/// in `D > 1`; the Fleischmann
/// LM correction does NOT apply to relativistic regimes (treated as
/// Standard if requested).
pub fn hllc_srhd<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> Cons<S, D> {
    let regime = crate::srhd::Srhd;

    // Quirk fallback — same shape as the newtonian gate. the carrier-
    // generic mask routes per cell at S = Gv; at S = f64 it's a bool short-circuit.
    if quirk_gate_active::<D>(shock_smoother) {
        let mask = quirk_strong_shock(prim_l.pre, prim_r.pre);
        return S::branch(mask,
            || hlle(&regime, eos, prim_l, prim_r, nhat, vface),
            || hllc_srhd_body(eos, prim_l, prim_r, nhat, vface),
        );
    }

    hllc_srhd_body(eos, prim_l, prim_r, nhat, vface)
}

/// the SRHD HLLC body (no Quirk gate). split out so the outer function can
/// route Quirk + strong-shock cells to HLLE without re-emitting the body.
#[inline]
fn hllc_srhd_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
) -> Cons<S, D> {
    let regime = crate::srhd::Srhd;
    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (a_l, a_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);

    S::branch(a_l.cmp_ge(vface),
        || f_l - u_l * vface,
        || S::branch(a_r.cmp_le(vface),
            || f_r - u_r * vface,
            || {
                let (a_star, p_star) = srhd_contact_props(&u_l, &u_r, &f_l, &f_r, nhat, a_l, a_r);
                S::branch(a_star.cmp_ge(vface),
                    || {
                        let us = srhd_star_state(prim_l, &u_l, a_l, a_star, p_star, nhat);
                        f_l + (us - u_l) * a_l - us * vface
                    },
                    || {
                        let us = srhd_star_state(prim_r, &u_r, a_r, a_star, p_star, nhat);
                        f_r + (us - u_r) * a_r - us * vface
                    },
                )
            }
        )
    )
}

// =============================================================================
// RMHD HLLC (mignone-bodo 2006) — null vs non-null normal B-field branch.
// =============================================================================

/// HLLC for relativistic MHD — three-wave solver resolving the contact wave.
/// builds on the HLL intermediate state, solves a quadratic for the contact
/// speed `a_star`, branches on whether the normal B-field is null. ONE
/// function for all dimensions and directions. carrier-generic over `S`.
/// honors the `Quirk` fallback in `D > 1`; Fleischmann LM does not apply to relativistic
/// regimes (treated as Standard if requested). reads the pressure jump
/// from the hydro half of the MHD primitive (`prim_l.hydro.pre`).
pub fn hllc_rmhd<S: Scalar, const D: usize>(
    regime: &Rmhd,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> MhdCons<S, D> {
    // Quirk fallback. RMHD's primitive nests hydro inside MhdPrim, so the
    // pressure for the detector lives at `prim_l.hydro.pre`.
    if quirk_gate_active::<D>(shock_smoother) {
        let mask = quirk_strong_shock(prim_l.hydro.pre, prim_r.hydro.pre);
        return S::branch(mask,
            || hlle(regime, eos, prim_l, prim_r, nhat, vface),
            || hllc_rmhd_body(regime, eos, prim_l, prim_r, nhat, vface),
        );
    }

    hllc_rmhd_body(regime, eos, prim_l, prim_r, nhat, vface)
}

/// the RMHD HLLC body (no Quirk gate). split out so the outer function can
/// route Quirk + strong-shock cells to HLLE without re-emitting the body.
fn hllc_rmhd_body<S: Scalar, const D: usize>(
    regime: &Rmhd,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
) -> MhdCons<S, D> {
    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (a_l, a_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);

    S::branch(a_l.cmp_ge(vface),
        || f_l - u_l * vface,
        || S::branch(a_r.cmp_le(vface),
            || f_r - u_r * vface,
            || {
                let inv = S::ONE / (a_r - a_l);

                // HLL intermediate state + flux.
                let hll_state = (u_r * a_r - u_l * a_l - f_r + f_l) * inv;
                let hll_flux = (f_l * a_r - f_r * a_l + (u_r - u_l) * (a_l * a_r)) * inv;

                // normal B from HLL state (continuous across the interface).
                let bn = hll_state.mag.dot(nhat);
                let bt_hll = hll_state.mag - nhat.scale(bn);

                let uhlld = hll_state.den;
                let uhllm = hll_state.mom.dot(nhat);
                let uhlle = hll_state.nrg + uhlld;

                let fhllm = hll_flux.mom.dot(nhat);
                let fhlle = hll_flux.nrg + hll_flux.den;
                let ft_hll = hll_flux.mag - nhat.scale(hll_flux.mag.dot(nhat));

                // contact-wave quadratic: compute null-B AND non-null-B
                // coefficients in parallel, select via mask.
                let null_cond = bn.abs().cmp_lt(S::from_f64(NULL_FIELD_THRESHOLD));
                let fdb = ft_hll.dot(&bt_hll);
                let bpsq = bt_hll.dot(&bt_hll);
                let fbpsq = ft_hll.dot(&ft_hll);
                let a_coeff = S::select(null_cond, fhlle, fhlle - fdb);
                let b_coeff = S::select(null_cond,
                    -(fhllm + uhlle),
                    -(fhllm + uhlle) + bpsq + fbpsq);
                let c_coeff = S::select(null_cond, uhllm, uhllm - fdb);

                let disc = (b_coeff * b_coeff - S::from_f64(4.0) * a_coeff * c_coeff).max(S::ZERO);
                let sgn_b = S::select(b_coeff.cmp_ge(S::ZERO), S::ONE, -S::ONE);
                let quad = S::from_f64(-0.5) * (b_coeff + sgn_b * disc.sqrt());
                let a_star = S::select(
                    quad.abs().cmp_gt(S::from_f64(DIVZERO_GUARD)),
                    c_coeff / quad,
                    S::ZERO,
                );

                // safe_bn: avoid 0/0 in the non-null path when bn is tiny;
                // when null_cond fires the non-null arm is discarded by select.
                let safe_bn = S::select(null_cond, S::ONE, bn);

                // per-side star state + HLLC flux. carrier-generic via select.
                let side_flux = |u: &MhdCons<S, D>,
                                 f: &MhdCons<S, D>,
                                 prim_side: &MhdPrim<S, D>,
                                 ws: S|
                    -> MhdCons<S, D>
                {
                    let mn = u.mom.dot(nhat);
                    let umtrans = u.mom - nhat.scale(mn);
                    let fmtrans = f.mom - nhat.scale(f.mom.dot(nhat));
                    let etot = u.nrg + u.den;
                    let cfac = S::ONE / (ws - a_star);

                    let vn = prim_side.vel.dot(nhat);
                    let vs = (ws - vn) / (ws - a_star);
                    let ds = vs * u.den;

                    // null-B star state.
                    let p_null = -a_star * fhlle + fhllm;
                    let es_null = cfac * (ws * etot - mn + p_null * a_star);
                    let mn_null = (es_null + p_null) * a_star;
                    let btrans_side = prim_side.mag - nhat.scale(prim_side.mag.dot(nhat));
                    let us_null = MhdCons {
                        hydro: Cons {
                            den: ds,
                            mom: nhat.scale(mn_null) + umtrans.scale(vs),
                            nrg: es_null - ds,
                        },
                        mag: nhat.scale(bn) + btrans_side.scale(vs),
                    };

                    // non-null-B star state (safe_bn guards division).
                    let vtrans = (bt_hll.scale(a_star) - ft_hll).scale(S::ONE / safe_bn);
                    let invg2 = S::ONE - (a_star * a_star + vtrans.dot(&vtrans));
                    let vsdb = a_star * safe_bn + bt_hll.dot(&vtrans);
                    let p_nn = -a_star * (fhlle - safe_bn * vsdb) + fhllm
                        + safe_bn * safe_bn * invg2;
                    let es_nn = cfac * (ws * etot - mn + p_nn * a_star - vsdb * safe_bn);
                    let mn_nn = (es_nn + p_nn) * a_star - vsdb * safe_bn;
                    let mtrans = (umtrans.scale(ws) - fmtrans
                        - (bt_hll.scale(invg2) + vtrans.scale(vsdb)).scale(safe_bn))
                        .scale(cfac);
                    let us_nn = MhdCons {
                        hydro: Cons {
                            den: ds,
                            mom: nhat.scale(mn_nn) + mtrans,
                            nrg: es_nn - ds,
                        },
                        mag: nhat.scale(safe_bn) + bt_hll,
                    };

                    let us = <MhdCons<S, D> as Selectable<S>>::select(null_cond, us_null, us_nn);
                    *f + (us - *u) * ws - us * vface
                };

                let flux_l = side_flux(&u_l, &f_l, prim_l, a_l);
                let flux_r = side_flux(&u_r, &f_r, prim_r, a_r);
                <MhdCons<S, D> as Selectable<S>>::select(a_star.cmp_gt(S::ZERO), flux_l, flux_r)
            }
        )
    )
}

// =============================================================================
// newtonian MHD HLLC (Li 2005 / Gurski 2004) — contact-resolving 3-wave solver:
// S_L < S_M (contact) < S_R. transverse B is CONTINUOUS across the contact
// (HLL-averaged) — the rotational (alfven) discontinuities are NOT resolved
// (that is HLLD's job). consistent (F(U,U) == F(U)); physicality-gated to HLLE.
// =============================================================================

/// the Newtonian ideal-MHD HLLC flux. `shock_smoother` enables the D>1 Quirk
/// strong-shock fallback to HLLE (matching the other regimes' HLLC).
pub fn hllc_newtonian<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> MhdCons<S, D> {
    if quirk_gate_active::<D>(shock_smoother) {
        let mask = quirk_strong_shock(prim_l.hydro.pre, prim_r.hydro.pre);
        return S::branch(
            mask,
            || hlle(&NewtonianMhd, eos, prim_l, prim_r, nhat, vface),
            || hllc_nmhd_body(eos, prim_l, prim_r, nhat, vface),
        );
    }
    hllc_nmhd_body(eos, prim_l, prim_r, nhat, vface)
}

fn hllc_nmhd_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
) -> MhdCons<S, D> {
    let zero = S::ZERO;
    let one = S::ONE;
    let half = S::from_f64(0.5);
    let eps = S::from_f64(DIVZERO_GUARD);
    let regime = NewtonianMhd;

    let hlle_flux = hlle(&regime, eos, prim_l, prim_r, nhat, vface);

    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (sll, srl) = regime.wave_speeds(eos, prim_l, nhat);
    let (slr, srr) = regime.wave_speeds(eos, prim_r, nhat);
    let s_l = sll.min(srl);
    let s_r = slr.max(srr);

    let un_l = prim_l.vel.dot(nhat);
    let un_r = prim_r.vel.dot(nhat);
    let bn = (prim_l.mag.dot(nhat) + prim_r.mag.dot(nhat)) * half;
    let rho_l = prim_l.rho;
    let rho_r = prim_r.rho;
    let pt_l = prim_l.pre + half * prim_l.mag.dot(&prim_l.mag);
    let pt_r = prim_r.pre + half * prim_r.mag.dot(&prim_r.mag);

    let cl = (s_l - un_l) * rho_l;
    let cr = (s_r - un_r) * rho_r;
    let dm = cr - cl;
    let dm_s = S::select(dm.abs().cmp_lt(eps), eps, dm);
    let s_m = (cr * un_r - cl * un_l - pt_r + pt_l) / dm_s;
    let pt_star = (cr * pt_l - cl * pt_r + cl * cr * (un_r - un_l)) / dm_s;

    // HLL state -> the transverse B held CONTINUOUS across the contact.
    let inv_dwave = one / S::select((s_r - s_l).abs().cmp_lt(eps), eps, s_r - s_l);
    let u_hll = (u_r * s_r - u_l * s_l - (f_r - f_l)) * inv_dwave;
    let tang = |v: &Tensor<S, D>, vn: S| -> Tensor<S, D> { *v - nhat.scale(vn) };
    let bt_star = tang(&u_hll.mag, u_hll.mag.dot(nhat));
    let b_star = nhat.scale(bn) + bt_star;

    // per-side single-star (*) state: normal velocity S_M, transverse v from the
    // transverse-momentum jump with the continuous B*, energy from the energy jump.
    let star = |u_k: &MhdCons<S, D>, prim_k: &MhdPrim<S, D>, f_k: &MhdCons<S, D>,
                s_k: S, un_k: S, rho_k: S, pt_k: S, c_k: S|
        -> (MhdCons<S, D>, MhdCons<S, D>, S) {
        let smk_s = S::select((s_k - s_m).abs().cmp_lt(eps), eps, s_k - s_m);
        let rho_star = rho_k * (s_k - un_k) / smk_s;
        let c_safe = S::select(c_k.abs().cmp_lt(eps), eps, c_k); // rho_K(S_K - u_K)
        let vt_k = tang(&prim_k.vel, un_k);
        let bt_k = tang(&prim_k.mag, bn);
        let vt_star = vt_k - (bt_star - bt_k).scale(bn / c_safe);
        let v_star = nhat.scale(s_m) + vt_star;
        let e_k = u_k.nrg;
        let vdb_k = prim_k.vel.dot(&prim_k.mag);
        let vdb_s = v_star.dot(&b_star);
        let e_star = ((s_k - un_k) * e_k - pt_k * un_k + pt_star * s_m + bn * (vdb_k - vdb_s)) / smk_s;
        let u_star = MhdCons { hydro: Cons { den: rho_star, mom: v_star.scale(rho_star), nrg: e_star }, mag: b_star };
        let f_star = *f_k + (u_star - *u_k) * s_k;
        (u_star, f_star, rho_star)
    };
    let (us_l, fs_l, rs_l) = star(&u_l, prim_l, &f_l, s_l, un_l, rho_l, pt_l, cl);
    let (us_r, fs_r, rs_r) = star(&u_r, prim_r, &f_r, s_r, un_r, rho_r, pt_r, cr);

    let reg = |f: MhdCons<S, D>, u: MhdCons<S, D>| -> MhdCons<S, D> { f - u * vface };
    let pick = MhdCons::select(vface.cmp_lt(s_l), reg(f_l, u_l),
               MhdCons::select(vface.cmp_lt(s_m), reg(fs_l, us_l),
               MhdCons::select(vface.cmp_lt(s_r), reg(fs_r, us_r),
               reg(f_r, u_r))));
    let ok = rs_l.cmp_gt(zero) & rs_r.cmp_gt(zero) & pt_star.cmp_gt(zero);
    MhdCons::select(ok, pick, hlle_flux)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn hllc_uniform_state_1d() {
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.5]), pre: 1.0 };
        let nhat = Tensor::unit(0);
        let flux = hllc(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard);
        let regime = Newtonian;
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
        assert!(approx(flux.mom[0], exact.mom[0]));
        assert!(approx(flux.nrg, exact.nrg));
    }

    #[test]
    fn hllc_uniform_state_2d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim { rho: 2.0, vel: Tensor::new([0.5, -0.3]), pre: 2.5 };

        let nhat_x = Tensor::unit(0);
        let flux_x = hllc(&eos, &prim, &prim, &nhat_x, 0.0, ShockwaveLimiter::Standard);
        let regime = Newtonian;
        let exact_x = regime.to_flux(&prim, &nhat_x, &eos);
        assert!(approx(flux_x.den, exact_x.den));
        assert!(approx(flux_x.mom[0], exact_x.mom[0]));
        assert!(approx(flux_x.mom[1], exact_x.mom[1]));
        assert!(approx(flux_x.nrg, exact_x.nrg));

        let nhat_y = Tensor::unit(1);
        let flux_y = hllc(&eos, &prim, &prim, &nhat_y, 0.0, ShockwaveLimiter::Standard);
        let exact_y = regime.to_flux(&prim, &nhat_y, &eos);
        assert!(approx(flux_y.den, exact_y.den));
        assert!(approx(flux_y.mom[0], exact_y.mom[0]));
        assert!(approx(flux_y.mom[1], exact_y.mom[1]));
        assert!(approx(flux_y.nrg, exact_y.nrg));
    }

    #[test]
    fn hllc_sod_shock_tube() {
        let eos = IdealGas { gamma: 1.4 };
        let prim_l = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let prim_r = Prim { rho: 0.125, vel: Tensor::new([0.0]), pre: 0.1 };
        let nhat = Tensor::unit(0);

        let flux = hllc(&eos, &prim_l, &prim_r, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert!(flux.den > 0.0);
        assert!(flux.nrg > 0.0);
    }

    #[test]
    fn hllc_symmetric_2d() {
        // x-problem vs y-problem with velocities swapped — proves rotational
        // invariance of the nhat-parametrized solver.
        let eos = IdealGas { gamma: 1.4 };

        let prim_l_x = Prim { rho: 1.0, vel: Tensor::new([1.0, 0.0]), pre: 1.0 };
        let prim_r_x = Prim { rho: 0.5, vel: Tensor::new([0.0, 0.0]), pre: 0.5 };
        let flux_x = hllc(&eos, &prim_l_x, &prim_r_x, &Tensor::unit(0), 0.0, ShockwaveLimiter::Standard);

        let prim_l_y = Prim { rho: 1.0, vel: Tensor::new([0.0, 1.0]), pre: 1.0 };
        let prim_r_y = Prim { rho: 0.5, vel: Tensor::new([0.0, 0.0]), pre: 0.5 };
        let flux_y = hllc(&eos, &prim_l_y, &prim_r_y, &Tensor::unit(1), 0.0, ShockwaveLimiter::Standard);

        assert!(approx(flux_x.den, flux_y.den));
        assert!(approx(flux_x.nrg, flux_y.nrg));
        assert!(approx(flux_x.mom[0], flux_y.mom[1]));
    }

    #[test]
    fn hllc_fleischmann_uniform_matches_standard() {
        // a uniform state has zero LM correction — Fleischmann reduces to the
        // exact regime flux just like Standard.
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.01]), pre: 1.0 };
        let nhat = Tensor::unit(0);
        let flux = hllc(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Fleischmann);
        let regime = Newtonian;
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
    }

    #[test]
    fn hllc_quirk_is_a_noop_in_1d() {
        // the Quirk fallback is gated on `D > 1` (`if constexpr (rank > 1)`).
        // in 1D the gate fires `false` AT COMPILE TIME and the standard HLLC
        // body runs — so Quirk and Standard must be bit-identical even on a
        // strong-shock 1D problem (sod). protects against regression in the
        // const-generic D > 1 guard.
        let eos = IdealGas { gamma: 1.4 };
        let prim_l = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let prim_r = Prim { rho: 0.125, vel: Tensor::new([0.0]), pre: 0.1 };
        let nhat = Tensor::unit(0);
        let f_std = hllc(&eos, &prim_l, &prim_r, &nhat, 0.0, ShockwaveLimiter::Standard);
        let f_quirk = hllc(&eos, &prim_l, &prim_r, &nhat, 0.0, ShockwaveLimiter::Quirk);
        assert!(approx(f_std.den, f_quirk.den));
        assert!(approx(f_std.mom[0], f_quirk.mom[0]));
        assert!(approx(f_std.nrg, f_quirk.nrg));
    }

    #[test]
    fn hllc_quirk_smooth_state_2d_matches_standard() {
        // a smooth (uniform) 2D state has zero pressure jump — the
        // `quirk_strong_shock` detector returns false and the standard HLLC
        // body runs. Quirk must match Standard in this regime.
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.3, -0.1]), pre: 1.0 };
        let nhat = Tensor::unit(0);
        let f_std = hllc(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard);
        let f_quirk = hllc(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Quirk);
        assert!(approx(f_std.den, f_quirk.den));
        assert!(approx(f_std.mom[0], f_quirk.mom[0]));
        assert!(approx(f_std.mom[1], f_quirk.mom[1]));
        assert!(approx(f_std.nrg, f_quirk.nrg));
    }

    #[test]
    fn hllc_quirk_strong_shock_2d_falls_back_to_hlle() {
        // a strong pressure jump in 2D triggers the Quirk detector — the flux
        // MUST equal the HLLE flux on that face (not HLLC). this is the actual
        // safety guarantee Quirk provides: carbuncle-prone shocks take the
        // dissipative two-wave solver instead of the contact-resolving three-
        // wave one.
        let eos = IdealGas { gamma: 1.4 };
        let prim_l = Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 };
        let prim_r = Prim { rho: 0.125, vel: Tensor::new([0.0, 0.0]), pre: 0.1 };
        let nhat = Tensor::unit(0);
        let regime = Newtonian;
        let f_quirk = hllc(&eos, &prim_l, &prim_r, &nhat, 0.0, ShockwaveLimiter::Quirk);
        let f_hlle = hlle(&regime, &eos, &prim_l, &prim_r, &nhat, 0.0);
        assert!(approx(f_quirk.den, f_hlle.den), "Quirk strong shock must equal HLLE: {} vs {}", f_quirk.den, f_hlle.den);
        assert!(approx(f_quirk.mom[0], f_hlle.mom[0]));
        assert!(approx(f_quirk.mom[1], f_hlle.mom[1]));
        assert!(approx(f_quirk.nrg, f_hlle.nrg));
    }

    #[test]
    fn quirk_strong_shock_detector_threshold() {
        // direct test of the detector — fires when `|pr - pl| / min(pl, pr) > 1e-4`.
        use crate::dissipation::quirk_strong_shock;
        // exact threshold: 1e-4 relative jump (1.0 vs 1.0001) — just at the line.
        assert!(!quirk_strong_shock::<f64>(1.0, 1.00005)); // 5e-5 < 1e-4, no shock
        assert!( quirk_strong_shock::<f64>(1.0, 1.0002));  // 2e-4 > 1e-4, shock
        // sod-class jump (1.0 vs 0.1) is firmly a shock — 9x relative.
        assert!( quirk_strong_shock::<f64>(1.0, 0.1));
        // symmetric — pl/pr swap doesn't change the verdict.
        assert!( quirk_strong_shock::<f64>(0.1, 1.0));
    }

    #[test]
    fn hllc_srhd_uniform_state() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = crate::srhd::Srhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.3]), pre: 1.0 };
        let nhat = Tensor::unit(0);
        let flux = hllc_srhd(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard);
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
        assert!(approx(flux.mom[0], exact.mom[0]));
        assert!(approx(flux.nrg, exact.nrg));
    }

    #[test]
    fn hllc_rmhd_uniform_state() {
        let eos = IdealGas { gamma: 2.0 };
        let regime = Rmhd;
        let prim = MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([0.3, 0.0, 0.0]), pre: 1.0 },
            mag: Tensor::new([0.5, 1.0, 0.0]),
        };
        let nhat = Tensor::unit(0);
        let flux = hllc_rmhd(&regime, &eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard);
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den), "den: {} vs {}", flux.den, exact.den);
        for dd in 0..3 {
            assert!(approx(flux.mom[dd], exact.mom[dd]), "mom[{}]: {} vs {}", dd, flux.mom[dd], exact.mom[dd]);
        }
        assert!(approx(flux.nrg, exact.nrg), "nrg: {} vs {}", flux.nrg, exact.nrg);
    }

    #[test]
    fn hllc_rmhd_balsara_shock() {
        let eos = IdealGas { gamma: 2.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0, 0.0]), pre: 1.0 },
            mag: Tensor::new([0.5, 1.0, 0.0]),
        };
        let prim_r = MhdPrim {
            hydro: Prim { rho: 0.125, vel: Tensor::new([0.0, 0.0, 0.0]), pre: 0.1 },
            mag: Tensor::new([0.5, -1.0, 0.0]),
        };
        let nhat = Tensor::unit(0);
        let flux = hllc_rmhd(&regime, &eos, &prim_l, &prim_r, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert!(flux.den > 0.0, "density flux should be positive: {}", flux.den);
    }

    // ---- Newtonian MHD HLLC ----

    fn nm_prim(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> MhdPrim<f64, 3> {
        MhdPrim { hydro: Prim { rho, vel: Tensor::new(v), pre: p }, mag: Tensor::new(b) }
    }

    fn assert_mhd_flux_eq(got: &MhdCons<f64, 3>, want: &MhdCons<f64, 3>, ctx: &str) {
        assert!(approx(got.den, want.den), "{ctx} den: {} vs {}", got.den, want.den);
        for dd in 0..3 {
            assert!(approx(got.mom[dd], want.mom[dd]), "{ctx} mom[{dd}]: {} vs {}", got.mom[dd], want.mom[dd]);
            assert!(approx(got.mag[dd], want.mag[dd]), "{ctx} mag[{dd}]: {} vs {}", got.mag[dd], want.mag[dd]);
        }
        assert!(approx(got.nrg, want.nrg), "{ctx} nrg: {} vs {}", got.nrg, want.nrg);
    }

    #[test]
    fn hllc_newtonian_uniform_is_physical_flux() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let cases = [
            nm_prim(1.0, [0.0, 0.0, 0.0], 1.0, [0.5, 1.0, 0.0]),
            nm_prim(0.7, [0.3, -0.2, 0.1], 0.9, [0.4, 0.2, -0.6]),
            nm_prim(1.2, [-0.4, 0.0, 0.3], 1.5, [0.0, 0.8, -0.3]), // Bn = 0
        ];
        for (ii, prim) in cases.iter().enumerate() {
            let flux = hllc_newtonian(&eos, prim, prim, &nhat, 0.0, ShockwaveLimiter::Standard);
            let exact = NewtonianMhd.to_flux(prim, &nhat, &eos);
            assert_mhd_flux_eq(&flux, &exact, &format!("uniform case {ii}"));
        }
    }

    #[test]
    fn hllc_newtonian_b_zero_matches_hydro_flux() {
        let eos = IdealGas { gamma: 1.4 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let prim = nm_prim(1.3, [0.4, -0.2, 0.1], 0.7, [0.0, 0.0, 0.0]);
        let flux = hllc_newtonian(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard);
        let exact = NewtonianMhd.to_flux(&prim, &nhat, &eos);
        assert_mhd_flux_eq(&flux, &exact, "b=0 uniform");
    }

    #[test]
    fn hllc_newtonian_supersonic_upwinds() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let pl = nm_prim(1.0, [5.0, 0.2, 0.0], 1.0, [0.3, 0.5, 0.0]);
        let pr = nm_prim(0.5, [5.0, -0.1, 0.2], 0.4, [0.3, -0.4, 0.1]);
        let f = hllc_newtonian(&eos, &pl, &pr, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert_mhd_flux_eq(&f, &NewtonianMhd.to_flux(&pl, &nhat, &eos), "supersonic-right");
        // mirror: supersonic-left
        let pl2 = nm_prim(1.0, [-5.0, 0.2, 0.0], 1.0, [0.3, 0.5, 0.0]);
        let pr2 = nm_prim(0.5, [-5.0, -0.1, 0.2], 0.4, [0.3, -0.4, 0.1]);
        let f2 = hllc_newtonian(&eos, &pl2, &pr2, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert_mhd_flux_eq(&f2, &NewtonianMhd.to_flux(&pr2, &nhat, &eos), "supersonic-left");
    }

    #[test]
    fn hllc_newtonian_brio_wu_finite_and_divb_clean() {
        let eos = IdealGas { gamma: 2.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let pl = nm_prim(1.0, [0.0, 0.0, 0.0], 1.0, [0.75, 1.0, 0.0]);
        let pr = nm_prim(0.125, [0.0, 0.0, 0.0], 0.1, [0.75, -1.0, 0.0]);
        let f = hllc_newtonian(&eos, &pl, &pr, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert!(f.den.is_finite() && f.nrg.is_finite(), "finite");
        assert!(f.mag[0].abs() < 1e-12, "normal-B flux must vanish: {}", f.mag[0]);
    }

    // ---- carrier gate: newtonian wave_properties q-factor sqrt ----

    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::passes::scalarize::scalarize;
    use symbi_ir::{begin_trace, end_trace, Gv};

    // trace `wave_properties` at S = Gv, scalarize each of the three signal-speed
    // outputs, and CPU-interpret them at the given f64 state. proves the body
    // renders (non-empty LoweredFn) AND that the traced graph (which evaluates
    // BOTH select arms) produces finite results matching the f64 physics path.
    fn wave_properties_gv(state: &[f64; 9]) -> [f64; 3] {
        let names = [
            "rho_l", "rho_r", "pre_l", "pre_r", "vn_l", "vn_r", "cs_l", "cs_r", "gamma",
        ];
        begin_trace();
        let p: Vec<Gv> = names.iter().map(|n| Gv::param(n)).collect();
        let (s_l, s_r, s_star) = wave_properties::<Gv>(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
        );
        let kernel = end_trace();
        let mut out = [0.0f64; 3];
        for (kk, root) in [s_l, s_r, s_star].iter().enumerate() {
            let lowered = scalarize(&kernel.graph, root.node(), "wave_props_probe");
            assert!(!lowered.body.is_empty(), "wave_properties output {kk} rendered an empty kernel");
            out[kk] = Cpu.eval_elemental(&lowered, state)[0];
        }
        out
    }

    fn wave_properties_f64(state: &[f64; 9]) -> [f64; 3] {
        let (s_l, s_r, s_star) = wave_properties::<f64>(
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8],
        );
        [s_l, s_r, s_star]
    }

    #[test]
    fn hllc_wave_properties_carrier_equiv_strong_rarefaction() {
        // strong double rarefaction (gas pulling apart fast) drives p_star -> 0.
        // at gamma < 1 the q-factor radicand `1 + k*(p_star/pre - 1)` goes
        // NEGATIVE (k = (gamma+1)/(2*gamma) > 1). the f64 select discards the
        // q_alt arm (p_star <= pre), so the physical result is the rarefaction
        // q = 1 and is unaffected. but at S = Gv BOTH arms trace: without the
        // `.max(S::ZERO)` clamp the discarded arm would trace `sqrt(neg)` = NaN
        // into the kernel. this asserts the clamp keeps the traced graph finite
        // and bit-equal to the f64 path. gamma chosen < 1 to actually hit the
        // negative radicand (the exact landmine the clamp guards).
        let gamma = 0.8_f64;
        let cs = (gamma * 1.0 / 1.0_f64).sqrt();
        let state = [1.0, 1.0, 1.0, 1.0, -3.0, 3.0, cs, cs, gamma];

        let want = wave_properties_f64(&state);
        let got = wave_properties_gv(&state);
        for kk in 0..3 {
            assert!(got[kk].is_finite(), "gv signal speed {kk} not finite: {}", got[kk]);
            assert!(approx(want[kk], got[kk]), "carrier mismatch at {kk}: f64 {} vs gv {}", want[kk], got[kk]);
        }
    }

    #[test]
    fn hllc_wave_properties_clamp_is_identity_on_physical_state() {
        // physical gamma > 1 strong rarefaction: the radicand stays positive, so
        // the `.max(S::ZERO)` clamp is an IDENTITY. the f64 and Gv paths must
        // agree, confirming the fix does not perturb non-degenerate states.
        let gamma = 1.4_f64;
        let cs = (gamma * 1.0 / 1.0_f64).sqrt();
        let state = [1.0, 1.0, 1.0, 1.0, -3.0, 3.0, cs, cs, gamma];

        let want = wave_properties_f64(&state);
        let got = wave_properties_gv(&state);
        for kk in 0..3 {
            assert!(got[kk].is_finite(), "gv signal speed {kk} not finite");
            assert!(approx(want[kk], got[kk]), "carrier mismatch at {kk}: f64 {} vs gv {}", want[kk], got[kk]);
        }
    }
}
