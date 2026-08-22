// =============================================================================
// riemann/hllc.rs
//
// the HLLC three-wave riemann solvers — one function per regime, all
// rotationally and dimensionally invariant (nhat-parametrized, generic over
// `S: Scalar` and `const D: usize`). a `ShockwaveLimiter` parameter
// selects the variant (Standard / Fleischmann LM); the
// relativistic regimes ignore it.
//
//   newtonian  `hllc`       — toro eq 10.37-10.39 star state, +/- the HLLC+ corrections.
//   rhd       `hllc_rhd`  — mignone & bodo (2005) star state.
//   rmhd       `hllc_rmhd`  — mignone & bodo (2006), null/non-null-B branch.
//
// every solver is GPU-traceable (`S::branch` / `S::select` on the
// carrier-generic mask) and `vface`-aware (the ALE grid velocity is
// subtracted from the conserved flux post-star).
// =============================================================================

use super::hlle::hlle;
use crate::dissipation::{ShockwaveLimiter, mach_scale, shear_weight};
use crate::eos::Eos;
use crate::mhd_state::{MhdCons, MhdPrim};
use crate::newtonian::Newtonian;
use crate::newtonian_mhd::NewtonianMhd;
use crate::regime::Regime;
use crate::rmhd::Rmhd;
use crate::spatial_metric::SpatialMetric;
use crate::state::{Cons, Prim};
use symbi_algebra::Tensor;
use symbi_ir::algebra::{Scalar, Selectable};

// =============================================================================
// newtonian HLLC — toro section 9.5.2 adaptive estimates + the HLLC+ corrections.
// =============================================================================

/// wave properties for newtonian HLLC: signal speeds + contact speed.
/// implements toro section 9.5.2 adaptive estimates (pvrs / two-rarefaction
/// / two-shock). returns `(s_l, s_r, s_star)`.
#[inline]
fn wave_properties<S: Scalar>(
    rho_l: S,
    rho_r: S,
    pre_l: S,
    pre_r: S,
    vn_l: S,
    vn_r: S,
    cs_l: S,
    cs_r: S,
    gamma: S,
) -> (S, S, S) {
    let half = S::HALF;
    let one = S::ONE;
    let two = S::TWO;

    // pvrs estimate
    let rho_bar = half * (rho_l + rho_r);
    let c_bar = half * (cs_l + cs_r);
    let pvrs = half * (pre_l + pre_r) - half * (vn_r - vn_l) * rho_bar * c_bar;
    let p_min = pre_l.min(pre_r);
    let p_max = pre_l.max(pre_r);

    let q_user = S::TWO;

    // pvrs when the pressure ratio is mild and pvrs is bounded, else rarefaction
    // (if pvrs <= p_min) or shock. the mask conjunction uses `&` on `S::Mask` (the
    // carrier's bitwise BitAnd; native `&&` would lock to a host carrier).
    //
    // the estimates live in lazy `S::cond` arms — a smooth-flow face pays only
    // the pvrs arithmetic; the two-rarefaction arm's three `powf` calls and the
    // two-shock arm's roots run only on the faces that reject pvrs. an eager
    // select spelling would pay all three estimates at every face, which dominates
    // the hllc kernel's cost (~49 ns/zone against ~14 for hlle). carrier-equivalent:
    // every carrier takes the same arm for the same input, bit-identically.
    let cond_pvrs = (p_max / p_min).cmp_le(q_user) & p_min.cmp_le(pvrs) & pvrs.cmp_le(p_max);
    let p_star = S::cond(
        cond_pvrs,
        || S::ZERO.max(pvrs),
        || {
            S::cond(
                pvrs.cmp_le(p_min),
                || {
                    // two-rarefaction (toro eq 9.41)
                    let gf = (gamma - one) / (two * gamma);
                    let pl_pow = pre_l.powf(gf);
                    let pr_pow = pre_r.powf(gf);
                    let num = cs_l + cs_r - half * (gamma - one) * (vn_r - vn_l);
                    let den = cs_l / pl_pow + cs_r / pr_pow;
                    let arg = num / den;
                    S::select(arg.cmp_gt(S::ZERO), arg.powf(one / gf), S::ZERO)
                },
                || {
                    // two-shock (toro eq 9.42)
                    let gp1 = gamma + one;
                    let gm1 = gamma - one;
                    let alpha_l = two / (gp1 * rho_l);
                    let alpha_r = two / (gp1 * rho_r);
                    let beta_l = gm1 / gp1 * pre_l;
                    let beta_r = gm1 / gp1 * pre_r;
                    let p0 = S::ZERO.max(pvrs);
                    let g_l = (alpha_l / (p0 + beta_l)).sqrt();
                    let g_r = (alpha_r / (p0 + beta_r)).sqrt();
                    S::ZERO.max((g_l * pre_l + g_r * pre_r - (vn_r - vn_l)) / (g_l + g_r))
                },
            )
        },
    );

    // q factors (toro eq 9.43). carrier gate: both select arms trace at S = Gv,
    // so the radicand is clamped to >= 0 ahead of the sqrt (matching the RMHD HLLC
    // disc clamp). on the shock arm (p_star > pre_k) the radicand is already >= 1,
    // so the clamp is the identity there and acts solely on the discarded arm.
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

/// the single newtonian HLLC star state. given one side `(prim, u_k, s_k)`
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
    let nrg_star = den_star * (u_k.nrg / prim.rho + (s_star - vn) * (s_star + prim.pre / chi_k));
    Cons {
        chi: Default::default(),
        den: den_star,
        mom: mom_star,
        nrg: nrg_star,
    }
}

/// the stencil quantities the HLLC+ corrections read, which a pointwise Riemann solve cannot
/// compute for itself: both are properties of the cells around the face rather than of its two
/// states.
///
/// `pressure_ratio` is the smallest `min(p_a/p_b, p_b/p_a)` across any interface of either
/// adjoining cell, measuring how much pressure structure the neighborhood carries.
/// `shocked` is one where a characteristic speed `u -+ a` reverses sign between neighbors and
/// zero elsewhere, which separates a front from a steep hydrostatic gradient: an atmosphere
/// bound to a point mass carries a large pressure ratio across every cell and reverses nothing.
#[derive(Clone, Copy)]
pub struct HllcPlusSensors<S> {
    pub pressure_ratio: S,
    pub shocked: S,
}

/// HLLC for newtonian (compressible Euler) — toro eq 10.37-10.39. one function
/// for all dimensions / directions / shock-limiter modes.
///
/// `shock_smoother`:
///   - `Standard`     — plain HLLC.
///   - `HllcPlus`     — classical HLLC plus the two velocity-jump dissipation rescalings.
///
/// `phi_floor` raises the acoustic-dissipation scaling to at least the supplied value, for a
/// face whose flow speed reports a mach number the surrounding physics does not set. the
/// scaling's premise is that a vanishing face-normal mach number means smooth subsonic flow, so
/// the acoustic dissipation may fall with it; where a velocity is imposed instead of evolved
/// that premise is empty and the floor restores the classical amount. `None` leaves the
/// published scaling as the whole rule and adds no arithmetic to the traced graph.
#[allow(clippy::too_many_arguments)]
pub fn hllc<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
    shear: Option<HllcPlusSensors<S>>,
) -> Cons<S, D> {
    hllc_newtonian_body(eos, prim_l, prim_r, nhat, vface, shock_smoother, shear)
}

/// the newtonian HLLC body — Standard / Fleischmann star-state dispatch,
/// cells to HLLE before reaching this point). callable directly for
/// regression diff harnesses.
#[inline]
#[allow(clippy::too_many_arguments)]
fn hllc_newtonian_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
    shear: Option<HllcPlusSensors<S>>,
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
        prim_l.rho, prim_r.rho, prim_l.pre, prim_r.pre, vn_l, vn_r, cs_l, cs_r, gamma,
    );

    let chi_l = prim_l.rho * (s_l - vn_l);
    let chi_r = prim_r.rho * (s_r - vn_r);

    match shock_smoother {
        // standard HLLC: branchless three-way dispatch on the signal speeds.
        // the upwind side uses its own star state (toro 10.21); supersonic
        // states pass through with the ALE `vface` correction.
        ShockwaveLimiter::Standard => S::branch(
            s_l.cmp_ge(vface),
            || f_l - u_l * vface,
            || {
                S::branch(
                    s_r.cmp_le(vface),
                    || f_r - u_r * vface,
                    || {
                        S::branch(
                            s_star.cmp_ge(vface),
                            || {
                                let us = star_state(prim_l, &u_l, s_l, s_star, chi_l, nhat);
                                f_l + (us - u_l) * s_l - us * vface
                            },
                            || {
                                let us = star_state(prim_r, &u_r, s_r, s_star, chi_r, nhat);
                                f_r + (us - u_r) * s_r - us * vface
                            },
                        )
                    },
                )
            },
        ),
        // the anti-dissipation pressure correction (Chen et al., J. Comput. Phys. 456:111027,
        // 2022, eq. 39): standard HLLC plus an additive term that rescales one identified
        // piece of the flux — the dissipation proportional to the face's normal velocity
        // jump, `P_d = chi_l chi_r / (chi_r - chi_l) * (u_R - u_L) * (0, nhat, S_*)`, whose
        // magnitude is `rho c du`.
        //
        // that term is the low-mach accuracy defect on its own: it carries the acoustic
        // impedance, so it sits an order in `1/Ma` above the convective flux it corrects and
        // drives pressure fluctuations to `O(Ma)` where the continuous Euler system gives
        // `O(Ma^2)`. adding `(g - 1) P_d` rescales it to the convective magnitude, restoring
        // the `Ma^2` law; the correction is inert at and above the sonic point, where `g = 1`.
        //
        // the star states, both signal speeds, the contact speed and the contact pressure keep
        // their classical values, so the flux stays the classical one everywhere the velocity
        // jump vanishes. a stagnant stratified column presents `u_R = u_L` exactly, `P_d` is
        // zero there, and the hydrostatic truncation residual keeps the full pressure-jump
        // dissipation that damps it.
        ShockwaveLimiter::HllcPlus => S::branch(
            s_l.cmp_ge(vface),
            || f_l - u_l * vface,
            || {
                S::branch(
                    s_r.cmp_le(vface),
                    || f_r - u_r * vface,
                    || {
                        let hllc = S::branch(
                            s_star.cmp_ge(vface),
                            || {
                                let us = star_state(prim_l, &u_l, s_l, s_star, chi_l, nhat);
                                f_l + (us - u_l) * s_l - us * vface
                            },
                            || {
                                let us = star_state(prim_r, &u_r, s_r, s_star, chi_r, nhat);
                                f_r + (us - u_r) * s_r - us * vface
                            },
                        );
                        // `chi_l < 0 < chi_r` holds across the whole subsonic fan this branch
                        // covers, so the denominator is bounded away from zero by the sum of
                        // the two acoustic impedances.
                        let impedance = chi_l * chi_r / (chi_r - chi_l);
                        let mach = mach_scale(prim_l.vel.norm(), prim_r.vel.norm(), cs_l, cs_r);
                        // the shock restraint, carried as one where a characteristic speed
                        // reverses across the face's neighborhood and zero elsewhere. a shock
                        // needs the whole of the classical dissipation: the velocity-jump term
                        // is what damps a compression, and a face inside a stagnating front
                        // reads a low mach number while carrying exactly such a compression, so
                        // the accuracy rescaling would strip the damping where the flow most
                        // needs it. lifting the mach number to one at those faces returns the
                        // flux to classical HLLC there.
                        let shocked = shear.map_or(S::ZERO, |s| s.shocked);
                        let mach = mach + shocked * (S::ONE - mach);
                        let p_d = impedance * (vn_r - vn_l);
                        let scale = mach - S::ONE;
                        let normal = hllc
                            + Cons {
                                chi: Default::default(),
                                den: S::ZERO,
                                mom: nhat.scale(p_d * scale),
                                nrg: p_d * scale * s_star,
                            };
                        // the transverse shear viscosity that carries shock stability (Chen,
                        // Lin, Li & Yan 2020, eq. 22). the grid-aligned instability grows
                        // through the transverse velocity jump, which the normal-jump term
                        // above leaves untouched, so this damps it directly: a dissipation
                        // proportional to the part of the velocity jump lying in the face
                        // plane, weighted to appear at shocks and vanish in smooth flow.
                        //
                        // `S_K / (S_K - S_*)` takes the upwind side's signal speed, matching
                        // the star state the branch selected. the two speeds carry opposite
                        // signs across the subsonic fan, so the factor lies in [0, 1] and
                        // reduces the viscosity as the contact approaches the outer wave.
                        //
                        // the momentum flux alone carries it: the mass and energy equations
                        // are the same in one dimension and in several, and the shear wave
                        // exists only in the multidimensional momentum balance.
                        match shear {
                            None => normal,
                            Some(sensors) => {
                                let s_k = S::select(s_star.cmp_ge(vface), s_l, s_r);
                                let upwind = s_k / (s_k - s_star);
                                let dv = prim_r.vel - prim_l.vel;
                                let dv_perp = dv - nhat.scale(dv.dot(nhat));
                                // the same restraint, in the opposite polarity: the transverse
                                // viscosity exists for the front and is absent everywhere else.
                                let weight = impedance
                                    * upwind
                                    * sensors.shocked
                                    * shear_weight(sensors.pressure_ratio, mach);
                                normal
                                    + Cons {
                                        chi: Default::default(),
                                        den: S::ZERO,
                                        mom: dv_perp.scale(weight),
                                        nrg: S::ZERO,
                                    }
                            }
                        }
                    },
                )
            },
        ),
    }
}

// =============================================================================
// RHD HLLC (mignone-bodo 2005) — the relativistic three-wave solver, with the
// low-mach acoustic-dissipation scaling riding the Mignone-Bodo star states.
// =============================================================================

/// contact properties for RHD: solve quadratic on HLL intermediate state.
/// returns `(a_star, p_star)` — contact wave speed and pressure.
#[inline]
fn rhd_contact_props<S: Scalar, const D: usize>(
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

    // rhd total energy: e = tau + D (nrg + den).
    let ee = hll_den.nrg + hll_den.den;
    let s_norm = hll_den.mom.dot(nhat);
    let fe = hll_flux.nrg + hll_flux.den;
    let fs_norm = hll_flux.mom.dot(nhat);

    // quadratic: a x^2 + b x + c = 0 with numerically-stable sign-of-b form.
    let aa = fe;
    let bb = -(ee + fs_norm);
    let cc = s_norm;
    let disc = bb * bb - S::FOUR * aa * cc;
    let disc_sqrt = disc.abs().sqrt();
    let sgn_b = S::select(bb.cmp_ge(S::ZERO), S::ONE, -S::ONE);
    let half = S::HALF;
    let quad = -half * (bb + sgn_b * disc_sqrt);
    // guard the contact-speed divide against the degenerate `quad -> 0` root (bb == 0 with fe or
    // s_norm == 0), where the raw divide returns NaN/Inf and poisons the flux. mirrors the RMHD
    // `a_star` guard. Gv evaluates both arms and selects the Inf away, so it stays clear of every
    // output — the carrier-safe pattern the RMHD path uses.
    let quad_scale = bb.abs().max(disc_sqrt);
    let quad_ok = quad
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * quad_scale);
    let quad_divisor = S::select(quad_ok, quad, S::ONE);
    let a_star = S::select(quad_ok, cc / quad_divisor, S::ZERO);
    let p_star = -a_star * fe + fs_norm;
    (a_star, p_star)
}

/// RHD star state: intermediate state between contact and signal wave.
/// uses `(a, a_star, p_star)` from mignone-bodo (2005).
#[inline]
fn rhd_star_state<S: Scalar, const D: usize>(
    prim: &Prim<S, D>,
    cons: &Cons<S, D>,
    a: S,
    a_star: S,
    p_star: S,
    nhat: &Tensor<S, D>,
    metric: &SpatialMetric<S, D>,
) -> Cons<S, D> {
    let vn = metric.contract_contra(&prim.vel, nhat); // v.n = gamma_{ij} v^i n^j (contravariant)
    let ee = cons.nrg + cons.den;
    let fac = S::ONE / (a - a_star);
    let ds = fac * (a - vn) * cons.den;
    let ms = cons.mom.scale(a - vn).scale(fac) + nhat.scale((p_star - prim.pre) * fac);
    let es = fac * (ee * (a - vn) + p_star * a_star - prim.pre * vn);
    // rhd convention: nrg = tau = e - D.
    Cons {
        chi: Default::default(),
        den: ds,
        mom: ms,
        nrg: es - ds,
    }
}

/// HLLC for special-relativistic hydrodynamics (mignone-bodo 2005). one function for all
/// dimensions and directions.
///
/// `shear` both selects and parameterizes the HLLC+ transverse viscosity, the shock-stability
/// half of the scheme. the low-mach accuracy half stays newtonian: separating the velocity-jump dissipation
/// from the pressure-jump dissipation in the relativistic flux is a distinct derivation, and the
/// defect it corrects is a subsonic one. `None` is classical HLLC exactly.
pub fn hllc_rhd<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shear: Option<HllcPlusSensors<S>>,
) -> Cons<S, D> {
    hllc_rhd_body(eos, prim_l, prim_r, nhat, vface, shear)
}

/// the RHD HLLC body, split out so the outer function can wrap it without re-emitting it.
#[inline]
fn hllc_rhd_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shear: Option<HllcPlusSensors<S>>,
) -> Cons<S, D> {
    let regime = crate::rhd::Rhd;
    // flat/orthonormal frame -> identity metric (bit-identical to euclidean .dot); this parameter
    // is the seam a GR face metric enters through.
    let metric = SpatialMetric::flat();
    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (a_l, a_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);

    S::branch(
        a_l.cmp_ge(vface),
        || f_l - u_l * vface,
        || {
            S::branch(
                a_r.cmp_le(vface),
                || f_r - u_r * vface,
                || {
                    let (a_star, p_star) =
                        rhd_contact_props(&u_l, &u_r, &f_l, &f_r, nhat, a_l, a_r);
                    let classical = S::branch(
                        a_star.cmp_ge(vface),
                        || {
                            let us =
                                rhd_star_state(prim_l, &u_l, a_l, a_star, p_star, nhat, &metric);
                            f_l + (us - u_l) * a_l - us * vface
                        },
                        || {
                            let us =
                                rhd_star_state(prim_r, &u_r, a_r, a_star, p_star, nhat, &metric);
                            f_r + (us - u_r) * a_r - us * vface
                        },
                    );
                    match shear {
                        None => classical,
                        Some(sensors) => {
                            // the transverse shear viscosity, carried onto the Mignone-Bodo star
                            // states. the grid-aligned shock instability grows through the
                            // transverse velocity jump in the multidimensional momentum balance,
                            // which is a statement about the momentum equation rather than about
                            // the newtonian closure, so the cure carries into the relativistic
                            // regime with the inertia rewritten.
                            //
                            // what changes is the coefficient. the newtonian dissipation on a
                            // velocity jump is the density times the wave-frame flux,
                            // `rho (S - u)`; relativistically the transverse momentum is
                            // `rho h W^2 v`, so the inertia carrying that jump is the enthalpy
                            // density `rho h W^2 = e + p` and the coefficient becomes
                            // `(e + p)(a - v)`. at `h -> 1`, `W -> 1` the two coincide, which is
                            // the limit `hllc_plus_shear_reduces_to_the_newtonian_coefficient`
                            // pins.
                            let inertia = |cons: &Cons<S, D>, prim: &Prim<S, D>, a: S, vn: S| {
                                (cons.nrg + cons.den + prim.pre) * (a - vn)
                            };
                            let vn_l = prim_l.vel.dot(nhat);
                            let vn_r = prim_r.vel.dot(nhat);
                            let chi_l = inertia(&u_l, prim_l, a_l, vn_l);
                            let chi_r = inertia(&u_r, prim_r, a_r, vn_r);
                            let cs_l =
                                crate::rhd::sound_speed_sq(eos, prim_l.rho, prim_l.pre).sqrt();
                            let cs_r =
                                crate::rhd::sound_speed_sq(eos, prim_r.rho, prim_r.pre).sqrt();
                            // the mach number stays a ratio of coordinate speeds: the imbalance
                            // being corrected is between wave speeds multiplying differences of
                            // conserved states, and a proper-velocity mach number would carry the
                            // full lorentz factor.
                            let mach = mach_scale(prim_l.vel.norm(), prim_r.vel.norm(), cs_l, cs_r);
                            let a_k = S::select(a_star.cmp_ge(vface), a_l, a_r);
                            let upwind = a_k / (a_k - a_star);
                            let dv = prim_r.vel - prim_l.vel;
                            let dv_perp = dv - nhat.scale(dv.dot(nhat));
                            let weight = chi_l * chi_r / (chi_r - chi_l)
                                * upwind
                                * sensors.shocked
                                * shear_weight(sensors.pressure_ratio, mach);
                            classical
                                + Cons {
                                    chi: Default::default(),
                                    den: S::ZERO,
                                    mom: dv_perp.scale(weight),
                                    nrg: S::ZERO,
                                }
                        }
                    }
                },
            )
        },
    )
}

// =============================================================================
// RMHD HLLC (mignone-bodo 2006) — null vs non-null normal B-field branch.
// =============================================================================

/// HLLC for relativistic MHD — three-wave solver resolving the contact wave.
/// builds on the HLL intermediate state, solves a quadratic for the contact
/// speed `a_star`, branches on whether the normal B-field is null. one
/// function for all dimensions and directions. carrier-generic over `S`.
/// the magnetized relativistic fan runs the Standard flavor, so a requested
/// Fleischmann LM resolves to Standard. reads the pressure jump
/// from the hydro half of the MHD primitive (`prim_l.hydro.pre`).
pub fn hllc_rmhd<S: Scalar, const D: usize>(
    regime: &Rmhd,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    // taken for signature uniformity with the hydro solvers; the magnetized HLLC ships a
    // single flavor, so the selector is inert here.
    _shock_smoother: ShockwaveLimiter,
) -> MhdCons<S, D> {
    hllc_rmhd_body(regime, eos, prim_l, prim_r, nhat, vface)
}

/// the RMHD HLLC body, split out so the outer function can wrap it without re-emitting it.
fn hllc_rmhd_body<S: Scalar, const D: usize>(
    regime: &Rmhd,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
) -> MhdCons<S, D> {
    // flat/orthonormal frame -> identity metric (bit-identical to euclidean .dot); this parameter
    // is the seam a GR face metric enters through.
    let metric = SpatialMetric::flat();
    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (a_l, a_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);

    S::branch(
        a_l.cmp_ge(vface),
        || f_l - u_l * vface,
        || {
            S::branch(
                a_r.cmp_le(vface),
                || f_r - u_r * vface,
                || {
                    let inv = S::ONE / (a_r - a_l);

                    // HLL intermediate state + flux.
                    let hll_state = (u_r * a_r - u_l * a_l - f_r + f_l) * inv;
                    let hll_flux = (f_l * a_r - f_r * a_l + (u_r - u_l) * (a_l * a_r)) * inv;

                    // normal B from HLL state (continuous across the interface). B is contravariant B^i.
                    let bn = metric.contract_contra(&hll_state.mag, nhat);
                    let bt_hll = metric.project_transverse(&hll_state.mag, nhat);

                    let uhlld = hll_state.den;
                    let uhllm = hll_state.mom.dot(nhat); // conserved-momentum (covariant S_i . n^i) -> metric-free
                    let uhlle = hll_state.nrg + uhlld;

                    let fhllm = hll_flux.mom.dot(nhat); // momentum-flux . n -> metric-free (variance settled by C1)
                    let fhlle = hll_flux.nrg + hll_flux.den;
                    let ft_hll = metric.project_transverse(&hll_flux.mag, nhat); // transverse magnetic flux (contravariant)

                    // contact-wave quadratic: compute null-B and non-null-B
                    // coefficients in parallel, select via mask.
                    let b_scale = metric
                        .norm_sq_contra(&prim_l.mag)
                        .max(metric.norm_sq_contra(&prim_r.mag))
                        .sqrt();
                    let null_cond = bn.abs().cmp_le(S::from_f64(32.0 * f64::EPSILON) * b_scale);
                    let fdb = metric.contract_contra(&ft_hll, &bt_hll);
                    let bpsq = metric.norm_sq_contra(&bt_hll);
                    let fbpsq = metric.norm_sq_contra(&ft_hll);
                    let a_coeff = S::select(null_cond, fhlle, fhlle - fdb);
                    let b_coeff =
                        S::select(null_cond, -(fhllm + uhlle), -(fhllm + uhlle) + bpsq + fbpsq);
                    let c_coeff = S::select(null_cond, uhllm, uhllm - fdb);

                    let disc = (b_coeff * b_coeff - S::FOUR * a_coeff * c_coeff).max(S::ZERO);
                    let sgn_b = S::select(b_coeff.cmp_ge(S::ZERO), S::ONE, -S::ONE);
                    let quad = S::from_f64(-0.5) * (b_coeff + sgn_b * disc.sqrt());
                    let quad_scale = b_coeff.abs().max(disc.sqrt());
                    let quad_ok = quad
                        .abs()
                        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * quad_scale);
                    let quad_divisor = S::select(quad_ok, quad, S::ONE);
                    let a_star = S::select(quad_ok, c_coeff / quad_divisor, S::ZERO);

                    // safe_bn: keeps the non-null path's divide well-defined when bn is
                    // tiny; where null_cond fires the select keeps the null arm.
                    let safe_bn = S::select(null_cond, S::ONE, bn);

                    // per-side star state + HLLC flux. carrier-generic via select.
                    let side_flux = |u: &MhdCons<S, D>,
                                     f: &MhdCons<S, D>,
                                     prim_side: &MhdPrim<S, D>,
                                     ws: S|
                     -> MhdCons<S, D> {
                        // momentum-class (conserved S_i / flux-of-momentum . n^i) -> metric-free (C1/Tier-2).
                        let mn = u.mom.dot(nhat);
                        let umtrans = u.mom - nhat.scale(mn);
                        let fmtrans = f.mom - nhat.scale(f.mom.dot(nhat));
                        let etot = u.nrg + u.den;
                        let cfac = S::ONE / (ws - a_star);

                        let vn = metric.contract_contra(&prim_side.vel, nhat);
                        let vs = (ws - vn) / (ws - a_star);
                        let ds = vs * u.den;

                        // null-B star state.
                        let p_null = -a_star * fhlle + fhllm;
                        let es_null = cfac * (ws * etot - mn + p_null * a_star);
                        let mn_null = (es_null + p_null) * a_star;
                        let btrans_side = metric.project_transverse(&prim_side.mag, nhat);
                        let us_null = MhdCons {
                            hydro: Cons {
                                chi: Default::default(),
                                den: ds,
                                mom: nhat.scale(mn_null) + umtrans.scale(vs),
                                nrg: es_null - ds,
                            },
                            mag: nhat.scale(bn) + btrans_side.scale(vs),
                        };

                        // non-null-B star state (safe_bn guards division).
                        let vtrans = (bt_hll.scale(a_star) - ft_hll).scale(S::ONE / safe_bn);
                        let invg2 = S::ONE - (a_star * a_star + metric.norm_sq_contra(&vtrans));
                        let vsdb = a_star * safe_bn + metric.contract_contra(&bt_hll, &vtrans);
                        let p_nn =
                            -a_star * (fhlle - safe_bn * vsdb) + fhllm + safe_bn * safe_bn * invg2;
                        let es_nn = cfac * (ws * etot - mn + p_nn * a_star - vsdb * safe_bn);
                        let mn_nn = (es_nn + p_nn) * a_star - vsdb * safe_bn;
                        let mtrans = (umtrans.scale(ws)
                            - fmtrans
                            - (bt_hll.scale(invg2) + vtrans.scale(vsdb)).scale(safe_bn))
                        .scale(cfac);
                        let us_nn = MhdCons {
                            hydro: Cons {
                                chi: Default::default(),
                                den: ds,
                                mom: nhat.scale(mn_nn) + mtrans,
                                nrg: es_nn - ds,
                            },
                            mag: nhat.scale(safe_bn) + bt_hll,
                        };

                        let us =
                            <MhdCons<S, D> as Selectable<S>>::select(null_cond, us_null, us_nn);
                        *f + (us - *u) * ws - us * vface
                    };

                    let flux_l = side_flux(&u_l, &f_l, prim_l, a_l);
                    let flux_r = side_flux(&u_r, &f_r, prim_r, a_r);
                    <MhdCons<S, D> as Selectable<S>>::select(a_star.cmp_gt(S::ZERO), flux_l, flux_r)
                },
            )
        },
    )
}

// =============================================================================
// newtonian MHD HLLC (Li 2005 / Gurski 2004) — contact-resolving 3-wave solver:
// S_L < S_M (contact) < S_R. transverse B is continuous across the contact
// (HLL-averaged); the three-wave fan resolves the fast pair and the contact and
// leaves the rotational (alfven) discontinuities to HLLD. consistent
// (F(U,U) == F(U)); physicality-gated to HLLE.
// =============================================================================

/// the Newtonian ideal-MHD HLLC flux.
pub fn hllc_newtonian<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    // taken for signature uniformity with the hydro solvers; the magnetized HLLC ships a
    // single flavor, so the selector is inert here.
    _shock_smoother: ShockwaveLimiter,
) -> MhdCons<S, D> {
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
    let half = S::HALF;
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
    let dm_ok = dm
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * cr.abs().max(cl.abs()));
    let dm_s = S::select(dm_ok, dm, one);
    let s_m = (cr * un_r - cl * un_l - pt_r + pt_l) / dm_s;
    let pt_star = (cr * pt_l - cl * pt_r + cl * cr * (un_r - un_l)) / dm_s;

    // HLL state -> the transverse B held continuous across the contact.
    let dwave = s_r - s_l;
    let dwave_ok = dwave
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_r.abs().max(s_l.abs()));
    let inv_dwave = one / S::select(dwave_ok, dwave, one);
    let u_hll = (u_r * s_r - u_l * s_l - (f_r - f_l)) * inv_dwave;
    let tang = |v: &Tensor<S, D>, vn: S| -> Tensor<S, D> { *v - nhat.scale(vn) };
    let bt_star = tang(&u_hll.mag, u_hll.mag.dot(nhat));
    let b_star = nhat.scale(bn) + bt_star;

    // per-side single-star (*) state: normal velocity S_M, transverse v from the
    // transverse-momentum jump with the continuous B*, energy from the energy jump.
    let star = |u_k: &MhdCons<S, D>,
                prim_k: &MhdPrim<S, D>,
                f_k: &MhdCons<S, D>,
                s_k: S,
                un_k: S,
                rho_k: S,
                pt_k: S,
                c_k: S|
     -> (MhdCons<S, D>, MhdCons<S, D>, S) {
        let smk = s_k - s_m;
        let smk_ok = smk
            .abs()
            .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_k.abs().max(s_m.abs()));
        let smk_s = S::select(smk_ok, smk, one);
        let rho_star = rho_k * (s_k - un_k) / smk_s;
        let c_ok = c_k
            .abs()
            .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * rho_k.abs() * s_k.abs().max(un_k.abs()));
        let c_safe = S::select(c_ok, c_k, one); // rho_K(S_K - u_K)
        let vt_k = tang(&prim_k.vel, un_k);
        let bt_k = tang(&prim_k.mag, bn);
        let vt_star = vt_k - (bt_star - bt_k).scale(bn / c_safe);
        let v_star = nhat.scale(s_m) + vt_star;
        let e_k = u_k.nrg;
        let vdb_k = prim_k.vel.dot(&prim_k.mag);
        let vdb_s = v_star.dot(&b_star);
        let e_star =
            ((s_k - un_k) * e_k - pt_k * un_k + pt_star * s_m + bn * (vdb_k - vdb_s)) / smk_s;
        let u_star = MhdCons {
            hydro: Cons {
                chi: Default::default(),
                den: rho_star,
                mom: v_star.scale(rho_star),
                nrg: e_star,
            },
            mag: b_star,
        };
        let f_star = *f_k + (u_star - *u_k) * s_k;
        (u_star, f_star, rho_star)
    };
    let (us_l, fs_l, rs_l) = star(&u_l, prim_l, &f_l, s_l, un_l, rho_l, pt_l, cl);
    let (us_r, fs_r, rs_r) = star(&u_r, prim_r, &f_r, s_r, un_r, rho_r, pt_r, cr);

    let reg = |f: MhdCons<S, D>, u: MhdCons<S, D>| -> MhdCons<S, D> { f - u * vface };
    let pick = MhdCons::select(
        vface.cmp_lt(s_l),
        reg(f_l, u_l),
        MhdCons::select(
            vface.cmp_lt(s_m),
            reg(fs_l, us_l),
            MhdCons::select(vface.cmp_lt(s_r), reg(fs_r, us_r), reg(f_r, u_r)),
        ),
    );
    let smk_l_ok = (s_l - s_m)
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_l.abs().max(s_m.abs()));
    let smk_r_ok = (s_r - s_m)
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_r.abs().max(s_m.abs()));
    let c_l_ok = cl
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * rho_l.abs() * s_l.abs().max(un_l.abs()));
    let c_r_ok = cr
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * rho_r.abs() * s_r.abs().max(un_r.abs()));
    let ok = dm_ok
        & dwave_ok
        & smk_l_ok
        & smk_r_ok
        & c_l_ok
        & c_r_ok
        & rs_l.cmp_gt(zero)
        & rs_r.cmp_gt(zero)
        & pt_star.cmp_gt(zero);
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
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 1.0,
        };
        let nhat = Tensor::unit(0);
        let flux = hllc(
            &eos,
            &prim,
            &prim,
            &nhat,
            0.0,
            ShockwaveLimiter::Standard,
            None,
        );
        let regime = Newtonian;
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
        assert!(approx(flux.mom[0], exact.mom[0]));
        assert!(approx(flux.nrg, exact.nrg));
    }

    #[test]
    fn hllc_uniform_state_2d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim {
            rho: 2.0,
            vel: Tensor::new([0.5, -0.3]),
            pre: 2.5,
        };

        let nhat_x = Tensor::unit(0);
        let flux_x = hllc(
            &eos,
            &prim,
            &prim,
            &nhat_x,
            0.0,
            ShockwaveLimiter::Standard,
            None,
        );
        let regime = Newtonian;
        let exact_x = regime.to_flux(&prim, &nhat_x, &eos);
        assert!(approx(flux_x.den, exact_x.den));
        assert!(approx(flux_x.mom[0], exact_x.mom[0]));
        assert!(approx(flux_x.mom[1], exact_x.mom[1]));
        assert!(approx(flux_x.nrg, exact_x.nrg));

        let nhat_y = Tensor::unit(1);
        let flux_y = hllc(
            &eos,
            &prim,
            &prim,
            &nhat_y,
            0.0,
            ShockwaveLimiter::Standard,
            None,
        );
        let exact_y = regime.to_flux(&prim, &nhat_y, &eos);
        assert!(approx(flux_y.den, exact_y.den));
        assert!(approx(flux_y.mom[0], exact_y.mom[0]));
        assert!(approx(flux_y.mom[1], exact_y.mom[1]));
        assert!(approx(flux_y.nrg, exact_y.nrg));
    }

    #[test]
    fn hllc_sod_shock_tube() {
        let eos = IdealGas { gamma: 1.4 };
        let prim_l = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        };
        let prim_r = Prim {
            rho: 0.125,
            vel: Tensor::new([0.0]),
            pre: 0.1,
        };
        let nhat = Tensor::unit(0);

        let flux = hllc(
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            0.0,
            ShockwaveLimiter::Standard,
            None,
        );
        assert!(flux.den > 0.0);
        assert!(flux.nrg > 0.0);
    }

    #[test]
    fn hllc_symmetric_2d() {
        // x-problem vs y-problem with velocities swapped — proves rotational
        // invariance of the nhat-parametrized solver.
        let eos = IdealGas { gamma: 1.4 };

        let prim_l_x = Prim {
            rho: 1.0,
            vel: Tensor::new([1.0, 0.0]),
            pre: 1.0,
        };
        let prim_r_x = Prim {
            rho: 0.5,
            vel: Tensor::new([0.0, 0.0]),
            pre: 0.5,
        };
        let flux_x = hllc(
            &eos,
            &prim_l_x,
            &prim_r_x,
            &Tensor::unit(0),
            0.0,
            ShockwaveLimiter::Standard,
            None,
        );

        let prim_l_y = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 1.0]),
            pre: 1.0,
        };
        let prim_r_y = Prim {
            rho: 0.5,
            vel: Tensor::new([0.0, 0.0]),
            pre: 0.5,
        };
        let flux_y = hllc(
            &eos,
            &prim_l_y,
            &prim_r_y,
            &Tensor::unit(1),
            0.0,
            ShockwaveLimiter::Standard,
            None,
        );

        assert!(approx(flux_x.den, flux_y.den));
        assert!(approx(flux_x.nrg, flux_y.nrg));
        assert!(approx(flux_x.mom[0], flux_y.mom[1]));
    }

    #[test]
    fn hllc_rhd_uniform_state() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = crate::rhd::Rhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.3]),
            pre: 1.0,
        };
        let nhat = Tensor::unit(0);
        let flux = hllc_rhd(&eos, &prim, &prim, &nhat, 0.0, None);
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
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.3, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 1.0, 0.0]),
        };
        let nhat = Tensor::unit(0);
        let flux = hllc_rmhd(
            &regime,
            &eos,
            &prim,
            &prim,
            &nhat,
            0.0,
            ShockwaveLimiter::Standard,
        );
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(
            approx(flux.den, exact.den),
            "den: {} vs {}",
            flux.den,
            exact.den
        );
        for dd in 0..3 {
            assert!(
                approx(flux.mom[dd], exact.mom[dd]),
                "mom[{}]: {} vs {}",
                dd,
                flux.mom[dd],
                exact.mom[dd]
            );
        }
        assert!(
            approx(flux.nrg, exact.nrg),
            "nrg: {} vs {}",
            flux.nrg,
            exact.nrg
        );
    }

    #[test]
    fn hllc_rmhd_balsara_shock() {
        let eos = IdealGas { gamma: 2.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 1.0, 0.0]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.125,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 0.1,
            },
            mag: Tensor::new([0.5, -1.0, 0.0]),
        };
        let nhat = Tensor::unit(0);
        let flux = hllc_rmhd(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            0.0,
            ShockwaveLimiter::Standard,
        );
        assert!(
            flux.den > 0.0,
            "density flux should be positive: {}",
            flux.den
        );
    }

    // ---- Newtonian MHD HLLC ----

    fn nm_prim(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> MhdPrim<f64, 3> {
        MhdPrim {
            hydro: Prim {
                rho,
                vel: Tensor::new(v),
                pre: p,
            },
            mag: Tensor::new(b),
        }
    }

    fn assert_mhd_flux_eq(got: &MhdCons<f64, 3>, want: &MhdCons<f64, 3>, ctx: &str) {
        assert!(
            approx(got.den, want.den),
            "{ctx} den: {} vs {}",
            got.den,
            want.den
        );
        for dd in 0..3 {
            assert!(
                approx(got.mom[dd], want.mom[dd]),
                "{ctx} mom[{dd}]: {} vs {}",
                got.mom[dd],
                want.mom[dd]
            );
            assert!(
                approx(got.mag[dd], want.mag[dd]),
                "{ctx} mag[{dd}]: {} vs {}",
                got.mag[dd],
                want.mag[dd]
            );
        }
        assert!(
            approx(got.nrg, want.nrg),
            "{ctx} nrg: {} vs {}",
            got.nrg,
            want.nrg
        );
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
        assert_mhd_flux_eq(
            &f,
            &NewtonianMhd.to_flux(&pl, &nhat, &eos),
            "supersonic-right",
        );
        // mirror: supersonic-left
        let pl2 = nm_prim(1.0, [-5.0, 0.2, 0.0], 1.0, [0.3, 0.5, 0.0]);
        let pr2 = nm_prim(0.5, [-5.0, -0.1, 0.2], 0.4, [0.3, -0.4, 0.1]);
        let f2 = hllc_newtonian(&eos, &pl2, &pr2, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert_mhd_flux_eq(
            &f2,
            &NewtonianMhd.to_flux(&pr2, &nhat, &eos),
            "supersonic-left",
        );
    }

    #[test]
    fn hllc_newtonian_brio_wu_finite_and_divb_clean() {
        let eos = IdealGas { gamma: 2.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let pl = nm_prim(1.0, [0.0, 0.0, 0.0], 1.0, [0.75, 1.0, 0.0]);
        let pr = nm_prim(0.125, [0.0, 0.0, 0.0], 0.1, [0.75, -1.0, 0.0]);
        let f = hllc_newtonian(&eos, &pl, &pr, &nhat, 0.0, ShockwaveLimiter::Standard);
        assert!(f.den.is_finite() && f.nrg.is_finite(), "finite");
        assert!(
            f.mag[0].abs() < 1e-12,
            "normal-B flux must vanish: {}",
            f.mag[0]
        );
    }

    // ---- carrier gate: newtonian wave_properties q-factor sqrt ----

    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::{Gv, begin_trace, end_trace};

    // trace `wave_properties` at S = Gv, scalarize each of the three signal-speed
    // outputs, and CPU-interpret them at the given f64 state. proves the body
    // renders (non-empty LoweredFn) and that the traced graph, which evaluates
    // both select arms, produces finite results matching the f64 physics path.
    fn wave_properties_gv(state: &[f64; 9]) -> [f64; 3] {
        let names = [
            "rho_l", "rho_r", "pre_l", "pre_r", "vn_l", "vn_r", "cs_l", "cs_r", "gamma",
        ];
        begin_trace();
        let p: Vec<Gv> = names.iter().map(|n| Gv::param(n)).collect();
        let (s_l, s_r, s_star) =
            wave_properties::<Gv>(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);
        let kernel = end_trace();
        // the kernel lowering: `Op::IfElse` (the pressure-estimate lazy branch)
        // lowers only through `scalarize_kernel`; the single-root elemental
        // path refuses it. the interpreter evaluates the IfElse statement
        // directly, so the lowered kernel adapts into a LoweredFn verbatim.
        let outputs = [s_l.node(), s_r.node(), s_star.node()];
        let k = symbi_ir::passes::scalarize::scalarize_kernel(&kernel.graph, &outputs);
        assert!(
            !k.body.is_empty(),
            "wave_properties rendered an empty kernel"
        );
        let lowered = symbi_ir::passes::scalarize::LoweredFn {
            name: "wave_props_probe".to_string(),
            params: k.params,
            body: k.body,
            results: k.outputs,
            result_element: symbi_ir::ElementTy::F64,
            result_shape: Vec::new(),
        };
        let vals = Cpu.eval_elemental(&lowered, state);
        [vals[0], vals[1], vals[2]]
    }

    fn wave_properties_f64(state: &[f64; 9]) -> [f64; 3] {
        let (s_l, s_r, s_star) = wave_properties::<f64>(
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            state[8],
        );
        [s_l, s_r, s_star]
    }

    #[test]
    fn hllc_wave_properties_carrier_equiv_strong_rarefaction() {
        // strong double rarefaction (gas pulling apart fast) drives p_star -> 0.
        // at gamma < 1 the q-factor radicand `1 + k*(p_star/pre - 1)` turns
        // negative (k = (gamma+1)/(2*gamma) > 1). the f64 select discards the
        // q_alt arm (p_star <= pre), so the physical result is the rarefaction
        // q = 1. at S = Gv both arms trace, and the `.max(S::ZERO)` clamp is what
        // keeps the discarded arm from tracing `sqrt(neg)` = NaN into the kernel.
        // this asserts the clamp keeps the traced graph finite and bit-equal to
        // the f64 path. gamma < 1 puts the state on the negative radicand, the
        // exact landmine the clamp guards.
        let gamma = 0.8_f64;
        let cs = (gamma * 1.0 / 1.0_f64).sqrt();
        let state = [1.0, 1.0, 1.0, 1.0, -3.0, 3.0, cs, cs, gamma];

        let want = wave_properties_f64(&state);
        let got = wave_properties_gv(&state);
        for kk in 0..3 {
            assert!(
                got[kk].is_finite(),
                "gv signal speed {kk} not finite: {}",
                got[kk]
            );
            assert!(
                approx(want[kk], got[kk]),
                "carrier mismatch at {kk}: f64 {} vs gv {}",
                want[kk],
                got[kk]
            );
        }
    }

    #[test]
    fn hllc_wave_properties_clamp_is_identity_on_physical_state() {
        // physical gamma > 1 strong rarefaction: the radicand stays positive, so
        // the `.max(S::ZERO)` clamp is the identity. the f64 and Gv paths agree:
        // the clamp passes non-degenerate states through unchanged.
        let gamma = 1.4_f64;
        let cs = (gamma * 1.0 / 1.0_f64).sqrt();
        let state = [1.0, 1.0, 1.0, 1.0, -3.0, 3.0, cs, cs, gamma];

        let want = wave_properties_f64(&state);
        let got = wave_properties_gv(&state);
        for kk in 0..3 {
            assert!(got[kk].is_finite(), "gv signal speed {kk} not finite");
            assert!(
                approx(want[kk], got[kk]),
                "carrier mismatch at {kk}: f64 {} vs gv {}",
                want[kk],
                got[kk]
            );
        }
    }

    // =========================================================================
    // the relativistic extension of HLLC-LM
    // =========================================================================

    #[test]
    fn the_central_formulation_is_an_identity_for_the_relativistic_star_states() {
        // the whole prerequisite for HLLC-LM in any regime. the intermediate flux has two
        // derivations — from the left wave, and from the right wave plus the contact:
        //
        //   F_*L = F_L + a_L (U_*L - U_L)
        //   F_*L = F_R + a_R (U_*R - U_R) + a_* (U_*L - U_*R)
        //
        // the central formulation is their average, so it equals the classical intermediate flux
        // exactly when both hold — which requires the star states to satisfy the Rankine-Hugoniot
        // conditions across the two outer waves and the contact simultaneously. that is a property
        // of the star-state construction rather than of the flux algebra, and it is what makes the
        // reformulation available to scale.
        //
        // hold it and the reduced-dissipation flux is a genuine modification of HLLC; lose it and
        // the result is a different, inconsistent solver whose error masquerades as ordinary
        // numerical dissipation.
        let eos = IdealGas {
            gamma: 4.0 / 3.0f64,
        };
        let n = Tensor::new([1.0, 0.0]);
        let regime = crate::rhd::Rhd;
        let metric = SpatialMetric::flat();
        let cases = [
            (
                "mildly relativistic",
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.2, 0.0]),
                    pre: 1.0,
                },
                Prim {
                    rho: 0.5,
                    vel: Tensor::new([-0.1, 0.0]),
                    pre: 0.4,
                },
            ),
            (
                // the grid-aligned shock: vanishing normal velocity, relativistic transverse motion.
                "vanishing normal velocity, relativistic transverse",
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([1.0e-6, 0.99]),
                    pre: 1.0,
                },
                Prim {
                    rho: 1.2,
                    vel: Tensor::new([-1.0e-6, 0.99]),
                    pre: 1.1,
                },
            ),
            (
                "strong jump",
                Prim {
                    rho: 10.0,
                    vel: Tensor::new([0.4, 0.3]),
                    pre: 20.0,
                },
                Prim {
                    rho: 0.1,
                    vel: Tensor::new([-0.3, 0.1]),
                    pre: 0.05,
                },
            ),
        ];
        for (name, l, r) in cases {
            let u_l = regime.to_conserved(&eos, &l);
            let u_r = regime.to_conserved(&eos, &r);
            let f_l = regime.to_flux(&l, &n, &eos);
            let f_r = regime.to_flux(&r, &n, &eos);
            let (a_l, a_r) = regime.extremal_speeds(&eos, &l, &r, &n);
            assert!(
                a_l < 0.0 && a_r > 0.0,
                "{name}: the fan must straddle the face for the intermediate flux to be the answer \
                 at all (a_L = {a_l}, a_R = {a_r})"
            );
            let (a_star, p_star) = rhd_contact_props(&u_l, &u_r, &f_l, &f_r, &n, a_l, a_r);
            let usl = rhd_star_state(&l, &u_l, a_l, a_star, p_star, &n, &metric);
            let usr = rhd_star_state(&r, &u_r, a_r, a_star, p_star, &n, &metric);

            let from_left = f_l + (usl - u_l) * a_l;
            let from_right = f_r + (usr - u_r) * a_r + (usl - usr) * a_star;

            // scaled by the size of the terms being differenced: the near-degenerate case has a
            // result close to zero built from O(1) pieces, so a result-relative bound would
            // demand cancellation beyond what the arithmetic delivers.
            let scale = f_l
                .den
                .abs()
                .max(f_r.den.abs())
                .max(f_l.nrg.abs())
                .max(f_r.nrg.abs())
                .max(1.0);
            for (what, a, b) in [
                ("den", from_left.den, from_right.den),
                ("mom_n", from_left.mom[0], from_right.mom[0]),
                ("mom_t", from_left.mom[1], from_right.mom[1]),
                ("nrg", from_left.nrg, from_right.nrg),
            ] {
                assert!(
                    (a - b).abs() <= 1.0e-12 * scale,
                    "{name}/{what}: the two derivations of the intermediate flux disagree \
                     ({a:e} vs {b:e}, scale {scale:e}). the Mignone-Bodo star states do not satisfy \
                     the contact jump condition, so the central formulation is not an identity and \
                     scaling it does not yield a modified HLLC"
                );
            }
        }
    }

    #[test]
    fn the_relativistic_sound_speed_is_not_the_newtonian_one() {
        // the trap in porting the scaling. the relativistic sound speed carries the specific
        // enthalpy that the newtonian expression leaves out, and on a relativistic gas that
        // newtonian value exceeds the speed of light — a mach number built on it runs low by that
        // factor and pushes faces onto the ramp that belong at phi = 1.
        let eos = IdealGas {
            gamma: 4.0 / 3.0f64,
        };
        let (rho, pre) = (1.0f64, 1.0f64);
        let newtonian = (4.0 / 3.0 * pre / rho).sqrt();
        let relativistic = crate::rhd::sound_speed_sq(&eos, rho, pre).sqrt();
        assert!(
            newtonian > 1.0,
            "this state must be one where the newtonian expression is superluminal, got {newtonian}"
        );
        assert!(
            relativistic < 1.0,
            "the relativistic sound speed must be subluminal, got {relativistic}"
        );
        // the acoustic wave speeds of a state at rest are exactly +/- cs, which pins which of the
        // two the solver uses.
        let at_rest = Prim {
            rho,
            vel: Tensor::new([0.0, 0.0]),
            pre,
        };
        let (a_l, a_r) =
            crate::rhd::Rhd.extremal_speeds(&eos, &at_rest, &at_rest, &Tensor::new([1.0, 0.0]));
        assert!(
            (a_r - relativistic).abs() < 1.0e-12 && (a_l + relativistic).abs() < 1.0e-12,
            "a state at rest must have acoustic speeds +/- cs_rel = +/-{relativistic}, got \
             ({a_l}, {a_r})"
        );
    }

    #[test]
    fn the_relativistic_shear_coefficient_reduces_to_the_newtonian_one() {
        // the coefficient carrying the transverse viscosity is the inertia the transverse
        // momentum equation gives that jump: the mass density `rho` in the newtonian momentum
        // `rho v`, and the enthalpy density `rho h W^2 = e + p` in the relativistic `rho h W^2 v`.
        // the two agree in the limit the relativistic system reduces in, and this pins that:
        // as the flow slows and the gas cools, `h -> 1` and `W -> 1` and the relativistic
        // coefficient must approach the newtonian one it generalizes.
        //
        // the coefficient is `(e + p)(a - v)` against `rho (S - u)`, so the ratio measured is
        // `(e + p)/rho = h W^2`, which carries the whole departure.
        let gamma = 5.0f64 / 3.0;
        let eos = IdealGas { gamma };
        let nhat = Tensor::new([1.0, 0.0]);
        let mut previous = f64::INFINITY;
        // colder and slower at each step: the two knobs that carry `h` and `W` to one.
        for (speed, theta) in [
            (0.3f64, 0.3f64),
            (0.1, 3.0e-2),
            (0.03, 3.0e-3),
            (0.01, 3.0e-4),
        ] {
            let rho = 1.0;
            let prim = Prim {
                rho,
                vel: Tensor::new([speed, 0.0]),
                pre: theta * rho,
            };
            let cons = crate::rhd::Rhd.to_conserved(&eos, &prim);
            // `e = tau + D` is the total energy density; `e + p = rho h W^2`.
            let enthalpy_density = cons.nrg + cons.den + prim.pre;
            let departure = (enthalpy_density / rho - 1.0).abs();
            assert!(
                departure < previous,
                "at v = {speed}, p/rho = {theta} the relativistic inertia departs from the \
                 newtonian one by {departure:.3e}, no closer than the previous {previous:.3e}; \
                 the coefficient must converge on `rho` as the gas cools and slows"
            );
            previous = departure;
            let _ = nhat;
        }
        assert!(
            previous < 1.0e-3,
            "at the coldest, slowest state the relativistic inertia still departs from the \
             newtonian one by {previous:.3e}; the reduction is not being reached"
        );
    }

    #[test]
    fn the_relativistic_shear_term_vanishes_without_a_transverse_velocity_jump() {
        // the term carries the in-plane velocity jump as a factor, so a face whose two states
        // share their transverse velocity receives the classical Mignone-Bodo flux however
        // strong the shock across it. this is what keeps the viscosity out of a
        // one-dimensional problem, where no shear wave exists to damp.
        let eos = IdealGas {
            gamma: 4.0f64 / 3.0,
        };
        let nhat = Tensor::new([1.0, 0.0]);
        // a strong relativistic shock, with the transverse component equal on both sides.
        for shared_vt in [0.0f64, 0.2, -0.35] {
            let l = Prim {
                rho: 1.0,
                vel: Tensor::new([0.6, shared_vt]),
                pre: 10.0,
            };
            let r = Prim {
                rho: 0.2,
                vel: Tensor::new([0.1, shared_vt]),
                pre: 0.5,
            };
            let sensors = HllcPlusSensors {
                // a strong shock as far as the sensors are concerned, so the weight is at full
                // strength and the vanishing is attributable to the jump alone.
                pressure_ratio: 0.05,
                shocked: 1.0,
            };
            let classical = hllc_rhd(&eos, &l, &r, &nhat, 0.0, None);
            let sheared = hllc_rhd(&eos, &l, &r, &nhat, 0.0, Some(sensors));
            assert_eq!(
                (
                    classical.den.to_bits(),
                    classical.mom[1].to_bits(),
                    classical.nrg.to_bits()
                ),
                (
                    sheared.den.to_bits(),
                    sheared.mom[1].to_bits(),
                    sheared.nrg.to_bits()
                ),
                "at a shared transverse velocity {shared_vt} the shear term moved the flux; it \
                 carries the in-plane velocity jump as a factor and that jump is zero here"
            );
        }
    }

    #[test]
    fn the_relativistic_shear_term_damps_a_transverse_velocity_jump() {
        // and where the jump exists the term opposes it: the transverse momentum flux moves
        // against the jump, which is what removes energy from the perturbation a grid-aligned
        // front grows through.
        let eos = IdealGas {
            gamma: 4.0f64 / 3.0,
        };
        let nhat = Tensor::new([1.0, 0.0]);
        let sensors = HllcPlusSensors {
            pressure_ratio: 0.05,
            shocked: 1.0,
        };
        for jump in [0.2f64, -0.2] {
            let l = Prim {
                rho: 1.0,
                vel: Tensor::new([0.6, -0.5 * jump]),
                pre: 10.0,
            };
            let r = Prim {
                rho: 0.2,
                vel: Tensor::new([0.1, 0.5 * jump]),
                pre: 0.5,
            };
            let classical = hllc_rhd(&eos, &l, &r, &nhat, 0.0, None);
            let sheared = hllc_rhd(&eos, &l, &r, &nhat, 0.0, Some(sensors));
            let delta = sheared.mom[1] - classical.mom[1];
            assert!(
                delta * jump < 0.0,
                "a transverse jump of {jump} moved the transverse momentum flux by \
                 {delta:.3e}, which carries the jump's own sign; the viscosity must oppose the \
                 jump rather than reinforce it"
            );
            assert!(
                delta.abs() > 1.0e-6,
                "the shear term moved the transverse momentum flux by only {delta:.3e} on a \
                 jump of {jump}; the term is present but inert and the sign check is vacuous"
            );
        }
    }

    #[test]
    fn the_hllc_plus_corrections_vanish_on_a_stagnant_stratified_face() {
        // the property that makes the correction safe on a hydrostatic background. the term it
        // adds carries the face's normal velocity jump as a factor, so two states at a common
        // velocity — the face a stratified column at rest presents, whatever pressure and
        // density contrast it carries across it — receive the classical HLLC flux with its
        // full pressure-jump dissipation intact. the scaling of the acoustic signal speeds
        // instead attenuates that dissipation on the same face, which is what leaves a
        // hydrostatic truncation residual undamped.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::new([1.0, 0.0]);
        // a strong stratification: a factor of two in density and pressure across one face,
        // both sides drifting at a common deeply subsonic velocity so the correction's mach
        // scaling is far from saturation and its absence is attributable to the jump alone.
        for common_v in [0.0f64, 1.0e-3, -4.0e-3] {
            let l = Prim {
                rho: 2.0,
                vel: Tensor::new([common_v, 0.0]),
                pre: 3.0,
            };
            let r = Prim {
                rho: 1.0,
                vel: Tensor::new([common_v, 0.0]),
                pre: 1.5,
            };
            let std = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::Standard, None);
            let plus = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::HllcPlus, None);
            assert_eq!(
                (std.den.to_bits(), std.mom[0].to_bits(), std.nrg.to_bits()),
                (
                    plus.den.to_bits(),
                    plus.mom[0].to_bits(),
                    plus.nrg.to_bits()
                ),
                "at a common velocity {common_v} the correction moved the flux; it carries the \
                 velocity jump as a factor and the jump is zero here"
            );
            // the premise: this face is deeply subsonic, so the correction is at full strength
            // wherever a velocity jump does exist, and its absence measures the jump.
            let cs = eos.sound_speed(l.rho, l.pre);
            assert!(
                common_v.abs() / cs < 0.1,
                "the face must sit deep in the low-mach regime for the vanishing to be \
                 attributable to the jump rather than to a saturated scaling"
            );
        }
    }

    #[test]
    fn hllc_plus_removes_the_normal_velocity_jump_dissipation_at_low_mach() {
        // the correction's purpose, stated as the term it cancels. the classical flux damps a
        // face's normal velocity jump at the acoustic impedance, `rho c du`, which at low mach
        // number exceeds the convective flux it corrects by an order in `1/Ma`. the correction
        // rescales that term to `Ma * rho c du`, so the residual dissipation falls linearly
        // with the mach number and the difference between the two fluxes approaches the whole
        // classical term.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::new([1.0]);
        let cs = eos.sound_speed(1.0, 1.0);
        let mut previous = f64::INFINITY;
        for mach in [0.2f64, 0.1, 0.05, 0.025] {
            let v = mach * cs;
            // a symmetric compression: the two sides converge on the face, so the velocity jump
            // is the whole of the face data and the pressure jump is zero.
            let l = Prim {
                rho: 1.0,
                vel: Tensor::new([v]),
                pre: 1.0,
            };
            let r = Prim {
                rho: 1.0,
                vel: Tensor::new([-v]),
                pre: 1.0,
            };
            let std = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::Standard, None);
            let plus = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::HllcPlus, None);
            // the surviving fraction of the classical velocity-jump dissipation, read off the
            // normal momentum flux against the jump-free reference the two states share.
            let reference = 1.0f64;
            let surviving = (plus.mom[0] - reference).abs() / (std.mom[0] - reference).abs();
            assert!(
                surviving < previous,
                "at Ma = {mach} the surviving dissipation fraction {surviving:.4} did not fall \
                 below the previous {previous:.4}; the correction is meant to scale it with the \
                 mach number"
            );
            // the scaling is `min(1, Ma)`, so the survivor is the mach number itself to within
            // the difference between the face mach number and the two states' own.
            assert!(
                (surviving - mach).abs() < 0.1 * mach,
                "at Ma = {mach} the surviving fraction is {surviving:.4}; the correction scales \
                 the velocity-jump dissipation by the local mach number"
            );
            previous = surviving;
        }
    }
}
