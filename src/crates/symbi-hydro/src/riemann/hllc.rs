// =============================================================================
// riemann/hllc.rs
//
// the HLLC three-wave riemann solvers — one function per regime, all
// rotationally and dimensionally invariant (nhat-parametrized, generic over
// `S: Scalar` and `const D: usize`). a `ShockwaveLimiter` parameter
// selects the variant (Standard / Fleischmann LM); the
// relativistic regimes ignore it.
//
//   newtonian  `hllc`       — toro eq 10.37-10.39 star state, +/- fleischmann LM.
//   rhd       `hllc_rhd`  — mignone & bodo (2005) star state.
//   rmhd       `hllc_rmhd`  — mignone & bodo (2006), null/non-null-B branch.
//
// every solver is GPU-traceable (`S::branch` / `S::select` on the
// carrier-generic mask) and `vface`-aware (the ALE grid velocity is
// subtracted from the conserved flux post-star).
// =============================================================================

use super::hlle::hlle;
use crate::dissipation::{ShockwaveLimiter, acoustic_phi, adaptive_phi, fleischmann_phi};
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
// newtonian HLLC — toro section 9.5.2 adaptive estimates + fleischmann LM.
// =============================================================================

/// wave properties for newtonian HLLC: signal speeds + contact speed.
/// implements toro section 9.5.2 adaptive estimates (PVRS / two-rarefaction
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

    // pvrs if the pressure ratio is mild AND pvrs is bounded, else rarefaction
    // (if pvrs <= p_min) or shock. mask AND uses `&` on `S::Mask` (the carrier's
    // bitwise BitAnd; native `&&` would lock to a host carrier).
    //
    // the estimates live in LAZY `S::cond` arms — a smooth-flow face pays only
    // the pvrs arithmetic; the two-rarefaction arm's three `powf` calls and the
    // two-shock arm's roots run only on the faces that reject pvrs. an eager
    // select spelling paid all three estimates at every face and dominated the
    // hllc kernel's cost (~49 ns/zone vs ~14 for hlle). carrier-equivalent:
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
    let nrg_star = den_star * (u_k.nrg / prim.rho + (s_star - vn) * (s_star + prim.pre / chi_k));
    Cons {
        chi: Default::default(),
        den: den_star,
        mom: mom_star,
        nrg: nrg_star,
    }
}

/// HLLC for newtonian (compressible Euler) — toro eq 10.37-10.39. ONE function
/// for all dimensions / directions / shock-limiter modes.
///
/// `shock_smoother`:
///   - `Standard`     — plain HLLC.
///   - `Fleischmann`  — symmetric flux (fleischmann eq 11) with adaptive phi.
pub fn hllc<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
    mach_limit: S,
) -> Cons<S, D> {
    hllc_newtonian_body(eos, prim_l, prim_r, nhat, vface, shock_smoother, mach_limit)
}

/// the newtonian HLLC body — Standard / Fleischmann star-state dispatch,
/// cells to HLLE before reaching this point). callable directly for
/// regression diff harnesses.
#[inline]
fn hllc_newtonian_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
    mach_limit: S,
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
        // the upwind side uses its OWN star state (toro 10.21); supersonic
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
        // fleischmann et al. (2020) HLLC-LM: the CENTRAL formulation of the intermediate flux
        // (their eq 19) with the acoustic signal speeds scaled by an adaptive factor `phi`.
        //
        // the central form averages the two Rankine-Hugoniot derivations of `F_*` — from the left
        // and from the right — which separates the flux into a central part and a dissipation part
        // the way the Roe flux does. that separation is what makes a targeted reduction of the
        // ACOUSTIC dissipation possible: `S_L` and `S_R` carry it, while the contact speed `S_*`
        // carries the advective dissipation and is left alone.
        //
        // `phi` scales ONLY this final sum. the wave speeds, the contact speed and both star states
        // are built from the unscaled `s_l` / `s_r` above — the paper is explicit that every
        // preceding step uses the original values, and scaling them earlier would change the star
        // states themselves rather than the dissipation they generate.
        //
        // the supersonic branches are the same as standard HLLC: the central form equals the
        // subsonic intermediate flux, and where the whole fan travels one way the physical flux is
        // the upwind one. dropping those branches leaves a supersonic face carrying `F_* != F_L`,
        // an error of several percent that no amount of dissipation tuning removes.
        ShockwaveLimiter::Fleischmann
        | ShockwaveLimiter::FleischmannClamped
        | ShockwaveLimiter::Acoustic => S::branch(
            s_l.cmp_ge(vface),
            || f_l - u_l * vface,
            || {
                S::branch(
                    s_r.cmp_le(vface),
                    || f_r - u_r * vface,
                    || {
                        let u_star_l = star_state(prim_l, &u_l, s_l, s_star, chi_l, nhat);
                        let u_star_r = star_state(prim_r, &u_r, s_r, s_star, chi_r, nhat);

                        // the acoustic dissipation scaling. `Acoustic` keys on the wave
                        // content of the face data and carries no reference mach number; the
                        // balance term is zero here because the riemann solver sees no body
                        // force, so a face held by one reads as fully acoustic — the
                        // conservative reading.
                        let phi = match shock_smoother {
                            ShockwaveLimiter::Acoustic => acoustic_phi(
                                vn_l, vn_r, cs_l, cs_r, prim_l.pre, prim_r.pre, prim_l.rho,
                                prim_r.rho, S::ZERO,
                            ),
                            ShockwaveLimiter::FleischmannClamped => {
                                adaptive_phi(vn_l, vn_r, cs_l, cs_r, prim_l.pre, prim_r.pre)
                            }
                            // `Standard` never reaches this branch; `Fleischmann` is the
                            // published ramp, with no clamp on the pressure jump.
                            _ => fleischmann_phi(vn_l, vn_r, cs_l, cs_r, mach_limit),
                        };
                        let s_l_lm = phi * s_l;
                        let s_r_lm = phi * s_r;

                        let face_star = <Cons<S, D> as Selectable<S>>::select(
                            s_star.cmp_ge(vface),
                            u_star_l,
                            u_star_r,
                        );
                        let half = S::from_f64(0.5);
                        (f_l + f_r) * half
                            + ((u_star_l - u_l) * s_l_lm
                                + (u_star_l - u_star_r) * s_star.abs()
                                + (u_star_r - u_r) * s_r_lm)
                                * half
                            - face_star * vface
                    },
                )
            },
        ),
    }
}

// =============================================================================
// RHD HLLC (mignone-bodo 2005) — relativistic; no Fleischmann LM correction.
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
    let disc = bb * bb - S::from_f64(4.0) * aa * cc;
    let disc_sqrt = disc.abs().sqrt();
    let sgn_b = S::select(bb.cmp_ge(S::ZERO), S::ONE, -S::ONE);
    let half = S::from_f64(0.5);
    let quad = -half * (bb + sgn_b * disc_sqrt);
    // guard the contact-speed divide against the degenerate `quad -> 0` root (bb == 0 with fe or
    // s_norm == 0): unguarded this returns NaN/Inf and poisons the flux. mirrors the proven RMHD
    // guard (the `a_star` select above). Gv evaluates both arms, but the Inf is selected away, never
    // combined into an output — the same carrier-safe pattern the RMHD path uses.
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

/// HLLC for special-relativistic hydrodynamics (mignone-bodo 2005). ONE
/// function for all dimensions and directions. honors the
/// `Fleischmann` low-mach acoustic-dissipation scaling.
pub fn hllc_rhd<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> Cons<S, D> {
    hllc_rhd_body(eos, prim_l, prim_r, nhat, vface, shock_smoother)
}

/// the RHD HLLC body, split out so the outer function can wrap it without re-emitting it.
#[inline]
fn hllc_rhd_body<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &Prim<S, D>,
    prim_r: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    shock_smoother: ShockwaveLimiter,
) -> Cons<S, D> {
    let regime = crate::rhd::Rhd;
    // flat/orthonormal frame -> identity metric (bit-identical to euclidean .dot); the GR face
    // metric threads in once the flux path carries it.
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
                    match shock_smoother {
                        ShockwaveLimiter::Standard => S::branch(
                            a_star.cmp_ge(vface),
                            || {
                                let us = rhd_star_state(
                                    prim_l, &u_l, a_l, a_star, p_star, nhat, &metric,
                                );
                                f_l + (us - u_l) * a_l - us * vface
                            },
                            || {
                                let us = rhd_star_state(
                                    prim_r, &u_r, a_r, a_star, p_star, nhat, &metric,
                                );
                                f_r + (us - u_r) * a_r - us * vface
                            },
                        ),
                        // HLLC-LM in the relativistic regime: the same central formulation and the
                        // same acoustic-dissipation scaling as the newtonian case, on the
                        // Mignone-Bodo star states.
                        //
                        // the reformulation carries over because it rests on one property, not on
                        // the newtonian flux algebra: the star states must satisfy the
                        // Rankine-Hugoniot conditions across both outer waves AND the contact, so
                        // that deriving the intermediate flux from the left and from the right give
                        // the same answer and their average is an identity. the Mignone-Bodo
                        // construction does — its contact speed and pressure are fixed by exactly
                        // that consistency — and `the_central_formulation_is_an_identity_for_the_
                        // relativistic_star_states` pins it.
                        //
                        // what does NOT carry over untouched is the sound speed. the relativistic
                        // value carries the specific enthalpy, `cs^2 = gamma p / (rho h)`, and the
                        // newtonian expression evaluated on a relativistic gas exceeds the speed of
                        // light — at gamma = 4/3, p = rho = 1 it gives 1.15 against the true 0.516.
                        // a mach number built on it would be low by that factor and would move faces
                        // onto the ramp that belong at phi = 1.
                        //
                        // the mach number stays a ratio of COORDINATE speeds, with no lorentz
                        // factors. the imbalance being corrected is between the acoustic
                        // dissipation carried by `a_l`, `a_r` and the advective dissipation carried
                        // by `a_star`, and those are coordinate wave speeds multiplying differences
                        // of conserved states. a proper-velocity mach number would carry the FULL
                        // lorentz factor, which at a grid-aligned shock is dominated by the
                        // TRANSVERSE motion — reintroducing exactly the contamination that keying on
                        // the face-normal component removes.
                        ShockwaveLimiter::Fleischmann
                        | ShockwaveLimiter::FleischmannClamped
                        | ShockwaveLimiter::Acoustic => {
                            let cs_l =
                                crate::rhd::sound_speed_sq(eos, prim_l.rho, prim_l.pre).sqrt();
                            let cs_r =
                                crate::rhd::sound_speed_sq(eos, prim_r.rho, prim_r.pre).sqrt();
                            let vn_l = prim_l.vel.dot(nhat);
                            let vn_r = prim_r.vel.dot(nhat);
                            let phi = match shock_smoother {
                                ShockwaveLimiter::Acoustic => acoustic_phi(
                                    vn_l, vn_r, cs_l, cs_r, prim_l.pre, prim_r.pre,
                                    prim_l.rho, prim_r.rho, S::ZERO,
                                ),
                                ShockwaveLimiter::FleischmannClamped => adaptive_phi(
                                    vn_l, vn_r, cs_l, cs_r, prim_l.pre, prim_r.pre,
                                ),
                                // the relativistic arm carries no runtime reference mach
                                // number: its LM selector is the clamped one, so this
                                // branch is unreachable and the published default keeps the
                                // expression well-defined without inventing a knob the
                                // relativistic dispatch cannot bind.
                                _ => fleischmann_phi(
                                    vn_l,
                                    vn_r,
                                    cs_l,
                                    cs_r,
                                    S::from_f64(crate::dissipation::MACH_LIMIT),
                                ),
                            };

                            let usl =
                                rhd_star_state(prim_l, &u_l, a_l, a_star, p_star, nhat, &metric);
                            let usr =
                                rhd_star_state(prim_r, &u_r, a_r, a_star, p_star, nhat, &metric);
                            let face_star = <Cons<S, D> as Selectable<S>>::select(
                                a_star.cmp_ge(vface),
                                usl,
                                usr,
                            );
                            let half = S::from_f64(0.5);
                            (f_l + f_r) * half
                                + ((usl - u_l) * (phi * a_l)
                                    + (usl - usr) * a_star.abs()
                                    + (usr - u_r) * (phi * a_r))
                                    * half
                                - face_star * vface
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
/// speed `a_star`, branches on whether the normal B-field is null. ONE
/// function for all dimensions and directions. carrier-generic over `S`.
/// Fleischmann LM does not apply to relativistic
/// regimes (treated as Standard if requested). reads the pressure jump
/// from the hydro half of the MHD primitive (`prim_l.hydro.pre`).
pub fn hllc_rmhd<S: Scalar, const D: usize>(
    regime: &Rmhd,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    // taken for signature uniformity with the hydro solvers; the magnetized HLLC has no
    // low-mach variant, so there is no flavor to select.
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
    // flat/orthonormal frame -> identity metric (bit-identical to euclidean .dot); the GR face
    // metric threads in once the flux path carries it.
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

                    // contact-wave quadratic: compute null-B AND non-null-B
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

                    let disc =
                        (b_coeff * b_coeff - S::from_f64(4.0) * a_coeff * c_coeff).max(S::ZERO);
                    let sgn_b = S::select(b_coeff.cmp_ge(S::ZERO), S::ONE, -S::ONE);
                    let quad = S::from_f64(-0.5) * (b_coeff + sgn_b * disc.sqrt());
                    let quad_scale = b_coeff.abs().max(disc.sqrt());
                    let quad_ok = quad
                        .abs()
                        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * quad_scale);
                    let quad_divisor = S::select(quad_ok, quad, S::ONE);
                    let a_star = S::select(quad_ok, c_coeff / quad_divisor, S::ZERO);

                    // safe_bn: avoid 0/0 in the non-null path when bn is tiny;
                    // when null_cond fires the non-null arm is discarded by select.
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
// S_L < S_M (contact) < S_R. transverse B is CONTINUOUS across the contact
// (HLL-averaged) — the rotational (alfven) discontinuities are NOT resolved
// (that is HLLD's job). consistent (F(U,U) == F(U)); physicality-gated to HLLE.
// =============================================================================

/// the Newtonian ideal-MHD HLLC flux.
pub fn hllc_newtonian<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    // taken for signature uniformity with the hydro solvers; the magnetized HLLC has no
    // low-mach variant, so there is no flavor to select.
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
    let half = S::from_f64(0.5);
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

    // HLL state -> the transverse B held CONTINUOUS across the contact.
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
    use crate::dissipation::MACH_LIMIT;
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
        let flux = hllc(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard, MACH_LIMIT);
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
        let flux_x = hllc(&eos, &prim, &prim, &nhat_x, 0.0, ShockwaveLimiter::Standard, MACH_LIMIT);
        let regime = Newtonian;
        let exact_x = regime.to_flux(&prim, &nhat_x, &eos);
        assert!(approx(flux_x.den, exact_x.den));
        assert!(approx(flux_x.mom[0], exact_x.mom[0]));
        assert!(approx(flux_x.mom[1], exact_x.mom[1]));
        assert!(approx(flux_x.nrg, exact_x.nrg));

        let nhat_y = Tensor::unit(1);
        let flux_y = hllc(&eos, &prim, &prim, &nhat_y, 0.0, ShockwaveLimiter::Standard, MACH_LIMIT);
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
            MACH_LIMIT,
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
            MACH_LIMIT,
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
            MACH_LIMIT,
        );

        assert!(approx(flux_x.den, flux_y.den));
        assert!(approx(flux_x.nrg, flux_y.nrg));
        assert!(approx(flux_x.mom[0], flux_y.mom[1]));
    }

    #[test]
    fn hllc_fleischmann_equals_standard_on_a_supersonic_face() {
        // eq 18: where the whole wave fan travels one way the flux is the upwind one, `F_L` if
        // `S_L >= 0` and `F_R` if `S_R <= 0`. the central formulation of the intermediate flux is
        // an identity for the SUBSONIC fan only — evaluated on a supersonic face it returns
        // `F_L + S_L (U_*L - U_L)`, which differs from `F_L` by several percent whenever the star
        // state differs from the upwind state.
        //
        // low mach and supersonic are exclusive, so the scaling cannot mask this: a supersonic face
        // has phi = 1 and the two solvers must agree exactly.
        let eos = IdealGas { gamma: 1.4f64 };
        let nhat = Tensor::new([1.0, 0.0]);
        let cases = [
            // left-supersonic (S_L >= 0) and right-supersonic (S_R <= 0), each with a genuine
            // jump so the star state is NOT the upwind state.
            (5.0, 1.0, 1.0, 5.0, 1.3, 1.2),
            (-5.0, 1.3, 1.2, -5.0, 1.0, 1.0),
        ];
        for (vl, pl, rl, vr, pr, rr) in cases {
            let l = Prim {
                rho: rl,
                vel: Tensor::new([vl, 0.0]),
                pre: pl,
            };
            let r = Prim {
                rho: rr,
                vel: Tensor::new([vr, 0.0]),
                pre: pr,
            };
            let std = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::Standard, MACH_LIMIT);
            let lm = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::Fleischmann, MACH_LIMIT);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0e-30);
            assert!(
                rel(std.den, lm.den) < 1.0e-14 && rel(std.nrg, lm.nrg) < 1.0e-14,
                "supersonic face (v = {vl} -> {vr}): standard gives den {} nrg {}, HLLC-LM gives \
                 den {} nrg {}. the upwind branches are missing, so the intermediate flux is being \
                 returned where the fan is entirely one-sided",
                std.den,
                std.nrg,
                lm.den,
                lm.nrg
            );
        }

        // the premise: the star state must actually differ from the upwind state, or the two
        // formulations agree for reasons unrelated to the branch.
        let l = Prim {
            rho: 1.0,
            vel: Tensor::new([5.0, 0.0]),
            pre: 1.0,
        };
        let r = Prim {
            rho: 1.2,
            vel: Tensor::new([5.0, 0.0]),
            pre: 1.3,
        };
        let u_l = l.to_conserved(&eos);
        let cs_l = eos.sound_speed(l.rho, l.pre);
        let cs_r = eos.sound_speed(r.rho, r.pre);
        let (s_l, s_r, s_star) =
            wave_properties(l.rho, r.rho, l.pre, r.pre, 5.0, 5.0, cs_l, cs_r, 1.4);
        assert!(
            s_l >= 0.0,
            "the left state must be supersonic, got S_L = {s_l}"
        );
        let us = star_state(&l, &u_l, s_l, s_star, l.rho * (s_l - 5.0), &nhat);
        assert!(
            (us.den - u_l.den).abs() > 1.0e-3,
            "the star state equals the upwind state here, so returning either would pass"
        );
        let _ = s_r;
    }

    #[test]
    fn hllc_fleischmann_uniform_matches_standard() {
        // a uniform state has zero LM correction — Fleischmann reduces to the
        // exact regime flux just like Standard.
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.01]),
            pre: 1.0,
        };
        let nhat = Tensor::unit(0);
        let flux = hllc(
            &eos,
            &prim,
            &prim,
            &nhat,
            0.0,
            ShockwaveLimiter::Fleischmann,
            MACH_LIMIT,
        );
        let regime = Newtonian;
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
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
        let flux = hllc_rhd(&eos, &prim, &prim, &nhat, 0.0, ShockwaveLimiter::Standard);
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
    // renders (non-empty LoweredFn) AND that the traced graph (which evaluates
    // BOTH select arms) produces finite results matching the f64 physics path.
    fn wave_properties_gv(state: &[f64; 9]) -> [f64; 3] {
        let names = [
            "rho_l", "rho_r", "pre_l", "pre_r", "vn_l", "vn_r", "cs_l", "cs_r", "gamma",
        ];
        begin_trace();
        let p: Vec<Gv> = names.iter().map(|n| Gv::param(n)).collect();
        let (s_l, s_r, s_star) =
            wave_properties::<Gv>(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);
        let kernel = end_trace();
        // the KERNEL lowering: `Op::IfElse` (the pressure-estimate lazy branch)
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
        // the `.max(S::ZERO)` clamp is an IDENTITY. the f64 and Gv paths must
        // agree: the clamp leaves non-degenerate states untouched.
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
        // the ENTIRE prerequisite for HLLC-LM in any regime. the intermediate flux has two
        // derivations — from the left wave, and from the right wave plus the contact:
        //
        //   F_*L = F_L + a_L (U_*L - U_L)
        //   F_*L = F_R + a_R (U_*R - U_R) + a_* (U_*L - U_*R)
        //
        // the central formulation is their average, so it equals the classical intermediate flux
        // only if BOTH hold — which requires the star states to satisfy the Rankine-Hugoniot
        // conditions across the two outer waves and the contact simultaneously. that is a property
        // of the star-state construction, not of the flux algebra, and it is what makes the
        // reformulation available to scale.
        //
        // if it failed, the reduced-dissipation flux would not be a modification of HLLC at all: it
        // would be a different, inconsistent solver whose error would look like ordinary numerical
        // dissipation.
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
                // the grid-aligned shock: vanishing NORMAL velocity, relativistic transverse motion.
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

            // scaled by the size of the terms being differenced, not by the result: the
            // near-degenerate case has a result close to zero built from O(1) pieces, so a
            // result-relative bound would demand cancellation the arithmetic cannot deliver.
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

    /// the acoustic-consistency solver must reduce to classical HLLC wherever the face
    /// carries a genuine wave — the impedance ratio saturates there, so the scaling is a
    /// no-op and the flux is the unmodified one. this is the property that makes it safe at
    /// shocks: it cannot be less dissipative than HLLC where HLLC's dissipation is what the
    /// data calls for.
    #[test]
    fn acoustic_hllc_equals_classical_hllc_on_a_wave_bearing_face() {
        let eos = IdealGas { gamma: 1.4f64 };
        let nhat = Tensor::new([1.0, 0.0]);
        // faces whose pressure jump MEETS OR EXCEEDS the impedance relation, which is what a
        // compression carries. a merely fast face does not qualify — a large velocity jump
        // with a pressure jump well under `rho c du` is not acoustic, and the sensor is right
        // to keep scaling it down.
        let cases = [
            (0.0, 1.0, 1.0, -3.0, 20.0, 4.0),
            (0.5, 1.0, 1.0, -0.5, 6.0, 3.0),
        ];
        for (vl, pl, rl, vr, pr, rr) in cases {
            let l = Prim {
                rho: rl,
                vel: Tensor::new([vl, 0.0]),
                pre: pl,
            };
            let r = Prim {
                rho: rr,
                vel: Tensor::new([vr, 0.0]),
                pre: pr,
            };
            // NON-VACUITY: the equivalence only holds where the scaling is inactive, so the
            // face has to actually saturate the sensor. a case that quietly stopped doing so
            // would make this test compare a solver against itself.
            let cs_l = (1.4f64 * l.pre / l.rho).sqrt();
            let cs_r = (1.4f64 * r.pre / r.rho).sqrt();
            let phi = crate::dissipation::acoustic_phi(
                l.vel[0], r.vel[0], cs_l, cs_r, l.pre, r.pre, l.rho, r.rho, 0.0,
            );
            assert_eq!(
                phi, 1.0,
                "this face does not saturate the sensor (phi = {phi}); it cannot test the \
                 phi = 1 equivalence"
            );
            let std = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::Standard, MACH_LIMIT);
            let acu = hllc(&eos, &l, &r, &nhat, 0.0, ShockwaveLimiter::Acoustic, MACH_LIMIT);
            let scale = std.den.abs().max(std.nrg.abs()).max(1.0);
            for (what, a, b) in [
                ("den", std.den, acu.den),
                ("mom_n", std.mom[0], acu.mom[0]),
                ("nrg", std.nrg, acu.nrg),
            ] {
                assert!(
                    (a - b).abs() <= 1.0e-12 * scale,
                    "{what}: a wave-bearing face must give classical HLLC exactly \
                     ({a:e} vs {b:e})"
                );
            }
        }
    }

    /// the payoff, measured through the flux rather than the sensor: on a SMOOTH low-mach
    /// face the acoustic scaling sits strictly closer to the central (non-dissipative) flux
    /// than the Fleischmann ramp does, because it scales to the mach number instead of
    /// saturating at a tenth of it. the central flux is the zero-dissipation reference, so
    /// distance from it IS the numerical dissipation the face applies.
    #[test]
    fn acoustic_hllc_is_less_dissipative_than_fleischmann_at_low_mach() {
        let eos = IdealGas { gamma: 1.4f64 };
        let nhat = Tensor::new([1.0, 0.0]);
        let (rho, p) = (1.0f64, 1.0f64);
        let cs = (1.4 * p / rho).sqrt();
        for &mach in &[1.0e-3, 1.0e-2, 5.0e-2] {
            let u = mach * cs;
            let du = 1.0e-2 * u;
            // the pressure jump the momentum balance supports across this velocity jump
            let dp = rho * u * du;
            let l = Prim {
                rho,
                vel: Tensor::new([u + 0.5 * du, 0.0]),
                pre: p + 0.5 * dp,
            };
            let r = Prim {
                rho,
                vel: Tensor::new([u - 0.5 * du, 0.0]),
                pre: p - 0.5 * dp,
            };
            let central = {
                let regime = Newtonian;
                let fl = regime.to_flux(&l, &nhat, &eos);
                let fr = regime.to_flux(&r, &nhat, &eos);
                (fl + fr) * 0.5
            };
            let dissipation = |lim| {
                let f = hllc(&eos, &l, &r, &nhat, 0.0, lim, MACH_LIMIT);
                (f.mom[0] - central.mom[0]).abs()
            };
            let fleisch = dissipation(ShockwaveLimiter::Fleischmann);
            let acoustic = dissipation(ShockwaveLimiter::Acoustic);
            assert!(
                acoustic < fleisch,
                "at Ma = {mach:e} the acoustic scaling was not less dissipative \
                 (acoustic {acoustic:e} vs fleischmann {fleisch:e})"
            );
            // and the reduction should track the ratio of the two sensors, which at this
            // mach is more than a factor of five.
            assert!(
                acoustic * 5.0 < fleisch,
                "at Ma = {mach:e} the reduction was only {:.2}x; the sensors differ by \
                 far more than that",
                fleisch / acoustic
            );
        }
    }

    #[test]
    fn relativistic_hllc_lm_equals_relativistic_hllc_wherever_the_scaling_is_inactive() {
        // above the mach limit `phi = 1` and the two must agree exactly — the reformulation is an
        // identity, so any disagreement is a defect in the central form rather than the intended
        // reduction. covers both the subsonic fan (where the star flux applies) and the supersonic
        // fans (where the upwind flux does).
        let eos = IdealGas {
            gamma: 4.0 / 3.0f64,
        };
        let n = Tensor::new([1.0, 0.0]);
        let cases = [
            // subsonic fan, normal mach well above the limit
            (
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.3, 0.1]),
                    pre: 1.0,
                },
                Prim {
                    rho: 0.6,
                    vel: Tensor::new([0.2, -0.1]),
                    pre: 0.5,
                },
            ),
            // left-supersonic and right-supersonic
            (
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.99, 0.0]),
                    pre: 0.01,
                },
                Prim {
                    rho: 1.2,
                    vel: Tensor::new([0.99, 0.0]),
                    pre: 0.02,
                },
            ),
            (
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([-0.99, 0.0]),
                    pre: 0.01,
                },
                Prim {
                    rho: 1.2,
                    vel: Tensor::new([-0.99, 0.0]),
                    pre: 0.02,
                },
            ),
        ];
        for (l, r) in cases {
            let cs = |p: &Prim<f64, 2>| crate::rhd::sound_speed_sq(&eos, p.rho, p.pre).sqrt();
            let ma = (l.vel[0] / cs(&l)).abs().max((r.vel[0] / cs(&r)).abs());
            assert!(
                ma > crate::dissipation::MACH_LIMIT,
                "this case sits ON the ramp (Ma = {ma}); it cannot test the phi = 1 equivalence"
            );
            let std = hllc_rhd(&eos, &l, &r, &n, 0.0, ShockwaveLimiter::Standard);
            let lm = hllc_rhd(&eos, &l, &r, &n, 0.0, ShockwaveLimiter::Fleischmann);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1.0e-30);
            assert!(
                rel(std.den, lm.den) < 1.0e-13
                    && rel(std.mom[0], lm.mom[0]) < 1.0e-13
                    && rel(std.nrg, lm.nrg) < 1.0e-13,
                "at Ma = {ma:.3} (phi = 1) the two solvers differ: den {} vs {}, nrg {} vs {}",
                std.den,
                lm.den,
                std.nrg,
                lm.nrg
            );
        }
    }

    #[test]
    fn the_relativistic_scaling_collapses_the_transverse_acoustic_dissipation() {
        // the mechanism, in the relativistic regime. a grid-aligned shock has a large velocity along
        // its propagation and a vanishing component across it, so the transverse-face Riemann
        // problem is at Ma ~ 0 while its acoustic wave speeds stay at O(c_s). classical HLLC then
        // applies dissipation proportional to c_s to a face whose flow is nearly at rest — the
        // scaling failure that drives the carbuncle.
        let eos = IdealGas {
            gamma: 4.0 / 3.0f64,
        };
        // a converging velocity perturbation along the front — velocities jump, pressure and
        // density UNIFORM. pressure uniformity is the physical state along a shock front and the
        // condition under which the low-mach reduction applies at all: a face-normal pressure
        // jump far above the incompressible `dp/p ~ gamma Ma^2` scale is a stratified/acoustic
        // structure, where the compressibility-consistency clamp restores classical dissipation
        // instead. the velocity jump generates the wrongly-scaled term this scheme exists to
        // remove — the momentum-flux dissipation `~ c_s rho du`, applied at O(c_s) to a face
        // whose flow is nearly at rest — while HLLC's contact resolution already keeps the
        // density channel clean.
        let l = Prim {
            rho: 1.0,
            vel: Tensor::new([1.0e-3, 0.99]),
            pre: 1.0,
        };
        let r = Prim {
            rho: 1.0,
            vel: Tensor::new([-1.0e-3, 0.99]),
            pre: 1.0,
        };
        let across = Tensor::new([1.0, 0.0]);
        let along = Tensor::new([0.0, 1.0]);

        // the premise: this really is the degenerate configuration — the outer speeds are O(c_s)
        // while the normal velocity is negligible.
        let regime = crate::rhd::Rhd;
        let (a_l, a_r) = regime.extremal_speeds(&eos, &l, &r, &across);
        assert!(
            a_l < -0.1 && a_r > 0.1,
            "the transverse face must carry O(c_s) acoustic speeds, got ({a_l}, {a_r})"
        );

        let std = hllc_rhd(&eos, &l, &r, &across, 0.0, ShockwaveLimiter::Standard);
        let lm = hllc_rhd(&eos, &l, &r, &across, 0.0, ShockwaveLimiter::Fleischmann);

        // the exact momentum flux across this symmetric face is the uniform pressure plus a
        // convective term of order rho du^2 ~ 1e-6; everything classical HLLC adds beyond that
        // is acoustic dissipation `~ c_s rho du`, and the scaling must remove most of it.
        let spurious_std = (std.mom[0] - 1.0).abs();
        let spurious_lm = (lm.mom[0] - 1.0).abs();
        assert!(
            spurious_std > 1.0e-4,
            "the classical momentum flux carries no measurable acoustic dissipation \
             ({spurious_std:e}); the probe is vacuous"
        );
        assert!(
            spurious_lm < 0.1 * spurious_std,
            "the spurious transverse momentum flux is {spurious_lm:e} under HLLC-LM against \
             {spurious_std:e} under HLLC; the acoustic dissipation has not been reduced"
        );

        // and the shock-normal face must be untouched: there the flow is supersonic and the scheme
        // is classical HLLC exactly. a scaling that fired in both directions would be smoothing the
        // shock itself.
        let std_along = hllc_rhd(&eos, &l, &r, &along, 0.0, ShockwaveLimiter::Standard);
        let lm_along = hllc_rhd(&eos, &l, &r, &along, 0.0, ShockwaveLimiter::Fleischmann);
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1.0e-30);
        assert!(
            rel(std_along.den, lm_along.den) < 1.0e-13,
            "the shock-normal face changed ({} vs {}); the flow there is supersonic and must get \
             classical HLLC untouched",
            std_along.den,
            lm_along.den
        );
    }

    #[test]
    fn a_hot_gas_above_the_mach_limit_is_not_put_on_the_ramp_by_the_wrong_sound_speed() {
        // the discriminating case for WHICH sound speed the scaling uses. the two differ by the
        // square root of the specific enthalpy, `cs_newt / cs_rel = sqrt(h)`, so on a hot gas they
        // straddle the mach limit: a face genuinely above it — where the scheme must be classical
        // HLLC — reads as low-mach under the newtonian expression and has its acoustic dissipation
        // cut by more than half.
        //
        // the earlier gates cannot see this. they sit either far below the limit (where both
        // definitions give a tiny phi) or far above it (where both give one). only a state placed
        // BETWEEN the two thresholds separates them.
        let eos = IdealGas {
            gamma: 4.0 / 3.0f64,
        };
        let n = Tensor::new([1.0, 0.0]);
        let (rho, pre) = (1.0f64, 10.0f64);
        let cs_rel = crate::rhd::sound_speed_sq(&eos, rho, pre).sqrt();
        let cs_newt = (4.0 / 3.0 * pre / rho).sqrt();
        // 1.5x the limit on the relativistic sound speed.
        let vn = 1.5 * crate::dissipation::MACH_LIMIT * cs_rel;

        // the premise: the two definitions must genuinely straddle the limit here, or the case
        // proves nothing about which one is in use.
        assert!(
            vn / cs_rel > crate::dissipation::MACH_LIMIT
                && vn / cs_newt < crate::dissipation::MACH_LIMIT,
            "the mach numbers do not straddle the limit: relativistic {:.4}, newtonian {:.4}",
            vn / cs_rel,
            vn / cs_newt
        );

        let l = Prim {
            rho,
            vel: Tensor::new([vn, 0.0]),
            pre,
        };
        let r = Prim {
            rho: rho * 1.05,
            vel: Tensor::new([vn, 0.0]),
            pre: pre * 1.05,
        };
        let std = hllc_rhd(&eos, &l, &r, &n, 0.0, ShockwaveLimiter::Standard);
        let lm = hllc_rhd(&eos, &l, &r, &n, 0.0, ShockwaveLimiter::Fleischmann);
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1.0e-30);
        assert!(
            rel(std.den, lm.den) < 1.0e-13 && rel(std.nrg, lm.nrg) < 1.0e-13,
            "this face is at Ma = {:.3} on the relativistic sound speed, above the limit, so \
             HLLC-LM must reduce nothing — yet it differs from HLLC (den {} vs {}, nrg {} vs {}). \
             the scaling is using the newtonian sound speed, which is larger by sqrt(h) = {:.2} and \
             puts this face on the ramp",
            vn / cs_rel,
            std.den,
            lm.den,
            std.nrg,
            lm.nrg,
            cs_newt / cs_rel
        );
    }

    #[test]
    fn the_relativistic_sound_speed_is_not_the_newtonian_one() {
        // the trap in porting the scaling. the newtonian sound speed omits the specific enthalpy,
        // and on a relativistic gas it exceeds the speed of light — so a mach number built on it is
        // low by that factor and pushes faces onto the ramp that belong at phi = 1.
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
        // the acoustic wave speeds of a state at rest ARE +/- cs, which pins which of the two the
        // solver actually uses.
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
}
