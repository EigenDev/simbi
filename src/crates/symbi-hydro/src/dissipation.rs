// =============================================================================
// dissipation.rs
//
// adaptive numerical dissipation for the HLLC riemann solver.
// fleischmann et al. (2020) low-mach fix: adaptive phi in [0,1].
// all detectors take nhat (the unit normal vector) as the direction argument.
//
// usage:
//   let nhat = Tensor::unit(0);
//   let phi = adaptive_phi(&prim_l, &prim_r, &nhat, gamma);
// =============================================================================

use crate::state::Prim;
use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// shockwave limiter selector for the HLLC riemann solver. picks the flavor of
/// HLLC the regime emits at a face:
///
///   - `Standard`     — plain HLLC (toro / mignone-bodo star state).
///   - `Fleischmann`  — newtonian only: HLLC + fleischmann et al. (2020)
///                      adaptive-phi low-mach correction. relativistic
///                      regimes ignore (no relativistic LM correction).
///   - `Quirk`        — RESERVED. falls back to HLLE in 2D+ when the
///                      `quirk_strong_shock` detector fires. the detector
///                      and threshold are not yet implemented; the variant
///                      is enumerated so a future patch lands without API
///                      churn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShockwaveLimiter {
    Standard,
    Fleischmann,
    Quirk,
}

impl Default for ShockwaveLimiter {
    fn default() -> Self {
        ShockwaveLimiter::Standard
    }
}

/// relative pressure-jump threshold for the Quirk strong-shock detector.
/// `QUIRK_THRESHOLD = 1e-4`.
pub const QUIRK_THRESHOLD: f64 = 1e-4;

/// Quirk strong-shock detector — fires when the relative pressure jump
/// across the face exceeds `QUIRK_THRESHOLD`:
///
/// ```text
///   bool quirk_strong_shock(real pl, real pr) {
///       return |pr - pl| / min(pl, pr) > QUIRK_THRESHOLD;
///   }
/// ```
///
/// returns `Self::Mask` so the carrier-generic
/// dispatch via `S::branch` works uniformly at S = f64 (host bool) and
/// S = Gv (graph mask). callers gate the HLLC -> HLLE fallback on this
/// mask; the gate is meaningful only in `D > 1` (1D doesn't carbuncle).
#[inline]
pub fn quirk_strong_shock<S: Scalar>(p_l: S, p_r: S) -> S::Mask {
    let jump = (p_r - p_l).abs();
    let p_min = p_l.min(p_r);
    (jump / p_min).cmp_gt(S::from_f64(QUIRK_THRESHOLD))
}

/// local mach number: max of left and right |v| / cs.
#[inline]
pub fn local_mach<S: Scalar, const D: usize>(left: &Prim<S, D>, right: &Prim<S, D>, gamma: S) -> S {
    let cs_l = (gamma * left.pre / left.rho).sqrt();
    let cs_r = (gamma * right.pre / right.rho).sqrt();
    let ma_l = left.vel.norm() / cs_l;
    let ma_r = right.vel.norm() / cs_r;
    ma_l.max(ma_r)
}

/// shock detector: entropy production + velocity convergence along nhat.
/// branchless: AND of two conditions via multiplication (both are 0/1 from cmp_*).
#[inline]
pub fn detect_shock<S: Scalar, const D: usize>(
    left: &Prim<S, D>,
    right: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    gamma: S,
) -> S {
    let s_l = left.pre.ln() - gamma * left.rho.ln();
    let s_r = right.pre.ln() - gamma * right.rho.ln();
    let entropy_production = s_r - s_l;

    let vn_l = left.vel.dot(nhat);
    let vn_r = right.vel.dot(nhat);
    let velocity_convergence = vn_l - vn_r;

    // AND = product of 0/1 masks. result is 1 if both conditions hold, else 0.
    // (mask -> S via select(m, ONE, ZERO); `cmp_*` returns `S::Mask`.)
    let c1 = S::select(
        entropy_production.cmp_gt(S::from_f64(0.01)),
        S::ONE,
        S::ZERO,
    );
    let c2 = S::select(velocity_convergence.cmp_gt(S::ZERO), S::ONE, S::ZERO);
    c1 * c2
}

/// interface detector: large density jump with small pressure jump.
/// branchless via cmp_* masks.
#[inline]
pub fn detect_interface<S: Scalar, const D: usize>(left: &Prim<S, D>, right: &Prim<S, D>) -> S {
    let half = S::from_f64(0.5);
    let rho_avg = half * (left.rho + right.rho);
    let pre_avg = half * (left.pre + right.pre);
    let rho_jump = (left.rho - right.rho).abs() / rho_avg;
    let pre_jump = (left.pre - right.pre).abs() / pre_avg;

    let c1 = S::select(rho_jump.cmp_gt(S::from_f64(0.1)), S::ONE, S::ZERO);
    let c2 = S::select(pre_jump.cmp_lt(S::from_f64(0.05)), S::ONE, S::ZERO);
    S::from_f64(0.4) * c1 * c2
}

/// alignment detector: high-speed flow aligned with nhat.
/// branchless: guards division by |v| via select on safe denominator,
/// then combines conditions with product of 0/1 masks.
#[inline]
pub fn detect_alignment<S: Scalar, const D: usize>(
    left: &Prim<S, D>,
    right: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    gamma: S,
) -> S {
    let v_l_mag = left.vel.norm();
    let v_r_mag = right.vel.norm();
    let eps = S::from_f64(1e-10);

    // guard against zero |v| to avoid 0/0 = NaN on GPU
    let safe_v_l = S::select(v_l_mag.cmp_gt(eps), v_l_mag, S::ONE);
    let safe_v_r = S::select(v_r_mag.cmp_gt(eps), v_r_mag, S::ONE);

    let vn_l = left.vel.dot(nhat).abs();
    let vn_r = right.vel.dot(nhat).abs();
    let align_l = vn_l / safe_v_l;
    let align_r = vn_r / safe_v_r;
    let max_align = align_l.max(align_r);

    let cs_l = (gamma * left.pre / left.rho).sqrt();
    let cs_r = (gamma * right.pre / right.rho).sqrt();
    let avg_mach = S::from_f64(0.5) * (v_l_mag / cs_l + v_r_mag / cs_r);

    // all four conditions ANDed via 0/1 mask product
    let c_vl = S::select(v_l_mag.cmp_gt(eps), S::ONE, S::ZERO);
    let c_vr = S::select(v_r_mag.cmp_gt(eps), S::ONE, S::ZERO);
    let c_align = S::select(max_align.cmp_gt(S::from_f64(0.8)), S::ONE, S::ZERO);
    let c_mach = S::select(avg_mach.cmp_gt(S::from_f64(0.5)), S::ONE, S::ZERO);
    c_vl * c_vr * c_align * c_mach
}

/// adaptive dissipation parameter phi. fleischmann et al. (2020).
/// nhat-parametrized: ONE function for all directions.
#[inline]
pub fn adaptive_phi<S: Scalar, const D: usize>(
    left: &Prim<S, D>,
    right: &Prim<S, D>,
    nhat: &Tensor<S, D>,
    gamma: S,
) -> S {
    let mach_lim = S::from_f64(0.1);
    let half_pi = S::from_f64(std::f64::consts::FRAC_PI_2);
    let ma = local_mach(left, right, gamma);

    let ratio = (ma / mach_lim).min(S::ONE);
    let mut phi = (ratio * half_pi).sin();

    let shock = detect_shock(left, right, nhat, gamma);
    let interface = detect_interface(left, right);
    let alignment = detect_alignment(left, right, nhat, gamma);

    phi = phi.max(shock);
    phi = phi.max(interface);
    phi = phi.max(alignment);
    phi.min(S::ONE)
}
