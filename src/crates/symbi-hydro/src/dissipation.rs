// =============================================================================
// dissipation.rs
//
// the low-mach acoustic-dissipation scaling of the HLLC-LM riemann solver
// (Fleischmann, Adami & Adams, J. Comput. Phys. 423:109762, 2020).
//
// a grid-aligned shock has a vanishing velocity component transverse to its propagation, so the
// transverse-face Riemann problems run at a local mach number near zero. There the acoustic
// dissipation of a classical HLLC flux scales with the sound speed rather than the flow speed, and
// that mismatch drives the grid-aligned shock instability (the carbuncle). Scaling the acoustic
// signal speeds by `phi(Ma_local)` removes the excess.
//
// the scaling is keyed on the FACE-NORMAL velocity component, and it is the only modulation of the
// dissipation — see `adaptive_phi`.
//
// usage:
//   let nhat = Tensor::unit(0);
//   let phi = adaptive_phi(&prim_l, &prim_r, &nhat, gamma);
// =============================================================================

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

/// the local mach number of a face's Riemann problem: `max(|u_L/c_L|, |u_R/c_R|)` where `u` is the
/// velocity component ALONG THE FACE NORMAL and `c` the sound speed.
///
/// takes the projected velocities and sound speeds directly rather than the states, because the
/// sound speed is a property of the REGIME: the newtonian `sqrt(gamma p / rho)` and the
/// relativistic `sqrt(gamma p / (rho h))` differ by the specific enthalpy, and the newtonian form
/// evaluated on a relativistic gas readily exceeds the speed of light. every caller has already
/// computed the value its own wave speeds are built from; recomputing one here would be a second,
/// silently regime-wrong definition.
///
/// the DIRECTION is the mechanism. a grid-aligned shock carries a large velocity along its
/// propagation direction and a vanishing component transverse to it, so the transverse faces run at
/// a local mach number near zero. keyed on the speed instead, those faces read as supersonic and the
/// correction does nothing on the only faces it exists for.
#[inline]
pub fn local_mach<S: Scalar>(vn_l: S, vn_r: S, cs_l: S, cs_r: S) -> S {
    (vn_l / cs_l).abs().max((vn_r / cs_r).abs())
}

/// reference mach number below which the acoustic dissipation is reduced. above it the scheme is
/// classical HLLC exactly. `MACH_LIMIT = 0.1` — the modification acts only where the local flow
/// component is under a tenth of the local sound speed.
pub const MACH_LIMIT: f64 = 0.1;

/// the acoustic-dissipation scaling `phi = sin(min(1, Ma_local / Ma_limit) * pi/2)`.
///
/// applied to the acoustic signal speeds `S_L` and `S_R` alone, so the acoustic dissipation falls
/// off in proportion to the local flow speed rather than the sound speed, while the advective
/// dissipation carried by the contact speed is untouched. the sine gives a smooth decay with zero
/// derivative at the crossover, so a face drifting across `Ma_limit` does not see a kink in its
/// flux; `phi = 1` recovers classical HLLC identically.
///
/// nothing else modulates this. a detector that raised `phi` back toward one at shocks, at contact
/// discontinuities, or in grid-aligned high-mach flow would be adding dissipation, which is the
/// opposite of what the scheme is for — every competing shock-stable HLLC variant stabilizes by
/// adding dissipation somewhere, and the point of this one is that it removes it instead. Where a
/// strong shock genuinely needs a more dissipative flux, that is the job of a solver fallback
/// (`ShockwaveLimiter::Quirk`), not of a term buried in the low-mach scaling.
#[inline]
pub fn adaptive_phi<S: Scalar>(vn_l: S, vn_r: S, cs_l: S, cs_r: S) -> S {
    let half_pi = S::from_f64(std::f64::consts::FRAC_PI_2);
    let ma = local_mach(vn_l, vn_r, cs_l, cs_r);
    let ratio = (ma / S::from_f64(MACH_LIMIT)).min(S::ONE);
    (ratio * half_pi).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scaling_depends_only_on_the_ratio_of_the_two_speeds() {
        // `phi` is a function of a mach number, which is dimensionless, so scaling the flow speed
        // and the sound speed together must leave it unchanged. a formula that compared a velocity
        // against an absolute threshold would instead move with the units, making the solver's
        // dissipation depend on whether lengths are metres or parsecs.
        let reference = adaptive_phi(0.03, 0.03, 1.0, 1.0);
        assert!(
            reference < 1.0,
            "this state must sit on the ramp for the invariance to be non-trivial, got {reference}"
        );
        for scale in [1e-100, 1e-3, 1.0, 1e3, 1e100] {
            let got = adaptive_phi(0.03 * scale, 0.03 * scale, scale, scale);
            // exact: `phi` divides the two before anything transcendental, and both scale
            // identically, so the quotient is bit-for-bit the same.
            assert_eq!(
                got, reference,
                "rescaling both speeds by {scale:e} changed phi"
            );
        }
    }

    #[test]
    fn the_scaling_takes_the_larger_mach_number_of_the_two_sides() {
        // eq 25 is a max over the two sides, each against its OWN sound speed. taking one side, or
        // an average, would let a face with one nearly-stagnant side reduce its dissipation while
        // the other side is moving — dissipation set by half the Riemann problem.
        let hot = adaptive_phi(0.001, 0.5, 1.0, 1.0);
        let both = adaptive_phi(0.5, 0.5, 1.0, 1.0);
        assert_eq!(hot, both, "the moving side must set phi");
        assert_eq!(
            adaptive_phi(0.5, 0.001, 1.0, 1.0),
            both,
            "and it must not matter which side it is"
        );
        // the sound speeds are per-side too: the colder side reaches the limit at a lower velocity.
        let cold_right = adaptive_phi(0.01, 0.01, 1.0, 0.05);
        assert!(
            cold_right > adaptive_phi(0.01, 0.01, 1.0, 1.0),
            "a colder right state raises its own mach number and so raises phi"
        );
    }

    #[test]
    fn the_sign_of_the_velocity_does_not_matter() {
        // the mach number is a magnitude: a face is equally in the low-mach regime whether the flow
        // crosses it one way or the other, and a signed ratio would make the scaling asymmetric
        // under a reflection of the grid.
        assert_eq!(
            adaptive_phi(0.02, -0.03, 1.0, 1.0),
            adaptive_phi(-0.02, 0.03, 1.0, 1.0)
        );
        assert_eq!(local_mach(-0.05, 0.01, 1.0, 1.0), 0.05);
    }
}
