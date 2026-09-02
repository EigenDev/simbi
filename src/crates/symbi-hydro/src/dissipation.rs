// =============================================================================
// dissipation.rs
//
// the two dissipation rescalings of the HLLC+ riemann solver (Chen, Lin, Li & Yan,
// SIAM J. Sci. Comput. 42:B921, 2020; the accuracy half restated as a framework in
// Chen, Li, Li, Yuan & Gao, J. Comput. Phys. 456:111027, 2022).
//
// a Godunov flux damps the two velocity jumps a face carries at the acoustic impedance
// `rho c`, and each mis-serves a distinct regime. on the normal jump that damping exceeds the
// convective flux it corrects by an order in `1/Ma`, so low-mach pressure fluctuations pick up
// an `O(Ma)` error where the continuous Euler system gives `O(Ma^2)`. on the transverse jump it
// is instead too weak along a grid-aligned shock, where the front is smooth in its own plane,
// and a perturbation of the front grows through those faces into the carbuncle.
//
// both terms are rescaled in place, leaving every signal speed, the contact speed and both star
// states classical: `mach_scale` cuts the normal-jump damping to the convective magnitude, and
// `shear_weight` raises the transverse-jump damping across a shock. each reads the local flow
// alone and saturates at the sonic point, so the solver carries no reference mach number.
//
// usage:
//   let g = mach_scale(speed_l, speed_r, cs_l, cs_r);
//   let w = shear_weight(neighborhood_pressure_ratio, g);
// =============================================================================

use symbi_carrier::Scalar;

/// shockwave limiter selector for the HLLC riemann solver. picks the flavor of
/// HLLC the regime emits at a face:
///
///   - `Standard` — plain HLLC (toro / mignone-bodo star state).
///   - `HllcPlus` — plain HLLC plus two additive corrections that rescale the dissipation on
///                  the normal velocity jump (`mach_scale`) and on the transverse one
///                  (`shear_weight`), leaving every signal speed, the contact speed, the
///                  contact pressure and both star states at their classical values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShockwaveLimiter {
    Standard,
    HllcPlus,
}

impl Default for ShockwaveLimiter {
    fn default() -> Self {
        ShockwaveLimiter::Standard
    }
}

/// keyed on the full speed of each side rather than on its face-normal component. the two
/// scalings this feeds act on the velocity jump, whose own direction already carries the
/// geometry — the normal jump for the accuracy term, the in-plane jump for the shear term —
/// so what the mach number has to report is how the flow speed compares to the acoustic speed,
/// and a face whose flow runs along it is no more incompressible for that. reading the normal
/// component instead sends the shear weight to zero across exactly the shock-transverse faces
/// the weight exists to reach, since a planar front carries its whole velocity across the face
/// rather than through it.
#[inline]
pub fn mach_scale<S: Scalar>(speed_l: S, speed_r: S, cs_l: S, cs_r: S) -> S {
    (speed_l / cs_l)
        .abs()
        .max((speed_r / cs_r).abs())
        .min(S::ONE)
}

/// the strength of the transverse shear viscosity that makes the anti-dissipation family
/// shock-stable (Chen, Lin, Li & Yan, SIAM J. Sci. Comput. 42:B921, 2020, eq. 23):
///
///   `g = 1 - h^Ma`,
///
/// where `h` is the smallest pressure ratio `min(p_a/p_b, p_b/p_a)` across any interface of
/// either cell adjoining this face, and `Ma` the local mach number.
///
/// the grid-aligned shock instability is carried by the transverse velocity jump: along a
/// planar front the flow is smooth, so the transverse Riemann problems see a vanishing
/// velocity component and receive almost no dissipation, and a perturbation of the front
/// grows through them. adding a dissipation proportional to that transverse jump damps the
/// mechanism directly, which is what a scaling of the normal-jump term leaves untouched.
///
/// the weight turns that dissipation on where a shock is and off everywhere else, through two
/// factors that must both be large. `h` measures pressure structure: a shock drives it toward
/// zero and `g` toward one, while smooth flow holds it near one and `g` near zero, so the
/// shear viscosity is absent from a boundary layer or a contact. the exponent is the mach
/// number, which sends `g` to zero throughout a subsonic region whatever the pressure ratio,
/// so a stratified atmosphere — carrying a large pressure ratio across every cell and no
/// shock at all — receives none of it.
///
/// the pressure ratio is read over the neighborhood rather than across this face alone. a
/// shock-transverse face is precisely one whose own two states are nearly equal, so its own
/// ratio reports smooth flow; the pressure structure that identifies it sits on the
/// shock-normal interfaces of the same two cells.
#[inline]
pub fn shear_weight<S: Scalar>(pressure_ratio: S, mach: S) -> S {
    // `ratio^mach` composed from the logarithm rather than taken as a general power. the
    // ratio is `min(a/b, b/a)` over positive pressures and so lies in (0, 1]; the floor
    // keeps the logarithm finite where a floor-level pressure drives the ratio to zero,
    // which is also where the general power's `0^0` convention and this composition would
    // otherwise disagree.
    //
    // both limits that carry physical meaning stay exact. at `ratio = 1` (smooth flow)
    // `ln 1 = 0` and `exp 0 = 1`, and at `mach = 0` (a subsonic region) the product is zero
    // and the exponential is one, so each returns a weight of exactly zero and the shear
    // viscosity is absent bit-for-bit rather than to rounding.
    let base = pressure_ratio.max(S::from_f64(f64::MIN_POSITIVE));
    S::ONE - (mach * base.ln()).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the shear weight switches the transverse viscosity off exactly, not to rounding, in
    /// the two regimes where it must be absent: smooth flow (a unit pressure ratio) and a
    /// subsonic region (zero mach). a residue of order epsilon here would seed viscosity in
    /// a stratified atmosphere, which carries a large pressure ratio across every cell and
    /// no shock at all.
    #[test]
    fn the_shear_weight_vanishes_exactly_where_the_viscosity_must_be_absent() {
        for mach in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let w = shear_weight(1.0_f64, mach);
            assert_eq!(w, 0.0, "smooth flow at mach {mach} left a weight {w:e}");
        }
        for ratio in [1.0e-12, 1.0e-4, 0.01, 0.5, 1.0] {
            let w = shear_weight(ratio, 0.0_f64);
            assert_eq!(w, 0.0, "subsonic at ratio {ratio} left a weight {w:e}");
        }
        // a floor-level pressure drives the ratio to zero; the weight stays a number.
        assert!(shear_weight(0.0_f64, 0.0_f64).is_finite());
        assert!(shear_weight(0.0_f64, 1.0_f64).is_finite());
        // and it stays a weight: bounded in [0, 1] wherever it is consulted.
        for ratio in [0.0, 1.0e-8, 0.3, 1.0] {
            for mach in [0.0, 0.4, 1.0] {
                let w = shear_weight(ratio, mach);
                assert!(
                    (0.0..=1.0).contains(&w),
                    "weight {w} out of range at ({ratio}, {mach})"
                );
            }
        }
    }

    #[test]
    fn the_scalings_depend_only_on_ratios_of_speeds() {
        // both scalings are functions of a mach number, which is dimensionless, so scaling the
        // flow speed and the sound speed together leaves them unchanged. a formula comparing a
        // velocity against an absolute threshold would move with the units, making the solver's
        // dissipation depend on whether lengths are meters or parsecs.
        //
        // the rescalings are powers of two, where a float carries them in the exponent alone, so
        // the quotient the scaling forms is the same to the last bit and the invariance is
        // testable exactly. a decimal rescaling rounds twice — once into the scaled speed and
        // once out of the division — and is checked below against that rounding instead.
        let reference = mach_scale(0.03, 0.03, 1.0, 1.0);
        assert!(
            reference < 1.0,
            "this state must sit below saturation for the invariance to be non-trivial, got \
             {reference}"
        );
        for exponent in [-300i32, -10, 0, 10, 300] {
            let scale = (2.0f64).powi(exponent);
            assert_eq!(
                mach_scale(0.03 * scale, 0.03 * scale, scale, scale),
                reference,
                "rescaling both speeds by 2^{exponent} changed the scaling"
            );
        }
        // a rescaling that is not a power of two lands within one rounding of the same value.
        for scale in [1.0e-100f64, 1.0e-3, 1.0e3, 1.0e100] {
            let got = mach_scale(0.03 * scale, 0.03 * scale, scale, scale);
            assert!(
                (got - reference).abs() <= 4.0 * f64::EPSILON * reference,
                "rescaling both speeds by {scale:e} moved the scaling to {got} from \
                 {reference}, beyond the rounding of the two decimal conversions"
            );
        }
    }

    #[test]
    fn the_mach_scaling_takes_the_faster_of_the_two_sides() {
        // a max over the two sides, each against its own sound speed. taking one side, or an
        // average, would let a face with one nearly-stagnant side reduce its dissipation while
        // the other side is moving — dissipation set by half the Riemann problem.
        let both = mach_scale(0.5, 0.5, 1.0, 1.0);
        assert_eq!(
            mach_scale(0.001, 0.5, 1.0, 1.0),
            both,
            "the moving side sets it"
        );
        assert_eq!(
            mach_scale(0.5, 0.001, 1.0, 1.0),
            both,
            "and it must not matter which side it is"
        );
        // the sound speeds are per-side too: the colder side reaches saturation at a lower speed.
        assert!(
            mach_scale(0.01, 0.01, 1.0, 0.05) > mach_scale(0.01, 0.01, 1.0, 1.0),
            "a colder right state raises its own mach number and so raises the scaling"
        );
    }

    #[test]
    fn the_mach_scaling_saturates_at_the_sonic_point() {
        // at and above the sonic point the correction is inert: the scaling returns one, the
        // rescaling factor `g - 1` vanishes, and the flux is classical HLLC exactly. this is
        // what makes the solver safe on a shock without any reference mach number to set.
        for speed in [1.0, 1.5, 40.0] {
            assert_eq!(
                mach_scale(speed, speed, 1.0, 1.0),
                1.0,
                "the scaling must saturate at Ma = {speed}"
            );
        }
        // and it is a magnitude: a face is equally low-mach whichever way the flow crosses it.
        assert_eq!(
            mach_scale(-0.05, 0.01, 1.0, 1.0),
            mach_scale(0.05, 0.01, 1.0, 1.0)
        );
    }

    #[test]
    fn the_shear_weight_needs_both_a_pressure_jump_and_a_flow() {
        // the transverse viscosity appears at a shock and nowhere else, which takes two
        // conditions at once: pressure structure, and a flow fast enough to carry it.
        // a smooth interface (ratio one) is exempt whatever the speed.
        assert_eq!(shear_weight(1.0, 1.0), 0.0);
        assert_eq!(shear_weight(1.0, 0.01), 0.0);
        // a stagnant stratified column carries a large pressure ratio and no flow; the mach
        // exponent empties the weight, which is what keeps the viscosity out of a hydrostatic
        // atmosphere.
        assert!(
            shear_weight(0.2, 0.0) == 0.0,
            "at rest the weight must vanish however strong the stratification"
        );
        assert!(
            shear_weight(0.2, 1.0e-3) < 1.0e-2,
            "a deeply subsonic stratified face must stay effectively exempt"
        );
        // a strong shock carries both, and the weight approaches its full strength.
        assert!(
            shear_weight(0.02, 1.0) > 0.97,
            "a strong shock at the sonic point must receive nearly the whole viscosity"
        );
        // monotone in the jump at fixed speed: a stronger shock draws more viscosity.
        assert!(shear_weight(0.02, 1.0) > shear_weight(0.5, 1.0));
    }
}
