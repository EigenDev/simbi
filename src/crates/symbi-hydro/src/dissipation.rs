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
// the scaling is keyed on the FACE-NORMAL velocity component, floored by a
// compressibility-consistency clamp on the face pressure jump: the low-mach reduction is valid
// only under the incompressible fluctuation scaling `dp/p ~ gamma Ma^2`, and a face whose
// pressure data contradicts that premise (a gravity-stratified balance, `dp/p ~ dx/H` at any
// mach) recovers classical HLLC dissipation — see `adaptive_phi`.
//
// usage:
//   let phi = adaptive_phi(vn_l, vn_r, cs_l, cs_r, p_l, p_r);
// =============================================================================

use symbi_ir::algebra::Scalar;

/// shockwave limiter selector for the HLLC riemann solver. picks the flavor of
/// HLLC the regime emits at a face:
///
///   - `Standard`     — plain HLLC (toro / mignone-bodo star state).
///   - `Fleischmann`  — newtonian only: HLLC-LM exactly as published (`fleischmann_phi`),
///                      the sine ramp on the acoustic signal speeds cut off at `MACH_LIMIT`.
///                      relativistic regimes ignore it (no relativistic LM correction).
///   - `FleischmannClamped` — newtonian only: the same ramp floored by a
///                      compressibility-consistency clamp on the face pressure jump
///                      (`adaptive_phi`). NOT in the published scheme; it restores classical
///                      dissipation wherever the face pressure data contradicts the
///                      incompressible fluctuation scaling, which on a gravity-stratified
///                      background is most faces.
///   - `Acoustic`     — newtonian only: the same acoustic-dissipation scaling keyed on the
///                      ACOUSTIC CONTENT of the face data (`acoustic_phi`) rather than on a
///                      reference mach number. carries no tuned constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShockwaveLimiter {
    Standard,
    Fleischmann,
    FleischmannClamped,
    Acoustic,
}

impl Default for ShockwaveLimiter {
    fn default() -> Self {
        ShockwaveLimiter::Standard
    }
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

/// margin of the compressibility-consistency clamp: the acoustic dissipation is restored once
/// the face pressure jump exceeds `STRAT_MARGIN * Ma_local^2` relative pressure units — several
/// times the `dp/p ~ gamma Ma^2` fluctuation scale of genuinely low-mach (incompressible-limit)
/// flow, so smooth subsonic turbulence never engages it, while a gravity-stratified balance
/// (`dp/p ~ dx/H` at any mach — two orders above the incompressible scale in an accretor
/// atmosphere) restores classical HLLC exactly where the hydrostatic residual must be damped.
pub const STRAT_MARGIN: f64 = 4.0;

/// floor of the clamp's reference scale: the stagnation-pressure ceiling of ramp-active flow.
/// the largest pressure structure a flow below `MACH_LIMIT` can build is the stagnation bump
/// `dp/p ~ gamma Ma_limit^2 / 2 ~ 0.008`; a face jump above twice that cannot come from
/// incompressible dynamics in the band where the ramp reduces dissipation, however small the
/// face's own normal mach is. without this floor, a stagnation face inside a vortex (normal
/// mach ~ 0, finite jump from the neighboring flow) would fire the clamp inside the very
/// turbulence the low-mach scheme exists for.
pub const INCOMP_JUMP_CEIL: f64 = 0.02;

/// the acoustic-dissipation scaling AS PUBLISHED (Fleischmann, Adami & Adams 2020):
///
///   `phi = sin( min(1, Ma_local / MACH_LIMIT) * pi/2 )`,
///
/// applied to the acoustic signal speeds `S_L` and `S_R` alone, so the acoustic dissipation
/// falls off in proportion to the local flow speed rather than the sound speed while the
/// advective dissipation carried by the contact speed is untouched. the sine gives a smooth
/// decay with zero derivative at the crossover, so a face drifting across `MACH_LIMIT` sees no
/// kink in its flux; `phi = 1` recovers classical HLLC identically.
///
/// this is the whole modification the paper specifies — §4: "the effective range of the
/// shock-transverse Mach number modification in the HLLC-LM solver is always limited to local
/// Mach numbers lower than 0.1". the paper's validation suite includes a gravitational
/// Rayleigh-Taylor instability, where the reduced contact-line dissipation is the reported
/// benefit, so a stratified background is within the scheme's demonstrated range.
///
/// the scheme leaves the hydrostatic residual of a STAGNANT stratified column undamped: a
/// limited reconstruction leaves an O(dx^2) face jump on a curved profile, the flux at rest is
/// the upwind dissipation acting on it, and removing that dissipation lets the residual ring.
/// the cure is to remove the jump — see `crate::hydrostatic` — not to restore the dissipation,
/// which costs the low-mach accuracy everywhere the stratification reaches.
/// `mach_limit` is the reference mach number the ramp saturates at, `MACH_LIMIT` in the paper's
/// own experiments. it is a PARAMETER rather than a constant because it sets how much of the
/// flow the reduction reaches, and the right value depends on what the run is resolving: the
/// paper reports 0.1 for its shock suite, while a deeply subsonic flow whose whole dynamic range
/// sits below that is unaffected by the ramp unless the limit is raised to meet it. two limits
/// are degenerate rather than invalid — 0 reduces nothing and recovers classical HLLC
/// everywhere, 1 reduces all the way to the sonic point — so the value is range-checked where a
/// user sets it, not on this path.
#[inline]
pub fn fleischmann_phi<S: Scalar>(vn_l: S, vn_r: S, cs_l: S, cs_r: S, mach_limit: S) -> S {
    let half_pi = S::from_f64(std::f64::consts::FRAC_PI_2);
    let ma = local_mach(vn_l, vn_r, cs_l, cs_r);
    let ratio = (ma / mach_limit).min(S::ONE);
    (ratio * half_pi).sin()
}

/// the acoustic-dissipation scaling: the fleischmann sine ramp, floored by the
/// compressibility-consistency clamp,
///
///   `phi = max( sin(min(1, Ma/Ma_limit) * pi/2),
///               min(1, (|dp|/p_min) / max(STRAT_MARGIN Ma^2, INCOMP_JUMP_CEIL)) )`.
///
/// applied to the acoustic signal speeds `S_L` and `S_R` alone, so the acoustic dissipation falls
/// off in proportion to the local flow speed rather than the sound speed, while the advective
/// dissipation carried by the contact speed is untouched. the sine gives a smooth decay with zero
/// derivative at the crossover, so a face drifting across `Ma_limit` does not see a kink in its
/// flux; `phi = 1` recovers classical HLLC identically.
///
/// the clamp enforces the ramp's own domain of validity. the low-mach reduction is derived under
/// the incompressible-limit scaling `dp/p = O(gamma Ma^2)`; a gravity-stratified balance violates
/// that premise categorically (`dp/p ~ dx/H`, mach-independent), and there the acoustic
/// dissipation is exactly what damps the hydrostatic residual — removing it lets the residual
/// ring dispersively and the entropy floor `K >= K_0` fails on a sealed stratified column (the
/// regime a solid accretor surface holds its masked cells in). the clamp restores dissipation
/// only up to classical HLLC, never beyond, and only where the pressure data contradicts the
/// incompressible premise:
///   - a contact discontinuity has no pressure jump, so the clamp NEVER fires there and the
///     scheme's low contact dissipation is preserved by construction;
///   - a face transverse to a grid-aligned shock sees near-uniform pressure along the front, so
///     the shock-stability mechanism (the reason the ramp exists) is untouched;
///   - shock/contact detectors that ADD dissipation beyond classical HLLC remain out of scope
///     here — robustness at strong shocks is the job of a solver fallback, not this scaling.
#[inline]
pub fn adaptive_phi<S: Scalar>(vn_l: S, vn_r: S, cs_l: S, cs_r: S, p_l: S, p_r: S) -> S {
    let ma = local_mach(vn_l, vn_r, cs_l, cs_r);
    // the ramp is the published scaling verbatim; the clamp below is the only departure, so
    // there is one definition of the ramp and the two arms cannot drift apart. the reference
    // mach number is FIXED at `MACH_LIMIT` on this arm: `INCOMP_JUMP_CEIL` is derived from it
    // (the stagnation bump a flow below the limit can build, `gamma Ma_limit^2`), so a limit
    // that moved without the ceiling moving with it would compare a jump against the wrong
    // incompressible scale. the clamp-free arm carries the runtime knob.
    let ramp = fleischmann_phi(vn_l, vn_r, cs_l, cs_r, S::from_f64(MACH_LIMIT));
    // relative face pressure jump against the largest jump an incompressible flow in the
    // ramp-active band can present: the local `STRAT_MARGIN Ma^2` fluctuation scale, floored
    // by the stagnation-pressure ceiling `INCOMP_JUMP_CEIL`. the floor is what keeps a
    // grid-aligned stagnation face inside a low-mach vortex (face-normal mach ~ 0, finite
    // neighbor-driven jump) from dividing by zero and firing the clamp inside legitimate
    // turbulence — there the clamp stays bounded by `jump / INCOMP_JUMP_CEIL`, well under
    // the ramp of any face that moves. a stratified balance exceeds BOTH scales and
    // saturates the clamp to classical HLLC.
    let jump = (p_l - p_r).abs() / p_l.min(p_r);
    let scale = (S::from_f64(STRAT_MARGIN) * ma * ma).max(S::from_f64(INCOMP_JUMP_CEIL));
    let clamp = (jump / scale).min(S::ONE);
    ramp.max(clamp)
}

/// the smallest velocity jump, in units of the local sound speed, that is treated as resolved
/// rather than as roundoff. this is a floating-point robustness floor, NOT a physical
/// threshold: it exists so a face with identically equal states divides by something, and its
/// value is far below any jump a discretization can represent.
pub const JUMP_EPS: f64 = 1.0e-30;

/// the acoustic-consistency scaling of the acoustic dissipation: scale by how much of the face
/// data is ACOUSTIC, measured against the impedance relation, rather than by a flow speed.
///
/// the scaling is the LARGER of two dimensionless demands on the face, capped at one:
///
///   `phi = min(1, max( Ma_local, |dp - dp_balance| / (rho c^2) ))`.
///
/// the first term is the low-mach requirement: the acoustic dissipation must fall with the
/// flow speed or it overwhelms the advective flux as `Ma -> 0` (Guillard & Viozat). saturating
/// at `Ma = 1` is where the acoustic and advective scales genuinely meet, so no reference mach
/// number is chosen. the second is a floor set by the UNSUPPORTED pressure structure the face
/// carries: a pressure jump that no body force holds up and no flow explains has to be damped,
/// whatever the mach number is.
///
/// taking the LARGER of the two is what separates the two ways a face can present a pressure
/// jump with no velocity behind it:
///
///   - the transverse face of a grid-aligned shock carries neither — the front is smooth along
///     itself, so both terms are small and the acoustic dissipation is reduced. that reduction
///     IS the carbuncle cure, and a sensor built as the RATIO of the two terms gets this
///     backwards: the ratio diverges when the velocity jump vanishes faster than the pressure
///     jump, which is exactly this configuration, and restores the dissipation that drives the
///     instability. measured, not argued — see `odd_even_decoupling.rs`;
///   - a face in FORCE BALANCE carries a large pressure jump that IS explained, so subtracting
///     `dp_balance` empties the second term and the low-mach reduction survives across a
///     stratified atmosphere instead of being switched off throughout it. whatever the balance
///     fails to account for is the residual, and it raises the floor in proportion — the
///     property a stratified column needs, since an undamped hydrostatic residual rings at
///     grid scale.
///
/// MEASURED LIMITATION — this sensor does NOT yet hold the adiabatic entropy floor on a
/// stagnant stratified column when `dp_balance` is zero. the floor it sets there is
/// `(dp/p) / gamma`, some fifteen times weaker than what damps the hydrostatic residual, and
/// a sealed column loses ~1.7 percent of its entropy where the mach-limited ramp loses none.
/// the two demands genuinely oppose: a transverse shock face and a hydrostatic residual both
/// present a small pressure jump behind a vanishing velocity jump, and face-local data alone
/// does not separate them — a floor strong enough for the second re-creates the first. supply
/// `dp_balance`, or use the mach-limited ramp, on any run with a stratified background.
///
/// the remaining cases follow from the same expression: smooth low-mach flow carries
/// `dp ~ rho u du`, so the jump term is `O(Ma^2)` and the mach term wins, giving `phi ~ Ma`;
/// a shock runs at `Ma >= 1` and saturates; a contact carries no pressure jump at all and is
/// left to the contact wave, which this scaling never touches.
///
/// `dp_balance` is the pressure jump the face's momentum sources support across it,
/// `rho_bar (f . n) dx`, for ANY body force `f` — gravity, rotation, magnetic tension,
/// radiation. pass zero when the run has none, or when the balance is not available: the
/// sensor then reads a balanced stratification as fully acoustic and returns `phi = 1`, which
/// is the conservative reading and reproduces a compressibility clamp without carrying one.
///
/// SELF-CORRECTING AGAINST CHECKERBOARD. the known hazard of scaling dissipation to `Ma` is
/// pressure-velocity decoupling in the incompressible limit. here the numerator IS the pressure
/// jump, so a grid-scale pressure oscillation raises `b`, raises `phi`, and restores the
/// dissipation that damps it. the mechanism that would run away supplies its own brake.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn acoustic_phi<S: Scalar>(
    vn_l: S,
    vn_r: S,
    cs_l: S,
    cs_r: S,
    p_l: S,
    p_r: S,
    rho_l: S,
    rho_r: S,
    dp_balance: S,
) -> S {
    let half = S::from_f64(0.5);
    let cs = (cs_l + cs_r) * half;
    let rho = (rho_l + rho_r) * half;
    // the DYNAMIC pressure jump in acoustic units: what the face carries beyond whatever a
    // body force holds up.
    let jump = ((p_l - p_r) - dp_balance).abs() / (rho * cs * cs);
    local_mach(vn_l, vn_r, cs_l, cs_r).max(jump).min(S::ONE)
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
        let reference = adaptive_phi(0.03, 0.03, 1.0, 1.0, 1.0, 1.0);
        assert!(
            reference < 1.0,
            "this state must sit on the ramp for the invariance to be non-trivial, got {reference}"
        );
        for scale in [1e-100, 1e-3, 1.0, 1e3, 1e100] {
            let got = adaptive_phi(0.03 * scale, 0.03 * scale, scale, scale, 1.0, 1.0);
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
        let hot = adaptive_phi(0.001, 0.5, 1.0, 1.0, 1.0, 1.0);
        let both = adaptive_phi(0.5, 0.5, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(hot, both, "the moving side must set phi");
        assert_eq!(
            adaptive_phi(0.5, 0.001, 1.0, 1.0, 1.0, 1.0),
            both,
            "and it must not matter which side it is"
        );
        // the sound speeds are per-side too: the colder side reaches the limit at a lower velocity.
        let cold_right = adaptive_phi(0.01, 0.01, 1.0, 0.05, 1.0, 1.0);
        assert!(
            cold_right > adaptive_phi(0.01, 0.01, 1.0, 1.0, 1.0, 1.0),
            "a colder right state raises its own mach number and so raises phi"
        );
    }

    #[test]
    fn the_sign_of_the_velocity_does_not_matter() {
        // the mach number is a magnitude: a face is equally in the low-mach regime whether the flow
        // crosses it one way or the other, and a signed ratio would make the scaling asymmetric
        // under a reflection of the grid.
        assert_eq!(
            adaptive_phi(0.02, -0.03, 1.0, 1.0, 1.0, 1.0),
            adaptive_phi(-0.02, 0.03, 1.0, 1.0, 1.0, 1.0)
        );
        assert_eq!(local_mach(-0.05, 0.01, 1.0, 1.0), 0.05);
    }
}
