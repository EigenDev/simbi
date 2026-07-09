// =============================================================================
// drain.rs
//
// the well-posed uniform-scaling volumetric drain (docs/ideas/accretor.md §2):
// the exact-exponential relaxation `U -> U exp(-chi dt / tau)`, applied
// post-hydro on the masked cells. scaling EVERY conserved component by the SAME
// factor `f` leaves each intensive primitive (velocity, specific internal
// energy, sound speed, temperature) pointwise invariant, so the drain is
// positivity-preserving for ANY dt (no CFL tax), injects no acoustic/entropy
// wave, and preserves the characteristic decomposition -- the no-reflection
// property at `r_mask` inside the sonic surface. the accretion rate is
// EMERGENT: the cell-integrated `U_old - U_new` reduced over the mask is a
// functional of the solved flow, never a target rate fed in.
//
// this is the well-posed replacement for the mass-only KMK04 sink
// (`effects::accretion_source`): a mass-only sink removes rho at fixed momentum,
// so it CHANGES the velocity when it drains, re-couples the primitive channels,
// injects a back-pressure artifact, and (in the property algebra) destroys the
// exact three-exponential decoupling. uniform scaling does none of these.
//
// usage:
//   let chi = drain_mask(dist, r_mask, w);
//   let tau = drain_timescale(dx, c_s, c_drain);
//   let (drained, delta) = drain_cell(&cons, chi, tau, dt, volume, body_idx);
// =============================================================================

use symbi_hydro::energy::{EnergyModel, EnergySlot};
use symbi_hydro::state::ConsG;
use symbi_ir::algebra::Scalar;

use crate::body_delta::BodyDelta;

/// the mollified spherical mask value at coordinate distance `dist` from the
/// mask center: `chi = 0.5 (1 - tanh((dist - r_mask) / w))`. `chi -> 1` well
/// inside `r_mask`, `-> 0` well outside, with a tanh ramp of width `w` across
/// the surface. the mollification keeps the drain isotropic on a cartesian grid
/// (no staircasing) and stops the mask edge from acting as a sharp reflecting
/// wall.
#[inline]
pub fn drain_mask<S: Scalar>(dist: S, r_mask: S, w: S) -> S {
    S::from_f64(0.5) * (S::ONE - ((dist - r_mask) / w).tanh())
}

/// the drain timescale `tau = c_drain * dx / c_s` (local sound speed). drains
/// fast enough that the mask stays evacuated and never backs up; in the
/// well-posed regime (`r_mask` inside the sonic surface) the emergent rate is
/// INSENSITIVE to `c_drain` once it is small enough -- the sonic surface
/// regulates the rate, not the drain. `c_drain` is a convergence-study dial,
/// never tuned to hit a target rate.
#[inline]
pub fn drain_timescale<S: Scalar>(dx: S, c_s: S, c_drain: S) -> S {
    c_drain * dx / c_s
}

/// the exact-exponential uniform-scaling drain on one masked cell's conserved
/// vector. returns the DRAINED state `U exp(-chi dt/tau)` and the cell's exact
/// contribution to the body -- the cell-integrated `U_old - U_new` (absorbed
/// mass and drag force), which makes gas+body conservation exact to machine
/// precision. the SAME scalar factor multiplies EVERY conserved component (the
/// accretor.md §2.2 DESIGN INVARIANT), including the energy slot, so the
/// intensive primitive state is untouched.
#[inline]
pub fn drain_cell<S: Scalar, const D: usize, E: EnergyModel>(
    cons: &ConsG<S, D, E>,
    chi: S,
    tau: S,
    dt: S,
    volume: S,
    idx: usize,
) -> (ConsG<S, D, E>, BodyDelta<S, D>) {
    // f = exp(-chi dt / tau) in (0, 1]; f = 1 (exact no-op) where chi = 0.
    let f = (-(chi * dt / tau)).exp();
    let drained = *cons * f; // UNIFORM scale of den, mom, AND the energy slot
    let absorbed = *cons - drained; // = cons * (1 - f): the body's exact gain
    let mut delta = BodyDelta::new(idx);
    // cell-integrated absorbed mass (dM), drag force (dP/dt), and total energy (dE). the
    // energy slot is zero for the isothermal regime (no energy channel). together these make
    // gas+body conservation of mass, momentum, AND energy exact to machine precision.
    delta.mass_delta = absorbed.den * volume;
    delta.force_delta = absorbed.mom.scale(volume / dt);
    delta.energy_delta = absorbed.nrg.value() * volume;
    (drained, delta)
}

/// the adiabatic sound speed recovered directly from the UPDATED conserved state. the
/// drain runs post-godunov but BEFORE c2p (spec ordering: hydro -> drain -> c2p+floors),
/// so the stored primitive is stale; the drain timescale's `c_s` must come from the
/// just-updated `cons`. `e_int = (nrg - 0.5 |mom|^2 / den) / den`, then
/// `c_s = sqrt(gamma (gamma - 1) e_int)`. a one-line inversion (no root-find: the sound
/// speed needs only the internal energy, not the full primitive). regime-local; the caller
/// supplies gamma. NOT used by the isothermal regime, which carries a fixed `c_s = c_iso`.
#[inline]
pub fn sound_speed_from_cons<S: Scalar>(den: S, mom_sq: S, nrg: S, gamma: S) -> S {
    let e_int = (nrg - S::from_f64(0.5) * mom_sq / den) / den;
    (gamma * (gamma - S::ONE) * e_int).sqrt()
}

/// the per-cell drain the post-godunov pass iterates: compose the mask (from the
/// cell's coordinate distance `dist` to the mask center), the local timescale (from
/// `dx` and the cell sound speed `c_s`), and the exact-exponential drain. `c_s` is
/// passed in (the caller has the local primitive) so this stays regime-agnostic. a
/// cell outside the mask (`chi -> 0`) is a bit-exact no-op that contributes nothing
/// to the body, so the pass may early-return on `dist` before calling this.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn drain_body_cell<S: Scalar, const D: usize, E: EnergyModel>(
    cons: &ConsG<S, D, E>,
    dist: S,
    r_mask: S,
    w: S,
    dx: S,
    c_s: S,
    c_drain: S,
    dt: S,
    volume: S,
    idx: usize,
) -> (ConsG<S, D, E>, BodyDelta<S, D>) {
    let chi = drain_mask(dist, r_mask, w);
    let tau = drain_timescale(dx, c_s, c_drain);
    drain_cell(cons, chi, tau, dt, volume, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Tensor;
    use symbi_hydro::energy::Adiabatic;

    type Cons3 = ConsG<f64, 3, Adiabatic>;

    fn sample_cons() -> Cons3 {
        // a moving, pressurized cell: den, mom = den*v, nrg = total energy.
        let den = 2.5;
        let v = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = 1.7; // specific internal energy
        let nrg = den * (e_int + 0.5 * v.dot(&v));
        ConsG { den, mom: v.scale(den), nrg }
    }

    // the DESIGN INVARIANT (accretor.md §2.2): uniform scaling leaves the intensive
    // primitive state pointwise invariant. a mass-only sink would change the velocity.
    #[test]
    fn uniform_scaling_leaves_intensive_primitives_invariant() {
        let cons = sample_cons();
        let (drained, _) = drain_cell(&cons, 0.8, 3.0, 0.4, 1.0, 0);
        // velocity v = mom / den: unchanged.
        for k in 0..3 {
            let v0 = cons.mom[k] / cons.den;
            let v1 = drained.mom[k] / drained.den;
            assert!((v0 - v1).abs() < 1e-14, "velocity {k} changed: {v0} -> {v1}");
        }
        // specific internal energy e_int = nrg/den - 0.5|v|^2: unchanged.
        let vsq0 = (0..3).map(|k| (cons.mom[k] / cons.den).powi(2)).sum::<f64>();
        let vsq1 = (0..3).map(|k| (drained.mom[k] / drained.den).powi(2)).sum::<f64>();
        let e0 = cons.nrg / cons.den - 0.5 * vsq0;
        let e1 = drained.nrg / drained.den - 0.5 * vsq1;
        assert!((e0 - e1).abs() < 1e-14, "specific internal energy changed: {e0} -> {e1}");
    }

    // exact gas+body conservation: the body's gain is exactly what the gas lost.
    #[test]
    fn conservation_is_exact() {
        let cons = sample_cons();
        let vol = 0.05;
        let (drained, delta) = drain_cell(&cons, 1.0, 2.0, 0.5, vol, 0);
        // mass: gas lost (den_old - den_new)*V equals the body's mass_delta.
        let gas_mass_lost = (cons.den - drained.den) * vol;
        assert!((gas_mass_lost - delta.mass_delta).abs() < 1e-15);
        // momentum: gas lost (mom_old - mom_new)*V equals the body's absorbed momentum (= F*dt).
        for k in 0..3 {
            let gas_mom_lost = (cons.mom[k] - drained.mom[k]) * vol;
            assert!((gas_mom_lost - delta.force_delta[k] * 0.5).abs() < 1e-15);
        }
        // energy: gas lost (nrg_old - nrg_new)*V equals the body's absorbed energy_delta.
        let gas_nrg_lost = (cons.nrg - drained.nrg) * vol;
        assert!((gas_nrg_lost - delta.energy_delta).abs() < 1e-15);
    }

    // positivity for ANY dt (the whole reason for the exponential): f = exp(-x) in
    // [0, 1] for x >= 0, so drained density is in [0, den] -- NEVER negative, never
    // NaN/Inf. an explicit stiff source would overshoot to negative density and need a
    // floor. at extreme dt the exponential underflows cleanly to 0 (fully drained).
    #[test]
    fn positivity_preserved_for_any_dt() {
        let cons = sample_cons();
        for &dt in &[0.1, 10.0, 1e6, 1e12] {
            let (drained, _) = drain_cell(&cons, 1.0, 1e-3, dt, 1.0, 0);
            assert!(drained.den >= 0.0 && drained.den.is_finite(),
                "density non-physical at dt={dt}: {}", drained.den);
            assert!(drained.den <= cons.den, "drain must not increase density");
        }
    }

    // chi = 0 (outside the mask) is an exact no-op: f = 1, nothing absorbed.
    #[test]
    fn zero_mask_is_exact_noop() {
        let cons = sample_cons();
        let (drained, delta) = drain_cell(&cons, 0.0, 1.0, 0.5, 1.0, 0);
        assert_eq!(drained.den, cons.den);
        assert_eq!(drained.nrg, cons.nrg);
        assert_eq!(delta.mass_delta, 0.0);
    }

    // c_s recovered from cons matches the analytic sqrt(gamma p / rho) (option 1: the drain
    // reads the just-updated cons, not the stale prim).
    #[test]
    fn sound_speed_from_cons_matches_analytic() {
        let (gamma, rho, p) = (5.0f64 / 3.0, 2.5f64, 1.4f64);
        let v = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = p / ((gamma - 1.0) * rho);
        let nrg = rho * (e_int + 0.5 * v.dot(&v));
        let mom_sq = v.scale(rho).dot(&v.scale(rho));
        let cs = sound_speed_from_cons(rho, mom_sq, nrg, gamma);
        let cs_analytic = (gamma * p / rho).sqrt();
        assert!((cs - cs_analytic).abs() < 1e-13, "c_s {cs} != analytic {cs_analytic}");
    }

    // the mask limits: fully inside -> 1, fully outside -> 0, on the surface -> 1/2.
    #[test]
    fn mask_ramps_from_one_to_zero() {
        let (r_mask, w) = (6.0f64, 1.0f64);
        assert!(drain_mask(0.0, r_mask, w) > 1.0 - 1e-4, "deep interior should be ~1");
        assert!(drain_mask(20.0, r_mask, w) < 1e-4, "far exterior should be ~0");
        assert!((drain_mask(r_mask, r_mask, w) - 0.5).abs() < 1e-14, "surface should be 1/2");
    }

    // the EMERGENT-RATE mechanics the post-godunov pass performs: reduce the per-cell
    // drain deltas over a radial mask. asserts (1) only masked cells contribute, (2)
    // gas+body mass conservation is exact across the whole reduction, (3) Mdot is a
    // FUNCTIONAL of the flow (doubling the density field doubles Mdot) -- never a
    // target fed in.
    #[test]
    fn emergent_rate_reduces_and_conserves_over_a_mask() {
        let (r_mask, w, dx, c_s, c_drain, dt) = (6.0, 1.0, 1.0, 1.0, 1.0, 0.2);
        let vol = dx * dx; // 2D-ish cell measure for the test
        // a radial density field rho(r) = 1 + 1/(1+r) over cells at r = 0, 2, ..., 18.
        let field = |scale: f64| -> (f64, f64, f64, f64) {
            let mut body = BodyDelta::<f64, 3>::new(0);
            let mut gas_lost = 0.0;
            let (mut inner_removed, mut outer_removed) = (0.0, 0.0);
            for i in 0..10 {
                let dist = 2.0 * i as f64;
                let rho = scale * (1.0 + 1.0 / (1.0 + dist));
                let cons = ConsG::<f64, 3, Adiabatic> {
                    den: rho, mom: Tensor::new([0.1 * rho, 0.0, 0.0]), nrg: 2.0 * rho,
                };
                let (drained, delta) =
                    drain_body_cell(&cons, dist, r_mask, w, dx, c_s, c_drain, dt, vol, 0);
                let removed = (cons.den - drained.den) * vol;
                gas_lost += removed;
                if i == 0 { inner_removed = removed; }
                if i == 9 { outer_removed = removed; }
                body += delta;
            }
            (body.mass_delta, gas_lost, inner_removed, outer_removed)
        };
        let (mdot_mass, gas_lost, inner, outer) = field(1.0);
        // (2) exact conservation across the reduction: body gained exactly what the gas lost.
        assert!((mdot_mass - gas_lost).abs() < 1e-13, "reduction non-conservative");
        // (1) the mask LOCALIZES the drain: the deep-interior cell (chi~1) drains O(1e8)x more
        // mass than a cell at ~3 r_mask (chi~0) -- the far field is untouched.
        assert!(outer / inner < 1e-6, "drain not localized: outer/inner = {}", outer / inner);
        // Mdot = absorbed mass / dt is a positive functional of the flow.
        assert!(mdot_mass / dt > 0.0);
        // (3) emergent, not targeted: doubling the density field doubles the absorbed mass.
        let (mdot_mass_2x, ..) = field(2.0);
        assert!((mdot_mass_2x - 2.0 * mdot_mass).abs() < 1e-13, "rate is not a linear flow-functional");
    }
}
