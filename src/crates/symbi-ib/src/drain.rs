// =============================================================================
// drain.rs
//
// the well-posed uniform-scaling volumetric drain:
// the exact-exponential relaxation `U -> U exp(-chi dt / tau)`, applied
// post-hydro on the masked cells. scaling every conserved component by one
// common factor `f` leaves each intensive primitive (velocity, specific internal
// energy, sound speed, temperature) pointwise invariant, so the drain is
// positivity-preserving at any dt (it carries its own stability, free of a CFL
// tax), stays silent in the acoustic and entropy waves, and preserves the
// characteristic decomposition -- outgoing characteristics pass cleanly through
// `r_mask` when it sits inside the sonic surface. the accretion rate is
// emergent: the cell-integrated `U_old - U_new` reduced over the mask is a
// functional of the solved flow, read out of the solution.
//
// this is the well-posed replacement for the mass-only KMK04 sink
// (`effects::accretion_source`): a mass-only sink removes rho at fixed momentum,
// so it shifts the velocity as it drains, re-couples the primitive channels,
// injects a back-pressure artifact, and (in the property algebra) breaks the
// exact three-exponential decoupling. uniform scaling keeps all four.
//
// usage:
//   let chi = drain_mask(dist, r_mask, w);
//   let tau = drain_timescale(dx, c_s, c_drain);
//   let (drained, delta) = drain_cell(&cons, chi, tau, dt, volume, body_idx);
// =============================================================================

use symbi_carrier::Scalar;
use symbi_hydro::energy::{EnergyModel, EnergySlot};
use symbi_hydro::state::ConsG;

use crate::body_delta::BodyDelta;

/// the mollified spherical mask value at coordinate distance `dist` from the
/// mask center: `chi = 0.5 (1 - tanh((dist - r_mask) / w))`. `chi -> 1` well
/// inside `r_mask`, `-> 0` well outside, with a tanh ramp of width `w` across
/// the surface. the mollification keeps the drain isotropic on a cartesian grid,
/// smoothing the staircase and softening the mask edge so it transmits rather
/// than reflects.
#[inline]
pub fn drain_mask<S: Scalar>(dist: S, r_mask: S, w: S) -> S {
    S::HALF * (S::ONE - ((dist - r_mask) / w).tanh())
}

/// the drain timescale `tau = c_drain * dx / c_s` (local sound speed). drains
/// fast enough to hold the mask evacuated against the inflow; in the
/// well-posed regime (`r_mask` inside the sonic surface) the emergent rate is
/// insensitive to `c_drain` once it is small enough -- the sonic surface
/// sets the emergent rate. `c_drain` is a convergence-study dial, swept for
/// plateau in the emergent rate.
#[inline]
pub fn drain_timescale<S: Scalar>(dx: S, c_s: S, c_drain: S) -> S {
    c_drain * dx / c_s
}

/// the spherical accretor's inverse drain timescale: no slower than either
/// the local signal crossing of one cell or free fall through the mask.
#[inline]
pub fn spherical_drain_rate<S: Scalar>(sound_rate: S, mass: S, r_acc: S) -> S {
    sound_rate.max((mass / (r_acc * r_acc * r_acc)).sqrt())
}

/// the instantaneous local drain rate `lambda_rho` for a slip-enabled cell: the fast-magnetosonic
/// cell-crossing rate `sqrt(cs^2 + |B|^2/rho) * inv_cd_dx` lifted by the free-fall rate
/// `sqrt(GM/r_acc^3)`, on the cell's own density and field. `inv_cd_dx = 1/(c_drain dx)`. the material
/// drain and the magnetic-slip coefficient evaluate this one rate law at a cell state: the drain
/// integrates it over its substep, and the slip coefficient `a_B` freezes its reciprocal
/// `tau_rho = 1/lambda_rho` at the magnetic predictor state -- one clock, read two ways.
#[inline]
pub fn local_drain_rate<S: Scalar>(cs: S, b_sq: S, den: S, inv_cd_dx: S, mass: S, r_acc: S) -> S {
    let sound_rate = (cs * cs + b_sq / den).sqrt() * inv_cd_dx;
    spherical_drain_rate(sound_rate, mass, r_acc)
}

/// the second-order material-drain factor over a masked step, exact across the acoustic/free-fall
/// branch crossing. the drain rate `lambda(rho) = max(lambda_ff, sqrt(cs^2 + b2/rho) inv_cd_dx)` has a
/// constant free-fall arm and an acoustic arm that stiffens as the density drains, so a cell entering
/// on the free-fall arm can cross to the acoustic arm partway through the step. the mask `chi` rescales
/// time uniformly, so a masked step of length `h` is the unmasked flow over `T = chi h`: drain the
/// free-fall arm exactly up to the crossing time, then midpoint-integrate the remaining acoustic
/// interval. the free-fall arm is linear in log-density (exact) and the acoustic arm is second-order
/// (midpoint), so the composite stays second-order accurate through the crossing. the returned factor
/// `f` in (0, 1] scales the conserved gas vector; `B` is untouched.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn event_split_drain_factor<S: Scalar>(
    rho: S,
    chi: S,
    h: S,
    cs: S,
    b2: S,
    inv_cd_dx: S,
    lambda_ff: S,
) -> S {
    let eps = S::from_f64(1e-300);
    // the mask rescales time: the masked step of length h is chi h of the unmasked drain flow.
    let t_total = chi * h;
    // the crossing satisfies sqrt(cs^2 + c_a2) inv_cd_dx = lambda_ff, i.e. c_a2 = (lambda_ff/inv_cd_dx)^2
    // - cs^2. free-fall lifts c_a2 = |B|^2/rho by exp(lambda_ff t); it meets the crossing at t_c. an arg
    // at or below one (cell already acoustic, or no free-fall arm) clamps t_c to zero.
    let lam_over = lambda_ff / inv_cd_dx;
    let c_a2_cross = lam_over * lam_over - cs * cs;
    let c_a2_0 = b2 / rho.max(eps);
    let arg = (c_a2_cross / c_a2_0.max(eps)).max(eps);
    let t_c = arg.ln() / lambda_ff.max(eps);
    let t_ff = t_c.max(S::ZERO).min(t_total);
    let f_ff = (S::ZERO - lambda_ff * t_ff).exp();
    let rho1 = rho * f_ff;
    let t_ac = t_total - t_ff;
    // the acoustic segment: midpoint on the true (max) rate, which past the crossing is the acoustic
    // arm. cs is invariant under the uniform drain, so only the Alfven term shifts between the predictor
    // and the corrector.
    let rate1 = ((cs * cs + b2 / rho1).sqrt() * inv_cd_dx).max(lambda_ff);
    let rho_star = rho1 * (S::ZERO - rate1 * t_ac * S::HALF).exp();
    let rate_star = ((cs * cs + b2 / rho_star).sqrt() * inv_cd_dx).max(lambda_ff);
    let f_ac = (S::ZERO - rate_star * t_ac).exp();
    f_ff * f_ac
}

/// the exact-exponential uniform-scaling drain on one masked cell's conserved
/// vector. returns the drained state `U exp(-chi dt/tau)` and the cell's exact
/// contribution to the body -- the cell-integrated `U_old - U_new` (absorbed
/// mass and drag force), which makes gas+body conservation exact to machine
/// precision. one scalar factor multiplies every conserved component, the energy
/// slot included; that uniformity is the invariant the scheme rests on, and it
/// leaves the intensive primitive state untouched.
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
    let drained = *cons * f; // uniform scale of den, mom, and the energy slot
    let absorbed = *cons - drained; // = cons * (1 - f): the body's exact gain
    let mut delta = BodyDelta::new(idx);
    // cell-integrated absorbed mass (dM), drag force (dP/dt), and total energy (dE). the
    // energy slot is zero for the isothermal regime, which carries no energy channel. together
    // these make gas+body conservation of mass, momentum, and energy exact to machine precision.
    delta.mass_delta = absorbed.den() * volume;
    delta.force_delta = absorbed.mom().scale(volume / dt);
    delta.energy_delta = absorbed.nrg().value() * volume;
    (drained, delta)
}

/// the adiabatic sound speed recovered directly from the updated conserved state. the
/// drain runs between godunov and c2p (ordering: hydro -> drain -> c2p+floors), so the
/// stored primitive is stale and the drain timescale's `c_s` comes from the
/// just-updated `cons`. `e_int = (nrg - 0.5 |mom|^2 / den) / den`, then
/// `c_s = sqrt(gamma (gamma - 1) e_int)`. a one-line inversion, since the sound speed
/// depends on the internal energy alone. regime-local; the caller supplies gamma. the
/// isothermal regime instead carries a fixed `c_s = c_iso`.
#[inline]
pub fn sound_speed_from_cons<S: Scalar>(den: S, mom_sq: S, nrg: S, gamma: S) -> S {
    let e_int = (nrg - S::HALF * mom_sq / den) / den;
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
mod event_split_drain_tests {
    use super::event_split_drain_factor;

    // the drain rate lambda(rho) = max( sqrt(cs^2 + b2/rho) inv_cd_dx, lambda_ff ): the acoustic arm
    // stiffens as rho drains (the Alfven term |B|^2/rho rises), the free-fall arm is constant, and the
    // nonsmooth max crosses between them.
    fn rate_of(cs: f64, b2: f64, inv_cd_dx: f64, lambda_ff: f64) -> impl Fn(f64) -> f64 {
        move |rho: f64| ((cs * cs + b2 / rho).sqrt() * inv_cd_dx).max(lambda_ff)
    }

    // the reference: y = log rho, dy/dt = -lambda(e^y), integrated with RK4 over `substeps` -- a
    // high-accuracy scalar solution of the exact drain flow (unmasked, chi = 1).
    fn reference(rho0: f64, t: f64, rate: &impl Fn(f64) -> f64, substeps: usize) -> f64 {
        let h = t / substeps as f64;
        let mut y = rho0.ln();
        let f = |y: f64| -rate(y.exp());
        for _ in 0..substeps {
            let k1 = f(y);
            let k2 = f(y + 0.5 * h * k1);
            let k3 = f(y + 0.5 * h * k2);
            let k4 = f(y + h * k3);
            y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        y.exp()
    }

    // the masked reference: dy/dt = -chi lambda(e^y), RK4 over `substeps`.
    #[allow(clippy::too_many_arguments)]
    fn reference_chi(rho0: f64, t: f64, rate: &impl Fn(f64) -> f64, chi: f64, substeps: usize) -> f64 {
        let h = t / substeps as f64;
        let mut y = rho0.ln();
        let f = |y: f64| -chi * rate(y.exp());
        for _ in 0..substeps {
            let k1 = f(y);
            let k2 = f(y + 0.5 * h * k1);
            let k3 = f(y + 0.5 * h * k2);
            let k4 = f(y + h * k3);
            y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        y.exp()
    }

    // one event-split drain step at mask `chi` and a fixed-time evolution of `steps` of them.
    #[allow(clippy::too_many_arguments)]
    fn step_chi(rho: f64, h: f64, chi: f64, cs: f64, b2: f64, inv_cd_dx: f64, lambda_ff: f64) -> f64 {
        rho * event_split_drain_factor(rho, chi, h, cs, b2, inv_cd_dx, lambda_ff)
    }
    #[allow(clippy::too_many_arguments)]
    fn evolve_chi(rho0: f64, t: f64, steps: usize, chi: f64, cs: f64, b2: f64, inv_cd_dx: f64, lambda_ff: f64) -> f64 {
        let h = t / steps as f64;
        let mut rho = rho0;
        for _ in 0..steps {
            rho = step_chi(rho, h, chi, cs, b2, inv_cd_dx, lambda_ff);
        }
        rho
    }

    // a fixed-time evolution at mask chi = 1.
    fn evolve(rho0: f64, t: f64, steps: usize, cs: f64, b2: f64, inv_cd_dx: f64, lambda_ff: f64) -> f64 {
        evolve_chi(rho0, t, steps, 1.0, cs, b2, inv_cd_dx, lambda_ff)
    }

    // second order on the density-dependent acoustic arm: the fixed-time error quarters when the step
    // halves.
    #[test]
    fn drain_is_second_order_on_the_acoustic_branch() {
        let (cs, b2, inv_cd_dx, ff) = (1.0, 4.0, 20.0, 0.0); // ff = 0: acoustic arm always active
        let (rho0, t) = (1.0, 0.02);
        let rate = rate_of(cs, b2, inv_cd_dx, ff);
        let refn = reference(rho0, t, &rate, 8192);
        let e_coarse = (evolve(rho0, t, 16, cs, b2, inv_cd_dx, ff) - refn).abs();
        let e_fine = (evolve(rho0, t, 32, cs, b2, inv_cd_dx, ff) - refn).abs();
        let ratio = e_coarse / e_fine.max(1e-300);
        println!("\nacoustic: E(h)={e_coarse:.3e} E(h/2)={e_fine:.3e} order={:.2}\n", ratio.log2());
        assert!(e_coarse > 1e-10, "vacuous ({e_coarse})");
        assert!(ratio > 3.6, "acoustic branch not second order: ratio {ratio:.2}");
    }

    // the free-fall arm is a constant rate, so the exponential drain is the exact flow and the scheme
    // reproduces it to roundoff (a constant-coefficient linear ODE has no scheme error).
    #[test]
    fn drain_is_exact_on_the_free_fall_branch() {
        let (cs, b2, inv_cd_dx, ff) = (0.0, 0.0, 1.0, 30.0); // acoustic = 0, so lambda = ff constant
        let rate = rate_of(cs, b2, inv_cd_dx, ff);
        let refn = reference(1.0, 0.03, &rate, 8192);
        let got = evolve(1.0, 0.03, 4, cs, b2, inv_cd_dx, ff);
        assert!((got - refn).abs() < 1e-11 * refn, "constant-rate drain not exact: {got:.15e} vs {refn:.15e}");
    }

    // the branch crossing is resolved by exact event splitting: a step that carries a cell from the
    // free-fall arm across to the acoustic arm keeps second order, because the free-fall segment is
    // integrated exactly to the crossing time and only the acoustic remainder carries the midpoint
    // error. this is the case that dropped to first order without the split.
    #[test]
    fn event_splitting_restores_second_order_across_the_crossover() {
        let (cs, b2, inv_cd_dx) = (1.0, 4.0, 5.0);
        let ff = (1.0 + 4.0 / 0.5f64).sqrt() * inv_cd_dx; // acoustic == ff at rho = 0.5
        let (rho0, t) = (1.0, 0.08); // the solution passes through rho = 0.5 within the interval
        let rate = rate_of(cs, b2, inv_cd_dx, ff);
        let refn = reference(rho0, t, &rate, 16384);
        let e_coarse = (evolve(rho0, t, 16, cs, b2, inv_cd_dx, ff) - refn).abs();
        let e_fine = (evolve(rho0, t, 32, cs, b2, inv_cd_dx, ff) - refn).abs();
        let order = (e_coarse / e_fine.max(1e-300)).log2();
        println!("\ncrossover order = {order:.2} (event-split)\n");
        assert!(e_coarse > 1e-10, "vacuous crossover test ({e_coarse})");
        assert!(order > 1.8, "event splitting did not restore second order across the crossing: {order:.2}");
    }

    // event splitting holds second order for crossings placed anywhere within the step and for
    // fractional masks: the crossing is located at an off-boundary fraction of the interval (never
    // aligned with a step edge at N = 16 or 32), the mask chi rescales the drain clock, and each case's
    // global error still quarters when the step halves.
    #[test]
    fn event_splitting_is_second_order_for_arbitrary_crossings_and_masks() {
        let (cs, b2, inv_cd_dx) = (1.0, 4.0, 5.0);
        // (label, crossing density rho_c, mask chi, off-boundary fraction of the interval at the cross)
        let cases: [(&str, f64, f64, f64); 4] = [
            ("early rho_c=0.7 chi=1.0", 0.7, 1.0, 0.31),
            ("mid   rho_c=0.5 chi=0.7", 0.5, 0.7, 0.53),
            ("late  rho_c=0.3 chi=0.2", 0.3, 0.2, 0.68),
            ("late  rho_c=0.4 chi=0.5", 0.4, 0.5, 0.79),
        ];
        for (label, rho_c, chi, frac) in cases {
            let ff = (cs * cs + b2 / rho_c).sqrt() * inv_cd_dx; // acoustic == ff at rho = rho_c
            let rho0 = 1.0;
            // the masked free-fall arm reaches rho_c at t_cross; size the interval so that crossing
            // lands at `frac` of it -- an off-boundary fraction for 16 and 32 uniform steps.
            let t_cross = (rho0 / rho_c).ln() / (chi * ff);
            let t = t_cross / frac;
            let rate = rate_of(cs, b2, inv_cd_dx, ff);
            let refn = reference_chi(rho0, t, &rate, chi, 32768);
            let e16 = (evolve_chi(rho0, t, 16, chi, cs, b2, inv_cd_dx, ff) - refn).abs();
            let e32 = (evolve_chi(rho0, t, 32, chi, cs, b2, inv_cd_dx, ff) - refn).abs();
            let order = (e16 / e32.max(1e-300)).log2();
            println!("{label}: E(h)={e16:.3e} E(h/2)={e32:.3e} order={order:.2}");
            assert!(e16 > 1e-10, "{label}: vacuous ({e16})");
            assert!(order > 1.8, "{label}: not second order across the crossing: order {order:.2}");
        }
    }

    // a step that drains a free-fall cell but stops short of the crossing stays exactly on the
    // free-fall arm: a constant rate, so the exponential drain is the exact flow to roundoff.
    #[test]
    fn a_step_short_of_the_crossing_stays_exactly_on_free_fall() {
        let (cs, b2, inv_cd_dx) = (1.0f64, 4.0, 5.0);
        let rho_c = 0.3f64;
        let ff = (cs * cs + b2 / rho_c).sqrt() * inv_cd_dx; // crossing far below the run
        let (chi, rho0) = (0.7, 1.0);
        // drain only to rho ~ 0.6, never reaching rho_c = 0.3: pure free-fall.
        let t = (rho0 / 0.6f64).ln() / (chi * ff);
        let rate = rate_of(cs, b2, inv_cd_dx, ff);
        let refn = reference_chi(rho0, t, &rate, chi, 32768);
        let got = evolve_chi(rho0, t, 4, chi, cs, b2, inv_cd_dx, ff);
        assert!(
            (got - refn).abs() < 1e-11 * refn,
            "a step short of the crossing is not exact free-fall: {got:.15e} vs {refn:.15e}"
        );
    }

    // the factor stays in [0, 1] and finite for arbitrarily stiff finite steps and for a vanishing
    // mask -- density scales down, never negative and never NaN; a step stiff enough to evacuate the
    // cell underflows the factor to zero (the full-drain limit), and chi = 0 is an exact no-op (f = 1).
    #[test]
    fn drain_factor_stays_in_the_unit_interval() {
        let (cs, b2, inv_cd_dx, ff) = (1.0f64, 4.0, 20.0, 15.0);
        for &chi in &[0.0f64, 0.3, 1.0] {
            for &h in &[1e-6f64, 1.0, 1e3, 1e9] {
                let f = event_split_drain_factor(1.0, chi, h, cs, b2, inv_cd_dx, ff);
                assert!(f.is_finite() && (0.0..=1.0).contains(&f), "factor left [0,1] at chi={chi} h={h}: f={f}");
            }
        }
        // chi = 0 is a bit-exact no-op.
        let f0 = event_split_drain_factor(1.0f64, 0.0, 3.0, cs, b2, inv_cd_dx, ff);
        assert_eq!(f0, 1.0, "zero mask must be an exact no-op, got {f0}");
    }

    // the drain and the slip operator read one instantaneous rate for a cell state: both consume
    // `local_drain_rate`, so the branch choice (acoustic vs free-fall) and the numeric value agree on
    // heterogeneous fields. this pins the coupling clock -- D and M share tau_rho.
    #[test]
    fn drain_and_slip_share_one_local_rate() {
        use super::{local_drain_rate, spherical_drain_rate};
        let inv_cd_dx = 5.0f64;
        let (mass, r_acc) = (2.0f64, 0.3f64);
        let lambda_ff = (mass / (r_acc * r_acc * r_acc)).sqrt();
        // acoustic-dominant, free-fall-dominant, and near-crossover cells.
        let cells: [(&str, f64, f64, f64); 3] = [
            ("acoustic", 1.0, 9.0, 0.4),   // sqrt(1 + 9/0.4)*5 = big > ff
            ("free-fall", 1.5, 0.01, 5.0), // tiny field: acoustic ~ cs*inv_cd_dx < ff
            ("crossover", 1.0, 4.0, 0.5),  // acoustic ~ ff
        ];
        for (name, cs, b2, rho) in cells {
            let rate = local_drain_rate(cs, b2, rho, inv_cd_dx, mass, r_acc);
            // the slip path builds tau_rho = 1/lambda_rho from the same call; the drain path feeds the
            // same rate into the free-fall max. both must equal this one value and pick one branch.
            let acoustic = (cs * cs + b2 / rho).sqrt() * inv_cd_dx;
            let expect = spherical_drain_rate(acoustic, mass, r_acc);
            assert_eq!(rate, expect, "{name}: local_drain_rate disagrees with the shared rate law");
            assert_eq!(rate, acoustic.max(lambda_ff), "{name}: branch choice differs from max(acoustic, ff)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Tensor;
    use symbi_hydro::energy::Adiabatic;
    use symbi_hydro::quantity::{Density, EnergyDensity};

    type Cons3 = ConsG<f64, 3, Adiabatic>;

    fn sample_cons() -> Cons3 {
        // a moving, pressurized cell: den, mom = den*v, nrg = total energy.
        let den = 2.5;
        let v = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = 1.7; // specific internal energy
        let nrg = den * (e_int + 0.5 * v.dot(&v));
        Cons3::adiabatic(Density(den), v.scale(den), EnergyDensity(nrg))
    }

    // the invariant the scheme rests on: uniform scaling leaves the intensive
    // primitive state pointwise invariant, where a mass-only sink shifts the velocity.
    #[test]
    fn uniform_scaling_leaves_intensive_primitives_invariant() {
        let cons = sample_cons();
        let (drained, _) = drain_cell(&cons, 0.8, 3.0, 0.4, 1.0, 0);
        // velocity v = mom / den: unchanged.
        for k in 0..3 {
            let v0 = cons.mom()[k] / cons.den();
            let v1 = drained.mom()[k] / drained.den();
            assert!(
                (v0 - v1).abs() < 1e-14,
                "velocity {k} changed: {v0} -> {v1}"
            );
        }
        // specific internal energy e_int = nrg/den - 0.5|v|^2: unchanged.
        let vsq0 = (0..3)
            .map(|k| (cons.mom()[k] / cons.den()).powi(2))
            .sum::<f64>();
        let vsq1 = (0..3)
            .map(|k| (drained.mom()[k] / drained.den()).powi(2))
            .sum::<f64>();
        let e0 = cons.nrg() / cons.den() - 0.5 * vsq0;
        let e1 = drained.nrg() / drained.den() - 0.5 * vsq1;
        assert!(
            (e0 - e1).abs() < 1e-14,
            "specific internal energy changed: {e0} -> {e1}"
        );
        let gamma = 5.0 / 3.0;
        let cs0 = sound_speed_from_cons(cons.den(), cons.mom().dot(&cons.mom()), cons.nrg(), gamma);
        let cs1 = sound_speed_from_cons(
            drained.den(),
            drained.mom().dot(&drained.mom()),
            drained.nrg(),
            gamma,
        );
        assert!(
            (cs0 - cs1).abs() < 1e-14,
            "sound speed changed: {cs0} -> {cs1}"
        );
    }

    #[test]
    fn spherical_rate_selects_the_faster_physical_clock() {
        let mass = 2.0_f64;
        let radius = 0.5_f64;
        let free_fall = (mass / (radius * radius * radius)).sqrt();
        assert_eq!(
            spherical_drain_rate(0.5 * free_fall, mass, radius),
            free_fall
        );
        assert_eq!(
            spherical_drain_rate(2.0 * free_fall, mass, radius),
            2.0 * free_fall
        );
    }

    #[test]
    fn bhl_default_resolves_to_the_historical_free_fall_floor() {
        let dx = 1.0_f64 / 64.0;
        let r_acc = 3.0 * dx;
        let sound_rate = 1.0 / dx;
        let free_fall = (1.0 / (r_acc * r_acc * r_acc)).sqrt();
        assert!(free_fall > sound_rate);
        assert_eq!(spherical_drain_rate(sound_rate, 1.0, r_acc), free_fall);
    }

    // exact gas+body conservation: the body's gain is exactly what the gas lost.
    #[test]
    fn conservation_is_exact() {
        let cons = sample_cons();
        let vol = 0.05;
        let (drained, delta) = drain_cell(&cons, 1.0, 2.0, 0.5, vol, 0);
        // mass: gas lost (den_old - den_new)*V equals the body's mass_delta.
        let gas_mass_lost = (cons.den() - drained.den()) * vol;
        assert!((gas_mass_lost - delta.mass_delta).abs() < 1e-15);
        // momentum: gas lost (mom_old - mom_new)*V equals the body's absorbed momentum (= F*dt).
        for k in 0..3 {
            let gas_mom_lost = (cons.mom()[k] - drained.mom()[k]) * vol;
            assert!((gas_mom_lost - delta.force_delta[k] * 0.5).abs() < 1e-15);
        }
        // energy: gas lost (nrg_old - nrg_new)*V equals the body's absorbed energy_delta.
        let gas_nrg_lost = (cons.nrg() - drained.nrg()) * vol;
        assert!((gas_nrg_lost - delta.energy_delta).abs() < 1e-15);
    }

    // positivity at any dt, which is why the update is an exponential: f = exp(-x) in
    // [0, 1] for x >= 0, so drained density stays in [0, den] and stays finite. an
    // explicit stiff source would overshoot to negative density and need a floor. at
    // extreme dt the exponential underflows cleanly to 0 (fully drained).
    #[test]
    fn positivity_preserved_for_any_dt() {
        let cons = sample_cons();
        for &dt in &[0.1, 10.0, 1e6, 1e12] {
            let (drained, _) = drain_cell(&cons, 1.0, 1e-3, dt, 1.0, 0);
            assert!(
                drained.den() >= 0.0 && drained.den().is_finite(),
                "density non-physical at dt={dt}: {}",
                drained.den()
            );
            assert!(
                drained.den() <= cons.den(),
                "drain must not increase density"
            );
        }
    }

    // chi = 0 (outside the mask) is an exact no-op: f = 1, nothing absorbed.
    #[test]
    fn zero_mask_is_exact_noop() {
        let cons = sample_cons();
        let (drained, delta) = drain_cell(&cons, 0.0, 1.0, 0.5, 1.0, 0);
        assert_eq!(drained.den(), cons.den());
        assert_eq!(drained.nrg(), cons.nrg());
        assert_eq!(delta.mass_delta, 0.0);
    }

    // c_s recovered from cons matches the analytic sqrt(gamma p / rho) (option 1: the drain
    // reads the just-updated cons).
    #[test]
    fn sound_speed_from_cons_matches_analytic() {
        let (gamma, rho, p) = (5.0f64 / 3.0, 2.5f64, 1.4f64);
        let v = Tensor::new([0.3, -0.2, 0.1]);
        let e_int = p / ((gamma - 1.0) * rho);
        let nrg = rho * (e_int + 0.5 * v.dot(&v));
        let mom_sq = v.scale(rho).dot(&v.scale(rho));
        let cs = sound_speed_from_cons(rho, mom_sq, nrg, gamma);
        let cs_analytic = (gamma * p / rho).sqrt();
        assert!(
            (cs - cs_analytic).abs() < 1e-13,
            "c_s {cs} != analytic {cs_analytic}"
        );
    }

    // the mask limits: fully inside -> 1, fully outside -> 0, on the surface -> 1/2.
    #[test]
    fn mask_ramps_from_one_to_zero() {
        let (r_mask, w) = (6.0f64, 1.0f64);
        assert!(
            drain_mask(0.0, r_mask, w) > 1.0 - 1e-4,
            "deep interior should be ~1"
        );
        assert!(
            drain_mask(20.0, r_mask, w) < 1e-4,
            "far exterior should be ~0"
        );
        assert!(
            (drain_mask(r_mask, r_mask, w) - 0.5).abs() < 1e-14,
            "surface should be 1/2"
        );
    }

    // the emergent-rate mechanics the post-godunov pass performs: reduce the per-cell
    // drain deltas over a radial mask. asserts (1) the masked cells alone contribute, (2)
    // gas+body mass conservation is exact across the whole reduction, (3) Mdot is a
    // functional of the flow -- doubling the density field doubles Mdot, so the rate
    // comes out of the solution.
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
                let cons = ConsG::<f64, 3, Adiabatic>::adiabatic(
                    Density(rho),
                    Tensor::new([0.1 * rho, 0.0, 0.0]),
                    EnergyDensity(2.0 * rho),
                );
                let (drained, delta) =
                    drain_body_cell(&cons, dist, r_mask, w, dx, c_s, c_drain, dt, vol, 0);
                let removed = (cons.den() - drained.den()) * vol;
                gas_lost += removed;
                if i == 0 {
                    inner_removed = removed;
                }
                if i == 9 {
                    outer_removed = removed;
                }
                body += delta;
            }
            (body.mass_delta, gas_lost, inner_removed, outer_removed)
        };
        let (mdot_mass, gas_lost, inner, outer) = field(1.0);
        // (2) exact conservation across the reduction: body gained exactly what the gas lost.
        assert!(
            (mdot_mass - gas_lost).abs() < 1e-13,
            "reduction non-conservative"
        );
        // (1) the mask localizes the drain: the deep-interior cell (chi~1) drains O(1e8)x more
        // mass than a cell at ~3 r_mask (chi~0), leaving the far field at its incoming state.
        assert!(
            outer / inner < 1e-6,
            "drain not localized: outer/inner = {}",
            outer / inner
        );
        // Mdot = absorbed mass / dt is a positive functional of the flow.
        assert!(mdot_mass / dt > 0.0);
        // (3) emergent: doubling the density field doubles the absorbed mass.
        let (mdot_mass_2x, ..) = field(2.0);
        assert!(
            (mdot_mass_2x - 2.0 * mdot_mass).abs() < 1e-13,
            "rate is not a linear flow-functional"
        );
    }
}
