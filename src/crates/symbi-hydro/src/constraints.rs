// =============================================================================
// constraints.rs
//
// the state-constraint algebra: every floor and ceiling a run may impose, expressed as one object.
//
// a constraint is a scalar inequality on the conserved state, `c(U) >= 0`. the family in force
// defines the acceptable set `C = intersect_k { c_k >= 0 }`, and every correction the scheme applies
// targets `G_safe = G intersect C` — the physically admissible set intersected with the run's declared
// constraints. the admissible set itself is the always-present member of the family, so one
// projection path covers G and the floors together and they stay in sync by construction.
//
// this collapses the usual zoo into one algebra:
//
//   density floor              c = D - D_min
//   temperature/pressure floor c = p(U) - f rho(U)
//   magnetization ceiling      c = sigma_max rho - |B|^2
//   admissibility (Wu & Tang)  c = (D, q, psi)
//
// the axioms, each carrying a proof obligation:
//
//  A1 concavity.   every `c_k` is concave in U, so `{c_k >= 0}` is convex and `G intersect C` inherits the
//                  convexity of G (Wu & Tang, arXiv:1709.05838, theorem 2.2). this is what makes the
//                  segment projection well-posed: a concave `c_k` restricted to the segment is
//                  concave in the blend parameter, so `{t : c_k >= 0}` is a single interval and the
//                  boundary is crossed once. `concavity_violation` checks it numerically, and
//                  membership in the family is conditional on passing, whatever the constraint
//                  models. a Lorentz-factor ceiling is the standard example of one that fails.
//  A2 projection.  the corrective operator is a projection: identity on `G intersect C` (an acceptable state
//                  passes through bit-for-bit), idempotent, and minimal along the correction segment.
//  A3 simultaneous. the family is applied as a single projection onto the intersection. a sequence
//                  of clamps would be order-dependent and would lose idempotence — the second clamp
//                  can push the state back out of the first's set.
//  A4 single site. the projection is applied at exactly one point in the operator sequence; c2p and
//                  the Riemann solver pass the state through untouched.
//  A5 accounting.  every firing emits `dU = Pi(U) - U`, booked per constraint. non-conservation
//                  surfaces as a reported budget with a constraint's name attached to it.
//  A6 vanishing.   a constraint standing in for truncation error carries a scale that vanishes with
//                  the mesh, so the injected quantity vanishes under refinement. a constraint
//                  standing in for physics (vacuum modeled as low-density gas) holds a finite scale
//                  under refinement and declares that by `is_model_term`, so the budget separates
//                  the two. those are different claims in a methods section and each one earns its
//                  own aggregate number.
//
// usage:
//  let family = [&WuTangAdmissibility { .. }, &TemperatureFloor { f_min }];
//  let (theta, binding) = joint_projection_theta(&family, &anchor, &candidate, iters);
// =============================================================================

use symbi_algebra::{Matrix, Tensor};
use symbi_ir::algebra::Scalar;

/// the regime-generic conserved state a constraint reads. absent slots are `None`: isothermal
/// regimes leave the energy slot empty, unmagnetized regimes the field slot. a constraint whose slot
/// is empty in the current regime returns a trivially satisfied residual, so one family can be
/// declared for every regime and each member stays inert where its slot is absent.
pub struct ConstraintState<'a, S: Scalar> {
    /// conserved rest-mass density `D`.
    pub den: S,
    /// conserved momentum `S_i` (covariant).
    pub mom: Tensor<S, 3>,
    /// the energy slot: `tau`/`ehat` for energy regimes, `None` for isothermal.
    pub nrg: Option<S>,
    /// cell-centered magnetic field `B^i`, `None` for unmagnetized regimes.
    pub mag: Option<Tensor<S, 3>>,
    /// the spatial metric and its inverse; identity on a flat cartesian chart.
    pub gm: &'a Matrix<S, 3>,
    pub gm_inv: &'a Matrix<S, 3>,
}

impl<S: Scalar> ConstraintState<'_, S> {
    /// `gamma^{ij} S_i S_j` — the momentum norm on the inverse metric (the valence `S_i` demands).
    pub fn mom_norm_sq(&self) -> S {
        let mut acc = S::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + self.gm_inv[(ii, jj)] * self.mom[ii] * self.mom[jj];
            }
        }
        acc
    }

    /// `gamma_{ij} B^i B^j` — the field norm on the covariant metric. zero when unmagnetized.
    pub fn mag_norm_sq(&self) -> S {
        match &self.mag {
            None => S::ZERO,
            Some(b) => {
                let mut acc = S::ZERO;
                for ii in 0..3 {
                    for jj in 0..3 {
                        acc = acc + self.gm[(ii, jj)] * b[ii] * b[jj];
                    }
                }
                acc
            }
        }
    }
}

/// one scalar constraint `c(U) >= 0` on the conserved state.
///
/// implementors are concave in `U` (axiom A1). the projection's uniqueness proof rests on that
/// concavity, and a single non-concave member silently breaks the whole family, its siblings
/// included. `concavity_violation` makes the property testable, and every implementor carries such
/// a test.
pub trait StateConstraint<S: Scalar> {
    /// the residual. `Some(c)` with `c >= 0` means satisfied; the projection drives `c` exactly to
    /// zero and stops there. `None` means the constraint is structurally inapplicable to this
    /// regime — an isothermal state carries an empty energy slot to floor, an unmagnetized one an
    /// empty field slot to cap.
    ///
    /// `None` is deliberately a signal distinct from "a residual that happens to be satisfied". a
    /// constraint that signalled inapplicability by returning a comfortable positive number would
    /// read exactly like one that ran and passed, so an inert member would silently inflate the
    /// count of constraints the run believes it is enforcing. the distinction is resolved once when
    /// the family is built and reused for every cell, so evaluation pays nothing for it.
    fn residual(&self, u: &ConstraintState<S>) -> Option<S>;

    /// the constraint's magnitude at mesh spacing `dx` (axiom A6).
    ///
    /// a numerical constraint stands in for truncation error and vanishes as `dx -> 0`, so that the
    /// injected quantity vanishes under refinement and the computation converges to the equations
    /// being claimed. a model term stands in for physics beyond the reach of the equations (vacuum)
    /// and holds a finite scale under refinement — it declares that by `is_model_term`. the question
    /// lives on the trait, so every new constraint answers it to compile.
    fn scale(&self, dx: f64) -> f64;

    /// whether this constraint encodes physics (a vacuum/atmosphere model that survives mesh
    /// refinement) or truncation error (a scale that vanishes with the mesh). the injection budget
    /// separates the two: mass added where the numerics lose a cancellation is a different claim
    /// from mass added because vacuum is modeled as thin gas.
    fn is_model_term(&self) -> bool {
        false
    }

    /// whether the residual is affine in the conserved state.
    ///
    /// an affine residual restricted to the correction segment is affine in the blend parameter, so
    /// its crossing is one exact division and one graph node, where the general case unrolls a
    /// bisection. that is the difference between a tractable and an enormous traced kernel once a
    /// family has several members. affinity is a strictly stronger claim than the concavity A1
    /// requires, so it is checked the same way: an affine residual's concavity violation is zero in
    /// both directions, where concavity alone only bounds it above by zero.
    fn is_affine(&self) -> bool {
        false
    }

    /// the ledger key for this constraint's injection budget.
    fn name(&self) -> &'static str;
}

/// the exact admissible set (Wu & Tang, arXiv:1709.05838, theorem 2.1) as a member of the family.
///
/// this constraint is always present, and its scale is identically zero: it marks the boundary of
/// physical representability, a property of the equations themselves, so it is mesh-independent and
/// sits outside the model-term budget. as an ordinary member of the family it lets the projection
/// reach `G intersect C` in a single operation.
pub struct WuTangAdmissibility<S: Scalar> {
    /// the relative margins, scaled by the cell's own state scale by the caller.
    pub eps_d: S,
    pub eps_q: S,
    pub eps_psi: S,
}

impl<S: Scalar> StateConstraint<S> for WuTangAdmissibility<S> {
    fn residual(&self, u: &ConstraintState<S>) -> Option<S> {
        let e = u.nrg.unwrap_or(S::ZERO);
        let b = u.mag.unwrap_or_else(Tensor::zeros);
        let (q, psi) =
            crate::admissible::rmhd_admissible_residuals(u.den, &u.mom, e, &b, u.gm_inv, u.gm);
        // the family's convention is one scalar per constraint, so the three conditions combine by
        // minimum — a min of concave functions is itself concave, so A1 survives the combination.
        let d_margin = u.den - self.eps_d;
        let q_margin = q - self.eps_q;
        let psi_margin = psi - self.eps_psi;
        Some(d_margin.min(q_margin).min(psi_margin))
    }

    /// a representability bound: the boundary of G is where a physical primitive ceases to exist at
    /// every resolution, so the scale is identically zero and refinement leaves it where it is.
    fn scale(&self, _dx: f64) -> f64 {
        0.0
    }

    fn name(&self) -> &'static str {
        "wu_tang_admissibility"
    }
}

/// a floor on temperature, `p >= f_min rho`, written in conserved variables.
///
/// this is the constraint for the cancellation failure: in a conservative scheme the gas pressure is
/// recovered as a residual of the total energy after the rest-mass, kinetic and magnetic parts are
/// removed, so once `p/E` falls to the scheme's own truncation error the recovered pressure is noise.
/// a dense, cold cell reaches that state at densities far above vacuum, where a density floor has
/// long since stopped acting.
///
/// in conserved variables the constraint is an energy floor: the eulerian energy must exceed the
/// rest-mass, kinetic and magnetic contributions by at least the floor's thermal share. that is
/// linear in `E` and in `D` at fixed momentum and field, hence concave.
pub struct TemperatureFloor<S: Scalar> {
    /// the minimum `p/rho`. numerical: it stands in for truncation error and vanishes with the
    /// mesh, so the caller supplies a mesh-dependent value.
    pub f_min: S,
}

impl<S: Scalar> StateConstraint<S> for TemperatureFloor<S> {
    fn residual(&self, u: &ConstraintState<S>) -> Option<S> {
        // an isothermal regime carries an empty energy slot, so the thermal floor reports
        // inapplicable and the family stays well-formed.
        let e = u.nrg?;
        // the thermal margin available in the energy slot, above the rest mass and the field. the
        // momentum norm is left to the admissibility residual, which keeps this expression linear
        // in the conserved slots and hence concave.
        let thermal = e - u.den - S::HALF * u.mag_norm_sq();
        Some(thermal - self.f_min * u.den)
    }

    /// second-order truncation in the energy update is what drives `p` below representability, so
    /// the floor tracks it at `dx^2`.
    fn scale(&self, dx: f64) -> f64 {
        dx * dx
    }

    fn is_affine(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "temperature_floor"
    }
}

/// a ceiling on magnetization, `|B|^2 / rho <= sigma_max`, i.e. `sigma_max D - |B|^2 >= 0`.
///
/// linear in `D` at the fixed field constrained transport owns, hence concave. this is the standard
/// guard for the magnetically dominated funnel, where the gas pressure is a vanishing fraction of the
/// magnetic energy and the recovery is worst-conditioned.
pub struct MagnetizationCeiling<S: Scalar> {
    pub sigma_max: S,
}

impl<S: Scalar> StateConstraint<S> for MagnetizationCeiling<S> {
    fn residual(&self, u: &ConstraintState<S>) -> Option<S> {
        u.mag.as_ref()?;
        Some(self.sigma_max * u.den - u.mag_norm_sq())
    }

    /// the recovery's conditioning at high sigma degrades with the same truncation error the
    /// temperature floor tracks, so the admissible ceiling rises as the mesh refines.
    fn scale(&self, dx: f64) -> f64 {
        dx * dx
    }

    fn is_affine(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "magnetization_ceiling"
    }
}

/// a floor on rest-mass density, `D >= D_min`.
///
/// linear, hence concave. this is a physical model term (axiom A6): it stands in for vacuum, which
/// lies beyond the reach of the equations — at `rho = 0` the conserved-to-primitive map degenerates
/// and the system loses strict hyperbolicity. its scale therefore survives mesh refinement, and the
/// mass it injects belongs in a separate line of the budget from the numerical constraints.
pub struct DensityFloor<S: Scalar> {
    /// evaluated by the caller at the cell position, so a radially scaled atmosphere is expressed
    /// here while the constraint stays chart-agnostic.
    pub den_min: S,
}

impl<S: Scalar> StateConstraint<S> for DensityFloor<S> {
    fn residual(&self, u: &ConstraintState<S>) -> Option<S> {
        Some(u.den - self.den_min)
    }

    /// mesh-independent by construction: this stands in for vacuum, which lies beyond the reach of
    /// the equations at every resolution — `rho = 0` stays ill-posed however fine the mesh.
    fn scale(&self, _dx: f64) -> f64 {
        1.0
    }

    fn is_model_term(&self) -> bool {
        true
    }

    fn is_affine(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "density_floor"
    }
}

/// the per-constraint blend threshold: the smallest `t` in [0, 1] for which
/// `c(anchor + t (cand - anchor))` is non-negative, by bisection.
///
/// `c` restricted to the segment is concave in `t` (A1) and non-negative at `t = 0` (the anchor is
/// acceptable), so the feasible set is an interval `[0, t_hi]` — in this orientation `t = 0` is the
/// anchor and `t = 1` the candidate, so the interval containing the anchor is the one that matters
/// and the threshold is its upper end. an acceptable candidate returns exactly 1 and therefore passes
/// through bit-for-bit (A2).
pub fn constraint_theta<'a, S, C, B>(constraint: &C, blend: &B, iters: usize) -> Option<S>
where
    S: Scalar + 'a,
    C: StateConstraint<S> + ?Sized,
    B: Fn(S) -> ConstraintState<'a, S>,
{
    // structurally inapplicable members produce `None` in place of a threshold, so the caller can
    // tell "this run leaves that constraint out of the family" apart from "it enforced it and the
    // constraint stayed slack".
    constraint.residual(&blend(S::ONE))?;
    let ok_at = |t: S| -> S::Mask {
        constraint
            .residual(&blend(t))
            .expect("applicability is fixed by the regime, not by the blend parameter")
            .cmp_gt(S::ZERO)
    };
    // the candidate itself is acceptable: exact passthrough, no bisection needed.
    let cand_ok = ok_at(S::ONE);
    let applies = |t: S| -> S {
        constraint
            .residual(&blend(t))
            .expect("applicability is fixed by the regime, not by the blend parameter")
    };
    if constraint.is_affine() {
        // c(t) = c0 + t (c1 - c0) with c0 >= 0 > c1 at a binding constraint, so the crossing is
        // c0 / (c0 - c1). exact, and one node in the trace where the bisection unrolls `iters`.
        let c0 = applies(S::ZERO);
        let c1 = applies(S::ONE);
        let drop = c0 - c1;
        // guard the division for the non-binding case; `cand_ok` selects it away regardless.
        let safe = S::select(drop.cmp_gt(S::ZERO), drop, S::ONE);
        return Some(S::select(cand_ok, S::ONE, (c0 / safe).max(S::ZERO)));
    }
    let mut lo = S::ZERO; // acceptable (the anchor)
    let mut hi = S::ONE; // infeasible end of the bracket (the candidate)
    for _ in 0..iters {
        let mid = S::HALF * (lo + hi);
        let ok = ok_at(mid);
        lo = S::select(ok, mid, lo);
        hi = S::select(ok, hi, mid);
    }
    Some(S::select(cand_ok, S::ONE, lo))
}

/// the per-constraint thresholds for a family, in declaration order.
///
/// the per-member cost buys the ledger; the mathematics is served more cheaply. bisecting the
/// envelope `min_k c_k` directly would be entirely valid and cheaper by a factor of the family
/// size: a pointwise min of concave functions is concave (`min(f,g) = -max(-f,-g)`, and a max of
/// convex functions is convex), so the envelope carries exactly the same single-crossing property
/// each member has. the per-member pass adds the one thing the envelope drops — the identity of the
/// constraint that bound the step — and a budget earns the word audit by naming what it charges.
/// that is what the extra bisections buy. anyone optimizing this later is trading attribution,
/// while correctness holds either way.
pub fn constraint_thetas<'a, S, B>(
    family: &[&dyn StateConstraint<S>],
    blend: &B,
    iters: usize,
) -> Vec<Option<S>>
where
    S: Scalar + 'a,
    B: Fn(S) -> ConstraintState<'a, S>,
{
    family
        .iter()
        .map(|c| constraint_theta(*c, blend, iters))
        .collect()
}

/// the joint blend: the most restrictive member's threshold (A3).
///
/// in this orientation `t = 0` is the anchor and `t = 1` the candidate, so the most restrictive
/// member is the smallest theta. an empty family retains the candidate whole.
pub fn joint_theta<S: Scalar>(thetas: &[Option<S>]) -> S {
    thetas.iter().flatten().fold(S::ONE, |acc, &t| acc.min(t))
}

/// the anchor-feasibility residual: the least constraint residual at `t = 0`.
///
/// every result in this module is conditional on the anchor being acceptable — A2's minimality and
/// the single-crossing argument both assume `c_k(anchor) >= 0`. that precondition belongs to the
/// caller, and it stays invisible to any gate that exercises the projection alone, since such a
/// gate supplies a feasible anchor by construction.
///
/// it is also easy to violate in a way that looks fine. an anchor certifies "this conserved state is
/// admissible", and that certificate holds alongside the magnetic field it was actually computed
/// against. constrained transport evolves `B` on shared faces, so a state assembled against the
/// stage-input field speaks to that field alone, and pairing it with the candidate's post-CT field
/// asserts the admissibility of a hybrid state that no stage ever assembled. every blend then leaves
/// the cell inadmissible, `t = 0` included, and the projection returns a perfectly well-formed theta
/// that happens to be useless.
///
/// the projection therefore reports this value outright: a negative one means the caller handed in
/// an infeasible anchor, a failure distinct from "the candidate needed correcting", and the separate
/// return keeps the two apart.
pub fn anchor_feasibility<'a, S, B>(family: &[&dyn StateConstraint<S>], blend: &B) -> S
where
    S: Scalar + 'a,
    B: Fn(S) -> ConstraintState<'a, S>,
{
    let at_anchor = blend(S::ZERO);
    family
        .iter()
        .filter_map(|c| c.residual(&at_anchor))
        .fold(S::from_f64(f64::INFINITY), |acc, r| acc.min(r))
}

/// the numerical concavity check for axiom A1: the largest violation of
/// `c((U1+U2)/2) >= (c(U1) + c(U2)) / 2` over the supplied sample pairs, normalized by the residual
/// scale. a non-positive result certifies concavity across the sample set.
///
/// this exists so a constraint proposed months from now is graded by ci in the session it is written,
/// while the failure is still local to one commit. it is the test that predicts the Lorentz-factor
/// ceiling's pathology numerically, ahead of any hand derivation.
pub fn concavity_violation<C>(
    constraint: &C,
    samples: &[(f64, Tensor<f64, 3>, Option<f64>, Option<Tensor<f64, 3>>)],
    gm: &Matrix<f64, 3>,
    gm_inv: &Matrix<f64, 3>,
) -> f64
where
    C: StateConstraint<f64> + ?Sized,
{
    let state = |den, mom, nrg, mag| ConstraintState {
        den,
        mom,
        nrg,
        mag,
        gm,
        gm_inv,
    };
    let mut worst = f64::NEG_INFINITY;
    for (ii, a) in samples.iter().enumerate() {
        for b in samples.iter().skip(ii + 1) {
            let (Some(ca), Some(cb)) = (
                constraint.residual(&state(a.0, a.1, a.2, a.3)),
                constraint.residual(&state(b.0, b.1, b.2, b.3)),
            ) else {
                continue;
            };
            let mid_mom = Tensor::new(std::array::from_fn(|k| 0.5 * (a.1[k] + b.1[k])));
            let mid_nrg = match (a.2, b.2) {
                (Some(x), Some(y)) => Some(0.5 * (x + y)),
                _ => None,
            };
            let mid_mag = match (&a.3, &b.3) {
                (Some(x), Some(y)) => {
                    Some(Tensor::new(std::array::from_fn(|k| 0.5 * (x[k] + y[k]))))
                }
                _ => None,
            };
            let Some(cm) =
                constraint.residual(&state(0.5 * (a.0 + b.0), mid_mom, mid_nrg, mid_mag))
            else {
                continue;
            };
            let scale = ca.abs().max(cb.abs()).max(1.0);
            // concavity: the midpoint residual sits at or above the chord.
            let violation = (0.5 * (ca + cb) - cm) / scale;
            worst = worst.max(violation);
        }
    }
    worst
}

/// one constraint's line in the injection budget (A5).
///
/// `model_term` is what keeps the report honest: mass added where the discretization loses a
/// cancellation and mass added because vacuum is being modeled as thin gas are different claims,
/// and an aggregate that merges them hides the modeling term inside the numerical one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintLedgerEntry {
    pub name: &'static str,
    pub model_term: bool,
    /// cell-substage events where this constraint was the binding one.
    pub firings: u64,
    /// the conserved increment this constraint's binding events introduced.
    pub injected_den: f64,
    pub injected_nrg: f64,
}

/// the per-constraint injection budget for a run.
#[derive(Clone, Debug, Default)]
pub struct ConstraintLedger {
    pub entries: Vec<ConstraintLedgerEntry>,
}

impl ConstraintLedger {
    pub fn new(family: &[&dyn StateConstraint<f64>]) -> Self {
        Self {
            entries: family
                .iter()
                .map(|c| ConstraintLedgerEntry {
                    name: c.name(),
                    model_term: c.is_model_term(),
                    ..Default::default()
                })
                .collect(),
        }
    }

    /// book one binding event. `injected` is `Pi(U) - U` for the slots the ledger tracks.
    pub fn book(&mut self, index: usize, injected_den: f64, injected_nrg: f64) {
        let e = &mut self.entries[index];
        e.firings += 1;
        e.injected_den += injected_den;
        e.injected_nrg += injected_nrg;
    }

    /// the injected totals split by A6 category: `(numerical, model)`. the numerical share is the
    /// one that must vanish under mesh refinement.
    pub fn split_by_category(&self) -> ((f64, f64), (f64, f64)) {
        let mut num = (0.0, 0.0);
        let mut model = (0.0, 0.0);
        for e in &self.entries {
            let slot = if e.model_term { &mut model } else { &mut num };
            slot.0 += e.injected_den;
            slot.1 += e.injected_nrg;
        }
        (num, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Matrix<f64, 3> {
        Matrix::identity()
    }

    /// a spread of magnetized states, all with positive density and energy.
    fn samples() -> Vec<(f64, Tensor<f64, 3>, Option<f64>, Option<Tensor<f64, 3>>)> {
        let mut out = Vec::new();
        for &d in &[0.05_f64, 0.4, 1.0, 3.0] {
            for &e_over_d in &[1.1_f64, 1.6, 4.0] {
                for &s in &[0.0_f64, 0.05, 0.2] {
                    out.push((
                        d,
                        Tensor::new([s * d, 0.5 * s * d, -0.25 * s * d]),
                        Some(e_over_d * d),
                        Some(Tensor::new([0.03, -0.02, 0.05])),
                    ));
                }
            }
        }
        out
    }

    #[test]
    fn every_shipped_constraint_is_concave() {
        let (gm, gi) = (identity(), identity());
        let s = samples();
        let checks: Vec<(&str, f64)> = vec![
            (
                "temperature_floor",
                concavity_violation(&TemperatureFloor { f_min: 1e-3 }, &s, &gm, &gi),
            ),
            (
                "magnetization_ceiling",
                concavity_violation(&MagnetizationCeiling { sigma_max: 50.0 }, &s, &gm, &gi),
            ),
            (
                "density_floor",
                concavity_violation(&DensityFloor { den_min: 1e-4 }, &s, &gm, &gi),
            ),
            (
                "wu_tang_admissibility",
                concavity_violation(
                    &WuTangAdmissibility {
                        eps_d: 0.0,
                        eps_q: 0.0,
                        eps_psi: 0.0,
                    },
                    &s,
                    &gm,
                    &gi,
                ),
            ),
        ];
        for (name, worst) in checks {
            assert!(
                worst <= 1e-12,
                "{name} violates concavity (axiom A1) by {worst:e}; the projection's \
                 unique-crossing proof does not hold for it"
            );
        }
    }

    /// the test that grades a proposed constraint. a Lorentz-factor ceiling is the canonical
    /// non-concave floor, and the checker rejects it numerically, ahead of any hand derivation —
    /// which is what gives A1 its force.
    #[test]
    fn the_concavity_checker_rejects_a_lorentz_ceiling() {
        struct LorentzCeiling {
            w_max: f64,
        }
        impl StateConstraint<f64> for LorentzCeiling {
            fn residual(&self, u: &ConstraintState<f64>) -> Option<f64> {
                // W ~ E / D for a cold flow; the ceiling W <= W_max is a ratio constraint, and a
                // ratio's sublevel set is non-convex.
                let e = u.nrg.unwrap_or(0.0);
                Some(self.w_max - e / u.den)
            }
            fn scale(&self, dx: f64) -> f64 {
                dx * dx
            }
            fn name(&self) -> &'static str {
                "lorentz_ceiling"
            }
        }
        let (gm, gi) = (identity(), identity());
        let worst = concavity_violation(&LorentzCeiling { w_max: 3.0 }, &samples(), &gm, &gi);
        assert!(
            worst > 1e-12,
            "the checker passed a ratio constraint; it cannot then be trusted to grade a new one"
        );
    }

    #[test]
    fn an_acceptable_candidate_passes_through_exactly() {
        let (gm, gi) = (identity(), identity());
        let floor = TemperatureFloor { f_min: 1e-3 };
        let blend = |t: f64| ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            // both endpoints comfortably above the floor.
            nrg: Some(2.0 + t * 0.5),
            mag: Some(Tensor::new([0.1, 0.0, 0.0])),
            gm: &gm,
            gm_inv: &gi,
        };
        assert_eq!(constraint_theta(&floor, &blend, 32), Some(1.0));
    }

    #[test]
    fn a_violating_candidate_is_pulled_exactly_to_the_boundary() {
        let (gm, gi) = (identity(), identity());
        let f_min = 0.25;
        let floor = TemperatureFloor { f_min };
        // residual(t) = (2.25 - 1.5 t) - D - f_min D = 1.0 - 1.5 t, so the anchor sits a clear 1.0
        // inside, the candidate 0.5 outside, and the crossing is analytic at t = 2/3 — which grades
        // the bisection against a known number.
        let blend = |t: f64| ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: Some(2.25 - t * 1.5),
            mag: None,
            gm: &gm,
            gm_inv: &gi,
        };
        let theta = constraint_theta(&floor, &blend, 48).expect("the floor applies here");
        assert!((theta - 2.0 / 3.0).abs() < 1e-12, "theta = {theta}");
        // and the projected state sits exactly on the boundary.
        let residual = floor.residual(&blend(theta)).expect("applicable");
        assert!(
            residual.abs() < 1e-12,
            "residual {residual:e} off the boundary"
        );
    }

    #[test]
    fn the_projection_is_idempotent() {
        // A2: projecting an already-projected state is a no-op. a clamp lacking idempotence moves
        // the state a little further every substage, and that surfaces as a slow leak, below the
        // threshold anyone would notice.
        let (gm, gi) = (identity(), identity());
        let family: Vec<&dyn StateConstraint<f64>> = vec![
            &TemperatureFloor { f_min: 0.2 },
            &MagnetizationCeiling { sigma_max: 1.0 },
        ];
        let anchor_nrg = 3.0;
        let blend = |t: f64| ConstraintState {
            den: 1.0 - t * 0.9,
            mom: Tensor::zeros(),
            nrg: Some(anchor_nrg - t * 2.0),
            mag: Some(Tensor::new([0.5, 0.0, 0.0])),
            gm: &gm,
            gm_inv: &gi,
        };
        let first = joint_theta(&constraint_thetas(&family, &blend, 48));
        assert!(first > 0.0 && first < 1.0, "setup must actually project");

        // re-project from the same anchor with the projected state as the new candidate: the
        // segment is now [anchor, Pi(U)], so the threshold is exactly 1 and the state stands.
        let reblend = |t: f64| blend(t * first);
        let second = joint_theta(&constraint_thetas(&family, &reblend, 48));
        assert_eq!(
            second, 1.0,
            "projecting an already-acceptable state moved it again"
        );
    }

    #[test]
    fn an_inapplicable_constraint_is_distinguishable_from_a_satisfied_one() {
        // the c2p-status failure mode, in this algebra: a constraint that signalled "inapplicable
        // to this regime" by returning a comfortable positive residual would read exactly like one
        // that ran and passed, so a run would believe it was enforcing constraints that were
        // structurally inert. `None` vs `Some(x >= 0)` keeps the two apart.
        let (gm, gi) = (identity(), identity());
        let iso = ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: None, // isothermal: the energy slot is empty
            mag: None, // unmagnetized: the field slot is empty
            gm: &gm,
            gm_inv: &gi,
        };
        assert_eq!(
            TemperatureFloor { f_min: 1e-3 }.residual(&iso),
            None,
            "a thermal floor on a regime with no energy slot must report inapplicable"
        );
        assert_eq!(
            MagnetizationCeiling { sigma_max: 50.0 }.residual(&iso),
            None,
            "a magnetization ceiling with no field must report inapplicable"
        );
        // the density floor applies to every regime, and reports a real residual.
        assert_eq!(
            DensityFloor { den_min: 0.25 }.residual(&iso),
            Some(0.75),
            "a density floor applies to every regime"
        );

        // and an inapplicable member returns `None` in place of a threshold, so it stays out of the
        // joint minimum entirely.
        let blend = |_t: f64| ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: None,
            mag: None,
            gm: &gm,
            gm_inv: &gi,
        };
        assert_eq!(
            constraint_theta(&TemperatureFloor { f_min: 1e-3 }, &blend, 16),
            None
        );
    }

    #[test]
    fn numerical_constraints_vanish_with_the_mesh_and_model_terms_do_not() {
        // A6, made structural. a numerical constraint stands in for truncation error, and its
        // shrinking under refinement is what carries the computation to the equations being
        // claimed. a model term stands in for vacuum, which stays beyond the equations' reach at
        // every mesh spacing.
        let coarse = 1.0e-1;
        let fine = 0.5e-1;
        for c in [
            &TemperatureFloor { f_min: 1.0 } as &dyn StateConstraint<f64>,
            &MagnetizationCeiling { sigma_max: 1.0 },
        ] {
            assert!(
                c.scale(fine) < c.scale(coarse),
                "{} is numerical but its scale does not vanish with the mesh",
                c.name()
            );
            assert!(!c.is_model_term(), "{} should be numerical", c.name());
        }
        let vacuum = DensityFloor { den_min: 1.0e-8 };
        assert_eq!(
            vacuum.scale(fine),
            vacuum.scale(coarse),
            "a vacuum model term must not pretend to vanish under refinement"
        );
        assert!(vacuum.is_model_term());
        // representability carries a mesh scale of exactly zero.
        let g = WuTangAdmissibility {
            eps_d: 0.0,
            eps_q: 0.0,
            eps_psi: 0.0,
        };
        assert_eq!(g.scale(coarse), 0.0);
        assert!(!g.is_model_term());
    }

    /// forces the bisection path on a constraint that would otherwise take the closed form, so the
    /// two can be compared against each other.
    struct ForceBisect<'c, S: Scalar>(&'c dyn StateConstraint<S>);
    impl<S: Scalar> StateConstraint<S> for ForceBisect<'_, S> {
        fn residual(&self, u: &ConstraintState<S>) -> Option<S> {
            self.0.residual(u)
        }
        fn scale(&self, dx: f64) -> f64 {
            self.0.scale(dx)
        }
        fn is_affine(&self) -> bool {
            false
        }
        fn name(&self) -> &'static str {
            self.0.name()
        }
    }

    #[test]
    fn every_affine_claim_is_actually_affine() {
        // `is_affine` selects a closed form whose validity rests on the residual truly being affine
        // — a wrong claim silently returns the wrong crossing. affinity means the midpoint sits
        // exactly on the chord, so the concavity violation vanishes in both directions, where A1
        // alone bounds it on one side.
        let (gm, gi) = (identity(), identity());
        let s = samples();
        let affine: Vec<&dyn StateConstraint<f64>> = vec![
            &TemperatureFloor { f_min: 1e-3 },
            &MagnetizationCeiling { sigma_max: 50.0 },
            &DensityFloor { den_min: 1e-4 },
        ];
        for c in affine {
            assert!(c.is_affine(), "{} should claim affinity", c.name());
            let v = concavity_violation(c, &s, &gm, &gi);
            assert!(
                v.abs() <= 1e-12,
                "{} claims affine but bends by {v:e}; the closed-form crossing is invalid for it",
                c.name()
            );
        }
        // and the nonlinear member declines the claim.
        let g = WuTangAdmissibility {
            eps_d: 0.0,
            eps_q: 0.0,
            eps_psi: 0.0,
        };
        assert!(!g.is_affine(), "the wu-tang residual is not affine");
    }

    #[test]
    fn the_closed_form_crossing_agrees_with_bisection() {
        // the closed form and the bisection locate the same crossing of the constraint residual.
        let (gm, gi) = (identity(), identity());
        let floor = TemperatureFloor { f_min: 0.25 };
        let blend = |t: f64| ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: Some(2.25 - t * 1.5),
            mag: None,
            gm: &gm,
            gm_inv: &gi,
        };
        let closed = constraint_theta(&floor, &blend, 48).expect("applicable");
        let bisected = constraint_theta(&ForceBisect(&floor), &blend, 48).expect("applicable");
        assert!(
            (closed - bisected).abs() < 1e-12,
            "closed form {closed} vs bisection {bisected}"
        );
        // the closed form lands on the exact rational, where the bisection converges toward it.
        assert_eq!(closed, 2.0 / 3.0);
    }

    #[test]
    fn a_slack_configured_constraint_still_reports_some() {
        // "configured so loose it stays slack" and "structurally inapplicable" are distinct states,
        // and collapsing them reopens the silent-pass hole one layer up: a ledger that reads "this
        // constraint was switched off" the same as "it was on and satisfied" is the A5 form of a
        // status field that defaults to zero. a slack constraint has a well-defined, comfortably
        // positive residual and reports it.
        let (gm, gi) = (identity(), identity());
        let live = ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: Some(3.0),
            mag: Some(Tensor::new([0.1, 0.0, 0.0])),
            gm: &gm,
            gm_inv: &gi,
        };
        // neutral / slack configurations: on, applicable, and comfortably clear of binding.
        for (name, r) in [
            (
                "temperature_floor",
                TemperatureFloor { f_min: 0.0 }.residual(&live),
            ),
            (
                "magnetization_ceiling",
                MagnetizationCeiling {
                    sigma_max: f64::INFINITY,
                }
                .residual(&live),
            ),
            (
                "density_floor",
                DensityFloor { den_min: 0.0 }.residual(&live),
            ),
        ] {
            let v = r.unwrap_or_else(|| {
                panic!(
                    "{name} configured slack reported INAPPLICABLE; a config value must not be \
                        able to masquerade as structural absence"
                )
            });
            assert!(
                v > 0.0,
                "{name} slack residual should be comfortably positive, got {v:e}"
            );
        }
        // only genuine structural absence yields None, and it does so regardless of configuration.
        let isothermal = ConstraintState {
            nrg: None,
            ..ConstraintState {
                den: 1.0,
                mom: Tensor::zeros(),
                nrg: None,
                mag: None,
                gm: &gm,
                gm_inv: &gi,
            }
        };
        assert_eq!(TemperatureFloor { f_min: 0.0 }.residual(&isothermal), None);
        assert_eq!(TemperatureFloor { f_min: 1e3 }.residual(&isothermal), None);
    }

    #[test]
    fn affinity_holds_across_the_whole_configurable_range() {
        // the scalars are runtime configurable, so affinity holds across the whole range a config
        // can set, well beyond the single value live when this test was written. it holds because
        // each scalar enters its residual as a coefficient of a conserved slot or as a constant
        // offset, which translates the affine form while leaving its slope intact; the sweep is
        // what turns that from an assumption into a measurement, since a wrong `is_affine` returns
        // a wrong crossing silently.
        let (gm, gi) = (identity(), identity());
        let s = samples();
        for &f_min in &[0.0_f64, 1e-12, 1e-6, 1e-2, 1.0, 1e3] {
            let v = concavity_violation(&TemperatureFloor { f_min }, &s, &gm, &gi);
            assert!(
                v.abs() <= 1e-12,
                "temperature_floor bends at f_min={f_min:e}: {v:e}"
            );
        }
        for &sigma_max in &[0.0_f64, 1e-3, 1.0, 50.0, 1e6, 1e12] {
            let v = concavity_violation(&MagnetizationCeiling { sigma_max }, &s, &gm, &gi);
            assert!(
                v.abs() <= 1e-12,
                "magnetization_ceiling bends at sigma_max={sigma_max:e}: {v:e}"
            );
        }
        for &den_min in &[0.0_f64, 1e-12, 1e-4, 1.0, 1e3] {
            let v = concavity_violation(&DensityFloor { den_min }, &s, &gm, &gi);
            assert!(
                v.abs() <= 1e-12,
                "density_floor bends at den_min={den_min:e}: {v:e}"
            );
        }
    }

    #[test]
    fn an_anchor_certified_against_a_different_field_is_reported_infeasible() {
        // an anchor certifies "this conserved state is admissible", and the certificate holds
        // alongside the magnetic field it was computed against. constrained transport advances B on
        // shared faces, so a stage-input-derived anchor paired with the candidate's post-CT field
        // asserts the admissibility of a hybrid state that no stage ever assembled. every blend
        // the cell inadmissible — the projection returns a well-formed theta that is simply
        // useless — so the infeasibility is reported on its own channel, apart from theta = 0.
        let (gm, gi) = (identity(), identity());
        let family: Vec<&dyn StateConstraint<f64>> = vec![&WuTangAdmissibility {
            eps_d: 0.0,
            eps_q: 0.0,
            eps_psi: 0.0,
        }];

        // a cold, weakly magnetized state: admissible alongside the field it was built with.
        let own_field = Tensor::new([0.05, 0.0, 0.0]);
        let with_own = |_t: f64| ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: Some(1.4),
            mag: Some(own_field),
            gm: &gm,
            gm_inv: &gi,
        };
        assert!(
            anchor_feasibility(&family, &with_own) > 0.0,
            "PREMISE: the anchor must be feasible against its OWN field, or the contrast below \
             says nothing"
        );

        // the same conserved state, now paired with a much stronger field — what CT would have
        // advanced to. psi carries the magnetic energy, so the certificate is void.
        let strong_field = Tensor::new([3.0, 0.0, 0.0]);
        let with_candidate_field = |_t: f64| ConstraintState {
            den: 1.0,
            mom: Tensor::zeros(),
            nrg: Some(1.4),
            mag: Some(strong_field),
            gm: &gm,
            gm_inv: &gi,
        };
        assert!(
            anchor_feasibility(&family, &with_candidate_field) < 0.0,
            "an anchor paired with a field it was never certified against was reported FEASIBLE; \
             the projection would then return a theta that recovers nothing and the caller would \
             have no signal that its anchor construction was wrong"
        );

        // and the projection over such an anchor is degenerate — which is exactly why the
        // feasibility residual is surfaced on its own, where theta alone stays ambiguous.
        let theta = joint_theta(&constraint_thetas(&family, &with_candidate_field, 32));
        assert_eq!(
            theta, 0.0,
            "an infeasible anchor collapses the blend, which is indistinguishable from a candidate \
             needing full correction unless feasibility is reported on its own"
        );
    }

    #[test]
    fn the_ledger_separates_numerical_from_model_injection() {
        // A6: the two categories answer different questions, so each keeps its own number.
        let family: Vec<&dyn StateConstraint<f64>> = vec![
            &TemperatureFloor { f_min: 1e-6 },
            &DensityFloor { den_min: 1e-8 },
        ];
        let mut ledger = ConstraintLedger::new(&family);
        assert!(
            !ledger.entries[0].model_term,
            "temperature floor is numerical"
        );
        assert!(ledger.entries[1].model_term, "density floor models vacuum");

        ledger.book(0, 0.0, 4.0e-7); // truncation-driven energy injection
        ledger.book(1, 2.0e-9, 0.0); // vacuum-model mass injection
        let ((num_d, num_e), (mod_d, mod_e)) = ledger.split_by_category();
        assert_eq!((num_d, num_e), (0.0, 4.0e-7));
        assert_eq!((mod_d, mod_e), (2.0e-9, 0.0));
    }

    #[test]
    fn the_joint_threshold_is_the_binding_constraint() {
        // A3: the family binds simultaneously, so the joint blend is the most restrictive member's.
        // in this orientation t = 0 is the anchor and t = 1 the candidate, so "most restrictive"
        // is the smallest theta (least of the candidate retained). the literature states the same
        // rule as a max because it measures t from the candidate; the two are t -> 1 - t apart.
        // applying the members in sequence would be order-dependent and non-idempotent.
        let (gm, gi) = (identity(), identity());
        let blend = |t: f64| ConstraintState {
            den: 1.0 - t * 0.9,
            mom: Tensor::zeros(),
            nrg: Some(2.0 - t * 1.0),
            mag: Some(Tensor::new([0.5, 0.0, 0.0])),
            gm: &gm,
            gm_inv: &gi,
        };
        let temp = TemperatureFloor { f_min: 0.2 };
        let sigma = MagnetizationCeiling { sigma_max: 1.0 };
        let family: Vec<&dyn StateConstraint<f64>> = vec![&temp, &sigma];
        let joint = joint_theta(&constraint_thetas(&family, &blend, 48));
        // the binding constraint's own residual is zero at the joint blend; the other is satisfied.
        assert!(temp.residual(&blend(joint)).expect("applicable") >= -1e-12);
        assert!(sigma.residual(&blend(joint)).expect("applicable") >= -1e-12);
        // and going past it violates the binding one, so the blend is minimal (A2).
        let past = joint + 1e-3;
        assert!(
            temp.residual(&blend(past)).expect("applicable") < 0.0
                || sigma.residual(&blend(past)).expect("applicable") < 0.0,
            "the joint blend was not the binding threshold"
        );
    }
}
