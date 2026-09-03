// =============================================================================
// admissibility.rs
//
// the law-indexed host admissibility witness: membership in a named
// constraint family is a proof — `Admissible<T, L>` pairs the judged state
// with the law's witness, and the law's `validate` is the only constructor.
// this is a different phase from `Recovered<T>`: recovery acceptance proves
// the strict c2p interior, membership here proves the named family, and no
// conversion between the two exists (pinned as a compile-fail doctest on
// `Admissible`).
//
// the one law instantiated here is the Wu & Tang RMHD admissible set over the
// undensitized eulerian conserved slots, with the production margin rule
// (eps_d = 1e-12 * scale, eps_q = floor * scale, eps_psi = floor * scale^{3/2},
// one shared local conserved-state scale). the anchored query returns the
// tri-state verdict the raw theta could never spell: membership, a binding
// constraint with its blend fraction, or an infeasible anchor — the anchor
// itself outside the set, which the bisection's theta = 0 used to shadow.
// the dormant `constraints` projection family stays unwired; this module
// types the host queries alone.
//
// usage:
//   match WuTang::judge_against_anchor(candidate, &anchor, &gm_inv, &gm, 40) {
//       WuTangVerdict::Member(proof) => consume(proof),
//       WuTangVerdict::Binding { theta } => blend(theta),
//       WuTangVerdict::InfeasibleAnchor { .. } => reject_anchor(),
//   }
// =============================================================================

use symbi_algebra::{Matrix, OrderedNumeric, Tensor};
use symbi_carrier::Scalar;

use crate::admissible::{
    ADMISSIBLE_REL_FLOOR, rmhd_admissible_residuals, rmhd_admissible_theta, rmhd_state_scale,
};

/// a named admissibility law over the state family it judges: the witness
/// type is the evidence `validate` proves and `Admissible` carries.
pub trait AdmissibilityLaw<T> {
    type Witness;
}

/// a state the law `L` accepted, carrying the law's witness. minted by the
/// law's `validate` alone —
///
/// ```compile_fail
/// use symbi_hydro::{Admissible, WuTang, WuTangState};
/// fn forge(s: WuTangState<f64>) -> Admissible<WuTangState<f64>, WuTang> {
///     Admissible { value: s, witness: todo!() }
/// }
/// ```
///
/// and a `Recovered` state proves the recovery interior only; broadening it
/// into a law-indexed membership has no conversion —
///
/// ```compile_fail
/// use symbi_hydro::{Admissible, Recovered, WuTang, WuTangState};
/// fn broaden(r: Recovered<WuTangState<f64>>) -> Admissible<WuTangState<f64>, WuTang> {
///     r.into()
/// }
/// ```
pub struct Admissible<T, L: AdmissibilityLaw<T>> {
    value: T,
    witness: L::Witness,
}

impl<T: std::fmt::Debug, L: AdmissibilityLaw<T>> std::fmt::Debug for Admissible<T, L>
where
    L::Witness: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Admissible")
            .field("value", &self.value)
            .field("witness", &self.witness)
            .finish()
    }
}

impl<T, L: AdmissibilityLaw<T>> Admissible<T, L> {
    pub fn value(&self) -> &T {
        &self.value
    }
    pub fn witness(&self) -> &L::Witness {
        &self.witness
    }
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// the eulerian conserved slots the Wu & Tang set judges: the undensitized
/// `D`, the covariant momentum `S_i`, the total eulerian energy
/// `total_energy = E = tau + D` (`MhdCons.nrg` stores tau; the distinct name
/// is this type's reason to exist), and the contravariant cell field `B^i`.
#[derive(Clone, Copy, Debug)]
pub struct WuTangState<S> {
    pub den: S,
    pub mom: Tensor<S, 3>,
    pub total_energy: S,
    pub mag: Tensor<S, 3>,
}

/// the finite-precision margins of the strict interior, from one shared local
/// conserved-state scale (the production rule the projection kernel bakes).
#[derive(Clone, Copy, Debug)]
pub struct WuTangMargins<S> {
    pub eps_d: S,
    pub eps_q: S,
    pub eps_psi: S,
}

/// the membership evidence: the tested density and residual pair, each
/// strictly inside its margin.
#[derive(Clone, Copy, Debug)]
pub struct WuTangWitness<S> {
    pub den: S,
    pub q: S,
    pub psi: S,
    pub margins: WuTangMargins<S>,
}

/// the complete rejection evidence: the tested density and residual pair with
/// the governing margins, so a density-only violation shows itself beside
/// apparently healthy residuals.
#[derive(Clone, Copy, Debug)]
pub struct WuTangOutside<S> {
    pub den: S,
    pub q: S,
    pub psi: S,
    pub margins: WuTangMargins<S>,
}

/// the Wu & Tang RMHD admissible set (arXiv:1610.06274 theorem-2.1 form):
/// `D > 0`, `q = E - sqrt(D^2 + |S|^2) > 0`, and the magnetized residual
/// `psi > 0`, each strictly inside its finite-precision margin.
pub struct WuTang;

impl<S: Scalar + OrderedNumeric> AdmissibilityLaw<WuTangState<S>> for WuTang {
    type Witness = WuTangWitness<S>;
}

/// the anchored verdict: membership (exact passthrough), a binding constraint
/// with the largest admissible blend fraction toward the anchor, or an
/// infeasible anchor — the anchor itself outside the set, so no blend along
/// the segment exists and theta carries no information.
pub enum WuTangVerdict<S: Scalar + OrderedNumeric> {
    Member(Admissible<WuTangState<S>, WuTang>),
    Binding { theta: S },
    InfeasibleAnchor { outside: WuTangOutside<S> },
}

impl WuTang {
    /// the production margin rule at a state's own scale.
    pub fn margins<S: Scalar + OrderedNumeric>(
        state: &WuTangState<S>,
        gm_inv: &Matrix<S, 3>,
        gm: &Matrix<S, 3>,
    ) -> WuTangMargins<S> {
        let scale = rmhd_state_scale(
            state.den,
            &state.mom,
            state.total_energy,
            &state.mag,
            gm_inv,
            gm,
        );
        WuTangMargins {
            eps_d: S::from_f64(1e-12) * scale,
            eps_q: S::from_f64(ADMISSIBLE_REL_FLOOR) * scale,
            eps_psi: S::from_f64(ADMISSIBLE_REL_FLOOR) * scale * scale.sqrt(),
        }
    }

    /// membership against explicit margins — the one primitive every query
    /// shares, and the sole mint of `Admissible<WuTangState, WuTang>`. the
    /// witness records the margins it proved, so a proof states which
    /// numerical interior holds.
    fn validate_with_margins<S: Scalar + OrderedNumeric>(
        state: WuTangState<S>,
        margins: WuTangMargins<S>,
        gm_inv: &Matrix<S, 3>,
        gm: &Matrix<S, 3>,
    ) -> Result<Admissible<WuTangState<S>, WuTang>, WuTangOutside<S>> {
        let (q, psi) = rmhd_admissible_residuals(
            state.den,
            &state.mom,
            state.total_energy,
            &state.mag,
            gm_inv,
            gm,
        );
        let den = state.den;
        if den > margins.eps_d && q > margins.eps_q && psi > margins.eps_psi {
            Ok(Admissible {
                value: state,
                witness: WuTangWitness {
                    den,
                    q,
                    psi,
                    margins,
                },
            })
        } else {
            Err(WuTangOutside {
                den,
                q,
                psi,
                margins,
            })
        }
    }

    /// membership in the strict interior at the state's own scale.
    pub fn validate<S: Scalar + OrderedNumeric>(
        state: WuTangState<S>,
        gm_inv: &Matrix<S, 3>,
        gm: &Matrix<S, 3>,
    ) -> Result<Admissible<WuTangState<S>, WuTang>, WuTangOutside<S>> {
        let margins = Self::margins(&state, gm_inv, gm);
        Self::validate_with_margins(state, margins, gm_inv, gm)
    }

    /// the anchored query on the candidate's magnetic slice: the candidate and
    /// the anchor share the candidate's `B` (a CT-owned field is not blended),
    /// and the verdict separates the three outcomes the raw theta conflated.
    /// margins come from the anchor's scale, matching the projection kernel.
    pub fn judge_against_anchor<S: Scalar + OrderedNumeric>(
        candidate: WuTangState<S>,
        anchor: &WuTangState<S>,
        gm_inv: &Matrix<S, 3>,
        gm: &Matrix<S, 3>,
        iters: usize,
    ) -> WuTangVerdict<S> {
        let anchor_slice = WuTangState {
            den: anchor.den,
            mom: anchor.mom,
            total_energy: anchor.total_energy,
            mag: candidate.mag,
        };
        // one margin set — the anchor slice's, matching the projection kernel —
        // governs anchor feasibility, candidate membership, and the theta
        // search alike, so `Member` means the production anchored query passes
        // through exactly.
        let margins = Self::margins(&anchor_slice, gm_inv, gm);
        if let Err(outside) = Self::validate_with_margins(anchor_slice, margins, gm_inv, gm) {
            return WuTangVerdict::InfeasibleAnchor { outside };
        }
        if let Ok(proof) = Self::validate_with_margins(candidate, margins, gm_inv, gm) {
            return WuTangVerdict::Member(proof);
        }
        let theta = rmhd_admissible_theta(
            candidate.den,
            candidate.mom,
            candidate.total_energy,
            anchor_slice.den,
            anchor_slice.mom,
            anchor_slice.total_energy,
            &candidate.mag,
            gm_inv,
            gm,
            margins.eps_d,
            margins.eps_q,
            margins.eps_psi,
            iters,
        );
        WuTangVerdict::Binding { theta }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Matrix<f64, 3> {
        Matrix::identity()
    }

    fn calm(b: [f64; 3]) -> WuTangState<f64> {
        WuTangState {
            den: 1.0,
            mom: Tensor::new([0.1, 0.0, 0.0]),
            total_energy: 2.0,
            mag: Tensor::new(b),
        }
    }

    /// a comfortably interior state validates and its witness carries the
    /// strictly positive residuals; an out-of-cone state returns the residual
    /// evidence instead of a proof.
    #[test]
    fn validate_separates_member_from_outside() {
        let gm = flat();
        let proof = WuTang::validate(calm([0.1, 0.0, 0.0]), &gm, &gm).expect("interior state");
        assert!(proof.witness().q > 0.0 && proof.witness().psi > 0.0);
        assert_eq!(proof.value().den, 1.0);

        let mut hot = calm([0.1, 0.0, 0.0]);
        hot.mom = Tensor::new([50.0, 0.0, 0.0]);
        let outside = WuTang::validate(hot, &gm, &gm).unwrap_err();
        assert!(outside.q <= outside.margins.eps_q);
    }

    /// the anchored verdict's three arms: a member passes through exactly, a
    /// bad candidate against a good anchor binds with theta in (0, 1), and a
    /// bad anchor is named infeasible rather than shadowed as theta = 0.
    #[test]
    fn anchored_verdict_separates_the_three_outcomes() {
        let gm = flat();
        let anchor = calm([0.1, 0.0, 0.0]);

        match WuTang::judge_against_anchor(calm([0.1, 0.0, 0.0]), &anchor, &gm, &gm, 40) {
            WuTangVerdict::Member(proof) => assert!(proof.witness().q > 0.0),
            other => panic!(
                "an admissible candidate must be a member, got {}",
                match other {
                    WuTangVerdict::Binding { .. } => "Binding",
                    WuTangVerdict::InfeasibleAnchor { .. } => "InfeasibleAnchor",
                    WuTangVerdict::Member(_) => unreachable!(),
                }
            ),
        }

        let mut bad = calm([0.1, 0.0, 0.0]);
        bad.den *= 0.5;
        bad.mom = bad.mom.scale(200.0);
        bad.total_energy *= 0.1;
        match WuTang::judge_against_anchor(bad, &anchor, &gm, &gm, 40) {
            WuTangVerdict::Binding { theta } => {
                assert!(theta < 1.0 && theta >= 0.0, "theta = {theta}")
            }
            _ => panic!("an out-of-cone candidate against a good anchor binds"),
        }

        let mut bad_anchor = anchor;
        bad_anchor.total_energy = 0.01;
        match WuTang::judge_against_anchor(bad, &bad_anchor, &gm, &gm, 40) {
            WuTangVerdict::InfeasibleAnchor { outside } => {
                assert!(outside.q <= 0.0, "q = {}", outside.q);
                assert!(outside.den > 0.0, "the evidence carries the tested density");
            }
            _ => panic!("an inadmissible anchor is named, never shadowed as theta = 0"),
        }
    }

    /// the binding theta lands the blended state inside the strict interior —
    /// the same law the projection kernel enforces.
    #[test]
    fn binding_theta_projects_into_the_interior() {
        let gm = flat();
        let anchor = calm([0.1, 0.0, 0.0]);
        let mut bad = calm([0.1, 0.0, 0.0]);
        bad.den *= 0.5;
        bad.mom = bad.mom.scale(200.0);
        bad.total_energy *= 0.1;
        let WuTangVerdict::Binding { theta } =
            WuTang::judge_against_anchor(bad, &anchor, &gm, &gm, 40)
        else {
            panic!("expected a binding constraint");
        };
        let blend = |a: f64, c: f64| a + theta * (c - a);
        let projected = WuTangState {
            den: blend(anchor.den, bad.den),
            mom: Tensor::new(std::array::from_fn(|k| blend(anchor.mom[k], bad.mom[k]))),
            total_energy: blend(anchor.total_energy, bad.total_energy),
            mag: bad.mag,
        };
        assert!(
            matches!(
                WuTang::judge_against_anchor(projected, &anchor, &gm, &gm, 40),
                WuTangVerdict::Member(_)
            ),
            "the blend at theta lies inside the anchor-relative strict interior"
        );
    }

    /// the anchored query judges membership under the anchor slice's margins:
    /// a candidate that is a member at its own scale can still bind against a
    /// far larger anchor, whose finite-precision interior it does not reach.
    /// the premise is asserted on both margin sets, so the test fails loudly
    /// if the constructed scales stop separating the arms.
    #[test]
    fn membership_is_judged_under_the_anchor_margins() {
        let gm = flat();
        // a near-cone candidate: q = 1e-8 at unit scale, comfortably inside
        // its own margins.
        let near = WuTangState {
            den: 1.0,
            mom: Tensor::new([1.0, 0.0, 0.0]),
            total_energy: 2.0f64.sqrt() + 1e-8,
            mag: Tensor::new([0.0, 0.0, 0.0]),
        };
        assert!(
            WuTang::validate(near, &gm, &gm).is_ok(),
            "premise: the candidate is a member at its own scale"
        );
        // an anchor six orders of magnitude larger: its margins dwarf the
        // candidate's residuals.
        let big = WuTangState {
            den: 1e6,
            mom: Tensor::new([1e5, 0.0, 0.0]),
            total_energy: 2e6,
            mag: Tensor::new([0.0, 0.0, 0.0]),
        };
        let margins = WuTang::margins(&big, &gm, &gm);
        assert!(
            WuTang::validate_with_margins(near, margins, &gm, &gm).is_err(),
            "premise: the candidate misses the anchor-scale interior"
        );
        match WuTang::judge_against_anchor(near, &big, &gm, &gm, 40) {
            WuTangVerdict::Binding { theta } => assert!(theta < 1.0),
            WuTangVerdict::Member(_) => {
                panic!("membership must be judged under the anchor margins")
            }
            WuTangVerdict::InfeasibleAnchor { .. } => panic!("the anchor is feasible"),
        }
    }
}
