// =============================================================================
// traced_recovery.rs
//
// the carrier interpretation of the recovery validity law: on a symbolic
// carrier the outcome is a product, because a Rust enum would branch while
// tracing. `TracedRecovery` pairs the branch-free candidate with
// `KernelC2pStatus`, the typed accept/reject fact of the recovery-interior
// predicate — the same acceptance meaning the host audits certify, on the
// carrier's mask algebra: finite positive density, finite positive pressure
// where the regime evolves energy, finite velocity components, finite cell-B
// components where the family is magnetized, and a subluminal metric velocity
// norm where the family is relativistic.
//
// the per-family constructors below are the only doors: each derives the
// status from the candidate it captures, so a status cannot be paired with a
// state it does not describe. the family carries the law because the
// candidate type alone cannot — `Prim<S, D>` serves both the newtonian and
// the relativistic regimes, and the relativistic norm contracts with the
// spatial metric, which only the call site holds. the channel renderer (the
// kernel materialization in symbi-discretize) owns the on-field
// representation of the fact.
//
// usage:
//   let rec = traced_recovery::relativistic(prim, &metric);
//   let (prim, status) = rec.into_parts();
//   // materialize: candidate field writes + the status channel write
// =============================================================================

use std::ops::BitAnd;

use symbi_algebra::Tensor;
use symbi_carrier::Scalar;

use crate::mhd_state::{IsoMhdPrim, MhdPrim};
use crate::spatial_metric::SpatialMetric;
use crate::state::Prim;

/// the kernel-side recovery fact: the acceptance mask of the interior
/// predicate, on the carrier's own mask type (`GvMask` when tracing, `bool`
/// on a host carrier). the field is private and the family constructors are
/// the only producers.
pub struct KernelC2pStatus<M> {
    accepted: M,
}

impl<M> KernelC2pStatus<M> {
    /// the acceptance mask, handed to the channel renderer that materializes
    /// the status write.
    #[inline]
    pub fn accepted(self) -> M {
        self.accepted
    }
}

/// the traced recovery outcome: the branch-free candidate plus the typed
/// status fact derived from it. both halves materialize together — candidate
/// field writes and a dedicated status channel write — so a failed pressure
/// is data, never the control signal.
pub struct TracedRecovery<T, M> {
    candidate: T,
    status: KernelC2pStatus<M>,
}

impl<T, M> TracedRecovery<T, M> {
    #[inline]
    fn certify(candidate: T, accepted: M) -> Self {
        Self {
            candidate,
            status: KernelC2pStatus { accepted },
        }
    }

    #[inline]
    pub fn into_parts(self) -> (T, KernelC2pStatus<M>) {
        (self.candidate, self.status)
    }
}

/// the shared gas interior on the carrier: finite positive density, finite
/// positive pressure when the family evolves energy, finite velocity
/// components. finiteness is `(v - v) == 0`, so NaN and both infinities
/// reject.
fn gas_accepted<S, const D: usize>(rho: S, pre: Option<S>, vel: &Tensor<S, D>) -> S::Mask
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let finite_pos = |v: S| (v - v).cmp_eq(S::ZERO) & v.cmp_gt(S::ZERO);
    let finite = |v: S| (v - v).cmp_eq(S::ZERO);
    let mut accepted = match pre {
        Some(pre) => finite_pos(rho) & finite_pos(pre),
        None => finite_pos(rho),
    };
    for k in 0..D {
        accepted = accepted & finite(vel[k]);
    }
    accepted
}

/// componentwise carrier finiteness of a tensor, folded onto a mask.
fn tensor_accepted<S, const D: usize>(acc: S::Mask, x: &Tensor<S, D>) -> S::Mask
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let mut accepted = acc;
    for k in 0..D {
        accepted = accepted & (x[k] - x[k]).cmp_eq(S::ZERO);
    }
    accepted
}

/// the newtonian family: the gas interior with the energy-evolving pressure.
pub fn newtonian<S, const D: usize>(prim: Prim<S, D>) -> TracedRecovery<Prim<S, D>, S::Mask>
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let accepted = gas_accepted(prim.rho(), Some(prim.pre()), prim.vel());
    TracedRecovery::certify(prim, accepted)
}

/// the isothermal family: the gas interior with the pressure slot excluded —
/// the substrate candidate carries a materialized closure pressure
/// (`cs^2 rho`), and its validity is the density's.
pub fn isothermal<S, const D: usize>(prim: Prim<S, D>) -> TracedRecovery<Prim<S, D>, S::Mask>
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let accepted = gas_accepted(prim.rho(), None, prim.vel());
    TracedRecovery::certify(prim, accepted)
}

/// the newtonian-MHD family: the newtonian interior plus finite cell-B.
pub fn newtonian_mhd<S, const D: usize>(
    prim: MhdPrim<S, D>,
) -> TracedRecovery<MhdPrim<S, D>, S::Mask>
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let gas = gas_accepted(prim.rho(), Some(prim.pre()), prim.vel());
    let accepted = tensor_accepted::<S, D>(gas, prim.mag());
    TracedRecovery::certify(prim, accepted)
}

/// the isothermal-MHD family: the isothermal interior plus finite cell-B.
pub fn isothermal_mhd<S, const D: usize>(
    prim: IsoMhdPrim<S, D>,
) -> TracedRecovery<IsoMhdPrim<S, D>, S::Mask>
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let gas = gas_accepted(prim.rho(), None, prim.vel());
    let accepted = tensor_accepted::<S, D>(gas, prim.mag());
    TracedRecovery::certify(prim, accepted)
}

/// the relativistic family: the newtonian interior plus a subluminal metric
/// velocity norm, `gamma_ij v^i v^j < 1` (identity gamma on a flat chart —
/// the contraction constant-folds to the euclidean norm). a NaN norm rejects
/// through the comparison.
pub fn relativistic<S, const D: usize>(
    prim: Prim<S, D>,
    metric: &SpatialMetric<S, D>,
) -> TracedRecovery<Prim<S, D>, S::Mask>
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let gas = gas_accepted(prim.rho(), Some(prim.pre()), prim.vel());
    let v_sq = metric.norm_sq_contra(prim.vel());
    let accepted = gas & v_sq.cmp_lt(S::ONE);
    TracedRecovery::certify(prim, accepted)
}

/// the relativistic-MHD family: the relativistic interior plus finite cell-B.
pub fn relativistic_mhd<S, const D: usize>(
    prim: MhdPrim<S, D>,
    metric: &SpatialMetric<S, D>,
) -> TracedRecovery<MhdPrim<S, D>, S::Mask>
where
    S: Scalar,
    S::Mask: BitAnd<Output = S::Mask>,
{
    let gas = gas_accepted(prim.rho(), Some(prim.pre()), prim.vel());
    let v_sq = metric.norm_sq_contra(prim.vel());
    let sublum = gas & v_sq.cmp_lt(S::ONE);
    let accepted = tensor_accepted::<S, D>(sublum, prim.mag());
    TracedRecovery::certify(prim, accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantity::{Density, Pressure};

    fn prim1(rho: f64, v: f64, pre: f64) -> Prim<f64, 1> {
        Prim::adiabatic(Density(rho), Tensor::new([v]), Pressure(pre))
    }

    /// at the host carrier each family door is the boolean interior law and
    /// the status describes the captured candidate.
    #[test]
    fn family_doors_certify_their_own_candidate() {
        let (_, s) = newtonian(prim1(1.0, 0.5, 0.2)).into_parts();
        assert!(s.accepted());
        let (_, s) = newtonian(prim1(-1.0, 0.5, 0.2)).into_parts();
        assert!(!s.accepted());
        let (_, s) = newtonian(prim1(1.0, 0.5, 0.0)).into_parts();
        assert!(!s.accepted());
        let (_, s) = newtonian(prim1(1.0, f64::INFINITY, 0.2)).into_parts();
        assert!(!s.accepted());

        let (_, s) = isothermal(prim1(1.0, 0.5, 0.0)).into_parts();
        assert!(s.accepted(), "iso ignores the pressure slot");
        let (_, s) = isothermal(prim1(0.0, f64::INFINITY, 0.0)).into_parts();
        assert!(!s.accepted());
    }

    /// the relativistic door adds the subluminal norm; the magnetized doors
    /// add cell-B finiteness.
    #[test]
    fn family_doors_carry_their_extra_conditions() {
        let flat = SpatialMetric::flat();
        let (_, s) = relativistic(prim1(1.0, 0.5, 0.2), &flat).into_parts();
        assert!(s.accepted());
        let (_, s) = relativistic(prim1(1.0, 1.5, 0.2), &flat).into_parts();
        assert!(!s.accepted(), "superluminal norm rejects");

        let mhd = MhdPrim {
            hydro: prim1(1.0, 0.5, 0.2),
            mag: Tensor::new([f64::NAN]),
        };
        let (_, s) = newtonian_mhd(mhd).into_parts();
        assert!(!s.accepted(), "non-finite cell-B rejects");
        let (_, s) = relativistic_mhd(mhd, &flat).into_parts();
        assert!(!s.accepted());
    }
}
