// =============================================================================
// spatial_metric.rs
//
// the spatial metric as a CARRIER-GENERIC VALUE the physics contracts with — the
// vehicle for the SR->GR pivot (B2). the carrier homomorphism transports it exactly
// like `eos`: at S=f64 it is concrete matrices, at S=Gv it traces into the kernel.
//
// holds gamma_{ij} (lowers a contravariant vector) and its inverse gamma^{ij}
// (raises a covariant one). the norm helpers NAME the variance of their input, so a
// caller cannot silently contract a covariant momentum with the wrong metric — a
// lightweight guard for the contravariant/covariant trap (the recovered v^i raised
// with gamma^{ij}, then re-contracted for the Lorentz norm with gamma_{ij}).
//
// flat / orthonormal frame: gamma = gamma_inv = identity, so every norm is the
// euclidean `.dot()` BIT-IDENTICALLY (the optimizer folds the identity). a genuine
// (GR) metric makes them the curved norms — no call site changes, only the value.
//
// usage:
//  let g = SpatialMetric::flat();              // SR / orthonormal -> identity
//  let s_sq = g.norm_sq_cov(&cons.mom);        // |S|^2 = gamma^{ij} S_i S_j
//  let v_sq = g.norm_sq_contra(&prim.vel);     // |v|^2 = gamma_{ij} v^i v^j
// =============================================================================

use symbi_algebra::{Matrix, Tensor};
use symbi_ir::algebra::Scalar;

/// the spatial metric gamma_{ij} and its inverse gamma^{ij} at a cell, as a
/// carrier-generic value. flat/orthonormal -> both identity (bit-identical to the
/// euclidean inner product).
#[derive(Clone, Copy, Debug)]
pub struct SpatialMetric<S: Scalar, const D: usize> {
    /// gamma_{ij} — contracts (lowers) a CONTRAVARIANT vector v^i.
    pub gamma: Matrix<S, D>,
    /// gamma^{ij} — contracts (raises) a COVARIANT vector w_i.
    pub gamma_inv: Matrix<S, D>,
}

impl<S: Scalar, const D: usize> SpatialMetric<S, D> {
    /// the flat / orthonormal-frame metric: gamma = gamma_inv = identity. every norm
    /// reduces to the euclidean `.dot()` bit-identically — the SR / curvilinear-flat case.
    pub fn flat() -> Self {
        Self { gamma: Matrix::identity(), gamma_inv: Matrix::identity() }
    }

    /// the squared norm of a COVARIANT vector w_i: `gamma^{ij} w_i w_j` (raise + contract).
    /// e.g. the conserved-momentum magnitude `|S|^2`.
    pub fn norm_sq_cov(&self, w: &Tensor<S, D>) -> S {
        self.gamma_inv.quadratic(w)
    }

    /// the squared norm of a CONTRAVARIANT vector v^i: `gamma_{ij} v^i v^j` (lower + contract).
    /// e.g. the velocity-squared `|v|^2` feeding the Lorentz factor.
    pub fn norm_sq_contra(&self, v: &Tensor<S, D>) -> S {
        self.gamma.quadratic(v)
    }
}
