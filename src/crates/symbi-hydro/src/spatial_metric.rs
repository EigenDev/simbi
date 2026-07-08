// =============================================================================
// spatial_metric.rs
//
// the spatial metric as a CARRIER-GENERIC VALUE the physics contracts with — the
// vehicle for the SR->GR generalization. the carrier homomorphism transports it exactly
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

/// the COVARIANT spatial metric gamma_{ij} — LOWERS a contravariant index. a DISTINCT type from
/// its inverse [`GammaInv`] so a builder cannot silently pass one where the other is meant (the
/// contravariant/covariant trap): the two matrices are numerically interchangeable but physically
/// opposite, and swapping them is invisible in a flat frame (both identity) yet a sign/scale bug on
/// a curved metric. the tag is asserted ONCE at construction; every contraction downstream carries
/// it by type.
#[derive(Clone, Copy, Debug)]
pub struct Gamma<S: Scalar, const D: usize>(Matrix<S, D>);

impl<S: Scalar, const D: usize> Gamma<S, D> {
    /// wrap a raw matrix ASSERTING it is the covariant metric gamma_{ij}. the single point where
    /// the covariant tag enters the type system.
    pub fn new(m: Matrix<S, D>) -> Self {
        Self(m)
    }
    /// the flat covariant metric (identity): lower is the identity map.
    pub fn identity() -> Self {
        Self(Matrix::identity())
    }
    /// lower a CONTRAVARIANT vector: `v_i = gamma_{ij} v^j`.
    pub fn lower(&self, v: &Tensor<S, D>) -> Tensor<S, D> {
        self.0.mul_vec(v)
    }
    /// the squared norm of a CONTRAVARIANT vector: `gamma_{ij} v^i v^j`.
    pub fn quadratic_contra(&self, v: &Tensor<S, D>) -> S {
        self.0.quadratic(v)
    }
    /// contract two CONTRAVARIANT vectors: `gamma_{ij} v^i w^j`.
    pub fn contract_contra(&self, v: &Tensor<S, D>, w: &Tensor<S, D>) -> S {
        self.0.contract(v, w)
    }
}

/// the CONTRAVARIANT spatial metric gamma^{ij} — RAISES a covariant index. the distinct-type
/// counterpart of [`Gamma`]; see it for the covariant/contravariant swap rationale.
#[derive(Clone, Copy, Debug)]
pub struct GammaInv<S: Scalar, const D: usize>(Matrix<S, D>);

impl<S: Scalar, const D: usize> GammaInv<S, D> {
    /// wrap a raw matrix ASSERTING it is the contravariant (inverse) metric gamma^{ij}.
    pub fn new(m: Matrix<S, D>) -> Self {
        Self(m)
    }
    /// the flat contravariant metric (identity): raise is the identity map.
    pub fn identity() -> Self {
        Self(Matrix::identity())
    }
    /// raise a COVARIANT vector: `w^i = gamma^{ij} w_j`.
    pub fn raise(&self, w: &Tensor<S, D>) -> Tensor<S, D> {
        self.0.mul_vec(w)
    }
    /// the squared norm of a COVARIANT vector: `gamma^{ij} w_i w_j`.
    pub fn quadratic_cov(&self, w: &Tensor<S, D>) -> S {
        self.0.quadratic(w)
    }
    /// the diagonal component gamma^{nn} — the coordinate light-cone speed is `alpha sqrt(gamma^{nn})`.
    pub fn diag(&self, n: usize) -> S {
        self.0[(n, n)]
    }
}

/// the spatial metric gamma_{ij} and its inverse gamma^{ij} at a cell, as a
/// carrier-generic value. flat/orthonormal -> both identity (bit-identical to the
/// euclidean inner product). the fields are DISTINCT TYPES ([`Gamma`] / [`GammaInv`]) so the
/// covariant/contravariant metric cannot be swapped at construction:
///
/// ```compile_fail
/// use symbi_hydro::spatial_metric::{SpatialMetric, Gamma, GammaInv};
/// use symbi_algebra::Matrix;
/// let g = Gamma::<f64, 3>::new(Matrix::identity());
/// let gi = GammaInv::<f64, 3>::new(Matrix::identity());
/// let _m = SpatialMetric::new(gi, g); // swapped -> covariant where contravariant is required
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpatialMetric<S: Scalar, const D: usize> {
    /// gamma_{ij} — lowers a CONTRAVARIANT vector v^i.
    pub gamma: Gamma<S, D>,
    /// gamma^{ij} — raises a COVARIANT vector w_i.
    pub gamma_inv: GammaInv<S, D>,
}

impl<S: Scalar, const D: usize> SpatialMetric<S, D> {
    /// assemble from the tagged covariant + contravariant metrics. the argument TYPES enforce the
    /// order — passing the inverse where the metric is expected fails to compile.
    pub fn new(gamma: Gamma<S, D>, gamma_inv: GammaInv<S, D>) -> Self {
        Self { gamma, gamma_inv }
    }

    /// the flat / orthonormal-frame metric: gamma = gamma_inv = identity. every norm
    /// reduces to the euclidean `.dot()` bit-identically — the SR / curvilinear-flat case.
    pub fn flat() -> Self {
        Self { gamma: Gamma::identity(), gamma_inv: GammaInv::identity() }
    }

    /// the squared norm of a COVARIANT vector w_i: `gamma^{ij} w_i w_j` (raise + contract).
    /// e.g. the conserved-momentum magnitude `|S|^2`.
    pub fn norm_sq_cov(&self, w: &Tensor<S, D>) -> S {
        self.gamma_inv.quadratic_cov(w)
    }

    /// the squared norm of a CONTRAVARIANT vector v^i: `gamma_{ij} v^i v^j` (lower + contract).
    /// e.g. the velocity-squared `|v|^2` feeding the Lorentz factor.
    pub fn norm_sq_contra(&self, v: &Tensor<S, D>) -> S {
        self.gamma.quadratic_contra(v)
    }

    /// the inner product of two CONTRAVARIANT vectors v^i, w^i: `gamma_{ij} v^i w^j` (lower one,
    /// contract). e.g. `v.B` in the magnetic four-vector, or `B.n` on a flux interface.
    pub fn contract_contra(&self, v: &Tensor<S, D>, w: &Tensor<S, D>) -> S {
        self.gamma.contract_contra(v, w)
    }

    /// decompose a CONTRAVARIANT vector v^i into its part along a COORDINATE-unit normal n^i
    /// (n^i = delta^i_dir) and the remainder: `v^i - n^i (v.n)` with the CONTRAVARIANT normal
    /// component `v.n = v^dir` (`.dot(nhat)`), NOT the gamma-lowered `v_dir`. this is the
    /// decomposition the coordinate induction/MHD flux uses (F(B^i) = B^i v^n - v^i B^n with
    /// v^n = v^dir, B^n = B^dir), so the transverse remainder is flux-consistent on ANY spatial
    /// metric — the HLLD fan's transverse fields telescope to the flux exactly. flat / orthonormal
    /// -> the euclidean `v - n (v.n)` bit-identically.
    pub fn project_transverse(&self, v: &Tensor<S, D>, nhat: &Tensor<S, D>) -> Tensor<S, D> {
        *v - nhat.scale(v.dot(nhat))
    }

    /// RAISE a COVARIANT vector w_i to its CONTRAVARIANT form `w^i = gamma^{ij} w_j`. flat /
    /// orthonormal -> identity -> `w` bit-identically (the optimizer folds the identity mul). the
    /// Valencia c2p recovers the CONTRAVARIANT velocity `v^i = gamma^{ij} S_j / (D + tau + p)` from
    /// the covariant conserved momentum `S_j` this way.
    pub fn raise(&self, w: &Tensor<S, D>) -> Tensor<S, D> {
        self.gamma_inv.raise(w)
    }

    /// LOWER a CONTRAVARIANT vector v^i to its COVARIANT form `v_i = gamma_{ij} v^j`. flat ->
    /// identity -> `v` bit-identically. the Valencia conserved momentum is the LOWERED velocity:
    /// `S_i = rho h W^2 v_i = rho h W^2 gamma_{ij} v^j`.
    pub fn lower(&self, v: &Tensor<S, D>) -> Tensor<S, D> {
        self.gamma.lower(v)
    }

    /// the ORTHONORMAL basis matrix E for a Riemann solve along coordinate axis `dir`: the columns
    /// are the orthonormal frame vectors e_hat_a (contravariant, in coordinate components) from a
    /// GRAM-SCHMIDT of the coordinate basis with respect to gamma, processing the TRANSVERSE axes
    /// FIRST and the normal `dir` LAST. because each transverse e_hat lives purely in the transverse
    /// coordinate plane (zero `dir` component), the normal maps cleanly: v^dir = E[dir][dir]
    /// V_hat^dir, so a contravariant/covariant/flux quantity transforms to and from the frame where
    /// the metric is the identity by E (and E^{-1}) with the single normal factor E[dir][dir]. this
    /// is the tetrad that reduces the GR MUB09 solve to the flat solver on ANY symmetric-positive
    /// spatial metric (diagonal Schwarzschild/KS -> E = diag(1/sqrt(gamma_ii)); non-diagonal Kerr ->
    /// the full triangular tetrad). the frame is: V_hat = E^{-1} v, B_hat = E^{-1} B.
    pub fn orthonormal_basis(&self, dir: usize) -> Matrix<S, D> {
        let one = S::ONE;
        // gram-schmidt order: every transverse axis in index order, then the normal last.
        let mut order = [0usize; D];
        let mut n = 0;
        for a in 0..D {
            if a != dir {
                order[n] = a;
                n += 1;
            }
        }
        order[D - 1] = dir;
        let mut cols: [Tensor<S, D>; D] = [Tensor::zeros(); D];
        for step in 0..D {
            let a = order[step];
            let mut u = Tensor::<S, D>::unit(a);
            for prev in 0..step {
                let b = order[prev];
                // subtract the gamma-projection onto the already-orthonormalized column b.
                let proj = self.contract_contra(&u, &cols[b]);
                u = u - cols[b].scale(proj);
            }
            let inv_norm = one / self.norm_sq_contra(&u).sqrt();
            cols[a] = u.scale(inv_norm);
        }
        Matrix::from_fn(|i, j| cols[j].data[i])
    }
}
