// =============================================================================
// rmhd/algebra.rs
//
// the RMHD elemental quantities — pure pointwise functions of the primitive
// MHD state, shared across the flux, the geometric source, and the solvers:
//   magnetic_pressure, total_pressure, the spatial magnetic four-vector, and
//   the curvilinear geometric-source quantities (rho h W^2, b^i, p_tot).
// carrier-generic (S: Scalar) single source.
// =============================================================================

use crate::eos::Eos;
use crate::mhd_state::MhdPrim;
use crate::rhd;
use crate::spatial_metric::SpatialMetric;
use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// magnetic pressure p_mag = 0.5*(B^2/W^2 + (v . B)^2) (= 0.5 * b^mu b_mu spatial).
#[inline]
pub fn magnetic_pressure<S: Scalar, const D: usize>(
    prim: &MhdPrim<S, D>,
    metric: &SpatialMetric<S, D>,
) -> S {
    let bsq = metric.norm_sq_contra(&prim.mag); // |B|^2 = gamma_{ij} B^i B^j
    let vsq = metric.norm_sq_contra(&prim.vel); // |v|^2
    let vdb = metric.contract_contra(&prim.vel, &prim.mag); // v.B = gamma_{ij} v^i B^j
    let w_sq = S::ONE / (S::ONE - vsq);
    S::from_f64(0.5) * (bsq / w_sq + vdb * vdb)
}

/// total pressure: p_gas + p_mag.
#[inline]
pub fn total_pressure<S: Scalar, const D: usize>(
    prim: &MhdPrim<S, D>,
    metric: &SpatialMetric<S, D>,
) -> S {
    prim.pre + magnetic_pressure(prim, metric)
}

/// the SPATIAL magnetic four-vector b^i = B^i / W + v^i W (v.B) — the spatial part of the
/// covariant four-vector b^mu. the SINGLE source for both the RMHD flux's magnetic tension
/// and the curvilinear geometric source's stress tension (T^{jk} = ... - b^j b^k).
#[inline]
pub fn magnetic_four_vector_spatial<S: Scalar, const D: usize>(
    prim: &MhdPrim<S, D>,
    metric: &SpatialMetric<S, D>,
) -> Tensor<S, D> {
    let ww = rhd::lorentz_factor(metric.norm_sq_contra(&prim.vel));
    let vdb = metric.contract_contra(&prim.vel, &prim.mag);
    prim.mag.scale(S::ONE / ww) + prim.vel.scale(ww * vdb)
}

/// the RMHD curvilinear geometric-source quantities: the GAS momentum density `rho h W^2`,
/// the spatial magnetic four-vector `b^i`, and the total pressure `p_tot`. these are the
/// regime-specific pieces of the relativistic-MHD stress `T^{jk} = (rho h W^2) v^j v^k +
/// p_tot gamma^{jk} - b^j b^k` that the substrate contracts with the Christoffels
/// (`S^i = -Gamma^i_{jk} T^{jk}`). RMHD needs its own quantities (unlike hydro/RHD, whose
/// `cons.mom` IS rho h W^2 v) because the RMHD `cons.mom` also carries B-momentum. the
/// carrier-generic single source for the substrate `rmhd_geometric_momentum_sources`.
#[inline]
pub fn rmhd_source_quantities<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &MhdPrim<S, D>,
    metric: &SpatialMetric<S, D>,
) -> (S, Tensor<S, D>, S) {
    let v_sq = metric.norm_sq_contra(&prim.vel);
    let wgam2 = rhd::enthalpy_density(eos, prim.rho, prim.pre, v_sq); // rho h W^2
    (
        wgam2,
        magnetic_four_vector_spatial(prim, metric),
        total_pressure(prim, metric),
    )
}
