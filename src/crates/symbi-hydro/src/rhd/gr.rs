// =============================================================================
// rhd/gr.rs
//
// `RhdGr` — the relativistic-hydro regime on a curved SPATIAL metric: the VALENCIA form
// (COVARIANT conserved momentum `S_i = rho h W^2 gamma_ij v^j`) with the Banyuls-Font COORDINATE
// wave speeds. it is the flat `Rhd` regime with three metric substitutions, and REDUCES to `Rhd`
// bit-for-bit at identity gamma + lapse 1 -> flat SR is untouched. carries the spatial metric
// gamma/gamma^{-1} (via `SpatialMetric`) + the lapse `alpha`.
//
// the GR flux kernel (the `_schw`/`_ks` bake) uses this via `riemann::hlle_with_speeds`, so GR is a
// DIVERGENT KERNEL, not a change to the shared flat `Rhd` — the same principle the densitization +
// shift-flux kernels already follow. `prim.vel` is the CONTRAVARIANT velocity `v^i` (the valencia
// velocity; = the physical V under identity gamma).
// =============================================================================

use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_ir::algebra::Scalar;

use crate::c2p_result::C2pResult;
use crate::eos::Eos;
use crate::regime::Regime;
use crate::regime_spec::RegimeSpec;
use crate::spatial_metric::SpatialMetric;
use crate::state::{Cons, Prim};

use super::cons::rhd_recover;
use super::wave_speeds::rhd_speeds_from_vn_gr;
use super::{enthalpy, lorentz_factor, sound_speed_sq, Rhd};

/// the maximum newton iterations for the metric-aware c2p (mirrors the flat host wrapper).
const MAX_ITER: usize = 100;

/// the relativistic-hydro regime on a curved spatial metric (Valencia covariant momentum + BF wave
/// speeds). `metric` = gamma_{ij}/gamma^{ij}; `alpha` = the lapse. reduces to `Rhd` at identity/1.
#[derive(Clone, Copy)]
pub struct RhdGr<S: Scalar, const D: usize> {
    pub metric: SpatialMetric<S, D>,
    pub alpha: S,
}

impl<S: Scalar, const D: usize> Regime<S, D> for RhdGr<S, D> {
    const SPEC: &'static RegimeSpec = <Rhd as Regime<S, D>>::SPEC;
    type Prim = Prim<S, D>;
    type Cons = Cons<S, D>;
    type Energy = crate::energy::Adiabatic;
    const CLAMP_EXTREMAL_TO_ZERO: bool = true;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        // Valencia: |v|^2 = gamma_ij v^i v^j (v^i CONTRAVARIANT); S_i = rho h W^2 gamma_ij v^j (LOWER).
        // identity gamma -> euclidean v.v + orthonormal S = rho h W^2 v (bit-identical to `Rhd`).
        let v_sq = self.metric.norm_sq_contra(&prim.vel);
        let ww = lorentz_factor(v_sq);
        let hh = enthalpy(eos, prim.rho, prim.pre);
        let den = prim.rho * ww;
        let rhw2 = prim.rho * hh * ww * ww;
        let mom = self.metric.lower(&prim.vel).scale(rhw2);
        let nrg = rhw2 - prim.pre - den;
        Cons { den, mom, nrg }
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        let dd = cons.den;
        if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
            let floored = Prim {
                rho: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
                vel: Tensor::zeros(),
                pre: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
            };
            return C2pResult::err(floored, code);
        }
        // the metric-aware recovery: |S|^2 = gamma^{ij} S_i S_j, then the raised v^i (`rhd_recover`
        // already contracts with `self.metric`). the SR->GR difference is the metric VALUE, not new code.
        let prim = rhd_recover(eos, cons, &self.metric, MAX_ITER);
        let v_sq = self.metric.norm_sq_contra(&prim.vel);
        let code = crate::c2p_result::relativistic_c2p_code(prim.rho, prim.pre, v_sq);
        if code.is_ok() { C2pResult::ok(prim) } else { C2pResult::err(prim, code) }
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
        // the Valencia SPATIAL flux (no shift; the shift-advection rides the discretize shift-flux
        // kernel): F_D = D v^n, F_{S_j} = S_j v^n + p n_j, F_tau = (tau + p) v^n. with v^n = v^i n_i
        // (CONTRAVARIANT normal for a coordinate-unit nhat). identity gamma + the covariant S from
        // to_conserved -> bit-identical to `Rhd::to_flux` ((tau+p)v^n == mn - D v^n at identity).
        let cons = self.to_conserved(eos, prim);
        let vn = prim.vel.dot(nhat);
        Cons {
            den: cons.den * vn,
            mom: cons.mom.scale(vn) + nhat.scale(prim.pre),
            nrg: (cons.nrg + prim.pre) * vn,
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        // Banyuls-Font coordinate speed (Font 2008 eq 37). the two velocities are DISTINCT and must
        // NOT be conflated: `vn` = the CONTRAVARIANT normal v^n = v^i n_i (the transport term + inside
        // the radical), `v_sq` = gamma_ij v^i v^j = the PHYSICAL speed squared (the `1 - v^2` factors).
        // feeding the physical velocity sqrt(gamma_nn) v^n to BOTH slots drove the discriminant
        // negative (NaN Riemann fan) once |v| approached alpha near the horizon — the gr_bondi crash.
        // gamma^{nn} is the inverse-metric normal. identity gamma + alpha 1 (v_sq = vn^2) -> flat.
        let cs_sq = sound_speed_sq(eos, prim.rho, prim.pre);
        let vn = prim.vel.dot(nhat); // contravariant v^n
        let v_sq = self.metric.norm_sq_contra(&prim.vel); // gamma_ij v^i v^j (physical norm)
        let gamma_nn_inv = self.metric.norm_sq_cov(nhat); // gamma^{nn} (coordinate-unit normal)
        rhd_speeds_from_vn_gr(cs_sq, vn, v_sq, gamma_nn_inv, self.alpha)
    }

    #[inline]
    fn effective_inertia(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        // rho h W^2 with the metric Lorentz factor |v|^2 = gamma_ij v^i v^j.
        crate::rhd::enthalpy_density(eos, prim.rho, prim.pre, self.metric.norm_sq_contra(&prim.vel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use crate::spatial_metric::{Gamma, GammaInv};
    use symbi_algebra::Matrix;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn rhd_gr_reduces_to_flat_at_identity() {
        // RhdGr on the flat metric (gamma = identity, alpha = 1) MUST equal the flat Rhd regime,
        // component-for-component: conserved, flux, AND wave speeds. the flat path stays untouched.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let flat = Rhd;
        let gr = RhdGr::<f64, 1> { metric: SpatialMetric::flat(), alpha: 1.0 };
        let nhat = Tensor::unit(0);
        for &v in &[0.0_f64, 0.3, -0.5, 0.85] {
            let prim = Prim { rho: 1.3, vel: Tensor::new([v]), pre: 0.5 };
            let (cf, cg) = (flat.to_conserved(&eos, &prim), gr.to_conserved(&eos, &prim));
            assert!(approx(cf.den, cg.den) && approx(cf.mom[0], cg.mom[0]) && approx(cf.nrg, cg.nrg), "cons v={v}");
            let (ff, fg) = (flat.to_flux(&prim, &nhat, &eos), gr.to_flux(&prim, &nhat, &eos));
            assert!(approx(ff.den, fg.den) && approx(ff.mom[0], fg.mom[0]) && approx(ff.nrg, fg.nrg), "flux v={v}");
            let ((slf, srf), (slg, srg)) = (flat.wave_speeds(&eos, &prim, &nhat), gr.wave_speeds(&eos, &prim, &nhat));
            assert!(approx(slf, slg) && approx(srf, srg), "speeds v={v}");
        }
    }

    #[test]
    fn rhd_gr_stores_covariant_momentum_on_schwarzschild() {
        // Schwarzschild r=10, M=1: f = 0.8, gamma_rr = 1/f = 1.25, gamma^{rr} = f. the covariant
        // momentum S_r = rho h W^2 * gamma_rr * v^r (v^r CONTRAVARIANT); W from |v|^2 = gamma_rr (v^r)^2.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let (f, vr) = (0.8_f64, 0.2_f64);
        let grr = 1.0 / f;
        let metric = SpatialMetric::<f64, 1>::new(
            Gamma::new(Matrix::diag(Tensor::new([grr]))),
            GammaInv::new(Matrix::diag(Tensor::new([f]))),
        );
        let gr = RhdGr { metric, alpha: f.sqrt() };
        let prim = Prim { rho: 1.0, vel: Tensor::new([vr]), pre: 0.1 };
        let c = gr.to_conserved(&eos, &prim);
        let v_sq = grr * vr * vr;
        let w = 1.0 / (1.0 - v_sq).sqrt();
        let h = 1.0 + (4.0 / 3.0) / (4.0 / 3.0 - 1.0) * 0.1 / 1.0; // 1 + Gamma/(Gamma-1) p/rho
        let rhw2 = 1.0 * h * w * w;
        assert!(approx(c.den, w), "D = rho W");
        assert!(approx(c.mom[0], grr * vr * rhw2), "S_r covariant = gamma_rr v^r rho h W^2");
        // and the c2p round-trips it back to the same contravariant v^r
        let back = rhd_recover(&eos, &c, &gr.metric, MAX_ITER);
        assert!(approx(back.vel[0], vr), "c2p recovers contravariant v^r");
    }
}
