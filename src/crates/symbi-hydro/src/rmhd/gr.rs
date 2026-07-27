// =============================================================================
// rmhd/gr.rs
//
// `RmhdGr` — the relativistic-MHD regime on a curved SPATIAL metric: the VALENCIA form
// (COVARIANT conserved momentum `S_i = (rho h W^2 + B^2) v_i - (v.B) B_i`) with the
// fast-magnetosonic BOUND through the Banyuls-Font coordinate wave-speed transform. it
// is the flat `Rmhd` regime with the metric threaded through every contraction, and
// REDUCES to `Rmhd` at identity gamma + lapse 1 -> flat SRMHD is untouched. carries
// gamma/gamma^{-1} (via `SpatialMetric`) + the lapse `alpha`, exactly like `RhdGr`.
//
// the GR flux kernel uses this via `riemann::hlle_with_speeds` + the shift-in-the-fan,
// so GRMHD is a DIVERGENT KERNEL that leaves the shared flat `Rmhd` untouched. `prim.vel` is
// the CONTRAVARIANT valencia velocity `v^i`; `prim.mag`/`cons.mag` the CONTRAVARIANT
// eulerian field `B^i`.
//
// wave speeds: the flat Mignone & Del Zanna magnetosonic quartic is a flat-frame
// construct; the GR fan uses the fast-speed BOUND c_ms^2 = c_s^2 + v_A^2 - c_s^2 v_A^2
// (v_A^2 = b^2/(rho h + b^2)) in the same two-velocity Banyuls-Font form the RHD fan
// uses — unconditionally OUTSIDE the true fan, so HLL stays consistent (at worst mildly
// more diffusive than the exact quartic).
// =============================================================================

use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_ir::algebra::Scalar;

use crate::c2p_result::C2pResult;
use crate::eos::Eos;
use crate::mhd_state::{MhdCons, MhdPrim};
use crate::regime::Regime;
use crate::regime_spec::RegimeSpec;
use crate::rhd::{enthalpy, lorentz_factor, lorentz_factor_sq, sound_speed_sq};
use crate::spatial_metric::SpatialMetric;
use crate::state::{Cons, Prim};

use super::Rmhd;
use super::algebra::{magnetic_four_vector_spatial, total_pressure};
use super::cons::rmhd_recover;

/// the maximum KKC false-position iterations (mirrors the flat host wrapper).
const MAX_ITER: usize = 100;

/// the relativistic-MHD regime on a curved spatial metric (Valencia covariant momentum +
/// the BF-transformed fast-speed bound). `metric` = gamma_{ij}/gamma^{ij}; `alpha` = the
/// lapse. reduces to `Rmhd` at identity/1.
#[derive(Clone, Copy)]
pub struct RmhdGr<S: Scalar, const D: usize> {
    pub metric: SpatialMetric<S, D>,
    pub alpha: S,
}

impl<S: Scalar, const D: usize> RmhdGr<S, D> {
    /// the frame-local magnetic invariants at a prim state: (B^2, v.B, b^2) with
    /// B^2 = gamma_ij B^i B^j, v.B = gamma_ij v^i B^j, b^2 = B^2/W^2 + (v.B)^2 (the
    /// four-vector norm b_mu b^mu).
    #[inline]
    fn mag_invariants(&self, prim: &MhdPrim<S, D>, w_sq: S) -> (S, S, S) {
        let bsq = self.metric.norm_sq_contra(&prim.mag);
        let vdb = self.metric.contract_contra(&prim.vel, &prim.mag);
        (bsq, vdb, bsq / w_sq + vdb * vdb)
    }
}

impl<S: Scalar, const D: usize> Regime<S, D> for RmhdGr<S, D> {
    const SPEC: &'static RegimeSpec = <Rmhd as Regime<S, D>>::SPEC;
    // relativistic: clamp the HLLE fan to include the stationary state (as `Rmhd`).
    const CLAMP_EXTREMAL_TO_ZERO: bool = true;
    type Prim = MhdPrim<S, D>;
    type Cons = MhdCons<S, D>;
    type Energy = crate::energy::Adiabatic;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        // Valencia: S_i = (rho h W^2 + B^2) v_i - (v.B) B_i with v_i/B_i LOWERED;
        // tau = rho h W^2 + B^2 - (p + b^2/2) - D. identity gamma -> lower = id, every
        // contraction euclidean: the flat `Rmhd::to_conserved` term-for-term.
        let vsq = self.metric.norm_sq_contra(&prim.vel);
        let ww = lorentz_factor(vsq);
        let w_sq = lorentz_factor_sq(vsq);
        let hh = enthalpy(eos, prim.rho, prim.pre);
        let (bsq, vdb, b_mu_sq) = self.mag_invariants(prim, w_sq);

        let den = prim.rho * ww;
        let rhw2 = prim.rho * hh * w_sq;
        let mom_fac = rhw2 + bsq;
        let mom =
            self.metric.lower(&prim.vel).scale(mom_fac) - self.metric.lower(&prim.mag).scale(vdb);
        let half = S::from_f64(0.5);
        let p_tot = prim.pre + half * b_mu_sq;
        let nrg = rhw2 + bsq - p_tot - den;

        MhdCons {
            hydro: Cons { den, mom, nrg },
            mag: prim.mag,
        }
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        let dd = cons.den;
        if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
            let floored = MhdPrim {
                hydro: Prim {
                    rho: S::from_f64(crate::c2p_result::C2P_FAILURE_SENTINEL),
                    vel: Tensor::zeros(),
                    pre: S::from_f64(crate::c2p_result::C2P_FAILURE_SENTINEL),
                },
                mag: cons.mag,
            };
            return C2pResult::err(floored, code);
        }
        // the metric-aware KKC recovery: the invariants r^2/B^2/r.B form with gamma, the
        // recovered v^i is contravariant. the SR->GR difference is the metric VALUE.
        let prim = rmhd_recover(eos, cons, &self.metric, MAX_ITER);
        let v_sq = self.metric.norm_sq_contra(&prim.vel);
        let code = crate::c2p_result::relativistic_c2p_code(prim.rho, prim.pre, v_sq);
        if code.is_ok() {
            C2pResult::ok(prim)
        } else {
            C2pResult::err(prim, code)
        }
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
        // the Valencia SPATIAL flux (no shift; the shift rides the GR flux kernel's fan):
        //   F_D     = D v^n
        //   F_{S_j} = S_j v^n + p_tot n_j - b_j B^n / W      (b_j LOWERED four-vector)
        //   F_tau   = (tau + p_tot) v^n - (v.B) B^n          ((v.B) B^n = alpha b^0 B^n / W)
        //   F_{B^i} = B^i v^n - v^i B^n                      (contravariant, metric-free)
        // v^n = v.nhat / B^n = B.nhat are coordinate COMPONENTS (coordinate-unit nhat).
        // identity gamma -> the flat `Rmhd::to_flux` (its tau flux `S.n - D v^n` equals
        // this form by the flat momentum identity).
        let cons = self.to_conserved(eos, prim);
        let vn = prim.vel.dot(nhat);
        let bn = prim.mag.dot(nhat);
        let vsq = self.metric.norm_sq_contra(&prim.vel);
        let ww = lorentz_factor(vsq);
        let vdb = self.metric.contract_contra(&prim.vel, &prim.mag);
        let b_lo = self
            .metric
            .lower(&magnetic_four_vector_spatial(prim, &self.metric));
        let p_tot = total_pressure(prim, &self.metric);

        MhdCons {
            hydro: Cons {
                den: cons.den * vn,
                mom: cons.mom.scale(vn) + nhat.scale(p_tot) - b_lo.scale(bn / ww),
                nrg: (cons.nrg + p_tot) * vn - vdb * bn,
            },
            mag: prim.mag.scale(vn) - prim.vel.scale(bn),
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        // the fast-magnetosonic bound c_ms^2 = c_s^2 + v_A^2 - c_s^2 v_A^2 through the
        // Banyuls-Font two-velocity transform (vn = contravariant v^n; v_sq = the physical
        // norm; gamma^{nn} the inverse-metric normal) — the RHD fan with c_ms for c_s.
        let cs_sq = sound_speed_sq(eos, prim.rho, prim.pre);
        let vsq = self.metric.norm_sq_contra(&prim.vel);
        let w_sq = lorentz_factor_sq(vsq);
        let (_bsq, _vdb, b_mu_sq) = self.mag_invariants(prim, w_sq);
        let hh = enthalpy(eos, prim.rho, prim.pre);
        let va_sq = b_mu_sq / (prim.rho * hh + b_mu_sq);
        let cms_sq = cs_sq + va_sq - cs_sq * va_sq;
        let vn = prim.vel.dot(nhat);
        let gamma_nn = self.metric.norm_sq_cov(nhat);
        crate::rhd::rhd_speeds_from_vn_gr(cms_sq, vn, vsq, gamma_nn, self.alpha)
    }

    fn effective_inertia(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        // rho h W^2 + B^2 with metric contractions (the flat form's euclidean dots
        // replaced by gamma) — the momentum-density scale of the magnetized fluid.
        let vsq = self.metric.norm_sq_contra(&prim.vel);
        let w_sq = lorentz_factor_sq(vsq);
        let hh = enthalpy(eos, prim.rho, prim.pre);
        prim.rho * hh * w_sq + self.metric.norm_sq_contra(&prim.mag)
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

    /// a kerr-like non-diagonal SPD spatial metric (gamma_{r phi} != 0) with its exact inverse.
    fn curved_metric() -> SpatialMetric<f64, 3> {
        let gamma = Matrix::<f64, 3> {
            data: [[1.5, 0.0, -0.3], [0.0, 9.0, 0.0], [-0.3, 0.0, 4.0]],
        };
        let det = 1.5 * 4.0 - 0.3 * 0.3;
        let gamma_inv = Matrix::<f64, 3> {
            data: [
                [4.0 / det, 0.0, 0.3 / det],
                [0.0, 1.0 / 9.0, 0.0],
                [0.3 / det, 0.0, 1.5 / det],
            ],
        };
        SpatialMetric::new(Gamma::new(gamma), GammaInv::new(gamma_inv))
    }

    #[test]
    fn rmhd_gr_reduces_to_flat_at_identity() {
        // RmhdGr on the flat metric (gamma = identity, alpha = 1) MUST equal the flat Rmhd
        // regime component-for-component: conserved, flux, wave speeds, and the recovery.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let flat = Rmhd;
        let gr = RmhdGr::<f64, 3> {
            metric: SpatialMetric::flat(),
            alpha: 1.0,
        };
        let nhat = Tensor::unit(0);
        for &(v, b) in &[
            ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
            ([0.3, -0.1, 0.2], [0.4, 0.7, -0.2]),
            ([0.0, 0.6, 0.0], [1.0, 0.0, 0.5]),
        ] {
            let prim = MhdPrim {
                hydro: Prim {
                    rho: 1.3,
                    vel: Tensor::new(v),
                    pre: 0.5,
                },
                mag: Tensor::new(b),
            };
            let (cf, cg) = (flat.to_conserved(&eos, &prim), gr.to_conserved(&eos, &prim));
            for k in 0..3 {
                assert!(approx(cf.mom[k], cg.mom[k]), "cons mom{k} v={v:?}");
            }
            assert!(
                approx(cf.den, cg.den) && approx(cf.nrg, cg.nrg),
                "cons v={v:?}"
            );
            let (ff, fg) = (
                flat.to_flux(&prim, &nhat, &eos),
                gr.to_flux(&prim, &nhat, &eos),
            );
            for k in 0..3 {
                assert!(approx(ff.mom[k], fg.mom[k]), "flux mom{k} v={v:?}");
                assert!(approx(ff.mag[k], fg.mag[k]), "flux mag{k} v={v:?}");
            }
            assert!(
                approx(ff.den, fg.den) && approx(ff.nrg, fg.nrg),
                "flux v={v:?}"
            );
            let fp = flat.to_primitive(&eos, &cf).value;
            let gp = gr.to_primitive(&eos, &cg).value;
            assert!(
                approx(fp.hydro.rho, gp.hydro.rho) && approx(fp.hydro.pre, gp.hydro.pre),
                "c2p v={v:?}"
            );
        }
    }

    #[test]
    fn rmhd_gr_wave_speed_bound_contains_flat_fan_at_identity() {
        // at identity/1 the BF-transformed bound must CONTAIN the exact flat MDZ fan
        // (the bound is c_ms >= every magnetosonic speed) and stay inside the light cone.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let flat = Rmhd;
        let gr = RmhdGr::<f64, 3> {
            metric: SpatialMetric::flat(),
            alpha: 1.0,
        };
        let nhat = Tensor::unit(0);
        for &(v, b) in &[
            ([0.3, -0.1, 0.2], [0.4, 0.7, -0.2]),
            ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            ([0.5, 0.0, 0.0], [0.0, 0.0, 1.5]),
        ] {
            let prim = MhdPrim {
                hydro: Prim {
                    rho: 1.0,
                    vel: Tensor::new(v),
                    pre: 0.1,
                },
                mag: Tensor::new(b),
            };
            let (fl, fr) = flat.wave_speeds(&eos, &prim, &nhat);
            let (gl, gr_) = gr.wave_speeds(&eos, &prim, &nhat);
            assert!(
                gl <= fl + 1e-14 && gr_ >= fr - 1e-14,
                "bound contains fan: ({gl},{gr_}) vs ({fl},{fr})"
            );
            assert!(gl >= -1.0 && gr_ <= 1.0, "inside the light cone");
        }
    }

    #[test]
    fn rmhd_gr_round_trips_on_a_non_diagonal_metric() {
        // prim -> conserved (hand-checkable valencia forms) -> KKC recovery on a
        // kerr-like gamma with gamma_{r phi} != 0: the recovered state must match the
        // input, exercising the metric contractions in BOTH directions (lower on p2c,
        // the invariants + raise on c2p).
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let m = curved_metric();
        let gr = RmhdGr::<f64, 3> {
            metric: m,
            alpha: 0.8,
        };
        for &(v, b) in &[
            ([0.15, 0.02, 0.10], [0.3, 0.1, -0.1]),
            ([0.0, 0.0, 0.12], [0.6, 0.0, 0.05]),
            ([-0.2, 0.05, 0.0], [0.0, 0.4, 0.2]),
        ] {
            let prim = MhdPrim {
                hydro: Prim {
                    rho: 1.0,
                    vel: Tensor::new(v),
                    pre: 0.2,
                },
                mag: Tensor::new(b),
            };
            let vsq = m.norm_sq_contra(&prim.hydro.vel);
            assert!(vsq < 1.0, "test state must be subluminal: {vsq}");
            let cons = gr.to_conserved(&eos, &prim);
            let back = gr.to_primitive(&eos, &cons);
            assert!(back.is_ok(), "recovery must classify ok");
            let p = back.value;
            assert!((p.hydro.rho - 1.0).abs() < 1e-10, "rho: {}", p.hydro.rho);
            assert!((p.hydro.pre - 0.2).abs() < 1e-10, "pre: {}", p.hydro.pre);
            for k in 0..3 {
                assert!(
                    (p.hydro.vel[k] - v[k]).abs() < 1e-10,
                    "v{k}: {} vs {}",
                    p.hydro.vel[k],
                    v[k]
                );
            }
        }
    }
}
