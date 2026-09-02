// =============================================================================
// rhd/gr.rs
//
// `RhdGr` — the relativistic-hydro regime on a curved spacetime, in the fully densitized
// free-index-down form (Gammie et al. 2003; Stone et al. 2024 eq. 20):
//
//   U   = sqrt(-g) [ rho u^t,  T^t_i,  -(T^t_t + rho u^t) ]
//   F^j = sqrt(-g) [ rho u^j,  T^j_i,  -(T^j_t + rho u^j) ]
//   d_t U + d_j F^j = sqrt(-g) [ 0,  (1/2) (d_i g_ab) T^ab,  0 ]
//
// with `T^mu_nu = rho h u^mu u_nu + p delta^mu_nu`. both the state and the flux carry the same
// measure `sqrt(-g) = alpha sqrt(det gamma)`, so the divergence is plain coordinate differencing
// on every chart and there is no lapse to place; the energy source vanishes identically because
// the metric is stationary (the t-row of `(1/2)(d_nu g_ab) T^ab`).
//
// the ADM spelling of the same state is `sqrt(gamma) [ D, S_i, alpha tau + (alpha-1) D -
// beta^i S_i ]`: `sqrt(-g) rho u^t = sqrt(gamma) D` and `sqrt(-g) T^t_i = sqrt(gamma) S_i` with
// `S_i = rho h W^2 gamma_ij v^j` the Valencia covariant momentum. that is how it is built here,
// so the flat limit (`sqrt(gamma) = 1`, `alpha = 1`, `beta = 0`) reduces to `Rhd` component for
// component and flat SR is untouched.
//
// `sqrt_gamma` is the determinant of the full spacetime chart, not of the possibly-reduced `D`
// momentum block: on a 1D radial spherical grid the measure is still `r^2 sin(theta) sqrt(gamma_rr)`,
// and the reduced 1x1 block would drop the `r^2 sin(theta)` that carries the geometry.
//
// `prim.vel` is the contravariant velocity `v^i` (the valencia velocity; = the physical V under
// identity gamma).
// =============================================================================

use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_carrier::Scalar;

use crate::c2p_result::C2pResult;
use crate::eos::Eos;
use crate::regime::Regime;
use crate::regime_spec::RegimeSpec;
use crate::spatial_metric::SpatialMetric;
use crate::state::{Cons, Prim};

use super::cons::rhd_recover;
use super::wave_speeds::rhd_speeds_from_vn_gr;
use super::{Rhd, enthalpy, lorentz_factor, sound_speed_sq};

/// the shared relativistic c2p iteration cap (`C2P_MAX_ITER`).
const MAX_ITER: usize = crate::c2p_result::C2P_MAX_ITER;

/// the densitized relativistic-hydro regime on a curved spacetime. `metric` = gamma_{ij}/gamma^{ij}
/// of the momentum block; `alpha` = the lapse; `shift` = the contravariant shift `beta^i`;
/// `sqrt_gamma` = sqrt(det gamma) of the full chart (every spatial coordinate, gridded or not), so
/// `sqrt(-g) = alpha * sqrt_gamma` is the complete four-volume measure. every conserved slot and
/// every flux slot carries that measure. reduces to `Rhd` at identity gamma, lapse 1, zero shift,
/// unit measure.
#[derive(Clone, Copy)]
pub struct RhdGr<S: Scalar, const D: usize> {
    pub metric: SpatialMetric<S, D>,
    pub alpha: S,
    pub shift: Tensor<S, D>,
    pub sqrt_gamma: S,
}

/// the undensitized Valencia pieces of a primitive on this chart: `(D, S_i, tau)` with
/// `D = rho W`, `S_i = rho h W^2 gamma_ij v^j`, `tau = rho h W^2 - p - D`, plus the Lorentz factor
/// and the total enthalpy density `rho h` the flux assembly reuses.
struct ValenciaParts<S: Scalar, const D: usize> {
    den: S,
    mom: Tensor<S, D>,
    tau: S,
    ww: S,
    rho_h: S,
}

impl<S: Scalar, const D: usize> RhdGr<S, D> {
    /// the undensitized Valencia decomposition of a primitive on this chart. `|v|^2 = gamma_ij v^i
    /// v^j` (v^i CONTRAVARIANT) and `S_i = rho h W^2 gamma_ij v^j` is lowered; identity gamma gives
    /// the euclidean norm and the orthonormal `S = rho h W^2 v`.
    #[inline]
    fn valencia_parts(&self, eos: &impl Eos<S>, prim: &Prim<S, D>) -> ValenciaParts<S, D> {
        let v_sq = self.metric.norm_sq_contra(&prim.vel);
        let ww = lorentz_factor(v_sq);
        let rho_h = prim.rho * enthalpy(eos, prim.rho, prim.pre);
        let den = prim.rho * ww;
        let rhw2 = rho_h * ww * ww;
        ValenciaParts {
            den,
            mom: self.metric.lower(&prim.vel).scale(rhw2),
            tau: rhw2 - prim.pre - den,
            ww,
            rho_h,
        }
    }
}

impl<S: Scalar, const D: usize> Regime<S, D> for RhdGr<S, D> {
    const SPEC: &'static RegimeSpec = <Rhd as Regime<S, D>>::SPEC;
    type Prim = Prim<S, D>;
    type Cons = Cons<S, D>;
    type Energy = crate::energy::Adiabatic;
    const CLAMP_EXTREMAL_TO_ZERO: bool = true;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        // U = sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)], spelled in ADM variables as
        // sqrt(gamma)[D, S_i, alpha tau + (alpha-1) D - beta^i S_i].
        let p = self.valencia_parts(eos, prim);
        let ehat = self.alpha * p.tau + (self.alpha - S::ONE) * p.den - self.shift.dot(&p.mom);
        Cons {
            chi: Default::default(),
            den: self.sqrt_gamma * p.den,
            mom: p.mom.scale(self.sqrt_gamma),
            nrg: self.sqrt_gamma * ehat,
        }
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        // undensitize by the known measure sqrt(-g)(x) first — the metric is a fixed function of
        // position, so this is exact and the inversion below is the unchanged Valencia recovery.
        let inv_dens = S::ONE / self.sqrt_gamma;
        let cons = Cons {
            chi: Default::default(),
            den: cons.den * inv_dens,
            mom: cons.mom.scale(inv_dens),
            nrg: cons.nrg * inv_dens,
        };
        let cons = &cons;
        let dd = cons.den;
        if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
            let floored = Prim {
                rho: S::from_f64(crate::c2p_result::C2P_FAILURE_SENTINEL),
                vel: Tensor::zeros(),
                pre: S::from_f64(crate::c2p_result::C2P_FAILURE_SENTINEL),
            };
            return C2pResult::err(floored, code);
        }
        // recover the Valencia tau from the covariant energy first (invert ehat = alpha tau +
        // (alpha-1) D - beta^i S_i): tau = (ehat + (1-alpha) D + beta^i S_i) / alpha. alpha=1,
        // beta=0 -> tau = ehat, so the flat recovery is untouched. then the shared metric-aware
        // newton on (D, S_i, tau) — the recovery physics never sees the energy re-split.
        let tau = (cons.nrg + (S::ONE - self.alpha) * dd + self.shift.dot(&cons.mom)) / self.alpha;
        let cons_tau = Cons {
            chi: Default::default(),
            den: dd,
            mom: cons.mom,
            nrg: tau,
        };
        // the metric-aware recovery: |S|^2 = gamma^{ij} S_i S_j, then the raised v^i (`rhd_recover`
        // already contracts with `self.metric`). the SR->GR difference lives entirely in the metric value; the code path is shared.
        let prim = rhd_recover(eos, &cons_tau, &self.metric, MAX_ITER);
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
        // F^n = sqrt(-g)[rho u^n, T^n_i, -(T^n_t + rho u^n)]. in ADM variables, with the transport
        // speed vt^n = alpha v^n - beta^n and the coordinate-unit normal n_i:
        //   sqrt(-g) rho u^n = sqrt(gamma) D vt^n
        //   sqrt(-g) T^n_i   = sqrt(gamma) (S_i vt^n + alpha p n_i)
        // the shift rides inside the flux, so no separate advection transform downstream.
        let parts = self.valencia_parts(eos, prim);
        let vt = self.alpha * prim.vel.dot(nhat) - self.shift.dot(nhat);
        // the free-index-down energy flux -sqrt(-g)(T^n_t + rho u^n) = -sqrt(-g) u^n (rho h u_t +
        // rho). the coordinate 4-velocity is u^t = W/alpha, u^i = W(v^i - beta^i/alpha), and
        // u_t = g_tt u^t + beta_j u^j with g_tt = -alpha^2 + beta^k beta_k. alpha = 1, beta = 0
        // gives u_t = -W and the flux collapses to the Valencia (tau + p) v^n.
        let ut = parts.ww / self.alpha;
        let u_sp = prim.vel.scale(parts.ww) - self.shift.scale(parts.ww / self.alpha);
        let beta_low = self.metric.lower(&self.shift);
        let g_tt = S::ZERO - self.alpha * self.alpha + self.shift.dot(&beta_low);
        let u_t = g_tt * ut + beta_low.dot(&u_sp);
        let un = u_sp.dot(nhat);
        let sqrt_neg_g = self.alpha * self.sqrt_gamma;
        Cons {
            chi: Default::default(),
            den: self.sqrt_gamma * parts.den * vt,
            mom: (parts.mom.scale(vt) + nhat.scale(self.alpha * prim.pre)).scale(self.sqrt_gamma),
            nrg: S::ZERO - sqrt_neg_g * un * (parts.rho_h * u_t + prim.rho),
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        // Banyuls-Font coordinate speed (Font 2008 eq 37). the two velocities are distinct and must
        // not be conflated: `vn` = the contravariant normal v^n = v^i n_i (the transport term + inside
        // the radical), `v_sq` = gamma_ij v^i v^j = the physical speed squared (the `1 - v^2` factors).
        // feeding the physical velocity sqrt(gamma_nn) v^n to both slots drove the discriminant
        // negative (NaN Riemann fan) once |v| approached alpha near the horizon — the gr_bondi crash.
        // gamma^{nn} is the inverse-metric normal. identity gamma + alpha 1 (v_sq = vn^2) -> flat.
        let cs_sq = sound_speed_sq(eos, prim.rho, prim.pre);
        let vn = prim.vel.dot(nhat); // contravariant v^n
        let v_sq = self.metric.norm_sq_contra(&prim.vel); // gamma_ij v^i v^j (physical norm)
        let gamma_nn_inv = self.metric.norm_sq_cov(nhat); // gamma^{nn} (coordinate-unit normal)
        let (s_l, s_r) = rhd_speeds_from_vn_gr(cs_sq, vn, v_sq, gamma_nn_inv, self.alpha);
        // the characteristic speeds of d_t U + d_n F^n: the flux carries the shift, so the
        // eigenvalues are the full coordinate speeds lambda^n - beta^n. zero shift -> unchanged.
        let beta_n = self.shift.dot(nhat);
        (s_l - beta_n, s_r - beta_n)
    }

    #[inline]
    fn effective_inertia(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        // rho h W^2 with the metric Lorentz factor |v|^2 = gamma_ij v^i v^j.
        crate::rhd::enthalpy_density(
            eos,
            prim.rho,
            prim.pre,
            self.metric.norm_sq_contra(&prim.vel),
        )
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
        // RhdGr on the flat metric (gamma = identity, alpha = 1) must equal the flat Rhd regime,
        // component-for-component: conserved, flux, and wave speeds. the flat path stays untouched.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let flat = Rhd;
        let gr = RhdGr::<f64, 1> {
            metric: SpatialMetric::flat(),
            alpha: 1.0,
            shift: Tensor::zeros(),
            sqrt_gamma: 1.0,
        };
        let nhat = Tensor::unit(0);
        for &v in &[0.0_f64, 0.3, -0.5, 0.85] {
            let prim = Prim {
                rho: 1.3,
                vel: Tensor::new([v]),
                pre: 0.5,
            };
            let (cf, cg) = (flat.to_conserved(&eos, &prim), gr.to_conserved(&eos, &prim));
            assert!(
                approx(cf.den, cg.den) && approx(cf.mom[0], cg.mom[0]) && approx(cf.nrg, cg.nrg),
                "cons v={v}"
            );
            let (ff, fg) = (
                flat.to_flux(&prim, &nhat, &eos),
                gr.to_flux(&prim, &nhat, &eos),
            );
            assert!(
                approx(ff.den, fg.den) && approx(ff.mom[0], fg.mom[0]) && approx(ff.nrg, fg.nrg),
                "flux v={v}"
            );
            let ((slf, srf), (slg, srg)) = (
                flat.wave_speeds(&eos, &prim, &nhat),
                gr.wave_speeds(&eos, &prim, &nhat),
            );
            assert!(approx(slf, slg) && approx(srf, srg), "speeds v={v}");
        }
    }

    #[test]
    fn rhd_gr_stores_the_densitized_momentum_on_schwarzschild() {
        // Schwarzschild r=10, M=1: f = 0.8, gamma_rr = 1/f = 1.25, gamma^{rr} = f, alpha = sqrt(f).
        // the stored momentum is sqrt(-g) T^t_r = sqrt(gamma) S_r with S_r = rho h W^2 gamma_rr v^r
        // (v^r contravariant) and W from |v|^2 = gamma_rr (v^r)^2. the measure is the full spherical
        // one, r^2 sin(theta) sqrt(gamma_rr) at the equator, so the r^2 that the 1x1 radial block
        // drops is present and the densitization is exercised far from 1.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let (f, vr, r) = (0.8_f64, 0.2_f64, 10.0_f64);
        let grr = 1.0 / f;
        let sqrt_gamma = r * r * grr.sqrt();
        let metric = SpatialMetric::<f64, 1>::new(
            Gamma::new(Matrix::diag(Tensor::new([grr]))),
            GammaInv::new(Matrix::diag(Tensor::new([f]))),
        );
        let alpha = f.sqrt();
        let gr = RhdGr {
            metric,
            alpha,
            shift: Tensor::new([0.0]),
            sqrt_gamma,
        };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([vr]),
            pre: 0.1,
        };
        let c = gr.to_conserved(&eos, &prim);
        let v_sq = grr * vr * vr;
        let w = 1.0 / (1.0 - v_sq).sqrt();
        let h = 1.0 + (4.0 / 3.0) / (4.0 / 3.0 - 1.0) * 0.1 / 1.0; // 1 + Gamma/(Gamma-1) p/rho
        let rhw2 = 1.0 * h * w * w;
        assert!(
            approx(c.den, sqrt_gamma * w),
            "mass slot = sqrt(gamma) rho W"
        );
        assert!(
            approx(c.mom[0], sqrt_gamma * grr * vr * rhw2),
            "momentum slot = sqrt(gamma) gamma_rr v^r rho h W^2"
        );
        // the radial fluxes carry the same measure and the lapse: sqrt(-g) rho u^r =
        // sqrt(gamma) D alpha v^r, sqrt(-g) T^r_r = sqrt(gamma)(S_r alpha v^r + alpha p).
        let fx = gr.to_flux(&prim, &Tensor::unit(0), &eos);
        assert!(
            approx(fx.den, sqrt_gamma * w * alpha * vr),
            "mass flux = sqrt(gamma) D alpha v^r"
        );
        assert!(
            approx(
                fx.mom[0],
                sqrt_gamma * (grr * vr * rhw2 * alpha * vr + alpha * 0.1)
            ),
            "momentum flux = sqrt(gamma)(S_r alpha v^r + alpha p)"
        );
        // and the c2p undensitizes and round-trips back to the same contravariant v^r.
        let back = gr.to_primitive(&eos, &c).unwrap();
        assert!(approx(back.vel[0], vr), "c2p recovers contravariant v^r");
        assert!(
            approx(back.rho, 1.0) && approx(back.pre, 0.1),
            "c2p recovers rho and p"
        );
    }

    #[test]
    fn rhd_gr_energy_slot_is_the_harm_current_on_kerr_schild() {
        // ingoing kerr-schild at r = 6, M = 1 (beta^r != 0), equatorial. the energy conserved and
        // its flux must equal the free-index-down current -sqrt(-g)(T^t_t + rho u^t) and
        // -sqrt(-g)(T^r_t + rho u^r), computed independently straight from the SchwarzschildKS line
        // element by autodiff — the transcription check on the conserved vector, not a restatement
        // of the regime's own algebra. `to_primitive` then inverts the densitized state back.
        use symbi_geometry::SchwarzschildKS;
        use symbi_geometry::grhd_source::coord_energy_cons_flux;
        use symbi_carrier::Dual;
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let (m, r, vr) = (1.0_f64, 6.0_f64, 0.15_f64);
        let a2 = 2.0 * m / r; // 1/3
        let alpha = 1.0 / (1.0 + a2).sqrt();
        let beta = 2.0 * m / (r + 2.0 * m); // beta^r contravariant
        let grr = 1.0 + a2;
        let sqrt_gamma = r * r * grr.sqrt(); // r^2 sin(theta) sqrt(gamma_rr), theta = pi/2
        let metric = SpatialMetric::<f64, 1>::new(
            Gamma::new(Matrix::diag(Tensor::new([grr]))),
            GammaInv::new(Matrix::diag(Tensor::new([1.0 / grr]))),
        );
        let gr = RhdGr {
            metric,
            alpha,
            shift: Tensor::new([beta]),
            sqrt_gamma,
        };
        let prim = Prim {
            rho: 1.3,
            vel: Tensor::new([vr]),
            pre: 0.4,
        };
        let c = gr.to_conserved(&eos, &prim);
        let fx = gr.to_flux(&prim, &Tensor::unit(0), &eos);
        let hh = 1.0 + (4.0 / 3.0) / (1.0 / 3.0) * 0.4 / 1.3;
        let (e_ref, f_ref): (f64, Tensor<f64, 3>) = coord_energy_cons_flux(
            &SchwarzschildKS {
                mass: Dual::constant(m),
            },
            Tensor::<f64, 3>::new([r, std::f64::consts::FRAC_PI_2, 0.0]),
            1.3,
            1.3 * hh,
            Tensor::<f64, 3>::new([vr, 0.0, 0.0]),
            0.4,
        );
        assert!(
            (c.nrg - e_ref).abs() < 1e-9 * e_ref.abs().max(1.0),
            "energy slot {} vs -sqrt(-g)(T^t_t + rho u^t) {e_ref}",
            c.nrg
        );
        assert!(
            (fx.nrg - f_ref[0]).abs() < 1e-9 * f_ref[0].abs().max(1.0),
            "energy flux {} vs -sqrt(-g)(T^r_t + rho u^r) {}",
            fx.nrg,
            f_ref[0]
        );
        // the densitized state inverts back to the primitive through the shared newton.
        let back = gr.to_primitive(&eos, &c).unwrap();
        assert!(
            approx(back.rho, 1.3) && approx(back.vel[0], vr) && approx(back.pre, 0.4),
            "densitized c2p round-trips the primitive"
        );
        // the shift enters the fan as the coordinate speed lambda - beta^r; both characteristics
        // shift by the same amount, so their separation is unchanged.
        let (s_l, s_r) = gr.wave_speeds(&eos, &prim, &Tensor::unit(0));
        let unshifted = RhdGr {
            metric,
            alpha,
            shift: Tensor::new([0.0]),
            sqrt_gamma,
        };
        let (u_l, u_r) = unshifted.wave_speeds(&eos, &prim, &Tensor::unit(0));
        assert!(
            approx(s_l, u_l - beta) && approx(s_r, u_r - beta),
            "fan carries -beta^r"
        );
    }
}
