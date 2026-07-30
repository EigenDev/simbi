// =============================================================================
// rmhd.rs (module root)
//
// the relativistic magnetohydrodynamics regime: `impl Regime<S, D> for Rmhd`. the
// trait interface lives here; the bulk physics lives in the concern submodules,
// which this delegates to (the regime supplies only its physics):
//   algebra      magnetic pressure / four-vector / geometric-source quantities
//   cons         the KKC false-position cons->prim recovery
//   wave_speeds  the Mignone & Del Zanna magnetosonic quartic + polynomial solvers
//
// state types: MhdPrim<S,D>/MhdCons<S,D> with magnetic field.
//   den = D = rho*W
//   mom = S = (rho*h*W^2 + b^2)*v - b^0*b
//   nrg = tau = (rho*h*W^2 + b^2) - p_tot - D
//   mag = B (same in prim and cons — evolved by induction equation)
//
// all functions are pure math — elemental, GPU-callable, no allocation.
// =============================================================================

use crate::c2p_result::C2pResult;
use crate::eos::Eos;
use crate::mhd_state::{MhdCons, MhdPrim};
use crate::regime::Regime;
use crate::rhd;
use crate::spatial_metric::SpatialMetric;
use crate::state::Cons;
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_ir::algebra::Scalar;

mod algebra;
mod cons;
mod gr;
mod wave_speeds;

pub use algebra::{
    magnetic_four_vector_spatial, magnetic_pressure, rmhd_source_quantities, total_pressure,
};
pub use cons::rmhd_recover;
use cons::rmhd_to_primitive;
pub use gr::{RmhdGr, rmhd_gr_wave_speeds_axis};
pub use wave_speeds::{rmhd_magnetosonic_cfl_speeds, rmhd_magnetosonic_cfl_speeds_gr};
use wave_speeds::rmhd_wave_speeds;

/// relativistic magnetohydrodynamics.
#[derive(Clone, Copy, Debug)]
pub struct Rmhd;

impl<S: Scalar, const D: usize> Regime<S, D> for Rmhd {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::RMHD_SPEC;
    // relativistic: clamp the HLLE fan to include the stationary state.
    const CLAMP_EXTREMAL_TO_ZERO: bool = true;
    type Prim = MhdPrim<S, D>;
    type Cons = MhdCons<S, D>;
    type Energy = crate::energy::Adiabatic;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        // flat/orthonormal frame -> identity metric (bit-identical to euclidean .dot); the GR
        // metric threads in once the flux path carries it.
        let metric = SpatialMetric::flat();
        let vsq = metric.norm_sq_contra(&prim.vel);
        let ww = rhd::lorentz_factor(vsq);
        let w_sq = rhd::lorentz_factor_sq(vsq);
        let hh = rhd::enthalpy(eos, prim.rho, prim.pre);

        let bsq = metric.norm_sq_contra(&prim.mag);
        let vdb = metric.contract_contra(&prim.vel, &prim.mag);

        // magnetic four-vector components
        let _b0 = ww * vdb;
        let b_mu_sq = bsq / w_sq + vdb * vdb;

        // conserved density
        let den = prim.rho * ww;

        // conserved momentum: (rho*h*W^2 + b^2)*v - b^0*b/W
        let rhw2 = prim.rho * hh * w_sq;
        let mom_fac = rhw2 + bsq; // note: bsq not b_mu_sq here (3+1 decomposition)
        let mom = prim.vel.scale(mom_fac) - prim.mag.scale(vdb);

        // conserved energy (tau = E_total - D)
        let half = S::from_f64(0.5);
        let p_tot = prim.pre + half * b_mu_sq;
        let nrg = rhw2 + bsq - p_tot - den;

        MhdCons {
            hydro: Cons { den, mom, nrg },
            mag: prim.mag,
        }
    }

    #[inline]
    fn to_conserved_covariant(
        &self,
        eos: &impl Eos<S>,
        prim: &Self::Prim,
        gamma: &SpatialMetric<S, D>,
        alpha: S,
        shift: Tensor<S, D>,
        _sqrt_gamma: S,
    ) -> Self::Cons {
        // the covariant storage: the Valencia conserved (S_i = (rho h W^2 + B^2) v_i - (v.B) B_i),
        // then the energy slot re-split to the covariant (killing) energy ehat = alpha tau +
        // (alpha-1) D - beta^i S_i (source-free on a stationary metric).
        // den/mom/mag are the Valencia state the KKC c2p inverts — only the energy slot changes.
        let c = RmhdGr {
            metric: *gamma,
            alpha,
        }
        .to_conserved(eos, prim);
        let nrg = alpha * c.nrg + (alpha - S::ONE) * c.den - shift.dot(&c.mom);
        MhdCons {
            hydro: Cons {
                den: c.den,
                mom: c.mom,
                nrg,
            },
            mag: c.mag,
        }
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        rmhd_to_primitive(eos, cons)
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
        let cons = self.to_conserved(eos, prim);
        let metric = SpatialMetric::flat();
        let vn = metric.contract_contra(&prim.vel, nhat);
        let bn = metric.contract_contra(&prim.mag, nhat);
        let ww = rhd::lorentz_factor(metric.norm_sq_contra(&prim.vel));

        // spatial magnetic four-vector (the SINGLE source, shared with the geometric tension).
        let b_mu = magnetic_four_vector_spatial(prim, &metric);

        let p_tot = total_pressure(prim, &metric);

        // induction flux: E = -v x B, F(B) = nhat x E = vn*B - bn*v
        let induction = prim.mag.scale(vn) - prim.vel.scale(bn);

        MhdCons {
            hydro: Cons {
                den: cons.den * vn,
                mom: cons.mom.scale(vn) + nhat.scale(p_tot) - b_mu.scale(bn / ww),
                nrg: cons.mom.dot(nhat) - cons.den * vn,
            },
            mag: induction,
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        rmhd_wave_speeds(eos, prim, nhat)
    }

    // extremal_speeds (clamped) + max_wave_speed (axis fold) are the Regime defaults; rmhd
    // sets CLAMP_EXTREMAL_TO_ZERO = true and reuses them — no per-regime copy.

    fn effective_inertia(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        // rho*h*W^2 + b^2 for geometric source terms
        let vsq = prim.vel.dot(&prim.vel);
        let w_sq = rhd::lorentz_factor_sq(vsq);
        let hh = rhd::enthalpy(eos, prim.rho, prim.pre);
        let bsq = prim.mag.dot(&prim.mag);
        prim.rho * hh * w_sq + bsq
    }

    // is_relativistic / is_mhd derive from SPEC.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use crate::state::Prim;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn rmhd_to_conserved_at_rest_no_b() {
        // zero B-field, at rest -> should match RHD
        let regime = Rmhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.0, 0.0]),
        };
        let cons = regime.to_conserved(&eos, &prim);
        assert!(approx(cons.den, 1.0, 1e-12));
        assert!(cons.mom[0].abs() < 1e-12);
        // tau = rho*h - p - D = 1*3.5 - 1 - 1 = 1.5 (same as RHD)
        let h = rhd::enthalpy(&eos, 1.0, 1.0);
        assert!(approx(cons.nrg, h - 1.0 - 1.0, 1e-12));
    }

    #[test]
    fn rmhd_roundtrip_weak_field() {
        let regime = Rmhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.1, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.1, 0.0]),
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(
            approx(prim.rho, prim2.rho, 1e-6),
            "rho: {} vs {}",
            prim.rho,
            prim2.rho
        );
        assert!(
            approx(prim.pre, prim2.pre, 1e-4),
            "pre: {} vs {}",
            prim.pre,
            prim2.pre
        );
        assert!(
            approx(prim.vel[0], prim2.vel[0], 1e-6),
            "vx: {} vs {}",
            prim.vel[0],
            prim2.vel[0]
        );
    }

    #[test]
    fn rmhd_wave_speeds_subluminal() {
        let regime = Rmhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([1.0, 0.0, 0.0]),
        };
        let nhat = Tensor::new([1.0, 0.0, 0.0]);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        assert!(sl > -1.0, "sl={} superluminal", sl);
        assert!(sr < 1.0, "sr={} superluminal", sr);
        assert!(sl < 0.0);
        assert!(sr > 0.0);
    }

    #[test]
    fn rmhd_max_wave_speed() {
        let regime = Rmhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.5, 0.0]),
        };
        let s = regime.max_wave_speed(&eos, &prim);
        assert!(s > 0.0);
        assert!(s < 1.0);
    }

    #[test]
    fn rmhd_negative_density_detected() {
        let regime = Rmhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let cons = MhdCons {
            hydro: Cons {
                den: -1.0,
                mom: Tensor::new([0.0, 0.0, 0.0]),
                nrg: 1.0,
            },
            mag: Tensor::new([0.5, 0.0, 0.0]),
        };
        let result = regime.to_primitive(&eos, &cons);
        assert!(
            result
                .error
                .contains(crate::c2p_result::ErrorCode::NEGATIVE_DENSITY)
        );
        assert!(result.value.rho > 0.0);
    }

    #[test]
    fn rmhd_valid_state_is_ok() {
        let regime = Rmhd;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.1, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.1, 0.0]),
        };
        let cons = regime.to_conserved(&eos, &prim);
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.is_ok());
    }
}
