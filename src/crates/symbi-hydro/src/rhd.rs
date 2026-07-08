// =============================================================================
// rhd.rs (module root)
//
// the special-relativistic hydrodynamics regime: `impl Regime<S, D> for Rhd`. the
// trait interface lives here; the physics lives in the concern submodules it
// delegates to (the regime supplies only its physics):
//   algebra      lorentz factor / relativistic enthalpy / sound speed (shared w/ RMHD)
//   cons         the Newton-Raphson cons->prim recovery
//   wave_speeds  the Mignone & Bodo relativistic acoustic speeds
//
// state types: same Prim<S,D>/Cons<S,D> structs as newtonian, but with relativistic
// meaning: den = D = rho*W, mom = S = rho*h*W^2*v, nrg = tau = rho*h*W^2 - p - D.
// all functions are pure math — elemental, GPU-callable, no allocation.
// =============================================================================

use symbi_algebra::{Tensor, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::state::{Prim, Cons};
use crate::regime::Regime;
use crate::c2p_result::C2pResult;

mod algebra;
mod cons;
mod gr;
mod wave_speeds;

pub use algebra::{enthalpy, enthalpy_density, lorentz_factor, lorentz_factor_sq, sound_speed_sq};
pub use gr::RhdGr;
pub(crate) use wave_speeds::rhd_speeds_from_vn_gr;
pub use cons::rhd_recover;
use cons::rhd_to_primitive;
use wave_speeds::rhd_speeds_from_vn;

/// special relativistic hydrodynamics.
#[derive(Clone, Copy, Debug)]
pub struct Rhd;

impl<S: Scalar, const D: usize> Regime<S, D> for Rhd {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::RHD_SPEC;
    // relativistic: clamp the HLLE fan to include the stationary state.
    const CLAMP_EXTREMAL_TO_ZERO: bool = true;
    type Prim = Prim<S, D>;
    type Cons = Cons<S, D>;
    type Energy = crate::energy::Adiabatic;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        let v_sq = prim.vel.dot(&prim.vel);
        let ww = lorentz_factor(v_sq);
        let hh = enthalpy(eos, prim.rho, prim.pre);
        let den = prim.rho * ww;
        let rhw2 = prim.rho * hh * ww * ww;
        let mom = prim.vel.scale(rhw2);
        let nrg = rhw2 - prim.pre - den;
        Cons { den, mom, nrg }
    }

    #[inline]
    fn to_conserved_covariant(
        &self,
        eos: &impl Eos<S>,
        prim: &Self::Prim,
        gamma: &crate::spatial_metric::SpatialMetric<S, D>,
        alpha: S,
    ) -> Self::Cons {
        // the Valencia covariant storage: delegate to `RhdGr` at the cell's spatial metric so the
        // initial conserved momentum is the covariant `S_i = rho h W^2 gamma_ij v^j` the c2p inverts.
        RhdGr { metric: *gamma, alpha }.to_conserved(eos, prim)
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where S: OrderedNumeric
    {
        rhd_to_primitive(eos, cons)
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
        let cons = self.to_conserved(eos, prim);
        let vn = prim.vel.dot(nhat);
        let mn = cons.mom.dot(nhat); // normal momentum
        Cons {
            den: cons.den * vn,
            mom: cons.mom.scale(vn) + nhat.scale(prim.pre),
            // rhd energy flux: S_n - D * v_n = mn - D * vn
            nrg: mn - cons.den * vn,
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        rhd_speeds_from_vn(sound_speed_sq(eos, prim.rho, prim.pre), prim.vel.dot(nhat))
    }

    #[inline]
    fn wave_speeds_axis(&self, eos: &impl Eos<S>, prim: &Self::Prim, axis: usize) -> (S, S) {
        // the 1D characteristic estimate depends only on the normal velocity (transverse
        // velocity does not enter) -> read vel[axis] directly, no unit-vector dot.
        rhd_speeds_from_vn(sound_speed_sq(eos, prim.rho, prim.pre), prim.vel[axis])
    }

    // extremal_speeds (clamped) + max_wave_speed (axis fold) are the Regime defaults; rhd
    // sets CLAMP_EXTREMAL_TO_ZERO = true and reuses them — no per-regime copy.

    // is_relativistic derives from SPEC.

    #[inline]
    fn effective_inertia(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        // rho * h * W^2: the relativistic inertial density.
        enthalpy_density(eos, prim.rho, prim.pre, prim.vel.dot(&prim.vel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use crate::newtonian::Newtonian;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }
    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn to_conserved_stationary() {
        // v=0: W=1, D=rho, S=0, tau = rho*h - p - rho = rho*(h-1) - p
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let cons = regime.to_conserved(&eos, &prim);
        let h = enthalpy(&eos, 1.0, 1.0);
        assert!(approx(cons.den, 1.0));
        assert!(approx(cons.mom[0], 0.0));
        assert!(approx(cons.nrg, h - 1.0 - 1.0));
    }

    #[test]
    fn to_conserved_moving() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let v = 0.5;
        let prim = Prim { rho: 1.0, vel: Tensor::new([v]), pre: 1.0 };
        let cons = regime.to_conserved(&eos, &prim);
        let h = enthalpy(&eos, 1.0, 1.0);
        let w = lorentz_factor(v * v);
        assert!(approx(cons.den, w));
        assert!(approx(cons.mom[0], h * w * w * v));
        assert!(approx(cons.nrg, h * w * w - 1.0 - w));
    }

    #[test]
    fn to_conserved_newtonian_limit() {
        // for v << 1, RHD conserved should approximate newtonian conserved
        let eos = IdealGas { gamma: 1.4 };
        let v = 1e-4;
        let prim = Prim { rho: 1.0, vel: Tensor::new([v]), pre: 1.0 };
        let cons_rhd = Rhd.to_conserved(&eos, &prim);
        let cons_newt = Newtonian.to_conserved(&eos, &prim);
        // den: D = rho*W ~ rho*(1 + 0.5*v^2) ~ rho
        assert!(approx_rel(cons_rhd.den, cons_newt.den, 1e-6));
        // mom: S = rho*h*W^2*v ~ rho*v for h~1, W~1
        assert!(approx_rel(cons_rhd.mom[0], cons_newt.mom[0], 1e-2));
    }

    #[test]
    fn flux_stationary() {
        // v=0: all fluxes zero except mom[dir] = p
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 2.0 };
        let ff = regime.to_flux(&prim, &Tensor::unit(0), &eos);
        assert!(approx(ff.den, 0.0));
        assert!(approx(ff.mom[0], 2.0)); // pressure only
        assert!(approx(ff.nrg, 0.0));
    }

    #[test]
    fn flux_uniform_recovery() {
        // for uniform state, hlle should return the physical flux
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.3]), pre: 1.0 };
        let ff = regime.to_flux(&prim, &Tensor::unit(0), &eos);
        let flux_hlle = crate::riemann::hlle(&regime, &eos, &prim, &prim, &Tensor::unit(0), 0.0);
        assert!(approx(ff.den, flux_hlle.den));
        assert!(approx(ff.mom[0], flux_hlle.mom[0]));
        assert!(approx(ff.nrg, flux_hlle.nrg));
    }

    #[test]
    fn rhd_flux_newtonian_limit() {
        // for v << 1, RHD flux should approximate newtonian euler flux
        let eos = IdealGas { gamma: 1.4 };
        let v = 1e-3;
        let prim = Prim { rho: 1.0, vel: Tensor::new([v]), pre: 1.0 };
        let flux_rhd = Rhd.to_flux(&prim, &Tensor::unit(0), &eos);
        let flux_newt = Newtonian.to_flux(&prim, &Tensor::unit(0), &eos);
        // density flux: rho*v for both
        assert!(approx_rel(flux_rhd.den, flux_newt.den, 1e-3));
        // momentum flux: rho*v^2 + p for both
        assert!(approx_rel(flux_rhd.mom[0], flux_newt.mom[0], 1e-2));
    }
}
