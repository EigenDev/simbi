// =============================================================================
// newtonian.rs
//
// newtonian (non-relativistic) hydrodynamics regime. implements the Regime
// trait with nhat-parametrized flux and wave speeds.
//
// all methods use dot(vel, nhat) for the normal velocity. one implementation
// handles all dimensions and all directions.
//
// usage:
//   let regime = Newtonian;
//   let nhat = Tensor::unit(0); // x-direction
//   let flux = regime.to_flux(&prim, &nhat, &eos);
//   let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
// =============================================================================

use crate::eos::EosFor;
use crate::quantity::{Density, Pressure};
use crate::recovery::Recovery;
use crate::regime::Regime;
use crate::state::{Cons, Prim};
use symbi_algebra::{FaceNormal, Normalized, OrderedNumeric, Physical};
use symbi_carrier::Scalar;

/// newtonian (non-relativistic) hydrodynamics.
#[derive(Clone, Copy, Debug)]
pub struct Newtonian;

impl<S: Scalar, const D: usize> Regime<S, D> for Newtonian {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::NEWTONIAN_SPEC;
    type Prim = Prim<S, D>;

    // the solver operates in the locally-flat orthonormal frame, so its
    // face normal is the physical-frame witness.
    type Normal = Normalized<Physical<S, D>>;
    type Cons = Cons<S, D>;
    type Energy = crate::energy::Adiabatic;

    #[inline]
    fn to_conserved(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim) -> Self::Cons {
        prim.to_conserved(eos)
    }

    #[inline]
    fn to_primitive(
        &self,
        eos: &impl EosFor<S, Self::Energy>,
        cons: &Self::Cons,
    ) -> Recovery<Self::Prim>
    where
        S: OrderedNumeric,
    {
        // raw IEEE math; no silent floors. the audit is an explicit signal
        // that leaves the math untouched: a rejected candidate travels
        // diagnostic-only in the failure, so any downstream NaN propagation
        // in an accepted state stays visible at the dt reduction.
        let prim = cons.to_primitive(eos);
        crate::recovery::judge(
            prim,
            crate::recovery::newtonian_prim_audit(prim.rho, prim.pre, &prim.vel),
        )
    }

    #[inline]
    fn to_flux(
        &self,
        prim: &Self::Prim,
        nhat: &Self::Normal,
        eos: &impl EosFor<S, Self::Energy>,
    ) -> Self::Cons {
        let nhat = nhat.components();
        let vn = prim.vel.dot(nhat); // normal velocity
        let cons = prim.to_conserved(eos);
        Cons {
            chi: Default::default(),
            den: cons.den * vn,
            mom: cons.mom.scale(vn) + nhat.scale(prim.pre),
            nrg: (cons.nrg + prim.pre) * vn,
        }
    }

    #[inline]
    fn wave_speeds(
        &self,
        eos: &impl EosFor<S, Self::Energy>,
        prim: &Self::Prim,
        nhat: &Self::Normal,
    ) -> (S, S) {
        let nhat = nhat.components();
        let a = eos.sound_speed(Density(prim.rho), Pressure(prim.pre));
        let vn = prim.vel.dot(nhat);
        (vn - a, vn + a)
    }

    #[inline]
    fn wave_speeds_axis(
        &self,
        eos: &impl EosFor<S, Self::Energy>,
        prim: &Self::Prim,
        axis: usize,
    ) -> (S, S) {
        // the speed depends only on the normal velocity -> read vel[axis] directly (no dot).
        let a = eos.sound_speed(Density(prim.rho), Pressure(prim.pre));
        let vn = prim.vel[axis];
        (vn - a, vn + a)
    }

    #[inline]
    fn max_wave_speed(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim) -> S {
        let a = eos.sound_speed(Density(prim.rho), Pressure(prim.pre));
        prim.vel.map(|v| v.abs() + a).component_max()
    }

    #[inline]
    fn effective_inertia(&self, _eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim) -> S {
        prim.rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use symbi_algebra::Tensor;
    use symbi_algebra::{FaceNormal, Normalized};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-13 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn newtonian_roundtrip_1d() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 2.0,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap().into_inner();
        assert!(approx(prim.rho, prim2.rho));
        assert!(approx(prim.vel[0], prim2.vel[0]));
        assert!(approx(prim.pre, prim2.pre));
    }

    #[test]
    fn newtonian_roundtrip_3d() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim {
            rho: 0.5,
            vel: Tensor::new([1.0, -0.3, 0.7]),
            pre: 0.1,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap().into_inner();
        assert!(approx(prim.rho, prim2.rho));
        for dd in 0..3 {
            assert!(approx(prim.vel[dd], prim2.vel[dd]));
        }
        assert!(approx(prim.pre, prim2.pre));
    }

    #[test]
    fn flux_x_direction() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([2.0, 0.0]),
            pre: 1.0,
        };
        let nhat = Normalized::axis(0); // x-direction
        let flux = regime.to_flux(&prim, &nhat, &eos);

        // f_den = rho * vn = 1.0 * 2.0 = 2.0
        assert!(approx(flux.den, 2.0));
        // f_mom_x = rho*vx*vn + p = 1.0*2.0*2.0 + 1.0 = 5.0
        assert!(approx(flux.mom[0], 5.0));
        // f_mom_y = rho*vy*vn = 0.0
        assert!(approx(flux.mom[1], 0.0));
    }

    #[test]
    fn flux_y_direction() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([2.0, 3.0]),
            pre: 1.0,
        };
        let nhat = Normalized::axis(1); // y-direction
        let flux = regime.to_flux(&prim, &nhat, &eos);

        // vn = dot(vel, nhat) = 3.0
        // f_den = rho * vn = 3.0
        assert!(approx(flux.den, 3.0));
        // f_mom_x = rho*vx*vn = 1.0*2.0*3.0 = 6.0
        assert!(approx(flux.mom[0], 6.0));
        // f_mom_y = rho*vy*vn + p = 1.0*3.0*3.0 + 1.0 = 10.0
        assert!(approx(flux.mom[1], 10.0));
    }

    #[test]
    fn wave_speeds_at_rest() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        };
        let nhat = Normalized::axis(0);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        let cs = (1.4f64 * 1.0 / 1.0).sqrt();
        assert!(approx(sl, -cs));
        assert!(approx(sr, cs));
    }

    #[test]
    fn wave_speeds_moving() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([1.0, 0.0]),
            pre: 1.0,
        };
        let nhat = Normalized::axis(0);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        let cs = (1.4f64).sqrt();
        assert!(approx(sl, 1.0 - cs));
        assert!(approx(sr, 1.0 + cs));
    }

    #[test]
    fn max_wave_speed_3d() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([1.0, -2.0, 0.5]),
            pre: 1.0,
        };
        let s = regime.max_wave_speed(&eos, &prim);
        let cs = (5.0f64 / 3.0).sqrt();
        // max of |vx|+cs, |vy|+cs, |vz|+cs = 2.0 + cs
        assert!(approx(s, 2.0 + cs));
    }

    // ---- error detection (no-floor policy: raw value is returned with flag) ----

    #[test]
    fn negative_density_flagged_with_raw_value() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let cons = Cons {
            chi: Default::default(),
            den: -1.0,
            mom: Tensor::new([0.0]),
            nrg: 1.0,
        };
        let failure = regime.to_primitive(&eos, &cons).unwrap_err();
        // the issue set signals the pathology...
        assert!(
            failure
                .issues()
                .contains(crate::recovery::RecoveryIssues::NEGATIVE_DENSITY)
        );
        // ...and the raw computed prim survives diagnostically (no silent floor).
        assert!(failure.candidate().snapshot().contains("rho: -1.0"));
    }

    #[test]
    fn negative_pressure_flagged_with_raw_value() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        // huge kinetic energy, small thermal -> negative pressure.
        let cons = Cons {
            chi: Default::default(),
            den: 1.0,
            mom: Tensor::new([100.0]),
            nrg: 1.0,
        };
        let failure = regime.to_primitive(&eos, &cons).unwrap_err();
        assert!(
            failure
                .issues()
                .contains(crate::recovery::RecoveryIssues::NEGATIVE_PRESSURE)
        );
        // the raw negative pressure survives diagnostically; an accepted state
        // never carries it.
        assert!(failure.candidate().snapshot().contains("pre: -"));
    }

    #[test]
    fn valid_state_is_ok() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 2.0,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.is_ok());
    }
}
