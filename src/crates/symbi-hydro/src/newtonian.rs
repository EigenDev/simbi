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

use crate::c2p_result::{C2pResult, ErrorCode};
use crate::eos::Eos;
use crate::regime::Regime;
use crate::state::{Cons, Prim};
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_carrier::Scalar;

/// newtonian (non-relativistic) hydrodynamics.
#[derive(Clone, Copy, Debug)]
pub struct Newtonian;

impl<S: Scalar, const D: usize> Regime<S, D> for Newtonian {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::NEWTONIAN_SPEC;
    type Prim = Prim<S, D>;
    type Cons = Cons<S, D>;
    type Energy = crate::energy::Adiabatic;

    #[inline]
    fn to_conserved(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        prim.to_conserved(eos)
    }

    #[inline]
    fn to_primitive(&self, eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        // raw IEEE math; no silent floors.
        // the diagnostic ErrorCode is preserved (it's an explicit
        // signal that leaves the math untouched). callers can
        // detect a pathology via `result.is_err()` and react; the
        // `value` is the raw unfloored computation with no
        // recovered floor substituted, so any downstream NaN propagation is
        // visible and can be caught at the dt reduction.
        let prim = cons.to_primitive(eos);
        let mut code = ErrorCode::NONE;
        if prim.rho <= S::ZERO {
            code = code.merge(ErrorCode::NEGATIVE_DENSITY);
        }
        if prim.pre <= S::ZERO {
            code = code.merge(ErrorCode::NEGATIVE_PRESSURE);
        }
        if !(prim.rho == prim.rho) || !(prim.pre == prim.pre) {
            code = code.merge(ErrorCode::NON_FINITE);
        }
        if code.is_ok() {
            C2pResult::ok(prim)
        } else {
            C2pResult::err(prim, code)
        }
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Tensor<S, D>, eos: &impl Eos<S>) -> Self::Cons {
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
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Tensor<S, D>) -> (S, S) {
        let a = eos.sound_speed(prim.rho, prim.pre);
        let vn = prim.vel.dot(nhat);
        (vn - a, vn + a)
    }

    #[inline]
    fn wave_speeds_axis(&self, eos: &impl Eos<S>, prim: &Self::Prim, axis: usize) -> (S, S) {
        // the speed depends only on the normal velocity -> read vel[axis] directly (no dot).
        let a = eos.sound_speed(prim.rho, prim.pre);
        let vn = prim.vel[axis];
        (vn - a, vn + a)
    }

    #[inline]
    fn max_wave_speed(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        let a = eos.sound_speed(prim.rho, prim.pre);
        prim.vel.map(|v| v.abs() + a).component_max()
    }

    #[inline]
    fn effective_inertia(&self, _eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        prim.rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use symbi_algebra::Tensor;

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
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
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
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
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
        let nhat = Tensor::unit(0); // x-direction
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
        let nhat = Tensor::unit(1); // y-direction
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
        let nhat = Tensor::unit(0);
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
        let nhat = Tensor::unit(0);
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
        let result = regime.to_primitive(&eos, &cons);
        // ErrorCode still signals the pathology...
        assert!(
            result
                .error
                .contains(crate::c2p_result::ErrorCode::NEGATIVE_DENSITY)
        );
        // ...but the value is the raw computed prim (no silent floor).
        assert_eq!(result.value.rho, -1.0);
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
        let result = regime.to_primitive(&eos, &cons);
        assert!(
            result
                .error
                .contains(crate::c2p_result::ErrorCode::NEGATIVE_PRESSURE)
        );
        // raw pressure is negative; downstream callers / dt reduction
        // are responsible for catching this.
        assert!(result.value.pre < 0.0);
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
