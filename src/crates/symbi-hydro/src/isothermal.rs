// =============================================================================
// isothermal.rs
//
// zero-overhead isothermal hydrodynamics via the energy model type system.
// uses ConsG<S, D, IsoModel> and PrimG<S, D, IsoModel> — the energy/pressure
// slots are Zero<S> (zst). no wasted memory, no wasted flops, no wasted
// bandwidth. accessing .nrg on isothermal cons returns Zero<S>, a zst placeholder.
//
// IsoNewtonian implements Regime<S, D> with isothermal types. HLLE works
// automatically (regime-generic). HLLC is not applicable (contact wave
// resolution requires the energy equation).
//
// usage:
//   let regime = IsoNewtonian;
//   let eos = Isothermal { cs: 1.0 };
//   let prim = PrimG::<f64, 1, IsoModel> {
//       rho: 1.0, vel: Tensor::new([0.5]), pre: Zero::default(),
//   };
//   let cons = regime.to_conserved(&eos, &prim);
// =============================================================================

use crate::quantity::{Density, Pressure, SpecificInternalEnergy};
use crate::c2p_result::C2pResult;
use crate::energy::{IsoModel, Zero};
use crate::eos::Eos;
use crate::regime::Regime;
use crate::state::{ConsG, PrimG};
use symbi_algebra::{FaceNormal, Normalized, OrderedNumeric, Physical};
use symbi_carrier::Scalar;

/// type aliases for isothermal state types.
pub type IsoCons<S, const D: usize> = ConsG<S, D, IsoModel>;
pub type IsoPrim<S, const D: usize> = PrimG<S, D, IsoModel>;

/// isothermal newtonian hydrodynamics. no energy equation.
/// pressure = cs^2 * rho, derived from the Isothermal EOS.
#[derive(Clone, Copy, Debug)]
pub struct IsoNewtonian;

impl<S: Scalar, const D: usize> Regime<S, D> for IsoNewtonian {
    const SPEC: &'static crate::regime_spec::RegimeSpec = &crate::regime_spec::ISO_NEWTONIAN_SPEC;
    type Prim = PrimG<S, D, IsoModel>;

    // the solver operates in the locally-flat orthonormal frame, so its
    // face normal is the physical-frame witness.
    type Normal = Normalized<Physical<S, D>>;
    type Cons = ConsG<S, D, IsoModel>;
    type Energy = IsoModel;

    #[inline]
    fn to_conserved(&self, _eos: &impl Eos<S>, prim: &Self::Prim) -> Self::Cons {
        ConsG {
            chi: Default::default(),
            den: prim.rho,
            mom: prim.vel.scale(prim.rho),
            nrg: Zero::default(),
        }
    }

    #[inline]
    fn to_primitive(&self, _eos: &impl Eos<S>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric,
    {
        // mul-by-reciprocal: matches the chalkboard kernel form so CPU
        // and GPU agree bit-for-bit. divide-by-zero produces inf/NaN
        // by IEEE rules — no silent floor. iso has no failure mode the
        // floor would meaningfully recover from, so always Ok.
        let inv_rho = S::ONE / cons.den;
        C2pResult::ok(PrimG {
            rho: cons.den,
            vel: cons.mom.scale(inv_rho),
            pre: Zero::default(),
        })
    }

    #[inline]
    fn to_flux(&self, prim: &Self::Prim, nhat: &Self::Normal, eos: &impl Eos<S>) -> Self::Cons {
        let nhat = nhat.components();
        let vn = prim.vel.dot(nhat);
        let pre = eos.pressure(Density(prim.rho), SpecificInternalEnergy(S::ZERO));
        ConsG {
            chi: Default::default(),
            den: prim.rho * vn,
            mom: prim.vel.scale(prim.rho * vn) + nhat.scale(pre),
            nrg: Zero::default(),
        }
    }

    #[inline]
    fn wave_speeds(&self, eos: &impl Eos<S>, prim: &Self::Prim, nhat: &Self::Normal) -> (S, S) {
        let nhat = nhat.components();
        let cs = eos.sound_speed(Density(prim.rho), Pressure(S::ZERO));
        let vn = prim.vel.dot(nhat);
        (vn - cs, vn + cs)
    }

    #[inline]
    fn max_wave_speed(&self, eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        let cs = eos.sound_speed(Density(prim.rho), Pressure(S::ZERO));
        prim.vel.map(|v| v.abs() + cs).component_max()
    }

    #[inline]
    fn effective_inertia(&self, _eos: &impl Eos<S>, prim: &Self::Prim) -> S {
        prim.rho
    }

    // has_energy derives from SPEC (ISO_NEWTONIAN_SPEC.has_energy = false).
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{FaceNormal, Normalized};
    use symbi_algebra::Tensor;
    use crate::energy::EnergySlot;
    use crate::eos::Isothermal;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-13 * a.abs().max(b.abs()).max(1.0)
    }

    fn iso_prim(rho: f64, vel: Tensor<f64, 1>) -> IsoPrim<f64, 1> {
        PrimG {
            rho,
            vel,
            pre: Zero::default(),
        }
    }

    fn iso_prim_2d(rho: f64, vel: Tensor<f64, 2>) -> IsoPrim<f64, 2> {
        PrimG {
            rho,
            vel,
            pre: Zero::default(),
        }
    }

    fn iso_prim_3d(rho: f64, vel: Tensor<f64, 3>) -> IsoPrim<f64, 3> {
        PrimG {
            rho,
            vel,
            pre: Zero::default(),
        }
    }

    // ---- IsoCons arithmetic ----

    #[test]
    fn iso_cons_arithmetic() {
        let a = IsoCons::<f64, 2> {
            chi: Default::default(),
            den: 1.0,
            mom: Tensor::new([2.0, 3.0]),
            nrg: Zero::default(),
        };
        let b = IsoCons::<f64, 2> {
            chi: Default::default(),
            den: 0.5,
            mom: Tensor::new([1.0, 0.5]),
            nrg: Zero::default(),
        };

        let sum = a + b;
        assert!(approx(sum.den, 1.5));
        assert!(approx(sum.mom[0], 3.0));
        assert!(approx(sum.mom[1], 3.5));

        let diff = a - b;
        assert!(approx(diff.den, 0.5));
        assert!(approx(diff.mom[0], 1.0));

        let neg = -a;
        assert!(approx(neg.den, -1.0));

        let scaled = a * 2.0;
        assert!(approx(scaled.den, 2.0));
        assert!(approx(scaled.mom[1], 6.0));
    }

    // ---- IsoPrim arithmetic ----

    #[test]
    fn iso_prim_arithmetic() {
        let a = IsoPrim::<f64, 2> {
            rho: 1.0,
            vel: Tensor::new([2.0, 3.0]),
            pre: Zero::default(),
        };
        let b = IsoPrim::<f64, 2> {
            rho: 0.5,
            vel: Tensor::new([1.0, 0.5]),
            pre: Zero::default(),
        };

        let sum = a + b;
        assert!(approx(sum.rho, 1.5));
        assert!(approx(sum.vel[0], 3.0));

        let diff = a - b;
        assert!(approx(diff.rho, 0.5));

        let neg = -a;
        assert!(approx(neg.rho, -1.0));

        let scaled = a * 2.0;
        assert!(approx(scaled.rho, 2.0));
        assert!(approx(scaled.vel[0], 4.0));
    }

    // ---- regime roundtrip ----

    #[test]
    fn iso_roundtrip_1d() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let prim = iso_prim(2.0, Tensor::new([0.7]));
        let cons = regime.to_conserved(&eos, &prim);
        assert!(approx(cons.den, 2.0));
        assert!(approx(cons.mom[0], 1.4));
        assert_eq!(cons.nrg.value(), 0.0); // isothermal: nrg is zst
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx(prim.rho, prim2.rho));
        assert!(approx(prim.vel[0], prim2.vel[0]));
    }

    #[test]
    fn iso_roundtrip_3d() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 2.0 };
        let prim = iso_prim_3d(0.5, Tensor::new([1.0, -0.3, 0.7]));
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx(prim.rho, prim2.rho));
        for dd in 0..3 {
            assert!(approx(prim.vel[dd], prim2.vel[dd]));
        }
    }

    // ---- flux ----

    #[test]
    fn iso_flux_x_direction() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let prim = iso_prim_2d(1.0, Tensor::new([2.0, 0.0]));
        let nhat = Normalized::axis(0);
        let flux = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, 2.0));
        assert!(approx(flux.mom[0], 5.0)); // rho*vx*vx + p = 1*4 + 1 = 5
        assert!(approx(flux.mom[1], 0.0));
        assert_eq!(flux.nrg.value(), 0.0);
    }

    #[test]
    fn iso_flux_y_direction() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 2.0 };
        let prim = iso_prim_2d(1.0, Tensor::new([2.0, 3.0]));
        let nhat = Normalized::axis(1);
        let flux = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, 3.0));
        assert!(approx(flux.mom[0], 6.0));
        assert!(approx(flux.mom[1], 13.0)); // rho*vy*vy + p = 1*9 + 4 = 13
    }

    // ---- wave speeds ----

    #[test]
    fn iso_wave_speeds_at_rest() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 2.0 };
        let prim = iso_prim_2d(1.0, Tensor::new([0.0, 0.0]));
        let nhat = Normalized::axis(0);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        assert!(approx(sl, -2.0));
        assert!(approx(sr, 2.0));
    }

    #[test]
    fn iso_wave_speeds_moving() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let prim = iso_prim(1.0, Tensor::new([1.5]));
        let nhat = Normalized::axis(0);
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        assert!(approx(sl, 0.5));
        assert!(approx(sr, 2.5));
    }

    #[test]
    fn iso_max_wave_speed() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let prim = iso_prim_3d(1.0, Tensor::new([1.0, -2.0, 0.5]));
        let s = regime.max_wave_speed(&eos, &prim);
        assert!(approx(s, 3.0));
    }

    // ---- HLLE (regime-generic, works automatically) ----

    #[test]
    fn iso_hlle_symmetric() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let left = iso_prim(1.0, Tensor::new([0.0]));
        let right = iso_prim(1.0, Tensor::new([0.0]));
        let nhat = Normalized::axis(0);
        let flux = crate::riemann::hlle(&regime, &eos, &left, &right, &nhat, 0.0);
        assert!(approx(flux.den, 0.0));
        // momentum flux = pressure = cs^2 * rho = 1.0
        assert!(approx(flux.mom[0], 1.0));
    }

    #[test]
    fn iso_hlle_sod_like() {
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let left = iso_prim(2.0, Tensor::new([0.0]));
        let right = iso_prim(1.0, Tensor::new([0.0]));
        let nhat = Normalized::axis(0);
        let flux = crate::riemann::hlle(&regime, &eos, &left, &right, &nhat, 0.0);
        assert!(flux.den > 0.0);
    }

    // ---- no-floor behavior ----

    #[test]
    fn iso_negative_density_passes_through() {
        // no silent floor: negative density is preserved verbatim and
        // velocity is the IEEE division mom / den. always Ok (iso has
        // no recoverable failure mode that the trait method could
        // meaningfully flag).
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let cons = IsoCons {
            chi: Default::default(),
            den: -1.0,
            mom: Tensor::new([2.0]),
            nrg: Zero::default(),
        };
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.is_ok());
        assert_eq!(result.value.rho, -1.0);
        assert_eq!(result.value.vel[0], -2.0);
    }

    #[test]
    fn iso_zero_density_yields_non_finite_velocity() {
        // divide-by-zero follows IEEE semantics and surfaces as +inf so callers
        // can detect and act (e.g., dt reduction).
        let regime = IsoNewtonian;
        let eos = Isothermal { cs: 1.0 };
        let cons = IsoCons {
            chi: Default::default(),
            den: 0.0,
            mom: Tensor::new([1.0]),
            nrg: Zero::default(),
        };
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.is_ok());
        assert_eq!(result.value.rho, 0.0);
        let v: f64 = result.value.vel[0];
        assert!(v.is_infinite());
    }

    #[test]
    fn iso_has_no_energy() {
        let regime = IsoNewtonian;
        assert!(!<IsoNewtonian as Regime<f64, 2>>::has_energy(&regime));
    }

    // ---- zero / default ----

    #[test]
    fn iso_zero_constructors() {
        let p = IsoPrim::<f64, 3>::zero();
        assert_eq!(p.rho, 0.0);
        assert_eq!(p.vel, Tensor::new([0.0, 0.0, 0.0]));
        assert_eq!(p.pre.value(), 0.0);

        let c = IsoCons::<f64, 2>::zero();
        assert_eq!(c.den, 0.0);
        assert_eq!(c.mom, Tensor::new([0.0, 0.0]));
        assert_eq!(c.nrg.value(), 0.0);
    }

    #[test]
    fn iso_default_is_zero() {
        let p = IsoPrim::<f64, 2>::default();
        assert_eq!(p.rho, 0.0);
        let c = IsoCons::<f64, 2>::default();
        assert_eq!(c.den, 0.0);
    }

    // ---- size assertions ----

    #[test]
    fn iso_cons_no_energy_overhead() {
        // 1d: den(8) + mom(8) = 16 bytes, no nrg
        assert_eq!(std::mem::size_of::<IsoCons<f64, 1>>(), 16);
        // 2d: den(8) + mom(16) = 24 bytes
        assert_eq!(std::mem::size_of::<IsoCons<f64, 2>>(), 24);
        // 3d: den(8) + mom(24) = 32 bytes
        assert_eq!(std::mem::size_of::<IsoCons<f64, 3>>(), 32);
    }

    #[test]
    fn iso_prim_no_pressure_overhead() {
        assert_eq!(std::mem::size_of::<IsoPrim<f64, 1>>(), 16);
        assert_eq!(std::mem::size_of::<IsoPrim<f64, 2>>(), 24);
        assert_eq!(std::mem::size_of::<IsoPrim<f64, 3>>(), 32);
    }
}
