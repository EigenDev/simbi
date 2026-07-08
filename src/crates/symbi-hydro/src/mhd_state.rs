// =============================================================================
// mhd_state.rs
//
// per-cell primitive and conservative state types for MHD (magnetohydrodynamics).
// extends the hydro state types with a magnetic field vector via composition,
// generic over scalar `S`, spatial dimension `D`, AND the energy model `E`
// (Adiabatic -> the energy/pressure slot is a scalar; IsoModel -> Zero<S> ZST,
// so isothermal MHD carries NO energy at zero memory/FLOP cost).
//
// the magnetic field B appears in BOTH primitive and conserved states because B
// is NOT evolved via conservation laws — it's evolved via the induction equation
// (constrained transport). B is the same physical field in both representations.
//
// MhdPrimG derefs to PrimG, MhdConsG derefs to ConsG — all hydro field accesses
// (.rho, .vel, .pre, .den, .mom, .nrg) work transparently. arithmetic delegates
// to the inner hydro type + mag component.
//
// `MhdPrim`/`MhdCons` (no E) are the Adiabatic aliases — the existing NMHD/RMHD
// call sites are unchanged. `IsoMhdPrim`/`IsoMhdCons` are the IsoModel aliases.
//
// usage:
//   let prim = MhdPrim { hydro: Prim { rho: 1.0, vel: v, pre: 1.0 }, mag: b };
//   prim.rho  // works via Deref
//   prim.mag  // direct field
// =============================================================================

use symbi_algebra::{Tensor, FieldElement};
use symbi_ir::algebra::Scalar;
use crate::energy::{Adiabatic, EnergyModel, IsoModel};
use crate::state::{ConsG, PrimG};
use std::ops::{Add, Sub, Neg, Mul, Deref, DerefMut};

/// MHD primitive variables: hydro primitives + magnetic field, generic over the
/// energy model `E` (Adiabatic scalar pressure slot, or IsoModel Zero<S> ZST).
/// (PartialEq is intentionally NOT derived — E::Slot is not PartialEq generically.)
#[derive(Clone, Copy, Debug)]
pub struct MhdPrimG<S: Scalar, const D: usize, E: EnergyModel = Adiabatic> {
    pub hydro: PrimG<S, D, E>,
    pub mag: Tensor<S, D>,
}

/// MHD conservative variables: hydro conservatives + magnetic field.
/// nrg (Adiabatic) = total energy; mag = B (evolved by induction, not flux).
#[derive(Clone, Copy, Debug)]
pub struct MhdConsG<S: Scalar, const D: usize, E: EnergyModel = Adiabatic> {
    pub hydro: ConsG<S, D, E>,
    pub mag: Tensor<S, D>,
}

// every MHD conserved state is Magnetic (disjoint from NonMagnetic by concrete type).
impl<S: Scalar, const D: usize, E: EnergyModel> crate::state::Magnetic for MhdConsG<S, D, E> {}

// MHD conserved state decomposes into its hydro ConsG + the magnetic 3-vector — the
// uniform IC-seeding join (see crate::state::SeedableCons).
impl<S: Scalar, const D: usize, E: EnergyModel> crate::state::SeedableCons<S, D> for MhdConsG<S, D, E> {
    type Energy = E;
    #[inline]
    fn hydro_part(&self) -> ConsG<S, D, E> { self.hydro }
    #[inline]
    fn mag_part(&self) -> Option<Tensor<S, D>> { Some(self.mag) }
    #[inline]
    fn from_parts(hydro: ConsG<S, D, E>, mag: Option<Tensor<S, D>>) -> Self {
        MhdConsG { hydro, mag: mag.expect("MHD cons_at requires the magnetic field") }
    }
}

/// adiabatic MHD state (the default) — NMHD/RMHD use these.
pub type MhdPrim<S, const D: usize> = MhdPrimG<S, D, Adiabatic>;
pub type MhdCons<S, const D: usize> = MhdConsG<S, D, Adiabatic>;

/// isothermal MHD state — no energy/pressure slot (Zero<S> ZST).
pub type IsoMhdPrim<S, const D: usize> = MhdPrimG<S, D, IsoModel>;
pub type IsoMhdCons<S, const D: usize> = MhdConsG<S, D, IsoModel>;

// ---- deref: transparent access to hydro fields ----

impl<S: Scalar, const D: usize, E: EnergyModel> Deref for MhdPrimG<S, D, E> {
    type Target = PrimG<S, D, E>;
    #[inline] fn deref(&self) -> &PrimG<S, D, E> { &self.hydro }
}

impl<S: Scalar, const D: usize, E: EnergyModel> DerefMut for MhdPrimG<S, D, E> {
    #[inline] fn deref_mut(&mut self) -> &mut PrimG<S, D, E> { &mut self.hydro }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Deref for MhdConsG<S, D, E> {
    type Target = ConsG<S, D, E>;
    #[inline] fn deref(&self) -> &ConsG<S, D, E> { &self.hydro }
}

impl<S: Scalar, const D: usize, E: EnergyModel> DerefMut for MhdConsG<S, D, E> {
    #[inline] fn deref_mut(&mut self) -> &mut ConsG<S, D, E> { &mut self.hydro }
}

// ---- constructors ----

impl<S: Scalar, const D: usize, E: EnergyModel> MhdPrimG<S, D, E> {
    pub fn zero() -> Self {
        MhdPrimG { hydro: PrimG::default(), mag: Tensor::zeros() }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> MhdConsG<S, D, E> {
    pub fn zero() -> Self {
        MhdConsG { hydro: ConsG::default(), mag: Tensor::zeros() }
    }
}

// ---- arithmetic: delegates to inner hydro type + mag ----

impl<S: Scalar, const D: usize, E: EnergyModel> Add for MhdPrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        MhdPrimG { hydro: self.hydro + rhs.hydro, mag: self.mag + rhs.mag }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Sub for MhdPrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        MhdPrimG { hydro: self.hydro - rhs.hydro, mag: self.mag - rhs.mag }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Neg for MhdPrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        MhdPrimG { hydro: -self.hydro, mag: -self.mag }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Mul<S> for MhdPrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn mul(self, s: S) -> Self {
        MhdPrimG { hydro: self.hydro * s, mag: self.mag.scale(s) }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Add for MhdConsG<S, D, E> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        MhdConsG { hydro: self.hydro + rhs.hydro, mag: self.mag + rhs.mag }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Sub for MhdConsG<S, D, E> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        MhdConsG { hydro: self.hydro - rhs.hydro, mag: self.mag - rhs.mag }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Neg for MhdConsG<S, D, E> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        MhdConsG { hydro: -self.hydro, mag: -self.mag }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Mul<S> for MhdConsG<S, D, E> {
    type Output = Self;
    #[inline]
    fn mul(self, s: S) -> Self {
        MhdConsG { hydro: self.hydro * s, mag: self.mag.scale(s) }
    }
}

// ---- Selectable impls ----

impl<S: Scalar, const D: usize, E: EnergyModel> symbi_ir::algebra::Selectable<S> for MhdConsG<S, D, E>
where
    S::Mask: Copy,
{
    #[inline]
    fn select(m: S::Mask, yes: Self, no: Self) -> Self {
        MhdConsG {
            hydro: <ConsG<S, D, E> as symbi_ir::algebra::Selectable<S>>::select(m, yes.hydro, no.hydro),
            mag: <Tensor<S, D> as symbi_ir::algebra::Selectable<S>>::select(m, yes.mag, no.mag),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> symbi_ir::algebra::Selectable<S> for MhdPrimG<S, D, E>
where
    S::Mask: Copy,
{
    #[inline]
    fn select(m: S::Mask, yes: Self, no: Self) -> Self {
        MhdPrimG {
            hydro: <PrimG<S, D, E> as symbi_ir::algebra::Selectable<S>>::select(m, yes.hydro, no.hydro),
            mag: <Tensor<S, D> as symbi_ir::algebra::Selectable<S>>::select(m, yes.mag, no.mag),
        }
    }
}

// ---- FieldElement impls (both energy models, both precisions) ----

unsafe impl<const D: usize> FieldElement for MhdConsG<f64, D, Adiabatic> { type Scalar = f64; }
unsafe impl<const D: usize> FieldElement for MhdConsG<f32, D, Adiabatic> { type Scalar = f32; }
unsafe impl<const D: usize> FieldElement for MhdPrimG<f64, D, Adiabatic> { type Scalar = f64; }
unsafe impl<const D: usize> FieldElement for MhdPrimG<f32, D, Adiabatic> { type Scalar = f32; }

unsafe impl<const D: usize> FieldElement for MhdConsG<f64, D, IsoModel> { type Scalar = f64; }
unsafe impl<const D: usize> FieldElement for MhdConsG<f32, D, IsoModel> { type Scalar = f32; }
unsafe impl<const D: usize> FieldElement for MhdPrimG<f64, D, IsoModel> { type Scalar = f64; }
unsafe impl<const D: usize> FieldElement for MhdPrimG<f32, D, IsoModel> { type Scalar = f32; }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Prim, Cons};
    use symbi_algebra::Tensor;

    #[test]
    fn test_mhd_prim_arithmetic() {
        let a: MhdPrim<f64, 3> = MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([1.0, 0.0, 0.0]), pre: 2.0 },
            mag: Tensor::new([0.0, 1.0, 0.0]),
        };
        let b: MhdPrim<f64, 3> = MhdPrim {
            hydro: Prim { rho: 0.5, vel: Tensor::new([0.0, 1.0, 0.0]), pre: 1.0 },
            mag: Tensor::new([1.0, 0.0, 0.0]),
        };
        let sum = a + b;
        assert!((sum.rho - 1.5_f64).abs() < 1e-14);
        assert!((sum.vel[0] - 1.0_f64).abs() < 1e-14);
        assert!((sum.vel[1] - 1.0_f64).abs() < 1e-14);
        assert!((sum.mag[0] - 1.0_f64).abs() < 1e-14);
        assert!((sum.mag[1] - 1.0_f64).abs() < 1e-14);
    }

    #[test]
    fn test_mhd_prim_deref() {
        let p: MhdPrim<f64, 3> = MhdPrim {
            hydro: Prim { rho: 2.0, vel: Tensor::new([1.0, 0.0, 0.0]), pre: 3.0 },
            mag: Tensor::new([0.0, 0.5, 0.0]),
        };
        assert!((p.rho - 2.0_f64).abs() < 1e-14);
        assert!((p.vel[0] - 1.0_f64).abs() < 1e-14);
        assert!((p.pre - 3.0_f64).abs() < 1e-14);
        assert!((p.mag[1] - 0.5_f64).abs() < 1e-14);
    }

    #[test]
    fn test_mhd_cons_arithmetic() {
        let u: MhdCons<f64, 3> = MhdCons {
            hydro: Cons { den: 1.0, mom: Tensor::new([2.0, 0.0, 0.0]), nrg: 5.0 },
            mag: Tensor::new([0.0, 1.0, 0.0]),
        };
        let scaled = u * 2.0;
        assert!((scaled.den - 2.0_f64).abs() < 1e-14);
        assert!((scaled.mom[0] - 4.0_f64).abs() < 1e-14);
        assert!((scaled.nrg - 10.0_f64).abs() < 1e-14);
        assert!((scaled.mag[1] - 2.0_f64).abs() < 1e-14);
    }

    #[test]
    fn test_mhd_cons_flux_differencing() {
        let u_l: MhdCons<f64, 3> = MhdCons {
            hydro: Cons { den: 1.0, mom: Tensor::new([1.0, 0.0, 0.0]), nrg: 3.0 },
            mag: Tensor::new([0.0, 1.0, 0.0]),
        };
        let u_r: MhdCons<f64, 3> = MhdCons {
            hydro: Cons { den: 0.5, mom: Tensor::new([0.5, 0.0, 0.0]), nrg: 1.5 },
            mag: Tensor::new([0.0, 0.5, 0.0]),
        };
        let diff = u_r - u_l;
        assert!((diff.den - (-0.5_f64)).abs() < 1e-14);
        assert!((diff.mag[1] - (-0.5_f64)).abs() < 1e-14);
    }
}
