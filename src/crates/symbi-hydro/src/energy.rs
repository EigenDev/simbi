// =============================================================================
// energy.rs
//
// type-level energy model for compressible hydrodynamics. encodes whether the
// energy equation is evolved (adiabatic) or absent (isothermal) as a phantom
// type parameter on conservative/primitive state types.
//
// adiabatic: Slot<S> = S (real energy field, full euler equations).
// isothermal: Slot<S> = Zero<S> (zero-sized, no energy — compile error if used).
//
// usage:
//   let cons: ConsG<f64, 2, Adiabatic> = ...; // has .nrg: f64
//   let iso:  ConsG<f64, 2, IsoModel>  = ...; // has .nrg: Zero<f64> (ZST)
// =============================================================================

use std::fmt::Debug;
use std::marker::PhantomData;
use symbi_ir::algebra::Scalar;

// ---- energy slot trait ----

/// operations on a scalar-or-absent energy/pressure slot. adiabatic implements
/// this with S directly; isothermal implements with Zero<S> (all ops are no-ops).
///
/// **PartialEq deliberately NOT a super-trait** (Tier 1.7): native `==` on a
/// generic `S: Scalar` is an A1 violation (Gv's `PartialEq` is non-physical
/// via `ord_key`). use `S::cmp_eq` returning `Self::Mask` for equality in
/// carrier-generic code; tests can `assert_eq!` on concrete `f64` values
/// because `f64: PartialEq` inherently.
pub trait EnergySlot<S: Scalar>: Copy + Default + Debug + Send + Sync + 'static {
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn neg(self) -> Self;
    fn scale(self, s: S) -> Self;

    /// extract the scalar value. adiabatic: returns self. isothermal: returns S::ZERO.
    fn value(self) -> S;

    /// construct from a scalar. adiabatic: identity. isothermal: discards the value.
    fn from_scalar(s: S) -> Self;

    /// conditional selection. adiabatic: delegates to S::select. isothermal: no-op.
    fn select_mask(m: S::Mask, yes: Self, no: Self) -> Self;
}

// ---- adiabatic: slot IS the scalar ----

impl<S: Scalar> EnergySlot<S> for S {
    #[inline(always)]
    fn zero() -> Self {
        S::ZERO
    }
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn scale(self, s: S) -> Self {
        self * s
    }
    #[inline(always)]
    fn value(self) -> S {
        self
    }
    #[inline(always)]
    fn from_scalar(s: S) -> Self {
        s
    }
    #[inline(always)]
    fn select_mask(m: S::Mask, yes: Self, no: Self) -> Self {
        <S as Scalar>::select(m, yes, no)
    }
}

// ---- isothermal: zero-sized energy slot ----

/// zero-sized placeholder for the energy/pressure slot in isothermal flows.
/// all arithmetic operations are no-ops. accessing .value() returns S::ZERO.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zero<S>(PhantomData<S>);

impl<S: Scalar> Default for Zero<S> {
    fn default() -> Self {
        Zero(PhantomData)
    }
}

impl<S: Scalar> EnergySlot<S> for Zero<S> {
    #[inline(always)]
    fn zero() -> Self {
        Zero(PhantomData)
    }
    #[inline(always)]
    fn add(self, _rhs: Self) -> Self {
        self
    }
    #[inline(always)]
    fn sub(self, _rhs: Self) -> Self {
        self
    }
    #[inline(always)]
    fn neg(self) -> Self {
        self
    }
    #[inline(always)]
    fn scale(self, _s: S) -> Self {
        self
    }
    #[inline(always)]
    fn value(self) -> S {
        S::ZERO
    }
    #[inline(always)]
    fn from_scalar(_s: S) -> Self {
        Zero(PhantomData)
    }
    #[inline(always)]
    fn select_mask(_m: S::Mask, _yes: Self, _no: Self) -> Self {
        Zero(PhantomData)
    }
}

// ---- energy model marker types ----

/// compile-time marker for energy model. determines the type of the
/// energy/pressure slot in conservative/primitive state types.
pub trait EnergyModel: Copy + 'static {
    /// the storage type for energy/pressure. S for adiabatic, Zero<S> for isothermal.
    type Slot<S: Scalar>: EnergySlot<S>;

    /// whether fields should allocate energy storage.
    const HAS_ENERGY: bool;
}

/// adiabatic energy model: full euler equations with energy equation evolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adiabatic;

impl EnergyModel for Adiabatic {
    type Slot<S: Scalar> = S;
    const HAS_ENERGY: bool = true;
}

/// isothermal energy model: no energy equation. pressure derived from EOS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsoModel;

impl EnergyModel for IsoModel {
    type Slot<S: Scalar> = Zero<S>;
    const HAS_ENERGY: bool = false;
}

// ---- dye slot ----

/// compile-time marker for the passive scalar (dye), the run-level opt-in that rides in the
/// conserved vector as `D_chi = rho chi`. it reuses [`EnergySlot`] for its storage operations:
/// that trait is an optional-scalar slot, and nothing in `zero / add / scale / value` is specific
/// to energy.
///
/// the point of putting the dye in the conserved state rather than carrying it alongside is that
/// EVERY operation which changes mass must change `D_chi` in proportion. expressed as a slot, an
/// operation that rebuilds a conserved state cannot omit the dye without failing to compile —
/// where a hand-wired dye can be, and has been, silently dropped by one path out of several.
///
/// orthogonal to [`EnergyModel`]: `D_chi = rho chi` involves no energy, so an isothermal run
/// carries a dye exactly as an adiabatic one does.
pub trait DyeModel: Copy + 'static {
    /// the storage type for the conserved dye. S when a dye is carried, Zero<S> when not.
    type Slot<S: Scalar>: EnergySlot<S>;

    /// whether fields should allocate dye storage.
    const HAS_DYE: bool;
}

/// a run carrying a passive scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dyed;

impl DyeModel for Dyed {
    type Slot<S: Scalar> = S;
    const HAS_DYE: bool = true;
}

/// a run with no passive scalar: the slot is zero-sized and every operation on it is a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Undyed;

impl DyeModel for Undyed {
    type Slot<S: Scalar> = Zero<S>;
    const HAS_DYE: bool = false;
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_zst() {
        assert_eq!(std::mem::size_of::<Zero<f64>>(), 0);
        assert_eq!(std::mem::size_of::<Zero<f32>>(), 0);
    }

    #[test]
    fn adiabatic_slot_is_scalar() {
        assert_eq!(
            std::mem::size_of::<<Adiabatic as EnergyModel>::Slot<f64>>(),
            8
        );
    }

    #[test]
    fn isothermal_slot_is_zst() {
        assert_eq!(
            std::mem::size_of::<<IsoModel as EnergyModel>::Slot<f64>>(),
            0
        );
    }

    #[test]
    fn adiabatic_slot_arithmetic() {
        let a: f64 = 3.0;
        let b: f64 = 2.0;
        assert_eq!(EnergySlot::add(a, b), 5.0);
        assert_eq!(EnergySlot::sub(a, b), 1.0);
        assert_eq!(EnergySlot::neg(a), -3.0);
        assert_eq!(EnergySlot::scale(a, 2.0), 6.0);
        assert_eq!(EnergySlot::value(a), 3.0);
        assert_eq!(<f64 as EnergySlot<f64>>::from_scalar(7.0), 7.0);
    }

    #[test]
    fn isothermal_slot_arithmetic() {
        let a = Zero::<f64>::zero();
        let b = Zero::<f64>::zero();
        assert_eq!(a.add(b), a);
        assert_eq!(a.sub(b), a);
        assert_eq!(a.neg(), a);
        assert_eq!(a.scale(999.0), a);
        assert_eq!(a.value(), 0.0);
        assert_eq!(<Zero<f64> as EnergySlot<f64>>::from_scalar(999.0), a);
    }

    #[test]
    fn energy_model_has_energy() {
        assert!(Adiabatic::HAS_ENERGY);
        assert!(!IsoModel::HAS_ENERGY);
    }
}

#[cfg(test)]
mod dye_slot_tests {
    use super::*;
    use crate::state::ConsG;

    // the undyed slot must cost nothing: a conserved state without a dye has to be byte-identical
    // to one that never had the field, or every hydro kernel pays for a feature it does not use.
    #[test]
    fn undyed_conserved_state_is_free() {
        assert_eq!(std::mem::size_of::<<Undyed as DyeModel>::Slot<f64>>(), 0);
        assert_eq!(
            std::mem::size_of::<ConsG<f64, 3, Adiabatic, Undyed>>(),
            std::mem::size_of::<f64>() * 5, // den + 3 momenta + nrg
        );
    }

    #[test]
    fn dyed_conserved_state_carries_one_more_scalar() {
        assert_eq!(std::mem::size_of::<<Dyed as DyeModel>::Slot<f64>>(), 8);
        assert_eq!(
            std::mem::size_of::<ConsG<f64, 3, Adiabatic, Dyed>>(),
            std::mem::size_of::<ConsG<f64, 3, Adiabatic, Undyed>>() + 8,
        );
    }

    // the dye is orthogonal to the energy model: an isothermal run carries one just as an
    // adiabatic run does, because `D_chi = rho chi` involves no energy.
    #[test]
    fn isothermal_can_carry_a_dye() {
        assert_eq!(
            std::mem::size_of::<ConsG<f64, 2, IsoModel, Dyed>>(),
            std::mem::size_of::<f64>() * 4, // den + 2 momenta + chi, no energy
        );
        assert!(Dyed::HAS_DYE);
        assert!(!Undyed::HAS_DYE);
    }
}
