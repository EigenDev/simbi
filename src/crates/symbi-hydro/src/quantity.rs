// =============================================================================
// quantity.rs
//
// semantic thermodynamic quantities for the EOS boundary. each is a
// zero-cost transparent wrapper naming what a scalar slot means, so the
// closure's positional arguments carry their identity in the type: density
// and pressure cannot swap, and the per-mass specific internal energy cannot
// stand in for the per-volume total energy density (or the reverse).
//
// construction and destructuring are explicit (`Density(x)` / `rho.0`):
// wrapping is the claim, made at the call boundary; formula interiors
// destructure at entry and stay carrier arithmetic.
//
// usage:
//  let a = eos.sound_speed(Density(prim.rho), Pressure(prim.pre));
//  fn sound_speed(&self, rho: Density<S>, pre: Pressure<S>) -> S {
//      let (Density(rho), Pressure(pre)) = (rho, pre);
//      ...
//  }
// =============================================================================

/// mass density `rho` (per unit volume).
///
/// the argument identities make a swapped call a type error —
///
/// ```compile_fail
/// use symbi_hydro::eos::{Eos, IdealGas};
/// use symbi_hydro::quantity::{Density, Pressure};
/// fn probe(eos: &IdealGas<f64>, rho: f64, pre: f64) {
///     let _ = eos.sound_speed(Pressure(pre), Density(rho)); // swapped slots
/// }
/// ```
///
/// as is confusing the per-mass internal energy with the per-volume total —
///
/// ```compile_fail
/// use symbi_hydro::eos::{Eos, IdealGas};
/// use symbi_hydro::quantity::{Density, EnergyDensity};
/// fn probe(eos: &IdealGas<f64>, rho: f64, nrg: f64) {
///     // pressure() consumes the specific internal energy e_int, per mass;
///     // the conserved nrg slot is a per-volume energy density.
///     let _ = eos.pressure(Density(rho), EnergyDensity(nrg));
/// }
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Density<S>(pub S);

/// thermodynamic pressure `p`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pressure<S>(pub S);

/// specific internal energy `e_int` (per unit mass): `p = p(rho, e_int)`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecificInternalEnergy<S>(pub S);

/// total energy density (per unit volume) — the conserved `nrg` slot.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyDensity<S>(pub S);

/// squared speed `|v|^2` — the kinetic argument of the energy closures.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocitySquared<S>(pub S);

/// squared sound speed `c_s^2` — the externally prescribed temperature an
/// isothermal recovery consumes (the isothermal state carries no energy
/// slot). keeping it a distinct type means an isothermal cs^2 cannot enter a
/// gamma-law recovery —
///
/// ```compile_fail
/// use symbi_hydro::eos::{Eos, IdealGas};
/// use symbi_hydro::quantity::{Density, SoundSpeedSquared, VelocitySquared};
/// fn probe(eos: &IdealGas<f64>, rho: f64, v2: f64, cs2: f64) {
///     // a gamma-law eos stores EnergyDensity; cs^2 is a different quantity.
///     let _ = eos.recover_pressure(
///         Density(rho),
///         VelocitySquared(v2),
///         SoundSpeedSquared(cs2),
///     );
/// }
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundSpeedSquared<S>(pub S);

/// the bare-slot boundary door: a storage or field slot holds a bare scalar,
/// and the closure is the authority on what that scalar means
/// (`Eos::RecoveryQuantity` — the evolved energy slot for an energy-evolving
/// gas, a prescribed temperature field for an isothermal one). crossing a
/// slot goes through these two named operations — reading interprets the
/// bare value under the closure's claim, writing strips the claim into the
/// slot.
pub trait StoredQuantity<S> {
    /// interpret a bare value read from a slot.
    fn from_stored(raw: S) -> Self;
    /// the bare value written into a slot.
    fn into_stored(self) -> S;
}

impl<S> StoredQuantity<S> for EnergyDensity<S> {
    fn from_stored(raw: S) -> Self {
        EnergyDensity(raw)
    }
    fn into_stored(self) -> S {
        self.0
    }
}

impl<S> StoredQuantity<S> for SoundSpeedSquared<S> {
    fn from_stored(raw: S) -> Self {
        SoundSpeedSquared(raw)
    }
    fn into_stored(self) -> S {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantities_are_layout_neutral() {
        assert_eq!(std::mem::size_of::<Density<f64>>(), 8);
        assert_eq!(std::mem::size_of::<Pressure<f32>>(), 4);
        assert_eq!(std::mem::size_of::<EnergyDensity<f64>>(), 8);
        assert_eq!(
            std::mem::align_of::<SpecificInternalEnergy<f64>>(),
            std::mem::align_of::<f64>()
        );
    }
}
