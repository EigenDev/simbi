// =============================================================================
// state.rs
//
// per-cell primitive and conservative state types for compressible hydro.
// generic over scalar type S, spatial dimension D, and energy model E.
//
// ConsG<S, D, E> / PrimG<S, D, E> are the full generic types. the type aliases
// Cons<S, D> and Prim<S, D> default to Adiabatic for backward compatibility.
//
// for isothermal flows, use ConsG<S, D, IsoModel> — the energy/pressure slot
// is zero-sized (Zero<S>). accessing .nrg on isothermal cons returns Zero<S>, a
// ZST; arithmetic with f64 on that slot does not compile.
//
// usage:
//   let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
//   let cons = prim.to_conserved(&eos);  // newtonian convenience
// =============================================================================

use crate::energy::{Adiabatic, DyeModel, EnergyModel, EnergySlot, IsoModel, Undyed};
use crate::eos::Eos;
use std::ops::{Add, Mul, Neg, Sub};
use symbi_algebra::{FieldElement, Tensor};
use symbi_ir::algebra::Scalar;

// ---- generic state types ----

/// conservative variables parameterized by energy model.
/// adiabatic: nrg is S (real energy). isothermal: nrg is Zero<S> (ZST).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsG<S: Scalar, const D: usize, E: EnergyModel = Adiabatic, X: DyeModel = Undyed> {
    pub den: S,
    /// momentum density. INVARIANT: PHYSICAL (orthonormal-frame) components, `rho*V_a` with
    /// `V_a = h_a v^a` — the frame the conservation form is written in. it is left
    /// a bare `Tensor` on purpose (no `symbi_algebra::Physical` wrapper): the interior physics is
    /// frame-CONSISTENT by construction (the Riemann solver / flux / wave-speeds are locally flat
    /// and never mix frames), so typing it would be a ~500-site tax that catches zero bugs. frames
    /// are crossed ONLY at boundaries, through the typed `Metric` morphisms (`to_physical` /
    /// `vector_to_cartesian`); a `.raw()` there is the audited escape hatch.
    pub mom: Tensor<S, D>,
    pub nrg: E::Slot<S>,
    /// the conserved passive scalar `D_chi = rho chi`. zero-sized unless the run carries a dye, so
    /// an undyed state is byte-identical to one without the slot. it lives HERE, in the conserved
    /// vector, so that any operation rebuilding a conserved state has to say what happens to the
    /// dye — a mass drain that forgets it would otherwise raise the concentration of the gas it
    /// leaves behind, silently and only on whichever code path did the forgetting.
    pub chi: X::Slot<S>,
}

/// primitive variables parameterized by energy model.
/// adiabatic: pre is S (real pressure). isothermal: pre is Zero<S> (ZST).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimG<S: Scalar, const D: usize, E: EnergyModel = Adiabatic> {
    pub rho: S,
    /// velocity. INVARIANT: PHYSICAL (orthonormal-frame) components `V_a` (= `Physical` in
    /// `symbi_algebra`); these orthonormal-frame values are distinct from the coordinate `v^i` — see `ConsG::mom` for why it stays a bare `Tensor`.
    pub vel: Tensor<S, D>,
    pub pre: E::Slot<S>,
}

// ---- uniform conserved-state decomposition for IC seeding ----

/// decompose a regime's conserved state into the hydro `ConsG` (mass / momentum /
/// energy, scattered to the cons FieldGroup) plus the optional magnetic 3-vector
/// (scattered to the MHD bcell). this is the ONE join that lets a single IC entry
/// point (`SimState::seed_cell`) seed every regime — energy-bearing or not (the
/// `EnergyModel` slot abstracts pressure/energy presence), MHD or pure hydro — via
/// `to_conserved` + `scatter_from`, with no per-regime hand-built `Cons { .. }`.
pub trait SeedableCons<S: Scalar, const D: usize> {
    type Energy: EnergyModel;
    fn hydro_part(&self) -> ConsG<S, D, Self::Energy>;
    fn mag_part(&self) -> Option<Tensor<S, D>>;
    /// reassemble from the hydro conserved state + the optional magnetic 3-vector — the INVERSE
    /// of `hydro_part`/`mag_part`, for cell read-back (`SimState::cons_at`). pure hydro ignores
    /// `mag`; MHD requires it (`Some`).
    fn from_parts(hydro: ConsG<S, D, Self::Energy>, mag: Option<Tensor<S, D>>) -> Self;
}

/// marker: a conserved state that carries NO magnetic field (pure hydro). gates the typestate
/// `set_initial` that reaches `Ready` in one call (no staggered faces owed). disjoint from
/// [`Magnetic`] by concrete type (`ConsG` vs `MhdConsG`) — no coherence overlap.
pub trait NonMagnetic {}

/// marker: a conserved state that carries a magnetic field (MHD). gates the typestate that owes
/// staggered face seeding (`seed_faces`) before reaching `Ready`.
pub trait Magnetic {}

// every pure-hydro ConsG is NonMagnetic.
impl<S: Scalar, const D: usize, E: EnergyModel> NonMagnetic for ConsG<S, D, E> {}

// pure hydro: the conserved state IS the hydro ConsG; no magnetic field.
impl<S: Scalar, const D: usize, E: EnergyModel> SeedableCons<S, D> for ConsG<S, D, E> {
    type Energy = E;
    #[inline]
    fn hydro_part(&self) -> ConsG<S, D, E> {
        *self
    }
    #[inline]
    fn mag_part(&self) -> Option<Tensor<S, D>> {
        None
    }
    #[inline]
    fn from_parts(hydro: ConsG<S, D, E>, _mag: Option<Tensor<S, D>>) -> Self {
        hydro
    }
}

/// build a primitive from its components laid out flat as `[rho, v_0 .. v_{D-1}, p]`, with the
/// pressure entry present exactly when the energy model carries one.
///
/// this is the shape a primitive state takes when it crosses a wire that has no types — a
/// configured stationary target arriving as a list of evaluated expressions, for instance — and
/// the trait is what lets one reader serve every energy-bearing and energy-free regime.
pub trait PrimFromSlots<S: Scalar, const D: usize> {
    fn from_slots(slots: &[S]) -> Self;
}

impl<S: Scalar, const D: usize, E: EnergyModel> PrimFromSlots<S, D> for PrimG<S, D, E> {
    #[inline]
    fn from_slots(slots: &[S]) -> Self {
        assert!(
            slots.len() >= 1 + D,
            "a {D}-dimensional primitive needs at least a density and {D} velocity component(s), \
             got {}",
            slots.len()
        );
        PrimG {
            rho: slots[0],
            vel: Tensor::new(std::array::from_fn(|k| slots[1 + k])),
            // an energy-free regime supplies no pressure and its slot discards whatever it is
            // handed, so the missing entry and a present one are the same state.
            pre: E::Slot::<S>::from_scalar(slots.get(1 + D).copied().unwrap_or(S::ZERO)),
        }
    }
}

// ---- backward-compatible type aliases ----

/// conservative variables (adiabatic). alias for ConsG<S, D, Adiabatic>.
pub type Cons<S, const D: usize> = ConsG<S, D, Adiabatic>;

/// primitive variables (adiabatic). alias for PrimG<S, D, Adiabatic>.
pub type Prim<S, const D: usize> = PrimG<S, D, Adiabatic>;

// ---- constructors ----

impl<S: Scalar, const D: usize, E: EnergyModel> PrimG<S, D, E> {
    pub fn zero() -> Self {
        PrimG {
            rho: S::ZERO,
            vel: Tensor::zeros(),
            pre: E::Slot::<S>::zero(),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> ConsG<S, D, E> {
    pub fn zero() -> Self {
        ConsG {
            chi: Default::default(),
            den: S::ZERO,
            mom: Tensor::zeros(),
            nrg: E::Slot::<S>::zero(),
        }
    }
}

// ---- conversions (adiabatic only — newtonian convenience methods) ----

impl<S: Scalar, const D: usize> Prim<S, D> {
    /// convert primitive to conservative variables.
    /// delegates nrg computation to eos.conserved_energy().
    /// for ideal gas: nrg = ke + rho * e_int (total energy).
    /// for isothermal: nrg = cs^2 (sound speed squared).
    pub fn to_conserved(&self, eos: &impl Eos<S>) -> Cons<S, D> {
        let rho = self.rho;
        let mom = self.vel.scale(rho);
        let v2 = self.vel.dot(&self.vel);
        let nrg = eos.conserved_energy(rho, v2, self.pre);
        Cons {
            chi: Default::default(),
            den: rho,
            mom,
            nrg,
        }
    }
}

impl<S: Scalar, const D: usize> Cons<S, D> {
    /// convert conservative to primitive variables.
    /// delegates pressure recovery to eos.recover_pressure().
    /// for ideal gas: inverts total energy to get e_int, then p.
    /// for isothermal: reads cs^2 from nrg, p = cs^2 * rho.
    pub fn to_primitive(&self, eos: &impl Eos<S>) -> Prim<S, D> {
        let rho = self.den;
        let vel = self.mom.map(|m| m / rho);
        let v2 = vel.dot(&vel);
        let pre = eos.recover_pressure(rho, v2, self.nrg);
        Prim { rho, vel, pre }
    }
}

// ---- Default ----

impl<S: Scalar, const D: usize, E: EnergyModel> Default for PrimG<S, D, E> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Default for ConsG<S, D, E> {
    fn default() -> Self {
        Self::zero()
    }
}

// ---- arithmetic on PrimG (for reconstruction, averaging, time integration) ----

impl<S: Scalar, const D: usize, E: EnergyModel> Add for PrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        PrimG {
            rho: self.rho + rhs.rho,
            vel: self.vel + rhs.vel,
            pre: self.pre.add(rhs.pre),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Sub for PrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        PrimG {
            rho: self.rho - rhs.rho,
            vel: self.vel - rhs.vel,
            pre: self.pre.sub(rhs.pre),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Neg for PrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        PrimG {
            rho: -self.rho,
            vel: -self.vel,
            pre: self.pre.neg(),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> Mul<S> for PrimG<S, D, E> {
    type Output = Self;
    #[inline]
    fn mul(self, s: S) -> Self {
        PrimG {
            rho: self.rho * s,
            vel: self.vel.scale(s),
            pre: self.pre.scale(s),
        }
    }
}

// ---- arithmetic on ConsG (for flux differencing) ----

impl<S: Scalar, const D: usize, E: EnergyModel, X: DyeModel> Add for ConsG<S, D, E, X> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        ConsG {
            den: self.den + rhs.den,
            mom: self.mom + rhs.mom,
            nrg: self.nrg.add(rhs.nrg),
            chi: self.chi.add(rhs.chi),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel, X: DyeModel> Sub for ConsG<S, D, E, X> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        ConsG {
            den: self.den - rhs.den,
            mom: self.mom - rhs.mom,
            nrg: self.nrg.sub(rhs.nrg),
            chi: self.chi.sub(rhs.chi),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel, X: DyeModel> Neg for ConsG<S, D, E, X> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        ConsG {
            den: -self.den,
            mom: -self.mom,
            nrg: self.nrg.neg(),
            chi: self.chi.neg(),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel, X: DyeModel> Mul<S> for ConsG<S, D, E, X> {
    type Output = Self;
    #[inline]
    fn mul(self, s: S) -> Self {
        ConsG {
            den: self.den * s,
            mom: self.mom.scale(s),
            nrg: self.nrg.scale(s),
            chi: self.chi.scale(s),
        }
    }
}

// ---- Selectable impls ----

impl<S: Scalar, const D: usize, E: EnergyModel> symbi_ir::algebra::Selectable<S> for ConsG<S, D, E>
where
    S::Mask: Copy,
{
    #[inline]
    fn select(m: S::Mask, yes: Self, no: Self) -> Self {
        ConsG {
            chi: Default::default(),
            den: <S as Scalar>::select(m, yes.den, no.den),
            mom: <Tensor<S, D> as symbi_ir::algebra::Selectable<S>>::select(m, yes.mom, no.mom),
            nrg: EnergySlot::select_mask(m, yes.nrg, no.nrg),
        }
    }
}

impl<S: Scalar, const D: usize, E: EnergyModel> symbi_ir::algebra::Selectable<S> for PrimG<S, D, E>
where
    S::Mask: Copy,
{
    #[inline]
    fn select(m: S::Mask, yes: Self, no: Self) -> Self {
        PrimG {
            rho: <S as Scalar>::select(m, yes.rho, no.rho),
            vel: <Tensor<S, D> as symbi_ir::algebra::Selectable<S>>::select(m, yes.vel, no.vel),
            pre: EnergySlot::select_mask(m, yes.pre, no.pre),
        }
    }
}

// ---- FieldElement impls ----

// safety: ConsG<f64, D, Adiabatic> is (f64, Tensor<f64, D>, f64) — contiguous, fixed-size,
// zero bytes produce valid values (0.0, zeros, 0.0).
unsafe impl<const D: usize> FieldElement for ConsG<f64, D, Adiabatic> {
    type Scalar = f64;
}
unsafe impl<const D: usize> FieldElement for ConsG<f32, D, Adiabatic> {
    type Scalar = f32;
}

// safety: PrimG<f64, D, Adiabatic> is (f64, Tensor<f64, D>, f64) — same layout guarantees.
unsafe impl<const D: usize> FieldElement for PrimG<f64, D, Adiabatic> {
    type Scalar = f64;
}
unsafe impl<const D: usize> FieldElement for PrimG<f32, D, Adiabatic> {
    type Scalar = f32;
}

// safety: ConsG<f64, D, IsoModel> is (f64, Tensor<f64, D>) — Zero<f64> is ZST.
// zero bytes produce valid values.
unsafe impl<const D: usize> FieldElement for ConsG<f64, D, IsoModel> {
    type Scalar = f64;
}
unsafe impl<const D: usize> FieldElement for ConsG<f32, D, IsoModel> {
    type Scalar = f32;
}

// safety: PrimG<f64, D, IsoModel> is (f64, Tensor<f64, D>) — Zero<f64> is ZST.
unsafe impl<const D: usize> FieldElement for PrimG<f64, D, IsoModel> {
    type Scalar = f64;
}
unsafe impl<const D: usize> FieldElement for PrimG<f32, D, IsoModel> {
    type Scalar = f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::{EnergySlot, Zero};
    use crate::eos::IdealGas;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-13 * a.abs().max(b.abs()).max(1.0)
    }

    // ---- adiabatic tests (backward compat — unchanged behavior) ----

    #[test]
    fn prim_cons_roundtrip_1d() {
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 2.0,
        };
        let cons = prim.to_conserved(&eos);
        let prim2 = cons.to_primitive(&eos);
        assert!(approx(prim.rho, prim2.rho));
        assert!(approx(prim.vel[0], prim2.vel[0]));
        assert!(approx(prim.pre, prim2.pre));
    }

    #[test]
    fn prim_cons_roundtrip_3d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim {
            rho: 0.5,
            vel: Tensor::new([1.0, -0.3, 0.7]),
            pre: 0.1,
        };
        let cons = prim.to_conserved(&eos);
        let prim2 = cons.to_primitive(&eos);
        assert!(approx(prim.rho, prim2.rho));
        for dd in 0..3 {
            assert!(approx(prim.vel[dd], prim2.vel[dd]));
        }
        assert!(approx(prim.pre, prim2.pre));
    }

    #[test]
    fn conserved_values_1d() {
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        };
        let cons = prim.to_conserved(&eos);
        assert!(approx(cons.den, 1.0));
        assert!(approx(cons.mom[0], 0.0));
        // nrg = 0.5*rho*v^2 + p/(gamma-1) = 0 + 1.0/0.4 = 2.5
        assert!(approx(cons.nrg, 2.5));
    }

    #[test]
    fn cons_arithmetic() {
        let a = Cons::<f64, 2> {
            chi: Default::default(),
            den: 1.0,
            mom: Tensor::new([2.0, 3.0]),
            nrg: 4.0,
        };
        let b = Cons::<f64, 2> {
            chi: Default::default(),
            den: 0.5,
            mom: Tensor::new([1.0, 0.5]),
            nrg: 2.0,
        };
        let sum = a + b;
        assert!(approx(sum.den, 1.5));
        assert!(approx(sum.mom[0], 3.0));
        assert!(approx(sum.mom[1], 3.5));
        assert!(approx(sum.nrg, 6.0));

        let diff = a - b;
        assert!(approx(diff.den, 0.5));
        assert!(approx(diff.mom[0], 1.0));

        let neg = -a;
        assert!(approx(neg.den, -1.0));

        let scaled = a * 2.0;
        assert!(approx(scaled.den, 2.0));
        assert!(approx(scaled.mom[1], 6.0));
    }

    // ---- isothermal conversions ----

    #[test]
    fn isothermal_cons_stores_cs_squared() {
        let eos = crate::eos::Isothermal { cs: 2.0 };
        let prim = Prim {
            rho: 3.0,
            vel: Tensor::new([1.0]),
            pre: 12.0,
        }; // p = cs^2*rho = 4*3
        let cons = prim.to_conserved(&eos);
        assert!(approx(cons.den, 3.0));
        assert!(approx(cons.mom[0], 3.0));
        // nrg slot holds cs^2 = 4.0 in the isothermal model; it carries no total energy
        assert!(approx(cons.nrg, 4.0));
    }

    #[test]
    fn isothermal_roundtrip_1d() {
        let eos = crate::eos::Isothermal { cs: 1.5 };
        let prim = Prim {
            rho: 2.0,
            vel: Tensor::new([0.7]),
            pre: 4.5,
        }; // p = cs^2*rho = 2.25*2
        let cons = prim.to_conserved(&eos);
        let prim2 = cons.to_primitive(&eos);
        assert!(approx(prim.rho, prim2.rho));
        assert!(approx(prim.vel[0], prim2.vel[0]));
        assert!(approx(prim.pre, prim2.pre));
    }

    #[test]
    fn isothermal_roundtrip_3d() {
        let eos = crate::eos::Isothermal { cs: 0.5 };
        let rho = 4.0;
        let pre = 0.25 * rho; // cs^2 * rho = 0.25 * 4 = 1.0
        let prim = Prim {
            rho,
            vel: Tensor::new([1.0, -0.5, 0.3]),
            pre,
        };
        let cons = prim.to_conserved(&eos);
        let prim2 = cons.to_primitive(&eos);
        assert!(approx(prim.rho, prim2.rho));
        for dd in 0..3 {
            assert!(approx(prim.vel[dd], prim2.vel[dd]));
        }
        assert!(approx(prim.pre, prim2.pre));
    }

    #[test]
    fn locally_isothermal_recover() {
        // simulate locally isothermal: different cs^2 per cell stored in nrg.
        // use a "dummy" global eos — recover_pressure reads cs^2 from nrg.
        let eos = crate::eos::Isothermal { cs: 0.0 }; // global cs irrelevant
        let local_cs_sq = 9.0; // local sound speed squared
        let cons = Cons {
            chi: Default::default(),
            den: 2.0,
            mom: Tensor::new([1.0]),
            nrg: local_cs_sq,
        };
        let prim = cons.to_primitive(&eos);
        assert!(approx(prim.rho, 2.0));
        assert!(approx(prim.vel[0], 0.5));
        // p = nrg * rho = 9.0 * 2.0 = 18.0
        assert!(approx(prim.pre, 18.0));
    }

    #[test]
    fn prim_arithmetic() {
        let a = Prim::<f64, 2> {
            rho: 1.0,
            vel: Tensor::new([2.0, 3.0]),
            pre: 4.0,
        };
        let b = Prim::<f64, 2> {
            rho: 0.5,
            vel: Tensor::new([1.0, 0.5]),
            pre: 2.0,
        };

        let sum = a + b;
        assert!(approx(sum.rho, 1.5));
        assert!(approx(sum.vel[0], 3.0));
        assert!(approx(sum.vel[1], 3.5));
        assert!(approx(sum.pre, 6.0));

        let diff = a - b;
        assert!(approx(diff.rho, 0.5));
        assert!(approx(diff.vel[0], 1.0));
        assert!(approx(diff.pre, 2.0));

        let neg = -a;
        assert!(approx(neg.rho, -1.0));
        assert!(approx(neg.vel[1], -3.0));

        let scaled = a * 2.0;
        assert!(approx(scaled.rho, 2.0));
        assert!(approx(scaled.vel[0], 4.0));
        assert!(approx(scaled.pre, 8.0));
    }

    #[test]
    fn zero_constructors() {
        let p = Prim::<f64, 3>::zero();
        assert_eq!(p.rho, 0.0);
        assert_eq!(p.vel, Tensor::new([0.0, 0.0, 0.0]));
        assert_eq!(p.pre, 0.0);

        let c = Cons::<f64, 2>::zero();
        assert_eq!(c.den, 0.0);
        assert_eq!(c.mom, Tensor::new([0.0, 0.0]));
        assert_eq!(c.nrg, 0.0);
    }

    // ---- correctness layer: isothermal ConsG/PrimG ----

    #[test]
    fn iso_cons_is_smaller() {
        // adiabatic 1d: den(8) + mom(8) + nrg(8) = 24
        assert_eq!(std::mem::size_of::<Cons<f64, 1>>(), 24);
        // isothermal 1d: den(8) + mom(8) + Zero(0) = 16
        assert_eq!(std::mem::size_of::<ConsG<f64, 1, IsoModel>>(), 16);
    }

    #[test]
    fn iso_prim_is_smaller() {
        assert_eq!(std::mem::size_of::<Prim<f64, 1>>(), 24);
        assert_eq!(std::mem::size_of::<PrimG<f64, 1, IsoModel>>(), 16);
    }

    #[test]
    fn iso_consg_arithmetic() {
        let a = ConsG::<f64, 2, IsoModel> {
            chi: Default::default(),
            den: 1.0,
            mom: Tensor::new([2.0, 3.0]),
            nrg: Zero::default(),
        };
        let b = ConsG::<f64, 2, IsoModel> {
            chi: Default::default(),
            den: 0.5,
            mom: Tensor::new([1.0, 0.5]),
            nrg: Zero::default(),
        };

        let sum = a + b;
        assert!(approx(sum.den, 1.5));
        assert!(approx(sum.mom[0], 3.0));
        assert_eq!(sum.nrg.value(), 0.0); // always zero

        let diff = a - b;
        assert!(approx(diff.den, 0.5));
        assert_eq!(diff.nrg.value(), 0.0);

        let neg = -a;
        assert!(approx(neg.den, -1.0));
        assert_eq!(neg.nrg.value(), 0.0);

        let scaled = a * 2.0;
        assert!(approx(scaled.den, 2.0));
        assert_eq!(scaled.nrg.value(), 0.0);
    }

    #[test]
    fn iso_primg_arithmetic() {
        let a = PrimG::<f64, 2, IsoModel> {
            rho: 1.0,
            vel: Tensor::new([2.0, 3.0]),
            pre: Zero::default(),
        };
        let b = PrimG::<f64, 2, IsoModel> {
            rho: 0.5,
            vel: Tensor::new([1.0, 0.5]),
            pre: Zero::default(),
        };

        let sum = a + b;
        assert!(approx(sum.rho, 1.5));
        assert!(approx(sum.vel[0], 3.0));
        assert_eq!(sum.pre.value(), 0.0);

        let scaled = a * 2.0;
        assert!(approx(scaled.rho, 2.0));
        assert_eq!(scaled.pre.value(), 0.0);
    }

    #[test]
    fn iso_consg_zero() {
        let c = ConsG::<f64, 3, IsoModel>::zero();
        assert_eq!(c.den, 0.0);
        assert_eq!(c.mom, Tensor::new([0.0, 0.0, 0.0]));
        assert_eq!(c.nrg.value(), 0.0);
    }

    #[test]
    fn iso_consg_default_is_zero() {
        let c = ConsG::<f64, 2, IsoModel>::default();
        assert_eq!(c.den, 0.0);
        assert_eq!(c.nrg.value(), 0.0);
    }
}
