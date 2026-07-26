// =============================================================================
// primitives.rs
//
// LAYER 0 UNIVERSAL PRIMITIVES — categorical structures over the substrate.
//
// these are the type-level disciplines that physics (Layer 1, in symbi-hydro)
// depends on, but the IR substrate is paradigm-agnostic about. they discharge
// three axioms:
//
//   A3 (algebra)            — LinearSpace: opt-in, nominal algebraic structure.
//   A4 (indices + geometry) — Variance markers (re-exported), Geometry trait.
//   A5 (scope)              — Scoped<Sc, T> for multi-rank correctness.
//
// what's here:
//   - variance              — Indexed<V, S, D> (re-exported from symbi-algebra);
//                             same-variance arithmetic compiles, mixed is a
//                             compile error; metric-free contraction.
//   - Scope + Scoped        — per-rank vs cross-rank value discipline (DORMANT
//                             until multi-rank code arrives; retained so the
//                             eventual MPI lift is a non-breaking add).
//   - LinearSpace           — types that declare additive + scalar-multiplicative
//                             structure (Cons + Cons OK; Prim + Prim won't compile
//                             because Prim doesn't impl LinearSpace).
//   - Geometry              — the metric-bearing manifold. FlatCartesian (ZST,
//                             identity metric) is the working impl; Cylindrical
//                             and Spherical are forward placeholders until the
//                             Regime-trait geometry lift forces them.
//
// these traits have no downstream consumers (physics uses the legacy
// `symbi_algebra::Scalar` + ad-hoc trait bounds).
//
// references:
//   - crate::algebra        — the new `Scalar` / `Mask` these traits build on
//   - symbi-algebra::variance — `Indexed`/`Upper`/`Lower` (used unchanged)
// =============================================================================

#![deny(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

use crate::algebra::{Scalar, Selectable};

// =============================================================================
// section 1 — variance: re-exported from symbi-algebra (where the concrete
// `Indexed<V, S, D>` repr-transparent over `Tensor<S, D>` already lives).
//
// arithmetic only within same variance (Upper + Upper compiles, Upper + Lower
// has no impl → compile error). `contract(v, w)` is the metric-free pairing.
// see crates/symbi-algebra/src/variance.rs for the implementation.
// =============================================================================

pub use symbi_algebra::{Contravariant, Covariant, Indexed, Lower, Upper, contract};

// =============================================================================
// section 2 — Scope: per-rank vs cross-rank value discipline (A5).
//
// arithmetic between same-scope values compiles; cross-scope arithmetic has
// NO impl → compile error. crossing scopes is explicit via `elevate` (Local
// → Global, triggers communication when ranks > 1) or `localize` (Global
// → Local, trivial — every rank already agrees).
//
// dormant: no multi-rank consumer; retained for the MPI lift as a
// non-breaking type-discipline addition.
// =============================================================================

pub trait Scope: 'static + Copy + Default {}

/// "this rank only" — value computed from local data, still awaiting cross-rank reconciliation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Local;
impl Scope for Local {}

/// "agreed across all ranks in the default communicator" — value reconciled
/// via collective communication.
#[derive(Clone, Copy, Debug, Default)]
pub struct Global;
impl Scope for Global {}

/// a value annotated with the scope over which it is meaningful. zero runtime
/// cost (`#[repr(transparent)]` over `T`); cross-scope arithmetic is a compile
/// error.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Scoped<Sc: Scope, T> {
    value: T,
    _scope: PhantomData<Sc>,
}

impl<Sc: Scope, T> Scoped<Sc, T> {
    /// wrap a value at the given scope. caller asserts the value genuinely
    /// lives in this scope; this is a compile-time TYPE assertion and performs no runtime check.
    pub const fn new(value: T) -> Self {
        Self {
            value,
            _scope: PhantomData,
        }
    }
    /// borrow the inner value.
    pub fn get(&self) -> &T {
        &self.value
    }
    /// consume and return the inner value (loses the scope tag).
    pub fn into_inner(self) -> T {
        self.value
    }
    /// scope-preserving map: U inherits the scope of self.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Scoped<Sc, U> {
        Scoped {
            value: f(self.value),
            _scope: PhantomData,
        }
    }
}

impl<T: Copy> Scoped<Global, T> {
    /// every rank already agrees on this value → narrowing to Local is free.
    pub fn localize(self) -> Scoped<Local, T> {
        Scoped {
            value: self.value,
            _scope: PhantomData,
        }
    }
}

// arithmetic preserved within a single scope; cross-scope has no impl → compile error.

impl<Sc: Scope, T: Add<Output = T>> Add for Scoped<Sc, T> {
    type Output = Scoped<Sc, T>;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}
impl<Sc: Scope, T: Sub<Output = T>> Sub for Scoped<Sc, T> {
    type Output = Scoped<Sc, T>;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}
impl<Sc: Scope, T: Mul<Output = T>> Mul for Scoped<Sc, T> {
    type Output = Scoped<Sc, T>;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}
impl<Sc: Scope, T: Neg<Output = T>> Neg for Scoped<Sc, T> {
    type Output = Scoped<Sc, T>;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}

// =============================================================================
// section 3 — LinearSpace: opt-in algebraic structure for state types (A3).
//
// nominal — a type declares membership by impl. the compiler enforces what
// arithmetic compiles. for hydro: `Cons<S, D>` impls LinearSpace (Cons + Cons,
// Cons * scalar); `Prim<S, D>` deliberately does NOT (Prim + Prim won't
// compile because no `Add` impl).
//
// a future `#[derive(LinearSpace)]` macro will verify every field impls
// LinearSpace over the same Scalar and emit Add/Sub/Mul/select fieldwise.
// =============================================================================

pub trait LinearSpace:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<<Self as LinearSpace>::Field, Output = Self>
    + Selectable<<Self as LinearSpace>::Field>
{
    type Field: Scalar;
}

/// every Scalar is trivially a linear space over itself.
impl<S: Scalar> LinearSpace for S {
    type Field = S;
}

// =============================================================================
// section 4 — Geometry: the metric-bearing manifold (A4).
//
// the index-to-physics bridge: methods take cell index (or a position derived
// from one) and return metric quantities. FlatCartesian is a ZST with the
// identity metric — the optimizer eliminates every method call through
// monomorphization. cylindrical/spherical/schwarzschild-ks add a non-trivial metric.
//
// NOT connected to the Regime trait in symbi-hydro — physics handles
// curvilinear cases ad-hoc. lifting Geometry into Regime (so flux/wave_speed
// take `&G`) is the contract this trait defines.
// =============================================================================

pub trait Geometry<S: Scalar, const D: usize>: 'static + Copy + Default {
    /// physical coordinate for a given cell index. for flat cartesian with
    /// unit cells: `x_i = ii`.
    fn physical_coord(&self, idx: [isize; D]) -> Contravariant<S, D>;

    /// sqrt of the metric determinant at position `x`; identity (== 1) for flat.
    fn sqrt_det_g(&self, x: &Contravariant<S, D>) -> S;

    /// cell volume at position `x` (includes the `sqrt_det_g` factor).
    fn cell_volume(&self, x: &Contravariant<S, D>) -> S;

    /// face area perpendicular to `axis` at position `x`.
    fn face_area(&self, x: &Contravariant<S, D>, axis: usize) -> S;
}

/// flat cartesian — identity metric, unit cells. ZST: every method
/// monomorphizes to a constant the optimizer eliminates.
#[derive(Clone, Copy, Debug, Default)]
pub struct FlatCartesian;

impl<S: Scalar, const D: usize> Geometry<S, D> for FlatCartesian {
    fn physical_coord(&self, idx: [isize; D]) -> Contravariant<S, D> {
        let mut arr = [S::zero(); D];
        for ii in 0..D {
            arr[ii] = S::from_f64(idx[ii] as f64);
        }
        Indexed::from_array(arr)
    }
    fn sqrt_det_g(&self, _x: &Contravariant<S, D>) -> S {
        S::one()
    }
    fn cell_volume(&self, _x: &Contravariant<S, D>) -> S {
        S::one()
    }
    fn face_area(&self, _x: &Contravariant<S, D>, _axis: usize) -> S {
        S::one()
    }
}

/// cylindrical (r, phi, z) — PLACEHOLDER. real impl lands when the
/// Regime-trait geometry lift forces curvature-aware face areas / volumes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cylindrical;

/// spherical (r, theta, phi) — PLACEHOLDER. same caveat as `Cylindrical`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spherical;

// =============================================================================
// section 5 — tests: contract sanity.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Scope: ABI ------------------------------------------------------
    #[test]
    fn scoped_is_repr_transparent_over_t() {
        assert_eq!(
            std::mem::size_of::<Scoped<Local, f64>>(),
            std::mem::size_of::<f64>()
        );
        assert_eq!(
            std::mem::size_of::<Scoped<Global, f64>>(),
            std::mem::size_of::<f64>()
        );
    }

    // ---- Scope: same-scope arithmetic ------------------------------------
    #[test]
    fn scoped_same_scope_adds() {
        let aa: Scoped<Local, f64> = Scoped::new(1.0);
        let bb: Scoped<Local, f64> = Scoped::new(2.0);
        let cc = aa + bb;
        assert_eq!(*cc.get(), 3.0);
    }

    #[test]
    fn scoped_global_arithmetic_independent_of_local() {
        let aa: Scoped<Global, f64> = Scoped::new(5.0);
        let bb: Scoped<Global, f64> = Scoped::new(7.0);
        let cc = aa - bb;
        assert_eq!(*cc.get(), -2.0);
    }

    // ---- Scope: Global → Local is free -----------------------------------
    #[test]
    fn scoped_global_localize_preserves_value() {
        let gg: Scoped<Global, f64> = Scoped::new(42.0);
        let ll = gg.localize();
        assert_eq!(*ll.get(), 42.0);
    }

    // ---- Scope: map preserves scope --------------------------------------
    #[test]
    fn scoped_map_preserves_scope() {
        let gg: Scoped<Global, f64> = Scoped::new(2.0);
        let gg2 = gg.map(|x| x * 3.0);
        assert_eq!(*gg2.get(), 6.0);
    }

    // ---- LinearSpace blanket for Scalar ----------------------------------
    #[test]
    fn scalar_is_linear_space_over_itself() {
        fn assert_ls<L: LinearSpace>() {}
        assert_ls::<f64>();
    }

    // ---- Geometry: FlatCartesian invariants ------------------------------
    #[test]
    fn flat_cartesian_metric_is_identity() {
        let geom = FlatCartesian;
        let xx: Contravariant<f64, 3> = geom.physical_coord([2, 3, 5]);
        assert_eq!(xx[0], 2.0);
        assert_eq!(xx[1], 3.0);
        assert_eq!(xx[2], 5.0);
        assert_eq!(
            <FlatCartesian as Geometry<f64, 3>>::sqrt_det_g(&geom, &xx),
            1.0
        );
        assert_eq!(
            <FlatCartesian as Geometry<f64, 3>>::cell_volume(&geom, &xx),
            1.0
        );
        assert_eq!(
            <FlatCartesian as Geometry<f64, 3>>::face_area(&geom, &xx, 1),
            1.0
        );
    }

    #[test]
    fn flat_cartesian_zst() {
        assert_eq!(std::mem::size_of::<FlatCartesian>(), 0);
    }

    // ---- variance re-export sanity ---------------------------------------
    #[test]
    fn variance_indexed_constructible_via_reexport() {
        let vv: Contravariant<f64, 3> = Indexed::from_array([1.0, 2.0, 3.0]);
        let ww: Covariant<f64, 3> = Indexed::from_array([4.0, 5.0, 6.0]);
        // metric-free contraction: v^i w_i = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(contract(&vv, &ww), 32.0);
    }
}
