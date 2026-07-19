// =============================================================================
// algebra.rs
//
// minimal structural numeric primitives needed by `Tensor` / `Matrix` /
// `Indexed` impls inside this crate. NOT a re-implementation of `Scalar` —
// the production `Scalar` (and `Selectable`) live in `symbi_ir::algebra`.
// this trait is `pub(crate)`-equivalent: nothing outside `symbi-algebra` should
// reference it. its role is to provide arithmetic / sqrt / min / max / abs /
// ZERO / ONE / from_f64 so dot / norm / det / inv etc. compile without
// depending on `symbi-ir` (that would be a circular dep — `symbi-ir` already
// depends on `symbi-algebra` for `Tensor`, `FieldElement`, `Domain`).
//
// the production `Scalar` in `symbi_ir::algebra` carries this structural surface
// as part of its trait bag, so a workspace-wide `S: symbi_ir::algebra::Scalar`
// automatically satisfies `S: symbi_algebra::Numeric` and the matrix/tensor
// methods work over that carrier.
// =============================================================================

use std::ops::{
    Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg,
};

/// structural numeric bag: what the in-crate `Tensor` / `Matrix` / `Indexed`
/// methods need from a scalar. NOT a production trait — `symbi_ir::algebra::Scalar`
/// is the production carrier-generic surface; this is the minimal subset that
/// breaks the `symbi-algebra` <-> `symbi-ir` dep cycle.
///
/// **deliberately NO `PartialOrd`** (Tier 1.7, 2026-05-30). the new `Scalar`
/// has `Numeric` as a super-trait, so `S: Scalar` does NOT have `PartialOrd`
/// — native `<` / `>` / `<=` / `>=` on a generic `S` does NOT compile, and
/// the A1 discipline (use `cmp_*` returning `Self::Mask`, then `select` or
/// `branch`) is compile-enforced. methods that genuinely need host-side
/// ordering bound on `OrderedNumeric` (below) instead, which is impl'd only
/// by host numeric types (f64/f32). tracing carriers (Gv) cannot be smuggled
/// into those methods — preserving "no silent A1 violation on Gv."
pub trait Numeric:
    Copy + 'static + Default
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign + SubAssign + MulAssign + DivAssign
{
    const ZERO: Self;
    const ONE:  Self;

    fn from_f64(v: f64) -> Self;

    fn sqrt(self) -> Self;
    fn abs (self) -> Self;
    fn min (self, other: Self) -> Self;
    fn max (self, other: Self) -> Self;
}

/// `Numeric` + host-side ordering. impl'd only by concrete host numeric types
/// (`f64`, `f32`) — tracing carriers (`Gv`, future `Sym` / `Dual`) impl
/// `Numeric` but NOT `OrderedNumeric`, so they cannot be smuggled into methods
/// that branch on a host bool (`Tensor::normalize`, `Matrix::is_symmetric`,
/// and the host-only Riemann solvers `riemann/hlld.rs` / `riemann/hllc.rs`).
/// these methods are host computations on concrete tensors / matrices, not
/// part of carrier-generic physics.
pub trait OrderedNumeric: Numeric + PartialOrd {}

impl Numeric for f64 {
    const ZERO: Self = 0.0;
    const ONE:  Self = 1.0;
    #[inline(always)] fn from_f64(v: f64) -> Self { v }
    #[inline(always)] fn sqrt(self) -> Self { f64::sqrt(self) }
    // ternary form `x < 0 ? -x : x` / `a < b ? a : b` / `a > b ? a : b` — NOT
    // f64::abs/min/max. the std methods are NaN-symmetric and normalize signed
    // zero; the ternary is order-asymmetric. the traced IR lowers Min/Max/Abs to
    // exactly this select (scalarize), so the host carrier MUST match it bit-for-
    // bit or carrier equivalence breaks at NaN / signed-zero cells (tier-1 #2b).
    #[inline(always)] fn abs (self) -> Self { if self < 0.0 { -self } else { self } }
    #[inline(always)] fn min (self, other: Self) -> Self { if self < other { self } else { other } }
    #[inline(always)] fn max (self, other: Self) -> Self { if self > other { self } else { other } }
}
impl OrderedNumeric for f64 {}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE:  Self = 1.0;
    #[inline(always)] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline(always)] fn sqrt(self) -> Self { f32::sqrt(self) }
    // ternary form (see the f64 impl) — matches the traced IR's Min/Max/Abs
    // select lowering, whose NaN behavior differs from f32::abs/min/max.
    #[inline(always)] fn abs (self) -> Self { if self < 0.0 { -self } else { self } }
    #[inline(always)] fn min (self, other: Self) -> Self { if self < other { self } else { other } }
    #[inline(always)] fn max (self, other: Self) -> Self { if self > other { self } else { other } }
}
impl OrderedNumeric for f32 {}
