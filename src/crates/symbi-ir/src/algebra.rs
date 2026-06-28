// =============================================================================
// algebra.rs
//
// THE SYMBI CODEGEN SUBSTRATE — mathematical contract.
//
// status: TIER 0 CLOSED (2026-05-30). the trait surface is feature-complete
// against the known kernel set (iterate/iterate_vec with freeze law, hyperbolics
// for RMHD quartic, infinity/nan/is_nan, branch default for state-typed
// conditionals, FieldLoad/IterateInline payload pinned in doc). still NOT
// wired into the workspace as the active `Scalar` trait — the Tier 1 atomic
// cutover from `symbi_algebra::Scalar` is the next step (see TODO_2026-05-30.md).
//
// constitution voice: this file preserves UPPERCASE EMPHASIS on load-bearing
// invariants (A1, FREEZE LAW, NEVER USE) and lowercase for narration. that's
// the deliberate register; new additions match it.
//
// the IR is a FREE ALGEBRA over the operation signature defined in section 1.
// every Carrier (each `Scalar` impl) is a TOTAL HOMOMORPHISM from this free
// algebra into a concrete algebra: `f64` evaluates, `Gv` traces into IR,
// future `Sym` interprets symbolically, future `Dual<C>` carries derivatives.
// lowering (CPU emit, CUDA emit) is also a homomorphism — into a source-code
// algebra. rewrites are equational laws preserved by every homomorphism.
//
// invariant: this module contains ZERO `panic!`/`unwrap`/`expect`. category
// errors (carrier dialect, variance, scope, algebra, theory composition,
// naming) MUST be compile errors. admitted panics live only at I/O / driver
// boundaries (config parse, NVRTC compile, HDF5 write) — outside this file.
//
// references:
//   - docs/design/00_axioms.md             — the axioms this code discharges
//   - docs/FOUNDATION_gv_reconciliation.md — the Gv carrier story (Gen-3)
//   - reviews/v2-synthesis.md §5           — the runtime-panic hole this closes
// =============================================================================

#![deny(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use std::ops::{
    Add, AddAssign, BitAnd, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

use symbi_algebra::FieldElement;

// =============================================================================
// section 1 — the Op signature.
//
// the IR is `Free<Op>` — terms built by composition of the symbols below.
// adding to this set is a ONE-WAY DOOR: every existing Carrier impl must be
// extended in lockstep. defer additions until a real consumer forces them.
// =============================================================================

/// the algebraic laws every `Scalar` carrier satisfies. these are not prose: each is an
/// EXECUTABLE PROPERTY swept over a deterministic sample grid on both concrete carriers
/// (f64 + f32) in `tests/carrier_laws.rs`, classified exact-vs-floating; the homomorphism
/// law (f64 == traced-Gv) is in `tests/carrier_oracle_new.rs`. (they also drive the rewrite
/// system, deferred per the rent test.)
pub mod laws {
    //! ## ring + abelian-group
    //! - Add: associative, commutative, identity Zero.
    //! - Mul: associative, commutative, identity One.
    //! - Neg: involution. `Neg(Neg(x)) == x`.
    //! - Sub: defined by `Sub(x, y) == Add(x, Neg(y))`.
    //! - Div: defined by `Div(x, y) == Mul(x, Recip(y))`. `Recip` undefined at zero.
    //! - distrib: Mul distributes over Add.
    //!
    //! ## comparisons
    //! - CmpEq: reflexive, symmetric.
    //! - CmpLt: irreflexive, transitive. `CmpLt(x, x) == false`.
    //!
    //! ## select
    //! - `Select(true,  t, _) == t`
    //! - `Select(false, _, f) == f`
    //! - Select distributes over scalar ops: `Select(m, op(t), op(u)) == op(Select(m, t, u))`
    //!
    //! ## transcendentals
    //! - `Sqrt(x * x) == Abs(x)`
    //! - `Sqrt(x) * Sqrt(x) == x`     when `x >= 0`
    //! - `Exp(Ln(x))  == x`           when `x >  0`
    //! - `Ln(Exp(x))  == x`
    //!
    //! ## hyperbolics
    //! - `Cosh(x)^2 - Sinh(x)^2 == 1`
    //! - `Tanh(x) == Sinh(x) / Cosh(x)`
    //! - `Sinh(2x) == 2 * Sinh(x) * Cosh(x)`
    //! - `Asinh(Sinh(x))  == x`            (total on R)
    //! - `Acosh(Cosh(x))  == Abs(x)`       (asymmetric, like `Sqrt(x*x)`)
    //! - `Atanh(x)` total on `(-1, 1)`; `Acosh(x)` total on `[1, +inf)`.
    //!
    //! ## iteration — the FREEZE LAW (load-bearing)
    //! `IterateInline` is fixed-count, but every Carrier MUST return the OLD
    //! accumulator from the step where `converged` first fires:
    //!
    //! ```text
    //! after step k where converged(acc_k, body(acc_k)) holds,
    //! the accumulator stays at acc_k for all remaining steps.
    //! ```
    //!
    //! host carriers (f64, f32): early-return on convergence.
    //! traced carriers (Gv): `acc' = select(converged, acc, body(acc))` per
    //!   component; the body runs once, the count bakes in, the freeze
    //!   preserves the converged value through the remaining baked steps.
    //! VIOLATING THIS makes the traced kernel disagree with the host loop on
    //! non-convergent inputs (the c2p / kepler regression class).
    //!
    //! ## NaN
    //! NaN is the only value where `x.cmp_eq(x) == false`. callers MUST use
    //! `is_nan(x)`, NEVER `x.cmp_eq(S::nan())` — the latter is always false.
    //!
    //! ## the homomorphism law (A1 stated precisely)
    //! for every Carrier `S`, every Op `f` of arity N, and every term `a_i`:
    //!
    //! ```text
    //! interp_S(f(a_1, ..., a_N)) == f_S(interp_S(a_1), ..., interp_S(a_N))
    //! ```
    //!
    //! this is what "carrier polymorphism" means. semantically-equivalent
    //! terms in the free algebra must interpret to equal values in S.
}

// =============================================================================
// section 2 — Mask: the carrier-polymorphic boolean.
//
// on `f64`, Mask = `bool`. on `Gv`, Mask = a graph-node handle. there is
// intentionally NO `Into<bool>`: that would re-open A1's runtime-panic hole
// (Gv → bool has no total definition).
// =============================================================================

pub trait Mask:
    Copy + Send + Sync + 'static + BitAnd<Output = Self> + BitOr<Output = Self> + Not<Output = Self>
{
}

impl Mask for bool {}

// =============================================================================
// section 3 — Scalar: a TOTAL homomorphism from Free<Op> into a concrete algebra.
//
// invariants:
//   1. every Op in §1 has a corresponding method on this trait. partial impls
//      are an A1 violation (see reviews/v2-synthesis.md §5).
//   2. NO `PartialOrd`. native `<`, `>`, `<=`, `>=` on `S: Scalar` must NOT
//      compile in generic code. comparisons return `S::Mask`.
//   3. NO `to_f64` on the trait. extracting a concrete value from a Carrier
//      is a debug-only side channel that does not live in the algebraic core.
// =============================================================================

pub trait Scalar:
    Copy
    + Send
    + Sync
    + 'static
    + Default
    + FieldElement<Scalar = Self>
    + symbi_algebra::algebra::Numeric
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + std::iter::Sum
    + std::fmt::Debug
    + std::fmt::Display
{
    /// the carrier-polymorphic boolean. `bool` for `f64`; a node handle for `Gv`.
    type Mask: Mask;

    // NOTE: `ZERO`, `ONE`, `from_f64`, `sqrt`, `abs`, `min`, `max` are inherited
    // from the `symbi_algebra::algebra::Numeric` super-trait — DO NOT redeclare
    // them here (name-resolution ambiguity if both traits have them). Numeric is
    // a structural sub-bag needed by `Tensor` / `Matrix` / `Indexed` methods that
    // live in `symbi-algebra` (downstream of `symbi-ir`); Scalar requiring it
    // lets carrier-generic code uniformly call `S::ZERO` / `S::sqrt()` etc.
    // while keeping the dep graph `symbi-algebra <- symbi-ir`.

    // ── IEEE sentinels exclusive to Scalar (not in Numeric) ───────────────
    /// the positive IEEE infinity. wave-speed init / fold sentinel.
    const INFINITY: Self;
    /// the negative IEEE infinity.
    const NEG_INFINITY: Self;
    /// IEEE NaN. WARNING: `x.cmp_eq(Self::NAN)` is ALWAYS false (NaN is the
    /// only value where `x == x` fails). use `is_nan(x)` instead.
    const NAN: Self;

    // ── method-form alias of `Numeric::ZERO` / `Numeric::ONE` (ergonomics) ─
    #[inline]
    fn zero() -> Self {
        <Self as symbi_algebra::algebra::Numeric>::ZERO
    }
    #[inline]
    fn one() -> Self {
        <Self as symbi_algebra::algebra::Numeric>::ONE
    }

    /// HOST-BOUNDARY ESCAPE — NOT for carrier-generic physics.
    ///
    /// on `f64`/`f32` this is the identity. on tracing Carriers (Gv) this
    /// PANICS for a non-literal node — extracting a concrete value from a
    /// trace is an A1 violation by construction.
    ///
    /// callers using this MUST be at the host/emitter boundary (eos parameter
    /// read-back, host-side reduction reading device buffers, test diffs
    /// against analytic). new code in carrier-generic physics MUST NOT use
    /// this — decide with `cmp_*` / `select` / `branch` instead.
    fn to_f64(self) -> f64;

    // ── comparisons — return Mask, NEVER bool ─────────────────────────────
    fn cmp_lt(self, other: Self) -> Self::Mask;
    fn cmp_le(self, other: Self) -> Self::Mask;
    fn cmp_gt(self, other: Self) -> Self::Mask;
    fn cmp_ge(self, other: Self) -> Self::Mask;
    fn cmp_eq(self, other: Self) -> Self::Mask;

    // ── branch-free conditional ───────────────────────────────────────────
    /// invariant: on tracing Carriers, BOTH `t` and `f` are evaluated when
    /// the graph is built. callers MUST clamp NaN/Inf-producing ops (sqrt of
    /// maybe-negative, `1/x` near 0) BEFORE handing them to `select`. use
    /// `safe_sqrt` / `safe_p` / `g_clamp` idioms when unsure.
    fn select(m: Self::Mask, t: Self, f: Self) -> Self;

    // ── ordered-field operations ──────────────────────────────────────────
    // `sqrt` / `abs` / `min` / `max` are inherited from `Numeric`.
    fn recip(self) -> Self;

    // ── carrier-safe clamp idioms (CLAUDE.md §4.3) ────────────────────────
    // these make the prescribed `safe_sqrt` / `g_clamp` idioms CALLABLE rather than
    // hand-rolled per site. the Gv carrier evaluates BOTH arms of a `select`, so a
    // maybe-NaN op (sqrt of a negative, an out-of-domain transcendental) must be clamped
    // BEFORE the select or it traces a NaN into the kernel. these are for guarding the
    // UNSELECTED arm / roundoff on a provably-physical quantity — NOT for masking a
    // genuinely unphysical state, which must surface NaN ([[feedback_no_silent_floors]]).

    /// `sqrt(max(self, 0))` — the canonical clamp-before-sqrt. use when the radicand is
    /// non-negative for physical inputs and the clamp only guards the unselected arm /
    /// roundoff; never to hide a negative radicand that signals an unphysical state.
    #[inline]
    fn safe_sqrt(self) -> Self {
        self.max(Self::ZERO).sqrt()
    }

    /// clamp into `[lo, hi]` = `max(lo).min(hi)`. keeps a transcendental argument in-domain
    /// on BOTH carrier arms before a `select` (e.g., `acos` to `[-1, 1]`, `acosh` to
    /// `[1, +inf)`). callers pass `lo <= hi`.
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    // ── transcendentals (total on stated domain) ──────────────────────────
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, other: Self) -> Self;

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log10(self) -> Self;

    fn powi(self, n: i32) -> Self;
    fn powf(self, e: Self) -> Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;

    // ── hyperbolics (RMHD quartic, Cardano-Vieta hyperbolic branch) ───────
    /// total on R.
    fn sinh(self) -> Self;
    /// total on R; grows ~e^|x|/2 (overflow above |x| ~= 710).
    fn cosh(self) -> Self;
    /// total on R; bounded in `(-1, 1)`.
    fn tanh(self) -> Self;
    /// total on R.
    fn asinh(self) -> Self;
    /// total on `[1, +inf)`; NaN for `x < 1`.
    fn acosh(self) -> Self;
    /// total on `(-1, 1)`; NaN outside, +/-inf at the endpoints.
    fn atanh(self) -> Self;

    // ── IEEE sentinel methods (defaulted via consts) ──────────────────────
    #[inline]
    fn infinity() -> Self {
        Self::INFINITY
    }
    #[inline]
    fn neg_infinity() -> Self {
        Self::NEG_INFINITY
    }
    #[inline]
    fn nan() -> Self {
        Self::NAN
    }
    /// IEEE-correct NaN test: true for any NaN bit pattern, false for any
    /// finite value or +/-inf. defaulted via the `x == x` identity.
    #[inline]
    fn is_nan(self) -> Self::Mask {
        !self.cmp_eq(self)
    }

    // ── HIGHER-ORDER: trace-safe conditional for state-typed results ──────
    /// **NEVER use native `if cond { ... } else { ... }`** on a value
    /// derived from `S: Scalar`. that's an A1 VIOLATION on Gv: only the
    /// branch evaluated by the host is traced, and the traced kernel
    /// silently disagrees with the host's f64 eval. use `branch` instead.
    ///
    /// both `t` and `f` are evaluated; `Selectable::select` picks. on
    /// traced Carriers, both branches are built into the graph (IR
    /// semantics demand it).
    #[inline]
    fn branch<R: Selectable<Self>>(
        m: Self::Mask,
        t: impl FnOnce() -> R,
        f: impl FnOnce() -> R,
    ) -> R {
        R::select(m, t(), f())
    }

    // ── HIGHER-ORDER: trace-safe LAZY branch — the DUAL of `iterate` ───────
    /// a data-dependent conditional that evaluates ONLY the taken arm at
    /// runtime — unlike `select` / `branch`, which evaluate BOTH. this is the
    /// carrier-portable form of an early-out `if`: it lets carrier-generic
    /// physics skip a whole expensive arm (e.g., the RMHD quartic on a
    /// fast-path cell) instead of paying compute-all-paths.
    ///
    /// use `cond` where the arms have a LARGE cost asymmetry. for cheap,
    /// symmetric arms prefer `select` — a real branch adds CPU branch cost and
    /// (on GPU) warp-divergence risk that a `select` blend does not.
    ///
    /// semantics:
    /// - `S = f64`/`f32`: a real `if m { t() } else { f() }` — one arm runs.
    /// - `S = Gv`: traces each arm into its own subgraph and emits an
    ///   `Op::IfElse`, rendered as a real `if/else` on CPU AND CUDA. one arm
    ///   executes at runtime (a warp whose lanes disagree runs both — never
    ///   worse than `select`'s unconditional both-arms).
    /// - carrier-equivalence holds: f64 and the traced kernel take the SAME
    ///   arm for the same input, bit-identical.
    ///
    /// DEFAULT is the eager fallback (`select` of both arms) so any carrier is
    /// correct without an override; `f64`/`f32` override with a real branch and
    /// `Gv` overrides to trace `Op::IfElse`. scalar-result only (mirrors
    /// `scope`); the vector form `cond_vec` (dual of `iterate_vec`, for tuple
    /// returns) lands on `Op::IfElse`'s result vectors.
    #[inline]
    fn cond(m: Self::Mask, t: impl FnOnce() -> Self, f: impl FnOnce() -> Self) -> Self {
        Self::select(m, t(), f())
    }

    /// the N-OUTPUT lazy branch — the dual of `iterate_vec`. one branch, N
    /// results: the SHARED arm computation runs once and both outputs come from
    /// the SAME taken arm. this is what lets a `(sl, sr)` wave-speed fast-path
    /// skip the WHOLE quartic when `vsq ~ 0` / `bn ~ 0` (computing the quartic
    /// once in the else arm, not twice as two scalar `cond`s would, and not at
    /// all on the fast path). DEFAULT is eager componentwise `select`;
    /// `f64`/`f32` take a real branch; `Gv` traces ONE `Op::IfElse` with N
    /// results and returns N `Op::Proj` outputs. carrier-equivalent: every
    /// carrier takes the SAME arm for the same input.
    #[inline]
    fn cond_vec<const N: usize>(
        m: Self::Mask,
        t: impl FnOnce() -> [Self; N],
        f: impl FnOnce() -> [Self; N],
    ) -> [Self; N] {
        let tv = t();
        let fv = f();
        std::array::from_fn(|j| Self::select(m, tv[j], fv[j]))
    }

    // ── HIGHER-ORDER: bounded-pressure scope (docs/design/23) ─────────────
    /// declare a **bounded-pressure phase**: run `body`, return its result;
    /// on tracing Carriers, intermediates that were created inside the
    /// closure die at the closure's closing brace. lets nvcc / rustc see
    /// shorter live ranges → tighter register allocation.
    ///
    /// **call site usage:** group naturally-cohesive computation —
    /// reconstruction, wave-speed quartic, HLLE flux — inside its own
    /// `S::scope(|| { ... })` block. authors stay in idiomatic Rust nested
    /// blocks; the codegen handles the rest.
    ///
    /// **semantics:**
    /// - at `S = f64`: identity. the closure executes immediately; locals
    ///   die at the brace per normal Rust rules. zero overhead, no IR.
    /// - at `S = Gv` (step 3a, this session): identity, same as `f64`. the
    ///   closure traces nodes into the parent frame; no new `ScalarStmt::Scope`
    ///   is emitted yet. **step 3b** (separate sub-step) wires the Gv
    ///   override to open/close a frame on the trace and emit an `Op::Scope`
    ///   that scalarize lowers into a `ScalarStmt::Scope`. once step 3b
    ///   lands, all consumers that use this method get the perf win
    ///   automatically — no call-site changes.
    /// - any future carrier (smid/avx) can override per its own discipline.
    ///
    /// see `docs/design/23_bounded_pressure_ir.md` for the formal framing
    /// (interval-graph coloring, Sethi-Ullman pathwidth).
    ///
    /// **return type:** `Self` — the scope returns a single scalar of the
    /// same carrier. for multi-output phases (`(lo, hi)` tuples, etc.),
    /// use two adjacent scopes — the perf win is preserved because each
    /// scope's interior temps still die at its closing brace. constraining
    /// to `Self` is what lets the Gv override extract a NodeId from the
    /// result; a fully generic `R` would require a `Traceable` bound that
    /// callers would have to thread through every helper.
    #[inline]
    fn scope<F>(body: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        body()
    }

    // ── HIGHER-ORDER: bounded iteration, FREEZE LAW ───────────────────────
    /// fixed-count iteration with the FREEZE LAW (see `mod laws` and
    /// `Op::IterateInline`). `converged` returns `Self::Mask`, never `bool`.
    /// the body MUST be pure (`Fn`, not `FnMut`).
    ///
    /// **freeze law** — after the first step where `converged(acc, body(acc))`
    /// holds, the accumulator stays at the OLD acc for all remaining steps.
    /// host: early-return. traced: `acc' = select(converged, acc, body(acc))`
    /// — both branches built, the frozen arm preserves acc through the
    /// remaining baked steps. VIOLATING THIS reopens the kepler c2p
    /// regression class (host loop early-breaks, traced kernel runs past).
    fn iterate(
        self,
        max_steps: usize,
        body: impl Fn(Self) -> Self,
        converged: impl Fn(Self, Self) -> Self::Mask,
    ) -> Self;

    /// `N`-accumulator iteration. `result` selects which accumulator
    /// becomes the scalar return. the FREEZE LAW applies componentwise.
    fn iterate_vec<const N: usize>(
        init: [Self; N],
        max_steps: usize,
        body: impl Fn([Self; N]) -> [Self; N],
        converged: impl Fn([Self; N], [Self; N]) -> Self::Mask,
        result: usize,
    ) -> Self;
}

// =============================================================================
// section 4 — Selectable: state types lift `select` componentwise.
//
// every state type (`Cons`, `Prim`, `Flux`, etc.) participating in branch-free
// physics implements this. a future `#[derive(Selectable)]` will verify every
// field is `Selectable<S>` and emit `select` fieldwise.
// =============================================================================

pub trait Selectable<S: Scalar>: Sized + Copy {
    fn select(m: S::Mask, t: Self, f: Self) -> Self;

    /// compatibility alias. takes a Self-typed condition (encoded 0/1 in the
    /// 0.0/1.0 convention) instead of an explicit Mask. defaults to
    /// `select(cond.cmp_gt(ZERO), ...)` — i.e., "non-zero is true". prefer
    /// `select` with an explicit Mask in new code.
    #[inline]
    fn sel(cond: S, t: Self, f: Self) -> Self {
        Self::select(cond.cmp_gt(S::ZERO), t, f)
    }
}

/// every Scalar is trivially Selectable for itself.
impl<S: Scalar> Selectable<S> for S {
    #[inline(always)]
    fn select(m: S::Mask, t: Self, f: Self) -> Self {
        <S as Scalar>::select(m, t, f)
    }
}

/// tuple state types lift `select` componentwise. enables HLLC-style branches
/// where each arm returns a `(Cons, Flux)` pair.
impl<S: Scalar, A: Selectable<S>, B: Selectable<S>> Selectable<S> for (A, B)
where
    S::Mask: Copy,
{
    #[inline(always)]
    fn select(m: S::Mask, t: (A, B), f: (A, B)) -> (A, B) {
        (A::select(m, t.0, f.0), B::select(m, t.1, f.1))
    }
}

/// fixed-rank vector lift: select each component independently. lives HERE so
/// it can refer to the production trait while keeping `Tensor` in
/// `symbi-algebra`.
impl<S: Scalar, const N: usize> Selectable<S> for symbi_algebra::Tensor<S, N>
where
    S::Mask: Copy,
{
    #[inline]
    fn select(m: S::Mask, t: Self, f: Self) -> Self {
        symbi_algebra::Tensor::new(std::array::from_fn(|ii| {
            <S as Scalar>::select(m, t[ii], f[ii])
        }))
    }
}

// =============================================================================
// section 5 — SourceLoc: provenance preserved by every homomorphism.
//
// attached to every IR node at trace time. the homomorphism law forces every
// Target to preserve it through lowering — pretty rendering, source-mapping
// comments, audit displays all derive from this single annotation.
// =============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceLoc {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
    pub fn_name: &'static str,
    /// the let-binding name when known (e.g., `"cons_l"`); None for anonymous.
    /// supplied by callers via `source_loc!(binding = "cons_l")`.
    pub binding: Option<&'static str>,
}

impl SourceLoc {
    pub const UNKNOWN: Self = Self {
        file: "<unknown>",
        line: 0,
        column: 0,
        fn_name: "<unknown>",
        binding: None,
    };
}

/// caller-side helper for trace points.
///
/// usage:
///     let cons_l = trace_node(source_loc!(binding = "cons_l"), || { ... });
#[macro_export]
macro_rules! source_loc {
    () => {
        $crate::algebra::SourceLoc {
            file: file!(),
            line: line!(),
            column: column!(),
            fn_name: "<unknown>",
            binding: None,
        }
    };
    (binding = $name:literal) => {
        $crate::algebra::SourceLoc {
            file: file!(),
            line: line!(),
            column: column!(),
            fn_name: "<unknown>",
            binding: Some($name),
        }
    };
}

// =============================================================================
// section 6 — RenderPolicy: natural transformation across Target homomorphisms.
//
// the same IR renders multiple ways. each render is a homomorphism into a
// distinct source-code algebra. RenderPolicy selects which.
// =============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderPolicy {
    /// production — minified. no comments, anonymous temporaries. fastest
    /// downstream compile, smallest binary. the default for `cargo build --release`.
    Minified,
    /// debug / audit — preserves names, source-loc comments, section headers.
    /// for inspection; not the build path.
    Audit,
    /// RESERVED — not implemented. graph -> LaTeX is a non-trivial pretty-
    /// printer with no documentation-pipeline consumer today. emit returns
    /// `Err(RenderPolicyNotImplemented)` for this variant until a consumer
    /// earns it (rent test). do NOT thread this in production paths.
    Latex,
}

/// returned by an emitter when the caller selects a `RenderPolicy` whose
/// implementation does not yet exist (today: `RenderPolicy::Latex`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RenderPolicyNotImplemented {
    pub policy: RenderPolicy,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        RenderPolicy::Minified
    }
}

// =============================================================================
// section 7 — `Scalar for f64`: the canonical evaluating Carrier.
//
// every method is the immediate-evaluation implementation. the compiler inlines
// through `S: Scalar` to native f64 ops on hot paths.
// =============================================================================

impl Scalar for f64 {
    type Mask = bool;

    // ZERO / ONE inherited from `Numeric for f64`.
    const INFINITY: f64 = f64::INFINITY;
    const NEG_INFINITY: f64 = f64::NEG_INFINITY;
    const NAN: f64 = f64::NAN;

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn cmp_lt(self, b: f64) -> bool {
        self < b
    }
    #[inline(always)]
    fn cmp_le(self, b: f64) -> bool {
        self <= b
    }
    #[inline(always)]
    fn cmp_gt(self, b: f64) -> bool {
        self > b
    }
    #[inline(always)]
    fn cmp_ge(self, b: f64) -> bool {
        self >= b
    }
    #[inline(always)]
    fn cmp_eq(self, b: f64) -> bool {
        self == b
    }

    #[inline(always)]
    fn select(m: bool, t: f64, f: f64) -> f64 {
        if m { t } else { f }
    }

    // the lazy branch: a REAL `if` on the host — only the taken arm runs. this
    // is the f64 reference for the early-out branch cost (and the
    // oracle the traced `Op::IfElse` kernel is checked against).
    #[inline(always)]
    fn cond(m: bool, t: impl FnOnce() -> f64, f: impl FnOnce() -> f64) -> f64 {
        if m { t() } else { f() }
    }

    #[inline(always)]
    fn cond_vec<const N: usize>(
        m: bool,
        t: impl FnOnce() -> [f64; N],
        f: impl FnOnce() -> [f64; N],
    ) -> [f64; N] {
        if m { t() } else { f() }
    }

    // sqrt / abs / min / max inherited from `Numeric for f64`.
    #[inline(always)]
    fn recip(self) -> f64 {
        1.0 / self
    }

    #[inline(always)]
    fn sin(self) -> f64 {
        f64::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> f64 {
        f64::cos(self)
    }
    #[inline(always)]
    fn tan(self) -> f64 {
        f64::tan(self)
    }
    #[inline(always)]
    fn asin(self) -> f64 {
        f64::asin(self)
    }
    #[inline(always)]
    fn acos(self) -> f64 {
        f64::acos(self)
    }
    #[inline(always)]
    fn atan2(self, b: f64) -> f64 {
        f64::atan2(self, b)
    }

    #[inline(always)]
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    #[inline(always)]
    fn log10(self) -> f64 {
        f64::log10(self)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> f64 {
        f64::powi(self, n)
    }
    #[inline(always)]
    fn powf(self, e: f64) -> f64 {
        f64::powf(self, e)
    }

    #[inline(always)]
    fn floor(self) -> f64 {
        f64::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> f64 {
        f64::ceil(self)
    }

    #[inline(always)]
    fn sinh(self) -> f64 {
        f64::sinh(self)
    }
    #[inline(always)]
    fn cosh(self) -> f64 {
        f64::cosh(self)
    }
    #[inline(always)]
    fn tanh(self) -> f64 {
        f64::tanh(self)
    }
    #[inline(always)]
    fn asinh(self) -> f64 {
        f64::asinh(self)
    }
    #[inline(always)]
    fn acosh(self) -> f64 {
        f64::acosh(self)
    }
    #[inline(always)]
    fn atanh(self) -> f64 {
        f64::atanh(self)
    }

    // iterate: early-return on convergence. preserves the FREEZE LAW — the
    // returned `acc` is the value BEFORE the converging step, matching how
    // the traced kernel freezes via `select(converged, acc, body(acc))`.
    #[inline]
    fn iterate(
        self,
        max_steps: usize,
        body: impl Fn(f64) -> f64,
        converged: impl Fn(f64, f64) -> bool,
    ) -> f64 {
        let mut acc = self;
        for _ in 0..max_steps {
            let next = body(acc);
            if converged(acc, next) {
                return acc;
            }
            acc = next;
        }
        acc
    }

    // iterate_vec: same FREEZE LAW, N-accumulator. `result` selects the
    // returned component (often the c2p root, e.g., recovered pressure).
    #[inline]
    fn iterate_vec<const N: usize>(
        init: [f64; N],
        max_steps: usize,
        body: impl Fn([f64; N]) -> [f64; N],
        converged: impl Fn([f64; N], [f64; N]) -> bool,
        result: usize,
    ) -> f64 {
        let mut acc = init;
        for _ in 0..max_steps {
            let next = body(acc);
            if converged(acc, next) {
                return acc[result];
            }
            acc = next;
        }
        acc[result]
    }
}

// f32 mirrors f64 with the same impl pattern. needed because the workspace's
// `ConsG<f32, ...>` / `PrimG<f32, ...>` types in symbi-hydro state.rs require
// `f32: Scalar` to satisfy the struct's `S: Scalar` bound. ZERO / ONE / from_f64
// / sqrt / abs / min / max inherited from `Numeric for f32`.
impl Scalar for f32 {
    type Mask = bool;

    const INFINITY: f32 = f32::INFINITY;
    const NEG_INFINITY: f32 = f32::NEG_INFINITY;
    const NAN: f32 = f32::NAN;

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline(always)]
    fn cmp_lt(self, b: f32) -> bool {
        self < b
    }
    #[inline(always)]
    fn cmp_le(self, b: f32) -> bool {
        self <= b
    }
    #[inline(always)]
    fn cmp_gt(self, b: f32) -> bool {
        self > b
    }
    #[inline(always)]
    fn cmp_ge(self, b: f32) -> bool {
        self >= b
    }
    #[inline(always)]
    fn cmp_eq(self, b: f32) -> bool {
        self == b
    }

    #[inline(always)]
    fn select(m: bool, t: f32, f: f32) -> f32 {
        if m { t } else { f }
    }

    #[inline(always)]
    fn cond(m: bool, t: impl FnOnce() -> f32, f: impl FnOnce() -> f32) -> f32 {
        if m { t() } else { f() }
    }

    #[inline(always)]
    fn cond_vec<const N: usize>(
        m: bool,
        t: impl FnOnce() -> [f32; N],
        f: impl FnOnce() -> [f32; N],
    ) -> [f32; N] {
        if m { t() } else { f() }
    }

    #[inline(always)]
    fn recip(self) -> f32 {
        1.0 / self
    }

    #[inline(always)]
    fn sin(self) -> f32 {
        f32::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> f32 {
        f32::cos(self)
    }
    #[inline(always)]
    fn tan(self) -> f32 {
        f32::tan(self)
    }
    #[inline(always)]
    fn asin(self) -> f32 {
        f32::asin(self)
    }
    #[inline(always)]
    fn acos(self) -> f32 {
        f32::acos(self)
    }
    #[inline(always)]
    fn atan2(self, b: f32) -> f32 {
        f32::atan2(self, b)
    }

    #[inline(always)]
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> f32 {
        f32::ln(self)
    }
    #[inline(always)]
    fn log10(self) -> f32 {
        f32::log10(self)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> f32 {
        f32::powi(self, n)
    }
    #[inline(always)]
    fn powf(self, e: f32) -> f32 {
        f32::powf(self, e)
    }

    #[inline(always)]
    fn floor(self) -> f32 {
        f32::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> f32 {
        f32::ceil(self)
    }

    #[inline(always)]
    fn sinh(self) -> f32 {
        f32::sinh(self)
    }
    #[inline(always)]
    fn cosh(self) -> f32 {
        f32::cosh(self)
    }
    #[inline(always)]
    fn tanh(self) -> f32 {
        f32::tanh(self)
    }
    #[inline(always)]
    fn asinh(self) -> f32 {
        f32::asinh(self)
    }
    #[inline(always)]
    fn acosh(self) -> f32 {
        f32::acosh(self)
    }
    #[inline(always)]
    fn atanh(self) -> f32 {
        f32::atanh(self)
    }

    #[inline]
    fn iterate(
        self,
        max_steps: usize,
        body: impl Fn(f32) -> f32,
        converged: impl Fn(f32, f32) -> bool,
    ) -> f32 {
        let mut acc = self;
        for _ in 0..max_steps {
            let next = body(acc);
            if converged(acc, next) {
                return acc;
            }
            acc = next;
        }
        acc
    }

    #[inline]
    fn iterate_vec<const N: usize>(
        init: [f32; N],
        max_steps: usize,
        body: impl Fn([f32; N]) -> [f32; N],
        converged: impl Fn([f32; N], [f32; N]) -> bool,
        result: usize,
    ) -> f32 {
        let mut acc = init;
        for _ in 0..max_steps {
            let next = body(acc);
            if converged(acc, next) {
                return acc[result];
            }
            acc = next;
        }
        acc[result]
    }
}

// =============================================================================
// section 8 — Carrier locations (where each homomorphism lives in the workspace).
//
//   `Scalar for f64`         — this file, §7. evaluating Carrier.
//   `Scalar for f32`         — this file, §7 (sibling impl). single-precision.
//   `Scalar for Gv`          — crates/symbi-discretize/src/gv.rs. IR-tracing.
//                              lives there because the graph it builds is part
//                              of the IR substrate, not the Carrier interface.
//   `Scalar for Sym`         — future, crates/symbi-core/src/sym.rs.
//   `Scalar for Dual<C>`     — future, crates/symbi-core/src/dual.rs.
//
// each MUST be a total homomorphism. the totality lint (task A5) checks it.
// =============================================================================

// =============================================================================
// section 9 — tests: homomorphism + identity contract checks for f64.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_add_identity() {
        assert_eq!(3.0 + f64::zero(), 3.0);
    }
    #[test]
    fn f64_mul_identity() {
        assert_eq!(3.0 * f64::one(), 3.0);
    }
    #[test]
    fn f64_neg_involution() {
        assert_eq!(-(-3.0_f64), 3.0);
    }
    #[test]
    fn f64_sub_via_neg() {
        assert_eq!(3.0 - 1.0, 3.0 + (-1.0_f64));
    }
    #[test]
    fn f64_div_via_recip() {
        assert_eq!(6.0 / 2.0, 6.0 * Scalar::recip(2.0_f64));
    }

    #[test]
    fn f64_cmp_lt_irreflexive() {
        assert!(!(3.0_f64).cmp_lt(3.0));
    }
    #[test]
    fn f64_cmp_eq_reflexive() {
        assert!((3.0_f64).cmp_eq(3.0));
    }

    #[test]
    fn f64_select_true_picks_t() {
        assert_eq!(<f64 as Scalar>::select(true, 1.0, 2.0), 1.0);
    }
    #[test]
    fn f64_select_false_picks_f() {
        assert_eq!(<f64 as Scalar>::select(false, 1.0, 2.0), 2.0);
    }

    #[test]
    fn f64_sqrt_sq_is_abs() {
        let x = -2.5_f64;
        assert_eq!((x * x).sqrt(), x.abs());
    }
    #[test]
    fn f64_exp_ln_roundtrip() {
        let x = 2.5_f64;
        assert!((x.ln().exp() - x).abs() < 1e-12);
    }

    #[test]
    fn bool_mask_demorgan() {
        // !(a & b) == !a | !b for every (a, b).
        for &a in &[true, false] {
            for &b in &[true, false] {
                assert_eq!(!(a & b), !a | !b);
            }
        }
    }

    #[test]
    fn source_loc_macro_compiles() {
        let loc = source_loc!(binding = "cons_l");
        assert_eq!(loc.binding, Some("cons_l"));
        assert!(loc.line > 0);
    }

    // ── hyperbolic identities (the laws drive the carrier oracle) ─────────
    #[test]
    fn f64_cosh_sq_minus_sinh_sq_is_one() {
        for &x in &[-1.5_f64, 0.0, 0.7, 2.5] {
            let id = x.cosh() * x.cosh() - x.sinh() * x.sinh();
            assert!(
                (id - 1.0).abs() < 1e-12,
                "cosh^2-sinh^2 != 1 at x={x}: {id}"
            );
        }
    }
    #[test]
    fn f64_asinh_sinh_roundtrip() {
        for &x in &[-2.0_f64, -0.3, 0.0, 1.2] {
            assert!((x.sinh().asinh() - x).abs() < 1e-12);
        }
    }
    #[test]
    fn f64_acosh_cosh_is_abs() {
        // acosh(cosh(x)) == |x| — asymmetric, like sqrt(x*x).
        for &x in &[-1.7_f64, -0.4, 0.0, 0.9, 2.3] {
            assert!((x.cosh().acosh() - x.abs()).abs() < 1e-12);
        }
    }
    #[test]
    fn f64_acosh_partial_domain() {
        // acosh on x < 1 is NaN — documented in the trait + laws.
        assert!(<f64 as Scalar>::acosh(0.5).is_nan());
    }

    // ── IEEE sentinels ────────────────────────────────────────────────────
    #[test]
    fn f64_infinity_is_positive() {
        assert!(<f64 as Scalar>::infinity() > 0.0);
    }
    #[test]
    fn f64_neg_infinity_is_negative() {
        assert!(<f64 as Scalar>::neg_infinity() < 0.0);
    }
    #[test]
    fn f64_nan_is_not_self_equal() {
        // the LAW that defines is_nan: NaN is the only value with x != x.
        let n = <f64 as Scalar>::nan();
        assert!(!n.cmp_eq(n));
    }
    #[test]
    fn f64_is_nan_detects_nan() {
        // is_nan is true for any NaN, false for finite / +/-inf.
        assert!((<f64 as Scalar>::nan()).is_nan());
        assert!(!(0.0_f64).is_nan());
        assert!(!(<f64 as Scalar>::infinity()).is_nan());
        assert!(!(<f64 as Scalar>::neg_infinity()).is_nan());
    }
    #[test]
    fn f64_cmp_eq_with_nan_is_always_false() {
        // documents the trap: x.cmp_eq(S::nan()) is ALWAYS false.
        let n = <f64 as Scalar>::nan();
        assert!(!(1.0_f64).cmp_eq(n));
        assert!(!n.cmp_eq(1.0));
    }

    // ── branch: default routes through Selectable::select ─────────────────
    #[test]
    fn f64_branch_picks_true_arm() {
        let r = <f64 as Scalar>::branch(true, || 1.0_f64, || 2.0);
        assert_eq!(r, 1.0);
    }
    #[test]
    fn f64_branch_picks_false_arm() {
        let r = <f64 as Scalar>::branch(false, || 1.0_f64, || 2.0);
        assert_eq!(r, 2.0);
    }
    #[test]
    fn f64_branch_evaluates_both_closures() {
        // documents the A1 trap: native `if cond { yes() } else { no() }`
        // would only evaluate the chosen arm. `branch` evaluates BOTH (for
        // the f64 carrier this is wasted work; for Gv it is the trace).
        let counter = std::cell::Cell::new(0);
        let bump = |_| counter.set(counter.get() + 1);
        let _ = <f64 as Scalar>::branch(
            true,
            || {
                bump(0);
                1.0_f64
            },
            || {
                bump(0);
                2.0
            },
        );
        assert_eq!(counter.get(), 2);
    }

    // ── iterate: the FREEZE LAW on f64 ────────────────────────────────────
    #[test]
    fn f64_iterate_returns_pre_convergence_acc() {
        // body: acc -> 2*acc. converged: |next - acc| < 0.5.
        // start at 0.0: 0 -> 0 (converged immediately since |0-0|<0.5 -> returns 0).
        let r0 = (0.0_f64).iterate(10, |a| 2.0 * a, |a, n| ((n - a) as f64).abs() < 0.5);
        assert_eq!(r0, 0.0);
        // start at 1.0: 1 -> 2 (|2-1|=1 not converged), 2 -> 4 (|4-2|=2 not), ...
        // converges when consecutive close, which never happens in this chain.
        // result should be after max_steps = 10: 1 -> 2 -> 4 -> ... -> 1024.
        let r1 = (1.0_f64).iterate(10, |a| 2.0 * a, |a, n| ((n - a) as f64).abs() < 0.5);
        assert_eq!(r1, 1024.0);
    }
    #[test]
    fn f64_iterate_freeze_holds_pre_convergence_value() {
        // a fixed-point body (acc -> sqrt(acc)) converges towards 1.0 from 4.0.
        // returns acc BEFORE the converging step — that's the freeze: the
        // returned value is from BEFORE the convergence criterion fired.
        let r = (4.0_f64).iterate(50, |a| a.sqrt(), |a, n| (n - a).abs() < 1e-10);
        // converged: the result is ~1.0 (the fixed point), but the returned
        // value is acc from the PRE-converging step, so it's epsilon-close.
        assert!((r - 1.0).abs() < 1e-9);
    }

    // ── iterate_vec: fibonacci as the multi-state canary ──────────────────
    #[test]
    fn f64_iterate_vec_fibonacci() {
        // (a, b) -> (b, a + b). 5 steps from (1, 1) -> (1, 1) (1,2) (2,3) (3,5) (5,8) (8,13).
        // never-converged predicate; result index = 1 (the "b" component).
        let r = <f64 as Scalar>::iterate_vec::<2>(
            [1.0, 1.0],
            5,
            |acc| [acc[1], acc[0] + acc[1]],
            |_, _| false,
            1,
        );
        assert_eq!(r, 13.0);
    }
    #[test]
    fn f64_iterate_vec_freeze() {
        // same fibonacci body, but converge-immediately predicate; returns acc[1]
        // BEFORE the first body step — the seed value.
        let r = <f64 as Scalar>::iterate_vec::<2>(
            [3.0, 7.0],
            5,
            |acc| [acc[1], acc[0] + acc[1]],
            |_, _| true,
            1,
        );
        assert_eq!(r, 7.0);
    }

    // ----- docs/design/23 step 3a: Scalar::scope at S = f64 -----

    /// at `S = f64`, `scope` is identity — the closure runs immediately and
    /// returns its value. This is the FOUNDATION for step 3b: physics
    /// authors can use `S::scope(|| ...)` blocks today, even before the Gv
    /// override emits real `ScalarStmt::Scope`s.
    #[test]
    fn f64_scope_runs_closure_identity() {
        let r: f64 = <f64 as Scalar>::scope(|| 3.14_f64 * 2.0);
        assert_eq!(r, 6.28);

        // the closure can return any Scalar — it's not tied to f64 specifically.
        // here R is inferred as f64; later if we add tensor returns it works the same.
        let inputs = (2.0_f64, 5.0_f64);
        let r: f64 = <f64 as Scalar>::scope(|| {
            // simulate a small phase: locals die at brace.
            let vn = inputs.0;
            let cs2 = inputs.1;
            vn * vn + cs2
        });
        assert_eq!(r, 9.0);
    }

    /// scope CAN be nested arbitrarily. each nested call to `scope` produces
    /// an independent identity transform at f64. this proves the trait
    /// shape works for the nested-phase patterns the design doc describes
    /// (load → reconstruct → compute → store).
    #[test]
    fn f64_scope_nests_arbitrarily() {
        let inner = <f64 as Scalar>::scope(|| {
            let a = <f64 as Scalar>::scope(|| 1.0 + 2.0);
            let b = <f64 as Scalar>::scope(|| 4.0 + 5.0);
            a * b
        });
        assert_eq!(inner, 3.0 * 9.0);
    }

    /// `scope` accepts a `FnOnce` closure — captured values move into the
    /// scope, mirroring how Rust nested blocks work. proves the API matches
    /// what physics authors will write.
    #[test]
    fn f64_scope_captures_move_in() {
        let v: f64 = 6.0;
        let sum: f64 = <f64 as Scalar>::scope(move || v);
        assert_eq!(sum, 6.0);
    }
}
