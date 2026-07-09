// =============================================================================
// graph.rs
//
// the tensor IR graph: a DAG of `Node`s, each carrying an `Op` variant
// and a `TensorTy`. nodes reference each other by `NodeId` (a u32
// newtype indexing into the graph's vector).
//
// R.1.d ships the skeleton with only `Const` and `Param` op variants
// implemented. the remaining ops (Construct, Index, Reduce,
// ElementWise, Transcendental, Select, Broadcast) land in R.2.
//
// invariants:
//   - every `NodeId` is in-bounds of `nodes` and `types`.
//   - `types[i]` is the result type of node `i`.
//   - `Const` and `Param` nodes have no input edges.
//   - all other op variants carry their input edges via NodeId.
// =============================================================================

use std::collections::HashMap;

use proc_macro2::Span;

use crate::dim::{broadcast_shape, broadcasts_to};
use crate::error::ShapeError;
use crate::{DimExpr, ElementTy, Symbol, TensorTy};

// ----- node id -----

/// opaque index into a `Graph`'s node vector. `u32` is overkill for
/// any realistic kernel but matches the existing scalar IR's choice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

// ----- op variants (R.1.d: skeleton only) -----

/// payload of a `Const` node. one variant per supported `ElementTy`.
///
/// F2.A: Hash + Eq are implemented over the *bit pattern* of floats,
/// not their IEEE comparison. two F64(NaN) values with identical bit
/// patterns are considered equal; two NaNs with different bit patterns
/// are not. this is what hash-cons needs (structural identity).
#[derive(Clone, Debug)]
pub enum ConstValue {
    F64(f64),
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ConstValue::F64(a), ConstValue::F64(b)) => a.to_bits() == b.to_bits(),
            (ConstValue::F32(a), ConstValue::F32(b)) => a.to_bits() == b.to_bits(),
            (ConstValue::I32(a), ConstValue::I32(b)) => a == b,
            (ConstValue::U32(a), ConstValue::U32(b)) => a == b,
            (ConstValue::Bool(a), ConstValue::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ConstValue {}

impl std::hash::Hash for ConstValue {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        match self {
            ConstValue::F64(x) => {
                0u8.hash(h);
                x.to_bits().hash(h);
            }
            ConstValue::F32(x) => {
                1u8.hash(h);
                x.to_bits().hash(h);
            }
            ConstValue::I32(x) => {
                2u8.hash(h);
                x.hash(h);
            }
            ConstValue::U32(x) => {
                3u8.hash(h);
                x.hash(h);
            }
            ConstValue::Bool(b) => {
                4u8.hash(h);
                b.hash(h);
            }
        }
    }
}

impl ConstValue {
    /// element type matching the variant.
    pub fn element(&self) -> ElementTy {
        match self {
            ConstValue::F64(_) => ElementTy::F64,
            ConstValue::F32(_) => ElementTy::F32,
            ConstValue::I32(_) => ElementTy::I32,
            ConstValue::U32(_) => ElementTy::U32,
            ConstValue::Bool(_) => ElementTy::Bool,
        }
    }
}

// serde via the float BIT PATTERN, not the decimal value. kernels carry
// `ConstValue::F64(f64::NAN)` / `INFINITY` (c2p sentinels); serde_json maps a
// raw non-finite f64 to `null`, which would not round-trip. serializing the bits
// (u64/u32) is exact for every value and matches this enum's bit-pattern Hash/Eq.
#[derive(serde::Serialize, serde::Deserialize)]
enum ConstValueRepr {
    F64(u64),
    F32(u32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl serde::Serialize for ConstValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let repr = match self {
            ConstValue::F64(x) => ConstValueRepr::F64(x.to_bits()),
            ConstValue::F32(x) => ConstValueRepr::F32(x.to_bits()),
            ConstValue::I32(x) => ConstValueRepr::I32(*x),
            ConstValue::U32(x) => ConstValueRepr::U32(*x),
            ConstValue::Bool(x) => ConstValueRepr::Bool(*x),
        };
        repr.serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for ConstValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match ConstValueRepr::deserialize(d)? {
            ConstValueRepr::F64(b) => ConstValue::F64(f64::from_bits(b)),
            ConstValueRepr::F32(b) => ConstValue::F32(f32::from_bits(b)),
            ConstValueRepr::I32(v) => ConstValue::I32(v),
            ConstValueRepr::U32(v) => ConstValue::U32(v),
            ConstValueRepr::Bool(v) => ConstValue::Bool(v),
        })
    }
}

/// indexing dimension: either a compile-time literal (must be in
/// bounds against a Literal axis; against a Generic axis the bound is
/// only checkable at monomorph time) or a const-generic symbol used
/// inside generated loops during scalarization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimIndex {
    Literal(usize),
    Generic(Symbol),
}

/// element-wise op tag. binary ops broadcast their inputs to a common
/// shape; unary ops preserve shape. comparison ops return Bool;
/// classification ops (IsFinite/IsNaN) take a float and return Bool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementWiseOp {
    // arithmetic binary
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    // integer floor division (rounds toward negative infinity) — the index-space
    // primitive for refinement-lattice pullbacks (amr prolong: coarse parent of a
    // possibly-negative fine ghost index). integer-only: renders as `div_euclid`
    // (rust), an explicit floor-division ternary (cuda). never promotes to float.
    FloorDiv,
    // arithmetic unary
    Neg,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Trunc,
    // transcendental unary (input: float, result: float)
    Sin,
    Cos,
    Acos,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    // transcendental binary: Pow(a, b) = a^b (float)
    Pow,
    // numeric conversion (the usual-arithmetic-conversions primitive): `Cast(to)(x)`
    // converts `x` to element type `to`. inserted IMPLICITLY by `element_wise` to
    // promote a mixed-type op's narrower operand (e.g., an i32 index times an f64
    // grid width) — callers never write it. unary; result element is `to`.
    Cast(ElementTy),
    // comparison binary (result: Bool)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // classification unary (input: float, result: Bool)
    IsFinite,
    IsNaN,
    // bitwise / logical binary. on integer inputs: bitwise. on Bool
    // inputs: logical (Rust's `&&` / `||` / `^` reduce to these
    // after eager evaluation — pure-functional bodies don't observe
    // short-circuit). result element matches input.
    BitAnd,
    BitOr,
    BitXor,
    // bitwise / logical unary. on integer inputs: bitwise complement.
    // on Bool inputs: logical NOT (Rust's `!`). result element matches input.
    // backs the `Mask: Not` requirement on the carrier `Scalar::Mask`.
    BitNot,
}

impl ElementWiseOp {
    pub fn arity(self) -> usize {
        match self {
            ElementWiseOp::Add
            | ElementWiseOp::Sub
            | ElementWiseOp::Mul
            | ElementWiseOp::Div
            | ElementWiseOp::FloorDiv
            | ElementWiseOp::Min
            | ElementWiseOp::Max
            | ElementWiseOp::Eq
            | ElementWiseOp::Ne
            | ElementWiseOp::Lt
            | ElementWiseOp::Le
            | ElementWiseOp::Gt
            | ElementWiseOp::Ge
            | ElementWiseOp::BitAnd
            | ElementWiseOp::BitOr
            | ElementWiseOp::BitXor
            | ElementWiseOp::Pow => 2,
            _ => 1,
        }
    }

    /// does this op produce a Bool tensor regardless of input element?
    pub fn returns_bool(self) -> bool {
        matches!(
            self,
            ElementWiseOp::Eq
                | ElementWiseOp::Ne
                | ElementWiseOp::Lt
                | ElementWiseOp::Le
                | ElementWiseOp::Gt
                | ElementWiseOp::Ge
                | ElementWiseOp::IsFinite
                | ElementWiseOp::IsNaN
        )
    }

    /// does this op require a float input? (Sqrt + IsFinite + IsNaN.)
    pub fn requires_float(self) -> bool {
        matches!(
            self,
            ElementWiseOp::Sqrt
                | ElementWiseOp::IsFinite
                | ElementWiseOp::IsNaN
                | ElementWiseOp::Floor
                | ElementWiseOp::Ceil
                | ElementWiseOp::Round
                | ElementWiseOp::Trunc
                | ElementWiseOp::Sin
                | ElementWiseOp::Cos
                | ElementWiseOp::Acos
                | ElementWiseOp::Sinh
                | ElementWiseOp::Cosh
                | ElementWiseOp::Asinh
                | ElementWiseOp::Acosh
                | ElementWiseOp::Pow
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            ElementWiseOp::Add => "Add",
            ElementWiseOp::Sub => "Sub",
            ElementWiseOp::Mul => "Mul",
            ElementWiseOp::Div => "Div",
            ElementWiseOp::FloorDiv => "FloorDiv",
            ElementWiseOp::Min => "Min",
            ElementWiseOp::Max => "Max",
            ElementWiseOp::Neg => "Neg",
            ElementWiseOp::Abs => "Abs",
            ElementWiseOp::Sqrt => "Sqrt",
            ElementWiseOp::Floor => "Floor",
            ElementWiseOp::Ceil => "Ceil",
            ElementWiseOp::Round => "Round",
            ElementWiseOp::Trunc => "Trunc",
            ElementWiseOp::Eq => "Eq",
            ElementWiseOp::Ne => "Ne",
            ElementWiseOp::Lt => "Lt",
            ElementWiseOp::Le => "Le",
            ElementWiseOp::Gt => "Gt",
            ElementWiseOp::Ge => "Ge",
            ElementWiseOp::IsFinite => "IsFinite",
            ElementWiseOp::IsNaN => "IsNaN",
            ElementWiseOp::Sin => "Sin",
            ElementWiseOp::Cos => "Cos",
            ElementWiseOp::Acos => "Acos",
            ElementWiseOp::Sinh => "Sinh",
            ElementWiseOp::Cosh => "Cosh",
            ElementWiseOp::Asinh => "Asinh",
            ElementWiseOp::Acosh => "Acosh",
            ElementWiseOp::Pow => "Pow",
            ElementWiseOp::BitAnd => "BitAnd",
            ElementWiseOp::BitOr => "BitOr",
            ElementWiseOp::BitXor => "BitXor",
            ElementWiseOp::BitNot => "BitNot",
            ElementWiseOp::Cast(_) => "Cast",
        }
    }

    /// does this op participate in the usual arithmetic conversions — i.e., a binary
    /// numeric op whose mixed int/float operands should promote to a common float
    /// type? arithmetic + comparison promote; bitwise + the conversion op itself do
    /// not (bitwise needs matching int/bool; Cast is unary).
    pub fn promotes(self) -> bool {
        matches!(
            self,
            ElementWiseOp::Add
                | ElementWiseOp::Sub
                | ElementWiseOp::Mul
                | ElementWiseOp::Div
                | ElementWiseOp::Min
                | ElementWiseOp::Max
                | ElementWiseOp::Pow
                | ElementWiseOp::Eq
                | ElementWiseOp::Ne
                | ElementWiseOp::Lt
                | ElementWiseOp::Le
                | ElementWiseOp::Gt
                | ElementWiseOp::Ge
        )
    }
}

/// the usual arithmetic conversions, scoped to the ONE case the substrate needs:
/// an integer promoted to a float (an i32 index times an f64 grid width -> the
/// index becomes a physical-space real). returns the float operand's type.
///
/// deliberately NARROW: float-width mismatch (f32 vs f64) and int-width mismatch
/// stay strict errors. the substrate's precision is uniform per kernel (the graph
/// is f64; f32 is a render-time choice), so a genuine f32+f64 in one graph is an
/// anomaly worth catching, not silently widening. Bool never promotes.
fn numeric_promote(a: ElementTy, b: ElementTy) -> Option<ElementTy> {
    match (a.is_float(), b.is_float(), a.is_integer(), b.is_integer()) {
        // one float, one int -> the float type (the index promotes up to it).
        (true, false, false, true) => Some(a),
        (false, true, true, false) => Some(b),
        // equal types, float-width mismatch, int-width mismatch, or any Bool: no
        // implicit promotion (kept strict so accidental mixing is still caught).
        _ => None,
    }
}

/// reduction op tag. the variants here are the non-Sum reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Min,
    Max,
    Or,
    And,
    Xor,
}

impl ReduceOp {
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Min => "Min",
            ReduceOp::Max => "Max",
            ReduceOp::Or => "Or",
            ReduceOp::And => "And",
            ReduceOp::Xor => "Xor",
        }
    }

    /// is this op valid on the given element type?
    /// Min/Max: float or int.
    /// Or/And/Xor: int or bool.
    pub fn accepts_element(self, e: ElementTy) -> bool {
        match self {
            ReduceOp::Min | ReduceOp::Max => e.is_float() || e.is_integer(),
            ReduceOp::Or | ReduceOp::And | ReduceOp::Xor => e.is_integer() || e == ElementTy::Bool,
        }
    }
}

/// transcendental ops. all take float inputs, all produce Tainted
/// results because CPU libm and GPU libdevice may drift up to a few
/// ULP — same rule as the existing scalar IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TranscendentalOp {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Pow,
    Hypot,
}

impl TranscendentalOp {
    pub fn arity(self) -> usize {
        match self {
            TranscendentalOp::Atan2 | TranscendentalOp::Pow | TranscendentalOp::Hypot => 2,
            _ => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TranscendentalOp::Sin => "Sin",
            TranscendentalOp::Cos => "Cos",
            TranscendentalOp::Tan => "Tan",
            TranscendentalOp::Asin => "Asin",
            TranscendentalOp::Acos => "Acos",
            TranscendentalOp::Atan => "Atan",
            TranscendentalOp::Atan2 => "Atan2",
            TranscendentalOp::Exp => "Exp",
            TranscendentalOp::Exp2 => "Exp2",
            TranscendentalOp::Log => "Log",
            TranscendentalOp::Log2 => "Log2",
            TranscendentalOp::Log10 => "Log10",
            TranscendentalOp::Sinh => "Sinh",
            TranscendentalOp::Cosh => "Cosh",
            TranscendentalOp::Tanh => "Tanh",
            TranscendentalOp::Asinh => "Asinh",
            TranscendentalOp::Acosh => "Acosh",
            TranscendentalOp::Atanh => "Atanh",
            TranscendentalOp::Pow => "Pow",
            TranscendentalOp::Hypot => "Hypot",
        }
    }
}

/// the IR op carried by a node. R.1.d implemented Const and Param;
/// R.2.a adds Construct, Index, Broadcast. the remaining four ops
/// (Reduce, ElementWise, Transcendental, Select) land later
/// in R.2.
///
/// F2.A: Op + dependents implement Hash + Eq so the Graph can
/// structural-hash-cons identical (Op, output-type) pairs into a single
/// NodeId at construction. equal subgraphs collapse to shared NodeIds,
/// which means sharing is preserved through scalarize / emit without
/// the post-hoc-CSE recovery dance.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// rank-0 literal of the given element type.
    Const(ConstValue),
    /// typed input identified by an interned symbol.
    Param(Symbol),
    /// build a rank-(K+1) tensor by stacking N rank-K tensors along a
    /// new outermost axis of length N. all inputs must agree on
    /// element and shape.
    Construct(Vec<NodeId>),
    /// extract a rank-0 element. one DimIndex per axis of the input.
    Index(NodeId, Vec<DimIndex>),
    /// explicit broadcast to a target shape (the input's shape must
    /// broadcast to the target per § 1.5).
    Broadcast(NodeId, Vec<DimExpr>),
    /// element-wise op with broadcast-aware shape inference. arity is
    /// 1 or 2 depending on the op tag.
    ElementWise(ElementWiseOp, Vec<NodeId>),
    /// transcendental op (sin/cos/exp/etc.). arity 1 or 2 depending on
    /// the op tag.
    Transcendental(TranscendentalOp, Vec<NodeId>),
    /// non-sum reduction over named axes.
    Reduce(ReduceOp, Vec<u32>, NodeId),
    /// element-wise if/else. cond must be Bool; then/else share element
    /// and broadcast to a common shape with cond.
    Select(NodeId, NodeId, NodeId),
    /// arbitrary-coord field load. the symbol names a field-input key
    /// (matching a Param key registered for the same field via
    /// add_param / intern_read_param) so the dispatch side resolves a
    /// buffer index for it. components is a rank-(ndim) coordinate of
    /// rank-0 scalar NodeIds. emitted as `buf<idx>[flat_index(components)]`
    /// on GPU; rewritten away at the emit_kernel layer.
    ///
    /// the kernel-coord field-read pattern (Param + emit_kernel prelude)
    /// is the right primitive when the read coordinate IS the kernel
    /// iteration; LoadAt covers the second case (gather at a runtime
    /// source coord, e.g., ghost-fill remap).
    LoadAt(Symbol, Vec<NodeId>),
    /// F2.C: a first-class function value. references a `FnDef`
    /// stored in the graph's `lambdas` table by FnId. the lambda's
    /// signature lives in FnDef.params; the body is a sub-Graph with
    /// `FnDef.output` as its result NodeId. Lambda nodes carry a
    /// placeholder tensor type (rank-0 F64) — they're callable, not
    /// values, so they should only be consumed by `Op::Apply`.
    Lambda(FnId),
    /// F2.C: apply a Lambda value to arguments. `lambda` must be a
    /// NodeId pointing at an `Op::Lambda` in the same graph. `args`
    /// match the lambda's FnDef.params in count and type. result type
    /// equals the type of the lambda's body output.
    Apply { lambda: NodeId, args: Vec<NodeId> },
    /// F2.F: bounded fold. canonical catamorphism over the natural
    /// number `count`. `lambda` must be `Op::Lambda` with FnDef of
    /// shape `(Acc, Idx) -> Acc`. semantics:
    ///
    ///   let mut acc = init;
    ///   for i in 0..count {
    ///       acc = lambda(acc, i);
    ///   }
    ///   acc
    ///
    /// the accumulator type `Acc` can be any TensorTy (rank-0 for
    /// simple sums, rank-1 for tuples like `(lo, hi)` in bisection
    /// solvers). `init.ty == Acc == lambda.params[0].ty == lambda
    /// output.ty`. `count` is rank-0 integer (literal or runtime
    /// param); `lambda.params[1]` is rank-0 integer.
    ///
    /// Fold is the only iteration primitive in the IR — there is no
    /// `Op::While`. all iteration is statically bounded by `count`,
    /// preserving the DAG's termination-by-construction property.
    /// dynamic-termination patterns (Newton with adaptive step
    /// rejection, etc.) live at the host level outside the IR.
    Fold {
        lambda: NodeId,
        init: NodeId,
        count: NodeId,
    },
    /// component `idx` of the current accumulator VECTOR inside an `IterateInline`
    /// loop body — a rank-0 placeholder leaf the scalarizer resolves to the loop's
    /// `idx`-th mutable accumulator local. scalar loops use a 1-component vector
    /// (`IterAcc(0)`). only meaningful within the `steps` sub-DAGs of the
    /// `IterateInline` that owns it.
    IterAcc(u32),
    /// an INLINE bounded iteration — the loop counterpart of `iterate`'s unroll
    /// (docs/design/14). emits the body ONCE as a real `for` over an `N`-component
    /// accumulator VECTOR (N = `accs.len()`; scalar Newton is N=1, a bracketed
    /// false-position root-find — KKC RMHD c2p — is N>1):
    ///
    ///   let mut acc_0 = init_0; ... let mut acc_{N-1} = init_{N-1};
    ///   for i in 0..count {
    ///       // all `steps[j]` computed from the OLD acc_* (simultaneous / Jacobi)
    ///       acc_0 = steps[0]; ...; acc_{N-1} = steps[N-1];
    ///   }
    ///   acc_{result}
    ///
    /// each `steps[j]` lives in the MAIN graph, referencing the `accs` `IterAcc(_)`
    /// placeholders PLUS loop-invariant nodes directly. the scalarizer lowers the
    /// union acc-dependent cone inside the `for` (invariants stay outside), emits
    /// the `N` assigns AFTER the whole cone (so the update is simultaneous), and
    /// the node's value is `accs[result]` post-loop. `count` is a literal bound.
    IterateInline {
        accs: Vec<NodeId>,
        inits: Vec<NodeId>,
        steps: Vec<NodeId>,
        count: usize,
        result: u32,
        /// optional EARLY-BREAK predicate (rank-0 Bool). when `Some`, the
        /// scalarizer emits `if break_when { break; }` at the END of the loop
        /// body — after the step assigns — so once the freeze predicate from
        /// `iterate` / `iterate_vec` first fires, no further iterations run.
        /// without this, the steps' `select(conv, old, new)` would null out
        /// the WRITES but still pay for the body's arithmetic every iteration.
        ///
        /// for the RMHD c2p (max_iter = 100, typical convergence ~8 iters),
        /// this turns ~92 dead bodies/cell into a real break — measurable.
        break_when: Option<NodeId>,
    },
    /// docs/design/23: a bounded-pressure phase scope. `body`
    /// lists the NodeIds CREATED inside a `Gv::scope` closure (in insertion
    /// order); they are temporaries that the lowered emit will surround
    /// with `{ ... }` braces so the codegen sees their lifetimes end at
    /// the closing brace. `result` is the value the scope returns to the
    /// enclosing context.
    ///
    /// invariants:
    /// - `body` is non-empty (an empty scope is degenerate — the trace
    ///   should not produce one; `Gv::scope` over a body that creates no
    ///   new nodes just returns the result inline).
    /// - `result` is either a member of `body` OR a NodeId from the
    ///   ENCLOSING graph (in which case the scope is a "renaming" — the
    ///   body produces temps, the result is one of them or an external).
    /// - scalarize handles `Op::Scope` by emitting a `ScalarStmt::Scope`
    ///   wrapping the body's NodeIds in a brace block, with `result`
    ///   lowered as the block's tail expression.
    /// - `Op::Scope` BYPASSES hash-cons (see `push()`): each scope is a
    ///   distinct lexical region, even if two scopes happen to share the
    ///   same body+result shape.
    Scope { body: Vec<NodeId>, result: NodeId },
    /// the DUAL of `IterateInline`: a real data-dependent branch. `cond` is a
    /// rank-0 Bool (Mask). exactly ONE arm executes at runtime — `then_results`
    /// when `cond` is true, `else_results` otherwise. each `*_body` lists the
    /// NodeIds CREATED inside that arm's `S::cond` closure (insertion order);
    /// they are lowered INSIDE the arm's brace so the codegen evaluates them
    /// only on the taken path. this is what lets carrier-generic physics get
    /// the early-out cost (skip the whole quartic on a fast path)
    /// instead of paying compute-all-paths via `Op::Select`.
    ///
    /// `*_results` are vectors so the vector form (`cond_vec`, the dual of
    /// `iterate_vec`) lands as a thin addition; scalar `S::cond` uses length-1
    /// vectors. the node's value is `then_results[0]` / `else_results[0]`
    /// (component-`result` for the vec form), typed from `then_results[0]`.
    ///
    /// invariants (mirror `Op::Scope`):
    /// - shared upstream values (the `cond`, any pre-branch subexpression) are
    ///   created BEFORE the closures, so they sit OUTSIDE both bodies and are
    ///   computed unconditionally — exactly the cheap shared prefix.
    /// - cross-arm / leaks-outside hash-cons sharing is resolved by the
    ///   scalarizer's eviction pass (a shared NodeId is hoisted to the outer
    ///   level and computed once), identical to the `Op::Scope` body rule.
    /// - bypasses hash-cons (see `push()`): each branch is a distinct lexical
    ///   region.
    IfElse {
        cond: NodeId,
        then_body: Vec<NodeId>,
        then_results: Vec<NodeId>,
        else_body: Vec<NodeId>,
        else_results: Vec<NodeId>,
    },
    /// extract component `index` of a MULTI-OUTPUT node (the only such node: an `Op::IfElse`
    /// with N results, from `S::cond_vec`). the scalarizer binds the multi-
    /// output node to an N-component `Concrete` binding and `Proj` selects one
    /// component — the lightweight projection that lets `cond_vec` return N
    /// distinct scalar values from a SINGLE shared branch (the arm computation
    /// is traced once; each output is one `Proj`). rank-0 scalar.
    Proj { source: NodeId, index: u32 },
}

// =============================================================================
// the SINGLE SOURCE OF TRUTH for Op's NodeId-field topology.
//
// every pass that walks operand edges (splice's remap, scalarize's cone /
// in-degree / `inputs()`, const-folder / SSA-rewrite passes) needs the same
// answer: "for THIS variant, which fields are NodeIds?". that answer lives in
// exactly one place — `Op::try_map_inputs`, a fallible, in-place per-variant
// traversal of every NodeId field. the read-only `inputs()` view is DERIVED
// from it (a noop map that collects the visited ids). adding a NodeId field to
// a variant is one match-arm edit; splitting the answer across passes risks a
// silent miscompile-class bug when one copy is updated and the other is not.
// =============================================================================
impl Op {
    /// visit every NodeId field of this op, applying `f` to each in declared
    /// order. errors short-circuit. ONE arm per variant — adding a new
    /// variant or new NodeId field touches THIS method and nowhere else.
    pub fn try_map_inputs<E>(
        &mut self,
        mut f: impl FnMut(NodeId) -> Result<NodeId, E>,
    ) -> Result<(), E> {
        match self {
            // leaves: no NodeId fields.
            Op::Const(_) | Op::Param(_) | Op::Lambda(_) | Op::IterAcc(_) => Ok(()),

            // single NodeId field.
            Op::Index(t, _) | Op::Broadcast(t, _) | Op::Reduce(_, _, t) => {
                *t = f(*t)?;
                Ok(())
            }

            // Vec<NodeId> field.
            Op::ElementWise(_, ins)
            | Op::Transcendental(_, ins)
            | Op::Construct(ins)
            | Op::LoadAt(_, ins)
            | Op::Apply { args: ins, .. } => {
                for n in ins.iter_mut() {
                    *n = f(*n)?;
                }
                Ok(())
            }

            // three NodeId fields.
            Op::Select(c, t, e) => {
                *c = f(*c)?;
                *t = f(*t)?;
                *e = f(*e)?;
                Ok(())
            }
            Op::Fold {
                lambda,
                init,
                count,
            } => {
                *lambda = f(*lambda)?;
                *init = f(*init)?;
                *count = f(*count)?;
                Ok(())
            }

            // multiple Vec<NodeId> fields + optional NodeId.
            Op::IterateInline {
                accs,
                inits,
                steps,
                break_when,
                ..
            } => {
                for n in accs.iter_mut() {
                    *n = f(*n)?;
                }
                for n in inits.iter_mut() {
                    *n = f(*n)?;
                }
                for n in steps.iter_mut() {
                    *n = f(*n)?;
                }
                if let Some(bw) = break_when {
                    *bw = f(*bw)?;
                }
                Ok(())
            }

            // docs/design/23: `body` is a Vec<NodeId> of scope-local
            // temps, `result` is the value the scope returns. both get remapped.
            Op::Scope { body, result } => {
                for n in body.iter_mut() {
                    *n = f(*n)?;
                }
                *result = f(*result)?;
                Ok(())
            }

            // the IfElse dual: cond + both arm bodies + both result vecs are
            // all NodeId edges. listing the bodies (not just the results) keeps
            // arm-internal nodes reachable for DCE — same rule as Op::Scope.
            Op::IfElse {
                cond,
                then_body,
                then_results,
                else_body,
                else_results,
            } => {
                *cond = f(*cond)?;
                for n in then_body.iter_mut() {
                    *n = f(*n)?;
                }
                for n in then_results.iter_mut() {
                    *n = f(*n)?;
                }
                for n in else_body.iter_mut() {
                    *n = f(*n)?;
                }
                for n in else_results.iter_mut() {
                    *n = f(*n)?;
                }
                Ok(())
            }

            // single NodeId field (the multi-output source).
            Op::Proj { source, .. } => {
                *source = f(*source)?;
                Ok(())
            }
        }
    }

    /// read-only view: every NodeId this op references as input, in declared
    /// order. DERIVED from `try_map_inputs` — there is no second list to keep
    /// in sync with the enum shape.
    pub fn inputs(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut clone = self.clone();
        // infallible visit — the closure never errors, so the Result is `Ok`.
        let _: Result<(), std::convert::Infallible> = clone.try_map_inputs(|id| {
            out.push(id);
            Ok(id)
        });
        out
    }

    /// re-insert this op (already remapped to live in `target`'s NodeId space)
    /// by calling the matching public builder on `target`. ONE arm per variant,
    /// dispatched in a single call (alongside `try_map_inputs` and `inputs`).
    ///
    /// caller contract: every NodeId field of `self` MUST already be a valid
    /// id in `target` (i.e., `try_map_inputs` has been run against the splice
    /// remap). builders re-run shape inference / hash-consing on `target`.
    ///
    /// `Param` and `Lambda` are NOT generically dispatchable:
    ///   - `Param` needs a cross-graph symbol -> NodeId substitution table
    ///     (splice's `param_subst`, import_subgraph's `resolve_leaf`) — the
    ///     "remap" closure can't synthesize that.
    ///   - `Lambda(FnId)` references a `FnDef` in the source graph's `lambdas`
    ///     table; lifting it into `target` requires cloning the FnDef (and
    ///     recursively its body sub-graph), which the caller owns.
    /// both variants panic via `unreachable!`; call sites MUST handle them
    /// before delegating to `dispatch_builder`.
    pub fn dispatch_builder(self, target: &mut Graph, span: Option<Span>) -> NodeId {
        match self {
            Op::Const(v) => target.add_const(v, span),
            Op::Construct(elems) => target.construct(elems, span),
            Op::Index(input, dims) => target.index(input, dims, span),
            Op::Broadcast(input, shape) => target.broadcast(input, shape, span),
            Op::ElementWise(op, inputs) => target.element_wise(op, inputs, span),
            Op::Transcendental(op, inputs) => target.transcendental(op, inputs, span),
            Op::Reduce(op, axes, input) => target.reduce(op, axes, input, span),
            Op::Select(c, t, e) => target.select(c, t, e, span),
            Op::LoadAt(sym, components) => target.load_at(sym, components, span),
            Op::Apply { lambda, args } => target.apply(lambda, args, span),
            Op::Fold {
                lambda,
                init,
                count,
            } => target.fold(lambda, init, count, span),
            Op::IterAcc(idx) => target.iter_acc(idx, span),
            Op::IterateInline {
                accs,
                inits,
                steps,
                count,
                result,
                break_when,
            } => target.iterate_inline(accs, inits, steps, count, result, break_when, span),
            // docs/design/23: Op::Scope rebuilt by direct push
            // through the `scope_op` builder (which derives ty from result
            // and bypasses hash-cons).
            Op::Scope { body, result } => target.scope_op(body, result, span),
            // the IfElse dual rebuilt through `if_else` (derives ty from
            // then_results[0], bypasses hash-cons like scope_op).
            Op::IfElse {
                cond,
                then_body,
                then_results,
                else_body,
                else_results,
            } => target.if_else(cond, then_body, then_results, else_body, else_results, span),
            Op::Proj { source, index } => target.proj(source, index, span),
            // Param / Lambda CANNOT be dispatched generically — Param needs a
            // param_subst lookup (cross-graph), Lambda needs FnDef cloning
            // (cross-graph). callers handle these BEFORE calling dispatch_builder.
            Op::Param(_) | Op::Lambda(_) => unreachable!(
                "dispatch_builder: Param/Lambda require cross-graph context \
                 (param_subst / FnDef clone); call sites must handle them \
                 before delegating",
            ),
        }
    }
}

// ----- node -----

/// one IR node: its `Op` plus optional source span for diagnostics.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub span: Option<Span>,
}

// ----- graph -----

/// the tensor IR graph. nodes and types are kept in parallel vectors so
/// `types[i]` is always the result type of `nodes[i]`. shape errors
/// produced during construction accumulate on the side and are drained
/// at the macro boundary (per spec § 12.4).
/// F2.C: opaque handle into a Graph's `lambdas` table. distinguished
/// from NodeId so the type system catches accidental mixing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FnId(pub u32);

/// F2.C: structural function definition. lives in `Graph.lambdas`,
/// referenced by `Op::Lambda(FnId)`. `body` is a sub-Graph; `output`
/// is the NodeId within that sub-Graph that yields the result.
///
/// every parameter name in `params` matches an `Op::Param(sym)` in
/// `body` — at apply time the parameter NodeId is substituted with
/// the caller's argument NodeId. the FnDef.body is not physically
/// spliced at apply time: scalarize/emit walk it once per FnId
/// and produce one device function; Apply sites emit a call.
#[derive(Clone, Debug)]
pub struct FnDef {
    pub name: Symbol,
    pub params: Vec<(Symbol, TensorTy)>,
    pub body: Graph,
    pub output: NodeId,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    types: Vec<TensorTy>,
    params: Vec<(Symbol, NodeId)>,
    /// name -> param NodeId, for fast lookup during macro expansion.
    /// duplicates rejected at insertion time.
    param_index: HashMap<Symbol, NodeId>,
    output: Option<NodeId>,
    errors: Vec<ShapeError>,
    /// graph-level dim constraint. when set, restricts the
    /// kernel to a specific ndim during macro emission. the constraint is
    /// carried explicitly here.
    /// `None` means "no graph-level constraint".
    pinned_ndim: Option<u8>,
    /// F2.A: structural cache for hash-consing. keyed on (Op, output
    /// TensorTy); equal keys map to the same NodeId. populated by
    /// `push` after a successful insert. invariant: two semantically
    /// identical sub-graphs share a single NodeId, so downstream
    /// passes (scalarize, emit) walk shared subterms once.
    ///
    /// the cache is intentionally NOT used for `Op::Param` — params
    /// are already deduped via `param_index` keyed on the symbol,
    /// which is the externally-visible identity for inputs. duplicate
    /// non-param ops with mismatched span info still collapse; the
    /// kept node uses the FIRST span seen.
    hashcons: HashMap<(Op, TensorTy), NodeId>,
    /// F2.C: first-class function definitions. addressable by FnId
    /// (the index here). `Op::Lambda(FnId)` makes a NodeId for the
    /// function value; `Op::Apply { lambda, args }` invokes it.
    lambdas: Vec<FnDef>,
    /// F2.E: `FnDef.name` -> Lambda NodeId for that function. lets
    /// `get_or_register_lambda` return a stable NodeId across calls
    /// so every Apply for the same function references one Lambda.
    /// parallel to `param_index` (which serves the same role for params).
    lambda_index: HashMap<Symbol, NodeId>,
}

impl Graph {
    /// build an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// is this graph empty?
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// access a node by id. panics if `id` is out of bounds — the
    /// invariant is that every `NodeId` from this graph is in-bounds.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// access the result type of a node.
    pub fn ty(&self, id: NodeId) -> &TensorTy {
        &self.types[id.0 as usize]
    }

    /// iterate all `(NodeId, Node, TensorTy)` triples.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node, &TensorTy)> {
        self.nodes
            .iter()
            .zip(self.types.iter())
            .enumerate()
            .map(|(i, (n, t))| (NodeId(i as u32), n, t))
    }

    /// declared params, in insertion order.
    pub fn params(&self) -> &[(Symbol, NodeId)] {
        &self.params
    }

    /// look up a param by its symbol.
    pub fn param(&self, name: &Symbol) -> Option<NodeId> {
        self.param_index.get(name).copied()
    }

    /// set the graph's output node. last call wins.
    pub fn set_output(&mut self, id: NodeId) {
        self.output = Some(id);
    }

    /// the output node, if set.
    pub fn output(&self) -> Option<NodeId> {
        self.output
    }

    /// F5.4-retire: pin a graph-level ndim constraint. last call wins.
    /// the macro pipeline reads `pinned_ndim()` and intersects it with
    /// any other ndim constraints during emission.
    pub fn pin_ndim(&mut self, ndim: u8) {
        self.pinned_ndim = Some(ndim);
    }

    /// returns the graph-level pinned ndim, if any.
    pub fn pinned_ndim(&self) -> Option<u8> {
        self.pinned_ndim
    }

    // ----- error accumulator (R.1.e) -----

    /// has any builder recorded a shape error so far?
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// peek at accumulated errors without consuming.
    pub fn errors(&self) -> &[ShapeError] {
        &self.errors
    }

    /// drain accumulated errors. the typical macro pattern: build the
    /// whole graph, then `take_errors()`; if non-empty, emit them as
    /// `compile_error!`s and abort code generation.
    pub fn take_errors(&mut self) -> Vec<ShapeError> {
        std::mem::take(&mut self.errors)
    }

    /// record a shape error. used internally by op builders in R.2+;
    /// public to let test code exercise the accumulator without
    /// constructing a real op-failure path.
    pub fn record_error(&mut self, err: ShapeError) {
        self.errors.push(err);
    }

    // ----- builders (R.1.d: Const + Param only; R.2 adds the rest) -----

    /// add a rank-0 constant. type is derived from the value's element.
    /// span is optional; pass None at non-macro construction sites.
    pub fn add_const(&mut self, value: ConstValue, span: Option<Span>) -> NodeId {
        let ty = TensorTy::scalar(value.element());
        self.push(Op::Const(value), ty, span)
    }

    /// add a typed parameter. the same symbol can only be registered
    /// once per graph; subsequent attempts return the existing NodeId
    /// without altering the graph.
    pub fn add_param(&mut self, name: Symbol, ty: TensorTy, span: Option<Span>) -> NodeId {
        if let Some(existing) = self.param_index.get(&name) {
            return *existing;
        }
        let id = self.push(Op::Param(name.clone()), ty, span);
        self.params.push((name.clone(), id));
        self.param_index.insert(name, id);
        id
    }

    /// shorthand: scalar parameter of the given element type.
    pub fn add_scalar_param(&mut self, name: &str, element: ElementTy) -> NodeId {
        let sym = Symbol::intern(name);
        self.add_param(sym, TensorTy::scalar(element), None)
    }

    // ----- R.2.a builders: Construct, Index, Broadcast -----

    /// stack N tensors of rank K into one tensor of rank K+1 with a
    /// new outermost axis of length N. all inputs must agree on
    /// element and shape (mismatches record errors and produce a
    /// best-effort poison node).
    pub fn construct(&mut self, elems: Vec<NodeId>, span: Option<Span>) -> NodeId {
        if elems.is_empty() {
            self.record_error(ShapeError::Other {
                message: "Construct requires at least one input element".to_string(),
                span,
            });
            // poison: rank-0 placeholder so downstream queries don't panic.
            return self.push(Op::Construct(elems), TensorTy::scalar(ElementTy::F64), span);
        }
        let first_ty = self.types[elems[0].0 as usize].clone();

        let first_span = self.nodes[elems[0].0 as usize].span;
        for (i, id) in elems.iter().enumerate().skip(1) {
            // clone fields out before any record_error borrow, since
            // record_error needs &mut self.
            let t_element = self.types[id.0 as usize].element;
            let t_rank = self.types[id.0 as usize].rank;
            let t_shape = self.types[id.0 as usize].shape.clone();
            let t_span = self.nodes[id.0 as usize].span;

            if t_element != first_ty.element {
                self.record_error(ShapeError::ElementMismatch {
                    left: first_ty.element,
                    right: t_element,
                    span_a: first_span,
                    span_b: t_span,
                    context: format!("Construct input {}", i),
                });
            }
            if t_shape != first_ty.shape {
                self.record_error(ShapeError::RankMismatch {
                    expected: first_ty.rank,
                    found: t_rank,
                    span: t_span,
                    context: format!("Construct input {}", i),
                });
            }
        }

        // output: rank K+1 with Literal(N) prepended.
        let mut new_shape = Vec::with_capacity(first_ty.shape.len() + 1);
        new_shape.push(DimExpr::Literal(elems.len()));
        new_shape.extend(first_ty.shape.iter().cloned());
        let out_ty = TensorTy {
            element: first_ty.element,
            rank: first_ty.rank + 1,
            shape: new_shape,
        };
        self.push(Op::Construct(elems), out_ty, span)
    }

    /// extract a rank-0 element from a tensor. one DimIndex per axis;
    /// Literal indices against Literal axes are bounds-checked at build
    /// time.
    pub fn index(&mut self, tensor: NodeId, idxs: Vec<DimIndex>, span: Option<Span>) -> NodeId {
        let t = self.types[tensor.0 as usize].clone();
        if idxs.len() as u32 != t.rank {
            self.record_error(ShapeError::RankMismatch {
                expected: t.rank,
                found: idxs.len() as u32,
                span,
                context: "Index".to_string(),
            });
        } else {
            for (axis, idx) in idxs.iter().enumerate() {
                if let (DimIndex::Literal(k), DimExpr::Literal(dim)) = (idx, &t.shape[axis])
                    && *k >= *dim
                {
                    self.record_error(ShapeError::Other {
                        message: format!(
                            "Index out of bounds on axis {}: index {} >= dim {}",
                            axis, k, dim
                        ),
                        span,
                    });
                }
            }
        }
        let out_ty = TensorTy {
            element: t.element,
            rank: 0,
            shape: vec![],
        };
        self.push(Op::Index(tensor, idxs), out_ty, span)
    }

    /// explicitly broadcast a tensor to a target shape. the input's
    /// shape must broadcast to `target` per § 1.5.
    pub fn broadcast(
        &mut self,
        tensor: NodeId,
        target: Vec<DimExpr>,
        span: Option<Span>,
    ) -> NodeId {
        let t = self.types[tensor.0 as usize].clone();
        if !broadcasts_to(&t.shape, &target) {
            self.record_error(ShapeError::BroadcastIncompatible {
                left: t.shape.clone(),
                right: target.clone(),
                span,
                context: "Broadcast".to_string(),
            });
        }
        let out_ty = TensorTy {
            element: t.element,
            rank: target.len() as u32,
            shape: target.clone(),
        };
        self.push(Op::Broadcast(tensor, target), out_ty, span)
    }

    // ----- R.2.b builders: ElementWise + Transcendental -----

    /// element-wise op (arithmetic, comparison, classification). arity
    /// is fixed per op tag. binary ops broadcast their inputs to a
    /// common shape; unary ops preserve shape. result element follows
    /// `returns_bool()`.
    pub fn element_wise(
        &mut self,
        op: ElementWiseOp,
        inputs: Vec<NodeId>,
        span: Option<Span>,
    ) -> NodeId {
        let want_arity = op.arity();
        if inputs.len() != want_arity {
            self.record_error(ShapeError::Other {
                message: format!(
                    "ElementWise({}) requires {} input(s), got {}",
                    op.name(),
                    want_arity,
                    inputs.len()
                ),
                span,
            });
            return self.push(
                Op::ElementWise(op, inputs),
                TensorTy::scalar(if op.returns_bool() {
                    ElementTy::Bool
                } else {
                    ElementTy::F64
                }),
                span,
            );
        }

        // gather input types up front to dodge the borrow-checker for record_error.
        let mut inputs = inputs;
        let mut in_tys: Vec<TensorTy> = inputs
            .iter()
            .map(|id| self.types[id.0 as usize].clone())
            .collect();
        let in_spans: Vec<Option<Span>> = inputs
            .iter()
            .map(|id| self.nodes[id.0 as usize].span)
            .collect();

        // the usual arithmetic conversions: a binary numeric op with mixed int/float
        // operands promotes the narrower operand to the common float type via an
        // IMPLICIT `Cast` (e.g., an i32 index times an f64 grid width -> both f64).
        // homogeneous ops are untouched (no Cast inserted), so existing kernels are
        // byte-identical. this is the one bridge from index space to physical space.
        if want_arity == 2 && op.promotes() {
            if let Some(common) = numeric_promote(in_tys[0].element, in_tys[1].element) {
                for k in 0..2 {
                    if in_tys[k].element != common {
                        let mut ty = in_tys[k].clone();
                        ty.element = common;
                        let arg = inputs[k];
                        inputs[k] = self.push(
                            Op::ElementWise(ElementWiseOp::Cast(common), vec![arg]),
                            ty,
                            span,
                        );
                    }
                }
                in_tys = inputs
                    .iter()
                    .map(|id| self.types[id.0 as usize].clone())
                    .collect();
            }
        }

        // element checks
        let first_elem = in_tys[0].element;
        for (i, t) in in_tys.iter().enumerate().skip(1) {
            if t.element != first_elem {
                self.record_error(ShapeError::ElementMismatch {
                    left: first_elem,
                    right: t.element,
                    span_a: in_spans[0],
                    span_b: in_spans[i],
                    context: format!("ElementWise({}) input {}", op.name(), i),
                });
            }
        }
        if op.requires_float() && !first_elem.is_float() {
            self.record_error(ShapeError::Other {
                message: format!(
                    "ElementWise({}) requires a float input, got {}",
                    op.name(),
                    first_elem
                ),
                span,
            });
        }
        // floor division is the index-space primitive: integer-only by design.
        // a float operand means physical-space data leaked into an index
        // expression — the float->int gather smell the lattice algebra bans.
        if matches!(op, ElementWiseOp::FloorDiv) && !first_elem.is_integer() {
            self.record_error(ShapeError::Other {
                message: format!("ElementWise(FloorDiv) requires integer inputs, got {first_elem}"),
                span,
            });
        }

        // shape broadcast
        let out_shape = if want_arity == 1 {
            in_tys[0].shape.clone()
        } else {
            match broadcast_shape(&in_tys[0].shape, &in_tys[1].shape) {
                Some(s) => s,
                None => {
                    self.record_error(ShapeError::BroadcastIncompatible {
                        left: in_tys[0].shape.clone(),
                        right: in_tys[1].shape.clone(),
                        span,
                        context: format!("ElementWise({})", op.name()),
                    });
                    in_tys[0].shape.clone()
                }
            }
        };

        // Cast(to) always produces the TARGET element — the input type is what
        // is being cast FROM. for every other op the output element follows
        // the (homogeneous) input element. returns_bool ops always produce
        // Bool. without this Cast case, an external caller that re-inserts a
        // standalone Cast (splice / import_subgraph going through
        // `dispatch_builder`) would type the result as the input element,
        // miss hashcons against the existing target-typed node, and produce a
        // duplicate that downstream promotion then double-wraps.
        let out_element = if op.returns_bool() {
            ElementTy::Bool
        } else if let ElementWiseOp::Cast(to) = op {
            to
        } else {
            first_elem
        };
        let out_ty = TensorTy {
            element: out_element,
            rank: out_shape.len() as u32,
            shape: out_shape,
        };
        // arithmetic-identity smart-constructor fold. catches the
        // patterns the IR is most prone to emit when contracting against unit
        // vectors / one-hot tensors: `x + 0`, `x * 1`, `x - 0`, `x / 1`. these
        // never enter the graph — `self.push` is never reached for them, so
        // they're never hash-consed, never lowered, never CSE'd, never emitted.
        // the absorbing `x * 0 -> 0` pattern is NOT folded here OR in the
        // ScalarExpr-level fold (see `passes/scalarize.rs::fold_arith_identity`):
        // IEEE-754 `inf * 0 = NaN`, and `feedback_no_silent_floors` requires
        // that NaN propagate so dt-reduction / regression machinery sees it.
        if want_arity == 2
            && let Some(folded) = self.fold_arith_identity(op, inputs[0], inputs[1])
        {
            return folded;
        }
        self.push(Op::ElementWise(op, inputs), out_ty, span)
    }

    /// recognise rank-0 F64/F32/I32/U32 literals equal to `target` (0.0 or 1.0).
    /// floats compare under `==`; literal `0.0`/`1.0` constants from the IR are
    /// always bit-exact zero / one, so this is the safe case the smart
    /// constructor needs (no signed-zero / `NaN==NaN` traps).
    fn const_eq(&self, id: NodeId, target: f64) -> bool {
        let Op::Const(c) = &self.nodes[id.0 as usize].op else {
            return false;
        };
        match c {
            ConstValue::F64(x) => *x == target,
            ConstValue::F32(x) => *x as f64 == target,
            ConstValue::I32(x) => *x as f64 == target,
            ConstValue::U32(x) => *x as f64 == target,
            ConstValue::Bool(_) => false,
        }
    }

    /// arithmetic-identity peephole used by `element_wise`. returns the folded
    /// `NodeId` if an identity matches; `None` otherwise. SAFE patterns only:
    ///   `x + 0` / `0 + x` -> `x`
    ///   `x - 0`           -> `x`
    ///   `x * 1` / `1 * x` -> `x`
    ///   `x / 1`           -> `x`
    /// shape-preserving and IEEE-safe across all numeric element types.
    ///
    /// the absorbing pattern `x * 0 -> 0` is DELIBERATELY ABSENT — IEEE-754
    /// `inf * 0 = NaN` and the project's `feedback_no_silent_floors` policy
    /// requires NaN to propagate, not be masked. the sibling fold in
    /// `passes/scalarize.rs::fold_arith_identity` MUST match this set
    /// exactly (`{ Add[0], Sub[0], Mul[1], Div[1] }`); changes to either
    /// layer require a matching change to the other.
    fn fold_arith_identity(&self, op: ElementWiseOp, a: NodeId, b: NodeId) -> Option<NodeId> {
        match op {
            ElementWiseOp::Add => {
                if self.const_eq(b, 0.0) {
                    Some(a)
                } else if self.const_eq(a, 0.0) {
                    Some(b)
                } else {
                    None
                }
            }
            ElementWiseOp::Sub => {
                if self.const_eq(b, 0.0) {
                    Some(a)
                } else {
                    None
                }
            }
            ElementWiseOp::Mul => {
                if self.const_eq(b, 1.0) {
                    Some(a)
                } else if self.const_eq(a, 1.0) {
                    Some(b)
                } else {
                    None
                }
            }
            ElementWiseOp::Div => {
                if self.const_eq(b, 1.0) {
                    Some(a)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// transcendental op. arity 1 or 2 per op tag. inputs must be
    /// float; all inputs share element + shape (broadcast-aware for
    /// binary).
    pub fn transcendental(
        &mut self,
        op: TranscendentalOp,
        inputs: Vec<NodeId>,
        span: Option<Span>,
    ) -> NodeId {
        let want_arity = op.arity();
        if inputs.len() != want_arity {
            self.record_error(ShapeError::Other {
                message: format!(
                    "Transcendental({}) requires {} input(s), got {}",
                    op.name(),
                    want_arity,
                    inputs.len()
                ),
                span,
            });
            return self.push(
                Op::Transcendental(op, inputs),
                TensorTy::scalar(ElementTy::F64),
                span,
            );
        }

        let in_tys: Vec<TensorTy> = inputs
            .iter()
            .map(|id| self.types[id.0 as usize].clone())
            .collect();
        let in_spans: Vec<Option<Span>> = inputs
            .iter()
            .map(|id| self.nodes[id.0 as usize].span)
            .collect();

        let first_elem = in_tys[0].element;
        if !first_elem.is_float() {
            self.record_error(ShapeError::Other {
                message: format!(
                    "Transcendental({}) requires a float input, got {}",
                    op.name(),
                    first_elem
                ),
                span,
            });
        }
        for (i, t) in in_tys.iter().enumerate().skip(1) {
            if t.element != first_elem {
                self.record_error(ShapeError::ElementMismatch {
                    left: first_elem,
                    right: t.element,
                    span_a: in_spans[0],
                    span_b: in_spans[i],
                    context: format!("Transcendental({}) input {}", op.name(), i),
                });
            }
        }

        let out_shape = if want_arity == 1 {
            in_tys[0].shape.clone()
        } else {
            match broadcast_shape(&in_tys[0].shape, &in_tys[1].shape) {
                Some(s) => s,
                None => {
                    self.record_error(ShapeError::BroadcastIncompatible {
                        left: in_tys[0].shape.clone(),
                        right: in_tys[1].shape.clone(),
                        span,
                        context: format!("Transcendental({})", op.name()),
                    });
                    in_tys[0].shape.clone()
                }
            }
        };

        let out_ty = TensorTy {
            element: first_elem,
            rank: out_shape.len() as u32,
            shape: out_shape,
        };
        self.push(Op::Transcendental(op, inputs), out_ty, span)
    }

    // ----- R.2.c builders: Reduce + Select -----

    /// reduce a tensor along the given axes. axes must be sorted, in
    /// bounds, with no duplicates. Min/Max accept float or int inputs;
    /// Or/And/Xor accept int or bool.
    pub fn reduce(
        &mut self,
        op: ReduceOp,
        axes: Vec<u32>,
        input: NodeId,
        span: Option<Span>,
    ) -> NodeId {
        let t = self.types[input.0 as usize].clone();

        // validate axes: sorted, in bounds, no duplicates
        let mut prev: Option<u32> = None;
        let mut axes_ok = true;
        for a in &axes {
            if *a >= t.rank {
                self.record_error(ShapeError::Other {
                    message: format!(
                        "Reduce({}) axis {} out of bounds for rank {}",
                        op.name(),
                        a,
                        t.rank
                    ),
                    span,
                });
                axes_ok = false;
            }
            if let Some(p) = prev {
                if *a <= p {
                    self.record_error(ShapeError::Other {
                        message: format!(
                            "Reduce({}) axes must be sorted ascending without duplicates; got {:?}",
                            op.name(),
                            axes
                        ),
                        span,
                    });
                    axes_ok = false;
                    break;
                }
            }
            prev = Some(*a);
        }

        // element compatibility
        if !op.accepts_element(t.element) {
            self.record_error(ShapeError::Other {
                message: format!(
                    "Reduce({}) does not accept element type {}",
                    op.name(),
                    t.element
                ),
                span,
            });
        }

        // output shape: drop reduced axes (when axis list is valid)
        let out_shape: Vec<DimExpr> = if axes_ok {
            t.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !axes.contains(&(*i as u32)))
                .map(|(_, d)| d.clone())
                .collect()
        } else {
            t.shape.clone()
        };

        let out_ty = TensorTy {
            element: t.element,
            rank: out_shape.len() as u32,
            shape: out_shape,
        };
        self.push(Op::Reduce(op, axes, input), out_ty, span)
    }

    /// element-wise if/else. cond.element must be Bool; then/else must
    /// share element and broadcast-compatible shapes; cond participates
    /// in the broadcast too.
    pub fn select(
        &mut self,
        cond: NodeId,
        then_branch: NodeId,
        else_branch: NodeId,
        span: Option<Span>,
    ) -> NodeId {
        let c = self.types[cond.0 as usize].clone();
        let t = self.types[then_branch.0 as usize].clone();
        let e = self.types[else_branch.0 as usize].clone();
        let c_span = self.nodes[cond.0 as usize].span;
        let t_span = self.nodes[then_branch.0 as usize].span;
        let e_span = self.nodes[else_branch.0 as usize].span;

        if c.element != ElementTy::Bool {
            self.record_error(ShapeError::Other {
                message: format!("Select condition must have element Bool, got {}", c.element),
                span: c_span.or(span),
            });
        }
        if t.element != e.element {
            self.record_error(ShapeError::ElementMismatch {
                left: t.element,
                right: e.element,
                span_a: t_span,
                span_b: e_span,
                context: "Select then/else".to_string(),
            });
        }

        // broadcast cond, then, else into one shape
        let shape_te = match broadcast_shape(&t.shape, &e.shape) {
            Some(s) => s,
            None => {
                self.record_error(ShapeError::BroadcastIncompatible {
                    left: t.shape.clone(),
                    right: e.shape.clone(),
                    span,
                    context: "Select then/else".to_string(),
                });
                t.shape.clone()
            }
        };
        let out_shape = match broadcast_shape(&shape_te, &c.shape) {
            Some(s) => s,
            None => {
                self.record_error(ShapeError::BroadcastIncompatible {
                    left: shape_te.clone(),
                    right: c.shape.clone(),
                    span,
                    context: "Select cond vs then/else".to_string(),
                });
                shape_te
            }
        };

        let out_ty = TensorTy {
            element: t.element,
            rank: out_shape.len() as u32,
            shape: out_shape,
        };
        self.push(Op::Select(cond, then_branch, else_branch), out_ty, span)
    }

    /// arbitrary-coord field load. `field_key` must name a field that
    /// is also registered as an input Param (the chalkboard dispatch
    /// resolves the buffer index from the same key). `components` must
    /// be rank-0 scalars; the count is the kernel's spatial ndim.
    /// result is a rank-0 scalar of the field's element type (f64).
    pub fn load_at(
        &mut self,
        field_key: Symbol,
        components: Vec<NodeId>,
        span: Option<Span>,
    ) -> NodeId {
        if components.is_empty() {
            self.record_error(ShapeError::Other {
                message: "LoadAt requires at least one coordinate component".to_string(),
                span,
            });
        }
        for (i, c) in components.iter().enumerate() {
            let t = &self.types[c.0 as usize];
            if t.rank != 0 {
                self.record_error(ShapeError::Other {
                    message: format!("LoadAt component {} must be rank-0, got rank {}", i, t.rank),
                    span,
                });
            }
        }
        self.push(
            Op::LoadAt(field_key, components),
            TensorTy::scalar(ElementTy::F64),
            span,
        )
    }

    // ----- F2.C: first-class functions -----

    /// register a function in the graph's `lambdas` table and return
    /// a NodeId for the corresponding `Op::Lambda(FnId)`. the FnDef
    /// is moved into the graph; its `body` becomes graph-owned.
    pub fn add_lambda(&mut self, fn_def: FnDef, span: Option<Span>) -> NodeId {
        let fn_id = FnId(self.lambdas.len() as u32);
        // bounds-check that `output` is a valid NodeId in the body.
        if (fn_def.output.0 as usize) >= fn_def.body.len() {
            self.record_error(ShapeError::Other {
                message: format!(
                    "add_lambda: FnDef.output NodeId {:?} is out of bounds \
                     for body of length {}",
                    fn_def.output,
                    fn_def.body.len()
                ),
                span,
            });
        }
        let name = fn_def.name.clone();
        self.lambdas.push(fn_def);
        // Lambda carries a placeholder rank-0 F64 type. it is a callable
        // value, not a tensor; only Op::Apply consumes it directly.
        let nid = self.push(Op::Lambda(fn_id), TensorTy::scalar(ElementTy::F64), span);
        // F2.E: register the by-name index so subsequent
        // `get_or_register_lambda` calls return THIS NodeId rather than
        // allocating another Lambda for the same function.
        self.lambda_index.entry(name).or_insert(nid);
        nid
    }

    /// F2.E: return the Lambda NodeId for `fn_def.name`, registering
    /// `fn_def` only if no Lambda for that name already exists in this
    /// graph. ensures every call site referring to function `f` uses
    /// the same Lambda NodeId, so `Op::Apply` hash-cons can merge
    /// (lambda, args) tuples across uses.
    pub fn get_or_register_lambda(&mut self, fn_def: FnDef, span: Option<Span>) -> NodeId {
        if let Some(&existing) = self.lambda_index.get(&fn_def.name) {
            return existing;
        }
        self.add_lambda(fn_def, span)
    }

    /// F2.E: lookup-only variant — does this graph already have a
    /// Lambda for the given function name?
    pub fn find_lambda(&self, name: &Symbol) -> Option<NodeId> {
        self.lambda_index.get(name).copied()
    }

    /// look up the FnDef for a Lambda NodeId. panics if `lambda` is
    /// not an `Op::Lambda`.
    pub fn fn_def(&self, lambda: NodeId) -> &FnDef {
        match &self.nodes[lambda.0 as usize].op {
            Op::Lambda(fn_id) => &self.lambdas[fn_id.0 as usize],
            other => panic!(
                "fn_def: NodeId {:?} is not a Lambda (got {:?})",
                lambda, other
            ),
        }
    }

    /// access all registered FnDefs in declaration order. emitters
    /// walk this to assemble device-function preambles.
    pub fn fn_defs(&self) -> &[FnDef] {
        &self.lambdas
    }

    /// apply a lambda to arguments. arg count must match the lambda's
    /// FnDef.params; arg types must match in element + shape. result
    /// type is the type of the lambda's body output.
    pub fn apply(&mut self, lambda: NodeId, args: Vec<NodeId>, span: Option<Span>) -> NodeId {
        let fn_id = match &self.nodes[lambda.0 as usize].op {
            Op::Lambda(fn_id) => *fn_id,
            other => {
                self.record_error(ShapeError::Other {
                    message: format!(
                        "apply: NodeId {:?} is not a Lambda (got {:?})",
                        lambda, other
                    ),
                    span,
                });
                // poison: rank-0 F64 placeholder.
                return self.push(
                    Op::Apply { lambda, args },
                    TensorTy::scalar(ElementTy::F64),
                    span,
                );
            }
        };
        // clone the FnDef summary out before borrowing `self` mutably.
        let (param_count, expected_tys, result_ty) = {
            let fn_def = &self.lambdas[fn_id.0 as usize];
            let result_ty = fn_def.body.ty(fn_def.output).clone();
            let expected_tys: Vec<TensorTy> =
                fn_def.params.iter().map(|(_, t)| t.clone()).collect();
            (fn_def.params.len(), expected_tys, result_ty)
        };
        if args.len() != param_count {
            self.record_error(ShapeError::Other {
                message: format!(
                    "apply: arity mismatch — lambda expects {} args, got {}",
                    param_count,
                    args.len()
                ),
                span,
            });
        }
        for (i, (arg, expected)) in args.iter().zip(expected_tys.iter()).enumerate() {
            let arg_ty = &self.types[arg.0 as usize];
            if arg_ty.element != expected.element || arg_ty.shape != expected.shape {
                self.record_error(ShapeError::Other {
                    message: format!(
                        "apply: arg {} type mismatch — lambda expects {:?}{:?}, got {:?}{:?}",
                        i, expected.element, expected.shape, arg_ty.element, arg_ty.shape
                    ),
                    span,
                });
            }
        }
        self.push(Op::Apply { lambda, args }, result_ty, span)
    }

    /// F2.F: bounded fold. constructs `Op::Fold { lambda, init, count }`
    /// after validating:
    ///   - `lambda` is `Op::Lambda` whose FnDef has exactly 2 params
    ///     of shape `(Acc, Idx) -> Acc`,
    ///   - `params[1].ty` is rank-0 integer (I32 or U32),
    ///   - `output.ty == params[0].ty`,
    ///   - `init.ty == params[0].ty`,
    ///   - `count.ty` is rank-0 integer.
    ///
    /// the result type matches the accumulator type. errors record on
    /// the graph and the call still returns a placeholder NodeId so
    /// downstream queries don't panic.
    pub fn fold(
        &mut self,
        lambda: NodeId,
        init: NodeId,
        count: NodeId,
        span: Option<Span>,
    ) -> NodeId {
        let fn_id = match &self.nodes[lambda.0 as usize].op {
            Op::Lambda(fid) => *fid,
            other => {
                self.record_error(ShapeError::Other {
                    message: format!(
                        "fold: lambda NodeId {:?} is not a Lambda (got {:?})",
                        lambda, other
                    ),
                    span,
                });
                // poison: rank-0 F64.
                return self.push(
                    Op::Fold {
                        lambda,
                        init,
                        count,
                    },
                    TensorTy::scalar(ElementTy::F64),
                    span,
                );
            }
        };
        // copy out the FnDef summary before borrowing self mutably.
        let (n_params, acc_ty_expected, idx_ty_expected, body_output_ty) = {
            let fn_def = &self.lambdas[fn_id.0 as usize];
            let body_output_ty = fn_def.body.ty(fn_def.output).clone();
            let acc = fn_def
                .params
                .first()
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| TensorTy::scalar(ElementTy::F64));
            let idx = fn_def
                .params
                .get(1)
                .map(|(_, t)| t.clone())
                .unwrap_or_else(|| TensorTy::scalar(ElementTy::I32));
            (fn_def.params.len(), acc, idx, body_output_ty)
        };
        if n_params != 2 {
            self.record_error(ShapeError::Other {
                message: format!(
                    "fold: body lambda must have exactly 2 params (Acc, Idx); got {}",
                    n_params
                ),
                span,
            });
        }
        // index param must be rank-0 integer.
        let idx_is_integer = idx_ty_expected.rank == 0
            && matches!(idx_ty_expected.element, ElementTy::I32 | ElementTy::U32);
        if !idx_is_integer {
            self.record_error(ShapeError::Other {
                message: format!(
                    "fold: body's index param must be rank-0 integer; got {:?}{:?}",
                    idx_ty_expected.element, idx_ty_expected.shape
                ),
                span,
            });
        }
        // body output type must match accumulator type.
        if body_output_ty.element != acc_ty_expected.element
            || body_output_ty.shape != acc_ty_expected.shape
        {
            self.record_error(ShapeError::Other {
                message: format!(
                    "fold: body output type {:?}{:?} does not match \
                     accumulator type {:?}{:?}",
                    body_output_ty.element,
                    body_output_ty.shape,
                    acc_ty_expected.element,
                    acc_ty_expected.shape
                ),
                span,
            });
        }
        // init type must match accumulator type.
        let init_ty = self.types[init.0 as usize].clone();
        if init_ty.element != acc_ty_expected.element || init_ty.shape != acc_ty_expected.shape {
            self.record_error(ShapeError::Other {
                message: format!(
                    "fold: init type {:?}{:?} does not match accumulator \
                     type {:?}{:?}",
                    init_ty.element, init_ty.shape, acc_ty_expected.element, acc_ty_expected.shape
                ),
                span,
            });
        }
        // count must be rank-0 integer.
        let count_ty = &self.types[count.0 as usize];
        let count_is_integer =
            count_ty.rank == 0 && matches!(count_ty.element, ElementTy::I32 | ElementTy::U32);
        if !count_is_integer {
            self.record_error(ShapeError::Other {
                message: format!(
                    "fold: count must be rank-0 integer; got {:?}{:?}",
                    count_ty.element, count_ty.shape
                ),
                span,
            });
        }
        self.push(
            Op::Fold {
                lambda,
                init,
                count,
            },
            acc_ty_expected,
            span,
        )
    }

    /// docs/design/14: a fresh `IterAcc(idx)` placeholder (rank-0 F64) for inline
    /// loop accumulator component `idx`. each call is distinct (bypasses hash-cons)
    /// so two loops (or two components) never share a placeholder.
    pub fn iter_acc(&mut self, idx: u32, span: Option<Span>) -> NodeId {
        self.push(Op::IterAcc(idx), TensorTy::scalar(ElementTy::F64), span)
    }

    /// docs/design/23: build an `Op::Scope` node. `body` lists
    /// the NodeIds created inside the scope's closure (in insertion order —
    /// `Gv::scope` populates this via the `(snapshot..)` range of newly-
    /// pushed NodeIds). `result` is the value the scope returns; the
    /// resulting `Op::Scope` node carries `result`'s TensorTy so downstream
    /// consumers see the right type.
    ///
    /// bypasses hash-cons (see `push()`): two scopes with identical body+
    /// result vectors stay distinct NodeIds because each represents a
    /// distinct lexical region in the user's code.
    pub fn scope_op(&mut self, body: Vec<NodeId>, result: NodeId, span: Option<Span>) -> NodeId {
        let ty = self
            .types
            .get(result.0 as usize)
            .cloned()
            .unwrap_or_else(|| TensorTy::scalar(ElementTy::F64));
        self.push(Op::Scope { body, result }, ty, span)
    }

    /// the DUAL of `iterate_inline`: build an `Op::IfElse` node — a real
    /// data-dependent branch. `cond` is a rank-0 Bool. `then_body`/`else_body`
    /// list the NodeIds created inside each arm's `S::cond` closure (insertion
    /// order, via the `(snapshot..)` range — same convention as `scope_op`).
    /// `then_results`/`else_results` are the per-component values each arm
    /// yields (length 1 for scalar `cond`). the node's type is taken from
    /// `then_results[0]`. bypasses hash-cons (see `push()`): each branch is a
    /// distinct lexical region.
    pub fn if_else(
        &mut self,
        cond: NodeId,
        then_body: Vec<NodeId>,
        then_results: Vec<NodeId>,
        else_body: Vec<NodeId>,
        else_results: Vec<NodeId>,
        span: Option<Span>,
    ) -> NodeId {
        if then_results.len() != else_results.len() {
            self.record_error(ShapeError::Other {
                message: format!(
                    "if_else: then/else result count mismatch ({}/{})",
                    then_results.len(),
                    else_results.len()
                ),
                span,
            });
        }
        let ty = then_results
            .first()
            .and_then(|n| self.types.get(n.0 as usize))
            .cloned()
            .unwrap_or_else(|| TensorTy::scalar(ElementTy::F64));
        self.push(
            Op::IfElse {
                cond,
                then_body,
                then_results,
                else_body,
                else_results,
            },
            ty,
            span,
        )
    }

    /// extract component `index` of a multi-output node (an `Op::IfElse` with N
    /// results). the projection's type is the source's type (all `cond_vec`
    /// outputs are rank-0 scalars). lets `cond_vec` return N distinct values
    /// from one shared branch.
    pub fn proj(&mut self, source: NodeId, index: u32, span: Option<Span>) -> NodeId {
        let ty = self
            .types
            .get(source.0 as usize)
            .cloned()
            .unwrap_or_else(|| TensorTy::scalar(ElementTy::F64));
        self.push(Op::Proj { source, index }, ty, span)
    }

    /// docs/design/14: an inline bounded iteration over an N-component accumulator
    /// vector. `accs[j]` must be `IterAcc(j)` (referenced by the `steps`);
    /// `inits[j]`/`steps[j]` are rank-0 F64; `count` is the literal bound; the
    /// node's value is `accs[result]` post-loop. the N=1 case is a scalar Newton —
    /// see `iterate_inline_scalar`.
    pub fn iterate_inline(
        &mut self,
        accs: Vec<NodeId>,
        inits: Vec<NodeId>,
        steps: Vec<NodeId>,
        count: usize,
        result: u32,
        break_when: Option<NodeId>,
        span: Option<Span>,
    ) -> NodeId {
        if accs.len() != inits.len() || accs.len() != steps.len() {
            self.record_error(ShapeError::Other {
                message: format!(
                    "iterate_inline: accs/inits/steps length mismatch ({}/{}/{})",
                    accs.len(),
                    inits.len(),
                    steps.len()
                ),
                span,
            });
        }
        for (j, &a) in accs.iter().enumerate() {
            if !matches!(self.nodes[a.0 as usize].op, Op::IterAcc(_)) {
                self.record_error(ShapeError::Other {
                    message: format!("iterate_inline: accs[{j}] {a:?} is not an IterAcc"),
                    span,
                });
            }
        }
        let acc_ty = self
            .types
            .get(
                inits
                    .get(result as usize)
                    .map(|n| n.0 as usize)
                    .unwrap_or(0),
            )
            .cloned()
            .unwrap_or_else(|| TensorTy::scalar(ElementTy::F64));
        self.push(
            Op::IterateInline {
                accs,
                inits,
                steps,
                count,
                result,
                break_when,
            },
            acc_ty,
            span,
        )
    }

    /// the scalar (N=1) inline iteration — a fixed-bound masked Newton. wraps the
    /// vector form with one accumulator. `acc` must be `IterAcc(0)`.
    pub fn iterate_inline_scalar(
        &mut self,
        acc: NodeId,
        init: NodeId,
        step: NodeId,
        count: usize,
        break_when: Option<NodeId>,
        span: Option<Span>,
    ) -> NodeId {
        self.iterate_inline(
            vec![acc],
            vec![init],
            vec![step],
            count,
            0,
            break_when,
            span,
        )
    }

    // ----- cross-graph splice -----

    /// graft the pure sub-DAG rooted at `root` (living in `src`) into `self`, returning the
    /// `NodeId` in `self` that computes the same value. each source node is RECREATED via the
    /// structural builders, so `self`'s hash-consing + type inference run on the imported
    /// nodes (shared structure dedups against what `self` already holds — CSE across the splice
    /// boundary). a memo keyed on the source `NodeId` keeps shared subterms shared.
    ///
    /// LEAF resolution: every `Param(sym)` is handed to `resolve_leaf`. return `Some(dst)` to
    /// bind the leaf to an existing node in `self` (the splice point — e.g., a field read the
    /// destination already built); return `None` to recreate it as a fresh param of the same
    /// type. constants are recreated verbatim.
    ///
    /// this is the POINTWISE-physics splice (the use case: a carrier-generic `Regime::wave_speeds`
    /// traced at `S = Gv` grafted into a substrate kernel that owns the geometry). it handles the
    /// pointwise op subset; iteration / lambda / opaque-call / implicit-cast
    /// nodes panic loudly — grafting a whole iterative or lambda-bearing kernel is out of scope.
    pub fn import_subgraph(
        &mut self,
        src: &Graph,
        root: NodeId,
        resolve_leaf: impl Fn(&Symbol) -> Option<NodeId>,
    ) -> NodeId {
        let mut memo: HashMap<NodeId, NodeId> = HashMap::new();
        self.import_node(src, root, &resolve_leaf, &mut memo)
    }

    fn import_node(
        &mut self,
        src: &Graph,
        src_id: NodeId,
        resolve_leaf: &impl Fn(&Symbol) -> Option<NodeId>,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&dst) = memo.get(&src_id) {
            return dst;
        }
        // clone the op to release the `src` borrow before recursing / mutating `self`.
        let op = src.node(src_id).op.clone();
        // SPECIAL cases — cross-graph semantics that `dispatch_builder` can't
        // express. everything else goes through the generic recurse-+-remap-+-
        // dispatch path below (same Phase-3 contract as splice).
        let dst = match op {
            // Param: bind via the caller-supplied leaf resolver; fall back to
            // cloning the param into `self` with its source type. resolve_leaf
            // is the import-side analog of splice's `param_subst`.
            Op::Param(sym) => resolve_leaf(&sym).unwrap_or_else(|| {
                let ty = src.ty(src_id).clone();
                self.add_param(sym, ty, None)
            }),
            // an implicit promotion Cast is re-inserted by the destination builders as needed;
            // importing one standalone would mis-type the result. it never appears in float-only
            // pointwise physics — reject loudly rather than produce a wrong type.
            Op::ElementWise(ElementWiseOp::Cast(_), _) => {
                panic!("import_subgraph: implicit Cast node is not importable")
            }
            // pointwise-physics opset only — lambda / fold / iterate / apply /
            // loadat would either need cross-graph FnDef cloning or are not used
            // inside the grafted regime fragments.
            Op::Lambda(_)
            | Op::Apply { .. }
            | Op::Fold { .. }
            | Op::IterAcc(_)
            | Op::IterateInline { .. } => {
                panic!(
                    "import_subgraph: unsupported op {:?} (pointwise physics only)",
                    op
                )
            }
            // generic path: recurse on every NodeId field (populating `memo`),
            // then re-insert via `Op::dispatch_builder`. one match arm covers
            // every dispatchable variant — same contract as splice.rs.
            mut generic => {
                // first pass: populate `memo` for every child. this resolves
                // ALL recursive imports before mutating `generic`'s NodeId
                // fields, sidestepping the `&mut self` aliasing a per-variant
                // scatter would otherwise require.
                let mut children: Vec<NodeId> = Vec::new();
                let _: Result<(), std::convert::Infallible> =
                    generic.clone().try_map_inputs(|id| {
                        children.push(id);
                        Ok(id)
                    });
                for child in children {
                    let _ = self.import_node(src, child, resolve_leaf, memo);
                }
                // second pass: remap the cloned op's NodeId fields via `memo`,
                // then hand off to the canonical builder dispatcher.
                let _: Result<(), std::convert::Infallible> =
                    generic.try_map_inputs(|id| Ok(memo[&id]));
                generic.dispatch_builder(self, None)
            }
        };
        memo.insert(src_id, dst);
        dst
    }

    // ----- internal -----

    fn push(&mut self, op: Op, ty: TensorTy, span: Option<Span>) -> NodeId {
        // F2.A hash-cons. params bypass — they're externally keyed
        // by symbol and dedup'd in `add_param`. F2.C Lambda also
        // bypasses: each `add_lambda` is meant to allocate a new
        // FnId; sharing happens at the FnDef level (not yet) or via
        // the Apply call sites. Op::Apply still hash-conses.
        // IterAcc: each placeholder is a distinct loop accumulator — sharing two
        // would alias unrelated loops. bypass hash-cons like Param/Lambda.
        // docs/design/23: `Op::Scope` bypasses hash-cons too. each
        // `Gv::scope` call site is a distinct lexical region; deduping two
        // scopes with identical body+result vectors would collapse them into
        // one, but they could be CALLED from different outer contexts where
        // identity matters (e.g., two scopes inside an `Op::Fold` body would
        // each iterate, and merging them would conflate iterations).
        let bypass = matches!(
            &op,
            Op::Param(_) | Op::Lambda(_) | Op::IterAcc(_) | Op::Scope { .. } | Op::IfElse { .. }
        );
        if !bypass {
            if let Some(&existing) = self.hashcons.get(&(op.clone(), ty.clone())) {
                return existing;
            }
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            op: op.clone(),
            span,
        });
        self.types.push(ty.clone());
        debug_assert_eq!(
            self.nodes.len(),
            self.types.len(),
            "nodes and types out of sync"
        );
        if !bypass {
            self.hashcons.insert((op, ty), id);
        }
        id
    }
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DimExpr, error::ShapeError};

    fn lit(n: usize) -> DimExpr {
        DimExpr::Literal(n)
    }

    // ---- NodeId ----

    #[test]
    fn node_ids_are_distinct() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_const(ConstValue::F64(2.0), None);
        assert_ne!(a, b);
    }

    // ---- ConstValue element ----

    #[test]
    fn const_value_reports_correct_element() {
        assert_eq!(ConstValue::F64(0.0).element(), ElementTy::F64);
        assert_eq!(ConstValue::F32(0.0).element(), ElementTy::F32);
        assert_eq!(ConstValue::I32(0).element(), ElementTy::I32);
        assert_eq!(ConstValue::U32(0).element(), ElementTy::U32);
        assert_eq!(ConstValue::Bool(false).element(), ElementTy::Bool);
    }

    // ---- empty graph ----

    #[test]
    fn empty_graph() {
        let g = Graph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert!(g.params().is_empty());
        assert!(g.output().is_none());
    }

    // ---- const builders ----

    #[test]
    fn add_const_f64_is_scalar() {
        let mut g = Graph::new();
        let id = g.add_const(ConstValue::F64(3.14), None);
        let ty = g.ty(id);
        assert_eq!(ty.element, ElementTy::F64);
        assert_eq!(ty.rank, 0);
        assert!(ty.shape.is_empty());
    }

    #[test]
    fn add_const_bool() {
        let mut g = Graph::new();
        let id = g.add_const(ConstValue::Bool(true), None);
        assert_eq!(g.ty(id).element, ElementTy::Bool);
    }

    // ---- param builders ----

    #[test]
    fn add_param_records_in_params_list() {
        let mut g = Graph::new();
        let id = g.add_param(Symbol::intern("x"), TensorTy::scalar(ElementTy::F64), None);
        assert_eq!(g.params().len(), 1);
        assert_eq!(g.params()[0].0.as_str(), "x");
        assert_eq!(g.params()[0].1, id);
    }

    #[test]
    fn duplicate_param_returns_existing_id() {
        let mut g = Graph::new();
        let a = g.add_param(Symbol::intern("x"), TensorTy::scalar(ElementTy::F64), None);
        let b = g.add_param(Symbol::intern("x"), TensorTy::scalar(ElementTy::F64), None);
        assert_eq!(a, b);
        assert_eq!(g.params().len(), 1, "duplicate must not append");
    }

    #[test]
    fn param_can_be_looked_up_by_symbol() {
        let mut g = Graph::new();
        let id = g.add_param(
            Symbol::intern("rho"),
            TensorTy::scalar(ElementTy::F64),
            None,
        );
        let found = g.param(&Symbol::intern("rho")).unwrap();
        assert_eq!(found, id);
        assert!(g.param(&Symbol::intern("missing")).is_none());
    }

    #[test]
    fn add_scalar_param_shorthand_works() {
        let mut g = Graph::new();
        let id = g.add_scalar_param("v", ElementTy::F64);
        assert_eq!(g.ty(id), &TensorTy::scalar(ElementTy::F64));
        assert_eq!(g.params().len(), 1);
    }

    #[test]
    fn rank_n_param_is_supported() {
        let mut g = Graph::new();
        let ty = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        let id = g.add_param(Symbol::intern("v"), ty.clone(), None);
        assert_eq!(g.ty(id), &ty);
    }

    // ---- iter + ty ----

    #[test]
    fn iter_walks_in_insertion_order() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_const(ConstValue::F32(2.0), None);
        let c = g.add_param(Symbol::intern("p"), TensorTy::scalar(ElementTy::I32), None);

        let ids: Vec<NodeId> = g.iter().map(|(id, _, _)| id).collect();
        assert_eq!(ids, vec![a, b, c]);
    }

    #[test]
    fn iter_yields_matched_types() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_scalar_param("p", ElementTy::I32);

        let pairs: Vec<(NodeId, ElementTy)> =
            g.iter().map(|(id, _, ty)| (id, ty.element)).collect();
        assert_eq!(pairs, vec![(a, ElementTy::F64), (b, ElementTy::I32)]);
    }

    // ---- output ----

    #[test]
    fn output_round_trip() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        assert!(g.output().is_none());
        g.set_output(a);
        assert_eq!(g.output(), Some(a));
    }

    #[test]
    fn set_output_last_wins() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_const(ConstValue::F64(2.0), None);
        g.set_output(a);
        g.set_output(b);
        assert_eq!(g.output(), Some(b));
    }

    // ---- floor division (the index-space refinement primitive) ----

    #[test]
    fn floor_div_types_integer_to_integer() {
        let mut g = Graph::new();
        let a = g.add_param(Symbol::intern("a"), TensorTy::scalar(ElementTy::I32), None);
        let b = g.add_param(Symbol::intern("b"), TensorTy::scalar(ElementTy::I32), None);
        let q = g.element_wise(ElementWiseOp::FloorDiv, vec![a, b], None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(q).element, ElementTy::I32);
    }

    #[test]
    fn floor_div_rejects_float_inputs() {
        // a float operand means physical-space data leaked into an index
        // expression — must be a trace-time error, not a silent cast.
        let mut g = Graph::new();
        let a = g.add_param(Symbol::intern("a"), TensorTy::scalar(ElementTy::F64), None);
        let b = g.add_param(Symbol::intern("b"), TensorTy::scalar(ElementTy::F64), None);
        let _ = g.element_wise(ElementWiseOp::FloorDiv, vec![a, b], None);
        assert!(
            g.has_errors(),
            "FloorDiv on floats must record a shape error"
        );
    }

    // ---- error accumulator (R.1.e) ----

    #[test]
    fn fresh_graph_has_no_errors() {
        let g = Graph::new();
        assert!(!g.has_errors());
        assert!(g.errors().is_empty());
    }

    #[test]
    fn record_error_appends() {
        let mut g = Graph::new();
        g.record_error(ShapeError::Other {
            message: "boom".to_string(),
            span: None,
        });
        assert!(g.has_errors());
        assert_eq!(g.errors().len(), 1);
    }

    #[test]
    fn take_errors_drains() {
        let mut g = Graph::new();
        g.record_error(ShapeError::RankMismatch {
            expected: 2,
            found: 1,
            span: None,
            context: "x".to_string(),
        });
        let drained = g.take_errors();
        assert_eq!(drained.len(), 1);
        assert!(!g.has_errors(), "errors should be empty after take");
    }

    #[test]
    fn errors_accumulate_in_insertion_order() {
        let mut g = Graph::new();
        g.record_error(ShapeError::Other {
            message: "one".into(),
            span: None,
        });
        g.record_error(ShapeError::Other {
            message: "two".into(),
            span: None,
        });
        g.record_error(ShapeError::Other {
            message: "three".into(),
            span: None,
        });
        let drained = g.take_errors();
        assert_eq!(drained.len(), 3);
        // verify order
        let msgs: Vec<String> = drained.iter().map(|e| e.summary()).collect();
        assert_eq!(
            msgs,
            vec!["one".to_string(), "two".to_string(), "three".to_string()]
        );
    }

    #[test]
    fn errors_independent_of_node_construction() {
        // recording an error doesn't add a node, and adding nodes doesn't
        // record errors.
        let mut g = Graph::new();
        let n = g.add_const(ConstValue::F64(0.0), None);
        assert!(!g.has_errors());
        g.record_error(ShapeError::Other {
            message: "x".into(),
            span: None,
        });
        // node count unchanged
        assert_eq!(g.len(), 1);
        // and the node is still queryable
        assert_eq!(g.ty(n).element, ElementTy::F64);
    }

    // ---- R.2.a: Construct ----

    fn vec3(g: &mut Graph) -> NodeId {
        // helper: rank-1 dim-3 Untagged param of element F64.
        let ty = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        g.add_param(Symbol::intern("v3"), ty, None)
    }

    #[test]
    fn construct_stacks_three_scalars_into_rank_1() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_const(ConstValue::F64(2.0), None);
        let c = g.add_const(ConstValue::F64(3.0), None);
        let v = g.construct(vec![a, b, c], None);
        assert!(!g.has_errors());
        let ty = g.ty(v);
        assert_eq!(ty.rank, 1);
        assert_eq!(ty.shape, vec![lit(3)]);
        assert_eq!(ty.element, ElementTy::F64);
    }

    #[test]
    fn construct_stacks_two_vectors_into_matrix() {
        let mut g = Graph::new();
        let v1 = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let v2 = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let m = g.construct(vec![v1, v2], None);
        assert!(!g.has_errors());
        let ty = g.ty(m);
        assert_eq!(ty.rank, 2);
        assert_eq!(ty.shape, vec![lit(2), lit(3)]);
    }

    #[test]
    fn construct_empty_records_error() {
        let mut g = Graph::new();
        let _ = g.construct(vec![], None);
        assert!(g.has_errors());
        let err = g.errors()[0].summary();
        assert!(err.contains("at least one"), "{}", err);
    }

    #[test]
    fn construct_element_mismatch_records_error() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_const(ConstValue::F32(2.0), None);
        let _ = g.construct(vec![a, b], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("element")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn construct_shape_mismatch_records_error() {
        let mut g = Graph::new();
        let v1 = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let v2 = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(4)]),
            None,
        );
        let _ = g.construct(vec![v1, v2], None);
        assert!(g.has_errors());
    }

    // ---- R.2.a: Index ----

    #[test]
    fn index_extracts_scalar() {
        let mut g = Graph::new();
        let v = vec3(&mut g);
        let s = g.index(v, vec![DimIndex::Literal(1)], None);
        assert!(!g.has_errors());
        let ty = g.ty(s);
        assert_eq!(ty.rank, 0);
        assert_eq!(ty.element, ElementTy::F64);
    }

    #[test]
    fn index_rank_mismatch_errors() {
        let mut g = Graph::new();
        let v = vec3(&mut g);
        let _ = g.index(v, vec![DimIndex::Literal(0), DimIndex::Literal(1)], None);
        let err = g.errors()[0].summary();
        assert!(err.contains("rank"), "{}", err);
    }

    #[test]
    fn index_out_of_bounds_errors() {
        let mut g = Graph::new();
        let v = vec3(&mut g);
        let _ = g.index(v, vec![DimIndex::Literal(5)], None);
        let err = g.errors()[0].summary();
        assert!(err.contains("out of bounds"), "{}", err);
    }

    #[test]
    fn index_with_generic_index() {
        // DimIndex::Generic is for inside-loop access during scalarization.
        let mut g = Graph::new();
        let v = vec3(&mut g);
        let _ = g.index(v, vec![DimIndex::Generic(Symbol::intern("ii"))], None);
        assert!(!g.has_errors());
    }

    // ---- R.2.a: Broadcast ----

    #[test]
    fn broadcast_scalar_to_vector() {
        let mut g = Graph::new();
        let s = g.add_const(ConstValue::F64(1.0), None);
        let v = g.broadcast(s, vec![lit(3)], None);
        assert!(!g.has_errors());
        let ty = g.ty(v);
        assert_eq!(ty.rank, 1);
        assert_eq!(ty.shape, vec![lit(3)]);
    }

    #[test]
    fn broadcast_one_to_n_succeeds() {
        let mut g = Graph::new();
        let one = g.add_param(
            Symbol::intern("o"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(1), lit(3)]),
            None,
        );
        let _ = g.broadcast(one, vec![lit(4), lit(3)], None);
        assert!(!g.has_errors());
    }

    #[test]
    fn broadcast_incompatible_errors() {
        let mut g = Graph::new();
        let v = vec3(&mut g); // shape [3]
        let _ = g.broadcast(v, vec![lit(5)], None);
        let err = g.errors()[0].summary();
        assert!(err.contains("incompatible"), "{}", err);
    }

    #[test]
    fn broadcast_preserves_element() {
        let mut g = Graph::new();
        let s = g.add_const(ConstValue::F64(1.0), None);
        let r = g.broadcast(s, vec![lit(3)], None);
        assert_eq!(g.ty(r).element, ElementTy::F64);
    }

    // ---- R.2.b: ElementWise ----

    fn scalar_f64(g: &mut Graph, name: &str) -> NodeId {
        g.add_scalar_param(name, ElementTy::F64)
    }

    fn vec_f64(g: &mut Graph, name: &str, dim: usize) -> NodeId {
        g.add_param(
            Symbol::intern(name),
            TensorTy::from_shape(ElementTy::F64, vec![lit(dim)]),
            None,
        )
    }

    #[test]
    fn elementwise_add_two_scalars() {
        let mut g = Graph::new();
        let a = scalar_f64(&mut g, "a");
        let b = scalar_f64(&mut g, "b");
        let s = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        assert!(!g.has_errors());
        let ty = g.ty(s);
        assert_eq!(ty.element, ElementTy::F64);
        assert_eq!(ty.rank, 0);
    }

    #[test]
    fn elementwise_add_two_vectors() {
        let mut g = Graph::new();
        let a = vec_f64(&mut g, "a", 3);
        let b = vec_f64(&mut g, "b", 3);
        let s = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        assert!(!g.has_errors());
        let ty = g.ty(s);
        assert_eq!(ty.shape, vec![lit(3)]);
    }

    #[test]
    fn elementwise_broadcast_scalar_and_vector() {
        let mut g = Graph::new();
        let s = scalar_f64(&mut g, "s");
        let v = vec_f64(&mut g, "v", 4);
        let out = g.element_wise(ElementWiseOp::Mul, vec![s, v], None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(out).shape, vec![lit(4)]);
    }

    #[test]
    fn elementwise_neg_is_unary() {
        let mut g = Graph::new();
        let v = vec_f64(&mut g, "v", 3);
        let r = g.element_wise(ElementWiseOp::Neg, vec![v], None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).shape, vec![lit(3)]);
    }

    #[test]
    fn elementwise_arity_error() {
        let mut g = Graph::new();
        let a = scalar_f64(&mut g, "a");
        // Add wants 2, pass 1
        let _ = g.element_wise(ElementWiseOp::Add, vec![a], None);
        let err = g.errors()[0].summary();
        assert!(err.contains("Add"), "{}", err);
        assert!(err.contains("2"), "{}", err);
    }

    #[test]
    fn elementwise_element_mismatch() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F32);
        let _ = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("element")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn elementwise_broadcast_incompatible_records_error() {
        let mut g = Graph::new();
        let a = vec_f64(&mut g, "a", 3);
        let b = vec_f64(&mut g, "b", 5);
        let _ = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("incompatible")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn elementwise_comparison_returns_bool() {
        let mut g = Graph::new();
        let a = scalar_f64(&mut g, "a");
        let b = scalar_f64(&mut g, "b");
        let cmp = g.element_wise(ElementWiseOp::Lt, vec![a, b], None);
        assert_eq!(g.ty(cmp).element, ElementTy::Bool);
    }

    #[test]
    fn elementwise_is_nan_requires_float() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::I32);
        let _ = g.element_wise(ElementWiseOp::IsNaN, vec![a], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("float")),
            "{:?}",
            summaries
        );
    }

    // ---- R.2.b: Transcendental ----

    #[test]
    fn transcendental_sin_preserves_element() {
        let mut g = Graph::new();
        let a = scalar_f64(&mut g, "a");
        let r = g.transcendental(TranscendentalOp::Sin, vec![a], None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).element, ElementTy::F64);
    }

    #[test]
    fn transcendental_preserves_shape() {
        let mut g = Graph::new();
        let v = vec_f64(&mut g, "v", 3);
        let r = g.transcendental(TranscendentalOp::Cos, vec![v], None);
        assert_eq!(g.ty(r).shape, vec![lit(3)]);
    }

    #[test]
    fn transcendental_atan2_is_binary() {
        let mut g = Graph::new();
        let y = scalar_f64(&mut g, "y");
        let x = scalar_f64(&mut g, "x");
        let r = g.transcendental(TranscendentalOp::Atan2, vec![y, x], None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).element, ElementTy::F64);
    }

    #[test]
    fn transcendental_pow_arity_error() {
        let mut g = Graph::new();
        let b = scalar_f64(&mut g, "b");
        // Pow wants 2, pass 1
        let _ = g.transcendental(TranscendentalOp::Pow, vec![b], None);
        let err = g.errors()[0].summary();
        assert!(err.contains("Pow"), "{}", err);
    }

    #[test]
    fn transcendental_rejects_non_float() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::I32);
        let _ = g.transcendental(TranscendentalOp::Sin, vec![a], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("float")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn op_name_and_arity_round_trip() {
        // sanity that the metadata is consistent.
        assert_eq!(ElementWiseOp::Add.arity(), 2);
        assert_eq!(ElementWiseOp::Neg.arity(), 1);
        assert_eq!(ElementWiseOp::IsNaN.arity(), 1);
        assert!(ElementWiseOp::Lt.returns_bool());
        assert!(!ElementWiseOp::Mul.returns_bool());
        assert!(ElementWiseOp::Sqrt.requires_float());
        assert!(!ElementWiseOp::Add.requires_float());

        assert_eq!(TranscendentalOp::Atan2.arity(), 2);
        assert_eq!(TranscendentalOp::Sin.arity(), 1);
        assert_eq!(TranscendentalOp::Sin.name(), "Sin");
    }

    // ---- R.2.c: Reduce ----

    #[test]
    fn reduce_max_collapses_one_axis() {
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("m"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(4)]),
            None,
        );
        let r = g.reduce(ReduceOp::Max, vec![1], m, None);
        assert!(!g.has_errors());
        let ty = g.ty(r);
        assert_eq!(ty.shape, vec![lit(3)]);
        assert_eq!(ty.element, ElementTy::F64);
    }

    #[test]
    fn reduce_collapses_multiple_axes() {
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("m"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(4), lit(5)]),
            None,
        );
        let r = g.reduce(ReduceOp::Min, vec![0, 2], m, None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).shape, vec![lit(4)]);
    }

    #[test]
    fn reduce_to_scalar() {
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("m"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(4)]),
            None,
        );
        let r = g.reduce(ReduceOp::Max, vec![0, 1], m, None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).rank, 0);
    }

    #[test]
    fn reduce_axes_out_of_bounds_errors() {
        let mut g = Graph::new();
        let v = vec_f64(&mut g, "v", 3);
        let _ = g.reduce(ReduceOp::Min, vec![5], v, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("out of bounds")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn reduce_axes_unsorted_errors() {
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("m"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3), lit(4)]),
            None,
        );
        let _ = g.reduce(ReduceOp::Min, vec![2, 0], m, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("sorted")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn reduce_axes_duplicate_errors() {
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("m"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let _ = g.reduce(ReduceOp::Max, vec![0, 0], m, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("sorted")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn reduce_and_on_float_errors() {
        let mut g = Graph::new();
        let v = vec_f64(&mut g, "v", 3);
        let _ = g.reduce(ReduceOp::And, vec![0], v, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("element type")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn reduce_or_on_bool_succeeds() {
        let mut g = Graph::new();
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::Bool, vec![lit(3)]),
            None,
        );
        let r = g.reduce(ReduceOp::Or, vec![0], b, None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).element, ElementTy::Bool);
    }

    // ---- R.2.c: Select ----

    fn bool_scalar(g: &mut Graph, name: &str) -> NodeId {
        g.add_param(
            Symbol::intern(name),
            TensorTy::scalar(ElementTy::Bool),
            None,
        )
    }

    #[test]
    fn select_picks_then_or_else() {
        let mut g = Graph::new();
        let c = bool_scalar(&mut g, "c");
        let t = scalar_f64(&mut g, "t");
        let e = scalar_f64(&mut g, "e");
        let r = g.select(c, t, e, None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).element, ElementTy::F64);
        assert_eq!(g.ty(r).rank, 0);
    }

    #[test]
    fn select_broadcasts_cond_then_else() {
        let mut g = Graph::new();
        let c = bool_scalar(&mut g, "c"); // scalar
        let t = vec_f64(&mut g, "t", 4); // [4]
        let e = scalar_f64(&mut g, "e"); // scalar
        let r = g.select(c, t, e, None);
        assert!(!g.has_errors());
        assert_eq!(g.ty(r).shape, vec![lit(4)]);
    }

    #[test]
    fn select_non_bool_cond_errors() {
        let mut g = Graph::new();
        let c = scalar_f64(&mut g, "c"); // wrong: not Bool
        let t = scalar_f64(&mut g, "t");
        let e = scalar_f64(&mut g, "e");
        let _ = g.select(c, t, e, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("Bool")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn select_then_else_element_mismatch_errors() {
        let mut g = Graph::new();
        let c = bool_scalar(&mut g, "c");
        let t = g.add_scalar_param("t", ElementTy::F64);
        let e = g.add_scalar_param("e", ElementTy::F32);
        let _ = g.select(c, t, e, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("element")),
            "{:?}",
            summaries
        );
    }

    #[test]
    fn select_then_else_broadcast_incompatible() {
        let mut g = Graph::new();
        let c = bool_scalar(&mut g, "c");
        let t = vec_f64(&mut g, "t", 3);
        let e = vec_f64(&mut g, "e", 5);
        let _ = g.select(c, t, e, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("incompatible")),
            "{:?}",
            summaries
        );
    }

    // ---- F2.A: hash-cons contract ----

    #[test]
    fn hashcons_same_const_returns_same_id() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(2.5), None);
        let b = g.add_const(ConstValue::F64(2.5), None);
        assert_eq!(a, b, "same constant should share NodeId");
        assert_eq!(g.len(), 1, "duplicate const must not allocate a new node");
    }

    #[test]
    fn hashcons_different_const_distinct() {
        // sanity: hash-cons must NOT merge structurally-different values.
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(2.5), None);
        let b = g.add_const(ConstValue::F64(3.5), None);
        assert_ne!(a, b);
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn hashcons_nan_bit_patterns_disambiguate() {
        // NaN bit patterns differ → different consts. structurally
        // identical NaN bit patterns share.
        let nan1 = f64::from_bits(0x7ff8_0000_0000_0001);
        let nan2 = f64::from_bits(0x7ff8_0000_0000_0002);
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(nan1), None);
        let b = g.add_const(ConstValue::F64(nan2), None);
        let c = g.add_const(ConstValue::F64(nan1), None);
        assert_ne!(a, b, "different NaN bits should be distinct");
        assert_eq!(a, c, "same NaN bits should share");
    }

    #[test]
    fn hashcons_same_binop_returns_same_id() {
        // (a + b) built twice should produce the same NodeId.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let s1 = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let s2 = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        assert_eq!(s1, s2);
        // 2 params + 1 add = 3 nodes total. without hash-cons this would be 4.
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn hashcons_input_order_matters() {
        // (a + b) is structurally different from (b + a) at the IR
        // level (the inputs vec is ordered). hash-cons must NOT merge
        // them — that would be an unproven algebraic-equivalence claim.
        // simple operand-swap normalization belongs
        // in a separate canonicalization pass.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let s_ab = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let s_ba = g.element_wise(ElementWiseOp::Add, vec![b, a], None);
        assert_ne!(s_ab, s_ba);
    }

    #[test]
    fn hashcons_collapses_deep_sharing() {
        // build (x * x) * (x * x). without hash-cons the two (x*x)
        // sub-trees would be separate NodeIds; with it, they share.
        // total: 1 param + 1 mul + 1 outer mul = 3 nodes.
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let xx_a = g.element_wise(ElementWiseOp::Mul, vec![x, x], None);
        let xx_b = g.element_wise(ElementWiseOp::Mul, vec![x, x], None);
        assert_eq!(xx_a, xx_b);
        let outer = g.element_wise(ElementWiseOp::Mul, vec![xx_a, xx_b], None);
        // outer is Mul(xx, xx) where both operands are the same NodeId.
        // graph nodes: x, xx, outer = 3 nodes.
        assert_eq!(g.len(), 3);
        assert_ne!(outer, xx_a);
    }

    // ---- F2.C: Lambda + Apply contracts ----

    fn build_square_fn() -> FnDef {
        // λx. x * x — body has one param + one Mul.
        let mut body = Graph::new();
        let x = body.add_scalar_param("x", ElementTy::F64);
        let sq = body.element_wise(ElementWiseOp::Mul, vec![x, x], None);
        FnDef {
            name: Symbol::intern("square"),
            params: vec![(Symbol::intern("x"), TensorTy::scalar(ElementTy::F64))],
            body,
            output: sq,
        }
    }

    #[test]
    fn lambda_is_a_first_class_node() {
        let mut g = Graph::new();
        let sq = build_square_fn();
        let l = g.add_lambda(sq, None);
        // it's a graph node
        match &g.node(l).op {
            Op::Lambda(_) => {}
            other => panic!("expected Lambda, got {:?}", other),
        }
        // and the FnDef is retrievable via the graph
        assert_eq!(g.fn_def(l).name.as_str(), "square");
        assert_eq!(g.fn_defs().len(), 1);
    }

    #[test]
    fn apply_takes_lambda_and_produces_result_typed_like_body_output() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_square_fn(), None);
        let a = g.add_scalar_param("a", ElementTy::F64);
        let y = g.apply(l, vec![a], None);
        // result type = type of square's body output = scalar F64.
        assert_eq!(g.ty(y).element, ElementTy::F64);
        assert_eq!(g.ty(y).rank, 0);
        // structurally, Apply with the lambda + one arg.
        match &g.node(y).op {
            Op::Apply { lambda, args } => {
                assert_eq!(*lambda, l);
                assert_eq!(args, &vec![a]);
            }
            other => panic!("expected Apply, got {:?}", other),
        }
        assert!(!g.has_errors(), "no errors expected: {:?}", g.errors());
    }

    #[test]
    fn apply_arity_mismatch_records_error() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_square_fn(), None);
        // pass two args to a one-param lambda
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let _ = g.apply(l, vec![a, b], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("arity mismatch")),
            "expected arity error, got {:?}",
            summaries
        );
    }

    #[test]
    fn apply_arg_type_mismatch_records_error() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_square_fn(), None);
        // pass a rank-1 arg to a rank-0 param
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let _ = g.apply(l, vec![v], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("type mismatch")),
            "expected type-mismatch error, got {:?}",
            summaries
        );
    }

    #[test]
    fn apply_with_non_lambda_records_error() {
        let mut g = Graph::new();
        let c = g.add_const(ConstValue::F64(1.0), None);
        let a = g.add_scalar_param("a", ElementTy::F64);
        // c is not a Lambda — Apply must error and produce a poison node
        let _ = g.apply(c, vec![a], None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("not a Lambda")),
            "expected not-a-Lambda error, got {:?}",
            summaries
        );
    }

    #[test]
    fn apply_is_hashconsed() {
        // calling apply(l, [a]) twice should share — F2.A applies to
        // Apply nodes (Lambda nodes bypass via the FnId path).
        let mut g = Graph::new();
        let l = g.add_lambda(build_square_fn(), None);
        let a = g.add_scalar_param("a", ElementTy::F64);
        let y1 = g.apply(l, vec![a], None);
        let y2 = g.apply(l, vec![a], None);
        assert_eq!(y1, y2);
    }

    #[test]
    fn get_or_register_lambda_returns_existing_by_name() {
        // F2.E: same fn name => same NodeId.
        let mut g = Graph::new();
        let l1 = g.get_or_register_lambda(build_square_fn(), None);
        let l2 = g.get_or_register_lambda(build_square_fn(), None);
        assert_eq!(l1, l2, "second call must return the existing Lambda");
        // and the lambdas table holds ONE entry — the second call
        // skipped the add path because the name was already known.
        assert_eq!(g.fn_defs().len(), 1);
    }

    #[test]
    fn find_lambda_returns_none_for_unknown_name() {
        let g = Graph::new();
        assert!(g.find_lambda(&Symbol::intern("square")).is_none());
    }

    #[test]
    fn apply_via_get_or_register_lambda_shares_lambda_nid() {
        // build the Lambda twice via get_or_register; both Apply nodes
        // reference the SAME lambda NodeId, so Apply hash-cons
        // (F2.A + F2.C) collapses them.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let l1 = g.get_or_register_lambda(build_square_fn(), None);
        let y1 = g.apply(l1, vec![a], None);
        let l2 = g.get_or_register_lambda(build_square_fn(), None);
        let y2 = g.apply(l2, vec![a], None);
        assert_eq!(l1, l2);
        assert_eq!(y1, y2, "two Apply(same_lambda, same_args) must collapse");
    }

    #[test]
    fn separate_add_lambda_calls_allocate_distinct_fn_ids() {
        // by design — add_lambda bypasses hash-cons. each call is a
        // fresh FnDef. structural sharing across Lambdas comes later
        // via a separate canonicalization pass on FnDef bodies.
        let mut g = Graph::new();
        let l1 = g.add_lambda(build_square_fn(), None);
        let l2 = g.add_lambda(build_square_fn(), None);
        assert_ne!(l1, l2, "each add_lambda must produce a fresh node");
        assert_eq!(g.fn_defs().len(), 2);
    }

    // ---- F2.F: Op::Fold contracts ----

    /// build a lambda of shape `(acc: f64, i: i32) -> f64`. body: acc + 1.0.
    /// the iteration index `i` is unused — that's allowed and tests the
    /// "lambda may ignore the index" semantics.
    fn build_inc_fold_fn() -> FnDef {
        let mut body = Graph::new();
        let acc = body.add_scalar_param("acc", ElementTy::F64);
        let _i = body.add_scalar_param("i", ElementTy::I32);
        let one = body.add_const(ConstValue::F64(1.0), None);
        let out = body.element_wise(ElementWiseOp::Add, vec![acc, one], None);
        FnDef {
            name: Symbol::intern("inc_fold_body"),
            params: vec![
                (Symbol::intern("acc"), TensorTy::scalar(ElementTy::F64)),
                (Symbol::intern("i"), TensorTy::scalar(ElementTy::I32)),
            ],
            body,
            output: out,
        }
    }

    #[test]
    fn fold_constructs_with_right_type() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_inc_fold_fn(), None);
        let init = g.add_const(ConstValue::F64(0.0), None);
        let n = g.add_const(ConstValue::I32(60), None);
        let r = g.fold(l, init, n, None);
        // result type = accumulator type = scalar F64.
        assert_eq!(g.ty(r).rank, 0);
        assert_eq!(g.ty(r).element, ElementTy::F64);
        assert!(!g.has_errors(), "errors: {:?}", g.errors());
        match &g.node(r).op {
            Op::Fold {
                lambda,
                init: i,
                count: c,
            } => {
                assert_eq!(*lambda, l);
                assert_eq!(*i, init);
                assert_eq!(*c, n);
            }
            other => panic!("expected Fold, got {:?}", other),
        }
    }

    #[test]
    fn fold_rejects_non_lambda_first_arg() {
        let mut g = Graph::new();
        let init = g.add_const(ConstValue::F64(0.0), None);
        let n = g.add_const(ConstValue::I32(10), None);
        // pass init (a Const) where the lambda should be.
        let _ = g.fold(init, init, n, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("not a Lambda")),
            "expected not-a-Lambda error, got {:?}",
            summaries
        );
    }

    #[test]
    fn fold_rejects_init_type_mismatch() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_inc_fold_fn(), None);
        // init type is I32 but accumulator type is F64 — mismatch.
        let bad_init = g.add_const(ConstValue::I32(0), None);
        let n = g.add_const(ConstValue::I32(10), None);
        let _ = g.fold(l, bad_init, n, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("init type")),
            "expected init-type error, got {:?}",
            summaries
        );
    }

    #[test]
    fn fold_rejects_non_integer_count() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_inc_fold_fn(), None);
        let init = g.add_const(ConstValue::F64(0.0), None);
        // float count — not allowed; count must be integer.
        let bad_count = g.add_const(ConstValue::F64(60.0), None);
        let _ = g.fold(l, init, bad_count, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries
                .iter()
                .any(|s| s.contains("count must be rank-0 integer")),
            "expected count-type error, got {:?}",
            summaries
        );
    }

    #[test]
    fn fold_rejects_wrong_arity_lambda() {
        // build a one-param lambda (square); use it as a Fold body.
        let mut g = Graph::new();
        let l = g.add_lambda(build_square_fn(), None);
        let init = g.add_const(ConstValue::F64(0.0), None);
        let n = g.add_const(ConstValue::I32(10), None);
        let _ = g.fold(l, init, n, None);
        let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
        assert!(
            summaries.iter().any(|s| s.contains("exactly 2 params")),
            "expected arity error, got {:?}",
            summaries
        );
    }

    #[test]
    fn fold_is_hashconsed() {
        let mut g = Graph::new();
        let l = g.add_lambda(build_inc_fold_fn(), None);
        let init = g.add_const(ConstValue::F64(0.0), None);
        let n = g.add_const(ConstValue::I32(60), None);
        let r1 = g.fold(l, init, n, None);
        let r2 = g.fold(l, init, n, None);
        assert_eq!(
            r1, r2,
            "two folds with identical (lambda, init, count) collapse"
        );
    }

    #[test]
    fn hashcons_params_unaffected_by_cache() {
        // params dedup via the symbol-indexed path; the hash-cons
        // cache is a no-op for Op::Param. registering the same param
        // twice still returns the same id, params() still has one
        // entry per declared symbol.
        let mut g = Graph::new();
        let a1 = g.add_scalar_param("x", ElementTy::F64);
        let a2 = g.add_scalar_param("x", ElementTy::F64);
        assert_eq!(a1, a2);
        assert_eq!(g.params().len(), 1);
    }

    #[test]
    fn import_subgraph_grafts_pointwise_dag_remapping_leaves() {
        // a source graph computes `sqrt(a*a + b)` over its own leaves a, b. import it into a
        // destination that already holds the splice points (x, the const 7), remapping a -> x
        // and b -> 7, while an unmapped leaf is recreated fresh. structure + leaf binding
        // must be preserved; shared subterms collapse via the destination hash-cons.
        let mut src = Graph::new();
        let a = src.add_scalar_param("a", ElementTy::F64);
        let b = src.add_scalar_param("b", ElementTy::F64);
        let asq = src.element_wise(ElementWiseOp::Mul, vec![a, a], None);
        let sum = src.element_wise(ElementWiseOp::Add, vec![asq, b], None);
        let root = src.element_wise(ElementWiseOp::Sqrt, vec![sum], None);

        let mut dst = Graph::new();
        let x = dst.add_scalar_param("x", ElementTy::F64);
        let seven = dst.add_const(ConstValue::F64(7.0), None);
        let imported = dst.import_subgraph(&src, root, |sym| match sym.as_str() {
            "a" => Some(x),
            "b" => Some(seven),
            _ => None,
        });
        assert!(!dst.has_errors(), "import errors: {:?}", dst.errors());

        // root is Sqrt(Add(Mul(x, x), 7)) — the leaf `a` resolved to the dst param x.
        match &dst.node(imported).op {
            Op::ElementWise(ElementWiseOp::Sqrt, ins) => match &dst.node(ins[0]).op {
                Op::ElementWise(ElementWiseOp::Add, add_ins) => {
                    match &dst.node(add_ins[0]).op {
                        Op::ElementWise(ElementWiseOp::Mul, mul_ins) => {
                            assert_eq!(mul_ins[0], x, "leaf a should remap to dst param x");
                            assert_eq!(mul_ins[1], x, "the shared leaf collapses to one node");
                        }
                        other => panic!("expected Mul, got {other:?}"),
                    }
                    assert_eq!(add_ins[1], seven, "leaf b should remap to the dst const 7");
                }
                other => panic!("expected Add, got {other:?}"),
            },
            other => panic!("expected Sqrt root, got {other:?}"),
        }
        // no spurious params: only `x` was declared in dst (b mapped to a const, a to x).
        assert_eq!(dst.params().len(), 1, "no leaf should leak a new param");
    }

    #[test]
    fn import_subgraph_recreates_unmapped_leaf_as_param() {
        // a leaf the resolver does not know (returns None) is recreated as a fresh param of
        // the same type — the graceful default, not a panic.
        let mut src = Graph::new();
        let a = src.add_scalar_param("free", ElementTy::F64);
        let root = src.element_wise(ElementWiseOp::Neg, vec![a], None);
        let mut dst = Graph::new();
        let imported = dst.import_subgraph(&src, root, |_| None);
        assert!(!dst.has_errors());
        assert_eq!(
            dst.params().len(),
            1,
            "the unmapped leaf is recreated as a param"
        );
        assert_eq!(
            dst.param(&Symbol::intern("free")),
            Some(imported_child(&dst, imported))
        );
    }

    fn imported_child(g: &Graph, id: NodeId) -> NodeId {
        match &g.node(id).op {
            Op::ElementWise(_, ins) => ins[0],
            other => panic!("expected unary op, got {other:?}"),
        }
    }
}
