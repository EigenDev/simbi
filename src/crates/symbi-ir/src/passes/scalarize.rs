// =============================================================================
// lower.rs
//
// scalarization pass: tensor IR -> a "lowered" form (LoweredFn) that
// per-backend emitters (R.3.f CPU, R.3.g CUDA) turn into target source.
//
// the lowered form is a sequence of scalar let-statements followed by
// one or more output scalar expressions (one per scalar component of
// the output tensor). it is intentionally simpler than the existing
// scalar IR — no graph, no node IDs, just a list of bindings + a list
// of return values. this is enough for V1 literal-dim lowering.
//
// const-generic dim support (loops that survive into target source)
// is unimplemented; generic dims trigger a panic.
// =============================================================================

use std::collections::{HashMap, HashSet};

use crate::einsum::{Atom, EinsumSpec};
use crate::graph::{
    ConstValue, DimIndex, ElementWiseOp, Graph, NodeId, Op, ReduceOp, TranscendentalOp,
};
use crate::{DimExpr, ElementTy, Symbol, TensorTy};

// ----- the lowered form -----

/// operator-based binary form: rendered as `a OP b` in target source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // bitwise / logical (also works for bool in Rust):
    BitOr,
    BitAnd,
    BitXor,
}

impl BinaryKind {
    pub fn rust_operator(self) -> &'static str {
        match self {
            BinaryKind::Add => "+",
            BinaryKind::Sub => "-",
            BinaryKind::Mul => "*",
            BinaryKind::Div => "/",
            BinaryKind::Eq => "==",
            BinaryKind::Ne => "!=",
            BinaryKind::Lt => "<",
            BinaryKind::Le => "<=",
            BinaryKind::Gt => ">",
            BinaryKind::Ge => ">=",
            BinaryKind::BitOr => "|",
            BinaryKind::BitAnd => "&",
            BinaryKind::BitXor => "^",
        }
    }
}

/// operator-based unary form: rendered as `OP a` in target source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum UnaryKind {
    Neg,
    /// bitwise complement on int, logical NOT on bool. Rust's `!` operator
    /// covers both (CUDA's `!` is the bool form; `~` would be the int form,
    /// but the substrate uses this op only for Bool — Mask's `Not`).
    Not,
}

impl UnaryKind {
    pub fn rust_operator(self) -> &'static str {
        match self {
            UnaryKind::Neg => "-",
            UnaryKind::Not => "!",
        }
    }
}

/// a scalar expression in the lowered IR. emitters turn this into
/// per-backend source (Rust or CUDA C++).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ScalarExpr {
    /// rank-0 literal value.
    Const(ConstValue),
    /// reference to a named scalar (param or let-bound local).
    Var(String),
    /// binary operator: `a OP b`.
    BinOp(BinaryKind, Box<ScalarExpr>, Box<ScalarExpr>),
    /// unary operator: `OP a`.
    UnaryOp(UnaryKind, Box<ScalarExpr>),
    /// method call: `receiver.method(args...)`. used for `.abs()`,
    /// `.sqrt()`, `.min(b)`, `.sin()`, `.atan2(x)`, `.is_finite()`, etc.
    /// `method` is an owned `String` (not `&'static str`) so the lowered form
    /// is serde-deserializable — the serialized IR is the durable artifact
    /// (docs/design/15 §3); construction sites pass a literal `.to_string()`.
    MethodCall {
        receiver: Box<ScalarExpr>,
        method: String,
        args: Vec<ScalarExpr>,
    },
    /// ternary if/else: `if cond { then } else { else_ }`.
    Select {
        cond: Box<ScalarExpr>,
        then: Box<ScalarExpr>,
        else_: Box<ScalarExpr>,
    },
    /// indexed access into an array-shaped param: `container[index]`.
    /// used inside generic-dim loop bodies to read elements of a
    /// `[f64; D]`-shaped parameter. emitted as bracket-indexing in
    /// both Rust and CUDA source.
    IndexInto {
        container: String,
        index: Box<ScalarExpr>,
    },
    /// arbitrary-coord field load. `field_key` matches an IR-side field
    /// key (e.g., "prim_vel_0") that the dispatch / emit_kernel layer
    /// can resolve to a buffer index + layout. `components` is the
    /// runtime coord, one rank-0 scalar per axis (ndim entries).
    ///
    /// emitters: `emit_kernel.rs` walks the scalarized body once and
    /// rewrites every FieldLoadAt into a `Var("buf<idx>[<flat>]")` form
    /// for GPU; the chalkboard CPU body is the user's original Rust
    /// (not produced by tensor::emit_cpu) so `gather_at(...)` runtime
    /// calls survive as-is on CPU.
    FieldLoadAt {
        field_key: String,
        components: Vec<ScalarExpr>,
    },
    /// F1.B.8: free-function call by name with scalar args. emit lowers
    /// as `name(arg0, arg1, ...)` on both CPU and CUDA targets. the
    /// function definition lives outside this elemental — either a
    /// scalar elemental's `_cuda` accessor (for kernels that
    /// chain through F1.B.8's opaque-call substrate) or a host
    /// function on the CPU path.
    FreeCall { name: String, args: Vec<ScalarExpr> },
    /// numeric conversion `value as <to>` — the lowered form of
    /// `ElementWiseOp::Cast`, inserted by the graph's usual arithmetic conversions
    /// (e.g., an i32 index promoted to f64 to multiply a grid width). emits a backend
    /// cast: Rust `S::from_f64(<value> as f64)` (generic) / `(<value>) as f64`, CUDA
    /// `(<float-ty>)(<value>)`.
    Cast {
        to: ElementTy,
        value: Box<ScalarExpr>,
    },
}

impl ScalarExpr {
    /// the immediate sub-expressions this node owns, in evaluation order. THE single source for
    /// "which children do I have" (docs/design/38) — every WALK / TRANSFORM pass (cse var
    /// collect, free-call scan, var-use test, FieldLoadAt rewrite) recurses through this instead
    /// of re-matching all 10 variants inline. the EMIT backends (cpu/cuda/interp/jit) still match
    /// per-variant — producing target source/IR is the irreducible part — exactly the split the
    /// `ScalarStmt` SSOT (above) documents. adding a variant => update this once; the walks follow.
    pub fn children(&self) -> Vec<&ScalarExpr> {
        match self {
            ScalarExpr::Const(_) | ScalarExpr::Var(_) => Vec::new(),
            ScalarExpr::BinOp(_, a, b) => vec![a.as_ref(), b.as_ref()],
            ScalarExpr::UnaryOp(_, a) => vec![a.as_ref()],
            ScalarExpr::MethodCall { receiver, args, .. } => std::iter::once(receiver.as_ref())
                .chain(args.iter())
                .collect(),
            ScalarExpr::Select { cond, then, else_ } => {
                vec![cond.as_ref(), then.as_ref(), else_.as_ref()]
            }
            ScalarExpr::IndexInto { index, .. } => vec![index.as_ref()],
            ScalarExpr::FieldLoadAt { components, .. } => components.iter().collect(),
            ScalarExpr::FreeCall { args, .. } => args.iter().collect(),
            ScalarExpr::Cast { value, .. } => vec![value.as_ref()],
        }
    }

    /// mutable `children` for in-place transforms (the FieldLoadAt rewrite). same ownership map.
    pub fn children_mut(&mut self) -> Vec<&mut ScalarExpr> {
        match self {
            ScalarExpr::Const(_) | ScalarExpr::Var(_) => Vec::new(),
            ScalarExpr::BinOp(_, a, b) => vec![a.as_mut(), b.as_mut()],
            ScalarExpr::UnaryOp(_, a) => vec![a.as_mut()],
            ScalarExpr::MethodCall { receiver, args, .. } => std::iter::once(receiver.as_mut())
                .chain(args.iter_mut())
                .collect(),
            ScalarExpr::Select { cond, then, else_ } => {
                vec![cond.as_mut(), then.as_mut(), else_.as_mut()]
            }
            ScalarExpr::IndexInto { index, .. } => vec![index.as_mut()],
            ScalarExpr::FieldLoadAt { components, .. } => components.iter_mut().collect(),
            ScalarExpr::FreeCall { args, .. } => args.iter_mut().collect(),
            ScalarExpr::Cast { value, .. } => vec![value.as_mut()],
        }
    }
}

/// one statement in the lowered IR body. covers immutable lets,
/// mutable accumulators (for reductions), compound assignment
/// (`acc += value`), and const-generic for-loops.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ScalarStmt {
    /// `let name: element = value;`
    Let {
        name: String,
        element: ElementTy,
        value: ScalarExpr,
    },
    /// `let mut name: element = init;` — used for loop-form reduction
    /// accumulators (zero-init, then CompoundAssign in the loop body).
    LetMut {
        name: String,
        element: ElementTy,
        init: ScalarExpr,
    },
    /// `name <op>= value;` — applied to a previously-declared LetMut
    /// inside a loop body. op is restricted to the four "reduction
    /// associative" forms: Add, Mul, BitOr, BitAnd, BitXor.
    CompoundAssign {
        name: String,
        op: BinaryKind,
        value: ScalarExpr,
    },
    /// F2.F: `name = value;` — plain (non-compound) assignment.
    /// applied to a previously-declared `LetMut`. used by Op::Fold's
    /// lowering, where the body lambda returns a new accumulator that
    /// REPLACES (not accumulates into) the current one. CompoundAssign
    /// only covers reductive updates; Fold needs the general case.
    Assign { name: String, value: ScalarExpr },
    /// `for iter in 0..bound { body }` — const-generic loop. bound is
    /// either Literal (rarely used; existing code unrolls these) or
    /// Generic (the const-generic identifier from the surrounding
    /// context, e.g., "D"). emitters render as `for ii in 0..D` (Rust)
    /// or `#pragma unroll\nfor (int ii = 0; ii < D; ++ii)` (CUDA).
    For {
        iter: String,
        bound: DimExpr,
        body: Vec<ScalarStmt>,
    },
    /// `if cond { then_body }` — emitted by `IterateInline` lowering to host
    /// the optional early-`break` after the step assigns. emitters render as
    /// the same `if` syntax in both Rust and CUDA.
    If {
        cond: ScalarExpr,
        then_body: Vec<ScalarStmt>,
    },
    /// `break;` — only valid inside a `For` body. emitters render as
    /// `break;` (Rust + CUDA).
    Break,
    /// `let <name>: <element> = { <body>; <result> };` — an explicit
    /// **bounded-pressure scope**. all `Let`s introduced inside `body` are
    /// dead at the closing brace; only `result` survives, bound to `name` in
    /// the enclosing scope.
    ///
    /// docs/design/23: this is the IR primitive that lets the renderer
    /// communicate phase boundaries to nvcc / rustc. without it, the CSE
    /// pass hoists every shared subexpression to function scope, producing
    /// the 239-`__cse_N` flat blocks that blow `wave_speed_map`'s
    /// register pressure to 154/thread. scopes give the codegen lifetime
    /// information it can actually allocate against.
    ///
    /// the renderer translates this to:
    ///   - **Rust**: `let <name>: <element> = { <body>; <result> };`
    ///     (Rust block expressions are first-class; the body executes and
    ///     the final expression is the block's value.)
    ///   - **CUDA**: `<ty> <name>; { <body>; <name> = <result>; }`
    ///     (C/C++ doesn't have block expressions; the declaration is in the outer
    ///     scope, the write inside the inner, and the inner-scope locals die
    ///     at the closing brace.)
    ///
    /// the CSE pass treats the scope as a **hoisting barrier**
    /// (docs/design/23): CSE candidates whose uses are all inside the
    /// scope stay inside; candidates whose uses cross the boundary get
    /// hoisted to the LCA scope of all use sites.
    ///
    /// the CSE pass passes scopes through unchanged (treats them as ordinary
    /// statement containers); LCA-aware placement is not applied.
    Scope {
        name: String,
        element: ElementTy,
        body: Vec<ScalarStmt>,
        result: ScalarExpr,
    },
    /// the DUAL of the `For`/`Break` iterate lowering: a real data-dependent
    /// branch where ONLY the taken arm executes. lowered from `Op::IfElse`
    /// (`S::cond` -> 1 output, `S::cond_vec` -> N outputs). `outs` declares the
    /// N result slots in the enclosing scope; each arm body ENDS with one
    /// `Assign { outs[j].0, <arm result j> }` per slot, so the variant carries
    /// one immediate expr (`cond`) plus two sub-bodies — fitting the SSOT walk
    /// model. the renderer translates this to:
    ///   - **Rust**: `let o0: e0; ... let o{N-1}: e{N-1}; if cond { then_body }
    ///     else { else_body }` (definite-assignment: both arms assign every out).
    ///   - **CUDA**: `e0 o0; ...; if (cond) { then_body } else { else_body }`.
    /// arm-internal lets die at their brace — the lifetime/branch info the
    /// codegen needs to evaluate only the taken arm (the compute-all-paths fix).
    IfElse {
        outs: Vec<(String, ElementTy)>,
        cond: ScalarExpr,
        then_body: Vec<ScalarStmt>,
        else_body: Vec<ScalarStmt>,
    },
}

// =============================================================================
// the SINGLE SOURCE OF TRUTH for ScalarStmt structural walks.
//
// every transformation pass (cse, FieldLoadAt rewrite, uses-var detection, the
// fresh-name index scan) walks scalar statements the same way: visit the
// immediate scalar expression a stmt CARRIES, then recurse into any child
// statement bodies. those two notions are encoded ONCE here rather than
// respelled inline in every backend match.
//
// the four helpers below + `with_child_expr` are the one place that encodes
// "which exprs belong to me" and "which sub-bodies do I own". every walk-style
// pass derives from them. emit backends (cpu/cuda/interp) still match
// per-variant — that's the irreducible part of producing source. but the
// WALK / TRANSFORM passes are now one-liners.
//
// adding a new ScalarStmt variant becomes: update the enum (above), then add
// arms here (one per accessor) for the variant's exprs + bodies. the walk
// passes pick it up for free.
// =============================================================================
impl ScalarStmt {
    /// the scalar expression this statement HOLDS directly (not nested inside a
    /// child body). every variant has at most one — Let's `value`, LetMut's
    /// `init`, Assign / CompoundAssign's `value`, If's `cond`. For / Break have
    /// no immediate expression (For's bound is a `DimExpr`, not a ScalarExpr).
    pub fn child_expr(&self) -> Option<&ScalarExpr> {
        match self {
            ScalarStmt::Let { value, .. } => Some(value),
            ScalarStmt::LetMut { init, .. } => Some(init),
            ScalarStmt::Assign { value, .. } => Some(value),
            ScalarStmt::CompoundAssign { value, .. } => Some(value),
            ScalarStmt::If { cond, .. } => Some(cond),
            // Scope's IMMEDIATE child expression is its `result` — the value
            // bound to `name` in the enclosing scope. body is a separate
            // sub-list (see `child_stmts`).
            ScalarStmt::Scope { result, .. } => Some(result),
            // IfElse's immediate expr is the `cond`; the arm results live as
            // trailing Assigns inside the sub-bodies (walked via child_stmts).
            ScalarStmt::IfElse { cond, .. } => Some(cond),
            ScalarStmt::For { .. } | ScalarStmt::Break => None,
        }
    }

    /// mutable variant of `child_expr` for in-place transformations (the
    /// FieldLoadAt rewrite, for example).
    pub fn child_expr_mut(&mut self) -> Option<&mut ScalarExpr> {
        match self {
            ScalarStmt::Let { value, .. } => Some(value),
            ScalarStmt::LetMut { init, .. } => Some(init),
            ScalarStmt::Assign { value, .. } => Some(value),
            ScalarStmt::CompoundAssign { value, .. } => Some(value),
            ScalarStmt::If { cond, .. } => Some(cond),
            ScalarStmt::Scope { result, .. } => Some(result),
            ScalarStmt::IfElse { cond, .. } => Some(cond),
            ScalarStmt::For { .. } | ScalarStmt::Break => None,
        }
    }

    /// the child statement bodies this variant owns, as a list of sub-bodies.
    /// `For`/`If`/`Scope` own ONE body; `IfElse` owns TWO (then + else); every
    /// other variant owns none. walks that recurse use this to descend
    /// uniformly without naming variants — and it handles multi-body variants
    /// (IfElse) that a single `&[ScalarStmt]` return could not express.
    pub fn child_stmt_bodies(&self) -> Vec<&[ScalarStmt]> {
        match self {
            ScalarStmt::For { body, .. } => vec![body],
            ScalarStmt::If { then_body, .. } => vec![then_body],
            ScalarStmt::Scope { body, .. } => vec![body],
            ScalarStmt::IfElse {
                then_body,
                else_body,
                ..
            } => vec![then_body, else_body],
            _ => Vec::new(),
        }
    }

    /// mutable variant of `child_stmt_bodies`. the two `IfElse` arms are
    /// disjoint struct fields, so borrowing both mutably at once is sound.
    pub fn child_stmt_bodies_mut(&mut self) -> Vec<&mut [ScalarStmt]> {
        match self {
            ScalarStmt::For { body, .. } => vec![body.as_mut_slice()],
            ScalarStmt::If { then_body, .. } => vec![then_body.as_mut_slice()],
            ScalarStmt::Scope { body, .. } => vec![body.as_mut_slice()],
            ScalarStmt::IfElse {
                then_body,
                else_body,
                ..
            } => vec![then_body.as_mut_slice(), else_body.as_mut_slice()],
            _ => Vec::new(),
        }
    }

    /// the binding name this statement INTRODUCES, if any. Let / LetMut /
    /// Scope declare new locals; other variants don't. used by the CSE
    /// fresh-name scan to pick the next available temp index without
    /// colliding with existing lets.
    pub fn binding_name(&self) -> Option<&str> {
        match self {
            ScalarStmt::Let { name, .. }
            | ScalarStmt::LetMut { name, .. }
            | ScalarStmt::Scope { name, .. } => Some(name.as_str()),
            // IfElse declares N result slots; surface the first (binding_name
            // feeds only the `__cse` prefix scan, and `__br` uses a different
            // prefix, so one representative name is sufficient).
            ScalarStmt::IfElse { outs, .. } => outs.first().map(|(n, _)| n.as_str()),
            _ => None,
        }
    }

    /// rebuild this statement with its child expression transformed. Variants
    /// without a child expression (For, Break) pass through unchanged. used by
    /// the CSE rewrite pass to thread expression-rewrites through statements
    /// without dwelling on the variant shape.
    ///
    /// for `Scope`, the transformed expression is the `result` — body
    /// statements are left untouched (they're walked separately via
    /// `child_stmts`).
    pub fn with_child_expr(self, f: impl FnOnce(ScalarExpr) -> ScalarExpr) -> Self {
        match self {
            ScalarStmt::Let {
                name,
                element,
                value,
            } => ScalarStmt::Let {
                name,
                element,
                value: f(value),
            },
            ScalarStmt::LetMut {
                name,
                element,
                init,
            } => ScalarStmt::LetMut {
                name,
                element,
                init: f(init),
            },
            ScalarStmt::Assign { name, value } => ScalarStmt::Assign {
                name,
                value: f(value),
            },
            ScalarStmt::CompoundAssign { name, op, value } => ScalarStmt::CompoundAssign {
                name,
                op,
                value: f(value),
            },
            ScalarStmt::If { cond, then_body } => ScalarStmt::If {
                cond: f(cond),
                then_body,
            },
            ScalarStmt::Scope {
                name,
                element,
                body,
                result,
            } => ScalarStmt::Scope {
                name,
                element,
                body,
                result: f(result),
            },
            ScalarStmt::IfElse {
                outs,
                cond,
                then_body,
                else_body,
            } => ScalarStmt::IfElse {
                outs,
                cond: f(cond),
                then_body,
                else_body,
            },
            other => other,
        }
    }
}

/// one scalar parameter of the lowered function. when `array_len` is
/// Some, the param is a rank-1 array (`[element; len]`); otherwise
/// it's a plain scalar. multi-rank generic params are not supported
/// in V1 (rank-1 only).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoweredParam {
    pub name: String,
    pub element: ElementTy,
    pub array_len: Option<DimExpr>,
}

impl LoweredParam {
    /// short-hand: a scalar param (no array_len).
    pub fn scalar(name: String, element: ElementTy) -> Self {
        Self {
            name,
            element,
            array_len: None,
        }
    }

    /// short-hand: a rank-1 array param.
    pub fn array(name: String, element: ElementTy, len: DimExpr) -> Self {
        Self {
            name,
            element,
            array_len: Some(len),
        }
    }
}

/// a fully-lowered function: scalar params + let-statement body + one
/// or more output expressions (one per scalar component of the output
/// tensor). emitters turn this into target source.
#[derive(Clone, Debug)]
pub struct LoweredFn {
    pub name: String,
    pub params: Vec<LoweredParam>,
    pub body: Vec<ScalarStmt>,
    pub results: Vec<ScalarExpr>,
    pub result_element: ElementTy,
    pub result_shape: Vec<DimExpr>,
}

// ----- the scalarizer -----

/// per-node binding produced by scalarization. either a fully-expanded
/// list of scalar component expressions (the literal-dim path) or a
/// reference to a rank-1 array param indexed at runtime (the
/// const-generic dim path, V1 scope).
#[derive(Clone, Debug)]
enum Binding {
    /// N scalar expressions in row-major order. used for literal-dim
    /// nodes and rank-0 results.
    Concrete(Vec<ScalarExpr>),
    /// rank-1 array accessible as `container[ii]`. only produced by
    /// generic-dim param lowering in V1.
    Array { container: String },
}

impl Binding {
    /// reconstruct one component expression using an index-expression
    /// (typically a `Var(iter_name)` from inside a generated For loop).
    fn get_at(&self, idx: ScalarExpr) -> ScalarExpr {
        match self {
            Binding::Concrete(v) => match &idx {
                ScalarExpr::Const(ConstValue::U32(k)) => v[*k as usize].clone(),
                _ => panic!(
                    "Binding::get_at on Concrete requires a literal U32 index; got {:?}",
                    idx
                ),
            },
            Binding::Array { container } => ScalarExpr::IndexInto {
                container: container.clone(),
                index: Box::new(idx),
            },
        }
    }
}

/// walks a tensor IR graph in topological (insertion) order and
/// produces a LoweredFn by per-op lowering rules.
struct Scalarizer {
    bindings: HashMap<NodeId, Binding>,
    params: Vec<LoweredParam>,
    body: Vec<ScalarStmt>,
    /// counter for generated temp names (loop iters, accumulators).
    next_temp: usize,
    /// the current `IterateInline` accumulator local names (one per vector
    /// component), set while lowering a loop's `steps` cone so `Op::IterAcc(j)`
    /// resolves to `iter_acc[j]` (docs/design/14).
    iter_acc: Option<Vec<String>>,
}

impl Scalarizer {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            params: Vec::new(),
            body: Vec::new(),
            next_temp: 0,
            iter_acc: None,
        }
    }

    /// F1.B.10b: compute in-degree per NodeId by walking the graph
    /// once. used by `maybe_hoist_to_let` to share ScalarExpr trees
    /// across consumers — without this, each downstream reference
    /// CLONES the entire binding sub-tree, producing exponential
    /// scalarized text size for graphs with shared sub-nodes (RMHD
    /// wave-speed kernels were ~100s in scalarize before this fix).
    fn compute_in_degrees(graph: &Graph) -> HashMap<NodeId, u32> {
        let mut deg: HashMap<NodeId, u32> = HashMap::new();
        for (_, node, _) in graph.iter() {
            for inp in op_inputs(&node.op) {
                *deg.entry(inp).or_insert(0) += 1;
            }
        }
        deg
    }

    /// after lowering a node, if it produced a rank-0 scalar binding
    /// that's non-trivial AND has in-degree >= 2, hoist the expression
    /// to a Let with a fresh `__cse_<n>` name and replace the binding
    /// with `Var(name)`. subsequent consumers then clone a cheap Var
    /// instead of the full sub-tree.
    fn maybe_hoist_to_let(&mut self, id: NodeId, in_degrees: &HashMap<NodeId, u32>, ty: &TensorTy) {
        if in_degrees.get(&id).copied().unwrap_or(0) < 2 {
            return;
        }
        if ty.rank != 0 {
            // rank > 0 hoisting needs per-component temp names; skipping
            // here is sound — the per-component bindings are shared
            // through the Binding::Concrete vec inside HashMap, and
            // consumers that pull individual components via Index also
            // get hoisted on this code path when they themselves have
            // in-degree >= 2.
            return;
        }
        let binding = match self.bindings.get(&id) {
            Some(Binding::Concrete(v)) if v.len() == 1 => v.clone(),
            _ => return,
        };
        let expr = &binding[0];
        // trivial leaves (Var, Const) don't benefit from a temp — they
        // already have stable identity.
        if matches!(expr, ScalarExpr::Var(_) | ScalarExpr::Const(_)) {
            return;
        }
        let temp = self.fresh("cse");
        self.body.push(ScalarStmt::Let {
            name: temp.clone(),
            element: ty.element,
            value: expr.clone(),
        });
        self.bindings
            .insert(id, Binding::Concrete(vec![ScalarExpr::Var(temp)]));
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let n = self.next_temp;
        self.next_temp += 1;
        format!("__{}_{}", prefix, n)
    }

    /// concrete-shape requirement: panic with a clear message if this
    /// op's input is not Concrete (i.e., is an Array — the V1 generic
    /// path supports einsum-with-rank-0 only).
    fn require_concrete<'a>(&'a self, id: NodeId, op_name: &str) -> &'a [ScalarExpr] {
        match self.bindings.get(&id) {
            Some(Binding::Concrete(v)) => v,
            Some(Binding::Array { .. }) => panic!(
                "scalarization: op {} does not yet support const-generic dim inputs (V1 R.5.a \
                 covers einsum only; other ops need follow-up phases)",
                op_name
            ),
            None => panic!("scalarization: missing binding for input node {:?}", id),
        }
    }

    fn lower_node(&mut self, id: NodeId, op: &Op, ty: &TensorTy, graph: &Graph) {
        let binding = match op {
            Op::Const(v) => Binding::Concrete(self.lower_const(v.clone())),
            Op::Param(sym) => self.lower_param(sym, ty),
            Op::ElementWise(ewop, inputs) => {
                Binding::Concrete(self.lower_element_wise(*ewop, inputs, ty, graph))
            }
            Op::Transcendental(trop, inputs) => {
                Binding::Concrete(self.lower_transcendental(*trop, inputs, ty, graph))
            }
            Op::Construct(inputs) => Binding::Concrete(self.lower_construct(inputs)),
            Op::Index(tensor, idxs) => Binding::Concrete(self.lower_index(*tensor, idxs, graph)),
            Op::Broadcast(tensor, target_shape) => {
                Binding::Concrete(self.lower_broadcast(*tensor, target_shape, graph))
            }
            Op::Reduce(rop, axes, input) => {
                Binding::Concrete(self.lower_reduce(*rop, axes, *input, ty, graph))
            }
            Op::Select(cond, then_id, else_id) => {
                Binding::Concrete(self.lower_select(*cond, *then_id, *else_id, ty, graph))
            }
            Op::Einsum(spec, inputs) => self.lower_einsum(spec, inputs, ty, graph),
            Op::LoadAt(sym, comps) => Binding::Concrete(self.lower_load_at(sym, comps)),
            // F2.C: Lambda is a callable value, not a tensor. it's only
            // consumed by Op::Apply, which extracts the FnDef name and
            // emits a FreeCall referencing the device function. lower
            // Lambda to a placeholder zero so any accidental reads
            // surface as obvious zeros rather than panics.
            Op::Lambda(_) => {
                Binding::Concrete(vec![ScalarExpr::Const(crate::ConstValue::F64(0.0))])
            }
            // F2.C: Apply lowers to a `FreeCall(name, args)` on the scalar
            // side — resolve the device-function name via `graph.fn_def`.
            Op::Apply { lambda, args } => {
                let fn_name = graph.fn_def(*lambda).name.clone();
                Binding::Concrete(self.lower_opaque_call(&fn_name, args))
            }
            // F2.F: Fold lowers to a `LetMut acc = init; For i in 0..count
            // { acc = body(acc, i); }; acc`. accumulator is rank-0 for V1
            // (rank > 0 fold needs per-component LetMut + Assign — punted
            // until a real call site demands it). produces a Binding to
            // the accumulator's `Var(name)`.
            Op::Fold {
                lambda,
                init,
                count,
            } => Binding::Concrete(self.lower_fold(*lambda, *init, *count, ty, graph)),
            // F4.0c: Morphism lowering. each kind expands to a small
            // arithmetic expression over the two input args (the
            // proc-macro synthesizes `field[coord]` as args[0] and
            // `field[coord + e_axis]` as args[1] at the call site).
            Op::Morphism { kind, args } => Binding::Concrete(self.lower_morphism(*kind, args, ty)),
            // docs/design/14: the loop accumulator placeholder resolves to the
            // mutable local set up by `lower_iterate_inline` (only while lowering
            // a loop's `step` cone).
            Op::IterAcc(idx) => Binding::Concrete(vec![ScalarExpr::Var(
                self.iter_acc
                    .as_ref()
                    .expect("IterAcc lowered outside an IterateInline loop body")
                    [*idx as usize]
                    .clone(),
            )]),
            // extract one component of a multi-output node (an IfElse from
            // cond_vec, bound to an N-component Concrete). the source is lowered
            // first (smaller NodeId), so its binding is available.
            Op::Proj { source, index } => Binding::Concrete(vec![
                self.require_concrete(*source, "Op::Proj source")[*index as usize].clone(),
            ]),
            // IterateInline is cone-partitioned: scalarize_kernel handles it
            // directly (skips the cone in the main pass, lowers it inside the
            // `for`), so it never reaches normal per-node dispatch.
            Op::IterateInline { .. } => {
                unreachable!("IterateInline is lowered in scalarize_kernel, not lower_node",)
            }
            // docs/design/23: Op::Scope is body-partitioned like
            // IterateInline. scalarize_kernel collects scope-owned NodeIds,
            // skips them in the main pass, then lowers each Scope via
            // `lower_scope` which emits a ScalarStmt::Scope wrapping the
            // body Lets in braces.
            Op::Scope { .. } => {
                unreachable!("Op::Scope is lowered in scalarize_kernel, not lower_node",)
            }
            // the IfElse dual is arm-body-partitioned like Op::Scope —
            // scalarize_kernel collects each arm's owned NodeIds, skips them in
            // the main pass, and lowers via `lower_if_else`.
            Op::IfElse { .. } => {
                unreachable!("Op::IfElse is lowered in scalarize_kernel, not lower_node",)
            }
        };
        self.bindings.insert(id, binding);
    }

    /// F2.F: lower an Op::Fold into a `LetMut + For + Assign` sequence.
    /// returns a single-element binding holding `Var(acc_name)` so
    /// downstream consumers see the post-loop accumulator value.
    fn lower_fold(
        &mut self,
        lambda: NodeId,
        init: NodeId,
        count: NodeId,
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        // V1 scope: rank-0 accumulator.
        assert_eq!(
            out_ty.rank, 0,
            "lower_fold V1: rank-0 accumulator only (got rank {})",
            out_ty.rank
        );
        let acc_name = self.fresh("fold_acc");
        let iter_name = self.fresh("fold_i");
        let acc_elem = out_ty.element;

        let init_expr = self.require_concrete(init, "Fold init")[0].clone();
        let count_expr = self.require_concrete(count, "Fold count")[0].clone();

        // resolve the body lambda's FnDef.name for the FreeCall.
        let fn_name = graph.fn_def(lambda).name.as_str().to_string();
        let acc_var = ScalarExpr::Var(acc_name.clone());
        let idx_var = ScalarExpr::Var(iter_name.clone());
        let step_expr = ScalarExpr::FreeCall {
            name: fn_name,
            args: vec![acc_var.clone(), idx_var],
        };

        // top-level: LetMut + For containing one Assign.
        self.body.push(ScalarStmt::LetMut {
            name: acc_name.clone(),
            element: acc_elem,
            init: init_expr,
        });
        let loop_bound = match count_expr {
            ScalarExpr::Const(ConstValue::I32(n)) => DimExpr::Literal(n as usize),
            ScalarExpr::Const(ConstValue::U32(n)) => DimExpr::Literal(n as usize),
            // for non-literal counts the existing ScalarStmt::For's
            // DimExpr can't carry a runtime expression — that's an
            // emit-layer extension. literal counts required.
            other => panic!(
                "lower_fold V1: count must be a literal integer Const; got {:?}",
                other
            ),
        };
        self.body.push(ScalarStmt::For {
            iter: iter_name,
            bound: loop_bound,
            body: vec![ScalarStmt::Assign {
                name: acc_name.clone(),
                value: step_expr,
            }],
        });
        vec![acc_var]
    }

    /// docs/design/14: lower an `Op::IterateInline` over an N-component accumulator
    /// vector to `LetMut acc_j = inits[j]; For i { <union cone> acc_j = steps[j] }`.
    /// the `cone` is the acc-dependent slice of all `steps` (id/topo order); its
    /// loop-INVARIANT inputs are already lowered before the loop. the N assigns
    /// come AFTER the whole cone, so the update is SIMULTANEOUS (Jacobi): every
    /// `steps[j]` reads the OLD `acc_*`. binds the node to `Var(acc_result)`.
    /// docs/design/23: lower an `Op::Scope` node into a
    /// `ScalarStmt::Scope` brace-block. body NodeIds are lowered IN ORDER into
    /// a fresh sub-body that becomes the scope's inner statements; `result`'s
    /// expression becomes the scope's trailing tail value.
    ///
    /// the technique mirrors `lower_iterate_inline`: mark current
    /// `self.body.len()`, lower body NodeIds (their Lets get pushed onto
    /// `self.body`), then `split_off(mark)` captures just those Lets as the
    /// inner body. the result NodeId's binding is already populated at that
    /// point — `require_concrete` reads it cleanly.
    ///
    /// the scope's NodeId itself binds to `ScalarExpr::Var("__scope_<n>")` in
    /// the OUTER body, so downstream consumers reference the scope's
    /// observable value through that name.
    fn lower_scope(
        &mut self,
        scope_id: NodeId,
        body: &[NodeId],
        result: NodeId,
        result_ty: &TensorTy,
        scope_owner: &HashMap<NodeId, NodeId>,
        in_degrees: &HashMap<NodeId, u32>,
        graph: &Graph,
    ) {
        // capture the position where the scope's inner Lets will begin.
        let mark = self.body.len();
        // lower body NodeIds IN ORDER. for each:
        //   - if it belongs to a DEEPER scope (scope_owner maps it to !=
        //     scope_id), skip — it is lowered on recursion into that
        //     nested scope.
        //   - if it IS a nested `Op::Scope`, dispatch to `lower_scope`
        //     recursively so its body lands inside ITS brace block.
        //   - otherwise, lower normally and the resulting Let lands in the
        //     inner body (it'll be captured by `split_off` below).
        for &bid in body {
            // skip NodeIds owned by a deeper scope. they were added to the
            // body list because they were created during the closure's
            // execution, but they're actually scope-local to a nested
            // `Op::Scope` and will be lowered there.
            if scope_owner.get(&bid) != Some(&scope_id) {
                continue;
            }
            let bnode = graph.node(bid);
            let bty = graph.ty(bid).clone();
            // dispatch nested Op::Scope BEFORE lower_node — lower_node
            // panics on Op::Scope (the unreachable arm; scopes go through
            // this specialized path).
            if let Op::Scope {
                body: inner_body,
                result: inner_result,
            } = &bnode.op
            {
                self.lower_scope(
                    bid,
                    inner_body,
                    *inner_result,
                    &bty,
                    scope_owner,
                    in_degrees,
                    graph,
                );
                continue;
            }
            let bop = bnode.op.clone();
            self.lower_node(bid, &bop, &bty, graph);
            self.maybe_hoist_to_let(bid, in_degrees, &bty);
        }
        // resolve the result's expression NOW — bindings are populated and
        // this expression is evaluated AS THE SCOPE'S TAIL VALUE
        // (i.e., inside the brace block, after all the inner Lets, before the
        // closing brace).
        let result_expr = self.require_concrete(result, "Op::Scope result")[0].clone();
        // peel off the inner body's Lets — these are the scope-local temps.
        let scope_body = self.body.split_off(mark);
        // mint a fresh outer name for the scope's observable value.
        let scope_name = self.fresh("__scope");
        // emit the ScalarStmt::Scope into the OUTER body.
        self.body.push(ScalarStmt::Scope {
            name: scope_name.clone(),
            element: result_ty.element,
            body: scope_body,
            result: result_expr,
        });
        // bind the Op::Scope NodeId to the outer scope-name var, so any
        // downstream consumer that references the scope sees `Var(scope_name)`.
        self.bindings.insert(
            scope_id,
            Binding::Concrete(vec![ScalarExpr::Var(scope_name)]),
        );
    }

    /// lower an `Op::IfElse` (the DUAL of `lower_iterate_inline`): N outer-
    /// declared result slots plus two arm sub-bodies, each ending with one
    /// `Assign { outs[j], <arm result j> }` per slot. only the TAKEN arm runs at
    /// render time. N=1 is scalar `S::cond`; N>1 is `S::cond_vec` (the IfElse
    /// node binds to an N-component `Concrete`, consumed via `Op::Proj`). the
    /// `cond` is computed OUTSIDE the branch, so it is already lowered.
    #[allow(clippy::too_many_arguments)]
    fn lower_if_else(
        &mut self,
        ifelse_id: NodeId,
        cond: NodeId,
        then_body: &[NodeId],
        then_results: &[NodeId],
        else_body: &[NodeId],
        else_results: &[NodeId],
        graph: &Graph,
        branch_owner: &HashMap<NodeId, (NodeId, bool)>,
        in_degrees: &HashMap<NodeId, u32>,
    ) {
        let cond_expr = self.require_concrete(cond, "Op::IfElse cond")[0].clone();
        // one declared slot per output component, typed from each result node.
        let outs: Vec<(String, ElementTy)> = then_results
            .iter()
            .map(|&r| (self.fresh("__br"), graph.ty(r).element))
            .collect();
        let names: Vec<String> = outs.iter().map(|(n, _)| n.clone()).collect();
        let then_stmts = self.lower_branch_arm(
            ifelse_id,
            true,
            then_body,
            then_results,
            &names,
            branch_owner,
            in_degrees,
            graph,
        );
        let else_stmts = self.lower_branch_arm(
            ifelse_id,
            false,
            else_body,
            else_results,
            &names,
            branch_owner,
            in_degrees,
            graph,
        );
        self.body.push(ScalarStmt::IfElse {
            outs: outs,
            cond: cond_expr,
            then_body: then_stmts,
            else_body: else_stmts,
        });
        // the IfElse node binds to its N components; scalar consumers take [0],
        // `Op::Proj` selects component j.
        self.bindings.insert(
            ifelse_id,
            Binding::Concrete(names.into_iter().map(ScalarExpr::Var).collect()),
        );
    }

    /// lower ONE arm of an `Op::IfElse`. mirrors the cone body of
    /// `lower_iterate_inline` / `lower_scope`: mark the body length, lower the
    /// nodes THIS arm owns (skipping nodes evicted to the outer level or claimed
    /// by a nested region — those are lowered elsewhere), `split_off` the arm's
    /// Lets, and append one trailing `names[j] = <result j>` assign per output.
    #[allow(clippy::too_many_arguments)]
    fn lower_branch_arm(
        &mut self,
        ifelse_id: NodeId,
        is_then: bool,
        body: &[NodeId],
        results: &[NodeId],
        names: &[String],
        branch_owner: &HashMap<NodeId, (NodeId, bool)>,
        in_degrees: &HashMap<NodeId, u32>,
        graph: &Graph,
    ) -> Vec<ScalarStmt> {
        let mark = self.body.len();
        for &bid in body {
            // lower ONLY the nodes this arm owns — evicted (outer-shared) nodes
            // and nodes claimed by a NESTED region are lowered in their own
            // place; their bindings are visible here as `Var(...)`.
            if branch_owner.get(&bid) != Some(&(ifelse_id, is_then)) {
                continue;
            }
            let bnode = graph.node(bid);
            let bty = graph.ty(bid).clone();
            // dispatch a nested IfElse recursively (lower_node panics on it).
            if let Op::IfElse {
                cond,
                then_body,
                then_results,
                else_body,
                else_results,
            } = &bnode.op
            {
                self.lower_if_else(
                    bid,
                    *cond,
                    then_body,
                    then_results,
                    else_body,
                    else_results,
                    graph,
                    branch_owner,
                    in_degrees,
                );
                continue;
            }
            let bop = bnode.op.clone();
            self.lower_node(bid, &bop, &bty, graph);
            self.maybe_hoist_to_let(bid, in_degrees, &bty);
        }
        // capture result exprs BEFORE split_off (bindings still resolve here).
        let result_exprs: Vec<ScalarExpr> = results
            .iter()
            .map(|&r| self.require_concrete(r, "Op::IfElse arm result")[0].clone())
            .collect();
        let mut arm = self.body.split_off(mark);
        for (name, value) in names.iter().zip(result_exprs) {
            arm.push(ScalarStmt::Assign {
                name: name.clone(),
                value,
            });
        }
        arm
    }

    fn lower_iterate_inline(
        &mut self,
        iter_id: NodeId,
        inits: &[NodeId],
        steps: &[NodeId],
        count: usize,
        result: usize,
        break_when: Option<NodeId>,
        cone: &[NodeId],
        in_degrees: &HashMap<NodeId, u32>,
        graph: &Graph,
    ) {
        let iter_name = self.fresh("iter_i");
        // the mutable accumulator locals (one per component), BEFORE the loop.
        let acc_names: Vec<String> = (0..inits.len()).map(|_| self.fresh("iter_acc")).collect();
        for (j, &init) in inits.iter().enumerate() {
            let init_expr = self.require_concrete(init, "IterateInline init")[0].clone();
            self.body.push(ScalarStmt::LetMut {
                name: acc_names[j].clone(),
                element: ElementTy::F64,
                init: init_expr,
            });
        }
        // lower the union cone INSIDE the loop (split_off captures its lets);
        // `IterAcc(j)` resolves to `acc_names[j]` via `self.iter_acc`.
        self.iter_acc = Some(acc_names.clone());
        let mark = self.body.len();
        for &cid in cone {
            let cop = graph.node(cid).op.clone();
            let cty = graph.ty(cid).clone();
            self.lower_node(cid, &cop, &cty, graph);
            self.maybe_hoist_to_let(cid, in_degrees, &cty);
        }
        // capture all step expressions while IterAcc still resolves.
        let step_exprs: Vec<ScalarExpr> = steps
            .iter()
            .map(|&s| self.require_concrete(s, "IterateInline step")[0].clone())
            .collect();
        // resolve the break predicate, if any. it's part of the cone (above) so
        // its `Let`s are already in `loop_body`; its binding is read here.
        let break_expr: Option<ScalarExpr> =
            break_when.map(|bw| self.require_concrete(bw, "IterateInline break_when")[0].clone());
        self.iter_acc = None;
        let n = step_exprs.len();
        // temp names for the SIMULTANEOUS (Jacobi) update (N>1 only).
        let tmp_names: Vec<String> = if n > 1 {
            (0..n).map(|_| self.fresh("iter_next")).collect()
        } else {
            Vec::new()
        };
        let mut loop_body = self.body.split_off(mark);
        if n == 1 {
            // scalar: one accumulator, no aliasing — assign directly.
            loop_body.push(ScalarStmt::Assign {
                name: acc_names[0].clone(),
                value: step_exprs.into_iter().next().unwrap(),
            });
        } else {
            // vector: capture EVERY step into a temp (reading the OLD accumulators)
            // BEFORE any assign — the update must be SIMULTANEOUS (Jacobi). a
            // direct `acc_j = step_j` sequence would let a later assign read an
            // already-updated sibling accumulator (Gauss-Seidel), corrupting it.
            for (j, value) in step_exprs.into_iter().enumerate() {
                loop_body.push(ScalarStmt::Let {
                    name: tmp_names[j].clone(),
                    element: ElementTy::F64,
                    value,
                });
            }
            for j in 0..acc_names.len() {
                loop_body.push(ScalarStmt::Assign {
                    name: acc_names[j].clone(),
                    value: ScalarExpr::Var(tmp_names[j].clone()),
                });
            }
        }
        // early-out: if the break predicate is satisfied AFTER the assigns, exit
        // the loop. the assigns already wrote the converged values (the freeze
        // pattern means step_j = select(conv, old_j, new_j) is a no-op once conv
        // fires, so the accs hold the converged answer). subsequent iterations
        // would re-compute the same body and freeze the writes again — exactly
        // the dead work this break eliminates.
        if let Some(be) = break_expr {
            loop_body.push(ScalarStmt::If {
                cond: be,
                then_body: vec![ScalarStmt::Break],
            });
        }
        self.body.push(ScalarStmt::For {
            iter: iter_name,
            bound: DimExpr::Literal(count),
            body: loop_body,
        });
        self.bindings.insert(
            iter_id,
            Binding::Concrete(vec![ScalarExpr::Var(acc_names[result].clone())]),
        );
    }

    /// F4.0c: lower a Morphism node to its semantic scalar expression.
    /// the proc-macro synthesizes the args in a kind-specific order:
    ///   - Diff / FaceAvg: [field[coord], field[+ax]]
    ///   - Curl:           [E_J@coord, E_J@+e_K, E_K@coord, E_K@+e_J,
    ///                      inv_dx_J, inv_dx_K]
    /// each kind reduces to a small arithmetic combination over those
    /// reads. axis info lives in the MorphismKind variant for ndim
    /// inference; lowering itself works off the args.
    fn lower_morphism(
        &mut self,
        kind: crate::morphism::MorphismKind,
        args: &[NodeId],
        _out_ty: &TensorTy,
    ) -> Vec<ScalarExpr> {
        use crate::morphism::MorphismKind;
        let body = match kind {
            // Diff(f, axis) = f[+ax] - f[coord] = a1 - a0
            MorphismKind::Diff { .. } => {
                let a0 = self.require_concrete(args[0], "Diff arg 0")[0].clone();
                let a1 = self.require_concrete(args[1], "Diff arg 1")[0].clone();
                ScalarExpr::BinOp(BinaryKind::Sub, Box::new(a1), Box::new(a0))
            }
            // FaceAvg(f, axis) = 0.5 * (f[coord] + f[+ax]) = 0.5 * (a0 + a1)
            MorphismKind::FaceAvg { .. } => {
                let a0 = self.require_concrete(args[0], "FaceAvg arg 0")[0].clone();
                let a1 = self.require_concrete(args[1], "FaceAvg arg 1")[0].clone();
                ScalarExpr::BinOp(
                    BinaryKind::Mul,
                    Box::new(ScalarExpr::Const(ConstValue::F64(0.5))),
                    Box::new(ScalarExpr::BinOp(
                        BinaryKind::Add,
                        Box::new(a0),
                        Box::new(a1),
                    )),
                )
            } // F5.4-retire: `MorphismKind::Curl` and `::CtEdgeEmf` were
              // RMHD-specific. their stencil-building moved to the macro
              // dispatcher (`symbi-macros::ir_builder::dispatch_curl_morphism`
              // and `dispatch_ct_edge_emf_morphism`), where the curl is
              // emitted inline as `ElementWise` nodes and the CT-EMF as a
              // straight `OpaqueCall` tree. nothing for lower.rs to do.
        };
        vec![body]
    }

    fn lower_opaque_call(&mut self, name: &Symbol, args: &[NodeId]) -> Vec<ScalarExpr> {
        // each arg is rank-0 by construction (Apply's scalar-function contract).
        // pull its single scalar binding and emit a FreeCall.
        let arg_exprs: Vec<ScalarExpr> = args
            .iter()
            .map(|a| self.require_concrete(*a, "Apply arg")[0].clone())
            .collect();
        vec![ScalarExpr::FreeCall {
            name: name.as_str().to_string(),
            args: arg_exprs,
        }]
    }

    fn lower_load_at(&mut self, field_key: &Symbol, components: &[NodeId]) -> Vec<ScalarExpr> {
        // each component is rank-0 (enforced at graph.rs::load_at). pull
        // its single scalar binding and assemble a FieldLoadAt that the
        // emit_kernel rewrite pass will turn into `buf<idx>[<flat>]`.
        let comp_exprs: Vec<ScalarExpr> = components
            .iter()
            .map(|c| self.require_concrete(*c, "LoadAt component")[0].clone())
            .collect();
        vec![ScalarExpr::FieldLoadAt {
            field_key: field_key.as_str().to_string(),
            components: comp_exprs,
        }]
    }

    fn lower_const(&mut self, v: ConstValue) -> Vec<ScalarExpr> {
        // Const is rank-0 by construction (see graph.rs add_const).
        vec![ScalarExpr::Const(v)]
    }

    fn lower_element_wise(
        &mut self,
        op: ElementWiseOp,
        inputs: &[NodeId],
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        let out_dims = resolve_literal_dims(&out_ty.shape);
        let out_indices = iter_row_major(&out_dims);
        let arity = op.arity();
        debug_assert_eq!(inputs.len(), arity, "ElementWise builder enforced arity");
        let in_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|id| resolve_literal_dims(&graph.ty(*id).shape))
            .collect();

        let mut out = Vec::with_capacity(out_indices.len());
        for out_idx in &out_indices {
            let mut input_exprs: Vec<ScalarExpr> = Vec::with_capacity(arity);
            for (k, in_shape) in in_shapes.iter().enumerate() {
                let flat = flat_index_with_broadcast(in_shape, &out_dims, out_idx);
                input_exprs.push(
                    self.require_concrete(inputs[k], "ElementWise/Transcendental")[flat].clone(),
                );
            }
            out.push(scalar_element_wise(op, input_exprs));
        }
        out
    }

    fn lower_transcendental(
        &mut self,
        op: TranscendentalOp,
        inputs: &[NodeId],
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        let out_dims = resolve_literal_dims(&out_ty.shape);
        let out_indices = iter_row_major(&out_dims);
        let arity = op.arity();
        debug_assert_eq!(inputs.len(), arity, "Transcendental builder enforced arity");
        let in_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|id| resolve_literal_dims(&graph.ty(*id).shape))
            .collect();

        let mut out = Vec::with_capacity(out_indices.len());
        for out_idx in &out_indices {
            let mut input_exprs: Vec<ScalarExpr> = Vec::with_capacity(arity);
            for (k, in_shape) in in_shapes.iter().enumerate() {
                let flat = flat_index_with_broadcast(in_shape, &out_dims, out_idx);
                input_exprs.push(
                    self.require_concrete(inputs[k], "ElementWise/Transcendental")[flat].clone(),
                );
            }
            out.push(scalar_transcendental(op, input_exprs));
        }
        out
    }

    fn lower_construct(&mut self, inputs: &[NodeId]) -> Vec<ScalarExpr> {
        // each input is a rank-K tensor; output is rank-(K+1) with the
        // outermost dim = inputs.len(). bindings concatenate in order.
        let mut out = Vec::new();
        for id in inputs {
            out.extend(self.require_concrete(*id, "Construct").iter().cloned());
        }
        out
    }

    fn lower_index(&mut self, tensor: NodeId, idxs: &[DimIndex], graph: &Graph) -> Vec<ScalarExpr> {
        // V1: only literal indices into a literal-dim tensor produce a
        // determinate flat index. Generic indices require loop emission
        // (R.3.h); panics.
        let in_shape = resolve_literal_dims(&graph.ty(tensor).shape);
        let mut flat = Vec::with_capacity(idxs.len());
        for d in idxs {
            match d {
                DimIndex::Literal(k) => flat.push(*k),
                DimIndex::Generic(s) => panic!(
                    "scalarization does not yet support DimIndex::Generic('{}'); \
                     loop emission lands in R.3.h",
                    s
                ),
            }
        }
        let pos = flat_index(&in_shape, &flat);
        vec![self.require_concrete(tensor, "Index")[pos].clone()]
    }

    fn lower_broadcast(
        &mut self,
        tensor: NodeId,
        target_shape: &[DimExpr],
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        let in_shape = resolve_literal_dims(&graph.ty(tensor).shape);
        let out_shape = resolve_literal_dims(target_shape);
        let out_indices = iter_row_major(&out_shape);
        let mut out = Vec::with_capacity(out_indices.len());
        for out_idx in &out_indices {
            let flat = flat_index_with_broadcast(&in_shape, &out_shape, out_idx);
            out.push(self.require_concrete(tensor, "Broadcast")[flat].clone());
        }
        out
    }

    fn lower_reduce(
        &mut self,
        op: ReduceOp,
        axes: &[u32],
        input: NodeId,
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        let in_shape = resolve_literal_dims(&graph.ty(input).shape);
        let out_shape = resolve_literal_dims(&out_ty.shape);

        // partition input axes into kept (in output) vs reduced.
        let reduced_axes: Vec<usize> = axes.iter().map(|a| *a as usize).collect();
        let kept_axes: Vec<usize> = (0..in_shape.len())
            .filter(|i| !reduced_axes.contains(i))
            .collect();
        let reduced_dims: Vec<usize> = reduced_axes.iter().map(|a| in_shape[*a]).collect();

        let out_indices = iter_row_major(&out_shape);
        let red_indices = iter_row_major(&reduced_dims);

        let mut out = Vec::with_capacity(out_indices.len());
        for out_idx in &out_indices {
            // build the full input index by interleaving kept + reduced axis positions.
            let mut vals: Vec<ScalarExpr> = Vec::with_capacity(red_indices.len());
            for red_idx in &red_indices {
                let mut full = vec![0usize; in_shape.len()];
                for (k, &axis) in kept_axes.iter().enumerate() {
                    full[axis] = out_idx[k];
                }
                for (k, &axis) in reduced_axes.iter().enumerate() {
                    full[axis] = red_idx[k];
                }
                let flat = flat_index(&in_shape, &full);
                vals.push(self.require_concrete(input, "Reduce")[flat].clone());
            }
            out.push(fold_reduce(op, vals));
        }
        out
    }

    fn lower_select(
        &mut self,
        cond: NodeId,
        then_id: NodeId,
        else_id: NodeId,
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        let out_shape = resolve_literal_dims(&out_ty.shape);
        let cond_shape = resolve_literal_dims(&graph.ty(cond).shape);
        let then_shape = resolve_literal_dims(&graph.ty(then_id).shape);
        let else_shape = resolve_literal_dims(&graph.ty(else_id).shape);

        let out_indices = iter_row_major(&out_shape);
        let mut out = Vec::with_capacity(out_indices.len());
        for out_idx in &out_indices {
            let cf = flat_index_with_broadcast(&cond_shape, &out_shape, out_idx);
            let tf = flat_index_with_broadcast(&then_shape, &out_shape, out_idx);
            let ef = flat_index_with_broadcast(&else_shape, &out_shape, out_idx);
            let c = self.require_concrete(cond, "Select")[cf].clone();
            let t = self.require_concrete(then_id, "Select")[tf].clone();
            let e = self.require_concrete(else_id, "Select")[ef].clone();
            out.push(ScalarExpr::Select {
                cond: Box::new(c),
                then: Box::new(t),
                else_: Box::new(e),
            });
        }
        out
    }

    fn lower_einsum(
        &mut self,
        spec: &EinsumSpec,
        inputs: &[NodeId],
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Binding {
        // detect generic-dim path: any input with a non-literal shape
        // routes through the loop-form lowering (R.5.a V1: rank-0 output
        // and rank-1 generic inputs only).
        let any_generic = inputs.iter().any(|id| {
            graph
                .ty(*id)
                .shape
                .iter()
                .any(|d| matches!(d, DimExpr::Generic(_)))
        });
        if any_generic {
            return self.lower_einsum_loop(spec, inputs, out_ty, graph);
        }

        // literal-dim path: fully unroll all batched and contracted loops.
        Binding::Concrete(self.lower_einsum_literal(spec, inputs, out_ty, graph))
    }

    /// literal-dim path: existing fully-unrolled scalarization.
    fn lower_einsum_literal(
        &mut self,
        spec: &EinsumSpec,
        inputs: &[NodeId],
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Vec<ScalarExpr> {
        // gather per-input shape + context.
        let in_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|id| resolve_literal_dims(&graph.ty(*id).shape))
            .collect();

        let ctxs: Vec<EinCtx> = spec
            .inputs
            .iter()
            .enumerate()
            .map(|(i, atoms)| EinCtx::from_atoms(atoms, in_shapes[i].len()))
            .collect();

        // bind each named label to its concrete dim (first occurrence wins;
        // the IR builder already verified all occurrences agree).
        let mut label_dim: Vec<(char, usize)> = Vec::new();
        for (i, ctx) in ctxs.iter().enumerate() {
            for (label, axis) in ctx.named_axes() {
                if !label_dim.iter().any(|(c, _)| *c == label) {
                    label_dim.push((label, in_shapes[i][axis]));
                }
            }
        }
        let dim_of = |c: char| -> usize {
            label_dim
                .iter()
                .find(|(b, _)| *b == c)
                .map(|(_, d)| *d)
                .expect("label was validated by builder; should have a dim binding")
        };

        // output: ellipsis batch dims (if any) then named output labels.
        let out_shape = resolve_literal_dims(&out_ty.shape);
        let output_named: Vec<char> = spec
            .output
            .iter()
            .filter_map(|a| match a {
                Atom::Label(c) => Some(*c),
                _ => None,
            })
            .collect();
        let batch_size = out_shape.len() - output_named.len();
        let output_batch_shape = &out_shape[..batch_size];

        // labels in inputs but not in output are contracted.
        let mut all_input_labels: Vec<char> = Vec::new();
        for ctx in &ctxs {
            for (label, _) in ctx.named_axes() {
                if !all_input_labels.contains(&label) {
                    all_input_labels.push(label);
                }
            }
        }
        let contracted: Vec<char> = all_input_labels
            .iter()
            .filter(|c| !output_named.contains(c))
            .copied()
            .collect();
        let contracted_dims: Vec<usize> = contracted.iter().map(|c| dim_of(*c)).collect();

        // emit one ScalarExpr per output cell.
        let out_indices = iter_row_major(&out_shape);
        let contracted_indices = iter_row_major(&contracted_dims);

        let mut out = Vec::with_capacity(out_indices.len());
        for out_idx in &out_indices {
            let out_batch_idx = &out_idx[..batch_size];
            let out_label_idx = &out_idx[batch_size..];

            // for each contracted-axis tuple, build the product of input lookups.
            let mut products: Vec<ScalarExpr> = Vec::with_capacity(contracted_indices.len());
            for c_idx in &contracted_indices {
                // gather one scalar per input.
                let mut factors: Vec<ScalarExpr> = Vec::with_capacity(inputs.len());
                for (k, &nid) in inputs.iter().enumerate() {
                    let in_shape = &in_shapes[k];
                    let ctx = &ctxs[k];
                    let in_idx = compute_einsum_input_index(
                        ctx,
                        in_shape,
                        out_batch_idx,
                        out_label_idx,
                        c_idx,
                        &output_named,
                        &contracted,
                    );
                    let flat = flat_index(in_shape, &in_idx);
                    factors.push(self.require_concrete(nid, "Einsum")[flat].clone());
                }
                // product of all factors (left-associative Mul chain).
                let prod = factors
                    .into_iter()
                    .reduce(|a, b| ScalarExpr::BinOp(BinaryKind::Mul, Box::new(a), Box::new(b)))
                    .expect("einsum builder rejects zero-input specs");
                products.push(prod);
            }

            // sum of products.
            let sum = products
                .into_iter()
                .reduce(|a, b| ScalarExpr::BinOp(BinaryKind::Add, Box::new(a), Box::new(b)))
                .unwrap_or_else(|| {
                    // no contracted axes — output is just the product of inputs
                    // at the kept-index position. unreachable in practice: even
                    // the no-contraction case (e.g., "i,j->ij") produces a single
                    // c_idx = [], so products has exactly one entry.
                    panic!("einsum: empty product set — internal bug")
                });
            // suppress unused — batch info gets folded into out_batch_idx,
            // which the input-index computation already consumes.
            let _ = output_batch_shape;
            out.push(sum);
        }
        out
    }

    /// generic-dim path: emit a loop nest for contracted axes whose
    /// dim is Generic. V1 supports rank-0 output and rank-1 generic
    /// inputs only — broader shapes panic.
    fn lower_einsum_loop(
        &mut self,
        spec: &EinsumSpec,
        inputs: &[NodeId],
        out_ty: &TensorTy,
        graph: &Graph,
    ) -> Binding {
        // V1 restrictions
        assert!(
            out_ty.rank == 0,
            "scalarization R.5.a: const-generic einsum currently supports rank-0 outputs \
             only (got rank {}); broader output shapes need follow-up phases",
            out_ty.rank
        );
        for (i, id) in inputs.iter().enumerate() {
            let s = &graph.ty(*id).shape;
            assert!(
                s.len() <= 1,
                "scalarization R.5.a: const-generic einsum currently supports rank-1 inputs \
                 only (input {} has rank {}); rank-2+ generic needs follow-up phases",
                i,
                s.len()
            );
        }

        // gather: for each input, get either its Array binding (if generic)
        // or its single Concrete entry (if rank-0 literal).
        let input_bindings: Vec<Binding> = inputs
            .iter()
            .map(|id| self.bindings.get(id).expect("binding present").clone())
            .collect();

        // pull each contracted-axis label and find its dim. for V1, the
        // simplest case: spec "i,i->" with both inputs sharing axis 'i'.
        // walk the spec's first input's atoms; for each Label, find the
        // binding's dim and emit a loop.
        let atoms_first = &spec.inputs[0];
        let mut contracted_labels: Vec<(char, DimExpr)> = Vec::new();
        for atom in atoms_first {
            if let Atom::Label(c) = atom {
                // dim from the first input's shape; the einsum builder validated agreement.
                let dim = graph.ty(inputs[0]).shape[0].clone();
                contracted_labels.push((*c, dim));
            }
        }

        // emit one accumulator: `let mut __acc_N: f64 = 0.0;`
        let acc_name = self.fresh("acc");
        self.body.push(ScalarStmt::LetMut {
            name: acc_name.clone(),
            element: out_ty.element,
            init: ScalarExpr::Const(zero_const(out_ty.element)),
        });

        // emit nested For loops, one per contracted label.
        // rank-1 inputs only, so every label is a
        // single-axis dim and one for-loop is emitted per distinct label.
        // the "i,i->" case has one label.
        let iter_name = self.fresh("ii");
        let dim = contracted_labels[0].1.clone();

        // build the loop body: acc += prod(input_k[iter_name] for k)
        let factors: Vec<ScalarExpr> = input_bindings
            .iter()
            .map(|b| b.get_at(ScalarExpr::Var(iter_name.clone())))
            .collect();
        let prod = factors
            .into_iter()
            .reduce(|a, b| ScalarExpr::BinOp(BinaryKind::Mul, Box::new(a), Box::new(b)))
            .expect("einsum has at least one input");
        let body_stmt = ScalarStmt::CompoundAssign {
            name: acc_name.clone(),
            op: BinaryKind::Add,
            value: prod,
        };

        self.body.push(ScalarStmt::For {
            iter: iter_name,
            bound: dim,
            body: vec![body_stmt],
        });

        // result: a single scalar referring to the accumulator.
        Binding::Concrete(vec![ScalarExpr::Var(acc_name)])
    }

    fn lower_param(&mut self, sym: &Symbol, ty: &TensorTy) -> Binding {
        let base = sym.as_str().to_string();

        // V1 const-generic path: rank-1 with a single Generic dim
        // becomes an Array binding pointing at a rank-1 array param.
        if ty.rank == 1
            && let DimExpr::Generic(_) = &ty.shape[0]
        {
            self.params.push(LoweredParam::array(
                base.clone(),
                ty.element,
                ty.shape[0].clone(),
            ));
            return Binding::Array { container: base };
        }

        // literal-dim path (or rank-0): fan out into N scalar params.
        let dims = resolve_literal_dims(&ty.shape);
        let indices = iter_row_major(&dims);
        let mut out = Vec::with_capacity(indices.len());
        for idx in indices {
            let name = if idx.is_empty() {
                base.clone()
            } else {
                format!(
                    "{}{}",
                    base,
                    idx.iter().map(|i| format!("_{}", i)).collect::<String>()
                )
            };
            self.params
                .push(LoweredParam::scalar(name.clone(), ty.element));
            out.push(ScalarExpr::Var(name));
        }
        Binding::Concrete(out)
    }
}

// ----- entry point -----

/// scalarize a graph into a LoweredFn.
///
/// the output node's components become the function's results. the
/// function name is the caller's choice (typically the elemental or
/// kernel function name).
///
/// V1 limitation: every dim in every tensor type must be Literal.
/// Generic dims trigger a panic; const-generic loop support is
/// unimplemented.
pub fn scalarize(graph: &Graph, output: NodeId, name: &str) -> LoweredFn {
    let mut sc = Scalarizer::new();
    let in_degrees = Scalarizer::compute_in_degrees(graph);
    for (id, node, ty) in graph.iter() {
        sc.lower_node(id, &node.op, ty, graph);
        sc.maybe_hoist_to_let(id, &in_degrees, ty);
    }
    let results = match sc.bindings.get(&output) {
        Some(Binding::Concrete(v)) => v.clone(),
        Some(Binding::Array { .. }) => panic!(
            "scalarize: output node {:?} produced an Array binding; V1 R.5.a \
             only supports rank-0 outputs through the generic-dim path",
            output
        ),
        None => panic!("output node {:?} not lowered", output),
    };
    let out_ty = graph.ty(output).clone();
    let mut f = LoweredFn {
        name: name.to_string(),
        params: sc.params,
        body: sc.body,
        results,
        result_element: out_ty.element,
        result_shape: out_ty.shape,
    };
    // F1.B.10: CSE pass collapses duplicated sub-expressions to
    // __cse_<n> Lets. without this, large kernels emit single-line
    // CUDA source strings >700 KB and blow up rustc's memory during
    // string-literal lowering (40-50 GiB anon-RSS on the symbi crate).
    crate::passes::cse::cse_lowered_fn(&mut f);
    f
}

/// kernel-mode scalarization: walk the graph once and extract one
/// scalar result per element of `outputs`, sharing the let-binding
/// body across all outputs. used by the kernel emitter (R.6.d) to
/// emit a single __global__ that performs multiple buffer writes.
///
/// every output node must be rank-0 (writes to a buffer are scalar);
/// multi-component outputs panic with a clear message.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KernelScalarized {
    pub params: Vec<LoweredParam>,
    pub body: Vec<ScalarStmt>,
    pub outputs: Vec<ScalarExpr>,
}

pub fn scalarize_kernel(graph: &Graph, outputs: &[NodeId]) -> KernelScalarized {
    let mut sc = Scalarizer::new();
    let in_degrees = Scalarizer::compute_in_degrees(graph);
    // DCE: walk backward from `outputs`, collect transitively-reachable nodes.
    // any node NOT reachable from an output is dead — its arithmetic contributes
    // to no buffer write. without this filter, the iso flux (which traces the
    // Newtonian regime then drops the `flux.nrg` write) still lowers the entire
    // energy U/F sub-DAG into the body as orphan `__cse_*` let-bindings; nvcc
    // DCEs the SASS but pays the parse + register-pressure cost. closing the
    // dead path here removes them from the emitted CUDA source up front.
    //
    // IterateInline cone nodes (lowered INSIDE the loop body, not the main pass)
    // are reached via the IterateInline node's `steps` field through `Op::inputs`
    // — so a reachable IterateInline pulls its cone in automatically.
    let reachable = reachable_from_outputs(graph, outputs);
    // docs/design/14: an IterateInline emits its body ONCE as a `for`. its `step`
    // sub-DAG's acc-dependent cone must be lowered INSIDE the loop, not in the
    // main pass — collect those nodes to skip, and the cone per iterate node.
    //
    // docs/design/23: Op::Scope body NodeIds are similarly partitioned —
    // they belong to the Scope's lexical region and are lowered INSIDE a
    // `ScalarStmt::Scope` brace block, not at function root.
    //
    // for NESTED scopes (an Op::Scope inside another Op::Scope's body), a
    // single NodeId may appear in multiple scopes' body lists. `scope_owner`
    // maps each scoped NodeId to its INNERMOST owner — built by walking the
    // graph in NodeId order (inner scopes have smaller NodeIds than the
    // outer scopes containing them, so first-claim wins is correct).
    let mut skip: HashSet<NodeId> = HashSet::new();
    let mut iter_cones: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut scope_owner: HashMap<NodeId, NodeId> = HashMap::new();
    // docs/design/23: a NodeId can appear in a scope's
    // body list yet ALSO be referenced from outside the scope, because the
    // graph is hash-consed — a closure may compute `bn * bn` (pushed inside
    // the scope) and later outer code may produce a structurally-identical
    // `bn * bn` that hash-conses to the SAME NodeId. lowering it inside the
    // scope's brace breaks the outer reference. so scope-body
    // bookkeeping is collected first; a SECOND pass evicts shared NodeIds from
    // scope_owner so the main pass lowers them at the outer level.
    let mut scope_body: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut scope_result: HashMap<NodeId, NodeId> = HashMap::new();
    // `Op::IfElse` (the DUAL of IterateInline): each arm body is a lexical
    // region lowered INSIDE its `if`/`else` brace — same partition as Scope but
    // TWO regions per node. `branch_owner` maps each arm-owned NodeId to its
    // (ifelse_id, is_then_arm); innermost-wins via NodeId order (a nested
    // IfElse node has a smaller id than the arm containing it, so its arm nodes
    // are claimed first). `branch_body` records each arm's set for eviction.
    let mut branch_owner: HashMap<NodeId, (NodeId, bool)> = HashMap::new();
    for (id, node, _) in graph.iter() {
        if !reachable.contains(&id) {
            continue;
        }
        if let Op::IterateInline {
            accs,
            steps,
            break_when,
            ..
        } = &node.op
        {
            let cone = iterate_cone(graph, accs, steps, *break_when);
            for &c in &cone {
                skip.insert(c);
            }
            iter_cones.insert(id, cone);
        }
        if let Op::Scope { body, result } = &node.op {
            let body_set: HashSet<NodeId> = body.iter().copied().collect();
            scope_body.insert(id, body_set);
            scope_result.insert(id, *result);
            for &b in body {
                // first-seen wins → INNERMOST scope claims the NodeId.
                scope_owner.entry(b).or_insert(id);
            }
        }
        if let Op::IfElse {
            then_body,
            else_body,
            ..
        } = &node.op
        {
            for &b in then_body {
                branch_owner.entry(b).or_insert((id, true));
            }
            for &b in else_body {
                branch_owner.entry(b).or_insert((id, false));
            }
        }
    }
    // build the reverse `users` map for shared-claim eviction below. only
    // reachable nodes contribute; an unreachable user couldn't share anything.
    let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (id, node, _) in graph.iter() {
        if !reachable.contains(&id) {
            continue;
        }
        for input in node.op.inputs() {
            users.entry(input).or_default().push(id);
        }
    }
    // each kernel OUTPUT is a virtual user — a sentinel NodeId past the
    // graph's range. ensures eviction catches "scope body node is also a
    // top-level kernel output" — otherwise the output's binding would be
    // a Var(...) only visible inside the scope's brace.
    let output_sentinel = NodeId(u32::MAX);
    for out in outputs {
        users.entry(*out).or_default().push(output_sentinel);
    }
    // evict scope_owner claims for NodeIds with users OUTSIDE the claimed
    // scope's body. the scope's `result` NodeId is exempt — it's allowed
    // to be referenced by the Op::Scope node (which is in the OUTER body)
    // because that reference resolves to the scope's named output.
    //
    // a NESTED Op::Scope NodeId is also exempt: it's a STRUCTURAL child of
    // the outer scope (lowered via the recursive `lower_scope` dispatch),
    // and any "outside" use of its value lives in the OUTER scope's body —
    // which the recursive dispatch handles correctly. evicting nested
    // Op::Scope nodes would dump their body INTO the outer level, defeating
    // the bounded-pressure point.
    let mut evict: Vec<NodeId> = Vec::new();
    for (&x, &owner_scope) in scope_owner.iter() {
        if matches!(graph.node(x).op, Op::Scope { .. }) {
            continue; // nested scope: structural, handled via recursive dispatch.
        }
        let body_set = match scope_body.get(&owner_scope) {
            Some(b) => b,
            None => continue,
        };
        let user_list = match users.get(&x) {
            Some(u) => u,
            None => continue,
        };
        let is_result = scope_result.get(&owner_scope) == Some(&x);
        for &u in user_list {
            // the Op::Scope node references its result NodeId via its
            // structural `result` field — allowed ONLY for the result.
            if u == owner_scope && is_result {
                continue;
            }
            if body_set.contains(&u) {
                continue;
            }
            // user is OUTSIDE the scope's body — shared. evict so main pass
            // lowers X at the outer level (the scope body still names the
            // outer's let in its `Var(...)` references). this handles BOTH
            // non-result body members (the `bn * bn` hash-cons case) AND
            // result NodeIds that ALSO leak outside (an inner expression
            // hash-consed with an outer expression).
            evict.push(x);
            break;
        }
    }
    for x in evict {
        scope_owner.remove(&x);
    }
    // now populate `skip` from the FINAL scope_owner (so evicted nodes
    // remain visible to the main pass).
    for &b in scope_owner.keys() {
        skip.insert(b);
    }
    // branch-arm eviction (the IfElse dual of the scope eviction above). a
    // then/else-arm node that is used OUTSIDE its arm — by the sibling arm
    // (cross-arm hash-cons share) or by outer code — is hoisted to the outer
    // level and computed ONCE (shared work is unconditional, which is correct).
    // CRUCIAL difference from Scope: the IfElse node lists its OWN arm bodies as
    // inputs (for remap-safety), so it appears as a "user" of every arm node;
    // that self-reference is NOT an escape — exempt `u == ifelse_id` for every
    // arm node, else laziness would be destroyed (all arms hoisted out).
    // FIXPOINT eviction: an arm node X "escapes" if it has a REAL (dataflow)
    // user lying OUTSIDE the arm's CURRENT membership. evicting X moves it to
    // the outer level, so any in-arm INPUT of X must then escape too — hence
    // the iteration to a fixpoint (a single pass would miss the cascade, and
    // checking STATIC body membership would wrongly treat an already-evicted
    // body member as still-in-arm, leaving a dangling in-arm reference). a user
    // is NOT an escape when it is: the owning container itself (arm output /
    // structural self-listing); still owned by the SAME arm; or an OUTER
    // container that lists X only structurally (nested arm ranges overlap).
    let mut changed = true;
    while changed {
        changed = false;
        let candidates: Vec<NodeId> = branch_owner.keys().copied().collect();
        for x in candidates {
            let (ifelse_id, is_then) = branch_owner[&x];
            // nested IfElse node: structural, lowered via recursive dispatch —
            // never evicted (mirrors the nested-Op::Scope rule).
            if matches!(graph.node(x).op, Op::IfElse { .. }) {
                continue;
            }
            let Some(user_list) = users.get(&x) else {
                continue;
            };
            let escapes = user_list.iter().any(|&u| {
                if u == ifelse_id {
                    return false;
                } // owning container
                if branch_owner.get(&u) == Some(&(ifelse_id, is_then)) {
                    return false;
                } // same arm, still
                if is_structural_container_use(graph, u, x) {
                    return false;
                } // outer structural listing
                true
            });
            if escapes {
                branch_owner.remove(&x);
                changed = true;
            }
        }
    }
    // mixed scope/branch nesting (a Scope inside a cond arm, or vice versa) is
    // not supported — both containers would try to lower the
    // shared node. assert disjointness so it fails loudly, never silently
    // miscompiles. (the cubic-resolvent cond chain uses no scopes.)
    debug_assert!(
        scope_owner.keys().all(|k| !branch_owner.contains_key(k)),
        "scalarize: a NodeId is owned by BOTH a Scope and an IfElse arm — \
         mixed scope/branch nesting is not yet supported",
    );
    for &b in branch_owner.keys() {
        skip.insert(b);
    }
    for (id, node, ty) in graph.iter() {
        if !reachable.contains(&id) {
            continue; // unreachable from any output — DCE
        }
        if skip.contains(&id) {
            continue; // a loop cone OR scope body node — lowered inside its container
        }
        if let Op::IterateInline {
            inits,
            steps,
            count,
            result,
            break_when,
            ..
        } = &node.op
        {
            sc.lower_iterate_inline(
                id,
                inits,
                steps,
                *count,
                *result as usize,
                *break_when,
                &iter_cones[&id],
                &in_degrees,
                graph,
            );
            continue;
        }
        if let Op::Scope { body, result } = &node.op {
            sc.lower_scope(id, body, *result, ty, &scope_owner, &in_degrees, graph);
            continue;
        }
        if let Op::IfElse {
            cond,
            then_body,
            then_results,
            else_body,
            else_results,
        } = &node.op
        {
            sc.lower_if_else(
                id,
                *cond,
                then_body,
                then_results,
                else_body,
                else_results,
                graph,
                &branch_owner,
                &in_degrees,
            );
            continue;
        }
        sc.lower_node(id, &node.op, ty, graph);
        sc.maybe_hoist_to_let(id, &in_degrees, ty);
    }
    let lowered_outputs: Vec<ScalarExpr> = outputs
        .iter()
        .map(|out| match sc.bindings.get(out) {
            Some(Binding::Concrete(v)) if v.len() == 1 => v[0].clone(),
            Some(Binding::Concrete(v)) => panic!(
                "scalarize_kernel: output node {:?} has {} scalar components; \
                 kernel writes must be rank-0 scalars",
                out,
                v.len()
            ),
            Some(Binding::Array { .. }) => panic!(
                "scalarize_kernel: output node {:?} produced an Array binding; \
                 kernel writes must be rank-0 scalars",
                out
            ),
            None => panic!("scalarize_kernel: output node {:?} not lowered", out),
        })
        .collect();
    let mut k = KernelScalarized {
        params: sc.params,
        body: sc.body,
        outputs: lowered_outputs,
    };
    // F1.B.10: CSE pass. see comment in `scalarize`.
    crate::passes::cse::cse_kernel(&mut k);
    k
}

// ----- helpers -----

// `op_inputs` has been replaced by `Op::inputs()` — see `graph.rs`. the rule
// "which fields of this variant are NodeIds" lives in `Op::try_map_inputs`
// (single source of truth). callers invoke `node.op.inputs()`
// directly; this stub is kept only as a documentation anchor.
fn op_inputs(op: &Op) -> Vec<NodeId> {
    op.inputs()
}

/// transitive backward reachability from a set of output nodes. used by
/// `scalarize_kernel` to skip lowering of dead nodes — arithmetic that
/// contributes to no buffer write should never enter the emitted body.
///
/// `Op::inputs` enumerates a node's NodeId children (single source of truth
/// for graph topology, see `Op::try_map_inputs` in `graph.rs`), so walking
/// `inputs()` recursively from each output covers the entire live subgraph.
fn reachable_from_outputs(graph: &Graph, outputs: &[NodeId]) -> HashSet<NodeId> {
    let mut reachable = HashSet::new();
    let mut stack: Vec<NodeId> = outputs.to_vec();
    while let Some(id) = stack.pop() {
        if !reachable.insert(id) {
            continue;
        }
        for input in graph.node(id).op.inputs() {
            stack.push(input);
        }
    }
    reachable
}

/// docs/design/14: the acc-dependent CONE of an `IterateInline` — the nodes that
/// must be recomputed each iteration. = the union backward-reachable set of all
/// `steps` whose `dep` flag is true, where `dep[n] = accs.contains(n) ||
/// any(dep[input])`. returned in increasing-id (topological) order. loop-INVARIANT
/// nodes (dep == false) stay in the main pass as kernel locals computed once.
/// is `user`'s reference to `x` purely STRUCTURAL — i.e., `user` is a container
/// (IfElse / Scope) that lists `x` in its body for remap-safety/DCE but does
/// NOT consume `x` as dataflow (cond / result)? such a reference must not count
/// as a "use outside the arm" in the branch-eviction pass, otherwise nested
/// cond arms (whose body ranges overlap their outer containers) get flattened.
fn is_structural_container_use(graph: &Graph, user: NodeId, x: NodeId) -> bool {
    match &graph.node(user).op {
        Op::IfElse {
            cond,
            then_results,
            else_results,
            ..
        } => *cond != x && !then_results.contains(&x) && !else_results.contains(&x),
        Op::Scope { result, .. } => *result != x,
        _ => false,
    }
}

fn iterate_cone(
    graph: &Graph,
    accs: &[NodeId],
    steps: &[NodeId],
    break_when: Option<NodeId>,
) -> Vec<NodeId> {
    // union backward-reachable set from all `steps` (transitive inputs). also
    // seed from `break_when` so its acc-dependent expression is part of the
    // cone (lowered INSIDE the for-loop, not hoisted as a loop invariant —
    // hoisting would constant-fold `0.5 < initial_done` into a dead `false`).
    let mut back: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = steps.to_vec();
    if let Some(bw) = break_when {
        stack.push(bw);
    }
    while let Some(n) = stack.pop() {
        if !back.insert(n) {
            continue;
        }
        for inp in op_inputs(&graph.node(n).op) {
            stack.push(inp);
        }
    }
    // `dep` bottom-up over id order (inputs precede consumers in the arena).
    let mut dep: HashMap<NodeId, bool> = HashMap::new();
    let mut cone = Vec::new();
    for (id, node, _) in graph.iter() {
        if !back.contains(&id) {
            continue;
        }
        let d = accs.contains(&id)
            || op_inputs(&node.op)
                .iter()
                .any(|i| *dep.get(i).unwrap_or(&false));
        dep.insert(id, d);
        if d {
            cone.push(id);
        }
    }
    cone
}

/// resolve a shape to its literal-only dimensions. panics on generic.
fn resolve_literal_dims(shape: &[DimExpr]) -> Vec<usize> {
    shape
        .iter()
        .map(|d| match d {
            DimExpr::Literal(n) => *n,
            DimExpr::Generic(s) => panic!(
                "scalarization does not yet support const-generic dim '{}'; \
                 literal-dim only in V1 (R.3.h adds loop emission)",
                s
            ),
        })
        .collect()
}

/// row-major flat index for `idx` against `shape`.
fn flat_index(shape: &[usize], idx: &[usize]) -> usize {
    debug_assert_eq!(shape.len(), idx.len(), "shape and idx must match in rank");
    let mut acc = 0usize;
    for axis in 0..shape.len() {
        acc = acc * shape[axis] + idx[axis];
    }
    acc
}

/// given an input shape, the broadcast output shape, and an output
/// index, compute the flat index into the input's row-major bindings.
/// follows numpy-style broadcasting: right-align input against output,
/// and gather index 0 for any axis where the input dim is 1 or absent.
fn flat_index_with_broadcast(in_shape: &[usize], out_shape: &[usize], out_idx: &[usize]) -> usize {
    debug_assert!(
        in_shape.len() <= out_shape.len(),
        "input rank exceeds output"
    );
    debug_assert_eq!(out_shape.len(), out_idx.len(), "out_idx rank mismatch");
    let offset = out_shape.len() - in_shape.len();
    let mut in_idx = Vec::with_capacity(in_shape.len());
    for axis in 0..in_shape.len() {
        let out_axis = axis + offset;
        if in_shape[axis] == 1 {
            in_idx.push(0);
        } else {
            in_idx.push(out_idx[out_axis]);
        }
    }
    flat_index(in_shape, &in_idx)
}

/// build a scalar expression for an element-wise op given its scalar
/// inputs (already gathered with broadcast applied).
fn scalar_element_wise(op: ElementWiseOp, mut inputs: Vec<ScalarExpr>) -> ScalarExpr {
    // helper aliases
    let pop = |inputs: &mut Vec<ScalarExpr>| inputs.remove(0);
    match op {
        // operator-based binary
        ElementWiseOp::Add => binop_box(BinaryKind::Add, &mut inputs),
        ElementWiseOp::Sub => binop_box(BinaryKind::Sub, &mut inputs),
        ElementWiseOp::Mul => binop_box(BinaryKind::Mul, &mut inputs),
        ElementWiseOp::Div => binop_box(BinaryKind::Div, &mut inputs),
        ElementWiseOp::Eq => binop_box(BinaryKind::Eq, &mut inputs),
        ElementWiseOp::Ne => binop_box(BinaryKind::Ne, &mut inputs),
        ElementWiseOp::Lt => binop_box(BinaryKind::Lt, &mut inputs),
        ElementWiseOp::Le => binop_box(BinaryKind::Le, &mut inputs),
        ElementWiseOp::Gt => binop_box(BinaryKind::Gt, &mut inputs),
        ElementWiseOp::Ge => binop_box(BinaryKind::Ge, &mut inputs),
        // operator-based binary (bitwise / logical on Bool)
        ElementWiseOp::BitAnd => binop_box(BinaryKind::BitAnd, &mut inputs),
        ElementWiseOp::BitOr => binop_box(BinaryKind::BitOr, &mut inputs),
        ElementWiseOp::BitXor => binop_box(BinaryKind::BitXor, &mut inputs),
        ElementWiseOp::BitNot => ScalarExpr::UnaryOp(UnaryKind::Not, Box::new(inputs.remove(0))),
        // method-based binary. min/max stay as method calls — every backend
        // renders them as the `a<b?a:b` / `a>b?a:b` ternary:
        // the cuda special-case arm, the interp/jit ternary, and the f64/f32
        // `Numeric` carrier. so CPU and GPU agree at NaN/signed-zero (tier-1 #2b)
        // WITHOUT lowering to a scoped `if`-select — that inlines nested min/max
        // chains into deeply-nested lexical scopes and overflows rustc's debuginfo.
        ElementWiseOp::Min => method_binary("min", &mut inputs),
        ElementWiseOp::Max => method_binary("max", &mut inputs),
        // integer floor division: rust renders the method directly
        // (`a.div_euclid(b)`); the cuda emitter special-cases the name into an
        // explicit floor-division ternary; the interpreter floors the quotient.
        ElementWiseOp::FloorDiv => method_binary("div_euclid", &mut inputs),
        // operator-based unary
        ElementWiseOp::Neg => ScalarExpr::UnaryOp(UnaryKind::Neg, Box::new(pop(&mut inputs))),
        // method-based unary. abs stays a method call (same ternary rationale as
        // min/max above) — `x<0?-x:x`, never the IEEE-symmetric `fabs`.
        ElementWiseOp::Abs => method_unary("abs", &mut inputs),
        ElementWiseOp::Sqrt => method_unary("sqrt", &mut inputs),
        ElementWiseOp::Floor => method_unary("floor", &mut inputs),
        ElementWiseOp::Ceil => method_unary("ceil", &mut inputs),
        ElementWiseOp::Round => method_unary("round", &mut inputs),
        ElementWiseOp::Trunc => method_unary("trunc", &mut inputs),
        ElementWiseOp::IsFinite => method_unary("is_finite", &mut inputs),
        ElementWiseOp::IsNaN => method_unary("is_nan", &mut inputs),
        // transcendental unary (interp + CUDA emit already know these names)
        ElementWiseOp::Sin => method_unary("sin", &mut inputs),
        ElementWiseOp::Cos => method_unary("cos", &mut inputs),
        ElementWiseOp::Acos => method_unary("acos", &mut inputs),
        ElementWiseOp::Sinh => method_unary("sinh", &mut inputs),
        ElementWiseOp::Cosh => method_unary("cosh", &mut inputs),
        ElementWiseOp::Asinh => method_unary("asinh", &mut inputs),
        ElementWiseOp::Acosh => method_unary("acosh", &mut inputs),
        // transcendental binary: Rust .powf(b) / CUDA pow(a,b).
        ElementWiseOp::Pow => method_binary("powf", &mut inputs),
        // numeric conversion (inserted by the graph's type promotion).
        ElementWiseOp::Cast(to) => ScalarExpr::Cast {
            to,
            value: Box::new(pop(&mut inputs)),
        },
    }
}

fn binop_box(kind: BinaryKind, inputs: &mut Vec<ScalarExpr>) -> ScalarExpr {
    let a = inputs.remove(0);
    let b = inputs.remove(0);
    fold_arith_identity(kind, a, b)
}

/// recognize the exact constants `0` and `1` (across numeric types) so the
/// arithmetic-identity peephole can fire on them. returns `None` for any
/// non-constant or non-{0,1} constant — runtime values that
/// happen to round to these are never folded.
fn const_zero_or_one(e: &ScalarExpr) -> Option<f64> {
    let ScalarExpr::Const(c) = e else { return None };
    let v = match c {
        ConstValue::F64(x) => *x,
        ConstValue::F32(x) => *x as f64,
        ConstValue::I32(x) => *x as f64,
        ConstValue::U32(x) => *x as f64,
        ConstValue::Bool(_) => return None,
    };
    if v == 0.0 || v == 1.0 { Some(v) } else { None }
}

/// arithmetic-identity peephole on the scalar IR. eliminates the dead ops
/// emitted by substrate constructions that contract against unit vectors
/// (e.g., `v · ehat` for `ehat = (1, 0, 0)` lowers to `v0*1 + v1*0 + v2*0`).
///
/// SAFE set (the ONLY identities folded here):
///   `x + 0 -> x`,  `0 + x -> x`
///   `x - 0 -> x`
///   `x * 1 -> x`,  `1 * x -> x`
///   `x / 1 -> x`
///
/// safety: `Mul`-absorbing arms (`x * 0 -> 0` / `0 * x -> 0`) are
/// INTENTIONALLY OMITTED — IEEE-754 says `inf * 0 = NaN` and `NaN * 0 = NaN`,
/// and the user's `feedback_no_silent_floors` policy is that NaN must
/// propagate so dt-reduction / regression-test machinery sees it. while the
/// substrate's current emit only feeds SYNTACTIC `Op::Const(0.0)` into Mul
/// (so `inf` could not in practice reach the zero arm), any future builder
/// that drops a syntactic 0 into a flux-limiter / floor-clamp position could
/// pick up an `inf * x` upstream — and the absorbing fold would mask it
/// silently. removed pre-emptively rather than left as a tripwire.
///
/// the safe set MUST stay in sync with `Graph::fold_arith_identity` in
/// `graph.rs` (the graph-layer fold) — both layers fold exactly
/// `{ Add[0], Sub[0], Mul[1], Div[1] }`, nothing more.
///
/// comparison / bitwise kinds fall through to construction unchanged.
fn fold_arith_identity(kind: BinaryKind, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    let ca = const_zero_or_one(&a);
    let cb = const_zero_or_one(&b);
    match (kind, ca, cb) {
        (BinaryKind::Add, _, Some(v)) if v == 0.0 => return a,
        (BinaryKind::Add, Some(v), _) if v == 0.0 => return b,
        (BinaryKind::Sub, _, Some(v)) if v == 0.0 => return a,
        (BinaryKind::Mul, _, Some(v)) if v == 1.0 => return a,
        (BinaryKind::Mul, Some(v), _) if v == 1.0 => return b,
        (BinaryKind::Div, _, Some(v)) if v == 1.0 => return a,
        _ => {}
    }
    ScalarExpr::BinOp(kind, Box::new(a), Box::new(b))
}

fn method_unary(name: &'static str, inputs: &mut Vec<ScalarExpr>) -> ScalarExpr {
    let recv = inputs.remove(0);
    ScalarExpr::MethodCall {
        receiver: Box::new(recv),
        method: name.to_string(),
        args: vec![],
    }
}

/// per-input context for einsum scalarization: positions of the
/// left/right named-label spans + whether the input had an ellipsis.
struct EinCtx<'a> {
    left: &'a [Atom],
    right: &'a [Atom],
    has_ellipsis: bool,
    /// for inputs with an ellipsis, the count of axes the ellipsis covers:
    /// `in_rank - left.len() - right.len()`. zero when no ellipsis.
    ellipsis_rank: usize,
}

impl<'a> EinCtx<'a> {
    fn from_atoms(atoms: &'a [Atom], in_rank: usize) -> Self {
        let pos = atoms.iter().position(|a| matches!(a, Atom::Ellipsis));
        match pos {
            None => EinCtx {
                left: atoms,
                right: &[],
                has_ellipsis: false,
                ellipsis_rank: 0,
            },
            Some(idx) => {
                let left = &atoms[..idx];
                let right = &atoms[idx + 1..];
                let ellipsis_rank = in_rank.saturating_sub(left.len() + right.len());
                EinCtx {
                    left,
                    right,
                    has_ellipsis: true,
                    ellipsis_rank,
                }
            }
        }
    }

    /// iterate `(label, input_axis_index)` for every named label across
    /// this input's left + right spans, in input-axis order.
    fn named_axes(&self) -> impl Iterator<Item = (char, usize)> + '_ {
        let left_iter = self
            .left
            .iter()
            .enumerate()
            .filter_map(|(axis, atom)| match atom {
                Atom::Label(c) => Some((*c, axis)),
                _ => None,
            });
        let right_start = self.left.len() + self.ellipsis_rank;
        let right_iter = self
            .right
            .iter()
            .enumerate()
            .filter_map(move |(axis, atom)| match atom {
                Atom::Label(c) => Some((*c, right_start + axis)),
                _ => None,
            });
        left_iter.chain(right_iter)
    }
}

/// compute the full input index for one input given the current output
/// position (split into batch + label parts) and current contracted-
/// index tuple. handles ellipsis broadcast against the output batch.
fn compute_einsum_input_index(
    ctx: &EinCtx<'_>,
    in_shape: &[usize],
    out_batch_idx: &[usize],
    out_label_idx: &[usize],
    c_idx: &[usize],
    output_named: &[char],
    contracted: &[char],
) -> Vec<usize> {
    let mut in_idx = vec![0usize; in_shape.len()];

    let resolve_label = |label: char| -> usize {
        if let Some(pos) = output_named.iter().position(|c| *c == label) {
            out_label_idx[pos]
        } else if let Some(pos) = contracted.iter().position(|c| *c == label) {
            c_idx[pos]
        } else {
            // builder validated; unreachable.
            panic!("einsum: label '{}' not in output or contracted set", label)
        }
    };

    // left named labels.
    for (axis, atom) in ctx.left.iter().enumerate() {
        if let Atom::Label(c) = atom {
            in_idx[axis] = resolve_label(*c);
        }
    }

    // ellipsis axes (if any): broadcast against output batch.
    if ctx.has_ellipsis && ctx.ellipsis_rank > 0 {
        let ellipsis_start = ctx.left.len();
        let batch_offset = out_batch_idx.len() - ctx.ellipsis_rank;
        for axis_within in 0..ctx.ellipsis_rank {
            let in_axis = ellipsis_start + axis_within;
            let out_batch_axis = axis_within + batch_offset;
            if in_shape[in_axis] == 1 {
                in_idx[in_axis] = 0;
            } else {
                in_idx[in_axis] = out_batch_idx[out_batch_axis];
            }
        }
    }

    // right named labels.
    let right_start = ctx.left.len() + ctx.ellipsis_rank;
    for (axis_within, atom) in ctx.right.iter().enumerate() {
        if let Atom::Label(c) = atom {
            in_idx[right_start + axis_within] = resolve_label(*c);
        }
    }

    in_idx
}

/// zero literal for the given element type, used to initialize loop
/// accumulators.
fn zero_const(e: ElementTy) -> ConstValue {
    match e {
        ElementTy::F64 => ConstValue::F64(0.0),
        ElementTy::F32 => ConstValue::F32(0.0),
        ElementTy::I32 => ConstValue::I32(0),
        ElementTy::U32 => ConstValue::U32(0),
        ElementTy::Bool => ConstValue::Bool(false),
    }
}

/// fold N scalars into one via a Reduce op.
fn fold_reduce(op: ReduceOp, vals: Vec<ScalarExpr>) -> ScalarExpr {
    let mut it = vals.into_iter();
    let first = it.next().expect("reduce over empty axis");
    it.fold(first, |acc, x| match op {
        ReduceOp::Min => ScalarExpr::MethodCall {
            receiver: Box::new(acc),
            method: "min".to_string(),
            args: vec![x],
        },
        ReduceOp::Max => ScalarExpr::MethodCall {
            receiver: Box::new(acc),
            method: "max".to_string(),
            args: vec![x],
        },
        ReduceOp::Or => ScalarExpr::BinOp(BinaryKind::BitOr, Box::new(acc), Box::new(x)),
        ReduceOp::And => ScalarExpr::BinOp(BinaryKind::BitAnd, Box::new(acc), Box::new(x)),
        ReduceOp::Xor => ScalarExpr::BinOp(BinaryKind::BitXor, Box::new(acc), Box::new(x)),
    })
}

fn method_binary(name: &'static str, inputs: &mut Vec<ScalarExpr>) -> ScalarExpr {
    let recv = inputs.remove(0);
    let arg = inputs.remove(0);
    ScalarExpr::MethodCall {
        receiver: Box::new(recv),
        method: name.to_string(),
        args: vec![arg],
    }
}

/// build a scalar expression for a transcendental op. all are method
/// calls in Rust source (`x.sin()`, `y.atan2(x)`, etc.).
fn scalar_transcendental(op: TranscendentalOp, mut inputs: Vec<ScalarExpr>) -> ScalarExpr {
    match op {
        TranscendentalOp::Sin => method_unary("sin", &mut inputs),
        TranscendentalOp::Cos => method_unary("cos", &mut inputs),
        TranscendentalOp::Tan => method_unary("tan", &mut inputs),
        TranscendentalOp::Asin => method_unary("asin", &mut inputs),
        TranscendentalOp::Acos => method_unary("acos", &mut inputs),
        TranscendentalOp::Atan => method_unary("atan", &mut inputs),
        TranscendentalOp::Atan2 => method_binary("atan2", &mut inputs),
        TranscendentalOp::Exp => method_unary("exp", &mut inputs),
        TranscendentalOp::Exp2 => method_unary("exp2", &mut inputs),
        TranscendentalOp::Log => method_unary("ln", &mut inputs),
        TranscendentalOp::Log2 => method_unary("log2", &mut inputs),
        TranscendentalOp::Log10 => method_unary("log10", &mut inputs),
        TranscendentalOp::Sinh => method_unary("sinh", &mut inputs),
        TranscendentalOp::Cosh => method_unary("cosh", &mut inputs),
        TranscendentalOp::Tanh => method_unary("tanh", &mut inputs),
        TranscendentalOp::Asinh => method_unary("asinh", &mut inputs),
        TranscendentalOp::Acosh => method_unary("acosh", &mut inputs),
        TranscendentalOp::Atanh => method_unary("atanh", &mut inputs),
        TranscendentalOp::Pow => method_binary("powf", &mut inputs),
        TranscendentalOp::Hypot => method_binary("hypot", &mut inputs),
    }
}

/// enumerate row-major index tuples for a shape. rank-0 yields the
/// empty index, rank-N yields the full cartesian product in row-major
/// order.
fn iter_row_major(dims: &[usize]) -> Vec<Vec<usize>> {
    if dims.is_empty() {
        return vec![vec![]];
    }
    let total: usize = dims.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut current = vec![0; dims.len()];
    for _ in 0..total {
        out.push(current.clone());
        // increment innermost first (row-major next-index).
        for axis in (0..dims.len()).rev() {
            current[axis] += 1;
            if current[axis] < dims[axis] {
                break;
            }
            current[axis] = 0;
        }
    }
    out
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TensorTy, VarianceTag};

    fn lit(n: usize) -> DimExpr {
        DimExpr::Literal(n)
    }

    // ---- F2.F: Op::Fold lowering ----

    #[test]
    fn fold_lowers_to_letmut_for_assign() {
        // build a Fold over a one-step accumulator body. verify the
        // lowered LoweredFn has the shape:
        //   LetMut __fold_acc_N = <init>
        //   For __fold_i_N in 0..count { Assign __fold_acc_N = body(...) }
        //   results: [Var(__fold_acc_N)]
        use crate::Symbol;
        use crate::graph::FnDef;

        let mut body = Graph::new();
        let bacc = body.add_scalar_param("acc", ElementTy::F64);
        let _bi = body.add_scalar_param("i", ElementTy::I32);
        let one = body.add_const(ConstValue::F64(1.0), None);
        let bout = body.element_wise(ElementWiseOp::Add, vec![bacc, one], None);
        let fn_def = FnDef {
            name: Symbol::intern("inc"),
            params: vec![
                (Symbol::intern("acc"), TensorTy::scalar(ElementTy::F64)),
                (Symbol::intern("i"), TensorTy::scalar(ElementTy::I32)),
            ],
            body,
            output: bout,
        };

        let mut g = Graph::new();
        let l = g.add_lambda(fn_def, None);
        let init = g.add_const(ConstValue::F64(0.0), None);
        let n = g.add_const(ConstValue::I32(60), None);
        let r = g.fold(l, init, n, None);

        let f = scalarize(&g, r, "fold_test");

        // body must contain a LetMut then a For containing an Assign.
        let mut saw_letmut = false;
        let mut saw_for_with_assign = false;
        for s in &f.body {
            match s {
                ScalarStmt::LetMut { name, .. } if name.starts_with("__fold_acc_") => {
                    saw_letmut = true;
                }
                ScalarStmt::For { iter, bound, body } => {
                    assert!(iter.starts_with("__fold_i_"));
                    assert_eq!(*bound, DimExpr::Literal(60));
                    for inner in body {
                        if matches!(inner, ScalarStmt::Assign { name, .. }
                                    if name.starts_with("__fold_acc_"))
                        {
                            saw_for_with_assign = true;
                        }
                    }
                }
                _ => {}
            }
        }
        assert!(saw_letmut, "expected LetMut for accumulator: {:?}", f.body);
        assert!(
            saw_for_with_assign,
            "expected For containing Assign: {:?}",
            f.body
        );

        // result should be Var(__fold_acc_<N>).
        match &f.results[0] {
            ScalarExpr::Var(n) => assert!(n.starts_with("__fold_acc_")),
            other => panic!("expected Var, got {:?}", other),
        }
    }

    // ---- docs/design/14: Op::IterateInline lowering ----

    #[test]
    fn iterate_inline_lowers_to_one_loop_body_emitted_once() {
        // a fixed-iteration sqrt-Newton: acc <- 0.5*(acc + a/acc). `a` is
        // loop-INVARIANT (must stay outside the loop); the acc-dependent step
        // (a/acc, acc+.., 0.5*..) goes INSIDE — emitted ONCE, not unrolled.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let acc = g.iter_acc(0, None);
        let half = g.add_const(ConstValue::F64(0.5), None);
        let aoa = g.element_wise(ElementWiseOp::Div, vec![a, acc], None); // a/acc
        let sum = g.element_wise(ElementWiseOp::Add, vec![acc, aoa], None); // acc + a/acc
        let step = g.element_wise(ElementWiseOp::Mul, vec![half, sum], None); // 0.5*(...)
        let init = g.add_const(ConstValue::F64(1.0), None);
        let it = g.iterate_inline_scalar(acc, init, step, 8, None, None);
        assert!(!g.has_errors(), "graph errors: {:?}", g.errors());

        let k = scalarize_kernel(&g, &[it]);

        // exactly ONE loop (the body emitted once, not 8x unrolled).
        let fors: Vec<&ScalarStmt> = k
            .body
            .iter()
            .filter(|s| matches!(s, ScalarStmt::For { .. }))
            .collect();
        assert_eq!(fors.len(), 1, "expected exactly one loop: {:?}", k.body);

        // a LetMut accumulator precedes the loop.
        let acc_name = match k
            .body
            .iter()
            .find(|s| matches!(s, ScalarStmt::LetMut { .. }))
        {
            Some(ScalarStmt::LetMut { name, .. }) => name.clone(),
            _ => panic!(
                "expected a LetMut accumulator before the loop: {:?}",
                k.body
            ),
        };

        // the loop runs `count` times and ends with `acc = step`.
        if let ScalarStmt::For { bound, body, .. } = fors[0] {
            assert_eq!(*bound, DimExpr::Literal(8));
            assert!(
                matches!(body.last(), Some(ScalarStmt::Assign { name, .. }) if *name == acc_name),
                "loop body must end with `acc = step`: {:?}",
                body,
            );
            // the acc-dependent step IS inside the loop (a Div for a/acc).
            assert!(
                body.iter()
                    .any(|s| matches!(s, ScalarStmt::Assign { value, .. }
                    if expr_mentions_div(value))
                        || matches!(s, ScalarStmt::Let { value, .. } if expr_mentions_div(value))),
                "the acc-dependent step must be inside the loop: {:?}",
                body,
            );
        }

        // the output is the post-loop accumulator.
        assert!(matches!(&k.outputs[0], ScalarExpr::Var(n) if *n == acc_name));
    }

    fn expr_mentions_div(e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::BinOp(BinaryKind::Div, _, _) => true,
            ScalarExpr::BinOp(_, a, b) => expr_mentions_div(a) || expr_mentions_div(b),
            ScalarExpr::UnaryOp(_, a) => expr_mentions_div(a),
            _ => false,
        }
    }

    #[test]
    fn vector_iterate_updates_simultaneously() {
        // P1 (RMHD c2p): a 2-component accumulator — the Fibonacci recurrence
        // (a, b) -> (b, a+b). this is the cleanest probe of SIMULTANEOUS update:
        // sequential (`a = b; b = a + b`) would read the NEW `a` and corrupt it.
        // the substrate must emit BOTH assigns AFTER the step body, reading the OLD
        // accumulators. (the RMHD KKC c2p carries a 4-value bracket the same way.)
        let mut g = Graph::new();
        let a0 = g.iter_acc(0, None);
        let a1 = g.iter_acc(1, None);
        let one = g.add_const(ConstValue::F64(1.0), None);
        // step: next_a = b ; next_b = a + b.
        let step1 = g.element_wise(ElementWiseOp::Add, vec![a0, a1], None);
        let it = g.iterate_inline(
            vec![a0, a1],
            vec![one, one],
            vec![a1, step1],
            5,
            1,
            None,
            None,
        );
        assert!(!g.has_errors(), "graph errors: {:?}", g.errors());

        let k = scalarize_kernel(&g, &[it]);

        // two accumulator LetMuts before the loop.
        let accs: Vec<String> = k
            .body
            .iter()
            .filter_map(|s| match s {
                ScalarStmt::LetMut { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            accs.len(),
            2,
            "expected 2 accumulator LetMuts: {:?}",
            k.body
        );

        // the loop ends with the two assigns BOTH last (simultaneous), each reading
        // the OLD accumulator locals.
        let for_body = k
            .body
            .iter()
            .find_map(|s| match s {
                ScalarStmt::For { body, .. } => Some(body.clone()),
                _ => None,
            })
            .expect("a For loop");
        let assigns: Vec<&ScalarStmt> = for_body
            .iter()
            .filter(|s| matches!(s, ScalarStmt::Assign { .. }))
            .collect();
        assert_eq!(
            assigns.len(),
            2,
            "expected 2 simultaneous assigns: {:?}",
            for_body
        );
        // they must be the LAST two statements (nothing reads a partially-updated acc).
        assert!(
            matches!(for_body[for_body.len() - 2], ScalarStmt::Assign { .. })
                && matches!(for_body[for_body.len() - 1], ScalarStmt::Assign { .. }),
            "the N assigns must be the final statements (simultaneous update): {:?}",
            for_body,
        );
        // next_a = b: an Assign whose value is a bare Var (the other accumulator).
        assert!(
            assigns.iter().any(|s| matches!(
                s,
                ScalarStmt::Assign {
                    value: ScalarExpr::Var(_),
                    ..
                }
            )),
            "next_a should be `= b` (a bare accumulator Var): {:?}",
            assigns,
        );
        // result is the second component (b).
        assert!(matches!(&k.outputs[0], ScalarExpr::Var(n) if *n == accs[1]));
    }

    // ---- iter_row_major ----

    #[test]
    fn row_major_rank_0_is_single_empty() {
        // explicit element type: serde_json's `PartialEq<Value> for usize` would
        // otherwise leave `vec![vec![]]`'s inner type ambiguous.
        let expected: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(iter_row_major(&[]), expected);
    }

    #[test]
    fn row_major_rank_1() {
        assert_eq!(iter_row_major(&[3]), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn row_major_rank_2_increments_inner_first() {
        let r = iter_row_major(&[2, 3]);
        assert_eq!(
            r,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ]
        );
    }

    #[test]
    fn row_major_rank_3() {
        let r = iter_row_major(&[2, 2, 2]);
        assert_eq!(r.len(), 8);
        // inner-most increments first
        assert_eq!(r[0], vec![0, 0, 0]);
        assert_eq!(r[1], vec![0, 0, 1]);
        assert_eq!(r[2], vec![0, 1, 0]);
        assert_eq!(r[4], vec![1, 0, 0]);
    }

    // ---- Const lowering ----

    #[test]
    fn const_f64_is_one_scalar_expr() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(2.5), None);
        let l = scalarize(&g, a, "test");
        assert_eq!(l.params.len(), 0);
        assert_eq!(l.body.len(), 0);
        assert_eq!(l.results.len(), 1);
        assert!(matches!(
            l.results[0],
            ScalarExpr::Const(ConstValue::F64(_))
        ));
        assert_eq!(l.result_element, ElementTy::F64);
        assert!(l.result_shape.is_empty());
    }

    #[test]
    fn const_bool_round_trips_value() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::Bool(true), None);
        let l = scalarize(&g, a, "t");
        if let ScalarExpr::Const(ConstValue::Bool(b)) = &l.results[0] {
            assert!(*b);
        } else {
            panic!("expected Bool const, got {:?}", l.results[0]);
        }
    }

    // ---- Param lowering: rank-0 ----

    #[test]
    fn rank_0_param_emits_single_var() {
        let mut g = Graph::new();
        let p = g.add_scalar_param("x", ElementTy::F64);
        let l = scalarize(&g, p, "f");
        assert_eq!(l.params.len(), 1);
        assert_eq!(l.params[0].name, "x");
        assert_eq!(l.params[0].element, ElementTy::F64);
        assert_eq!(l.results.len(), 1);
        assert!(matches!(&l.results[0], ScalarExpr::Var(n) if n == "x"));
    }

    // ---- Param lowering: rank-N ----

    #[test]
    fn rank_1_param_expands_to_n_components() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let l = scalarize(&g, v, "f");
        assert_eq!(l.params.len(), 3);
        assert_eq!(l.params[0].name, "v_0");
        assert_eq!(l.params[1].name, "v_1");
        assert_eq!(l.params[2].name, "v_2");
        assert_eq!(l.results.len(), 3);
    }

    #[test]
    fn rank_2_param_expands_row_major() {
        // matrix M of shape [2, 3] -> 6 scalar params, row-major.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let l = scalarize(&g, m, "f");
        assert_eq!(l.params.len(), 6);
        let names: Vec<&str> = l.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["M_0_0", "M_0_1", "M_0_2", "M_1_0", "M_1_1", "M_1_2"]
        );
    }

    #[test]
    fn rank_1_param_variance_is_irrelevant_to_naming() {
        // upper / lower / untagged all produce the same scalar fan-out;
        // variance information is metadata, not emitted into target source.
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]).with_variance(VarianceTag::Upper),
            None,
        );
        let l = scalarize(&g, v, "f");
        assert_eq!(l.params.len(), 2);
        assert_eq!(l.params[0].name, "v_0");
        assert_eq!(l.params[1].name, "v_1");
    }

    // ---- R.5.a: rank-1 generic-dim param now produces an Array binding ----

    #[test]
    fn rank_1_generic_param_produces_array_param() {
        // After R.5.a, this no longer panics. It produces a single array
        // LoweredParam with array_len = Some(Generic("D")).
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![DimExpr::generic("D")]),
            None,
        );
        // need an output node; just return the param via an identity einsum-like op.
        // simplest: use the param node directly as the output even though it's rank-1.
        // since our output extraction expects Concrete, we wrap in a no-op einsum that
        // reduces it to rank-0... actually, easiest: try to use the param itself as output.
        // it'll panic with the "Array binding output" message — which is OK for V1.
        // for testing, instead build a 1-param dot product to validate the array path.
        let _ = v;
        // the actual loop-form path is tested via dot_product_with_generic_dim below.
    }

    #[test]
    #[should_panic(expected = "does not yet support const-generic dim")]
    fn rank_2_generic_panics_in_scalarization() {
        // V1 R.5.a only supports rank-1 generic params. rank-2 generic
        // (e.g., matrix with generic dims) still panics.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(
                ElementTy::F64,
                vec![DimExpr::generic("D"), DimExpr::generic("D")],
            ),
            None,
        );
        // wrap in an einsum to force lowering
        let r = g.einsum("ij,ij->", vec![m, m], None);
        let _ = scalarize(&g, r, "f");
    }

    #[test]
    fn dot_product_with_generic_dim_emits_loop_form() {
        let mut g = Graph::new();
        let a = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![DimExpr::generic("D")]),
            None,
        );
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![DimExpr::generic("D")]),
            None,
        );
        let r = g.einsum("i,i->", vec![a, b], None);
        let f = scalarize(&g, r, "dotd");
        // single rank-0 result referencing the accumulator
        assert_eq!(f.results.len(), 1);
        // params should be two arrays
        assert_eq!(f.params.len(), 2);
        assert!(matches!(&f.params[0].array_len, Some(DimExpr::Generic(_))));
        assert!(matches!(&f.params[1].array_len, Some(DimExpr::Generic(_))));
        // body should contain a LetMut accumulator, a For loop, and the
        // loop body should contain a CompoundAssign.
        assert!(
            f.body
                .iter()
                .any(|s| matches!(s, ScalarStmt::LetMut { .. }))
        );
        assert!(f.body.iter().any(|s| matches!(s, ScalarStmt::For { .. })));
        if let ScalarStmt::For { body, .. } = &f.body[1] {
            assert!(body.iter().any(|s| matches!(
                s,
                ScalarStmt::CompoundAssign {
                    op: BinaryKind::Add,
                    ..
                }
            )));
        } else {
            panic!("expected For at body[1]");
        }
    }

    // ---- R.3.b: ElementWise scalarization ----

    use crate::ElementWiseOp;
    use crate::TranscendentalOp;

    #[test]
    fn scalar_add_two_consts() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(2.0), None);
        let b = g.add_const(ConstValue::F64(3.0), None);
        let s = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let l = scalarize(&g, s, "f");
        assert_eq!(l.results.len(), 1);
        match &l.results[0] {
            ScalarExpr::BinOp(BinaryKind::Add, lhs, rhs) => {
                assert!(matches!(**lhs, ScalarExpr::Const(ConstValue::F64(_))));
                assert!(matches!(**rhs, ScalarExpr::Const(ConstValue::F64(_))));
            }
            other => panic!("expected BinOp(Add, ..), got {:?}", other),
        }
    }

    #[test]
    fn scalar_add_two_params() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let s = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let l = scalarize(&g, s, "f");
        // a + b
        if let ScalarExpr::BinOp(BinaryKind::Add, lhs, rhs) = &l.results[0] {
            assert!(matches!(&**lhs, ScalarExpr::Var(n) if n == "a"));
            assert!(matches!(&**rhs, ScalarExpr::Var(n) if n == "b"));
        } else {
            panic!("expected BinOp, got {:?}", l.results[0]);
        }
    }

    #[test]
    fn vector_add_pairs_components() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let w = g.add_param(
            Symbol::intern("w"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let s = g.element_wise(ElementWiseOp::Add, vec![v, w], None);
        let l = scalarize(&g, s, "f");
        assert_eq!(l.results.len(), 3);
        // each result_i = v_i + w_i
        for (i, e) in l.results.iter().enumerate() {
            if let ScalarExpr::BinOp(BinaryKind::Add, lhs, rhs) = e {
                let v_i = format!("v_{}", i);
                let w_i = format!("w_{}", i);
                assert!(
                    matches!(&**lhs, ScalarExpr::Var(n) if *n == v_i),
                    "lhs[{}]: {:?}",
                    i,
                    lhs
                );
                assert!(
                    matches!(&**rhs, ScalarExpr::Var(n) if *n == w_i),
                    "rhs[{}]: {:?}",
                    i,
                    rhs
                );
            } else {
                panic!("expected BinOp at {}, got {:?}", i, e);
            }
        }
    }

    #[test]
    fn broadcast_scalar_times_vector() {
        // s * v where s is scalar, v is [3]. each output_i = s * v_i.
        let mut g = Graph::new();
        let s = g.add_scalar_param("s", ElementTy::F64);
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let r = g.element_wise(ElementWiseOp::Mul, vec![s, v], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 3);
        for (i, e) in l.results.iter().enumerate() {
            if let ScalarExpr::BinOp(BinaryKind::Mul, lhs, rhs) = e {
                assert!(matches!(&**lhs, ScalarExpr::Var(n) if n == "s"));
                let v_i = format!("v_{}", i);
                assert!(matches!(&**rhs, ScalarExpr::Var(n) if *n == v_i));
            } else {
                panic!("expected BinOp(Mul) at {}, got {:?}", i, e);
            }
        }
    }

    #[test]
    fn broadcast_with_size_one_axis() {
        // [1, 3] + [4, 3] -> [4, 3]. each output[i, j] = a[0, j] + b[i, j].
        let mut g = Graph::new();
        let a = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(1), lit(3)]),
            None,
        );
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(4), lit(3)]),
            None,
        );
        let r = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 12);
        // verify result[i, j] = a_0_j + b_i_j by spot-checking i=2, j=1.
        let out_idx = 2 * 3 + 1;
        if let ScalarExpr::BinOp(BinaryKind::Add, lhs, rhs) = &l.results[out_idx] {
            assert!(matches!(&**lhs, ScalarExpr::Var(n) if n == "a_0_1"));
            assert!(matches!(&**rhs, ScalarExpr::Var(n) if n == "b_2_1"));
        } else {
            panic!("expected BinOp(Add), got {:?}", l.results[out_idx]);
        }
    }

    #[test]
    fn unary_neg_per_component() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]),
            None,
        );
        let n = g.element_wise(ElementWiseOp::Neg, vec![v], None);
        let l = scalarize(&g, n, "f");
        assert_eq!(l.results.len(), 2);
        for (i, e) in l.results.iter().enumerate() {
            if let ScalarExpr::UnaryOp(UnaryKind::Neg, inner) = e {
                let v_i = format!("v_{}", i);
                assert!(matches!(&**inner, ScalarExpr::Var(n) if *n == v_i));
            } else {
                panic!("expected UnaryOp(Neg) at {}, got {:?}", i, e);
            }
        }
    }

    #[test]
    fn method_unary_for_abs() {
        // Abs stays a method call (`.abs()`); the ternary `x<0?-x:x` semantics
        // live in each backend (cuda special-case, interp/jit, Numeric carrier).
        let mut g = Graph::new();
        let v = g.add_scalar_param("v", ElementTy::F64);
        let a = g.element_wise(ElementWiseOp::Abs, vec![v], None);
        let l = scalarize(&g, a, "f");
        if let ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } = &l.results[0]
        {
            assert!(matches!(&**receiver, ScalarExpr::Var(n) if n == "v"));
            assert_eq!(*method, "abs");
            assert!(args.is_empty());
        } else {
            panic!("expected MethodCall(abs), got {:?}", l.results[0]);
        }
    }

    #[test]
    fn method_binary_for_min() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let m = g.element_wise(ElementWiseOp::Min, vec![a, b], None);
        let l = scalarize(&g, m, "f");
        if let ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } = &l.results[0]
        {
            assert!(matches!(&**receiver, ScalarExpr::Var(n) if n == "a"));
            assert_eq!(*method, "min");
            assert_eq!(args.len(), 1);
            assert!(matches!(&args[0], ScalarExpr::Var(n) if n == "b"));
        } else {
            panic!("expected MethodCall(min), got {:?}", l.results[0]);
        }
    }

    #[test]
    fn comparison_emits_bool_binop() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let lt = g.element_wise(ElementWiseOp::Lt, vec![a, b], None);
        let l = scalarize(&g, lt, "f");
        assert_eq!(l.result_element, ElementTy::Bool);
        assert!(matches!(
            &l.results[0],
            ScalarExpr::BinOp(BinaryKind::Lt, _, _)
        ));
    }

    #[test]
    fn is_nan_emits_method_call() {
        let mut g = Graph::new();
        let v = g.add_scalar_param("v", ElementTy::F64);
        let n = g.element_wise(ElementWiseOp::IsNaN, vec![v], None);
        let l = scalarize(&g, n, "f");
        if let ScalarExpr::MethodCall { method, .. } = &l.results[0] {
            assert_eq!(*method, "is_nan");
        } else {
            panic!("expected is_nan method call");
        }
    }

    // ---- R.3.b: Transcendental scalarization ----

    #[test]
    fn transcendental_sin_emits_method_call() {
        let mut g = Graph::new();
        let v = g.add_scalar_param("v", ElementTy::F64);
        let s = g.transcendental(TranscendentalOp::Sin, vec![v], None);
        let l = scalarize(&g, s, "f");
        if let ScalarExpr::MethodCall {
            method,
            receiver,
            args,
        } = &l.results[0]
        {
            assert_eq!(*method, "sin");
            assert!(matches!(&**receiver, ScalarExpr::Var(n) if n == "v"));
            assert!(args.is_empty());
        } else {
            panic!("expected MethodCall(sin), got {:?}", l.results[0]);
        }
    }

    #[test]
    fn transcendental_log_emits_ln_method() {
        // Rust f64::log is base-N; ::ln is natural log. the Log variant
        // maps to .ln per convention.
        let mut g = Graph::new();
        let v = g.add_scalar_param("v", ElementTy::F64);
        let r = g.transcendental(TranscendentalOp::Log, vec![v], None);
        let l = scalarize(&g, r, "f");
        if let ScalarExpr::MethodCall { method, .. } = &l.results[0] {
            assert_eq!(*method, "ln");
        } else {
            panic!("expected ln method");
        }
    }

    #[test]
    fn transcendental_pow_emits_powf_with_arg() {
        let mut g = Graph::new();
        let b = g.add_scalar_param("b", ElementTy::F64);
        let e = g.add_scalar_param("e", ElementTy::F64);
        let r = g.transcendental(TranscendentalOp::Pow, vec![b, e], None);
        let l = scalarize(&g, r, "f");
        if let ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } = &l.results[0]
        {
            assert!(matches!(&**receiver, ScalarExpr::Var(n) if n == "b"));
            assert_eq!(*method, "powf");
            assert_eq!(args.len(), 1);
            assert!(matches!(&args[0], ScalarExpr::Var(n) if n == "e"));
        } else {
            panic!("expected powf method call");
        }
    }

    #[test]
    fn transcendental_atan2_emits_method_with_arg() {
        let mut g = Graph::new();
        let y = g.add_scalar_param("y", ElementTy::F64);
        let x = g.add_scalar_param("x", ElementTy::F64);
        let r = g.transcendental(TranscendentalOp::Atan2, vec![y, x], None);
        let l = scalarize(&g, r, "f");
        if let ScalarExpr::MethodCall { method, .. } = &l.results[0] {
            assert_eq!(*method, "atan2");
        } else {
            panic!("expected atan2 method call");
        }
    }

    #[test]
    fn transcendental_over_vector_unrolls() {
        // each element gets its own .sin() call.
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let s = g.transcendental(TranscendentalOp::Cos, vec![v], None);
        let l = scalarize(&g, s, "f");
        assert_eq!(l.results.len(), 3);
        for (i, e) in l.results.iter().enumerate() {
            if let ScalarExpr::MethodCall {
                receiver, method, ..
            } = e
            {
                assert_eq!(*method, "cos");
                let v_i = format!("v_{}", i);
                assert!(matches!(&**receiver, ScalarExpr::Var(n) if *n == v_i));
            } else {
                panic!("expected MethodCall(cos) at {}", i);
            }
        }
    }

    // ---- BinaryKind / UnaryKind metadata ----

    #[test]
    fn binary_kind_rust_operators() {
        assert_eq!(BinaryKind::Add.rust_operator(), "+");
        assert_eq!(BinaryKind::Eq.rust_operator(), "==");
        assert_eq!(BinaryKind::Le.rust_operator(), "<=");
    }

    #[test]
    fn unary_kind_rust_operator() {
        assert_eq!(UnaryKind::Neg.rust_operator(), "-");
    }

    // ---- R.3.c: Construct ----

    #[test]
    fn construct_three_scalars_into_rank_1() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::F64(1.0), None);
        let b = g.add_const(ConstValue::F64(2.0), None);
        let c = g.add_const(ConstValue::F64(3.0), None);
        let v = g.construct(vec![a, b, c], None);
        let l = scalarize(&g, v, "f");
        assert_eq!(l.results.len(), 3);
        assert!(matches!(
            &l.results[0],
            ScalarExpr::Const(ConstValue::F64(_))
        ));
    }

    #[test]
    fn construct_two_vectors_into_matrix() {
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
        let l = scalarize(&g, m, "f");
        assert_eq!(l.results.len(), 6); // 2x3 matrix
        // verify row-major layout: result[0..3] from a, result[3..6] from b
        if let ScalarExpr::Var(n) = &l.results[0] {
            assert_eq!(n, "a_0");
        } else {
            panic!()
        }
        if let ScalarExpr::Var(n) = &l.results[2] {
            assert_eq!(n, "a_2");
        } else {
            panic!()
        }
        if let ScalarExpr::Var(n) = &l.results[3] {
            assert_eq!(n, "b_0");
        } else {
            panic!()
        }
        if let ScalarExpr::Var(n) = &l.results[5] {
            assert_eq!(n, "b_2");
        } else {
            panic!()
        }
    }

    // ---- R.3.c: Index ----

    #[test]
    fn index_extracts_correct_scalar() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let s = g.index(v, vec![DimIndex::Literal(1)], None);
        let l = scalarize(&g, s, "f");
        assert_eq!(l.results.len(), 1);
        if let ScalarExpr::Var(n) = &l.results[0] {
            assert_eq!(n, "v_1");
        } else {
            panic!("expected Var(v_1), got {:?}", l.results[0]);
        }
    }

    #[test]
    fn index_matrix_picks_row_major_element() {
        // M[1, 2] of a [3, 4] matrix -> flat index = 1*4 + 2 = 6.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(4)]),
            None,
        );
        let s = g.index(m, vec![DimIndex::Literal(1), DimIndex::Literal(2)], None);
        let l = scalarize(&g, s, "f");
        if let ScalarExpr::Var(n) = &l.results[0] {
            assert_eq!(n, "M_1_2");
        } else {
            panic!("expected M_1_2");
        }
    }

    #[test]
    #[should_panic(expected = "DimIndex::Generic")]
    fn index_generic_panics_in_v1() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let s = g.index(v, vec![DimIndex::Generic(Symbol::intern("ii"))], None);
        let _ = scalarize(&g, s, "f");
    }

    // ---- R.3.c: Broadcast ----

    #[test]
    fn broadcast_scalar_replicates() {
        let mut g = Graph::new();
        let s = g.add_const(ConstValue::F64(2.5), None);
        let v = g.broadcast(s, vec![lit(3)], None);
        let l = scalarize(&g, v, "f");
        assert_eq!(l.results.len(), 3);
        // all three components should be the same Const expression.
        for r in &l.results {
            assert!(matches!(r, ScalarExpr::Const(ConstValue::F64(_))));
        }
    }

    #[test]
    fn broadcast_size_one_axis_tiles() {
        // [1, 3] -> [4, 3]: each row gets the same 3 source values.
        let mut g = Graph::new();
        let row = g.add_param(
            Symbol::intern("r"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(1), lit(3)]),
            None,
        );
        let m = g.broadcast(row, vec![lit(4), lit(3)], None);
        let l = scalarize(&g, m, "f");
        assert_eq!(l.results.len(), 12);
        // every M[i, j] should be r_0_j (broadcast across i).
        for i in 0..4 {
            for j in 0..3 {
                let pos = i * 3 + j;
                if let ScalarExpr::Var(n) = &l.results[pos] {
                    let expected = format!("r_0_{}", j);
                    assert_eq!(n, &expected, "at [{}, {}]: {} vs {}", i, j, n, expected);
                } else {
                    panic!("expected Var");
                }
            }
        }
    }

    #[test]
    fn broadcast_rank_1_to_rank_2_replicates_along_leading() {
        // [3] -> [4, 3]: leading axis broadcast.
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let m = g.broadcast(v, vec![lit(4), lit(3)], None);
        let l = scalarize(&g, m, "f");
        assert_eq!(l.results.len(), 12);
        // M[i, j] = v_j for all i.
        for i in 0..4 {
            for j in 0..3 {
                let pos = i * 3 + j;
                if let ScalarExpr::Var(n) = &l.results[pos] {
                    let expected = format!("v_{}", j);
                    assert_eq!(n, &expected);
                } else {
                    panic!();
                }
            }
        }
    }

    // ---- R.3.d: Reduce ----

    use crate::ReduceOp;

    #[test]
    fn reduce_max_over_vector_to_scalar() {
        // max-reduce a [3] vector -> rank-0. result = v_0.max(v_1).max(v_2).
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let r = g.reduce(ReduceOp::Max, vec![0], v, None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 1);
        // result is acc.max(v_2) where acc is v_0.max(v_1).
        if let ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } = &l.results[0]
        {
            assert_eq!(method, "max");
            assert_eq!(args.len(), 1);
            // arg is v_2
            assert!(matches!(&args[0], ScalarExpr::Var(n) if n == "v_2"));
            // receiver is v_0.max(v_1)
            if let ScalarExpr::MethodCall {
                receiver: r2,
                method: m2,
                args: a2,
            } = &**receiver
            {
                assert_eq!(m2, "max");
                assert!(matches!(&**r2, ScalarExpr::Var(n) if n == "v_0"));
                assert!(matches!(&a2[0], ScalarExpr::Var(n) if n == "v_1"));
            } else {
                panic!("expected nested .max chain, got {:?}", receiver);
            }
        } else {
            panic!("expected MethodCall(max)");
        }
    }

    #[test]
    fn reduce_min_over_inner_axis_keeps_outer() {
        // [3, 4] reduce axis 1 -> [3]. each result[i] = M_i_0.min(M_i_1)...
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(4)]),
            None,
        );
        let r = g.reduce(ReduceOp::Min, vec![1], m, None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 3);
        // result[2] should be a chain of .min over M_2_0..M_2_3.
        // verify by walking down the chain and collecting the args.
        let mut visited: Vec<String> = vec![];
        let mut cur = &l.results[2];
        while let ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } = cur
        {
            assert_eq!(method, "min");
            if let ScalarExpr::Var(n) = &args[0] {
                visited.push(n.clone());
            }
            cur = &**receiver;
        }
        if let ScalarExpr::Var(n) = cur {
            visited.push(n.clone());
        }
        // visited captured (in reverse-chain order): [M_2_3, M_2_2, M_2_1, M_2_0]
        visited.reverse();
        assert_eq!(visited, vec!["M_2_0", "M_2_1", "M_2_2", "M_2_3"]);
    }

    #[test]
    fn reduce_full_collapse_to_scalar() {
        // [2, 3] reduce axes 0+1 -> scalar.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let r = g.reduce(ReduceOp::Max, vec![0, 1], m, None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 1);
        // count the depth of the chain — should be 6 method calls + 1 leaf var.
        let mut count = 0;
        let mut cur = &l.results[0];
        while let ScalarExpr::MethodCall {
            receiver, method, ..
        } = cur
        {
            assert_eq!(*method, "max");
            count += 1;
            cur = &**receiver;
        }
        assert!(matches!(cur, ScalarExpr::Var(_)));
        assert_eq!(count, 5, "expected 5 .max chains for 6 elements");
    }

    #[test]
    fn reduce_or_uses_binary_bitor() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::Bool, vec![lit(3)]),
            None,
        );
        let r = g.reduce(ReduceOp::Or, vec![0], v, None);
        let l = scalarize(&g, r, "f");
        // outer: BinOp(BitOr, BinOp(BitOr, v_0, v_1), v_2)
        if let ScalarExpr::BinOp(BinaryKind::BitOr, _, _) = &l.results[0] {
            // ok
        } else {
            panic!("expected BinOp(BitOr), got {:?}", l.results[0]);
        }
    }

    #[test]
    fn reduce_and_uses_binary_bitand() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::Bool, vec![lit(2)]),
            None,
        );
        let r = g.reduce(ReduceOp::And, vec![0], v, None);
        let l = scalarize(&g, r, "f");
        assert!(matches!(
            &l.results[0],
            ScalarExpr::BinOp(BinaryKind::BitAnd, _, _)
        ));
    }

    #[test]
    fn reduce_xor_uses_binary_bitxor() {
        let mut g = Graph::new();
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::I32, vec![lit(2)]),
            None,
        );
        let r = g.reduce(ReduceOp::Xor, vec![0], v, None);
        let l = scalarize(&g, r, "f");
        assert!(matches!(
            &l.results[0],
            ScalarExpr::BinOp(BinaryKind::BitXor, _, _)
        ));
    }

    // ---- R.3.d: Select ----

    #[test]
    fn select_scalar_picks_branch() {
        let mut g = Graph::new();
        let c = g.add_scalar_param("c", ElementTy::Bool);
        let t = g.add_scalar_param("t", ElementTy::F64);
        let e = g.add_scalar_param("e", ElementTy::F64);
        let r = g.select(c, t, e, None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 1);
        if let ScalarExpr::Select { cond, then, else_ } = &l.results[0] {
            assert!(matches!(&**cond, ScalarExpr::Var(n) if n == "c"));
            assert!(matches!(&**then, ScalarExpr::Var(n) if n == "t"));
            assert!(matches!(&**else_, ScalarExpr::Var(n) if n == "e"));
        } else {
            panic!("expected Select, got {:?}", l.results[0]);
        }
    }

    #[test]
    fn select_broadcasts_scalar_cond_with_vector_branches() {
        // c: scalar Bool, t: [3], e: [3] -> result [3], all using the same cond.
        let mut g = Graph::new();
        let c = g.add_scalar_param("c", ElementTy::Bool);
        let t = g.add_param(
            Symbol::intern("t"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let e = g.add_param(
            Symbol::intern("e"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let r = g.select(c, t, e, None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 3);
        for (i, expr) in l.results.iter().enumerate() {
            if let ScalarExpr::Select { cond, then, else_ } = expr {
                assert!(matches!(&**cond, ScalarExpr::Var(n) if n == "c"));
                let ti = format!("t_{}", i);
                let ei = format!("e_{}", i);
                assert!(matches!(&**then, ScalarExpr::Var(n) if *n == ti));
                assert!(matches!(&**else_, ScalarExpr::Var(n) if *n == ei));
            } else {
                panic!("expected Select at {}", i);
            }
        }
    }

    // ---- R.3.e: Einsum scalarization ----

    /// collect the leaf variable names referenced in a ScalarExpr (DFS).
    fn collect_vars(e: &ScalarExpr, out: &mut Vec<String>) {
        match e {
            ScalarExpr::Const(_) => {}
            ScalarExpr::Var(n) => out.push(n.clone()),
            ScalarExpr::BinOp(_, a, b) => {
                collect_vars(a, out);
                collect_vars(b, out);
            }
            ScalarExpr::UnaryOp(_, a) => collect_vars(a, out),
            ScalarExpr::MethodCall { receiver, args, .. } => {
                collect_vars(receiver, out);
                for a in args {
                    collect_vars(a, out);
                }
            }
            ScalarExpr::Select { cond, then, else_ } => {
                collect_vars(cond, out);
                collect_vars(then, out);
                collect_vars(else_, out);
            }
            ScalarExpr::IndexInto { container, index } => {
                out.push(container.clone());
                collect_vars(index, out);
            }
            ScalarExpr::FieldLoadAt {
                field_key,
                components,
            } => {
                out.push(field_key.clone());
                for c in components {
                    collect_vars(c, out);
                }
            }
            ScalarExpr::FreeCall { name, args } => {
                out.push(name.clone());
                for a in args {
                    collect_vars(a, out);
                }
            }
            ScalarExpr::Cast { value, .. } => collect_vars(value, out),
        }
    }

    /// count how many times `BinaryKind::op` appears in `e`.
    fn count_binop(e: &ScalarExpr, kind: BinaryKind) -> usize {
        match e {
            ScalarExpr::BinOp(k, a, b) => {
                let here = if *k == kind { 1 } else { 0 };
                here + count_binop(a, kind) + count_binop(b, kind)
            }
            ScalarExpr::UnaryOp(_, a) => count_binop(a, kind),
            ScalarExpr::MethodCall { receiver, args, .. } => {
                let mut c = count_binop(receiver, kind);
                for a in args {
                    c += count_binop(a, kind);
                }
                c
            }
            ScalarExpr::Select { cond, then, else_ } => {
                count_binop(cond, kind) + count_binop(then, kind) + count_binop(else_, kind)
            }
            _ => 0,
        }
    }

    #[test]
    fn einsum_dot_product_unrolls() {
        // i,i-> with N=3 yields  a_0*b_0 + a_1*b_1 + a_2*b_2.
        let mut g = Graph::new();
        let a = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let r = g.einsum("i,i->", vec![a, b], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 1);
        // 3 multiplies + 2 adds.
        assert_eq!(count_binop(&l.results[0], BinaryKind::Mul), 3);
        assert_eq!(count_binop(&l.results[0], BinaryKind::Add), 2);
        // refs in order: a_0, b_0, a_1, b_1, a_2, b_2.
        let mut vars = vec![];
        collect_vars(&l.results[0], &mut vars);
        assert_eq!(vars, vec!["a_0", "b_0", "a_1", "b_1", "a_2", "b_2"]);
    }

    #[test]
    fn einsum_matmul_unrolls() {
        // ij,jk->ik with [2,3]·[3,2] -> [2,2]. each output cell sums 3 products.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let n = g.add_param(
            Symbol::intern("N"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(2)]),
            None,
        );
        let r = g.einsum("ij,jk->ik", vec![m, n], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 4);
        // each result cell is sum of 3 products.
        for cell in &l.results {
            assert_eq!(count_binop(cell, BinaryKind::Mul), 3);
            assert_eq!(count_binop(cell, BinaryKind::Add), 2);
        }
        // result[0,0] = M_0_0*N_0_0 + M_0_1*N_1_0 + M_0_2*N_2_0
        let mut vars = vec![];
        collect_vars(&l.results[0], &mut vars);
        assert_eq!(
            vars,
            vec!["M_0_0", "N_0_0", "M_0_1", "N_1_0", "M_0_2", "N_2_0"]
        );
    }

    #[test]
    fn einsum_trace_unrolls() {
        // ii-> on [3,3] = M_0_0 + M_1_1 + M_2_2.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(3)]),
            None,
        );
        let r = g.einsum("ii->", vec![m], None);
        let l = scalarize(&g, r, "f");
        // trace has no multiplies (only one input, single factor per term).
        // 3 leaf vars, 2 adds.
        assert_eq!(count_binop(&l.results[0], BinaryKind::Add), 2);
        let mut vars = vec![];
        collect_vars(&l.results[0], &mut vars);
        assert_eq!(vars, vec!["M_0_0", "M_1_1", "M_2_2"]);
    }

    #[test]
    fn einsum_bilinear_form_unrolls() {
        // ij,i,j-> with [2,2] M, [2] a, [2] b: result = sum_{i,j} M_i_j * a_i * b_j.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(2)]),
            None,
        );
        let a = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]),
            None,
        );
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]),
            None,
        );
        let r = g.einsum("ij,i,j->", vec![m, a, b], None);
        let l = scalarize(&g, r, "f");
        // 4 product groups, each has 3 leaves * 2 mults = 8 mults total; 3 adds.
        assert_eq!(count_binop(&l.results[0], BinaryKind::Mul), 8);
        assert_eq!(count_binop(&l.results[0], BinaryKind::Add), 3);
        let mut vars = vec![];
        collect_vars(&l.results[0], &mut vars);
        // expected order: (i=0, j=0), (i=0, j=1), (i=1, j=0), (i=1, j=1).
        // for each cell: M_i_j, a_i, b_j.
        assert_eq!(
            vars,
            vec![
                "M_0_0", "a_0", "b_0", "M_0_1", "a_0", "b_1", "M_1_0", "a_1", "b_0", "M_1_1",
                "a_1", "b_1",
            ]
        );
    }

    #[test]
    fn einsum_outer_product_unrolls() {
        // i,j->ij with [2] a, [3] b -> [2,3] matrix; no contraction.
        let mut g = Graph::new();
        let a = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]),
            None,
        );
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let r = g.einsum("i,j->ij", vec![a, b], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 6);
        // each cell = a_i * b_j with no addition.
        for cell in &l.results {
            assert_eq!(count_binop(cell, BinaryKind::Mul), 1);
            assert_eq!(count_binop(cell, BinaryKind::Add), 0);
        }
        // result[1, 2] should be a_1 * b_2.
        let mut vars = vec![];
        collect_vars(&l.results[1 * 3 + 2], &mut vars);
        assert_eq!(vars, vec!["a_1", "b_2"]);
    }

    #[test]
    fn einsum_matvec_unrolls() {
        // ij,j->i with [2,3] M, [3] v -> [2]; each result_i = sum_j M_i_j * v_j.
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let r = g.einsum("ij,j->i", vec![m, v], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 2);
        // each row: 3 muls + 2 adds.
        for cell in &l.results {
            assert_eq!(count_binop(cell, BinaryKind::Mul), 3);
            assert_eq!(count_binop(cell, BinaryKind::Add), 2);
        }
        let mut vars = vec![];
        collect_vars(&l.results[1], &mut vars);
        assert_eq!(vars, vec!["M_1_0", "v_0", "M_1_1", "v_1", "M_1_2", "v_2"]);
    }

    #[test]
    fn einsum_batched_dot_unrolls() {
        // ...i,...i->...  with [2,3] and [2,3] -> [2]. each batch index gets
        // its own dot product.
        let mut g = Graph::new();
        let a = g.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let b = g.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(3)]),
            None,
        );
        let r = g.einsum("...i,...i->...", vec![a, b], None);
        let l = scalarize(&g, r, "f");
        assert_eq!(l.results.len(), 2);
        // batch 0 should reference a_0_0, b_0_0, a_0_1, b_0_1, a_0_2, b_0_2.
        let mut vars0 = vec![];
        collect_vars(&l.results[0], &mut vars0);
        assert_eq!(
            vars0,
            vec!["a_0_0", "b_0_0", "a_0_1", "b_0_1", "a_0_2", "b_0_2"]
        );
        // batch 1 uses the _1_ slice.
        let mut vars1 = vec![];
        collect_vars(&l.results[1], &mut vars1);
        assert_eq!(
            vars1,
            vec!["a_1_0", "b_1_0", "a_1_1", "b_1_1", "a_1_2", "b_1_2"]
        );
    }
}
