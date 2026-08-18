// =============================================================================
// lower.rs
//
// scalarization pass: tensor IR -> a "lowered" form (LoweredFn) that
// per-backend emitters (CPU, CUDA) turn into target source.
//
// the lowered form is a sequence of scalar let-statements followed by
// one or more output scalar expressions (one per scalar component of
// the output tensor). it is intentionally simpler than the existing
// scalar IR — no graph, no node IDs, just a list of bindings + a list
// of return values. every tensor dim must be a Literal: const-generic dim
// support (loops that survive into target source) is unimplemented, and a
// generic dim triggers a panic.
// =============================================================================

use std::collections::{HashMap, HashSet};

use crate::graph::{
    ConstValue, DimIndex, ElementWiseOp, Graph, NodeId, Op, ReduceOp,
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
    /// bitwise complement on int, logical not on bool. Rust's `!` operator
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
    /// is serde-deserializable — the serialized IR is the durable artifact;
    /// construction sites pass a literal `.to_string()`.
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
    /// free-function call by name with scalar args. emit lowers
    /// as `name(arg0, arg1, ...)` on both CPU and CUDA targets. the
    /// function definition lives outside this elemental — either a
    /// scalar elemental's `_cuda` accessor or a host function on the
    /// CPU path.
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
    /// the immediate sub-expressions this node owns, in evaluation order. the single source for
    /// child enumeration used by every walk / transform pass (cse var
    /// collect, free-call scan, var-use test, FieldLoadAt rewrite) recurses through this uniform
    /// accessor, with no per-variant re-matching. the emit backends (cpu/cuda/interp/jit) still match
    /// per-variant — producing target source/IR is the irreducible part — exactly the split the
    /// `ScalarStmt` ssot (above) documents. adding a variant => update this once; the walks follow.
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
    /// `name = value;` — plain (non-compound) assignment.
    /// applied to a previously-declared `LetMut`. used by Op::Fold's
    /// lowering, where the body lambda returns a new accumulator that
    /// replaces (not accumulates into) the current one. CompoundAssign
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
    /// this is the IR primitive that lets the renderer
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
    /// the CSE pass treats the scope as a **hoisting barrier**:
    /// CSE candidates whose uses are all inside the
    /// scope stay inside; candidates whose uses cross the boundary get
    /// hoisted to the lca scope of all use sites.
    ///
    /// the CSE pass passes scopes through unchanged (treats them as ordinary
    /// statement containers); lca-aware placement is not applied.
    Scope {
        name: String,
        element: ElementTy,
        body: Vec<ScalarStmt>,
        result: ScalarExpr,
    },
    /// the dual of the `For`/`Break` iterate lowering: a real data-dependent
    /// branch where only the taken arm executes. lowered from `Op::IfElse`
    /// (`S::cond` -> 1 output, `S::cond_vec` -> N outputs). `outs` declares the
    /// N result slots in the enclosing scope; each arm body ends with one
    /// `Assign { outs[j].0, <arm result j> }` per slot, so the variant carries
    /// one immediate expr (`cond`) plus two sub-bodies — fitting the ssot walk
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
// the single source of truth for ScalarStmt structural walks.
//
// every transformation pass (cse, FieldLoadAt rewrite, uses-var detection, the
// fresh-name index scan) walks scalar statements the same way: visit the
// immediate scalar expression a stmt carries, then recurse into any child
// statement bodies. those two notions are encoded once here rather than
// respelled inline in every backend match.
//
// the four helpers below + `with_child_expr` are the one place that encodes
// expression and sub-body ownership. every walk-style
// pass derives from them. emit backends (cpu/cuda/interp) still match
// per-variant — that's the irreducible part of producing source. but the
// walk / transform passes are now one-liners.
//
// adding a new ScalarStmt variant becomes: update the enum (above), then add
// arms here (one per accessor) for the variant's exprs + bodies. the walk
// passes pick it up for free.
// =============================================================================
impl ScalarStmt {
    /// the scalar expression this statement holds directly (not nested inside a
    /// child body). every variant has at most one — Let's `value`, LetMut's
    /// `init`, Assign / CompoundAssign's `value`, If's `cond`. For / Break have
    /// no immediate expression (For's bound is a `DimExpr`).
    pub fn child_expr(&self) -> Option<&ScalarExpr> {
        match self {
            ScalarStmt::Let { value, .. } => Some(value),
            ScalarStmt::LetMut { init, .. } => Some(init),
            ScalarStmt::Assign { value, .. } => Some(value),
            ScalarStmt::CompoundAssign { value, .. } => Some(value),
            ScalarStmt::If { cond, .. } => Some(cond),
            // Scope's immediate child expression is its `result` — the value
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
    /// `For`/`If`/`Scope` own one body; `IfElse` owns two (then + else); every
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

    /// the binding name this statement introduces, if any. Let / LetMut /
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
/// it's a plain scalar. generic params are rank-1 at most.
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

/// per-node binding produced by scalarization: a fully-expanded list of
/// scalar component expressions (the literal-dim path).
#[derive(Clone, Debug)]
enum Binding {
    /// N scalar expressions in row-major order. used for literal-dim
    /// nodes and rank-0 results.
    Concrete(Vec<ScalarExpr>),
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
    /// resolves to `iter_acc[j]`.
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

    /// compute in-degree per NodeId by walking the graph
    /// once. used by `maybe_hoist_to_let` to share ScalarExpr trees
    /// across consumers — without this, each downstream reference
    /// clones the entire binding sub-tree, producing exponential
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
    /// that's non-trivial and has in-degree >= 2, hoist the expression
    /// to a Let with a fresh `__cse_<n>` name and replace the binding
    /// with `Var(name)`. subsequent consumers then clone a cheap Var
    /// standing in for the full sub-tree.
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
    /// op's input has no binding.
    fn require_concrete<'a>(&'a self, id: NodeId, _op_name: &str) -> &'a [ScalarExpr] {
        match self.bindings.get(&id) {
            Some(Binding::Concrete(v)) => v,
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
            Op::LoadAt(sym, comps) => Binding::Concrete(self.lower_load_at(sym, comps)),
            // Lambda is a callable value. it's only
            // consumed by Op::Apply, which extracts the FnDef name and
            // emits a FreeCall referencing the device function. lower
            // Lambda to a placeholder zero so any accidental reads
            // surface as obvious zeros.
            Op::Lambda(_) => {
                Binding::Concrete(vec![ScalarExpr::Const(crate::ConstValue::F64(0.0))])
            }
            // Apply lowers to a `FreeCall(name, args)` on the scalar
            // side — resolve the device-function name via `graph.fn_def`.
            Op::Apply { lambda, args } => {
                let fn_name = graph.fn_def(*lambda).name.clone();
                Binding::Concrete(self.lower_opaque_call(&fn_name, args))
            }
            // Fold lowers to a `LetMut acc = init; For i in 0..count
            // { acc = body(acc, i); }; acc`. the accumulator is rank-0; a
            // rank > 0 fold would need per-component LetMut + Assign.
            // produces a Binding to the accumulator's `Var(name)`.
            Op::Fold {
                lambda,
                init,
                count,
            } => Binding::Concrete(self.lower_fold(*lambda, *init, *count, ty, graph)),
            // the loop accumulator placeholder resolves to the
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
            // Op::Scope is body-partitioned like
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

    /// lower an Op::Fold into a `LetMut + For + Assign` sequence.
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
        assert_eq!(
            out_ty.rank, 0,
            "lower_fold: rank-0 accumulator only (got rank {})",
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
                "lower_fold: count must be a literal integer Const; got {:?}",
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

    /// lower an `Op::IterateInline` over an N-component accumulator
    /// vector to `LetMut acc_j = inits[j]; For i { <union cone> acc_j = steps[j] }`.
    /// the `cone` is the acc-dependent slice of all `steps` (id/topo order); its
    /// loop-invariant inputs are already lowered before the loop. the N assigns
    /// come after the whole cone, so the update is simultaneous (Jacobi): every
    /// `steps[j]` reads the old `acc_*`. binds the node to `Var(acc_result)`.
    /// lower an `Op::Scope` node into a
    /// `ScalarStmt::Scope` brace-block. body NodeIds are lowered in order into
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
    /// the outer body, so downstream consumers reference the scope's
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
        // lower body NodeIds in order. for each:
        //   - if it belongs to a deeper scope (scope_owner maps it to !=
        //     scope_id), skip — it is lowered on recursion into that
        //     nested scope.
        //   - if it is a nested `Op::Scope`, dispatch to `lower_scope`
        //     recursively so its body lands inside its brace block.
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
            // dispatch nested Op::Scope before lower_node — lower_node
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
        // resolve the result's expression now — bindings are populated and
        // this expression is evaluated as the scope'S tail value
        // (i.e., inside the brace block, after all the inner Lets, before the
        // closing brace).
        let result_expr = self.require_concrete(result, "Op::Scope result")[0].clone();
        // peel off the inner body's Lets — these are the scope-local temps.
        let scope_body = self.body.split_off(mark);
        // mint a fresh outer name for the scope's observable value.
        let scope_name = self.fresh("__scope");
        // emit the ScalarStmt::Scope into the outer body.
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

    /// lower an `Op::IfElse` (the dual of `lower_iterate_inline`): N outer-
    /// declared result slots plus two arm sub-bodies, each ending with one
    /// `Assign { outs[j], <arm result j> }` per slot. only the taken arm runs at
    /// render time. N=1 is scalar `S::cond`; N>1 is `S::cond_vec` (the IfElse
    /// node binds to an N-component `Concrete`, consumed via `Op::Proj`). the
    /// `cond` is computed outside the branch, so it is already lowered.
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

    /// lower one arm of an `Op::IfElse`. mirrors the cone body of
    /// `lower_iterate_inline` / `lower_scope`: mark the body length, lower the
    /// nodes this arm owns (skipping nodes evicted to the outer level or claimed
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
            // lower only the nodes this arm owns — evicted (outer-shared) nodes
            // and nodes claimed by a nested region are lowered in their own
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
        // capture result exprs before split_off (bindings still resolve here).
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
        // the mutable accumulator locals (one per component), before the loop.
        let acc_names: Vec<String> = (0..inits.len()).map(|_| self.fresh("iter_acc")).collect();
        for (j, &init) in inits.iter().enumerate() {
            let init_expr = self.require_concrete(init, "IterateInline init")[0].clone();
            self.body.push(ScalarStmt::LetMut {
                name: acc_names[j].clone(),
                element: ElementTy::F64,
                init: init_expr,
            });
        }
        // lower the union cone inside the loop (split_off captures its lets);
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
        // temp names for the simultaneous (Jacobi) update (N>1 only).
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
            // vector: capture every step into a temp (reading the old accumulators)
            // before any assign — the update must be simultaneous (Jacobi). a
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
        // early-out: if the break predicate is satisfied after the assigns, exit
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
        // only literal indices into a literal-dim tensor produce a
        // determinate flat index. Generic indices require loop emission;
        // panics.
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

    fn lower_param(&mut self, sym: &Symbol, ty: &TensorTy) -> Binding {
        let base = sym.as_str().to_string();

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
/// every dim in every tensor type must be Literal. Generic dims trigger a
/// panic; const-generic loop support is unimplemented.
pub fn scalarize(graph: &Graph, output: NodeId, name: &str) -> LoweredFn {
    let mut sc = Scalarizer::new();
    let in_degrees = Scalarizer::compute_in_degrees(graph);
    for (id, node, ty) in graph.iter() {
        sc.lower_node(id, &node.op, ty, graph);
        sc.maybe_hoist_to_let(id, &in_degrees, ty);
    }
    let results = match sc.bindings.get(&output) {
        Some(Binding::Concrete(v)) => v.clone(),
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
    // CSE pass collapses duplicated sub-expressions to
    // __cse_<n> Lets. without this, large kernels emit single-line
    // CUDA source strings >700 KB and blow up rustc's memory during
    // string-literal lowering (40-50 GiB anon-rss on the symbi crate).
    crate::passes::cse::cse_lowered_fn(&mut f);
    f
}

/// kernel-mode scalarization: walk the graph once and extract one
/// scalar result per element of `outputs`, sharing the let-binding
/// body across all outputs. used by the kernel emitter to
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
    // dce: walk backward from `outputs`, collect transitively-reachable nodes.
    // any node not reachable from an output is dead — its arithmetic contributes
    // to no buffer write. without this filter, the iso flux (which traces the
    // Newtonian regime then drops the `flux.nrg` write) still lowers the entire
    // energy U/F sub-DAG into the body as orphan `__cse_*` let-bindings; nvcc
    // DCEs the sass but pays the parse + register-pressure cost. closing the
    // dead path here removes them from the emitted CUDA source up front.
    //
    // IterateInline cone nodes (lowered inside the loop body)
    // are reached via the IterateInline node's `steps` field through `Op::inputs`
    // — so a reachable IterateInline pulls its cone in automatically.
    let reachable = reachable_from_outputs(graph, outputs);
    // an IterateInline emits its body once as a `for`. its `step`
    // sub-DAG's acc-dependent cone must be lowered inside the loop
    // — collect those nodes to skip, and the cone per iterate node.
    //
    // Op::Scope body NodeIds are similarly partitioned —
    // they belong to the Scope's lexical region and are lowered inside a
    // `ScalarStmt::Scope` brace block.
    //
    // for nested scopes (an Op::Scope inside another Op::Scope's body), a
    // single NodeId may appear in multiple scopes' body lists. `scope_owner`
    // maps each scoped NodeId to its innermost owner — built by walking the
    // graph in NodeId order (inner scopes have smaller NodeIds than the
    // outer scopes containing them, so first-claim wins is correct).
    let mut skip: HashSet<NodeId> = HashSet::new();
    let mut iter_cones: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut scope_owner: HashMap<NodeId, NodeId> = HashMap::new();
    // a NodeId can appear in a scope's
    // body list yet also be referenced from outside the scope, because the
    // graph is hash-consed — a closure may compute `bn * bn` (pushed inside
    // the scope) and later outer code may produce a structurally-identical
    // `bn * bn` that hash-conses to the same NodeId. lowering it inside the
    // scope's brace breaks the outer reference. so scope-body
    // bookkeeping is collected first; a second pass evicts shared NodeIds from
    // scope_owner so the main pass lowers them at the outer level.
    let mut scope_body: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut scope_result: HashMap<NodeId, NodeId> = HashMap::new();
    // `Op::IfElse` (the dual of IterateInline): each arm body is a lexical
    // region lowered inside its `if`/`else` brace — same partition as Scope but
    // two regions per node. `branch_owner` maps each arm-owned NodeId to its
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
                // first-seen wins -> innermost scope claims the NodeId.
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
    // each kernel output is a virtual user — a sentinel NodeId past the
    // graph's range. ensures eviction catches "scope body node is also a
    // top-level kernel output" — otherwise the output's binding would be
    // a Var(...) only visible inside the scope's brace.
    let output_sentinel = NodeId(u32::MAX);
    for out in outputs {
        users.entry(*out).or_default().push(output_sentinel);
    }
    // evict scope_owner claims for NodeIds with users outside the claimed
    // scope's body. the scope's `result` NodeId is exempt — it's allowed
    // to be referenced by the Op::Scope node (which is in the outer body)
    // because that reference resolves to the scope's named output.
    //
    // a nested Op::Scope NodeId is also exempt: it's a structural child of
    // the outer scope (lowered via the recursive `lower_scope` dispatch),
    // and any "outside" use of its value lives in the outer scope's body —
    // which the recursive dispatch handles correctly. evicting nested
    // Op::Scope nodes would dump their body into the outer level, defeating
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
            // structural `result` field — allowed only for the result.
            if u == owner_scope && is_result {
                continue;
            }
            if body_set.contains(&u) {
                continue;
            }
            // user is outside the scope's body — shared. evict so main pass
            // lowers X at the outer level (the scope body still names the
            // outer's let in its `Var(...)` references). this handles both
            // non-result body members (the `bn * bn` hash-cons case) and
            // result NodeIds that also leak outside (an inner expression
            // hash-consed with an outer expression).
            evict.push(x);
            break;
        }
    }
    for x in evict {
        scope_owner.remove(&x);
    }
    // now populate `skip` from the final scope_owner (so evicted nodes
    // remain visible to the main pass).
    for &b in scope_owner.keys() {
        skip.insert(b);
    }
    // branch-arm eviction (the IfElse dual of the scope eviction above). a
    // then/else-arm node that is used outside its arm — by the sibling arm
    // (cross-arm hash-cons share) or by outer code — is hoisted to the outer
    // level and computed once (shared work is unconditional, which is correct).
    // crucial difference from Scope: the IfElse node lists its own arm bodies as
    // inputs (for remap-safety), so it appears as a "user" of every arm node;
    // that self-reference is not an escape — exempt `u == ifelse_id` for every
    // arm node, else laziness would be destroyed (all arms hoisted out).
    // fixpoint eviction: an arm node X "escapes" if it has a real (dataflow)
    // user lying outside the arm's current membership. evicting X moves it to
    // the outer level, so any in-arm input of X must then escape too — hence
    // the iteration to a fixpoint (a single pass would miss the cascade, and
    // checking static body membership would wrongly treat an already-evicted
    // body member as still-in-arm, leaving a dangling in-arm reference). a user
    // is not an escape when it is: the owning container itself (arm output /
    // structural self-listing); still owned by the same arm; or an outer
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
            continue; // unreachable from any output — dce
        }
        if skip.contains(&id) {
            continue; // a loop cone or scope body node — lowered inside its container
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
            None => panic!("scalarize_kernel: output node {:?} not lowered", out),
        })
        .collect();
    let mut k = KernelScalarized {
        params: sc.params,
        body: sc.body,
        outputs: lowered_outputs,
    };
    // CSE pass. see comment in `scalarize`.
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
    graph.reachable_from(outputs)
}

/// the acc-dependent cone of an `IterateInline` — the nodes that
/// must be recomputed each iteration. = the union backward-reachable set of all
/// `steps` whose `dep` flag is true, where `dep[n] = accs.contains(n) ||
/// any(dep[input])`. returned in increasing-id (topological) order. loop-invariant
/// nodes (dep == false) stay in the main pass as kernel locals computed once.
/// is `user`'s reference to `x` purely structural — i.e., `user` is a container
/// (IfElse / Scope) that lists `x` in its body for remap-safety/dce but does
/// not consume `x` as dataflow (cond / result)? such a reference must not count
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
    // cone (lowered inside the for-loop; hoisting it as a loop invariant
    // would constant-fold `0.5 < initial_done` into a dead `false`).
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

/// resolve a shape to its literal-only dimensions.
fn resolve_literal_dims(shape: &[DimExpr]) -> Vec<usize> {
    shape
        .iter()
        .map(|d| match d {
            DimExpr::Literal(n) => *n,
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
        // `Numeric` carrier. so CPU and GPU agree at NaN/signed-zero
        // without lowering to a scoped `if`-select — that inlines nested min/max
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
        ElementWiseOp::Tan => method_unary("tan", &mut inputs),
        ElementWiseOp::Asin => method_unary("asin", &mut inputs),
        ElementWiseOp::Atan => method_unary("atan", &mut inputs),
        ElementWiseOp::Exp => method_unary("exp", &mut inputs),
        ElementWiseOp::Exp2 => method_unary("exp2", &mut inputs),
        ElementWiseOp::Log => method_unary("ln", &mut inputs),
        ElementWiseOp::Log2 => method_unary("log2", &mut inputs),
        ElementWiseOp::Log10 => method_unary("log10", &mut inputs),
        ElementWiseOp::Tanh => method_unary("tanh", &mut inputs),
        ElementWiseOp::Atanh => method_unary("atanh", &mut inputs),
        ElementWiseOp::Atan2 => method_binary("atan2", &mut inputs),
        ElementWiseOp::Hypot => method_binary("hypot", &mut inputs),
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
/// (e.g., `contract(v, ehat)` for `ehat = (1, 0, 0)` lowers to `v0*1 + v1*0 + v2*0`).
///
/// safe set (the only identities folded here):
///   `x + 0 -> x`,  `0 + x -> x`
///   `x - 0 -> x`
///   `x * 1 -> x`,  `1 * x -> x`
///   `x / 1 -> x`
///
/// safety: `Mul`-absorbing arms (`x * 0 -> 0` / `0 * x -> 0`) are
/// intentionally omitted — IEEE-754 says `inf * 0 = NaN` and `NaN * 0 = NaN`,
/// and the user's `feedback_no_silent_floors` policy is that NaN must
/// propagate so dt-reduction / regression-test machinery sees it. while the
/// substrate's current emit only feeds syntactic `Op::Const(0.0)` into Mul
/// (so `inf` could not in practice reach the zero arm), any future builder
/// that drops a syntactic 0 into a flux-limiter / floor-clamp position could
/// pick up an `inf * x` upstream — and the absorbing fold would mask it
/// silently. removed pre-emptively.
///
/// the safe set is the one `arith_identity_elements` table in `graph.rs`, which the
/// graph-layer fold queries too — the two layers structurally cannot drift.
///
/// comparison / bitwise kinds fall through to construction unchanged.
fn fold_arith_identity(kind: BinaryKind, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    use crate::graph::{FoldableArith, arith_identity_elements};
    let foldable = match kind {
        BinaryKind::Add => Some(FoldableArith::Add),
        BinaryKind::Sub => Some(FoldableArith::Sub),
        BinaryKind::Mul => Some(FoldableArith::Mul),
        BinaryKind::Div => Some(FoldableArith::Div),
        _ => None,
    };
    if let Some(op) = foldable {
        let (left, right) = arith_identity_elements(op);
        let ca = const_zero_or_one(&a);
        let cb = const_zero_or_one(&b);
        if right.is_some() && cb == right {
            return a;
        }
        if left.is_some() && ca == left {
            return b;
        }
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
    use crate::TensorTy;

    fn lit(n: usize) -> DimExpr {
        DimExpr::Literal(n)
    }

    // ---- Op::Fold lowering ----

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

    // ---- Op::IterateInline lowering ----

    #[test]
    fn iterate_inline_lowers_to_one_loop_body_emitted_once() {
        // a fixed-iteration sqrt-Newton: acc <- 0.5*(acc + a/acc). `a` is
        // loop-invariant (must stay outside the loop); the acc-dependent step
        // (a/acc, acc+.., 0.5*..) goes inside — emitted once.
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

        // exactly one loop (the body emitted once).
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
            // the acc-dependent step is inside the loop (a Div for a/acc).
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
        // RMHD c2p: a 2-component accumulator — the Fibonacci recurrence
        // (a, b) -> (b, a+b). this is the cleanest probe of simultaneous update:
        // sequential (`a = b; b = a + b`) would read the new `a` and corrupt it.
        // the substrate must emit both assigns after the step body, reading the old
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

        // the loop ends with the two assigns both last (simultaneous), each reading
        // the old accumulator locals.
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
        // they must be the last two statements (nothing reads a partially-updated acc).
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

    // ---- ElementWise scalarization ----

    use crate::ElementWiseOp;

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

    // ---- Transcendental scalarization ----

    #[test]
    fn transcendental_sin_emits_method_call() {
        let mut g = Graph::new();
        let v = g.add_scalar_param("v", ElementTy::F64);
        let s = g.element_wise(ElementWiseOp::Sin, vec![v], None);
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
        let r = g.element_wise(ElementWiseOp::Log, vec![v], None);
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
        let r = g.element_wise(ElementWiseOp::Pow, vec![b, e], None);
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
        let r = g.element_wise(ElementWiseOp::Atan2, vec![y, x], None);
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
        let s = g.element_wise(ElementWiseOp::Cos, vec![v], None);
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

    // ---- Construct ----

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

    // ---- Index ----

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

    // ---- Broadcast ----

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

    // ---- Reduce ----

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

    // ---- Select ----

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
}
