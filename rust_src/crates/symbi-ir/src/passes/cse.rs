// =============================================================================
// cse.rs
//
// common-subexpression elimination pass over LoweredFn / KernelScalarized.
//
// motivation: the scalarizer produces fully-inlined ScalarExpr trees. when
// the source graph has shared sub-graphs (e.g. wave-speed coefficients
// reused across all conserved-flux components in a riemann kernel), the
// same subexpression is duplicated at every consumer. for large kernels
// the duplication is multiplicative — newton_face_flux_3d emitted single
// CUDA lines of 700 KB+ pre-CSE, with 5× duplication per output. rustc's
// memory peaks at 40-50 GiB just parsing the resulting `&'static str`
// literals.
//
// algorithm:
//   single-pass bottom-up hash recursion.
//
//   walk the tree once. at each node, recursively compute children's
//   hashes FIRST, then combine them with the operator's hash to get
//   the node's own hash. each subexpression is hashed exactly once;
//   the resulting work is O(N) where N is the total number of nodes
//   (including duplicated subtrees).
//
//   the first (count) pass returns each subexpression's hash AND
//   increments a counter keyed by that hash. a subexpression appearing
//   K times in the tree contributes K increments to its hash bucket;
//   the work per increment is O(1).
//
//   the second (rewrite) pass walks the tree again, also bottom-up
//   single-pass, and at each subexpression: if its hash matches a
//   "promote me" candidate, emit a Let with a fresh __cse_<n> name
//   (once per hash), then replace the subexpression with Var(name).
//
// previous (broken) implementation: per-subexpression `key_of(e)`
// built the FULL structural string of the subtree — O(subtree-size)
// work — and was invoked O(N) times. total work O(N^2) on duplicated
// trees, which for RMHD kernels (deeply-spliced wave-speed bodies)
// translated to multi-minute compile times and 50+ GiB rustc RSS.
//
// hash collisions: u64 + FxHash-style mixing is collision-safe for
// the sizes we care about (a kernel has at most ~10^4 distinct
// subexpressions; the collision probability is ~10^4 / 2^64 ~ 10^-15).
// we trust the hash. if a collision did occur, the worst outcome
// would be one bogus CSE that produces wrong code at runtime —
// which the existing physics-regression test suite would catch
// long before reaching production.
//
// V1 scope: skips into For-loop bodies (iter-var visibility). only
// F64-producing subexpressions are eligible — Bool / comparison
// results would need separate element tracking.
// =============================================================================

use std::collections::{HashMap, HashSet};

use crate::graph::ConstValue;
use crate::passes::scalarize::{LoweredFn, ScalarExpr, ScalarStmt};
use crate::{ElementTy, KernelScalarized};

/// run CSE over a `LoweredFn` (elemental-style). mutates body in place
/// to insert `__cse_<n>` Let statements; results are rewritten to
/// reference the temps.
pub fn cse_lowered_fn(f: &mut LoweredFn) {
    cse_in_place(&mut f.body, &mut f.results, "__cse_");
}

/// run CSE over a `KernelScalarized` (kernel-style). same shape as
/// `cse_lowered_fn` but operates on the kernel emitter's body + outputs.
pub fn cse_kernel(k: &mut KernelScalarized) {
    cse_in_place(&mut k.body, &mut k.outputs, "__cse_");
}

// ----- internals -----

fn cse_in_place(
    body:    &mut Vec<ScalarStmt>,
    outputs: &mut Vec<ScalarExpr>,
    prefix:  &str,
) {
    // **docs/design/23 step 2: scope-aware CSE.** before the flat-pass below
    // runs, recursively CSE each `Scope`'s body so duplicates LOCAL to a
    // scope land inside that scope's braces (the scope acts as a hoisting
    // barrier per the design). this is the structural fix for the
    // `wave_speed_map` 239-`__cse_N`-temp pathology: putting the quartic-
    // coefficient phase inside `S::scope` (step 3) localises its
    // intermediates and nvcc sees their lifetimes end at the `}`.
    //
    // `recursive_cse_scopes` ALSO walks into `For` / `If` bodies — those
    // stay opaque to the OUTER flat-pass below (V1 policy), but a `Scope`
    // nested inside them still benefits from per-scope CSE.
    recursive_cse_scopes(body, prefix);

    // pass 1: O(N) hash + count.
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for stmt in body.iter() {
        count_in_stmt(stmt, &mut counts);
    }
    for out in outputs.iter() {
        let _ = count_and_hash(out, &mut counts);
    }

    // **docs/design/23 step 4**: candidacy is now COST-AWARE. the count map
    // is threaded as-is into the rewrite state; the threshold check happens
    // at each rewrite site, gated on `cost_class(expr)`. cheap ops (add /
    // sub / mul / neg / select / casts / comparisons) need >= 4 uses;
    // medium (div / sqrt / abs / min / max / floor / ceil) need >= 2;
    // expensive (transcendentals, free calls) and memory loads need >= 2.
    //
    // early bail-out: nothing with count < 2 can EVER reach any threshold,
    // so if no hash hit 2 we have no work. cheap-only kernels with all
    // shares at 2-3 also bail at the per-class check during pass 2.
    if !counts.values().any(|&c| c >= 2) {
        return;
    }

    // pass 2: rewrite + emit Lets just-in-time.
    //
    // the scalarizer's `maybe_hoist_to_let` ALSO mints `{prefix}<n>` Lets, with a
    // counter independent of ours. when both fire (a large kernel where hoisting
    // misses some structural duplicates that the hash-based pass below catches —
    // first hit by the SRHD iterate), our `next_id` starting at 0 would re-declare
    // a hoisted `{prefix}0`. Rust tolerates the shadowing (the CPU kernel compiled
    // + ran), but CUDA C rejects the re-declaration — the GPU PTX gate surfaced it.
    // continue numbering ABOVE any pre-existing `{prefix}<n>` so names are unique
    // on every backend. for kernels where hoisting de-duped everything we returned
    // early above, so this is a no-op there (output byte-identical).
    let next_id = max_temp_index(body, prefix).map_or(0, |m| m + 1);
    let mut state = RewriteState {
        counts,
        emitted: HashMap::new(),
        next_id,
        prefix:  prefix.to_string(),
    };
    let body_take = std::mem::take(body);
    let mut rewritten: Vec<ScalarStmt> = Vec::with_capacity(body_take.len());
    for stmt in body_take {
        rewrite_stmt(stmt, &mut state, &mut rewritten);
    }
    let outs_take = std::mem::take(outputs);
    let mut new_outputs: Vec<ScalarExpr> = Vec::with_capacity(outs_take.len());
    for out in outs_take {
        let (e, _) = rewrite_expr(out, &mut state, &mut rewritten);
        new_outputs.push(e);
    }
    *body    = rewritten;
    *outputs = new_outputs;

    // pass 3: drop lets orphaned by pass 2's re-hoisting (see `dce_in_place`).
    // only runs on the rewrite path — the early-bail above leaves the body
    // byte-identical, and the scalarizer's own output carries no dead lets.
    dce_in_place(body, outputs);
}

// ---- pass 3: dead-let elimination ----
//
// the two-pass cse and the scalarizer's independent `maybe_hoist_to_let` can
// both mint a `{prefix}<n>` Let for the SAME value: the scalarizer hoists it
// once, then pass 2 (numbering ABOVE the scalarizer's temps) hoists the
// identical value again under a fresh name and rewrites every USE site to the
// new temp — orphaning the scalarizer's Let (e.g. `__cse_0 = __cse_1`, read by
// nobody). pass 2 only ever ADDS statements, so the orphan survives into the
// emitted kernel, inflating the body and the `pressure` metric (which counts
// every live Let). this final sweep drops every immutable `Let` whose name is
// read NOWHERE — in the body or the outputs — to a fixpoint (dropping one
// orphan can reveal its operands as dead too).

/// collect every variable NAME read by `e` into `out`.
fn mark_expr_vars(e: &ScalarExpr, out: &mut HashSet<String>) {
    // the var NAMES this node reads directly: a `Var`, or an `IndexInto`'s `container` (a name,
    // not a child expr — `children()` only yields the index sub-expr). then recurse the SSOT
    // children for every sub-expression.
    match e {
        ScalarExpr::Var(name) => { out.insert(name.clone()); }
        ScalarExpr::IndexInto { container, .. } => { out.insert(container.clone()); }
        _ => {}
    }
    for c in e.children() {
        mark_expr_vars(c, out);
    }
}

/// true if `e` contains a `FreeCall` — conservatively treated as possibly
/// side-effecting, so a Let bound to such a value is NEVER dropped even if its
/// name is unread. every other ScalarExpr form is a pure dataflow read.
fn expr_has_free_call(e: &ScalarExpr) -> bool {
    matches!(e, ScalarExpr::FreeCall { .. }) || e.children().iter().any(|c| expr_has_free_call(c))
}

/// gather every NAME read anywhere in `body` — each statement's immediate
/// expression plus, recursively, its child statement bodies — into `out`.
/// built on the ScalarStmt walk SSOT (`child_expr` / `child_stmt_bodies`) so a
/// new statement variant is covered for free.
fn collect_reads(body: &[ScalarStmt], out: &mut HashSet<String>) {
    for stmt in body {
        if let Some(e) = stmt.child_expr() {
            mark_expr_vars(e, out);
        }
        for sub in stmt.child_stmt_bodies() {
            collect_reads(sub, out);
        }
    }
}

/// drop immutable `Let`s at this body level whose name is not in `read` and
/// whose value is side-effect-free. returns whether anything was removed.
/// nested-body lets are handled where they are minted: `cse_in_place` runs per
/// `Scope` body (via `recursive_cse_scopes`), and cse never hoists into bare
/// `For` / `If` bodies — so the orphan classes all surface at a level this is
/// called on.
fn drop_dead_lets(body: &mut Vec<ScalarStmt>, read: &HashSet<String>) -> bool {
    let before = body.len();
    body.retain(|stmt| !matches!(stmt,
        ScalarStmt::Let { name, value, .. }
            if !read.contains(name) && !expr_has_free_call(value)));
    body.len() != before
}

/// remove dead immutable `Let`s from `body` to a fixpoint, treating `outputs`
/// and all nested reads as the live roots.
fn dce_in_place(body: &mut Vec<ScalarStmt>, outputs: &[ScalarExpr]) {
    loop {
        let mut read: HashSet<String> = HashSet::new();
        for out in outputs {
            mark_expr_vars(out, &mut read);
        }
        collect_reads(body, &mut read);
        if !drop_dead_lets(body, &read) {
            break;
        }
    }
}

/// **docs/design/23 step 2**: recurse into every `Scope` in `body` and run
/// `cse_in_place` on its `(body, result)` pair, then ALSO walk through
/// `For` / `If` bodies to find Scopes nested deeper. each Scope is
/// CSE'd INDEPENDENTLY with its own candidate set + `__cse_N` numbering —
/// duplicates within a scope hoist to the scope's start, never escaping.
///
/// the outer `cse_in_place` runs AFTER this, so the outer flat-pass sees
/// the post-recursion body. it CAN'T peek inside a Scope (V1 `count_in_stmt`
/// policy: only immediate child expressions) — it only sees the Scope's
/// `result` expression. that's the architectural intent: scopes are
/// hoisting barriers.
fn recursive_cse_scopes(body: &mut Vec<ScalarStmt>, prefix: &str) {
    use crate::graph::ConstValue;
    for stmt in body.iter_mut() {
        match stmt {
            ScalarStmt::Scope { body: scope_body, result, .. } => {
                // wrap the scope's `result` into a single-element outputs vec,
                // run a fresh CSE over the scope's body, write back the
                // (possibly Var-rewritten) result.
                let placeholder = ScalarExpr::Const(ConstValue::F64(0.0));
                let mut outputs = vec![std::mem::replace(result, placeholder)];
                cse_in_place(scope_body, &mut outputs, prefix);
                *result = outputs.into_iter().next().expect("scope outputs vec");
            }
            ScalarStmt::For { body: for_body, .. } => {
                // For/If bodies are opaque to the OUTER flat CSE (iter-var /
                // branch scope), but a Scope nested inside still benefits
                // from per-scope CSE — recurse to find it.
                recursive_cse_scopes(for_body, prefix);
            }
            ScalarStmt::If { then_body, .. } => {
                recursive_cse_scopes(then_body, prefix);
            }
            ScalarStmt::IfElse { then_body, else_body, .. } => {
                // both cond arms are opaque to the OUTER flat CSE (branch
                // scope), but a Scope nested inside an arm still benefits from
                // per-scope CSE — recurse into each arm to find it.
                recursive_cse_scopes(then_body, prefix);
                recursive_cse_scopes(else_body, prefix);
            }
            _ => {}
        }
    }
}

/// the highest `n` among `{prefix}{n}` Let/LetMut DECLARATIONS already present in
/// `body` (recursing into For bodies). lets the CSE pass continue its numbering
/// above names a prior pass (the scalarizer's hoist) minted with the same prefix,
/// so no temp is declared twice. only declarations can collide — assignments and
/// Var references reuse an existing name, so they're not scanned.
fn max_temp_index(body: &[ScalarStmt], prefix: &str) -> Option<usize> {
    fn scan(stmts: &[ScalarStmt], prefix: &str, max: &mut Option<usize>) {
        // derived from `ScalarStmt::binding_name` + `child_stmt_bodies`.
        for s in stmts {
            if let Some(name) = s.binding_name()
                && let Some(idx) = name.strip_prefix(prefix).and_then(|r| r.parse::<usize>().ok())
            {
                *max = Some(max.map_or(idx, |m: usize| m.max(idx)));
            }
            for body in s.child_stmt_bodies() {
                scan(body, prefix, max);
            }
        }
    }
    let mut max = None;
    scan(body, prefix, &mut max);
    max
}

// ---- bottom-up hashing (no string allocation) ----

/// FxHash-like 64-bit mix. fast, deterministic, low collision in our
/// regime. not cryptographic.
const SEED:  u64 = 0xcbf29ce484222325; // FNV-style seed
const PRIME: u64 = 0x100000001b3;

#[inline]
fn mix(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(PRIME)
}

/// distinct domain tags so structurally different shapes can't
/// accidentally collide (e.g. BinOp vs Select).
const TAG_CONST:        u64 = 0x01;
const TAG_VAR:          u64 = 0x02;
const TAG_BINOP:        u64 = 0x03;
const TAG_UNARYOP:      u64 = 0x04;
const TAG_METHODCALL:   u64 = 0x05;
const TAG_SELECT:       u64 = 0x06;
const TAG_INDEXINTO:    u64 = 0x07;
const TAG_FIELDLOADAT:  u64 = 0x08;
const TAG_FREECALL:     u64 = 0x09;
const TAG_CAST:         u64 = 0x0a;

fn hash_str(s: &str) -> u64 {
    let mut h = SEED;
    for b in s.bytes() { h = mix(h, b as u64); }
    h
}

fn hash_const(v: &ConstValue) -> u64 {
    let mut h = mix(SEED, TAG_CONST);
    match v {
        ConstValue::F64(x) => { h = mix(h, 0x64);  h = mix(h, x.to_bits()); }
        ConstValue::F32(x) => { h = mix(h, 0x32);  h = mix(h, x.to_bits() as u64); }
        ConstValue::I32(x) => { h = mix(h, 0x33);  h = mix(h, *x as u32 as u64); }
        ConstValue::U32(x) => { h = mix(h, 0x35);  h = mix(h, *x as u64); }
        ConstValue::Bool(b) => { h = mix(h, 0x36); h = mix(h, *b as u64); }
    }
    h
}

/// O(1) per node hash combinator. when called from a bottom-up walk
/// with child hashes already computed, the entire tree's hash table
/// fills in O(N) total work.
fn hash_expr_step(e: &ScalarExpr, child_hashes: &[u64]) -> u64 {
    use crate::passes::scalarize::BinaryKind;
    use crate::passes::scalarize::UnaryKind;
    match e {
        ScalarExpr::Const(v) => hash_const(v),
        ScalarExpr::Var(name) => {
            let mut h = mix(SEED, TAG_VAR);
            h = mix(h, hash_str(name));
            h
        }
        ScalarExpr::BinOp(op, _, _) => {
            // children: a, b. commutative ops hash operand-order-independently
            // so `a op b` and `b op a` collapse to a single cse temp. ieee add
            // and mul are bit-commutative, so reusing one order for the other
            // does not perturb the emitted numerics.
            let (op_tag, commutative) = match op {
                BinaryKind::Add   => (1, true),  BinaryKind::Sub   => (2, false),
                BinaryKind::Mul   => (3, true),  BinaryKind::Div   => (4, false),
                BinaryKind::Eq    => (5, true),  BinaryKind::Ne    => (6, true),
                BinaryKind::Lt    => (7, false), BinaryKind::Le    => (8, false),
                BinaryKind::Gt    => (9, false), BinaryKind::Ge    => (10, false),
                BinaryKind::BitOr => (11, true), BinaryKind::BitAnd => (12, true),
                BinaryKind::BitXor => (13, true),
            };
            let mut h = mix(SEED, TAG_BINOP);
            h = mix(h, op_tag);
            // sort the two child hashes for commutative ops; keep positional
            // order otherwise. full hash entropy is preserved either way.
            let (first, second) = if commutative && child_hashes[1] < child_hashes[0] {
                (child_hashes[1], child_hashes[0])
            } else {
                (child_hashes[0], child_hashes[1])
            };
            h = mix(h, first);
            h = mix(h, second);
            h
        }
        ScalarExpr::UnaryOp(op, _) => {
            let op_tag = match op { UnaryKind::Neg => 1, UnaryKind::Not => 2 };
            let mut h = mix(SEED, TAG_UNARYOP);
            h = mix(h, op_tag);
            h = mix(h, child_hashes[0]);
            h
        }
        ScalarExpr::Cast { to, .. } => {
            let mut h = mix(SEED, TAG_CAST);
            h = mix(h, *to as u64);
            h = mix(h, child_hashes[0]);
            h
        }
        ScalarExpr::MethodCall { method, .. } => {
            let mut h = mix(SEED, TAG_METHODCALL);
            h = mix(h, hash_str(method));
            for &c in child_hashes { h = mix(h, c); }
            h
        }
        ScalarExpr::Select { .. } => {
            let mut h = mix(SEED, TAG_SELECT);
            for &c in child_hashes { h = mix(h, c); }
            h
        }
        ScalarExpr::IndexInto { container, .. } => {
            let mut h = mix(SEED, TAG_INDEXINTO);
            h = mix(h, hash_str(container));
            for &c in child_hashes { h = mix(h, c); }
            h
        }
        ScalarExpr::FieldLoadAt { field_key, .. } => {
            let mut h = mix(SEED, TAG_FIELDLOADAT);
            h = mix(h, hash_str(field_key));
            for &c in child_hashes { h = mix(h, c); }
            h
        }
        ScalarExpr::FreeCall { name, .. } => {
            let mut h = mix(SEED, TAG_FREECALL);
            h = mix(h, hash_str(name));
            for &c in child_hashes { h = mix(h, c); }
            h
        }
    }
}

// ---- candidate eligibility: cost model ----

/// **docs/design/23 step 4: register-pressure cost model.**
///
/// classify each ScalarExpr by the cost of RECOMPUTING it vs the cost of
/// HOLDING it in a register across its live range. hoisting a value costs
/// ~1 SM cycle stall risk per warp (register pressure → spill); recomputing
/// a cheap op costs ~0-1 cycles. so cheap shared-twice values are a net
/// loss when hoisted, and transcendentals shared once are mandatory hoists.
///
/// the threshold table is fixed at build time. it is NOT a heuristic the
/// kernel author can tune — that would make pressure unpredictable. instead
/// the author opts into scoping (`S::scope`) to bound lifetimes; the cost
/// model decides per-op whether the share-count justifies a Let.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CostClass {
    /// `Const`, `Var` — leaf-like, recomputed for free. never hoist.
    Trivial,
    /// `+`, `-`, `*`, `Neg`, `Not`, `Select`, comparisons, bitops, `Cast` —
    /// one ALU op. hoist iff used >= 4 times.
    Cheap,
    /// `/`, `sqrt`, `abs`, `min`, `max`, `floor`, `ceil` — a few ALU
    /// ops or a single special-function unit op. hoist iff used >= 2 times.
    Medium,
    /// transcendentals: `sin`, `cos`, `tan`, `exp`, `ln`, `log10`, `pow`,
    /// `atan2`, `hypot`, hyperbolics. SFU-bound, many cycles. always
    /// hoist when shared (>= 2 uses).
    Expensive,
    /// `FieldLoadAt`, `IndexInto` — memory access. the load is the cost;
    /// the value, once in a register, is much cheaper to hold than to
    /// re-fetch. always hoist when shared.
    Memory,
}

fn cost_class(e: &ScalarExpr) -> CostClass {
    use crate::passes::scalarize::{BinaryKind, UnaryKind};
    match e {
        ScalarExpr::Const(_) | ScalarExpr::Var(_) => CostClass::Trivial,
        ScalarExpr::BinOp(op, _, _) => match op {
            // Div is the lone medium-cost binary op (FP divider).
            BinaryKind::Div => CostClass::Medium,
            // every other binary (add/sub/mul/comparisons/bitops) is
            // one ALU op = cheap.
            BinaryKind::Add | BinaryKind::Sub | BinaryKind::Mul
            | BinaryKind::Eq  | BinaryKind::Ne | BinaryKind::Lt | BinaryKind::Le
            | BinaryKind::Gt  | BinaryKind::Ge
            | BinaryKind::BitOr | BinaryKind::BitAnd | BinaryKind::BitXor
                => CostClass::Cheap,
        },
        ScalarExpr::UnaryOp(op, _) => match op {
            // both Neg and Not are single-instruction.
            UnaryKind::Neg | UnaryKind::Not => CostClass::Cheap,
        },
        ScalarExpr::Select { .. } => CostClass::Cheap,
        ScalarExpr::Cast { .. } => CostClass::Cheap,
        ScalarExpr::MethodCall { method, .. } => match method.as_str() {
            // medium: few-cycle ALU specials.
            "sqrt" | "abs" | "min" | "max" | "floor" | "ceil"
            | "is_finite" | "is_nan"
                => CostClass::Medium,
            // expensive: SFU / library calls. transcendentals + pow.
            "sin"  | "cos"  | "tan"  | "asin" | "acos" | "atan" | "atan2"
            | "exp" | "exp2" | "ln"  | "log2" | "log10"
            | "sinh"| "cosh" | "tanh"| "asinh"| "acosh"| "atanh"
            | "powf"| "powi" | "hypot"
                => CostClass::Expensive,
            // unknown methods default to Expensive — a conservative bias
            // toward hoisting unfamiliar ops (better one extra register
            // than a stalling recompute of something we mis-cheap'd).
            _ => CostClass::Expensive,
        },
        ScalarExpr::IndexInto   { .. } => CostClass::Memory,
        ScalarExpr::FieldLoadAt { .. } => CostClass::Memory,
        ScalarExpr::FreeCall    { .. } => CostClass::Expensive,
    }
}

/// the share-count threshold at which an expression of this class becomes
/// worth hoisting. `usize::MAX` means never. this is the BASE threshold; the
/// effective threshold at a rewrite site is depth-adjusted by
/// `effective_threshold` (a deep cheap chain is worth a register sooner).
#[inline]
fn hoist_threshold(c: CostClass) -> usize {
    match c {
        CostClass::Trivial   => usize::MAX,
        CostClass::Cheap     => 4,
        CostClass::Medium    => 2,
        CostClass::Expensive => 2,
        CostClass::Memory    => 2,
    }
}

/// minimum subtree height at which a CHEAP share drops to the 2-use threshold.
/// a `(v - w)` (height 1) stays inline at 2 uses; a `((a*b)*c)*d` ALU chain
/// (height >= 3) hoists at 2 — the recompute cost scales with chain depth, the
/// register cost does not.
const CHEAP_DEPTH_FLOOR: usize = 3;

/// true iff `e`'s subtree height is at least `k`. height(leaf) = 0,
/// height(node) = 1 + max(child heights). DEPTH-CAPPED: recurses at most `k`
/// levels (short-circuits on the first qualifying path), so it is O(k) per
/// call — it does NOT re-walk the whole subtree and so does not reintroduce
/// the O(N^2) the single-pass hash design eliminated.
fn height_ge(e: &ScalarExpr, k: usize) -> bool {
    if k == 0 {
        return true;
    }
    match e {
        ScalarExpr::Const(_) | ScalarExpr::Var(_)
        | ScalarExpr::IndexInto { .. } | ScalarExpr::FieldLoadAt { .. } => false,
        ScalarExpr::UnaryOp(_, a) => height_ge(a, k - 1),
        ScalarExpr::Cast { value, .. } => height_ge(value, k - 1),
        ScalarExpr::BinOp(_, a, b) => height_ge(a, k - 1) || height_ge(b, k - 1),
        ScalarExpr::Select { cond, then, else_ } => {
            height_ge(cond, k - 1) || height_ge(then, k - 1) || height_ge(else_, k - 1)
        }
        ScalarExpr::MethodCall { receiver, args, .. } => {
            height_ge(receiver, k - 1) || args.iter().any(|a| height_ge(a, k - 1))
        }
        ScalarExpr::FreeCall { args, .. } => args.iter().any(|a| height_ge(a, k - 1)),
    }
}

/// the depth-aware share-count threshold for hoisting `e`. identical to the
/// base `hoist_threshold` for every class EXCEPT a deep cheap subtree, which
/// drops to 2 uses. only ever LOWERS a threshold, never raises one.
fn effective_threshold(e: &ScalarExpr) -> usize {
    let class = cost_class(e);
    let base = hoist_threshold(class);
    if class == CostClass::Cheap && height_ge(e, CHEAP_DEPTH_FLOOR) {
        base.min(2)
    } else {
        base
    }
}

/// triviality test: leaf-like expressions aren't worth CSE'ing. retained
/// as a structural guard at the rewrite site even though the cost-model
/// threshold for `Trivial` is `usize::MAX` (defence in depth — a trivial
/// node hashing identically to a non-trivial twin still bails out).
fn is_trivial(e: &ScalarExpr) -> bool {
    matches!(e, ScalarExpr::Const(_) | ScalarExpr::Var(_))
}

/// for V1, only F64-typed expressions are eligible. Bool / comparison
/// results would need separate element tracking in the emitted Let.
fn is_f64_expr(e: &ScalarExpr) -> bool {
    use crate::passes::scalarize::BinaryKind;
    match e {
        ScalarExpr::Const(ConstValue::F64(_)) => true,
        ScalarExpr::Const(_) => false,
        ScalarExpr::Var(_) => true, // assume f64; conservative
        ScalarExpr::BinOp(op, a, _b) => {
            matches!(op,
                BinaryKind::Add | BinaryKind::Sub
              | BinaryKind::Mul | BinaryKind::Div)
            && is_f64_expr(a)
        }
        ScalarExpr::UnaryOp(_, a) => is_f64_expr(a),
        ScalarExpr::MethodCall { method, receiver, .. } => {
            // is_finite / is_nan return bool; everything else returns
            // the receiver's element type.
            !matches!(method.as_str(), "is_finite" | "is_nan") && is_f64_expr(receiver)
        }
        ScalarExpr::Select { then, .. } => is_f64_expr(then),
        ScalarExpr::IndexInto { .. } => true,
        ScalarExpr::FieldLoadAt { .. } => true,
        ScalarExpr::FreeCall { .. } => true,
        // a Cast to a float type is an f64-valued (CSE-eligible) expression.
        ScalarExpr::Cast { to, .. } => to.is_float(),
    }
}

// ---- pass 1: hash + count, single bottom-up walk ----

fn count_in_stmt(stmt: &ScalarStmt, counts: &mut HashMap<u64, usize>) {
    // CSE policy V1: only the immediate child expression participates in CSE;
    // descending into For / If sub-bodies would cross iter-var / branch scope
    // boundaries. derived from `ScalarStmt::child_expr` — every variant with
    // an immediate scalar expression is handled uniformly.
    if let Some(e) = stmt.child_expr() {
        let _ = count_and_hash(e, counts);
    }
}

/// recurse into children, compute their hashes, then combine into
/// THIS node's hash. each subexpression is hashed exactly once and
/// increments its hash bucket exactly once per occurrence in the tree.
fn count_and_hash(e: &ScalarExpr, counts: &mut HashMap<u64, usize>) -> u64 {
    let child_hashes = match e {
        ScalarExpr::Const(_) | ScalarExpr::Var(_) => Vec::new(),
        ScalarExpr::BinOp(_, a, b) => {
            vec![
                count_and_hash(a, counts),
                count_and_hash(b, counts),
            ]
        }
        ScalarExpr::UnaryOp(_, a) => vec![count_and_hash(a, counts)],
        ScalarExpr::MethodCall { receiver, args, .. } => {
            let mut v = Vec::with_capacity(1 + args.len());
            v.push(count_and_hash(receiver, counts));
            for a in args { v.push(count_and_hash(a, counts)); }
            v
        }
        ScalarExpr::Select { cond, then, else_ } => {
            vec![
                count_and_hash(cond,  counts),
                count_and_hash(then,  counts),
                count_and_hash(else_, counts),
            ]
        }
        ScalarExpr::IndexInto { index, .. } => vec![count_and_hash(index, counts)],
        ScalarExpr::FieldLoadAt { components, .. } => {
            components.iter().map(|c| count_and_hash(c, counts)).collect()
        }
        ScalarExpr::FreeCall { args, .. } => {
            args.iter().map(|a| count_and_hash(a, counts)).collect()
        }
        ScalarExpr::Cast { value, .. } => vec![count_and_hash(value, counts)],
    };
    let h = hash_expr_step(e, &child_hashes);
    *counts.entry(h).or_insert(0) += 1;
    h
}

// ---- pass 2: rewrite, single bottom-up walk ----

struct RewriteState {
    /// **docs/design/23 step 4**: hash -> raw share count (computed in
    /// pass 1). the cost-model threshold check at the rewrite site uses
    /// `cost_class(expr)` to pick the per-class minimum (cheap=4,
    /// medium/expensive/memory=2, trivial=never). storing counts (not a
    /// pre-bucketed set) lets the policy stay at the rewrite site, where
    /// the full ScalarExpr is still in hand.
    counts: HashMap<u64, usize>,
    /// hash -> temp name, once a Let has been physically emitted.
    emitted: HashMap<u64, String>,
    next_id: usize,
    prefix:  String,
}

impl RewriteState {
    fn fresh_name(&mut self) -> String {
        let n = self.next_id;
        self.next_id += 1;
        format!("{}{}", self.prefix, n)
    }
}

fn rewrite_stmt(
    stmt:     ScalarStmt,
    state:    &mut RewriteState,
    out_body: &mut Vec<ScalarStmt>,
) {
    // single source of truth via `with_child_expr`: thread the expression
    // rewriter through whatever expression this statement carries (none for
    // For / Break — both fall through). `rewrite_expr` may push CSE lets into
    // `out_body` BEFORE the rewritten value is captured, preserving the
    // "introduce-then-use" ordering.
    let rewritten = stmt.with_child_expr(|e| rewrite_expr(e, state, out_body).0);
    out_body.push(rewritten);
}

/// recurse into children, rewriting them (and emitting any pending
/// CSE Lets) FIRST, then decide for the current expression. returns
/// the rewritten expression and its structural hash so the parent
/// can compose its own hash without re-walking.
fn rewrite_expr(
    e:        ScalarExpr,
    state:    &mut RewriteState,
    out_body: &mut Vec<ScalarStmt>,
) -> (ScalarExpr, u64) {
    // recurse into children first.
    let (rewritten, child_hashes) = match e {
        ScalarExpr::Const(_) | ScalarExpr::Var(_) => {
            let h = hash_expr_step(&e, &[]);
            return (e, h);
        }
        ScalarExpr::BinOp(op, a, b) => {
            let (ra, ha) = rewrite_expr(*a, state, out_body);
            let (rb, hb) = rewrite_expr(*b, state, out_body);
            (ScalarExpr::BinOp(op, Box::new(ra), Box::new(rb)), vec![ha, hb])
        }
        ScalarExpr::UnaryOp(op, a) => {
            let (ra, ha) = rewrite_expr(*a, state, out_body);
            (ScalarExpr::UnaryOp(op, Box::new(ra)), vec![ha])
        }
        ScalarExpr::MethodCall { receiver, method, args } => {
            let (rr, hr) = rewrite_expr(*receiver, state, out_body);
            let mut new_args = Vec::with_capacity(args.len());
            let mut hashes = Vec::with_capacity(1 + args.len());
            hashes.push(hr);
            for a in args {
                let (ra, ha) = rewrite_expr(a, state, out_body);
                new_args.push(ra);
                hashes.push(ha);
            }
            (ScalarExpr::MethodCall { receiver: Box::new(rr), method, args: new_args }, hashes)
        }
        ScalarExpr::Select { cond, then, else_ } => {
            let (rc, hc) = rewrite_expr(*cond,  state, out_body);
            let (rt, ht) = rewrite_expr(*then,  state, out_body);
            let (re, he) = rewrite_expr(*else_, state, out_body);
            (ScalarExpr::Select {
                cond:  Box::new(rc),
                then:  Box::new(rt),
                else_: Box::new(re),
            }, vec![hc, ht, he])
        }
        ScalarExpr::IndexInto { container, index } => {
            let (ri, hi) = rewrite_expr(*index, state, out_body);
            (ScalarExpr::IndexInto { container, index: Box::new(ri) }, vec![hi])
        }
        ScalarExpr::FieldLoadAt { field_key, components } => {
            let mut new_components = Vec::with_capacity(components.len());
            let mut hashes = Vec::with_capacity(components.len());
            for c in components {
                let (rc, hc) = rewrite_expr(c, state, out_body);
                new_components.push(rc);
                hashes.push(hc);
            }
            (ScalarExpr::FieldLoadAt { field_key, components: new_components }, hashes)
        }
        ScalarExpr::FreeCall { name, args } => {
            let mut new_args = Vec::with_capacity(args.len());
            let mut hashes = Vec::with_capacity(args.len());
            for a in args {
                let (ra, ha) = rewrite_expr(a, state, out_body);
                new_args.push(ra);
                hashes.push(ha);
            }
            (ScalarExpr::FreeCall { name, args: new_args }, hashes)
        }
        ScalarExpr::Cast { to, value } => {
            let (rv, hv) = rewrite_expr(*value, state, out_body);
            (ScalarExpr::Cast { to, value: Box::new(rv) }, vec![hv])
        }
    };
    let self_hash = hash_expr_step(&rewritten, &child_hashes);

    // already emitted? -> return Var.
    if let Some(name) = state.emitted.get(&self_hash) {
        return (ScalarExpr::Var(name.clone()), self_hash);
    }

    // **docs/design/23 step 4**: candidate iff share-count meets the
    // cost-class threshold AND the expression is non-trivial AND F64.
    // each gate is a load-bearing structural check:
    //   - count: pass-1 hash buckets (could exceed the threshold);
    //   - cost_class: per-op policy from the table above;
    //   - is_trivial: defence in depth (Trivial class threshold is MAX, but
    //     a Const that hashes oddly should still bail);
    //   - is_f64_expr: V1 element-tracking — Bool-typed exprs would need a
    //     separate Let element.
    let count = state.counts.get(&self_hash).copied().unwrap_or(0);
    let threshold = effective_threshold(&rewritten);
    if count >= threshold
        && !is_trivial(&rewritten)
        && is_f64_expr(&rewritten)
    {
        let temp_name = state.fresh_name();
        state.emitted.insert(self_hash, temp_name.clone());
        out_body.push(ScalarStmt::Let {
            name:    temp_name.clone(),
            element: ElementTy::F64,
            value:   rewritten,
        });
        return (ScalarExpr::Var(temp_name), self_hash);
    }

    (rewritten, self_hash)
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstValue;
    use crate::passes::scalarize::{BinaryKind, ScalarExpr, ScalarStmt};

    fn v(name: &str) -> ScalarExpr { ScalarExpr::Var(name.to_string()) }

    /// regression: commutative ops hash operand-order-independently, so
    /// `a op b` and `b op a` produce the SAME hash (and collapse in cse);
    /// non-commutative ops keep order-sensitive hashes. threshold-independent.
    #[test]
    fn commutative_ops_hash_order_independent() {
        // the expr value's children are ignored by hash_expr_step (it reads
        // only the op tag + the supplied child hashes), so swapping the
        // child-hash array models swapping the operands.
        let ab = [11u64, 22u64];
        let ba = [22u64, 11u64];
        for (op, expect_equal) in [
            (BinaryKind::Add, true), (BinaryKind::Mul, true),
            (BinaryKind::Eq, true), (BinaryKind::Ne, true),
            (BinaryKind::BitOr, true), (BinaryKind::BitAnd, true),
            (BinaryKind::BitXor, true),
            (BinaryKind::Sub, false), (BinaryKind::Div, false),
            (BinaryKind::Lt, false), (BinaryKind::Le, false),
            (BinaryKind::Gt, false), (BinaryKind::Ge, false),
        ] {
            let e = ScalarExpr::BinOp(op, Box::new(v("a")), Box::new(v("b")));
            let h_ab = hash_expr_step(&e, &ab);
            let h_ba = hash_expr_step(&e, &ba);
            assert_eq!(h_ab == h_ba, expect_equal,
                "{:?}: order-equality should be {}", op, expect_equal);
        }
    }

    /// regression: `a*b` and `b*a` are the same commutative value, so their
    /// uses sum toward the hoist threshold. five total occurrences of the
    /// product (3 as a*b, 2 as b*a) cross the cheap-class threshold of 4 and
    /// collapse to ONE cse temp. before the commutative-hash fix the two
    /// orderings hashed apart (3 and 2 uses) and neither hoisted.
    #[test]
    fn commutative_product_collapses_across_operand_order() {
        let ab = || ScalarExpr::BinOp(BinaryKind::Mul, Box::new(v("a")), Box::new(v("b")));
        let ba = || ScalarExpr::BinOp(BinaryKind::Mul, Box::new(v("b")), Box::new(v("a")));
        // wrap each occurrence in a distinct parent so the product itself is
        // the shared subexpression (and the outer adds stay unique).
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(ab()), Box::new(v("p"))),
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(ab()), Box::new(v("q"))),
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(ab()), Box::new(v("r"))),
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(ba()), Box::new(v("s"))),
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(ba()), Box::new(v("t"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "a*b and b*a must collapse to one cse temp, got {} lets: {:?}",
            body.len(), body);
    }

    /// regression: depth-aware cheap threshold. a DEEP cheap chain (height >= 3,
    /// e.g. `((a*b)*c)*d`) shared twice hoists at 2 uses; a SHALLOW cheap share
    /// (height 1, e.g. `a-b`) shared twice stays inline (base threshold 4).
    #[test]
    fn depth_aware_cheap_threshold() {
        let mul = |a: ScalarExpr, b: ScalarExpr| {
            ScalarExpr::BinOp(BinaryKind::Mul, Box::new(a), Box::new(b))
        };
        // ((a*b)*c)*d : height 3, cheap. shared twice -> must hoist.
        let deep = || mul(mul(mul(v("a"), v("b")), v("c")), v("d"));
        assert!(height_ge(&deep(), 3), "fixture must be height >= 3");
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(deep()), Box::new(v("p"))),
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(deep()), Box::new(v("q"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "deep cheap chain shared twice must hoist, got {:?}", body);

        // a-b : height 1, cheap. shared twice -> must NOT hoist (base threshold 4).
        let shallow = || ScalarExpr::BinOp(BinaryKind::Sub, Box::new(v("a")), Box::new(v("b")));
        assert!(!height_ge(&shallow(), 3));
        let mut body2: Vec<ScalarStmt> = vec![];
        let mut outputs2 = vec![
            ScalarExpr::BinOp(BinaryKind::Mul, Box::new(shallow()), Box::new(shallow())),
        ];
        cse_in_place(&mut body2, &mut outputs2, "__cse_");
        assert!(body2.is_empty(),
            "shallow cheap share at 2 uses must stay inline, got {:?}", body2);
    }

    /// regression: pass 2 can re-hoist a value the scalarizer already hoisted,
    /// orphaning the original Let (`__cse_0 = __cse_1`, read by nobody). the
    /// pass-3 dce sweep must drop the orphan so it never reaches the kernel.
    #[test]
    fn dce_drops_let_orphaned_by_rehoist() {
        // a/b is medium-class (Div, threshold 2). pre-seed the body with the
        // scalarizer's hoist of a/b as __cse_0; the outputs use a/b twice, so
        // pass 2 re-hoists it under __cse_1 and rewrites the uses, orphaning
        // __cse_0.
        let div = || ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("a")), Box::new(v("b")));
        let mut body = vec![ScalarStmt::Let {
            name: "__cse_0".into(),
            element: ElementTy::F64,
            value: div(),
        }];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(div()), Box::new(v("c"))),
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(div()), Box::new(v("d"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        // exactly one surviving Let (the shared a/b), no orphan.
        assert_eq!(body.len(), 1, "orphaned let must be dropped, body={:?}", body);
        // every surviving let is actually read by the outputs.
        let mut read = HashSet::new();
        for o in &outputs {
            mark_expr_vars(o, &mut read);
        }
        for stmt in &body {
            if let Some(name) = stmt.binding_name() {
                assert!(read.contains(name), "surviving let `{name}` is dead");
            }
        }
    }

    #[test]
    fn no_cse_when_no_shared_subexpr() {
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(v("x")), Box::new(v("y"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert!(body.is_empty(), "no CSE Lets expected: {:?}", body);
        assert_eq!(
            outputs[0],
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(v("x")), Box::new(v("y"))),
        );
    }

    #[test]
    fn shared_subexpr_hoisted_to_let() {
        // step 4: Div is medium-class (threshold 2). a Div shared 2× still
        // hoists. swapping `Mul` (cheap, threshold 4) for `Div` keeps the
        // topology of the test while staying on the new policy's right side.
        let xx = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("z")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(xx.clone()), Box::new(v("a"))),
            ScalarExpr::BinOp(BinaryKind::Sub, Box::new(xx.clone()), Box::new(v("b"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1, "expected one CSE Let, got {:?}", body);
        let temp_name = match &body[0] {
            ScalarStmt::Let { name, value, .. } => {
                assert_eq!(*value, xx);
                name.clone()
            }
            other => panic!("expected Let, got {:?}", other),
        };
        assert_eq!(
            outputs[0],
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(v(&temp_name)), Box::new(v("a"))),
        );
        assert_eq!(
            outputs[1],
            ScalarExpr::BinOp(BinaryKind::Sub, Box::new(v(&temp_name)), Box::new(v("b"))),
        );
    }

    #[test]
    fn cse_numbering_continues_above_pre_hoisted_names() {
        // regression for the SRHD/GPU bug: the scalarizer's hoist pass already
        // minted `__cse_0`; CSE must NOT re-declare it (Rust shadows, CUDA C
        // errors). simulate a body that already declares `__cse_0`, then give the
        // outputs a DISTINCT shared subexpr CSE has to hoist. fixture uses Div
        // (medium-class, threshold 2) so the share triggers under step 4.
        let pre = ScalarExpr::BinOp(BinaryKind::Mul, Box::new(v("a")), Box::new(v("a")));
        let mut body: Vec<ScalarStmt> = vec![ScalarStmt::Let {
            name:    "__cse_0".to_string(),
            element: crate::ElementTy::F64,
            value:   pre,
        }];
        // one output READS the pre-hoisted __cse_0 so it is live (a realistic
        // scalarizer temp that is actually used) — otherwise pass-3 dce would
        // correctly drop it as dead and there would be no pre-hoisted name to
        // number above.
        let yy = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("y")), Box::new(v("z")));
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(yy.clone()), Box::new(v("p"))),
            ScalarExpr::BinOp(BinaryKind::Sub, Box::new(yy.clone()),
                Box::new(ScalarExpr::Var("__cse_0".to_string()))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");

        let names: Vec<String> = body
            .iter()
            .filter_map(|s| match s {
                ScalarStmt::Let { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        let unique: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "duplicate temp declaration: {names:?}");
        assert!(names.iter().any(|n| n == "__cse_0"), "pre-hoisted name kept: {names:?}");
        assert!(
            names.iter().any(|n| n == "__cse_1"),
            "CSE must continue numbering above the hoisted __cse_0: {names:?}"
        );
    }

    #[test]
    fn nested_shared_subexpr_emits_deeper_let_first() {
        // step 4: both fixtures must be medium-class so 2 uses suffices.
        // inner = Div(x, w); outer = Div(inner, y). both shared 2× across the
        // two outputs. deepest-first emission ordering is unchanged.
        let xx   = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("w")));
        let xxpy = ScalarExpr::BinOp(BinaryKind::Div, Box::new(xx.clone()), Box::new(v("y")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Mul, Box::new(xxpy.clone()), Box::new(v("a"))),
            ScalarExpr::BinOp(BinaryKind::Mul, Box::new(xxpy.clone()), Box::new(v("b"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 2, "expected two CSE Lets, got {:?}", body);
        let (name0, value0) = match &body[0] {
            ScalarStmt::Let { name, value, .. } => (name.clone(), value.clone()),
            _ => panic!(),
        };
        let (_name1, value1) = match &body[1] {
            ScalarStmt::Let { name, value, .. } => (name.clone(), value.clone()),
            _ => panic!(),
        };
        assert_eq!(value0, xx, "first Let must be deepest shared expr");
        assert_eq!(
            value1,
            ScalarExpr::BinOp(BinaryKind::Div, Box::new(v(&name0)), Box::new(v("y"))),
        );
    }

    #[test]
    fn no_cse_for_trivial_var_const() {
        let one = ScalarExpr::Const(ConstValue::F64(1.0));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(v("x")), Box::new(one.clone())),
            ScalarExpr::BinOp(BinaryKind::Sub, Box::new(v("x")), Box::new(one.clone())),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert!(body.is_empty(), "trivial subexprs must not be hoisted: {:?}", body);
    }

    #[test]
    fn idempotent_second_pass_is_no_op() {
        // step 4: use Div so the 2-share hits the medium-class threshold.
        let xx = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("z")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(xx.clone()), Box::new(v("a"))),
            ScalarExpr::BinOp(BinaryKind::Sub, Box::new(xx.clone()), Box::new(v("b"))),
        ];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        let body_after_first = body.clone();
        let outs_after_first = outputs.clone();
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body, body_after_first);
        assert_eq!(outputs, outs_after_first);
    }

    /// the load-bearing test: a chain of duplicated subtrees that
    /// would be quadratic-time in the old key_of implementation. this
    /// builds a 16-deep binary tree where each subtree is shared with
    /// its sibling — total node count is 2^17 - 1 = 131071, but the
    /// number of DISTINCT subexpressions is only 17. the old CSE would
    /// generate 17 keys totalling 131071 characters (cheap per key)
    /// but call count_in_expr 131071 times — each call's key_of
    /// walking the full subtree. that's ~10^10 character operations.
    /// the new CSE walks each node exactly once: ~10^5 ops total.
    ///
    /// the test budget here is conservative (no time assertion);
    /// completion in test mode confirms the algorithmic fix.
    #[test]
    fn deeply_duplicated_tree_completes_quickly() {
        // step 4: switch the duplicated op from Mul (cheap, threshold 4) to
        // Div (medium, threshold 2). every level is shared 2× via its
        // parent's `Div(e, e)`, so under the medium threshold the inner
        // 15 levels still hoist. the test's purpose remains: pass 1's
        // single-walk hash builder must stay O(N) — the change in
        // operator is fixture-only.
        let mut e = v("x");
        for _ in 0..16 {
            e = ScalarExpr::BinOp(BinaryKind::Div, Box::new(e.clone()), Box::new(e.clone()));
        }
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![e];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        // 15 CSE Lets: the bottom 15 of the 16 BinOp levels each
        // appear >= 2 times (their immediate parent uses them twice
        // via the Div(x, x) duplication). the topmost BinOp appears
        // exactly once (as the output) so it stays inline.
        // the leaf `x` is excluded by is_trivial.
        assert_eq!(body.len(), 15, "expected 15 CSE Lets, got {}", body.len());
        // the output is a Div whose two children are the same Var
        // pointing at the topmost CSE temp (one level below the root).
        match &outputs[0] {
            ScalarExpr::BinOp(BinaryKind::Div, a, b) => {
                match (&**a, &**b) {
                    (ScalarExpr::Var(na), ScalarExpr::Var(nb)) => assert_eq!(na, nb,
                        "topmost Div should reference the same CSE temp on both sides"),
                    other => panic!("expected (Var, Var), got {:?}", other),
                }
            }
            other => panic!("expected top-level Div, got {:?}", other),
        }
    }

    // ----- docs/design/23 step 2: scope-aware CSE tests -----

    /// helper: a Scope statement that wraps `inner_body` and binds the result
    /// of evaluating `inner_result` to a freshly-named outer local.
    fn scope_stmt(name: &str, body: Vec<ScalarStmt>, result: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Scope {
            name: name.to_string(),
            element: ElementTy::F64,
            body,
            result,
        }
    }

    /// helper: a Let with the standard f64 element.
    fn let_f64(name: &str, value: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Let { name: name.to_string(), element: ElementTy::F64, value }
    }

    /// **the load-bearing law**: a duplicated subexpression INSIDE a Scope is
    /// hoisted to that Scope's body (NOT to the outer function root). this is
    /// exactly the structural property we need for `wave_speed_map` — the
    /// quartic-coefficient phase's shared subexpressions should live INSIDE
    /// the phase's `{ }`, dying at the closing brace.
    #[test]
    fn shared_subexpr_inside_scope_hoists_inside_scope() {
        // outer body:
        //   let result: f64 = scope phase {
        //       let z: f64 = (x / y) * (x / y);  // (x/y) appears 2× → CSE candidate
        //       z + 1
        //   };
        // step 4: Div is medium-class (threshold 2). Add (cheap, threshold 4)
        // wouldn't hoist here. expected after CSE: the (x/y) hoists INSIDE
        // the scope's body.
        let xy_div = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("y")));
        let inner_let = let_f64(
            "z",
            ScalarExpr::BinOp(BinaryKind::Mul,
                Box::new(xy_div.clone()),
                Box::new(xy_div.clone())),
        );
        let inner_result = ScalarExpr::BinOp(
            BinaryKind::Add,
            Box::new(v("z")),
            Box::new(ScalarExpr::Const(ConstValue::F64(1.0))),
        );

        let mut body = vec![scope_stmt("phase", vec![inner_let], inner_result)];
        let mut outputs = vec![v("phase")];
        cse_in_place(&mut body, &mut outputs, "__cse_");

        // the OUTER body must still be just the one Scope statement.
        assert_eq!(body.len(), 1, "outer body should have exactly the Scope; got {:?}", body);
        // the Scope's BODY must have grown from 1 stmt (the inner let) to 2
        // (a hoisted __cse_N let for x/y, then the original `z = ...`).
        match &body[0] {
            ScalarStmt::Scope { body: scope_body, .. } => {
                assert!(
                    scope_body.len() >= 2,
                    "scope body should contain hoisted CSE let + original; got {} stmts: {:?}",
                    scope_body.len(), scope_body,
                );
                // the first stmt of the scope body should be a CSE let — the
                // hoisted `x / y`.
                match &scope_body[0] {
                    ScalarStmt::Let { name, value, .. } => {
                        assert!(name.starts_with("__cse_"),
                            "first stmt in scope body should be a CSE let, got name={name}");
                        // its value must be (x / y) — that's the hoisted candidate.
                        assert_eq!(*value, xy_div, "hoisted let must be (x / y); got {:?}", value);
                    }
                    other => panic!("first scope stmt should be a Let, got {:?}", other),
                }
            }
            other => panic!("outer body[0] should be a Scope, got {:?}", other),
        }
    }

    /// **scope sealing**: a duplicate that appears in TWO sibling scopes is NOT
    /// shared across them. each scope gets its own independent CSE pass. this
    /// is the per-scope isolation property — fancy LCA is a separate refinement
    /// (step 2b), not required for the wave-speed win.
    #[test]
    fn duplicate_across_sibling_scopes_stays_local() {
        // outer body:
        //   let a: f64 = scope sA { let p = (x / y) + (x / y); p };
        //   let b: f64 = scope sB { let q = (x / y) + (x / y); q };
        // step 4: each scope independently hoists Div (medium, threshold 2).
        let x1 = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("y")));
        let inner = |name: &str| -> Vec<ScalarStmt> {
            vec![let_f64(name, ScalarExpr::BinOp(BinaryKind::Add, Box::new(x1.clone()), Box::new(x1.clone())))]
        };

        let mut body = vec![
            scope_stmt("a", inner("p"), v("p")),
            scope_stmt("b", inner("q"), v("q")),
        ];
        let mut outputs = vec![v("a"), v("b")];
        cse_in_place(&mut body, &mut outputs, "__cse_");

        // outer body still 2 Scope stmts — neither merged nor hoisted out.
        assert_eq!(body.len(), 2, "outer body should remain 2 Scope stmts");

        // each scope's body should contain a hoisted CSE let for (x+1).
        for stmt in &body {
            match stmt {
                ScalarStmt::Scope { body: sb, .. } => {
                    let has_cse_let = sb.iter().any(|s| {
                        matches!(s, ScalarStmt::Let { name, .. } if name.starts_with("__cse_"))
                    });
                    assert!(has_cse_let, "each scope should have its own CSE let; body = {:?}", sb);
                }
                _ => panic!("expected Scope, got {:?}", stmt),
            }
        }
    }

    /// **flat-code regression**: no Scope means same behavior as before for
    /// hoist-eligible candidates. this is the "no patchwork" property —
    /// step 2 doesn't change anything for code that hasn't opted into scopes.
    #[test]
    fn flat_code_with_no_scope_unchanged_by_step_2() {
        // (x / y) * (x / y) — pure flat duplication of a medium-class op.
        let xy_div = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("y")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![ScalarExpr::BinOp(
            BinaryKind::Mul,
            Box::new(xy_div.clone()),
            Box::new(xy_div),
        )];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        // exactly one CSE let in the outer body (for x/y), output references it.
        assert_eq!(body.len(), 1, "expected exactly one CSE let in flat body");
        match &outputs[0] {
            ScalarExpr::BinOp(BinaryKind::Mul, a, b) => {
                let (ScalarExpr::Var(na), ScalarExpr::Var(nb)) = (a.as_ref(), b.as_ref()) else {
                    panic!("output should be Mul(Var, Var); got {:?}", outputs[0]);
                };
                assert_eq!(na, nb);
                assert!(na.starts_with("__cse_"));
            }
            other => panic!("output should be Mul; got {:?}", other),
        }
    }

    // ----- docs/design/23 step 4: cost-model CSE tests -----
    //
    // each test exercises ONE class of the cost table at the threshold
    // boundary: just-below stays inline (good — would have spent a register
    // for nothing), at-or-above hoists (good — actually saves recomputation
    // of an expensive node). transcendentals + memory + free-calls hoist at
    // 2; cheap ops require 4. trivial never hoists at any count.

    /// helper: build a chain of N identical outputs that each share `seed`.
    /// returns the populated outputs vec so the test asserts on `body.len()`
    /// after `cse_in_place`.
    fn shared_n_times(seed: ScalarExpr, n: usize) -> Vec<ScalarExpr> {
        (0..n)
            .map(|i| {
                ScalarExpr::BinOp(
                    BinaryKind::Add,
                    Box::new(seed.clone()),
                    Box::new(v(&format!("k{i}"))),
                )
            })
            .collect()
    }

    /// cheap-class at the threshold-1 boundary: a Mul shared 3 times must NOT
    /// hoist (cheap threshold = 4). this is the load-bearing policy reversal —
    /// the OLD pass would have hoisted any 2× share.
    #[test]
    fn cheap_below_threshold_stays_inline() {
        let mul = ScalarExpr::BinOp(BinaryKind::Mul, Box::new(v("x")), Box::new(v("y")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(mul, 3);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert!(body.is_empty(),
            "cheap-class Mul shared 3× must NOT hoist; got body = {:?}", body);
    }

    /// cheap-class at the threshold: a Mul shared 4 times MUST hoist.
    #[test]
    fn cheap_at_threshold_hoists() {
        let mul = ScalarExpr::BinOp(BinaryKind::Mul, Box::new(v("x")), Box::new(v("y")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(mul, 4);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "cheap-class Mul shared 4× must hoist exactly once; got body = {:?}", body);
    }

    /// medium-class (Div) at the threshold: 2 shares hoist.
    #[test]
    fn medium_at_threshold_hoists() {
        let div = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("y")));
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(div, 2);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "medium-class Div shared 2× must hoist exactly once; got body = {:?}", body);
    }

    /// medium-class (sqrt) at the threshold: 2 shares hoist. MethodCall
    /// path mirrors the BinOp medium path.
    #[test]
    fn medium_method_at_threshold_hoists() {
        let sqrt = ScalarExpr::MethodCall {
            receiver: Box::new(v("x")),
            method:   "sqrt".to_string(),
            args:     vec![],
        };
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(sqrt, 2);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "medium-class sqrt shared 2× must hoist; got body = {:?}", body);
    }

    /// expensive (transcendental) at the threshold: 2 shares hoist. this is
    /// the cost class where the recomputation pain is greatest.
    #[test]
    fn expensive_transcendental_at_threshold_hoists() {
        let sin = ScalarExpr::MethodCall {
            receiver: Box::new(v("x")),
            method:   "sin".to_string(),
            args:     vec![],
        };
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(sin, 2);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "expensive sin shared 2× must hoist; got body = {:?}", body);
    }

    /// expensive (exp) at single use: must NOT hoist. the policy is
    /// "always hoist when SHARED", not "always hoist".
    #[test]
    fn expensive_single_use_stays_inline() {
        let exp = ScalarExpr::MethodCall {
            receiver: Box::new(v("x")),
            method:   "exp".to_string(),
            args:     vec![],
        };
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![ScalarExpr::BinOp(
            BinaryKind::Add, Box::new(exp), Box::new(v("k")),
        )];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert!(body.is_empty(),
            "expensive single-use exp must NOT hoist; got body = {:?}", body);
    }

    /// memory-class (FieldLoadAt) at the threshold: 2 shares hoist. once the
    /// value is in a register, refetching is strictly worse than holding it.
    #[test]
    fn memory_field_load_at_threshold_hoists() {
        let load = ScalarExpr::FieldLoadAt {
            field_key:  "rho".to_string(),
            components: vec![v("i"), v("j")],
        };
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(load, 2);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "memory FieldLoadAt shared 2× must hoist; got body = {:?}", body);
    }

    /// trivial (Const, Var) at any share count: never hoists.
    #[test]
    fn trivial_never_hoists() {
        // Var shared 10 times: must NOT hoist (it's already a register name).
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(v("x"), 10);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert!(body.is_empty(),
            "trivial Var shared 10× must NOT hoist; got body = {:?}", body);

        // Const shared 10 times: must NOT hoist either.
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(ScalarExpr::Const(ConstValue::F64(2.5)), 10);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert!(body.is_empty(),
            "trivial Const shared 10× must NOT hoist; got body = {:?}", body);
    }

    /// FreeCall (unknown function): defaults to Expensive class, so 2 shares
    /// hoist. catches the conservative-bias rule.
    #[test]
    fn free_call_treated_as_expensive() {
        let call = ScalarExpr::FreeCall {
            name: "user_kernel".to_string(),
            args: vec![v("x"), v("y")],
        };
        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = shared_n_times(call, 2);
        cse_in_place(&mut body, &mut outputs, "__cse_");
        assert_eq!(body.len(), 1,
            "FreeCall shared 2× must hoist (expensive class); got body = {:?}", body);
    }

    /// **the load-bearing demonstration**: a kernel that mixes cheap and
    /// expensive operations. only the expensive shared values land in
    /// __cse_ lets; the cheap shared-twice values stay inline. this is
    /// EXACTLY the wave_speed_map win the design doc projects.
    #[test]
    fn mixed_kernel_hoists_only_expensive_shares() {
        // a 2-output kernel:
        //   out0 = sin(x) + (a * b)        // sin shared, mul shared
        //   out1 = sin(x) + (a * b) * 2    // sin shared 2×, mul shared 2×
        // sin → hoist (expensive ≥ 2). mul → stay inline (cheap < 4).
        let sin = ScalarExpr::MethodCall {
            receiver: Box::new(v("x")),
            method:   "sin".to_string(),
            args:     vec![],
        };
        let mul = ScalarExpr::BinOp(BinaryKind::Mul, Box::new(v("a")), Box::new(v("b")));
        let out0 = ScalarExpr::BinOp(BinaryKind::Add, Box::new(sin.clone()), Box::new(mul.clone()));
        let scaled = ScalarExpr::BinOp(BinaryKind::Mul,
            Box::new(mul.clone()), Box::new(ScalarExpr::Const(ConstValue::F64(2.0))));
        let out1 = ScalarExpr::BinOp(BinaryKind::Add, Box::new(sin.clone()), Box::new(scaled));

        let mut body: Vec<ScalarStmt> = vec![];
        let mut outputs = vec![out0, out1];
        cse_in_place(&mut body, &mut outputs, "__cse_");
        // exactly one CSE Let — for sin(x). a*b stays inline (cheap, only 2 shares).
        assert_eq!(body.len(), 1,
            "expected exactly one CSE Let (for sin); got body = {:?}", body);
        match &body[0] {
            ScalarStmt::Let { value, .. } => assert_eq!(*value, sin,
                "the lone hoist must be sin(x); got {:?}", value),
            other => panic!("expected Let, got {:?}", other),
        }
    }

    /// **the unit-cost-model API**: confirm `cost_class` returns the table
    /// labels for representative members of each class. guards against a
    /// careless edit of the classifier.
    #[test]
    fn cost_class_table_is_exact() {
        assert_eq!(cost_class(&v("x")),                                CostClass::Trivial);
        assert_eq!(cost_class(&ScalarExpr::Const(ConstValue::F64(1.0))), CostClass::Trivial);

        let bin = |op| ScalarExpr::BinOp(op, Box::new(v("a")), Box::new(v("b")));
        assert_eq!(cost_class(&bin(BinaryKind::Add)), CostClass::Cheap);
        assert_eq!(cost_class(&bin(BinaryKind::Sub)), CostClass::Cheap);
        assert_eq!(cost_class(&bin(BinaryKind::Mul)), CostClass::Cheap);
        assert_eq!(cost_class(&bin(BinaryKind::Div)), CostClass::Medium);
        assert_eq!(cost_class(&bin(BinaryKind::Lt)),  CostClass::Cheap);

        let mc = |name: &str| ScalarExpr::MethodCall {
            receiver: Box::new(v("x")),
            method:   name.to_string(),
            args:     vec![],
        };
        assert_eq!(cost_class(&mc("sqrt")), CostClass::Medium);
        assert_eq!(cost_class(&mc("abs")),  CostClass::Medium);
        assert_eq!(cost_class(&mc("sin")),  CostClass::Expensive);
        assert_eq!(cost_class(&mc("cos")),  CostClass::Expensive);
        assert_eq!(cost_class(&mc("exp")),  CostClass::Expensive);
        assert_eq!(cost_class(&mc("ln")),   CostClass::Expensive);
        assert_eq!(cost_class(&mc("powf")), CostClass::Expensive);

        let load = ScalarExpr::FieldLoadAt {
            field_key:  "u".to_string(),
            components: vec![v("i")],
        };
        assert_eq!(cost_class(&load), CostClass::Memory);
    }

    /// **threshold table**: trivial = MAX, cheap = 4, medium/expensive/memory = 2.
    #[test]
    fn hoist_threshold_table_is_exact() {
        assert_eq!(hoist_threshold(CostClass::Trivial),   usize::MAX);
        assert_eq!(hoist_threshold(CostClass::Cheap),     4);
        assert_eq!(hoist_threshold(CostClass::Medium),    2);
        assert_eq!(hoist_threshold(CostClass::Expensive), 2);
        assert_eq!(hoist_threshold(CostClass::Memory),    2);
    }

    /// **nested-scope correctness**: a Scope-inside-a-Scope still gets per-scope
    /// CSE. recursion descends correctly.
    #[test]
    fn nested_scopes_each_get_independent_cse() {
        // outer body:
        //   let r: f64 = scope outer {
        //       let m: f64 = scope inner {
        //           let p = (x / y) * (x / y);   // (x/y) ×2 INSIDE inner
        //           p
        //       };
        //       m + m                            // m ×2 INSIDE outer, AFTER inner closes
        //   };
        // step 4: Div is medium-class so the 2-share hoists.
        let x1 = ScalarExpr::BinOp(BinaryKind::Div, Box::new(v("x")), Box::new(v("y")));
        let inner_scope = scope_stmt(
            "m",
            vec![let_f64("p", ScalarExpr::BinOp(BinaryKind::Mul, Box::new(x1.clone()), Box::new(x1)))],
            v("p"),
        );
        let outer_scope = scope_stmt(
            "r",
            vec![inner_scope],
            // m is trivial (Var) so won't be hoisted, but the outer scope is
            // valid — we just want to verify the recursion runs cleanly.
            ScalarExpr::BinOp(BinaryKind::Add, Box::new(v("m")), Box::new(v("m"))),
        );

        let mut body = vec![outer_scope];
        let mut outputs = vec![v("r")];
        cse_in_place(&mut body, &mut outputs, "__cse_");

        // inner scope should have a CSE let for (x+1).
        match &body[0] {
            ScalarStmt::Scope { body: outer_body, .. } => {
                // outer scope's body has one statement: the inner Scope.
                assert_eq!(outer_body.len(), 1);
                match &outer_body[0] {
                    ScalarStmt::Scope { body: inner_body, .. } => {
                        let has_cse_let = inner_body.iter().any(|s| {
                            matches!(s, ScalarStmt::Let { name, .. } if name.starts_with("__cse_"))
                        });
                        assert!(has_cse_let,
                            "inner scope should contain hoisted CSE let; body = {:?}",
                            inner_body);
                    }
                    other => panic!("outer scope's first stmt should be inner Scope, got {:?}", other),
                }
            }
            other => panic!("outer body[0] should be outer Scope, got {:?}", other),
        }
    }
}
