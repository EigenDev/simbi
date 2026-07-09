// =============================================================================
// interp.rs
//
// the CPU backend: an in-process interpreter for the scalarized IR
// (`09_core_abstractions` abstraction #4 — the device-agnostic execution boundary).
//
// emit_cpu / emit_cuda / emit_kernel render a `LoweredFn` to SOURCE that is
// compiled later; this RUNS it directly. that closes the device-agnosticism
// hole — until now the substrate could emit CUDA but had no way to EXECUTE a
// generated computation on the host. `Cpu` is the first `Backend` instance;
// the source-emitting GPU path (emit_cuda / emit_kernel + the `symbi` runtime's
// JIT + dispatch) is the other. the cross-device kernel-over-field-buffers
// method extends this boundary — it needs the buffer / domain
// model that lives in `symbi-grid` / `symbi`.
//
// scope (slice 1): elemental `LoweredFn` over SCALAR params — the `scalarize`
// output for pointwise / elemental graphs (arithmetic, transcendentals,
// select). kernel-stencil evaluation (`FieldLoadAt` over field buffers + a
// coordinate loop), rank-1 array params, and generic-dim `for` loops are
// follow-ons; they panic with a clear message here.
// =============================================================================

use std::collections::HashMap;

use crate::graph::{ConstValue, Graph, NodeId};
use crate::passes::scalarize::{scalarize_kernel, BinaryKind, LoweredFn, ScalarExpr, ScalarStmt, UnaryKind};
use crate::backends::kernel::KernelEmitInputs;

/// a scalar value during interpretation: a float, or a boolean (from
/// comparisons / classifications, consumed by `Select` and bitwise-logical ops).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    F(f64),
    B(bool),
}

impl Value {
    fn as_f(self) -> f64 {
        match self {
            Value::F(x) => x,
            Value::B(b) => if b { 1.0 } else { 0.0 },
        }
    }
    fn as_b(self) -> bool {
        match self {
            Value::B(b) => b,
            Value::F(x) => x != 0.0,
        }
    }
}

type Env = HashMap<String, Value>;

/// resolves a stencil field read `field_key[coord]` to its value, for the
/// kernel interpreter. `None` in the elemental path (where `FieldLoadAt` cannot
/// appear). the coord is one integer index per spatial axis.
type FieldRead<'a> = Option<&'a dyn Fn(&str, &[i64]) -> f64>;

/// a compute backend for the scalarized IR. `Cpu` interprets in-process; the
/// GPU instances emit + JIT + dispatch (see the module header).
pub trait Backend {
    /// evaluate an elemental `LoweredFn` given its scalar input values in
    /// parameter order; returns one f64 per result component.
    fn eval_elemental(&self, f: &LoweredFn, inputs: &[f64]) -> Vec<f64>;
}

/// the CPU backend: a direct interpreter over the scalarized IR.
pub struct Cpu;

impl Backend for Cpu {
    fn eval_elemental(&self, f: &LoweredFn, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(
            f.params.len(),
            inputs.len(),
            "Cpu::eval_elemental: '{}' expects {} inputs, got {}",
            f.name, f.params.len(), inputs.len(),
        );
        let mut env = Env::new();
        for (p, &v) in f.params.iter().zip(inputs) {
            assert!(
                p.array_len.is_none(),
                "interp: rank-1 array param '{}' not supported in slice-1 elemental eval",
                p.name,
            );
            env.insert(p.name.clone(), Value::F(v));
        }
        for stmt in &f.body {
            // top-level break would escape the kernel — disallowed by lowering.
            let _ = exec_stmt(stmt, &mut env, None);
        }
        f.results.iter().map(|r| eval_expr(r, &env, None).as_f()).collect()
    }
}

/// `ScalarStmt::Break` propagates upward; the enclosing `For` catches it and
/// stops iterating. `If`/leaf statements emit `Continue`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow { Continue, Break }

fn exec_stmt(stmt: &ScalarStmt, env: &mut Env, fr: FieldRead) -> Flow {
    match stmt {
        ScalarStmt::Let { name, value, .. } | ScalarStmt::LetMut { name, init: value, .. } => {
            let v = eval_expr(value, env, fr);
            env.insert(name.clone(), v);
            Flow::Continue
        }
        ScalarStmt::Assign { name, value } => {
            let v = eval_expr(value, env, fr);
            env.insert(name.clone(), v);
            Flow::Continue
        }
        ScalarStmt::CompoundAssign { name, op, value } => {
            let rhs = eval_expr(value, env, fr);
            let cur = *env.get(name)
                .unwrap_or_else(|| panic!("interp: compound-assign to unbound '{name}'"));
            env.insert(name.clone(), eval_binop(*op, cur, rhs));
            Flow::Continue
        }
        ScalarStmt::For { iter, bound, body } => {
            use crate::dim::DimExpr;
            let DimExpr::Literal(n) = bound;
            'outer: for i in 0..*n {
                env.insert(iter.clone(), Value::F(i as f64));
                for s in body {
                    if exec_stmt(s, env, fr) == Flow::Break {
                        break 'outer;
                    }
                }
            }
            Flow::Continue
        }
        ScalarStmt::If { cond, then_body } => {
            if eval_expr(cond, env, fr).as_b() {
                for s in then_body {
                    if exec_stmt(s, env, fr) == Flow::Break {
                        return Flow::Break;
                    }
                }
            }
            Flow::Continue
        }
        ScalarStmt::Break => Flow::Break,
        ScalarStmt::Scope { name, body, result, .. } => {
            // Scope semantics for the interpreter: execute body (its lets
            // mutate env), evaluate `result`, then bind `name` to it in env.
            // **scoping is a CODEGEN concern** (lifetime hints to nvcc /
            // rustc); the interpreter doesn't restore env afterward because
            // doing so would diverge from the codegen path's semantics under
            // SSA-fresh names — the body's bindings are all fresh `__cse_N`
            // names that downstream stmts don't reference anyway. break does
            // not propagate out of a Scope (it's not a loop).
            for s in body {
                if exec_stmt(s, env, fr) == Flow::Break {
                    return Flow::Break;
                }
            }
            let v = eval_expr(result, env, fr);
            env.insert(name.clone(), v);
            Flow::Continue
        }
        ScalarStmt::IfElse { cond, then_body, else_body, .. } => {
            // a REAL branch: execute ONLY the taken arm (each arm ends with a
            // `name = <arm result>` Assign that binds `name` in env). this
            // matches BOTH the f64 host oracle and the rendered `if/else` — the
            // untaken arm's ops never run, so an out-of-domain transcendental in
            // the dead arm cannot poison the result. break does not propagate
            // out (not a loop).
            let arm = if eval_expr(cond, env, fr).as_b() { then_body } else { else_body };
            for s in arm {
                if exec_stmt(s, env, fr) == Flow::Break {
                    return Flow::Break;
                }
            }
            Flow::Continue
        }
    }
}

fn eval_expr(e: &ScalarExpr, env: &Env, fr: FieldRead) -> Value {
    match e {
        ScalarExpr::Const(c) => eval_const(c),
        ScalarExpr::Var(name) => *env.get(name)
            .unwrap_or_else(|| panic!("interp: unbound variable '{name}'")),
        ScalarExpr::BinOp(op, a, b) => eval_binop(*op, eval_expr(a, env, fr), eval_expr(b, env, fr)),
        ScalarExpr::UnaryOp(UnaryKind::Neg, a) => Value::F(-eval_expr(a, env, fr).as_f()),
        ScalarExpr::UnaryOp(UnaryKind::Not, a) => Value::B(!eval_expr(a, env, fr).as_b()),
        // numeric promotion: the interpreter is f64-based, so the int->float cast is
        // the identity (the value is already evaluated as f64).
        ScalarExpr::Cast { value, .. } => Value::F(eval_expr(value, env, fr).as_f()),
        ScalarExpr::Select { cond, then, else_ } => {
            if eval_expr(cond, env, fr).as_b() {
                eval_expr(then, env, fr)
            } else {
                eval_expr(else_, env, fr)
            }
        }
        ScalarExpr::MethodCall { receiver, method, args } => {
            let recv = eval_expr(receiver, env, fr).as_f();
            let argv: Vec<f64> = args.iter().map(|a| eval_expr(a, env, fr).as_f()).collect();
            eval_method(recv, method, &argv)
        }
        ScalarExpr::FieldLoadAt { field_key, components } => {
            let read = fr.unwrap_or_else(|| panic!(
                "interp: field stencil read '{field_key}[coord]' outside a kernel context"
            ));
            let coord: Vec<i64> = components.iter()
                .map(|c| eval_expr(c, env, fr).as_f().round() as i64)
                .collect();
            Value::F(read(field_key, &coord))
        }
        ScalarExpr::FreeCall { name, .. } => panic!(
            "interp: opaque free call '{name}(...)' is not interpretable (its body lives \
             outside the IR); supply a host implementation or inline it"
        ),
        ScalarExpr::IndexInto { container, .. } => panic!(
            "interp: array indexing into '{container}' not supported (rank-1 array params)"
        ),
    }
}

fn eval_const(c: &ConstValue) -> Value {
    match c {
        ConstValue::F64(v) => Value::F(*v),
        ConstValue::F32(v) => Value::F(*v as f64),
        ConstValue::I32(v) => Value::F(*v as f64),
        ConstValue::U32(v) => Value::F(*v as f64),
        ConstValue::Bool(b) => Value::B(*b),
    }
}

fn eval_binop(op: BinaryKind, a: Value, b: Value) -> Value {
    use BinaryKind::*;
    match op {
        Add => Value::F(a.as_f() + b.as_f()),
        Sub => Value::F(a.as_f() - b.as_f()),
        Mul => Value::F(a.as_f() * b.as_f()),
        Div => Value::F(a.as_f() / b.as_f()),
        Eq => Value::B(a.as_f() == b.as_f()),
        Ne => Value::B(a.as_f() != b.as_f()),
        Lt => Value::B(a.as_f() < b.as_f()),
        Le => Value::B(a.as_f() <= b.as_f()),
        Gt => Value::B(a.as_f() > b.as_f()),
        Ge => Value::B(a.as_f() >= b.as_f()),
        BitOr => Value::B(a.as_b() | b.as_b()),
        BitAnd => Value::B(a.as_b() & b.as_b()),
        BitXor => Value::B(a.as_b() ^ b.as_b()),
    }
}

fn eval_method(recv: f64, method: &str, args: &[f64]) -> Value {
    match method {
        "sqrt" => Value::F(recv.sqrt()),
        // abs/min/max use the plain `a<b?a:b` TERNARY, NOT the
        // NaN-symmetric f64::abs/min/max — so the interpreter (the carrier oracle)
        // matches the cuda emit, the cranelift jit, and the f64/f32 `Numeric`
        // carrier bit-for-bit at NaN / signed-zero.
        "abs" => Value::F(if recv < 0.0 { -recv } else { recv }),
        "min" => Value::F(if recv < args[0] { recv } else { args[0] }),
        "max" => Value::F(if recv > args[0] { recv } else { args[0] }),
        "floor" => Value::F(recv.floor()),
        "ceil" => Value::F(recv.ceil()),
        "round" => Value::F(recv.round()),
        "trunc" => Value::F(recv.trunc()),
        // integer floor division held in the f64-based interpreter: the operands
        // are exact small integers, so floor(a/b) is the exact euclidean quotient.
        "div_euclid" => Value::F((recv / args[0]).floor()),
        "sin" => Value::F(recv.sin()),
        "cos" => Value::F(recv.cos()),
        "tan" => Value::F(recv.tan()),
        "asin" => Value::F(recv.asin()),
        "acos" => Value::F(recv.acos()),
        "atan" => Value::F(recv.atan()),
        "atan2" => Value::F(recv.atan2(args[0])),
        "exp" => Value::F(recv.exp()),
        "exp2" => Value::F(recv.exp2()),
        "ln" => Value::F(recv.ln()),
        "log2" => Value::F(recv.log2()),
        "log10" => Value::F(recv.log10()),
        "sinh" => Value::F(recv.sinh()),
        "cosh" => Value::F(recv.cosh()),
        "tanh" => Value::F(recv.tanh()),
        "asinh" => Value::F(recv.asinh()),
        "acosh" => Value::F(recv.acosh()),
        "atanh" => Value::F(recv.atanh()),
        "powf" => Value::F(recv.powf(args[0])),
        "powi" => Value::F(recv.powi(args[0] as i32)),
        "hypot" => Value::F(recv.hypot(args[0])),
        "is_finite" => Value::B(recv.is_finite()),
        "is_nan" => Value::B(recv.is_nan()),
        other => panic!("interp: unsupported method '.{other}()'"),
    }
}

// ---- kernel interpreter (the stencil case, over host buffers) -----------
//
// runs a scalarized KERNEL over a domain on host f64 buffers: iterates the
// cells, resolves field reads (cell loads + stencil-shifted `FieldLoadAt`),
// evaluates the shared body, and writes the per-cell outputs. the SAME
// `KernelEmitInputs` spec either emits CUDA (`emit_kernel_from_lowering`) or
// runs here on CPU. buffers are separate in/out (a stencil sweep that wrote
// in-place would corrupt neighbour reads; in-place godunov is the caller's
// double-buffering concern). this is the device-agnostic execution closing
// the loop with the kernels the substrate generates.

/// a read-only host field buffer with its view-ABI layout: `lo[axis]` is the
/// buffer's per-axis origin, `extent[axis]` its per-axis size (the stride
/// source). indexing matches `emit::emit_flat_index`.
pub struct CpuField<'a> {
    pub data:   &'a [f64],
    pub lo:     &'a [i32],
    pub extent: &'a [u32],
}

/// a writable host field buffer (the kernel's output target).
pub struct CpuFieldMut<'a> {
    pub data:   &'a mut [f64],
    pub lo:     &'a [i32],
    pub extent: &'a [u32],
}

/// affine view offset `sum_a (coord[a] - lo[a]) * strides[a]`. the stride prefix
/// product is NOT re-spelled here — it is derived from the ONE canonical
/// definition `symbi_algebra::strides_from_extent` (the same formula `Layout` /
/// `Domain` / the runtime `View` and the AOT kernels all use). routing through it
/// means the interpreter is a faithful oracle for stencil (neighbour) reads and
/// cannot drift from the real `Field` layout (axis-0-fastest / physical-x-fastest).
fn flat_index(coord: &[i64], lo: &[i32], extent: &[u32], ndim: usize) -> usize {
    let ext_i64: [i64; 4] = std::array::from_fn(|a| if a < ndim { extent[a] as i64 } else { 1 });
    let mut strides = [0i64; 4];
    symbi_algebra::strides_from_extent(&ext_i64[..ndim], &mut strides[..ndim]);
    let mut idx: i64 = 0;
    for ax in 0..ndim {
        idx += (coord[ax] - lo[ax] as i64) * strides[ax];
    }
    assert!(idx >= 0, "interp: negative flat index (coord out of buffer bounds)");
    idx as usize
}

impl Cpu {
    /// run a scalarized kernel over the domain `[dom_lo, dom_lo + grid_size)`
    /// per axis. `inputs` / `outputs` / `scalars` line up with `spec`'s
    /// `field_inputs` / `field_writes` / `scalar_params`.
    pub fn run_kernel(
        &self,
        graph: &Graph,
        spec: &KernelEmitInputs,
        inputs: &[CpuField],
        outputs: &mut [CpuFieldMut],
        scalars: &[f64],
        grid_sizes: &[u32],
        dom_los: &[i32],
    ) {
        let ndim = spec.ndim as usize;
        assert_eq!(inputs.len(), spec.field_inputs.len(), "input buffer count");
        assert_eq!(outputs.len(), spec.field_writes.len(), "output buffer count");
        assert_eq!(scalars.len(), spec.scalar_params.len(), "scalar count");
        assert_eq!(grid_sizes.len(), ndim);
        assert_eq!(dom_los.len(), ndim);

        // scalarize once: a shared body + one output expr per write.
        let output_nodes: Vec<NodeId> = spec.field_writes.iter().map(|(_, _, n)| *n).collect();
        let sc = scalarize_kernel(graph, &output_nodes);

        // field_key -> input buffer, for cell loads and stencil reads.
        let mut in_map: HashMap<&str, &CpuField> = HashMap::new();
        for ((key, _), buf) in spec.field_inputs.iter().zip(inputs.iter()) {
            in_map.insert(key.as_str(), buf);
        }
        let field_read = |key: &str, coord: &[i64]| -> f64 {
            let f = in_map.get(key)
                .unwrap_or_else(|| panic!("interp: no buffer for field '{key}'"));
            f.data[flat_index(coord, f.lo, f.extent, ndim)]
        };

        let total: usize = grid_sizes.iter().map(|&g| g as usize).product();
        for flat in 0..total {
            // unflatten the iteration index into a domain coordinate.
            let mut coord = vec![0i64; ndim];
            let mut rem = flat;
            for ax in (0..ndim).rev() {
                let g = grid_sizes[ax] as usize;
                coord[ax] = dom_los[ax] as i64 + (rem % g) as i64;
                rem /= g;
            }
            // per-cell environment: coord components, field cell loads, scalars.
            let mut env = Env::new();
            for ax in 0..ndim {
                env.insert(format!("_coord_{ax}"), Value::F(coord[ax] as f64));
            }
            for (key, _) in spec.field_inputs {
                env.insert(key.clone(), Value::F(field_read(key, &coord)));
            }
            for (name, &v) in spec.scalar_params.iter().zip(scalars) {
                env.insert(name.clone(), Value::F(v));
            }
            for stmt in &sc.body {
                let _ = exec_stmt(stmt, &mut env, Some(&field_read));
            }
            for (out, expr) in outputs.iter_mut().zip(sc.outputs.iter()) {
                let v = eval_expr(expr, &env, Some(&field_read)).as_f();
                out.data[flat_index(&coord, out.lo, out.extent, ndim)] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementTy, ElementWiseOp, Graph, Symbol, TensorTy, scalarize};

    fn scalar_param(g: &mut Graph, name: &str) -> crate::graph::NodeId {
        g.add_param(Symbol::intern(name), TensorTy::scalar(ElementTy::F64), None)
    }

    // the LAYOUT chokepoint gate: the interpreter's view offset must agree
    // bit-for-bit with the canonical `symbi_algebra::Layout` over a sweep of
    // non-square / >=2D extents and asymmetric coords. if anyone re-spells the
    // stride formula in `flat_index` and drifts from the canonical definition
    // (the latent axis-order class of bug — last-axis-fastest vs axis-0-fastest),
    // this fails. `Layout` itself is already pinned == `Domain::flat_index` in
    // symbi-algebra, so transitively interp == Domain == kernels.
    #[test]
    fn flat_index_equals_canonical_layout() {
        use symbi_algebra::Layout;

        // deterministic splitmix prng — no external dep (workspace forbids rand
        // in build-affecting code), mirrors the symbi-algebra law-test rng.
        struct Rng(u64);
        impl Rng {
            fn bits(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^ (z >> 31)
            }
            fn in_range(&mut self, lo: i32, hi: i32) -> i32 {
                lo + (self.bits() % ((hi - lo) as u64)) as i32
            }
        }
        let mut rng = Rng(0xFA57_0FF5);

        for _ in 0..3000 {
            // random origin + non-square extents per axis, 2D and 3D.
            let lo3: [i32; 3] = std::array::from_fn(|_| rng.in_range(-4, 4));
            let ext3: [u32; 3] = std::array::from_fn(|_| rng.in_range(1, 9) as u32);

            for ndim in 2..=3usize {
                let lo = &lo3[..ndim];
                let ext = &ext3[..ndim];
                // sweep every cell in the allocated buffer (asymmetric coords).
                let total: usize = ext.iter().map(|&e| e as usize).product();
                for flat in 0..total {
                    let mut coord = [0i64; 3];
                    let mut rem = flat;
                    for ax in 0..ndim {
                        coord[ax] = lo[ax] as i64 + (rem % ext[ax] as usize) as i64;
                        rem /= ext[ax] as usize;
                    }
                    let got = flat_index(&coord[..ndim], lo, ext, ndim);
                    let want = if ndim == 2 {
                        Layout::<2>::new([lo[0], lo[1]], [ext[0], ext[1]])
                            .at([coord[0] as i32, coord[1] as i32])
                    } else {
                        Layout::<3>::new([lo[0], lo[1], lo[2]], [ext[0], ext[1], ext[2]])
                            .at([coord[0] as i32, coord[1] as i32, coord[2] as i32])
                    };
                    assert_eq!(got, want, "interp::flat_index != Layout::at at coord {coord:?}");
                }
            }
        }
    }

    #[test]
    fn evaluates_arithmetic_against_hand_computed_value() {
        // out = a * 2.0 + b   (the emit_kernel arithmetic graph, run on CPU).
        let mut g = Graph::new();
        let a = scalar_param(&mut g, "a");
        let b = scalar_param(&mut g, "b");
        let two = g.add_const(ConstValue::F64(2.0), None);
        let scaled = g.element_wise(ElementWiseOp::Mul, vec![a, two], None);
        let summed = g.element_wise(ElementWiseOp::Add, vec![scaled, b], None);
        let f = scalarize(&g, summed, "compute");

        // params come out in insertion order: [a, b].
        let out = Cpu.eval_elemental(&f, &[3.0, 4.0]);
        assert_eq!(out, vec![3.0 * 2.0 + 4.0]); // 10.0
        let out2 = Cpu.eval_elemental(&f, &[-1.0, 0.5]);
        assert_eq!(out2, vec![-1.0 * 2.0 + 0.5]); // -1.5
    }

    #[test]
    fn floor_div_floors_toward_negative_infinity() {
        // out = a floor-div b, the index-space primitive: must match
        // i64::div_euclid for positive divisors, including negative numerators
        // (a fine ghost index below zero mapping to its coarse parent).
        let mut g = Graph::new();
        let a = g.add_param(Symbol::intern("a"), TensorTy::scalar(ElementTy::I32), None);
        let b = g.add_param(Symbol::intern("b"), TensorTy::scalar(ElementTy::I32), None);
        let q = g.element_wise(ElementWiseOp::FloorDiv, vec![a, b], None);
        let f = scalarize(&g, q, "fdiv");

        for (aa, bb) in [(5i64, 2i64), (4, 2), (0, 2), (-1, 2), (-2, 2), (-3, 2), (-4, 2), (7, 3), (-7, 3)] {
            let out = Cpu.eval_elemental(&f, &[aa as f64, bb as f64]);
            assert_eq!(
                out, vec![aa.div_euclid(bb) as f64],
                "floor_div({aa}, {bb}) disagrees with div_euclid"
            );
        }
    }

    #[test]
    fn evaluates_a_transcendental() {
        // out = sqrt(x).
        let mut g = Graph::new();
        let x = scalar_param(&mut g, "x");
        let r = g.element_wise(ElementWiseOp::Sqrt, vec![x], None);
        let f = scalarize(&g, r, "root");
        let out = Cpu.eval_elemental(&f, &[9.0]);
        assert_eq!(out, vec![3.0]);
    }

    #[test]
    fn evaluates_a_select() {
        // out = if x < 0 { -x } else { x }  (abs via select).
        let mut g = Graph::new();
        let x = scalar_param(&mut g, "x");
        let zero = g.add_const(ConstValue::F64(0.0), None);
        let neg = g.element_wise(ElementWiseOp::Neg, vec![x], None);
        let cond = g.element_wise(ElementWiseOp::Lt, vec![x, zero], None);
        let sel = g.select(cond, neg, x, None);
        let f = scalarize(&g, sel, "absish");
        assert_eq!(Cpu.eval_elemental(&f, &[-5.0]), vec![5.0]);
        assert_eq!(Cpu.eval_elemental(&f, &[ 2.0]), vec![2.0]);
    }

    // ----- docs/design/23: Scope interpreter semantics test -----

    /// the interpreter must execute a `ScalarStmt::Scope` correctly: run the
    /// inner body, evaluate `result`, bind `result`'s value to the outer name.
    /// since scoping is a CODEGEN concern (it tells nvcc/rustc about lifetimes),
    /// the interpreter's numerical answer must be IDENTICAL whether the body
    /// uses Scope or flat lets. that property is what makes scope-aware lowering
    /// safe: it doesn't change semantics, only codegen.
    #[test]
    fn scope_is_semantically_transparent_in_interpreter() {
        use crate::passes::scalarize::{LoweredFn, LoweredParam, ScalarExpr, ScalarStmt, BinaryKind};

        // build a tiny LoweredFn BY HAND using a Scope for a controlled
        // test of the Scope arm in exec_stmt — independent of whatever the
        // scalarize pass might or might not produce.
        //
        // semantics: out = (a + b) * a  computed via a Scope binding `__t1 = a + b`
        // then returning `__t1 * a`.
        let scope_form = LoweredFn {
            name: "scope_form".to_string(),
            params: vec![
                LoweredParam::scalar("a".to_string(), ElementTy::F64),
                LoweredParam::scalar("b".to_string(), ElementTy::F64),
            ],
            body: vec![
                ScalarStmt::Scope {
                    name:    "out".to_string(),
                    element: ElementTy::F64,
                    body: vec![
                        ScalarStmt::Let {
                            name:    "__t1".to_string(),
                            element: ElementTy::F64,
                            value: ScalarExpr::BinOp(
                                BinaryKind::Add,
                                Box::new(ScalarExpr::Var("a".to_string())),
                                Box::new(ScalarExpr::Var("b".to_string())),
                            ),
                        },
                    ],
                    result: ScalarExpr::BinOp(
                        BinaryKind::Mul,
                        Box::new(ScalarExpr::Var("__t1".to_string())),
                        Box::new(ScalarExpr::Var("a".to_string())),
                    ),
                },
            ],
            results: vec![ScalarExpr::Var("out".to_string())],
            result_element: ElementTy::F64,
            result_shape:   vec![],
        };

        // equivalent FLAT form: same math, no scope.
        let flat_form = LoweredFn {
            name: "flat_form".to_string(),
            params: vec![
                LoweredParam::scalar("a".to_string(), ElementTy::F64),
                LoweredParam::scalar("b".to_string(), ElementTy::F64),
            ],
            body: vec![
                ScalarStmt::Let {
                    name:    "__t1".to_string(),
                    element: ElementTy::F64,
                    value: ScalarExpr::BinOp(
                        BinaryKind::Add,
                        Box::new(ScalarExpr::Var("a".to_string())),
                        Box::new(ScalarExpr::Var("b".to_string())),
                    ),
                },
                ScalarStmt::Let {
                    name:    "out".to_string(),
                    element: ElementTy::F64,
                    value: ScalarExpr::BinOp(
                        BinaryKind::Mul,
                        Box::new(ScalarExpr::Var("__t1".to_string())),
                        Box::new(ScalarExpr::Var("a".to_string())),
                    ),
                },
            ],
            results: vec![ScalarExpr::Var("out".to_string())],
            result_element: ElementTy::F64,
            result_shape:   vec![],
        };

        // run both, on the same inputs.
        for &(a, b) in &[(3.0_f64, 4.0), (-1.0, 0.5), (1.5, -2.0)] {
            let scope_out = Cpu.eval_elemental(&scope_form, &[a, b]);
            let flat_out  = Cpu.eval_elemental(&flat_form,  &[a, b]);
            assert_eq!(scope_out, flat_out,
                "scope_form and flat_form must produce IDENTICAL results for a={a}, b={b}");
            // sanity-check the actual value too.
            assert_eq!(scope_out, vec![(a + b) * a]);
        }
    }
}
