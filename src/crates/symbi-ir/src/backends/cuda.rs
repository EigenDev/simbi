// =============================================================================
// emit_cuda.rs
//
// CUDA backend for the tensor IR: LoweredFn -> CUDA C++ source string.
//
// emission shape (V1, elemental-style):
//
//   // (struct decl only when rank > 0)
//   struct <name>_out { <element> _0; <element> _1; ... };
//
//   __device__ inline <return_type> <name>(<scalar params>) {
//       <let statements>
//       return <return expression>;
//   }
//
// return type:
//   - rank-0: the scalar element type (e.g., `double`)
//   - rank-N: a struct with named fields _0, _1, ..., _{N-1}
//
// the macro layer (R.5) is the eventual consumer; CUDA syntax here is
// pure device-function code, no kernel-launch (__global__) shape yet.
// =============================================================================

use crate::graph::{ConstValue, Graph, NodeId};
use crate::passes::scalarize::scalarize;
use crate::{BinaryKind, ElementTy, LoweredFn, ScalarExpr, ScalarStmt, UnaryKind};

// =============================================================================
// per-cell source-kernel ABI emit (the primary path's sibling to the stencil
// emit in backends/kernel.rs). takes a scalar source graph + ordered param
// names + per-component output NodeIds and produces one `extern "C" __global__`
// over the source ABI:
//
//   extern "C" __global__ void <entry>(
//       const double* param_0, ... const double* param_{N-1},  // inputs
//       double* out_0,          ... double* out_{M-1},         // outputs
//       unsigned int n_cells                                    // grid size
//   ) {
//       unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//       if (i >= n_cells) return;
//       auto <param_name> = param_<k>[i];                      // per-cell reads
//       ...
//       { /* component k */ <body stmts> out_<k>[i] = <result>; }
//   }
//
// the per-component body comes from `scalarize` + the existing `emit_stmt` /
// `emit_expr` — the SAME machinery the stencil path uses. each component is its
// own brace scope so the `__cse_*` / `__t_*` temps don't collide across
// components. the param NAMES (`rho`, `vel_0`, ...) match `Op::Param` leaves in
// the graph, so the scalarized body references them transparently.
// =============================================================================

/// emit a per-cell source kernel as `extern "C" __global__` CUDA source over the
/// source ABI. `params` is the ordered input-buffer manifest (positional
/// `param_<k>`); `outputs` is one NodeId per output component (`out_<k>`).
pub fn emit_source_kernel(
    graph: &Graph,
    params: &[String],
    outputs: &[NodeId],
    entry_name: &str,
) -> String {
    let mut s = String::new();

    // signature.
    s.push_str(&format!("extern \"C\" __global__ void {entry_name}(\n"));
    for k in 0..params.len() {
        s.push_str(&format!("    const double* param_{k},\n"));
    }
    for k in 0..outputs.len() {
        s.push_str(&format!("    double* out_{k},\n"));
    }
    s.push_str("    unsigned int n_cells\n");
    s.push_str(") {\n");

    // thread index + bounds.
    s.push_str("    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    if (i >= n_cells) return;\n\n");

    // per-cell input reads. the param NAMES match the graph's `Op::Param`
    // leaves, so the scalarized body references them transparently. NVRTC
    // reads them as `auto` (CUDA C++14) — precision propagates from the
    // buffer type.
    for (k, p) in params.iter().enumerate() {
        s.push_str(&format!("    auto {p} = param_{k}[i];\n"));
    }
    s.push('\n');

    // per-component scoped body. each output is scalarized independently into
    // a LoweredFn; its body statements + result expression land inside the
    // component's brace block, so `__cse_*` / `__t_*` temps stay component-
    // local (no cross-component collision).
    for (k, &out_node) in outputs.iter().enumerate() {
        s.push_str(&format!("    {{ /* component {k} */\n"));
        let f = scalarize(graph, out_node, "source_term");
        for stmt in &f.body {
            s.push_str("        ");
            emit_stmt(&mut s, stmt);
            s.push('\n');
        }
        debug_assert_eq!(
            f.results.len(), 1,
            "source-kernel output node must be rank-0 (one scalar component)",
        );
        s.push_str(&format!("        out_{k}[i] = "));
        emit_expr(&mut s, &f.results[0]);
        s.push_str(";\n");
        s.push_str("    }\n");
    }

    s.push_str("}\n");
    s
}

/// emit a LoweredFn as a CUDA C++ source string. for rank > 0 returns,
/// a struct decl precedes the function.
pub fn emit_cuda(f: &LoweredFn) -> String {
    let mut out = String::new();
    let needs_struct = !f.result_shape.is_empty() || f.results.len() > 1;
    if needs_struct {
        emit_struct(&mut out, f);
        out.push('\n');
    }
    emit_signature(&mut out, f, needs_struct);
    out.push_str(" {\n");
    for stmt in &f.body {
        out.push_str("    ");
        emit_stmt(&mut out, stmt);
        out.push('\n');
    }
    out.push_str("    return ");
    emit_return(&mut out, f, needs_struct);
    out.push_str(";\n}\n");
    out
}

fn emit_struct(out: &mut String, f: &LoweredFn) {
    out.push_str("struct ");
    out.push_str(&f.name);
    out.push_str("_out { ");
    let ty = cuda_type_name(f.result_element);
    for i in 0..f.results.len() {
        out.push_str(ty);
        out.push_str(&format!(" _{}; ", i));
    }
    out.push_str("};");
}

fn emit_signature(out: &mut String, f: &LoweredFn, needs_struct: bool) {
    // collect const-generic identifiers used by array params; CUDA
    // doesn't have const-generic functions natively, so we emit a
    // C++ template<int D, ...> prefix.
    let mut generics: Vec<String> = Vec::new();
    for p in &f.params {
        if let Some(crate::DimExpr::Generic(sym)) = &p.array_len {
            let name = sym.as_str().to_string();
            if !generics.contains(&name) {
                generics.push(name);
            }
        }
    }
    if !generics.is_empty() {
        out.push_str("template<");
        for (i, g) in generics.iter().enumerate() {
            if i > 0 { out.push_str(", "); }
            out.push_str("int ");
            out.push_str(g);
        }
        out.push_str("> ");
    }

    out.push_str("__device__ inline ");
    if needs_struct {
        out.push_str(&f.name);
        out.push_str("_out");
    } else {
        out.push_str(cuda_type_name(f.result_element));
    }
    out.push(' ');
    out.push_str(&f.name);
    out.push('(');
    for (i, p) in f.params.iter().enumerate() {
        if i > 0 { out.push_str(", "); }
        match &p.array_len {
            None => {
                out.push_str(cuda_type_name(p.element));
                out.push(' ');
                out.push_str(&p.name);
            }
            Some(_) => {
                // arrays as `const <ty> (&name)[<len>]` for type-safe pass-by-reference;
                // since C++ template arrays decay to pointers easily, we emit by ref.
                out.push_str("const ");
                out.push_str(cuda_type_name(p.element));
                out.push_str(" (&");
                out.push_str(&p.name);
                out.push_str(")[");
                match &p.array_len {
                    Some(crate::DimExpr::Literal(n)) => out.push_str(&n.to_string()),
                    Some(crate::DimExpr::Generic(sym)) => out.push_str(sym.as_str()),
                    None => unreachable!(),
                }
                out.push(']');
            }
        }
    }
    out.push(')');
}

pub(crate) fn emit_stmt(out: &mut String, stmt: &ScalarStmt) {
    match stmt {
        ScalarStmt::Let { name, element, value } => {
            out.push_str(cuda_type_name(*element));
            out.push(' ');
            out.push_str(name);
            out.push_str(" = ");
            emit_expr(out, value);
            out.push(';');
        }
        ScalarStmt::LetMut { name, element, init } => {
            // CUDA doesn't distinguish mut/non-mut; emit a plain decl.
            out.push_str(cuda_type_name(*element));
            out.push(' ');
            out.push_str(name);
            out.push_str(" = ");
            emit_expr(out, init);
            out.push(';');
        }
        ScalarStmt::CompoundAssign { name, op, value } => {
            out.push_str(name);
            out.push(' ');
            out.push_str(cuda_binop(*op));
            out.push_str("= ");
            emit_expr(out, value);
            out.push(';');
        }
        ScalarStmt::Assign { name, value } => {
            // F2.F: plain assignment (Fold body's accumulator update).
            out.push_str(name);
            out.push_str(" = ");
            emit_expr(out, value);
            out.push(';');
        }
        ScalarStmt::For { iter, bound, body } => {
            // pragma unroll for compile-time constant bounds; nvcc will
            // unroll when D is a template parameter or literal.
            out.push_str("#pragma unroll\n    ");
            out.push_str("for (int ");
            out.push_str(iter);
            out.push_str(" = 0; ");
            out.push_str(iter);
            out.push_str(" < ");
            match bound {
                crate::DimExpr::Literal(n) => out.push_str(&n.to_string()),
                crate::DimExpr::Generic(sym) => out.push_str(sym.as_str()),
            }
            out.push_str("; ++");
            out.push_str(iter);
            out.push_str(") { ");
            for s in body {
                emit_stmt(out, s);
                out.push(' ');
            }
            out.push('}');
        }
        ScalarStmt::If { cond, then_body } => {
            out.push_str("if (");
            emit_expr(out, cond);
            out.push_str(") { ");
            for s in then_body {
                emit_stmt(out, s);
                out.push(' ');
            }
            out.push('}');
        }
        ScalarStmt::Break => {
            out.push_str("break;");
        }
        ScalarStmt::Scope { name, element, body, result } => {
            // CUDA has no block-expression form, so declare `name` in the
            // outer scope, write inside the inner `{ }`, and rely on the
            // brace to kill all inner locals. nvcc gets clean lifetime
            // information for the body's temps. see docs/design/23.
            out.push_str(cuda_type_name(*element));
            out.push(' ');
            out.push_str(name);
            out.push_str("; { ");
            for s in body {
                emit_stmt(out, s);
                out.push(' ');
            }
            out.push_str(name);
            out.push_str(" = ");
            emit_expr(out, result);
            out.push_str("; }");
        }
        ScalarStmt::IfElse { outs, cond, then_body, else_body } => {
            // declare the N result slots in the outer scope; each arm body ends
            // with `outs[j] = <arm result j>`. a real C `if (cond) { } else { }`
            // — ONE arm runs (a divergent warp runs both, never worse than a
            // blend). the carrier-portable early-out `if`, DUAL of iterate's `for`.
            for (name, element) in outs {
                out.push_str(cuda_type_name(*element));
                out.push(' ');
                out.push_str(name);
                out.push_str("; ");
            }
            out.push_str("if (");
            emit_expr(out, cond);
            out.push_str(") { ");
            for s in then_body {
                emit_stmt(out, s);
                out.push(' ');
            }
            out.push_str("} else { ");
            for s in else_body {
                emit_stmt(out, s);
                out.push(' ');
            }
            out.push('}');
        }
    }
}

fn emit_return(out: &mut String, f: &LoweredFn, needs_struct: bool) {
    if !needs_struct {
        emit_expr(out, &f.results[0]);
        return;
    }
    // brace-init list: `<name>_out { e_0, e_1, ... }`
    out.push_str(&f.name);
    out.push_str("_out { ");
    for (i, e) in f.results.iter().enumerate() {
        if i > 0 { out.push_str(", "); }
        emit_expr(out, e);
    }
    out.push_str(" }");
}

pub(crate) fn emit_expr(out: &mut String, e: &ScalarExpr) {
    match e {
        ScalarExpr::Const(v) => emit_const(out, v),
        ScalarExpr::Var(name) => out.push_str(name),
        ScalarExpr::BinOp(kind, a, b) => {
            out.push('(');
            emit_expr(out, a);
            out.push(' ');
            out.push_str(cuda_binop(*kind));
            out.push(' ');
            emit_expr(out, b);
            out.push(')');
        }
        ScalarExpr::UnaryOp(UnaryKind::Neg, a) => {
            out.push_str("(-");
            emit_expr(out, a);
            out.push(')');
        }
        ScalarExpr::UnaryOp(UnaryKind::Not, a) => {
            out.push_str("(!");
            emit_expr(out, a);
            out.push(')');
        }
        ScalarExpr::Cast { to, value } => {
            // numeric promotion: `(<float-ty>)(value)` — explicit cast (matches the
            // IR; handles f32 kernels rather than relying on C's int*double promote).
            out.push('(');
            out.push_str(cuda_type_name(*to));
            out.push_str(")(");
            emit_expr(out, value);
            out.push(')');
        }
        ScalarExpr::MethodCall { receiver, method, args } => {
            // emit `min`/`max`/`abs` as INLINE TERNARIES, not fmin/fmax/fabs. the
            // libdevice functions follow IEEE 754-2008 NaN / signed-zero semantics
            // that differ from the plain `a<b?a:b` ternary; at shock cells the
            // divergent semantics produced different fluxes CPU vs GPU -> macroscopic
            // Bx drift in MUB09. the f64/f32 `Numeric` carrier, the
            // interpreter, and the cranelift jit all use this SAME ternary, so the
            // CPU<->GPU bit-oracle (cpu_gpu_minmax_oracle.rs) holds.
            //   min(a, b) = a < b ? a : b
            //   max(a, b) = a > b ? a : b
            //   abs(x)    = x < 0 ? -x : x
            // no headers needed; works in both CUDA C and CUDA C++.
            match method.as_str() {
                "min" => {
                    out.push('(');
                    emit_expr(out, receiver);
                    out.push_str(" < ");
                    emit_expr(out, &args[0]);
                    out.push_str(" ? ");
                    emit_expr(out, receiver);
                    out.push_str(" : ");
                    emit_expr(out, &args[0]);
                    out.push(')');
                }
                "max" => {
                    out.push('(');
                    emit_expr(out, receiver);
                    out.push_str(" > ");
                    emit_expr(out, &args[0]);
                    out.push_str(" ? ");
                    emit_expr(out, receiver);
                    out.push_str(" : ");
                    emit_expr(out, &args[0]);
                    out.push(')');
                }
                "abs" => {
                    out.push('(');
                    emit_expr(out, receiver);
                    out.push_str(" < 0.0 ? -");
                    emit_expr(out, receiver);
                    out.push_str(" : ");
                    emit_expr(out, receiver);
                    out.push(')');
                }
                "powi" => {
                    // integer power. f64::powi raises a NEGATIVE base exactly
                    // (e.g., (-2)^2 = 4) whereas libdevice pow(neg, 2.0) = NaN,
                    // and the fallthrough arm would emit a bare `powi(...)` that
                    // does not exist in NVRTC. emit the unrolled multiply chain
                    // (exponentiation by squaring) so the grouping is BIT-IDENTICAL
                    // to f64::powi and the Gv carrier lowering (gv.rs::powi).
                    let n = const_i32(&args[0]).unwrap_or_else(|| {
                        panic!(
                            "emit_cuda::powi: exponent must be a compile-time integer \
                             constant, got {:?}",
                            args[0]
                        )
                    });
                    out.push_str(&powi_product(receiver, n));
                }
                "div_euclid" => {
                    // integer floor division (rust div_euclid semantics for the
                    // index-space FloorDiv op): c's `/` truncates toward zero, so
                    // subtract one when the remainder is nonzero and the operand
                    // signs differ (the xor sign test).
                    //   a.div_euclid(b) -> (a / b - ((a % b != 0 && (a ^ b) < 0) ? 1 : 0))
                    out.push('(');
                    emit_expr(out, receiver);
                    out.push_str(" / ");
                    emit_expr(out, &args[0]);
                    out.push_str(" - ((");
                    emit_expr(out, receiver);
                    out.push_str(" % ");
                    emit_expr(out, &args[0]);
                    out.push_str(" != 0 && (");
                    emit_expr(out, receiver);
                    out.push_str(" ^ ");
                    emit_expr(out, &args[0]);
                    out.push_str(") < 0) ? 1 : 0))");
                }
                _ => {
                    // every other method: emit as function call as before.
                    // sqrt, sin, cos, exp, log, etc. — these go through
                    // libdevice and match standard math semantics.
                    let fn_name = cuda_method_to_fn(method);
                    out.push_str(fn_name);
                    out.push('(');
                    emit_expr(out, receiver);
                    for a in args {
                        out.push_str(", ");
                        emit_expr(out, a);
                    }
                    out.push(')');
                }
            }
        }
        ScalarExpr::Select { cond, then, else_ } => {
            // C++ ternary.
            out.push('(');
            emit_expr(out, cond);
            out.push_str(" ? ");
            emit_expr(out, then);
            out.push_str(" : ");
            emit_expr(out, else_);
            out.push(')');
        }
        ScalarExpr::IndexInto { container, index } => {
            out.push_str(container);
            out.push('[');
            emit_expr(out, index);
            out.push(']');
        }
        ScalarExpr::FieldLoadAt { .. } => {
            // emit_kernel.rs MUST rewrite every FieldLoadAt to a Var
            // form before invoking the cuda emitter. seeing one here is
            // a bug in the rewrite pass — fail loudly rather than emit
            // an undefined identifier.
            panic!(
                "emit_cuda::emit_expr: encountered un-rewritten FieldLoadAt — \
                 emit_kernel::rewrite_field_load_at must run first"
            );
        }
        ScalarExpr::FreeCall { name, args } => {
            // F1.B.8: direct function call by name. the function
            // definition is supplied externally (by the scalar
            // elemental's _cuda accessor, embedded into the kernel
            // preamble by the kernel macro). emitter just produces the
            // call site.
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                emit_expr(out, a);
            }
            out.push(')');
        }
    }
}

fn emit_const(out: &mut String, v: &ConstValue) {
    match v {
        // non-finite consts spell the IEEE bit pattern via device intrinsics, NOT the
        // <math.h> macros (INFINITY) / functions (nan): nvcc includes math.h
        // implicitly but NVRTC does NOT, so `INFINITY` is undefined under runtime
        // compilation (docs/design/15 §1). __longlong_as_double / __int_as_float are
        // CUDA+HIP device builtins needing no header, and reinterpret the exact bits
        // — bit-identical to the macros, no numerical change.
        ConstValue::F64(x) => {
            if x.is_nan() {
                out.push_str("__longlong_as_double(0x7ff8000000000000LL)");
            } else if x.is_infinite() {
                out.push_str(if *x > 0.0 {
                    "__longlong_as_double(0x7ff0000000000000LL)"
                } else {
                    "(-__longlong_as_double(0x7ff0000000000000LL))"
                });
            } else {
                out.push_str(&format!("{:?}", x));
            }
        }
        ConstValue::F32(x) => {
            if x.is_nan() {
                out.push_str("__int_as_float(0x7fc00000)");
            } else if x.is_infinite() {
                out.push_str(if *x > 0.0 { "__int_as_float(0x7f800000)" } else { "(-__int_as_float(0x7f800000))" });
            } else {
                out.push_str(&format!("{:?}f", x));
            }
        }
        ConstValue::I32(x) => out.push_str(&format!("{}", x)),
        ConstValue::U32(x) => out.push_str(&format!("{}u", x)),
        ConstValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
    }
}

pub(crate) fn cuda_type_name(e: ElementTy) -> &'static str {
    match e {
        ElementTy::F64  => "double",
        ElementTy::F32  => "float",
        ElementTy::I32  => "int",
        ElementTy::U32  => "unsigned int",
        ElementTy::Bool => "bool",
    }
}

fn cuda_binop(kind: BinaryKind) -> &'static str {
    match kind {
        BinaryKind::Add => "+", BinaryKind::Sub => "-",
        BinaryKind::Mul => "*", BinaryKind::Div => "/",
        BinaryKind::Eq => "==", BinaryKind::Ne => "!=",
        BinaryKind::Lt => "<",  BinaryKind::Le => "<=",
        BinaryKind::Gt => ">",  BinaryKind::Ge => ">=",
        BinaryKind::BitOr  => "|",
        BinaryKind::BitAnd => "&",
        BinaryKind::BitXor => "^",
    }
}

/// translate a Rust method-call name to its CUDA libdevice / math.h
/// equivalent function name.
// extract a compile-time integer exponent from a powi argument. accepts the
// integer const forms and an exactly-integral f64 (the carrier lowers small
// int exponents through f64 in places). returns None for any non-constant or
// non-integral value so the caller can fail loudly.
fn const_i32(e: &ScalarExpr) -> Option<i32> {
    match e {
        ScalarExpr::Const(ConstValue::I32(n)) => Some(*n),
        ScalarExpr::Const(ConstValue::U32(n)) => Some(*n as i32),
        ScalarExpr::Const(ConstValue::F64(x)) if x.fract() == 0.0 => Some(*x as i32),
        ScalarExpr::Const(ConstValue::F32(x)) if x.fract() == 0.0 => Some(*x as i32),
        _ => None,
    }
}

// build the cuda source for `receiver^n` as an unrolled multiply chain using
// exponentiation by squaring. mirrors gv.rs::powi EXACTLY so the float grouping
// (and therefore the rounding) is bit-identical to f64::powi on host and Gv.
fn powi_product(receiver: &ScalarExpr, n: i32) -> String {
    if n == 0 {
        return "1.0".to_string();
    }
    let mut base = String::new();
    emit_expr(&mut base, receiver);
    let mut exp = n.unsigned_abs();
    let mut acc: Option<String> = None;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = Some(match acc {
                None => base.clone(),
                Some(a) => format!("({a} * {base})"),
            });
        }
        exp >>= 1;
        if exp > 0 {
            base = format!("({base} * {base})");
        }
    }
    let pos = acc.expect("n != 0 implies acc is set");
    if n < 0 {
        format!("(1.0 / {pos})")
    } else {
        pos
    }
}

fn cuda_method_to_fn(method: &str) -> &str {
    match method {
        // float-only math functions take the `f` suffix? we don't track
        // precision here per-call; emit the double-prec name. nvcc selects
        // overloads based on argument type at compile time, so this works
        // for both double and float inputs.
        // NOTE: `abs`, `min`, `max` are handled inline (ternary form) in
        // emit_expr's MethodCall arm — they intentionally do NOT use the libdevice
        // `fabs`/`fmin`/`fmax`, whose IEEE NaN / signed-zero semantics diverge
        // from the plain ternary. these fallthrough entries are kept
        // only for a hypothetical caller that bypasses the special-case path; they
        // are not reached during normal emission.
        "abs"    => "fabs",
        "sqrt"   => "sqrt",
        "floor"  => "floor",
        "ceil"   => "ceil",
        "round"  => "round",
        "trunc"  => "trunc",
        "min"    => "fmin",
        "max"    => "fmax",
        "is_finite" => "isfinite",
        "is_nan"    => "isnan",
        "sin" | "cos" | "tan" => method,
        "asin" | "acos" | "atan" | "atan2" => method,
        "exp" | "exp2" => method,
        // Rust .ln() -> C++ log(); .log2() / .log10() are direct.
        "ln"    => "log",
        "log2"  => "log2",
        "log10" => "log10",
        "sinh" | "cosh" | "tanh" => method,
        "asinh" | "acosh" | "atanh" => method,
        "powf"  => "pow",
        "hypot" => "hypot",
        // fallback: pass through and trust nvcc to either find or error.
        other => other,
    }
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConstValue, DimExpr, ElementTy, ElementWiseOp, Graph, Symbol, TensorTy,
        TranscendentalOp, scalarize,
    };

    fn lit(n: usize) -> DimExpr { DimExpr::Literal(n) }

    #[test]
    fn rank_0_emits_scalar_return() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let f = scalarize(&g, x, "ident");
        let src = emit_cuda(&f);
        assert!(src.contains("__device__ inline double ident(double x)"));
        assert!(src.contains("return x;"));
        // no struct decl for rank-0
        assert!(!src.contains("_out"), "{}", src);
    }

    #[test]
    fn rank_n_emits_struct_decl_and_brace_init() {
        let mut g = Graph::new();
        let s = g.add_scalar_param("s", ElementTy::F64);
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]),
            None,
        );
        let r = g.element_wise(ElementWiseOp::Mul, vec![s, v], None);
        let f = scalarize(&g, r, "scale2");
        let src = emit_cuda(&f);
        assert!(src.contains("struct scale2_out { double _0; double _1; };"));
        assert!(src.contains("__device__ inline scale2_out scale2(double s, double v_0, double v_1)"));
        assert!(src.contains("return scale2_out { (s * v_0), (s * v_1) };"));
    }

    #[test]
    fn add_uses_plus_operator() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let s = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let f = scalarize(&g, s, "addd");
        let src = emit_cuda(&f);
        assert!(src.contains("(a + b)"));
    }

    #[test]
    fn neg_uses_unary_minus() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let n = g.element_wise(ElementWiseOp::Neg, vec![x], None);
        let f = scalarize(&g, n, "neg");
        let src = emit_cuda(&f);
        assert!(src.contains("(-x)"));
    }

    #[test]
    fn abs_emits_ternary_matching_my_abs() {
        // emit `(x < 0.0 ? -x : x)` not `fabs(x)` so semantics match
        // the CPU carrier's ternary abs.
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let a = g.element_wise(ElementWiseOp::Abs, vec![x], None);
        let f = scalarize(&g, a, "absx");
        let src = emit_cuda(&f);
        assert!(src.contains("(x < 0.0 ? -x : x)"),
            "expected ternary, got:\n{}", src);
        assert!(!src.contains("fabs"),
            "should not emit fabs(), got:\n{}", src);
    }

    #[test]
    fn floor_div_emits_floor_division_correction() {
        // FloorDiv must render rust div_euclid semantics in C: truncating `/`
        // corrected by one when the remainder is nonzero and signs differ.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::I32);
        let b = g.add_scalar_param("b", ElementTy::I32);
        let q = g.element_wise(ElementWiseOp::FloorDiv, vec![a, b], None);
        let f = scalarize(&g, q, "fdiv");
        let src = emit_cuda(&f);
        assert!(
            src.contains("(a / b - ((a % b != 0 && (a ^ b) < 0) ? 1 : 0))"),
            "expected floor-division correction, got:\n{}", src
        );
        assert!(!src.contains("div_euclid"),
            "rust method name must not leak into cuda source:\n{}", src);
    }

    #[test]
    fn min_emits_ternary_matching_my_min() {
        // tier-1 #2b: emit `(a < b ? a : b)` not `fmin(a, b)`.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let m = g.element_wise(ElementWiseOp::Min, vec![a, b], None);
        let f = scalarize(&g, m, "mn");
        let src = emit_cuda(&f);
        assert!(src.contains("(a < b ? a : b)"),
            "expected ternary, got:\n{}", src);
        assert!(!src.contains("fmin"),
            "should not emit fmin(), got:\n{}", src);
    }

    #[test]
    fn max_emits_ternary_matching_my_max() {
        // tier-1 #2b: emit `(a > b ? a : b)` not `fmax(a, b)`.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let m = g.element_wise(ElementWiseOp::Max, vec![a, b], None);
        let f = scalarize(&g, m, "mx");
        let src = emit_cuda(&f);
        assert!(src.contains("(a > b ? a : b)"),
            "expected ternary, got:\n{}", src);
        assert!(!src.contains("fmax"),
            "should not emit fmax(), got:\n{}", src);
    }

    #[test]
    fn transcendental_sin_uses_sin_function() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let s = g.transcendental(TranscendentalOp::Sin, vec![x], None);
        let f = scalarize(&g, s, "sinx");
        let src = emit_cuda(&f);
        assert!(src.contains("sin(x)"));
    }

    #[test]
    fn transcendental_ln_maps_to_log() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let l = g.transcendental(TranscendentalOp::Log, vec![x], None);
        let f = scalarize(&g, l, "lnx");
        let src = emit_cuda(&f);
        // Rust's ln -> C++ log(double)
        assert!(src.contains("log(x)"), "{}", src);
        assert!(!src.contains("ln("), "should not emit Rust-style ln(), got:\n{}", src);
    }

    #[test]
    fn transcendental_pow_maps_to_pow() {
        let mut g = Graph::new();
        let b = g.add_scalar_param("b", ElementTy::F64);
        let e = g.add_scalar_param("e", ElementTy::F64);
        let r = g.transcendental(TranscendentalOp::Pow, vec![b, e], None);
        let f = scalarize(&g, r, "powbe");
        let src = emit_cuda(&f);
        assert!(src.contains("pow(b, e)"));
        assert!(!src.contains("powf("));
    }

    #[test]
    fn powi_emits_unrolled_product_not_libdevice() {
        // tier-1 #4: a `powi` MethodCall must NOT hit the `other => other`
        // fallthrough (which emits a bare `powi(...)` that NVRTC cannot compile).
        // it lowers to the exponentiation-by-squaring multiply chain, grouped
        // bit-identically to f64::powi / gv.rs::powi.
        let x = ScalarExpr::Var("x".to_string());
        let pow3 = ScalarExpr::MethodCall {
            receiver: Box::new(x.clone()),
            method:   "powi".to_string(),
            args:     vec![ScalarExpr::Const(ConstValue::I32(3))],
        };
        let mut src = String::new();
        emit_expr(&mut src, &pow3);
        assert_eq!(src, "(x * (x * x))", "x.powi(3) grouping");
        assert!(!src.contains("powi("), "rust method must not leak: {src}");
        assert!(!src.contains("pow("), "must not call libdevice pow: {src}");

        // negative exponent -> reciprocal of the positive chain.
        let powm2 = ScalarExpr::MethodCall {
            receiver: Box::new(x.clone()),
            method:   "powi".to_string(),
            args:     vec![ScalarExpr::Const(ConstValue::I32(-2))],
        };
        let mut src = String::new();
        emit_expr(&mut src, &powm2);
        assert_eq!(src, "(1.0 / (x * x))", "x.powi(-2)");

        // zero exponent -> 1.0.
        let pow0 = ScalarExpr::MethodCall {
            receiver: Box::new(x),
            method:   "powi".to_string(),
            args:     vec![ScalarExpr::Const(ConstValue::I32(0))],
        };
        let mut src = String::new();
        emit_expr(&mut src, &pow0);
        assert_eq!(src, "1.0", "x.powi(0)");
    }

    #[test]
    fn select_uses_ternary() {
        let mut g = Graph::new();
        let c = g.add_scalar_param("c", ElementTy::Bool);
        let t = g.add_scalar_param("t", ElementTy::F64);
        let e = g.add_scalar_param("e", ElementTy::F64);
        let r = g.select(c, t, e, None);
        let f = scalarize(&g, r, "sel");
        let src = emit_cuda(&f);
        assert!(src.contains("(c ? t : e)"));
    }

    #[test]
    fn const_int_no_suffix_unsigned_with_u() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::U32(5), None);
        let f = scalarize(&g, a, "five");
        let src = emit_cuda(&f);
        assert!(src.contains("return 5u;"));
        assert!(src.contains("unsigned int"));
    }

    #[test]
    fn const_bool_emits_true_false() {
        let mut g = Graph::new();
        let b = g.add_const(ConstValue::Bool(false), None);
        let f = scalarize(&g, b, "no");
        let src = emit_cuda(&f);
        assert!(src.contains("return false;"));
        assert!(src.contains("bool"));
    }

    #[test]
    fn dot_product_emits_full_cuda_source() {
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
        let f = scalarize(&g, r, "dot3");
        let src = emit_cuda(&f);
        // rank-0 result -> no struct decl
        assert!(!src.contains("_out"));
        assert!(src.contains("__device__ inline double dot3(double a_0, double a_1, double a_2, double b_0, double b_1, double b_2)"));
        // 3 muls + 2 adds in the return expression
        assert!(src.contains("(a_0 * b_0)"));
        assert!(src.contains("(a_2 * b_2)"));
    }

    #[test]
    fn comparison_emits_c_operator() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let lt = g.element_wise(ElementWiseOp::Lt, vec![a, b], None);
        let f = scalarize(&g, lt, "less");
        let src = emit_cuda(&f);
        assert!(src.contains("(a < b)"));
        assert!(src.contains("__device__ inline bool less"));
    }

    // ---- R.5.a: const-generic loop emission (CUDA) ----

    #[test]
    fn generic_dim_dot_emits_template_with_for_loop() {
        let mut g = Graph::new();
        let a = g.add_param(
            crate::Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![DimExpr::generic("D")]),
            None,
        );
        let b = g.add_param(
            crate::Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![DimExpr::generic("D")]),
            None,
        );
        let r = g.einsum("i,i->", vec![a, b], None);
        let f = scalarize(&g, r, "dot_d");
        let src = emit_cuda(&f);
        // template<int D> __device__ inline double dot_d(const double (&a)[D], const double (&b)[D])
        assert!(src.contains("template<int D>"), "missing template prefix: {}", src);
        assert!(src.contains("__device__ inline double dot_d"));
        assert!(src.contains("const double (&a)[D]"), "missing array-ref param: {}", src);
        assert!(src.contains("const double (&b)[D]"), "{}", src);
        // body: double __acc_N = 0.0; #pragma unroll for (int __ii_N = 0; ...) { __acc_N += a[ii] * b[ii]; }
        assert!(src.contains("double __acc_"), "missing acc decl: {}", src);
        assert!(src.contains("#pragma unroll"), "missing pragma: {}", src);
        assert!(src.contains("for (int __ii_"), "missing C-style for: {}", src);
        assert!(src.contains("a[__ii_"));
        assert!(src.contains("b[__ii_"));
        assert!(src.contains("+= ("), "missing compound assign: {}", src);
    }

    #[test]
    fn matmul_2x2_emits_struct_with_four_slots() {
        let mut g = Graph::new();
        let m = g.add_param(
            Symbol::intern("M"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(2)]),
            None,
        );
        let n = g.add_param(
            Symbol::intern("N"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2), lit(2)]),
            None,
        );
        let r = g.einsum("ij,jk->ik", vec![m, n], None);
        let f = scalarize(&g, r, "matmul22");
        let src = emit_cuda(&f);
        assert!(src.contains("struct matmul22_out { double _0; double _1; double _2; double _3; };"));
        // R[0,0] = M_0_0*N_0_0 + M_0_1*N_1_0
        assert!(src.contains("((M_0_0 * N_0_0) + (M_0_1 * N_1_0))"));
    }

    // ----- per-cell source-kernel ABI emit -----

    #[test]
    fn source_kernel_emits_abi_signature_and_per_cell_reads() {
        // graph: out_0 = a + b * sqrt(d) over scalar params a, b, d.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let d = g.add_scalar_param("d", ElementTy::F64);
        let sq = g.element_wise(ElementWiseOp::Sqrt, vec![d], None);
        let bsq = g.element_wise(ElementWiseOp::Mul, vec![b, sq], None);
        let out = g.element_wise(ElementWiseOp::Add, vec![a, bsq], None);

        let params = vec!["a".to_string(), "b".to_string(), "d".to_string()];
        let src = emit_source_kernel(&g, &params, &[out], "test_source");

        // signature: extern "C" __global__ with param/out ptrs + n_cells.
        assert!(src.contains("extern \"C\" __global__ void test_source("), "{src}");
        assert!(src.contains("const double* param_0,"));
        assert!(src.contains("const double* param_2,"));
        assert!(src.contains("double* out_0,"));
        assert!(src.contains("unsigned int n_cells"));
        // thread index + bounds.
        assert!(src.contains("unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;"));
        assert!(src.contains("if (i >= n_cells) return;"));
        // per-cell reads bind the param names the body references.
        assert!(src.contains("auto a = param_0[i];"));
        assert!(src.contains("auto b = param_1[i];"));
        assert!(src.contains("auto d = param_2[i];"));
        // component scope + the actual arithmetic (sqrt via libdevice, not method).
        assert!(src.contains("{ /* component 0 */"));
        assert!(src.contains("sqrt(d)"), "{src}");
        assert!(!src.contains(".sqrt()"), "cuda must not use carrier-generic method form");
        assert!(src.contains("out_0[i] = "));
    }

    #[test]
    fn source_kernel_select_emits_c_ternary_per_component() {
        // out_0 = (a < b) ? a : b — a Select node renders as the C ternary.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let lt = g.element_wise(ElementWiseOp::Lt, vec![a, b], None);
        let out = g.select(lt, a, b, None);

        let params = vec!["a".to_string(), "b".to_string()];
        let src = emit_source_kernel(&g, &params, &[out], "sel_source");

        assert!(src.contains(" ? ") && src.contains(" : "), "missing ternary; {src}");
        assert!(src.contains("(a < b)"), "{src}");
    }

    #[test]
    fn source_kernel_multi_component_isolates_temp_scopes() {
        // two outputs -> two `out_<k>` ptrs + two brace scopes. each scope is
        // independent so per-component temps cannot collide.
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let two = g.add_const(ConstValue::F64(2.0), None);
        let o0 = g.element_wise(ElementWiseOp::Mul, vec![a, two], None);
        let o1 = g.element_wise(ElementWiseOp::Add, vec![a, a], None);

        let params = vec!["a".to_string()];
        let src = emit_source_kernel(&g, &params, &[o0, o1], "two_source");

        assert!(src.contains("double* out_0,"));
        assert!(src.contains("double* out_1,"));
        assert!(src.contains("{ /* component 0 */"));
        assert!(src.contains("{ /* component 1 */"));
        assert!(src.contains("out_0[i] = "));
        assert!(src.contains("out_1[i] = "));
        // braces balanced (kernel body + two component scopes).
        assert_eq!(src.matches('{').count(), src.matches('}').count(), "{src}");
    }

    // ----- docs/design/23 step 1: ScalarStmt::Scope CUDA tests -----

    /// CUDA has no block-expression syntax, so a `Scope` lowers to:
    ///   `<ty> <name>; { <body>; <name> = <result>; }`
    /// the inner braces kill body-local lets; nvcc gets explicit liveness.
    #[test]
    fn scope_emits_cuda_decl_then_braced_assign() {
        use crate::passes::scalarize::{LoweredFn, LoweredParam, ScalarExpr, ScalarStmt, BinaryKind};
        let a_var = || ScalarExpr::Var("a".to_string());
        let b_var = || ScalarExpr::Var("b".to_string());

        let fun = LoweredFn {
            name: "scoped".to_string(),
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
                                Box::new(a_var()),
                                Box::new(b_var()),
                            ),
                        },
                    ],
                    result: ScalarExpr::BinOp(
                        BinaryKind::Mul,
                        Box::new(ScalarExpr::Var("__t1".to_string())),
                        Box::new(a_var()),
                    ),
                },
            ],
            results: vec![ScalarExpr::Var("out".to_string())],
            result_element: ElementTy::F64,
            result_shape: vec![],
        };

        let src = emit_cuda(&fun);
        // declare in outer, then a braced block that fills it. exact form per
        // the lowering rule documented in passes/scalarize.rs::ScalarStmt::Scope.
        assert!(
            src.contains("double out; { double __t1 = (a + b); out = (__t1 * a); }"),
            "expected canonical CUDA decl+braced-assign form; got:\n{src}",
        );
    }

    /// nested scopes produce nested braces in CUDA too. brace count must be
    /// balanced — render output must be at least syntactically valid C++.
    #[test]
    fn scope_nests_correctly_cuda() {
        use crate::passes::scalarize::{LoweredFn, LoweredParam, ScalarExpr, ScalarStmt, BinaryKind};
        let x = || ScalarExpr::Var("x".to_string());

        let fun = LoweredFn {
            name: "nested".to_string(),
            params: vec![LoweredParam::scalar("x".to_string(), ElementTy::F64)],
            body: vec![
                ScalarStmt::Scope {
                    name: "outer".to_string(),
                    element: ElementTy::F64,
                    body: vec![
                        ScalarStmt::Scope {
                            name: "inner".to_string(),
                            element: ElementTy::F64,
                            body: vec![
                                ScalarStmt::Let {
                                    name: "q".to_string(),
                                    element: ElementTy::F64,
                                    value: ScalarExpr::BinOp(
                                        BinaryKind::Mul,
                                        Box::new(x()),
                                        Box::new(x()),
                                    ),
                                },
                            ],
                            result: ScalarExpr::Var("q".to_string()),
                        },
                    ],
                    result: ScalarExpr::BinOp(
                        BinaryKind::Add,
                        Box::new(ScalarExpr::Var("inner".to_string())),
                        Box::new(x()),
                    ),
                },
            ],
            results: vec![ScalarExpr::Var("outer".to_string())],
            result_element: ElementTy::F64,
            result_shape: vec![],
        };

        let src = emit_cuda(&fun);
        // both nested scopes must appear as decl+brace pairs.
        assert!(src.contains("double outer;"), "missing outer decl:\n{src}");
        assert!(src.contains("double inner;"), "missing inner decl:\n{src}");
        // brace balance — count must match.
        let open = src.matches('{').count();
        let close = src.matches('}').count();
        assert_eq!(open, close, "unbalanced braces in:\n{src}");
    }
}
