// =============================================================================
// emit_cpu.rs
//
// CPU backend for the tensor IR: LoweredFn -> Rust source string.
//
// emission shape (V1, elemental-style):
//
//   pub fn <name>(<scalar params>) -> <return_type> {
//       <let statements>
//       <return expression>
//   }
//
// return_type:
//   - rank-0:  the scalar element type (e.g., `f64`)
//   - rank-N:  a tuple of length = product of literal dims
//              (e.g., `(f64, f64, f64)` for shape [3])
//
// the R.5 kernel macro will eventually use a different emission shape
// (writing per-component to Field<T> via SoA indices); this V1 shape
// targets elemental-style usage and the IR conformance tests.
//
// scalar expressions render with explicit parentheses to keep operator
// precedence unambiguous. type suffixes (`1.0_f64`) on every Const
// avoid ambiguity in inference contexts.
// =============================================================================

use crate::graph::ConstValue;
use crate::{ElementTy, LoweredFn, ScalarExpr, ScalarStmt, UnaryKind};

/// emit a LoweredFn as a Rust source string. the output is a complete
/// `pub fn` declaration.
pub fn emit_cpu(f: &LoweredFn) -> String {
    let mut out = String::new();
    emit_signature(&mut out, f);
    out.push_str(" {\n");
    // the elemental (V1) path is always f64 — the hand-written elementals are f64.
    for stmt in &f.body {
        out.push_str("    ");
        emit_stmt(&mut out, stmt, false);
        out.push('\n');
    }
    out.push_str("    ");
    emit_return(&mut out, f, false);
    out.push_str("\n}\n");
    out
}

fn emit_signature(out: &mut String, f: &LoweredFn) {
    // collect const-generic identifiers used by array_len so the fn
    // can carry them in its generic-parameter list. only Symbol-based
    // generics are emitted; literal lengths don't add anything.
    let mut generics: Vec<String> = Vec::new();
    for p in &f.params {
        if let Some(crate::DimExpr::Generic(sym)) = &p.array_len {
            let name = sym.as_str().to_string();
            if !generics.contains(&name) {
                generics.push(name);
            }
        }
    }

    out.push_str("pub fn ");
    out.push_str(&f.name);
    if !generics.is_empty() {
        out.push('<');
        for (i, g) in generics.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str("const ");
            out.push_str(g);
            out.push_str(": usize");
        }
        out.push('>');
    }
    out.push('(');
    for (i, p) in f.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&p.name);
        out.push_str(": ");
        emit_param_type(out, p);
    }
    out.push(')');
    out.push_str(" -> ");
    emit_return_type(out, f);
}

fn emit_param_type(out: &mut String, p: &crate::LoweredParam) {
    match &p.array_len {
        None => out.push_str(rust_type_name(p.element, false)),
        Some(crate::DimExpr::Literal(n)) => {
            out.push('[');
            out.push_str(rust_type_name(p.element, false));
            out.push_str(&format!("; {}]", n));
        }
        Some(crate::DimExpr::Generic(sym)) => {
            out.push('[');
            out.push_str(rust_type_name(p.element, false));
            out.push_str(&format!("; {}]", sym));
        }
    }
}

fn emit_return_type(out: &mut String, f: &LoweredFn) {
    if f.results.len() == 1 && f.result_shape.is_empty() {
        out.push_str(rust_type_name(f.result_element, false));
        return;
    }
    // rank > 0: a tuple of the element type repeated for each scalar.
    out.push('(');
    for i in 0..f.results.len() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(rust_type_name(f.result_element, false));
    }
    out.push(')');
}

pub(crate) fn emit_stmt(out: &mut String, stmt: &ScalarStmt, generic: bool) {
    match stmt {
        ScalarStmt::Let {
            name,
            element,
            value,
        } => {
            out.push_str("let ");
            out.push_str(name);
            out.push_str(": ");
            out.push_str(rust_type_name(*element, generic));
            out.push_str(" = ");
            emit_expr(out, value, generic);
            out.push(';');
        }
        ScalarStmt::LetMut {
            name,
            element,
            init,
        } => {
            out.push_str("let mut ");
            out.push_str(name);
            out.push_str(": ");
            out.push_str(rust_type_name(*element, generic));
            out.push_str(" = ");
            emit_expr(out, init, generic);
            out.push(';');
        }
        ScalarStmt::CompoundAssign { name, op, value } => {
            out.push_str(name);
            out.push(' ');
            out.push_str(op.rust_operator());
            out.push_str("= ");
            emit_expr(out, value, generic);
            out.push(';');
        }
        ScalarStmt::Assign { name, value } => {
            // F2.F: plain assignment (Fold body's accumulator update).
            out.push_str(name);
            out.push_str(" = ");
            emit_expr(out, value, generic);
            out.push(';');
        }
        ScalarStmt::For { iter, bound, body } => {
            out.push_str("for ");
            out.push_str(iter);
            out.push_str(" in 0..");
            match bound {
                crate::DimExpr::Literal(n) => out.push_str(&n.to_string()),
                crate::DimExpr::Generic(sym) => out.push_str(sym.as_str()),
            }
            out.push_str(" { ");
            for s in body {
                emit_stmt(out, s, generic);
                out.push(' ');
            }
            out.push('}');
        }
        ScalarStmt::If { cond, then_body } => {
            out.push_str("if ");
            emit_expr(out, cond, generic);
            out.push_str(" { ");
            for s in then_body {
                emit_stmt(out, s, generic);
                out.push(' ');
            }
            out.push('}');
        }
        ScalarStmt::Break => {
            out.push_str("break;");
        }
        ScalarStmt::Scope {
            name,
            element,
            body,
            result,
        } => {
            // Rust has first-class block expressions: `let name: ty = { body; result };`
            // — the inner lets die at the closing brace, only `result` survives.
            // see docs/design/23_bounded_pressure_ir.md.
            out.push_str("let ");
            out.push_str(name);
            out.push_str(": ");
            out.push_str(rust_type_name(*element, generic));
            out.push_str(" = { ");
            for s in body {
                emit_stmt(out, s, generic);
                out.push(' ');
            }
            emit_expr(out, result, generic);
            out.push_str(" };");
        }
        ScalarStmt::IfElse {
            outs,
            cond,
            then_body,
            else_body,
        } => {
            // declare the N result slots in the OUTER scope; each arm body ends
            // with `outs[j] = <arm result j>`. Rust definite-assignment accepts
            // the deferred init since BOTH arms assign every slot before any
            // use. only the taken arm runs — the carrier-portable early-out `if`
            // (avoids the compute-all-paths cost), the DUAL of the `For`/`Break` iterate.
            for (name, element) in outs {
                out.push_str("let ");
                out.push_str(name);
                out.push_str(": ");
                out.push_str(rust_type_name(*element, generic));
                out.push_str("; ");
            }
            out.push_str("if ");
            emit_expr(out, cond, generic);
            out.push_str(" { ");
            for s in then_body {
                emit_stmt(out, s, generic);
                out.push(' ');
            }
            out.push_str("} else { ");
            for s in else_body {
                emit_stmt(out, s, generic);
                out.push(' ');
            }
            out.push('}');
        }
    }
}

fn emit_return(out: &mut String, f: &LoweredFn, generic: bool) {
    if f.results.len() == 1 && f.result_shape.is_empty() {
        emit_expr(out, &f.results[0], generic);
        return;
    }
    out.push('(');
    for (i, e) in f.results.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        emit_expr(out, e, generic);
    }
    out.push(')');
}

pub(crate) fn emit_expr(out: &mut String, e: &ScalarExpr, generic: bool) {
    match e {
        ScalarExpr::Const(v) => emit_const(out, v, generic),
        ScalarExpr::Var(name) => out.push_str(name),
        ScalarExpr::BinOp(kind, a, b) => {
            out.push('(');
            emit_expr(out, a, generic);
            out.push(' ');
            out.push_str(kind.rust_operator());
            out.push(' ');
            emit_expr(out, b, generic);
            out.push(')');
        }
        ScalarExpr::UnaryOp(UnaryKind::Neg, a) => {
            out.push_str("(-");
            emit_expr(out, a, generic);
            out.push(')');
        }
        ScalarExpr::UnaryOp(UnaryKind::Not, a) => {
            out.push_str("(!");
            emit_expr(out, a, generic);
            out.push(')');
        }
        ScalarExpr::Cast { to: _, value } => {
            // numeric promotion: cast the (narrower) value to the kernel float type.
            // generic kernels go through `S::from_f64`; the f64 elemental path casts
            // directly. `value` is an integer index / narrower float -> `as f64`.
            if generic {
                out.push_str("S::from_f64((");
                emit_expr(out, value, generic);
                out.push_str(") as f64)");
            } else {
                out.push_str("((");
                emit_expr(out, value, generic);
                out.push_str(") as f64)");
            }
        }
        ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } => {
            out.push('(');
            emit_expr(out, receiver, generic);
            out.push_str(").");
            out.push_str(method);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                emit_expr(out, a, generic);
            }
            out.push(')');
        }
        ScalarExpr::Select { cond, then, else_ } => {
            out.push_str("(if ");
            emit_expr(out, cond, generic);
            out.push_str(" { ");
            emit_expr(out, then, generic);
            out.push_str(" } else { ");
            emit_expr(out, else_, generic);
            out.push_str(" })");
        }
        ScalarExpr::IndexInto { container, index } => {
            out.push_str(container);
            out.push('[');
            emit_expr(out, index, generic);
            out.push(']');
        }
        ScalarExpr::FieldLoadAt { .. } => {
            // FieldLoadAt is a chalkboard-kernel-only construct: it only
            // makes sense when there is a buffer-passing dispatch (the
            // kernel pipeline). CPU elemental emission has no buffer
            // dispatch — if an elemental tries to use gather_at the
            // graph builder should have rejected it upstream.
            panic!(
                "emit_cpu::emit_expr: FieldLoadAt is only meaningful in \
                 chalkboard kernel emission; elemental CPU emission cannot \
                 reference a buffer index"
            );
        }
        ScalarExpr::FreeCall { name, args } => {
            // F1.B.8: direct function call by name. Rust emission —
            // the function is resolved by name at the surrounding
            // module scope (the scalar elemental's Rust impl).
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                emit_expr(out, a, generic);
            }
            out.push(')');
        }
    }
}

// float constants: `generic` (kernel) renders the scalar-parametric form
// `S::lit(x)` / `S::nan()` (docs/design/15: kernels are `fn k<S: Scalar>`);
// `!generic` (the f64 elemental path) renders the concrete `{x}_f64` / `f64::NAN`.
// integer/bool constants are scalar-independent.
fn emit_const(out: &mut String, v: &ConstValue, generic: bool) {
    match v {
        ConstValue::F64(x) => emit_float_const(out, *x as f64, generic),
        ConstValue::F32(x) => emit_float_const(out, *x as f64, generic),
        ConstValue::I32(x) => out.push_str(&format!("{}_i32", x)),
        ConstValue::U32(x) => out.push_str(&format!("{}_u32", x)),
        ConstValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
    }
}

// `generic` spells the symbi_algebra::Scalar API: associated consts INFINITY /
// NEG_INFINITY / NAN and the `from_f64` constructor; `!generic` (elemental) stays
// concrete f64.
fn emit_float_const(out: &mut String, x: f64, generic: bool) {
    if x.is_nan() {
        out.push_str(if generic { "S::NAN" } else { "f64::NAN" });
    } else if x.is_infinite() {
        out.push_str(match (generic, x > 0.0) {
            (true, true) => "S::INFINITY",
            (true, false) => "S::NEG_INFINITY",
            (false, true) => "f64::INFINITY",
            (false, false) => "f64::NEG_INFINITY",
        });
    } else if generic {
        out.push_str(&format!("S::from_f64({:?})", x));
    } else {
        out.push_str(&format!("{:?}_f64", x));
    }
}

// float element types render `S` (generic kernel scalar) or `f64` (elemental);
// integer/bool types are scalar-independent.
pub(crate) fn rust_type_name(e: ElementTy, generic: bool) -> &'static str {
    match e {
        ElementTy::F64 | ElementTy::F32 => {
            if generic {
                "S"
            } else {
                "f64"
            }
        }
        ElementTy::I32 => "i32",
        ElementTy::U32 => "u32",
        ElementTy::Bool => "bool",
    }
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConstValue, DimExpr, ElementTy, ElementWiseOp, Graph, Symbol, TensorTy, TranscendentalOp,
        scalarize,
    };

    fn lit(n: usize) -> DimExpr {
        DimExpr::Literal(n)
    }

    /// helper: emit_cpu output must parse as syn::ItemFn.
    fn assert_parses(src: &str) {
        let parsed: Result<syn::ItemFn, _> = syn::parse_str(src);
        assert!(
            parsed.is_ok(),
            "emit_cpu output failed to parse:\n{}\nerror: {}",
            src,
            parsed.err().unwrap()
        );
    }

    #[test]
    fn const_scalar_fn() {
        let mut g = Graph::new();
        let c = g.add_const(ConstValue::F64(3.14), None);
        let f = scalarize(&g, c, "pi");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("pub fn pi"));
        assert!(src.contains("-> f64"));
        assert!(src.contains("3.14_f64"));
    }

    #[test]
    fn identity_param_fn() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let f = scalarize(&g, x, "ident");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("pub fn ident(x: f64) -> f64"));
        assert!(src.contains("    x\n"));
    }

    #[test]
    fn add_two_scalars_fn() {
        let mut g = Graph::new();
        let a = g.add_scalar_param("a", ElementTy::F64);
        let b = g.add_scalar_param("b", ElementTy::F64);
        let s = g.element_wise(ElementWiseOp::Add, vec![a, b], None);
        let f = scalarize(&g, s, "add");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("(a + b)"));
    }

    #[test]
    fn dot_product_rust_source_round_trips() {
        // a 3-vector dot product, end-to-end through the IR + lowering + emit.
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
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains(
            "pub fn dot3(a_0: f64, a_1: f64, a_2: f64, b_0: f64, b_1: f64, b_2: f64) -> f64"
        ));
        // body contains the unrolled sum of products.
        assert!(src.contains("(a_0 * b_0)"));
        assert!(src.contains("(a_2 * b_2)"));
    }

    #[test]
    fn rank_n_output_emits_tuple_return() {
        // vector add: scalar param + vector param -> vector result.
        let mut g = Graph::new();
        let s = g.add_scalar_param("s", ElementTy::F64);
        let v = g.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(2)]),
            None,
        );
        let r = g.element_wise(ElementWiseOp::Mul, vec![s, v], None);
        let f = scalarize(&g, r, "scale2");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(
            src.contains("-> (f64, f64)"),
            "tuple return missing: {}",
            src
        );
        assert!(
            src.contains("((s * v_0), (s * v_1))"),
            "tuple body wrong: {}",
            src
        );
    }

    #[test]
    fn unary_neg_emits_unary_operator() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let n = g.element_wise(ElementWiseOp::Neg, vec![x], None);
        let f = scalarize(&g, n, "neg");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("(-x)"));
    }

    #[test]
    fn method_call_emits_dot_notation() {
        // abs is emitted as a method call `(x).abs()`; for generic-S kernels this
        // resolves to the `Numeric` carrier's ternary `my_abs` (tier-1 #2b), no
        // scoped if-select (which would blow up rustc debuginfo on nested chains).
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let a = g.element_wise(ElementWiseOp::Abs, vec![x], None);
        let f = scalarize(&g, a, "absx");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("(x).abs()"));
    }

    #[test]
    fn transcendental_sin_emits_method() {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let s = g.transcendental(TranscendentalOp::Sin, vec![x], None);
        let f = scalarize(&g, s, "sinx");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("(x).sin()"));
    }

    #[test]
    fn select_emits_if_else_expression() {
        let mut g = Graph::new();
        let c = g.add_scalar_param("c", ElementTy::Bool);
        let t = g.add_scalar_param("t", ElementTy::F64);
        let e = g.add_scalar_param("e", ElementTy::F64);
        let r = g.select(c, t, e, None);
        let f = scalarize(&g, r, "sel");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("(if c { t } else { e })"));
    }

    #[test]
    fn nan_inf_consts_emit_named_constants() {
        let mut g = Graph::new();
        let _ = g.add_const(ConstValue::F64(f64::NAN), None);
        let n = g.add_const(ConstValue::F64(f64::INFINITY), None);
        let f = scalarize(&g, n, "inf_only");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("f64::INFINITY"), "INFINITY missing: {}", src);
    }

    #[test]
    fn bool_const_emits_keyword() {
        let mut g = Graph::new();
        let b = g.add_const(ConstValue::Bool(true), None);
        let f = scalarize(&g, b, "yes");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("    true\n"));
        assert!(src.contains("-> bool"));
    }

    #[test]
    fn integer_consts_emit_with_suffix() {
        let mut g = Graph::new();
        let a = g.add_const(ConstValue::I32(-5), None);
        let f = scalarize(&g, a, "neg5");
        let src = emit_cpu(&f);
        assert_parses(&src);
        assert!(src.contains("-5_i32"));
        assert!(src.contains("-> i32"));
    }

    // ---- R.5.a: const-generic loop emission ----

    #[test]
    fn generic_dim_dot_emits_for_loop_with_accumulator() {
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
        let src = emit_cpu(&f);
        // accept the source; it should be valid Rust.
        assert_parses(&src);
        // signature: pub fn dot_d<const D: usize>(a: [f64; D], b: [f64; D]) -> f64
        assert!(
            src.contains("pub fn dot_d<const D: usize>"),
            "signature missing: {}",
            src
        );
        assert!(src.contains("a: [f64; D]"), "{}", src);
        assert!(src.contains("b: [f64; D]"), "{}", src);
        assert!(src.contains("-> f64"));
        // body: let mut __acc_N: f64 = 0.0; for __ii_N in 0..D { __acc_N += a[ii] * b[ii]; }
        assert!(
            src.contains("let mut __acc_"),
            "missing accumulator: {}",
            src
        );
        assert!(src.contains("for __ii_"), "missing for loop: {}", src);
        assert!(src.contains("a[__ii_"), "missing a[ii] indexing: {}", src);
        assert!(src.contains("b[__ii_"), "missing b[ii] indexing: {}", src);
        assert!(
            src.contains("+= ("),
            "missing compound-assign with product: {}",
            src
        );
    }

    #[test]
    fn matmul_emits_full_unrolled_body() {
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
        let src = emit_cpu(&f);
        assert_parses(&src);
        // 8 scalar params (4 each for M and N), 4-tuple return.
        assert!(src.contains("M_0_0: f64") && src.contains("N_1_1: f64"));
        assert!(src.contains("-> (f64, f64, f64, f64)"));
        // result[0,0] = M_0_0*N_0_0 + M_0_1*N_1_0
        assert!(src.contains("((M_0_0 * N_0_0) + (M_0_1 * N_1_0))"));
    }

    // ----- docs/design/23 step 1: ScalarStmt::Scope tests -----

    /// build a `LoweredFn` by hand whose body contains a `Scope` statement,
    /// emit it, and verify the Rust output is the canonical block-expression
    /// form: `let <name>: <ty> = { <inner lets>; <result> };`.
    #[test]
    fn scope_emits_rust_block_expression() {
        use crate::passes::scalarize::{
            BinaryKind, LoweredFn, LoweredParam, ScalarExpr, ScalarStmt,
        };
        let a_var = || ScalarExpr::Var("a".to_string());
        let b_var = || ScalarExpr::Var("b".to_string());
        let t1_var = || ScalarExpr::Var("__t1".to_string());

        // result = scope { let __t1 = a + b; __t1 * a }
        let fun = LoweredFn {
            name: "scoped".to_string(),
            params: vec![
                LoweredParam::scalar("a".to_string(), ElementTy::F64),
                LoweredParam::scalar("b".to_string(), ElementTy::F64),
            ],
            body: vec![ScalarStmt::Scope {
                name: "out".to_string(),
                element: ElementTy::F64,
                body: vec![ScalarStmt::Let {
                    name: "__t1".to_string(),
                    element: ElementTy::F64,
                    value: ScalarExpr::BinOp(BinaryKind::Add, Box::new(a_var()), Box::new(b_var())),
                }],
                result: ScalarExpr::BinOp(BinaryKind::Mul, Box::new(t1_var()), Box::new(a_var())),
            }],
            results: vec![ScalarExpr::Var("out".to_string())],
            result_element: ElementTy::F64,
            result_shape: vec![],
        };

        let src = emit_cpu(&fun);
        // canonical Rust block-expression shape — the let on the outside, the
        // inner let dies at the closing brace, the result is the block's value.
        assert!(
            src.contains("let out: f64 = { let __t1: f64 = (a + b); (__t1 * a) };"),
            "expected canonical block-expression form; got:\n{src}",
        );
        // and the whole thing still parses as a Rust item — proves the brace
        // structure is balanced and syntactically valid.
        assert_parses(&src);
    }

    /// nested scopes emit nested braces and still parse cleanly. validates
    /// that the renderer is recursive and doesn't accidentally hoist.
    #[test]
    fn scope_nests_correctly() {
        use crate::passes::scalarize::{
            BinaryKind, LoweredFn, LoweredParam, ScalarExpr, ScalarStmt,
        };
        let inner_var = || ScalarExpr::Var("inner".to_string());
        let outer_var = || ScalarExpr::Var("outer".to_string());
        let x = || ScalarExpr::Var("x".to_string());

        // result = scope outer {
        //              let inner = scope { let q = x*x; q + x };
        //              inner * 2
        //          }
        let fun = LoweredFn {
            name: "nested".to_string(),
            params: vec![LoweredParam::scalar("x".to_string(), ElementTy::F64)],
            body: vec![ScalarStmt::Scope {
                name: "outer".to_string(),
                element: ElementTy::F64,
                body: vec![ScalarStmt::Scope {
                    name: "inner".to_string(),
                    element: ElementTy::F64,
                    body: vec![ScalarStmt::Let {
                        name: "q".to_string(),
                        element: ElementTy::F64,
                        value: ScalarExpr::BinOp(BinaryKind::Mul, Box::new(x()), Box::new(x())),
                    }],
                    result: ScalarExpr::BinOp(
                        BinaryKind::Add,
                        Box::new(ScalarExpr::Var("q".to_string())),
                        Box::new(x()),
                    ),
                }],
                result: ScalarExpr::BinOp(
                    BinaryKind::Mul,
                    Box::new(inner_var()),
                    Box::new(ScalarExpr::Const(ConstValue::F64(2.0))),
                ),
            }],
            results: vec![outer_var()],
            result_element: ElementTy::F64,
            result_shape: vec![],
        };

        let src = emit_cpu(&fun);
        assert!(
            src.contains("let inner: f64 = {"),
            "inner scope must render as a block expression:\n{src}"
        );
        assert!(
            src.contains("let outer: f64 = {"),
            "outer scope must render as a block expression:\n{src}"
        );
        // brace balance: any nested-scope kernel must still parse.
        assert_parses(&src);
    }
}
