// =============================================================================
// mask_form.rs
//
// branch-free spelling of a scalarized kernel body: float comparisons become
// `Scalar::cmp_*` method calls (returning `S::Mask`) and float-valued
// `Select` becomes the `S::select(mask, then, else)` free call, so the emitted
// Rust body is straight-line and LLVM's slp vectorizer can fuse it.
//
// default on, cost-gated: `select` computes both arms of every conditional,
// so the spelling pays only where arms are cheap. the arm gate below rejects
// transcendental / multi-division arms (the hllc star fan measured 2.6x
// slower mask-formed) and tolerates a single guarded division (the van-leer
// limiter idiom measured faster eager than branched). the win depends on
// long unit-stride trips — the row-elongated cover blocks are this
// spelling's other half.
//
// bit-identity: `select` evaluates both arms and returns the taken arm's
// value unchanged, so per-cell results are bit-for-bit identical to the
// if/else spelling. the trace carrier (Gv) already evaluates both arms of
// every select, so any kernel valid under tracing is valid here.
//
// a kernel is eligible only if it has no statement-level control flow
// (`For`/`Break`/`If`/`IfElse`/assignments — the iterative c2p class) and
// every bool-typed value is provably a mask: bool locals are retyped to
// `S::Mask` wholesale, so any native bool left in mask position (an integer
// comparison, a bool literal, an integer-valued select) bails the kernel.
// ineligible kernels are left untouched and emit the bool/if form.
//
// index space is never entered: `FieldLoadAt` components and `Cast` operands
// stay verbatim.
//
// usage:
//   let changed = mask_form::apply(&mut prepared.scalarized);
// =============================================================================

use crate::ElementTy;
use crate::graph::ConstValue;
use crate::passes::scalarize::{BinaryKind, KernelScalarized, ScalarExpr, ScalarStmt, UnaryKind};
use std::collections::HashMap;

/// rewrite `scalarized` into the mask/select spelling if eligible. returns
/// whether the rewrite was applied; ineligible bodies are left untouched.
pub fn apply(scalarized: &mut KernelScalarized) -> bool {
    // a bool scalar param is a host bool; it cannot join mask arithmetic.
    if scalarized
        .params
        .iter()
        .any(|p| p.element == ElementTy::Bool)
    {
        return false;
    }
    let mut env: HashMap<String, ElementTy> = scalarized
        .params
        .iter()
        .map(|p| (p.name.clone(), p.element))
        .collect();
    let mut check_env = env.clone();
    if !body_eligible(&scalarized.body, &mut check_env)
        || !scalarized
            .outputs
            .iter()
            .all(|e| expr_eligible(e, &check_env))
    {
        return false;
    }
    for stmt in &mut scalarized.body {
        rewrite_stmt(stmt, &mut env);
    }
    for out in &mut scalarized.outputs {
        rewrite_expr(out, &env);
    }
    true
}

fn is_float(t: Option<ElementTy>) -> bool {
    matches!(t, Some(ElementTy::F64) | Some(ElementTy::F32))
}

fn cmp_method(kind: BinaryKind) -> Option<&'static str> {
    match kind {
        BinaryKind::Lt => Some("cmp_lt"),
        BinaryKind::Le => Some("cmp_le"),
        BinaryKind::Gt => Some("cmp_gt"),
        BinaryKind::Ge => Some("cmp_ge"),
        BinaryKind::Eq => Some("cmp_eq"),
        // Ne rewrites to !cmp_eq — Mask has no cmp_ne.
        BinaryKind::Ne => None,
        _ => None,
    }
}

fn is_comparison(kind: BinaryKind) -> bool {
    matches!(
        kind,
        BinaryKind::Lt
            | BinaryKind::Le
            | BinaryKind::Gt
            | BinaryKind::Ge
            | BinaryKind::Eq
            | BinaryKind::Ne
    )
}

// methods on a float receiver that return a boolean.
fn method_returns_bool(name: &str) -> bool {
    matches!(name, "is_finite" | "is_nan" | "is_infinite")
}

/// bottom-up element type of an expression; `None` = unknown (poisons any
/// comparison / select decision that depends on it).
fn infer(e: &ScalarExpr, env: &HashMap<String, ElementTy>) -> Option<ElementTy> {
    match e {
        ScalarExpr::Const(v) => Some(match v {
            ConstValue::F64(_) => ElementTy::F64,
            ConstValue::F32(_) => ElementTy::F32,
            ConstValue::I32(_) => ElementTy::I32,
            ConstValue::U32(_) => ElementTy::U32,
            ConstValue::Bool(_) => ElementTy::Bool,
        }),
        ScalarExpr::Var(name) => env.get(name).copied(),
        ScalarExpr::BinOp(kind, a, b) => {
            if is_comparison(*kind) {
                Some(ElementTy::Bool)
            } else if matches!(
                kind,
                BinaryKind::BitAnd | BinaryKind::BitOr | BinaryKind::BitXor
            ) {
                infer(a, env)
            } else {
                // arithmetic: unify — a float side wins (numeric promotion).
                match (infer(a, env), infer(b, env)) {
                    (x, y) if is_float(x) => x.or(y),
                    (x, y) if is_float(y) => y.or(x),
                    (Some(x), _) => Some(x),
                    (None, y) => y,
                }
            }
        }
        ScalarExpr::UnaryOp(UnaryKind::Not, _) => Some(ElementTy::Bool),
        ScalarExpr::UnaryOp(UnaryKind::Neg, a) => infer(a, env),
        ScalarExpr::MethodCall {
            receiver, method, ..
        } => {
            if method_returns_bool(method) {
                Some(ElementTy::Bool)
            } else {
                infer(receiver, env)
            }
        }
        ScalarExpr::Select { then, else_, .. } => infer(then, env).or_else(|| infer(else_, env)),
        ScalarExpr::IndexInto { container, .. } => env.get(container).copied(),
        // a field load reads the kernel float type.
        ScalarExpr::FieldLoadAt { .. } => Some(ElementTy::F64),
        // scalar elementals return the kernel float type.
        ScalarExpr::FreeCall { .. } => Some(ElementTy::F64),
        ScalarExpr::Cast { to, .. } => Some(*to),
    }
}

// ---- pass 1: eligibility (read-only) ----------------------------------------

fn body_eligible(body: &[ScalarStmt], env: &mut HashMap<String, ElementTy>) -> bool {
    body.iter().all(|stmt| match stmt {
        ScalarStmt::Let {
            name,
            element,
            value,
        } => {
            let ok = expr_eligible(value, env);
            env.insert(name.clone(), *element);
            ok
        }
        ScalarStmt::Scope {
            name,
            element,
            body,
            result,
        } => {
            let mut inner = env.clone();
            let ok = body_eligible(body, &mut inner) && expr_eligible(result, &inner);
            env.insert(name.clone(), *element);
            ok
        }
        // statement-level control flow / mutation is the iterative-kernel
        // class (c2p root-finding, folds) — not maskable.
        ScalarStmt::LetMut { .. }
        | ScalarStmt::CompoundAssign { .. }
        | ScalarStmt::Assign { .. }
        | ScalarStmt::For { .. }
        | ScalarStmt::If { .. }
        | ScalarStmt::IfElse { .. }
        | ScalarStmt::Break => false,
    })
}

fn expr_eligible(e: &ScalarExpr, env: &HashMap<String, ElementTy>) -> bool {
    // after the rewrite every bool-typed value in the body must be an
    // `S::Mask` — a float `cmp_*` result or Not/And/Or combinations of them —
    // because bool locals are retyped to the mask type wholesale. anything
    // that would leave a native bool in mask position bails the kernel:
    // integer comparisons, bool literals, bool-xor (Mask has no BitXor),
    // integer-valued selects (their `if` needs a native bool cond).
    match e {
        // index space stays verbatim: components / cast operands are not entered.
        ScalarExpr::FieldLoadAt { .. } | ScalarExpr::Cast { .. } => return true,
        ScalarExpr::Const(ConstValue::Bool(_)) => return false,
        ScalarExpr::BinOp(kind, a, b) if is_comparison(*kind) => {
            if !(is_float(infer(a, env)) || is_float(infer(b, env))) {
                return false;
            }
        }
        ScalarExpr::BinOp(BinaryKind::BitXor, a, b) => {
            if matches!(infer(a, env), Some(ElementTy::Bool))
                || matches!(infer(b, env), Some(ElementTy::Bool))
            {
                return false;
            }
        }
        ScalarExpr::Select { then, else_, .. } => {
            if !(is_float(infer(then, env)) || is_float(infer(else_, env))) {
                return false;
            }
            // `select` evaluates both arms every cell, where the `if` spelling
            // evaluates one behind a (usually well-predicted) branch. an arm
            // whose inline cost is heavy — a division, an expensive method, a
            // free call — makes the trade a measured loss (the hllc star-state
            // fan runs 2.6x slower mask-formed). cse-hoisted lets referenced by
            // the arm are shared straight-line work outside the arm cost, so the walk
            // sees only what the arm computes inline.
            if arm_is_expensive(then) || arm_is_expensive(else_) {
                return false;
            }
        }
        _ => {}
    }
    e.children().iter().all(|c| expr_eligible(c, env))
}

// inline arm cost gate for the select rewrite: transcendental / power methods
// and free calls dominate the cost of computing a discarded arm, and so do
// multiple divisions. a single division is tolerated: the guarded-denominator
// limiter idiom `select(pos, 2ab/select(pos, a+b, 1), 0)` carries exactly one
// safe division per arm, and computing it eagerly measured faster than the
// branchy spelling (the mask-form flux body with van-leer arms ran 16.4
// vs 18.5 ns/zone bool/if) — the branch costs more than one discarded fdiv.
fn arm_is_expensive(e: &ScalarExpr) -> bool {
    fn scan(e: &ScalarExpr, divs: &mut u32) -> bool {
        let heavy = match e {
            ScalarExpr::BinOp(BinaryKind::Div, _, _) => {
                *divs += 1;
                *divs >= 2
            }
            ScalarExpr::MethodCall { method, .. } => {
                !matches!(method.as_str(), "min" | "max" | "abs")
            }
            ScalarExpr::FreeCall { .. } => true,
            _ => false,
        };
        heavy || e.children().iter().any(|c| scan(c, divs))
    }
    scan(e, &mut 0)
}

// ---- pass 2: rewrite (in place) ----------------------------------------------

fn rewrite_stmt(stmt: &mut ScalarStmt, env: &mut HashMap<String, ElementTy>) {
    match stmt {
        ScalarStmt::Let {
            name,
            element,
            value,
        } => {
            rewrite_expr(value, env);
            env.insert(name.clone(), *element);
        }
        ScalarStmt::Scope {
            name,
            element,
            body,
            result,
        } => {
            let mut inner = env.clone();
            for s in body.iter_mut() {
                rewrite_stmt(s, &mut inner);
            }
            rewrite_expr(result, &inner);
            env.insert(name.clone(), *element);
        }
        _ => unreachable!("mask_form: ineligible statement survived the eligibility pass"),
    }
}

fn rewrite_expr(e: &mut ScalarExpr, env: &HashMap<String, ElementTy>) {
    // index space stays verbatim.
    if matches!(e, ScalarExpr::FieldLoadAt { .. } | ScalarExpr::Cast { .. }) {
        return;
    }
    for c in e.children_mut() {
        rewrite_expr(c, env);
    }
    let replacement = match e {
        ScalarExpr::BinOp(kind, a, b) if is_comparison(*kind) => {
            let float = is_float(infer(a, env)) || is_float(infer(b, env));
            if !float {
                return;
            }
            match cmp_method(*kind) {
                Some(m) => Some(ScalarExpr::MethodCall {
                    receiver: a.clone(),
                    method: m.to_string(),
                    args: vec![(**b).clone()],
                }),
                // Ne: !(a.cmp_eq(b)) — Mask implements Not.
                None => Some(ScalarExpr::UnaryOp(
                    UnaryKind::Not,
                    Box::new(ScalarExpr::MethodCall {
                        receiver: a.clone(),
                        method: "cmp_eq".to_string(),
                        args: vec![(**b).clone()],
                    }),
                )),
            }
        }
        ScalarExpr::Select { cond, then, else_ } => {
            let float = is_float(infer(then, env)) || is_float(infer(else_, env));
            if !float {
                return;
            }
            Some(ScalarExpr::FreeCall {
                name: "S::select".to_string(),
                args: vec![(**cond).clone(), (**then).clone(), (**else_).clone()],
            })
        }
        _ => None,
    };
    if let Some(r) = replacement {
        *e = r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::scalarize::LoweredParam;

    fn f(x: f64) -> ScalarExpr {
        ScalarExpr::Const(ConstValue::F64(x))
    }
    fn var(n: &str) -> ScalarExpr {
        ScalarExpr::Var(n.to_string())
    }
    fn let_s(name: &str, value: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Let {
            name: name.to_string(),
            element: ElementTy::F64,
            value,
        }
    }
    fn param(name: &str, element: ElementTy) -> LoweredParam {
        LoweredParam {
            name: name.to_string(),
            element,
            array_len: None,
        }
    }
    fn kernel(
        params: Vec<LoweredParam>,
        body: Vec<ScalarStmt>,
        outputs: Vec<ScalarExpr>,
    ) -> KernelScalarized {
        KernelScalarized {
            params,
            body,
            outputs,
        }
    }

    #[test]
    fn float_comparison_becomes_cmp_method() {
        let mut k = kernel(
            vec![param("x", ElementTy::F64)],
            vec![ScalarStmt::Let {
                name: "m".into(),
                element: ElementTy::Bool,
                value: ScalarExpr::BinOp(BinaryKind::Gt, Box::new(var("x")), Box::new(f(0.0))),
            }],
            vec![var("x")],
        );
        assert!(apply(&mut k));
        match &k.body[0] {
            ScalarStmt::Let { value, .. } => match value {
                ScalarExpr::MethodCall { method, .. } => assert_eq!(method, "cmp_gt"),
                other => panic!("expected cmp_gt MethodCall, got {other:?}"),
            },
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn float_select_becomes_select_free_call() {
        let sel = ScalarExpr::Select {
            cond: Box::new(ScalarExpr::BinOp(
                BinaryKind::Lt,
                Box::new(var("x")),
                Box::new(f(0.0)),
            )),
            then: Box::new(f(1.0)),
            else_: Box::new(f(2.0)),
        };
        let mut k = kernel(
            vec![param("x", ElementTy::F64)],
            vec![let_s("y", sel)],
            vec![var("y")],
        );
        assert!(apply(&mut k));
        match &k.body[0] {
            ScalarStmt::Let { value, .. } => match value {
                ScalarExpr::FreeCall { name, args } => {
                    assert_eq!(name, "S::select");
                    assert_eq!(args.len(), 3);
                    // the cond was rewritten to a mask-producing cmp
                    assert!(
                        matches!(&args[0], ScalarExpr::MethodCall { method, .. } if method == "cmp_lt")
                    );
                }
                other => panic!("expected S::select FreeCall, got {other:?}"),
            },
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn integer_select_in_body_bails_whole_kernel() {
        // a body-space integer pick needs a native bool `if`, which cannot
        // coexist with mask-typed bool locals — the kernel keeps bool/if form.
        let sel = ScalarExpr::Select {
            cond: Box::new(ScalarExpr::BinOp(
                BinaryKind::Eq,
                Box::new(var("k")),
                Box::new(ScalarExpr::Const(ConstValue::I32(1))),
            )),
            then: Box::new(ScalarExpr::Const(ConstValue::I32(2))),
            else_: Box::new(ScalarExpr::Const(ConstValue::I32(3))),
        };
        let mut k = kernel(
            vec![param("k", ElementTy::I32)],
            vec![ScalarStmt::Let {
                name: "j".into(),
                element: ElementTy::I32,
                value: sel.clone(),
            }],
            vec![var("j")],
        );
        assert!(!apply(&mut k));
        assert!(matches!(
            &k.body[0],
            ScalarStmt::Let { value, .. } if *value == sel
        ));
    }

    #[test]
    fn ne_rewrites_to_not_cmp_eq() {
        let mut k = kernel(
            vec![param("x", ElementTy::F64)],
            vec![ScalarStmt::Let {
                name: "m".into(),
                element: ElementTy::Bool,
                value: ScalarExpr::BinOp(BinaryKind::Ne, Box::new(var("x")), Box::new(f(0.0))),
            }],
            vec![var("x")],
        );
        assert!(apply(&mut k));
        match &k.body[0] {
            ScalarStmt::Let { value, .. } => {
                assert!(matches!(
                    value,
                    ScalarExpr::UnaryOp(UnaryKind::Not, inner)
                        if matches!(&**inner, ScalarExpr::MethodCall { method, .. } if method == "cmp_eq")
                ));
            }
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn multiple_divisions_in_select_arm_bail_whole_kernel() {
        // both select arms run every cell under mask form; a multi-division
        // arm is the hllc-class regression (one guarded division is tolerated
        // — the limiter idiom measured faster eager). the kernel keeps
        // bool/if form.
        let div = |a: ScalarExpr, b: ScalarExpr| {
            ScalarExpr::BinOp(BinaryKind::Div, Box::new(a), Box::new(b))
        };
        let sel = ScalarExpr::Select {
            cond: Box::new(ScalarExpr::BinOp(
                BinaryKind::Gt,
                Box::new(var("x")),
                Box::new(f(0.0)),
            )),
            then: Box::new(div(div(f(1.0), var("x")), var("x"))),
            else_: Box::new(f(0.0)),
        };
        let mut k = kernel(
            vec![param("x", ElementTy::F64)],
            vec![let_s("y", sel.clone())],
            vec![var("y")],
        );
        assert!(!apply(&mut k));
        assert!(matches!(
            &k.body[0],
            ScalarStmt::Let { value, .. } if *value == sel
        ));
    }

    #[test]
    fn single_division_arm_stays_eligible() {
        // the guarded-denominator limiter: one division per arm masks.
        let sel = ScalarExpr::Select {
            cond: Box::new(ScalarExpr::BinOp(
                BinaryKind::Gt,
                Box::new(var("x")),
                Box::new(f(0.0)),
            )),
            then: Box::new(ScalarExpr::BinOp(
                BinaryKind::Div,
                Box::new(f(1.0)),
                Box::new(var("x")),
            )),
            else_: Box::new(f(0.0)),
        };
        let mut k = kernel(
            vec![param("x", ElementTy::F64)],
            vec![let_s("y", sel)],
            vec![var("y")],
        );
        assert!(apply(&mut k));
    }

    #[test]
    fn division_outside_select_arms_stays_eligible() {
        // a cse-hoisted division feeding cheap select arms by name is shared
        // straight-line work outside the arm cost — the kernel is rewritten.
        let div = ScalarExpr::BinOp(BinaryKind::Div, Box::new(f(1.0)), Box::new(var("x")));
        let sel = ScalarExpr::Select {
            cond: Box::new(ScalarExpr::BinOp(
                BinaryKind::Gt,
                Box::new(var("x")),
                Box::new(f(0.0)),
            )),
            then: Box::new(var("iv")),
            else_: Box::new(f(0.0)),
        };
        let mut k = kernel(
            vec![param("x", ElementTy::F64)],
            vec![let_s("iv", div), let_s("y", sel)],
            vec![var("y")],
        );
        assert!(apply(&mut k));
        assert!(matches!(
            &k.body[1],
            ScalarStmt::Let { value: ScalarExpr::FreeCall { name, .. }, .. } if name == "S::select"
        ));
    }

    #[test]
    fn control_flow_kernel_is_left_untouched() {
        let orig_body = vec![ScalarStmt::For {
            iter: "_it".into(),
            bound: crate::DimExpr::Literal(8),
            body: vec![ScalarStmt::Break],
        }];
        let mut k = kernel(vec![], orig_body.clone(), vec![f(0.0)]);
        assert!(!apply(&mut k));
        assert_eq!(k.body, orig_body);
    }

    #[test]
    fn unknown_typed_comparison_bails_whole_kernel() {
        // comparison on a var with no declared type: not provable, no rewrite.
        let cmp = ScalarExpr::BinOp(
            BinaryKind::Gt,
            Box::new(var("mystery")),
            Box::new(var("mystery2")),
        );
        let mut k = kernel(
            vec![],
            vec![ScalarStmt::Let {
                name: "m".into(),
                element: ElementTy::Bool,
                value: cmp.clone(),
            }],
            vec![f(0.0)],
        );
        assert!(!apply(&mut k));
        assert!(matches!(
            &k.body[0],
            ScalarStmt::Let { value, .. } if *value == cmp
        ));
    }

    #[test]
    fn field_load_components_stay_verbatim() {
        // an integer select inside a load's coord component must not be entered.
        let load = ScalarExpr::FieldLoadAt {
            field_key: "prim_rho".into(),
            components: vec![ScalarExpr::Select {
                cond: Box::new(ScalarExpr::BinOp(
                    BinaryKind::Eq,
                    Box::new(var("map_type")),
                    Box::new(ScalarExpr::Const(ConstValue::I32(1))),
                )),
                then: Box::new(var("_coord_0")),
                else_: Box::new(ScalarExpr::Const(ConstValue::I32(0))),
            }],
        };
        let mut k = kernel(
            vec![param("map_type", ElementTy::I32)],
            vec![let_s("x", load.clone())],
            vec![var("x")],
        );
        assert!(apply(&mut k));
        assert!(matches!(
            &k.body[0],
            ScalarStmt::Let { value, .. } if *value == load
        ));
    }
}
