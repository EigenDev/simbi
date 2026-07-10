// =============================================================================
// lazy_select.rs
//
// automatic lazy scheduling of expensive select arms. a `select(c, t, f)`
// evaluates BOTH arms at every cell; when one arm is expensive (a `powf`
// pressure estimate, a root-heavy fallback) and the condition usually picks
// the other, the discarded work dominates the kernel. this pass rewrites a
// float-valued `let x = select(c, t, f)` whose arm-exclusive cost crosses a
// threshold into the lazy statement form:
//
//   let x: S;
//   if c { <arm-exclusive lets> x = t; } else { <...> x = f; }
//
// sinking: after cse the expensive subexpressions live in SEPARATE top-level
// lets. a let sinks into an arm body iff every transitive use ends in that
// one arm of that one select; anything reaching the condition, an output,
// another statement kind, or a second arm stays rooted. per-cell values are
// identical by select semantics — the taken arm's value is unchanged and the
// arms are pure — so the rewrite is bit-exact on every carrier.
//
// runs in `prepare` (post-cse, pre-render): every kernel backend inherits it.
// on gpu a diverged warp runs both arms — never worse than the eager select.
// kernels gaining an `IfElse` drop out of mask_form (statement control flow),
// which is the intended trade: the threshold mirrors mask_form's arm-cost
// gate, so cheap-arm bodies (the vectorizable class) are never touched.
//
// usage:
//   let converted = lazy_select::apply(&mut prepared.scalarized);
// =============================================================================

use crate::ElementTy;
use crate::passes::scalarize::{BinaryKind, KernelScalarized, ScalarExpr, ScalarStmt};
use std::collections::HashMap;

/// minimum arm-exclusive weight that justifies a real branch. one division or
/// sqrt plus change stays eager (a predicted branch costs more than one
/// discarded root — the `S::cond` doc's own guidance); any transcendental
/// fires; multi-root composite arms fire.
const LAZY_THRESHOLD: u32 = 32;

// op weights: transcendentals / unknown calls dominate; roots and divisions
// are an order cheaper; field loads carry memory latency; plain arithmetic
// is ~free. shared in spirit with mask_form's boolean arm gate.
fn expr_weight(e: &ScalarExpr) -> u32 {
    let own = match e {
        ScalarExpr::MethodCall { method, .. } => match method.as_str() {
            "min" | "max" | "abs" => 1,
            "sqrt" | "safe_sqrt" | "recip" => 8,
            _ => 64,
        },
        ScalarExpr::FreeCall { .. } => 64,
        ScalarExpr::BinOp(BinaryKind::Div, _, _) => 8,
        ScalarExpr::BinOp(..) | ScalarExpr::UnaryOp(..) | ScalarExpr::Select { .. } => 1,
        ScalarExpr::FieldLoadAt { .. } => 8,
        _ => 0,
    };
    own + e.children().iter().map(|c| expr_weight(c)).sum::<u32>()
}

/// where a let's value is ultimately consumed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    /// unconditional: an output, a condition, a non-let statement, or arms of
    /// different selects.
    Root,
    /// exclusively one arm of one candidate select: (candidate index, is_then).
    Arm(usize, bool),
}

fn join(a: Option<Placement>, b: Placement) -> Placement {
    match a {
        None => b,
        Some(x) if x == b => x,
        Some(_) => Placement::Root,
    }
}

fn collect_vars<'a>(e: &'a ScalarExpr, out: &mut Vec<&'a str>) {
    if let ScalarExpr::Var(n) = e {
        out.push(n);
    }
    for c in e.children() {
        collect_vars(c, out);
    }
}

/// rewrite eligible selects into lazy `IfElse` statements. returns how many
/// selects were converted. bodies whose statements are not all plain lets are
/// left untouched (the iterative-kernel class owns its own control flow).
pub fn apply(scalarized: &mut KernelScalarized) -> usize {
    if !scalarized
        .body
        .iter()
        .all(|s| matches!(s, ScalarStmt::Let { .. }))
    {
        return 0;
    }

    // candidate selects: float-valued, the whole rhs of a top-level let.
    struct Candidate {
        stmt_idx: usize,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for (idx, stmt) in scalarized.body.iter().enumerate() {
        if let ScalarStmt::Let {
            element: ElementTy::F64 | ElementTy::F32,
            value: ScalarExpr::Select { .. },
            ..
        } = stmt
        {
            candidates.push(Candidate { stmt_idx: idx });
        }
    }
    if candidates.is_empty() {
        return 0;
    }
    let cand_of_stmt: HashMap<usize, usize> = candidates
        .iter()
        .enumerate()
        .map(|(ci, c)| (c.stmt_idx, ci))
        .collect();

    // placement analysis, bottom-up: a use contributes the USER's placement
    // (root for outputs / conditions; the arm slot for a candidate's arm; the
    // user-let's own placement otherwise). candidate lets themselves stay
    // rooted — no nested sinking in this pass.
    let mut placement: HashMap<String, Placement> = HashMap::new();
    let root_uses = |placement: &mut HashMap<String, Placement>, e: &ScalarExpr| {
        let mut vars = Vec::new();
        collect_vars(e, &mut vars);
        for v in vars {
            let cur = placement.get(v).copied();
            placement.insert(v.to_string(), join(cur, Placement::Root));
        }
    };
    for out in &scalarized.outputs {
        root_uses(&mut placement, out);
    }
    for (idx, stmt) in scalarized.body.iter().enumerate().rev() {
        let ScalarStmt::Let { name, value, .. } = stmt else {
            unreachable!()
        };
        if let Some(&ci) = cand_of_stmt.get(&idx) {
            let ScalarExpr::Select { cond, then, else_ } = value else {
                unreachable!()
            };
            root_uses(&mut placement, cond);
            for (arm, is_then) in [(then, true), (else_, false)] {
                let mut vars = Vec::new();
                collect_vars(arm, &mut vars);
                for v in vars {
                    let cur = placement.get(v).copied();
                    placement.insert(v.to_string(), join(cur, Placement::Arm(ci, is_then)));
                }
            }
            // the candidate's own result flows wherever its users put it, but
            // it is never sunk itself: pin it to root.
            placement.insert(name.clone(), Placement::Root);
        } else {
            // a normal let: its value's vars inherit THIS let's placement.
            let here = placement.get(name).copied().unwrap_or(Placement::Root);
            let mut vars = Vec::new();
            collect_vars(value, &mut vars);
            for v in vars {
                let cur = placement.get(v).copied();
                placement.insert(v.to_string(), join(cur, here));
            }
        }
    }

    // cost + conversion decision per candidate: arm-inline weight plus the
    // weight of every let sunk exclusively into that arm.
    let mut sunk_weight: HashMap<(usize, bool), u32> = HashMap::new();
    for stmt in &scalarized.body {
        let ScalarStmt::Let { name, value, .. } = stmt else {
            unreachable!()
        };
        if let Some(Placement::Arm(ci, is_then)) = placement.get(name).copied() {
            *sunk_weight.entry((ci, is_then)).or_insert(0) += expr_weight(value);
        }
    }
    let mut fire: Vec<bool> = Vec::with_capacity(candidates.len());
    for (ci, c) in candidates.iter().enumerate() {
        let ScalarStmt::Let {
            value: ScalarExpr::Select { then, else_, .. },
            ..
        } = &scalarized.body[c.stmt_idx]
        else {
            unreachable!()
        };
        let t_cost = expr_weight(then) + sunk_weight.get(&(ci, true)).copied().unwrap_or(0);
        let f_cost = expr_weight(else_) + sunk_weight.get(&(ci, false)).copied().unwrap_or(0);
        fire.push(t_cost.max(f_cost) >= LAZY_THRESHOLD);
    }
    if !fire.iter().any(|&f| f) {
        return 0;
    }

    // rewrite: walk the body once; lets placed in a FIRING candidate's arm
    // move into that arm's pending list (original order); a firing candidate
    // becomes an IfElse whose arms are its pending lets plus the result
    // assignment. lets placed in a non-firing candidate's arm stay rooted.
    let mut pending: HashMap<(usize, bool), Vec<ScalarStmt>> = HashMap::new();
    let mut new_body: Vec<ScalarStmt> = Vec::with_capacity(scalarized.body.len());
    let mut converted = 0usize;
    for (idx, stmt) in std::mem::take(&mut scalarized.body).into_iter().enumerate() {
        let ScalarStmt::Let {
            ref name,
            element,
            ref value,
        } = stmt
        else {
            unreachable!()
        };
        if let Some(&ci) = cand_of_stmt.get(&idx) {
            if fire[ci] {
                let ScalarExpr::Select { cond, then, else_ } = value.clone() else {
                    unreachable!()
                };
                let mut then_body = pending.remove(&(ci, true)).unwrap_or_default();
                then_body.push(ScalarStmt::Assign {
                    name: name.clone(),
                    value: *then,
                });
                let mut else_body = pending.remove(&(ci, false)).unwrap_or_default();
                else_body.push(ScalarStmt::Assign {
                    name: name.clone(),
                    value: *else_,
                });
                new_body.push(ScalarStmt::IfElse {
                    outs: vec![(name.clone(), element)],
                    cond: *cond,
                    then_body,
                    else_body,
                });
                converted += 1;
                continue;
            }
            new_body.push(stmt);
            continue;
        }
        match placement.get(name.as_str()).copied() {
            Some(Placement::Arm(ci, is_then)) if fire[ci] => {
                pending.entry((ci, is_then)).or_default().push(stmt);
            }
            _ => new_body.push(stmt),
        }
    }
    debug_assert!(
        pending.is_empty(),
        "lazy_select: sunk lets left unplaced (a candidate consumed lets defined after it)"
    );
    scalarized.body = new_body;
    converted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ConstValue;

    fn f(x: f64) -> ScalarExpr {
        ScalarExpr::Const(ConstValue::F64(x))
    }
    fn var(n: &str) -> ScalarExpr {
        ScalarExpr::Var(n.to_string())
    }
    fn let_f(name: &str, value: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Let {
            name: name.to_string(),
            element: ElementTy::F64,
            value,
        }
    }
    fn powf(recv: ScalarExpr, e: ScalarExpr) -> ScalarExpr {
        ScalarExpr::MethodCall {
            receiver: Box::new(recv),
            method: "powf".to_string(),
            args: vec![e],
        }
    }
    fn sel(cond: ScalarExpr, t: ScalarExpr, e: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Select {
            cond: Box::new(cond),
            then: Box::new(t),
            else_: Box::new(e),
        }
    }
    fn gt0(v: &str) -> ScalarExpr {
        ScalarExpr::BinOp(BinaryKind::Gt, Box::new(var(v)), Box::new(f(0.0)))
    }
    fn kernel(body: Vec<ScalarStmt>, outputs: Vec<ScalarExpr>) -> KernelScalarized {
        KernelScalarized {
            params: Vec::new(),
            body,
            outputs,
        }
    }

    #[test]
    fn expensive_exclusive_arm_converts_and_sinks() {
        // heavy = x.powf(0.3) feeds ONLY the then-arm: it must sink into the
        // if-body and the select becomes an IfElse.
        let mut k = kernel(
            vec![
                let_f("heavy", powf(var("x"), f(0.3))),
                let_f("y", sel(gt0("x"), var("heavy"), f(0.0))),
            ],
            vec![var("y")],
        );
        assert_eq!(apply(&mut k), 1);
        assert_eq!(k.body.len(), 1);
        match &k.body[0] {
            ScalarStmt::IfElse {
                outs,
                then_body,
                else_body,
                ..
            } => {
                assert_eq!(outs, &vec![("y".to_string(), ElementTy::F64)]);
                assert_eq!(then_body.len(), 2, "sunk let + assign: {then_body:?}");
                assert!(matches!(&then_body[0], ScalarStmt::Let { name, .. } if name == "heavy"));
                assert!(matches!(&then_body[1], ScalarStmt::Assign { name, .. } if name == "y"));
                assert_eq!(else_body.len(), 1);
            }
            other => panic!("expected IfElse, got {other:?}"),
        }
    }

    #[test]
    fn cheap_arms_stay_eager() {
        let mut k = kernel(
            vec![let_f(
                "y",
                sel(
                    gt0("x"),
                    ScalarExpr::BinOp(BinaryKind::Mul, Box::new(var("x")), Box::new(f(2.0))),
                    f(0.0),
                ),
            )],
            vec![var("y")],
        );
        assert_eq!(apply(&mut k), 0);
        assert!(matches!(
            &k.body[0],
            ScalarStmt::Let { value: ScalarExpr::Select { .. }, .. }
        ));
    }

    #[test]
    fn let_shared_by_both_arms_stays_rooted() {
        // shared feeds BOTH arms: it must not sink; with only Var arms left
        // the exclusive cost is zero and the select stays eager.
        let mut k = kernel(
            vec![
                let_f("shared", powf(var("x"), f(0.3))),
                let_f(
                    "y",
                    sel(
                        gt0("x"),
                        var("shared"),
                        ScalarExpr::BinOp(
                            BinaryKind::Mul,
                            Box::new(var("shared")),
                            Box::new(f(2.0)),
                        ),
                    ),
                ),
            ],
            vec![var("y")],
        );
        assert_eq!(apply(&mut k), 0);
        assert_eq!(k.body.len(), 2);
    }

    #[test]
    fn let_used_by_output_stays_rooted() {
        // heavy feeds the then-arm AND an output: it cannot sink, and the
        // remaining arm-inline cost (a var) is below threshold.
        let mut k = kernel(
            vec![
                let_f("heavy", powf(var("x"), f(0.3))),
                let_f("y", sel(gt0("x"), var("heavy"), f(0.0))),
            ],
            vec![var("y"), var("heavy")],
        );
        assert_eq!(apply(&mut k), 0);
        assert_eq!(k.body.len(), 2);
    }

    #[test]
    fn let_used_by_condition_stays_rooted() {
        // the condition needs the value unconditionally.
        let mut k = kernel(
            vec![
                let_f("heavy", powf(var("x"), f(0.3))),
                let_f("y", sel(gt0("heavy"), var("heavy"), f(0.0))),
            ],
            vec![var("y")],
        );
        assert_eq!(apply(&mut k), 0);
    }

    #[test]
    fn transitive_chain_sinks_whole_dependency_tail() {
        // a -> b -> then-arm: both sink, in definition order.
        let mut k = kernel(
            vec![
                let_f("a", powf(var("x"), f(0.3))),
                let_f(
                    "b",
                    ScalarExpr::BinOp(BinaryKind::Mul, Box::new(var("a")), Box::new(f(2.0))),
                ),
                let_f("y", sel(gt0("x"), var("b"), f(0.0))),
            ],
            vec![var("y")],
        );
        assert_eq!(apply(&mut k), 1);
        match &k.body[0] {
            ScalarStmt::IfElse { then_body, .. } => {
                assert!(matches!(&then_body[0], ScalarStmt::Let { name, .. } if name == "a"));
                assert!(matches!(&then_body[1], ScalarStmt::Let { name, .. } if name == "b"));
                assert!(matches!(&then_body[2], ScalarStmt::Assign { .. }));
            }
            other => panic!("expected IfElse, got {other:?}"),
        }
    }

    #[test]
    fn arms_of_two_different_selects_stay_rooted() {
        // heavy feeds arms of TWO selects: exclusive to neither.
        let mut k = kernel(
            vec![
                let_f("heavy", powf(var("x"), f(0.3))),
                let_f("y", sel(gt0("x"), var("heavy"), f(0.0))),
                let_f("z", sel(gt0("w"), var("heavy"), f(1.0))),
            ],
            vec![var("y"), var("z")],
        );
        assert_eq!(apply(&mut k), 0);
        assert_eq!(k.body.len(), 3);
    }

    #[test]
    fn converted_kernel_evaluates_identically() {
        // the strongest gate: interpret original vs converted on inputs that
        // exercise BOTH branch directions; values must be bit-equal.
        use crate::backends::interp::{Backend, Cpu};
        use crate::passes::scalarize::{LoweredFn, LoweredParam};
        let build = || {
            kernel(
                vec![
                    let_f("heavy", powf(var("x"), f(0.3))),
                    let_f(
                        "y",
                        sel(
                            gt0("x"),
                            ScalarExpr::BinOp(
                                BinaryKind::Add,
                                Box::new(var("heavy")),
                                Box::new(f(1.0)),
                            ),
                            f(7.0),
                        ),
                    ),
                ],
                vec![var("y")],
            )
        };
        let to_fn = |k: KernelScalarized| LoweredFn {
            name: "probe".to_string(),
            params: vec![LoweredParam {
                name: "x".to_string(),
                element: ElementTy::F64,
                array_len: None,
            }],
            body: k.body,
            results: k.outputs,
            result_element: ElementTy::F64,
            result_shape: Vec::new(),
        };
        let eager = to_fn(build());
        let mut lazy_k = build();
        assert_eq!(apply(&mut lazy_k), 1);
        let lazy = to_fn(lazy_k);
        for x in [2.5_f64, -3.0, 0.0] {
            let a = Cpu.eval_elemental(&eager, &[x])[0];
            let b = Cpu.eval_elemental(&lazy, &[x])[0];
            assert!(
                a == b || (a.is_nan() && b.is_nan()),
                "value drift at x={x}: eager {a} vs lazy {b}"
            );
        }
    }
}
