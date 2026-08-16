// =============================================================================
// unswitch.rs
//
// loop unswitching on a scalarized kernel body: a select whose condition
// depends only on scalar params (e.g. the reconstruction limiter pick
// `theta < 0`) is loop-invariant — every cell takes the same arm. `find`
// locates the most-used such condition; `specialize` partially evaluates the
// body at cond = true / false, deleting the untaken arm of every select
// gated on it. the emitter renders both specializations and a dispatcher
// that branches once per kernel call, so each specialized loop nest is free
// of the invariant conditional (and each can be mask-formed independently —
// an arm-cost-heavy branch keeps bool/if form while the cheap branch
// vectorizes).
//
// specialization is bit-identical by construction: replacing
// `select(c, t, f)` with `t` under `c == true` is the definition of select.
//
// usage:
//   if let Some(cand) = unswitch::find(&prepared.scalarized) {
//       let mut spec = prepared.scalarized.clone();
//       unswitch::specialize(&mut spec, &cand.cond_let, true);
//   }
// =============================================================================

use crate::ElementTy;
use crate::passes::scalarize::{KernelScalarized, ScalarExpr, ScalarStmt};
use std::collections::{HashMap, HashSet};

/// a select-count threshold below which duplicating the loop nest is not
/// worth the code size: one or two invariant selects cost ~nothing at
/// runtime (a predicted branch), while the specialized twin doubles the body.
const MIN_SELECT_USES: usize = 4;

/// an unswitchable condition: a bool local computed purely from scalar
/// params, gating `uses` float selects in the body.
pub struct Candidate {
    /// the name of the bool let holding the invariant condition.
    pub cond_let: String,
    /// the condition's defining expression (over params only) — rendered by
    /// the emitter as the dispatcher's branch condition.
    pub cond_expr: ScalarExpr,
    /// how many selects the condition gates.
    pub uses: usize,
}

/// every Var referenced by `e` is in `names`; index space (field loads) and
/// casts make an expression non-invariant conservatively.
fn refs_only(e: &ScalarExpr, names: &HashSet<&str>) -> bool {
    match e {
        ScalarExpr::Var(n) => names.contains(n.as_str()),
        ScalarExpr::FieldLoadAt { .. } | ScalarExpr::Cast { .. } => false,
        _ => e.children().iter().all(|c| refs_only(c, names)),
    }
}

fn count_cond_uses(e: &ScalarExpr, name: &str, uses: &mut usize) {
    if let ScalarExpr::Select { cond, .. } = e {
        if matches!(cond.as_ref(), ScalarExpr::Var(n) if n == name) {
            *uses += 1;
        }
    }
    for c in e.children() {
        count_cond_uses(c, name, uses);
    }
}

fn walk_stmt_exprs<'a>(stmt: &'a ScalarStmt, out: &mut Vec<&'a ScalarExpr>) {
    match stmt {
        ScalarStmt::Let { value, .. } => out.push(value),
        ScalarStmt::LetMut { init, .. } => out.push(init),
        ScalarStmt::CompoundAssign { value, .. } | ScalarStmt::Assign { value, .. } => {
            out.push(value)
        }
        ScalarStmt::Scope { body, result, .. } => {
            for s in body {
                walk_stmt_exprs(s, out);
            }
            out.push(result);
        }
        ScalarStmt::For { body, .. }
        | ScalarStmt::If {
            then_body: body, ..
        } => {
            for s in body {
                walk_stmt_exprs(s, out);
            }
        }
        ScalarStmt::IfElse {
            cond,
            then_body,
            else_body,
            ..
        } => {
            out.push(cond);
            for s in then_body.iter().chain(else_body.iter()) {
                walk_stmt_exprs(s, out);
            }
        }
        ScalarStmt::Break => {}
    }
}

/// find the param-invariant bool local gating the most selects, if any gates
/// at least `MIN_SELECT_USES`. one level only — no 2^n multi-cond expansion.
///
/// `scalar_params` must be the kernel's declared scalar params only — the
/// scalarized param list also carries field inputs (per-cell base reads),
/// which are not invariant.
pub fn find(scalarized: &KernelScalarized, scalar_params: &HashSet<String>) -> Option<Candidate> {
    let param_names: HashSet<&str> = scalarized
        .params
        .iter()
        .map(|p| p.name.as_str())
        .filter(|n| scalar_params.contains(*n))
        .collect();
    // invariant bool lets, in definition order (top-level only: a scope-local
    // bool cannot gate selects outside its scope anyway).
    let mut invariant: HashMap<&str, &ScalarExpr> = HashMap::new();
    for stmt in &scalarized.body {
        if let ScalarStmt::Let {
            name,
            element: ElementTy::Bool,
            value,
        } = stmt
        {
            if refs_only(value, &param_names) {
                invariant.insert(name.as_str(), value);
            }
        }
    }
    if invariant.is_empty() {
        return None;
    }
    let mut all_exprs: Vec<&ScalarExpr> = Vec::new();
    for stmt in &scalarized.body {
        walk_stmt_exprs(stmt, &mut all_exprs);
    }
    all_exprs.extend(scalarized.outputs.iter());
    let mut best: Option<Candidate> = None;
    for (name, expr) in invariant {
        let mut uses = 0;
        for e in &all_exprs {
            count_cond_uses(e, name, &mut uses);
        }
        if uses >= MIN_SELECT_USES && best.as_ref().is_none_or(|b| uses > b.uses) {
            best = Some(Candidate {
                cond_let: name.to_string(),
                cond_expr: (*expr).clone(),
                uses,
            });
        }
    }
    best
}

fn specialize_expr(e: &mut ScalarExpr, cond_let: &str, value: bool) {
    for c in e.children_mut() {
        specialize_expr(c, cond_let, value);
    }
    if let ScalarExpr::Select { cond, then, else_ } = e {
        if matches!(cond.as_ref(), ScalarExpr::Var(n) if n == cond_let) {
            let arm = if value {
                std::mem::replace(then.as_mut(), ScalarExpr::Var(String::new()))
            } else {
                std::mem::replace(else_.as_mut(), ScalarExpr::Var(String::new()))
            };
            *e = arm;
        }
    }
}

fn specialize_stmt(stmt: &mut ScalarStmt, cond_let: &str, value: bool) {
    match stmt {
        ScalarStmt::Let { value: v, .. } | ScalarStmt::LetMut { init: v, .. } => {
            specialize_expr(v, cond_let, value)
        }
        ScalarStmt::CompoundAssign { value: v, .. } | ScalarStmt::Assign { value: v, .. } => {
            specialize_expr(v, cond_let, value)
        }
        ScalarStmt::Scope { body, result, .. } => {
            for s in body {
                specialize_stmt(s, cond_let, value);
            }
            specialize_expr(result, cond_let, value);
        }
        ScalarStmt::For { body, .. }
        | ScalarStmt::If {
            then_body: body, ..
        } => {
            for s in body {
                specialize_stmt(s, cond_let, value);
            }
        }
        ScalarStmt::IfElse {
            cond,
            then_body,
            else_body,
            ..
        } => {
            specialize_expr(cond, cond_let, value);
            for s in then_body.iter_mut().chain(else_body.iter_mut()) {
                specialize_stmt(s, cond_let, value);
            }
        }
        ScalarStmt::Break => {}
    }
}

/// partially evaluate the body at `cond_let == value`: every select gated on
/// the condition collapses to its taken arm. the defining let stays (dead
/// code; the compiler removes it).
pub fn specialize(scalarized: &mut KernelScalarized, cond_let: &str, value: bool) {
    for stmt in &mut scalarized.body {
        specialize_stmt(stmt, cond_let, value);
    }
    for out in &mut scalarized.outputs {
        specialize_expr(out, cond_let, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ConstValue;
    use crate::passes::scalarize::{BinaryKind, LoweredParam};

    fn f(x: f64) -> ScalarExpr {
        ScalarExpr::Const(ConstValue::F64(x))
    }
    fn var(n: &str) -> ScalarExpr {
        ScalarExpr::Var(n.to_string())
    }
    fn sel(cond: &str, t: ScalarExpr, e: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Select {
            cond: Box::new(var(cond)),
            then: Box::new(t),
            else_: Box::new(e),
        }
    }
    fn param(name: &str, element: ElementTy) -> LoweredParam {
        LoweredParam {
            name: name.to_string(),
            element,
            array_len: None,
        }
    }
    fn scalars(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }
    fn theta_lt_zero_kernel(n_selects: usize) -> KernelScalarized {
        let mut body = vec![ScalarStmt::Let {
            name: "m".into(),
            element: ElementTy::Bool,
            value: ScalarExpr::BinOp(BinaryKind::Lt, Box::new(var("theta")), Box::new(f(0.0))),
        }];
        for i in 0..n_selects {
            body.push(ScalarStmt::Let {
                name: format!("y{i}"),
                element: ElementTy::F64,
                value: sel("m", f(1.0), f(2.0)),
            });
        }
        KernelScalarized {
            params: vec![param("theta", ElementTy::F64)],
            body,
            outputs: vec![var("y0")],
        }
    }

    #[test]
    fn finds_param_invariant_condition() {
        let k = theta_lt_zero_kernel(4);
        let c = find(&k, &scalars(&["theta"])).expect("candidate");
        assert_eq!(c.cond_let, "m");
        assert_eq!(c.uses, 4);
    }

    #[test]
    fn below_use_threshold_is_not_worth_duplicating() {
        let k = theta_lt_zero_kernel(3);
        assert!(find(&k, &scalars(&["theta"])).is_none());
    }

    #[test]
    fn field_dependent_condition_is_not_invariant() {
        let mut k = theta_lt_zero_kernel(4);
        // the condition reads a field: it varies per cell.
        if let ScalarStmt::Let { value, .. } = &mut k.body[0] {
            *value = ScalarExpr::BinOp(
                BinaryKind::Lt,
                Box::new(ScalarExpr::FieldLoadAt {
                    field_key: "prim_rho".into(),
                    components: vec![var("_coord_0")],
                }),
                Box::new(f(0.0)),
            );
        }
        assert!(find(&k, &scalars(&["theta"])).is_none());
    }

    #[test]
    fn specialize_collapses_gated_selects_only() {
        let mut k = theta_lt_zero_kernel(4);
        // add a select on a different (cell-varying) condition; it must survive.
        k.body.push(ScalarStmt::Let {
            name: "other".into(),
            element: ElementTy::F64,
            value: sel("cell_mask", f(3.0), f(4.0)),
        });
        specialize(&mut k, "m", true);
        for i in 0..4 {
            match &k.body[1 + i] {
                ScalarStmt::Let { value, .. } => assert_eq!(*value, f(1.0)),
                other => panic!("expected Let, got {other:?}"),
            }
        }
        specialize(&mut k, "m", false); // idempotent: gated selects already gone
        match k.body.last().unwrap() {
            ScalarStmt::Let { value, .. } => {
                assert!(matches!(value, ScalarExpr::Select { .. }))
            }
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn specialize_false_takes_else_arm() {
        let mut k = theta_lt_zero_kernel(4);
        specialize(&mut k, "m", false);
        match &k.body[1] {
            ScalarStmt::Let { value, .. } => assert_eq!(*value, f(2.0)),
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn nested_gated_selects_collapse() {
        // select(m, select(m, a, b), c) at m=true must reduce fully to a.
        let mut k = KernelScalarized {
            params: vec![param("theta", ElementTy::F64)],
            body: vec![
                ScalarStmt::Let {
                    name: "m".into(),
                    element: ElementTy::Bool,
                    value: ScalarExpr::BinOp(
                        BinaryKind::Lt,
                        Box::new(var("theta")),
                        Box::new(f(0.0)),
                    ),
                },
                ScalarStmt::Let {
                    name: "y".into(),
                    element: ElementTy::F64,
                    value: sel("m", sel("m", f(1.0), f(2.0)), f(3.0)),
                },
            ],
            outputs: vec![var("y")],
        };
        specialize(&mut k, "m", true);
        match &k.body[1] {
            ScalarStmt::Let { value, .. } => assert_eq!(*value, f(1.0)),
            other => panic!("expected Let, got {other:?}"),
        }
    }
}
