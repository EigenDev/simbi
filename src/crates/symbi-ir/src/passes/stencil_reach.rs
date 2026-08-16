// =============================================================================
// stencil_reach.rs
//
// pure analysis: the per-field, per-axis stencil reach of a scalarized kernel,
// read off its FieldLoadAt index expressions.
// each load component is classified as an affine function of the coord vars:
// - `_coord_a + c` (unit coefficient on the component's own axis) is a stencil
//   offset — the reach contribution is |c|
// - anything else (a scaled coord like `2*ii`, a lattice-map select, a runtime
//   var, an absolute constant, a transposed coord) is Unbounded — the load can
//   reach outside any fixed halo, so the analysis refuses to bound it
// cse hoists index arithmetic into immutable lets (`__cse_0 = _coord_0 - 2`),
// so vars resolve through the let environment in program order; mutable names
// (LetMut accumulators, For iterators, IfElse outs) never enter it.
// the reach is the max over every load of a field along each axis. consumers:
// the ghost-width law (allocated halo >= reach for every dispatched kernel),
// the cover-executor tile overlap, and the multi-node exchange width.
//
// usage:
//   let report = stencil_reach(&prepared.scalarized);
//   for (field, axes) in &report.per_field { ... }
//   let widest = report.max_bounded();
// =============================================================================

use std::collections::BTreeMap;

use super::scalarize::{BinaryKind, KernelScalarized, ScalarExpr, ScalarStmt, UnaryKind};
use crate::graph::ConstValue;

/// the reach of a field's loads along one axis: the max |offset| of its
/// unit-stride stencil reads, or Unbounded when any load's index cannot be
/// bounded (scaled, selected, runtime-dependent, or absolute addressing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisReach {
    Bounded(u32),
    Unbounded,
}

impl AxisReach {
    /// the join of two reaches: bounded maxes combine; Unbounded absorbs.
    fn join(self, other: AxisReach) -> AxisReach {
        match (self, other) {
            (AxisReach::Bounded(a), AxisReach::Bounded(b)) => AxisReach::Bounded(a.max(b)),
            _ => AxisReach::Unbounded,
        }
    }
}

/// per-field, per-axis reach of every FieldLoadAt in a kernel. fields read only
/// at the cell center (no FieldLoadAt) do not appear — their reach is zero by
/// construction. BTreeMap for deterministic iteration in test output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReachReport {
    pub per_field: BTreeMap<String, Vec<AxisReach>>,
}

impl ReachReport {
    /// the widest bounded reach across every field and axis; None when the
    /// kernel has no stencil loads at all.
    pub fn max_bounded(&self) -> Option<u32> {
        self.per_field
            .values()
            .flatten()
            .filter_map(|r| match r {
                AxisReach::Bounded(w) => Some(*w),
                AxisReach::Unbounded => None,
            })
            .max()
    }

    /// every (field, axis) whose reach the analysis could not bound.
    pub fn unbounded(&self) -> Vec<(&str, usize)> {
        self.per_field
            .iter()
            .flat_map(|(field, axes)| {
                axes.iter().enumerate().filter_map(move |(ax, r)| {
                    matches!(r, AxisReach::Unbounded).then_some((field.as_str(), ax))
                })
            })
            .collect()
    }

    /// fold one load's per-axis classification into the report.
    fn record(&mut self, field: &str, axis_reach: &[AxisReach]) {
        let entry = self
            .per_field
            .entry(field.to_string())
            .or_insert_with(|| vec![AxisReach::Bounded(0); axis_reach.len()]);
        // a field loaded with differing component counts cannot be reasoned
        // about per-axis; widen the entry and mark the extra axes unbounded.
        if entry.len() < axis_reach.len() {
            entry.resize(axis_reach.len(), AxisReach::Unbounded);
        }
        for (ax, r) in axis_reach.iter().enumerate() {
            entry[ax] = entry[ax].join(*r);
        }
    }
}

/// an index expression as a linear function of the coord vars: per-axis integer
/// coefficients plus an integer constant. None when the expression is not
/// affine-integer (division, method calls, selects, runtime vars).
#[derive(Clone, Debug, Default)]
struct Affine {
    coeffs: BTreeMap<usize, i64>,
    constant: i64,
}

impl Affine {
    fn constant(c: i64) -> Affine {
        Affine {
            coeffs: BTreeMap::new(),
            constant: c,
        }
    }

    fn coord(axis: usize) -> Affine {
        Affine {
            coeffs: BTreeMap::from([(axis, 1)]),
            constant: 0,
        }
    }

    fn add(mut self, other: Affine, sign: i64) -> Affine {
        for (axis, c) in other.coeffs {
            *self.coeffs.entry(axis).or_insert(0) += sign * c;
        }
        self.constant += sign * other.constant;
        self.coeffs.retain(|_, c| *c != 0);
        self
    }

    fn scale(mut self, k: i64) -> Affine {
        for c in self.coeffs.values_mut() {
            *c *= k;
        }
        self.constant *= k;
        self.coeffs.retain(|_, c| *c != 0);
        self
    }

    /// true when the expression is exactly `_coord_axis + constant` — the
    /// unit-stride stencil form the reach can bound.
    fn is_unit_stencil(&self, axis: usize) -> bool {
        self.coeffs.len() == 1 && self.coeffs.get(&axis) == Some(&1)
    }
}

/// the immutable-let environment: names whose bound value is affine-integer.
/// only ScalarStmt::Let enters — LetMut/Assign names are mutable and excluded.
type AffineEnv = BTreeMap<String, Affine>;

/// extract the affine form of an integer index expression, or None when the
/// expression is not a linear-integer function of the coord vars. mirrors the
/// integer index grammar the emitters accept (`render_index_expr`), minus the
/// data-independent select — a selected index is bounded only by its arms'
/// values, which this analysis does not chase. vars resolve through the
/// immutable-let environment (cse hoists index arithmetic into lets).
fn affine(e: &ScalarExpr, env: &AffineEnv) -> Option<Affine> {
    match e {
        ScalarExpr::Var(name) => name
            .strip_prefix("_coord_")
            .and_then(|axis| axis.parse::<usize>().ok())
            .map(Affine::coord)
            .or_else(|| env.get(name).cloned()),
        ScalarExpr::Const(c) => int_const(c).map(Affine::constant),
        ScalarExpr::BinOp(BinaryKind::Add, a, b) => Some(affine(a, env)?.add(affine(b, env)?, 1)),
        ScalarExpr::BinOp(BinaryKind::Sub, a, b) => Some(affine(a, env)?.add(affine(b, env)?, -1)),
        ScalarExpr::BinOp(BinaryKind::Mul, a, b) => {
            let (fa, fb) = (affine(a, env)?, affine(b, env)?);
            // linear times linear stays linear only when one side is constant.
            if fa.coeffs.is_empty() {
                Some(fb.scale(fa.constant))
            } else if fb.coeffs.is_empty() {
                Some(fa.scale(fb.constant))
            } else {
                None
            }
        }
        ScalarExpr::UnaryOp(UnaryKind::Neg, a) => Some(affine(a, env)?.scale(-1)),
        _ => None,
    }
}

/// the integer value of a constant, when it is exactly integral.
fn int_const(c: &ConstValue) -> Option<i64> {
    match c {
        ConstValue::F64(v) if v.fract() == 0.0 => Some(*v as i64),
        ConstValue::F32(v) if v.fract() == 0.0 => Some(*v as i64),
        ConstValue::I32(v) => Some(*v as i64),
        ConstValue::U32(v) => Some(*v as i64),
        _ => None,
    }
}

/// classify one load component at axis position `axis`: the |offset| of a
/// unit-stride stencil read, or Unbounded for everything else. a coefficient
/// other than 1 on the own axis, any cross-axis term, and a pure constant
/// (absolute addressing) all escape a cell-relative halo bound.
fn component_reach(e: &ScalarExpr, axis: usize, env: &AffineEnv) -> AxisReach {
    match affine(e, env) {
        Some(f) if f.is_unit_stencil(axis) => AxisReach::Bounded(f.constant.unsigned_abs() as u32),
        _ => AxisReach::Unbounded,
    }
}

/// recursively collect every FieldLoadAt in an expression tree into the report.
/// components are walked too: a nested load used as an index is itself recorded
/// (and the outer component that contains it classifies as Unbounded).
fn visit_expr(e: &ScalarExpr, env: &AffineEnv, report: &mut ReachReport) {
    if let ScalarExpr::FieldLoadAt {
        field_key,
        components,
    } = e
    {
        let axes: Vec<AxisReach> = components
            .iter()
            .enumerate()
            .map(|(ax, comp)| component_reach(comp, ax, env))
            .collect();
        report.record(field_key, &axes);
    }
    for child in e.children() {
        visit_expr(child, env, report);
    }
}

/// walk a statement body (through For/If/Scope/IfElse sub-bodies) collecting
/// loads, binding immutable lets into the environment in program order. kernel
/// let names are globally fresh (the scalarizer's fresh-name counter), so one
/// flat environment across nested scopes is sound.
fn visit_stmts(stmts: &[ScalarStmt], env: &mut AffineEnv, report: &mut ReachReport) {
    for stmt in stmts {
        // sub-bodies first: a Scope's result references lets declared inside
        // its own body, so the body's bindings must be in the environment
        // before the immediate expression is classified.
        for body in stmt.child_stmt_bodies() {
            visit_stmts(body, env, report);
        }
        if let Some(e) = stmt.child_expr() {
            visit_expr(e, env, report);
        }
        if let ScalarStmt::Let { name, value, .. } = stmt
            && let Some(f) = affine(value, env)
        {
            env.insert(name.clone(), f);
        }
    }
}

/// the per-field, per-axis stencil reach of a scalarized kernel: every
/// FieldLoadAt in the body and the output expressions, classified and joined.
pub fn stencil_reach(k: &KernelScalarized) -> ReachReport {
    let mut report = ReachReport::default();
    let mut env = AffineEnv::new();
    visit_stmts(&k.body, &mut env, &mut report);
    for out in &k.outputs {
        visit_expr(out, &env, &mut report);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementTy;

    fn coord(axis: usize) -> ScalarExpr {
        ScalarExpr::Var(format!("_coord_{axis}"))
    }

    fn shifted(axis: usize, offset: i64) -> ScalarExpr {
        ScalarExpr::BinOp(
            BinaryKind::Add,
            Box::new(coord(axis)),
            Box::new(ScalarExpr::Const(ConstValue::F64(offset as f64))),
        )
    }

    fn load(field: &str, components: Vec<ScalarExpr>) -> ScalarExpr {
        ScalarExpr::FieldLoadAt {
            field_key: field.to_string(),
            components,
        }
    }

    fn kernel(body: Vec<ScalarStmt>, outputs: Vec<ScalarExpr>) -> KernelScalarized {
        KernelScalarized {
            params: Vec::new(),
            body,
            outputs,
        }
    }

    fn let_stmt(name: &str, value: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Let {
            name: name.to_string(),
            element: ElementTy::F64,
            value,
        }
    }

    #[test]
    fn unit_stencil_offsets_bound_the_reach() {
        // a 1d plm-style stencil: loads at -2, -1, 0, +1 -> reach 2.
        let body = vec![
            let_stmt("qm2", load("prim_rho", vec![shifted(0, -2)])),
            let_stmt("qm1", load("prim_rho", vec![shifted(0, -1)])),
            let_stmt("q0", load("prim_rho", vec![coord(0)])),
            let_stmt("qp1", load("prim_rho", vec![shifted(0, 1)])),
        ];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(report.per_field["prim_rho"], vec![AxisReach::Bounded(2)]);
        assert_eq!(report.max_bounded(), Some(2));
        assert!(report.unbounded().is_empty());
    }

    #[test]
    fn per_axis_reach_is_independent() {
        // a 2d load offset only along axis 1: axis 0 reach 0, axis 1 reach 1.
        let body = vec![let_stmt(
            "q",
            load("prim_vel", vec![coord(0), shifted(1, -1)]),
        )];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(
            report.per_field["prim_vel"],
            vec![AxisReach::Bounded(0), AxisReach::Bounded(1)],
        );
    }

    #[test]
    fn scaled_coord_is_unbounded() {
        // a refinement-style index `2*ii` reaches proportionally to the coord,
        // so no fixed halo bounds it.
        let two_ii = ScalarExpr::BinOp(
            BinaryKind::Mul,
            Box::new(ScalarExpr::Const(ConstValue::F64(2.0))),
            Box::new(coord(0)),
        );
        let body = vec![let_stmt("q", load("coarse_rho", vec![two_ii]))];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(report.per_field["coarse_rho"], vec![AxisReach::Unbounded]);
        assert_eq!(report.unbounded(), vec![("coarse_rho", 0)]);
    }

    #[test]
    fn lattice_map_select_is_unbounded() {
        // a ghost-fill source coord picks periodic/reflect by a runtime map_type:
        // the index is data-independent but not affine — refuse to bound it.
        let sel = ScalarExpr::Select {
            cond: ScalarExpr::Var("map_type".to_string()).into(),
            then: shifted(0, 4).into(),
            else_: coord(0).into(),
        };
        let body = vec![let_stmt("q", load("prim_rho", vec![sel]))];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(report.per_field["prim_rho"], vec![AxisReach::Unbounded]);
    }

    #[test]
    fn transposed_and_absolute_indices_are_unbounded() {
        // component 0 reading _coord_1 is a transposed access; a bare constant
        // is absolute addressing. neither is a cell-relative stencil.
        let body = vec![
            let_stmt("qt", load("f_transposed", vec![coord(1)])),
            let_stmt(
                "qa",
                load("f_absolute", vec![ScalarExpr::Const(ConstValue::F64(0.0))]),
            ),
        ];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(report.per_field["f_transposed"], vec![AxisReach::Unbounded]);
        assert_eq!(report.per_field["f_absolute"], vec![AxisReach::Unbounded]);
    }

    #[test]
    fn loads_in_nested_bodies_and_outputs_are_found() {
        // loads hide inside For bodies, IfElse arms, Scope results, and the
        // output expressions; the walk must reach all of them.
        let body = vec![
            ScalarStmt::For {
                iter: "ii".to_string(),
                bound: crate::DimExpr::Literal(3),
                body: vec![let_stmt("a", load("f_for", vec![shifted(0, 1)]))],
            },
            ScalarStmt::IfElse {
                outs: vec![("o".to_string(), ElementTy::F64)],
                cond: ScalarExpr::Var("c".to_string()),
                then_body: vec![let_stmt("b", load("f_then", vec![shifted(0, -2)]))],
                else_body: vec![ScalarStmt::Assign {
                    name: "o".to_string(),
                    value: ScalarExpr::Const(ConstValue::F64(0.0)),
                }],
            },
            ScalarStmt::Scope {
                name: "s".to_string(),
                element: ElementTy::F64,
                body: Vec::new(),
                result: load("f_scope", vec![shifted(0, 3)]),
            },
        ];
        let outputs = vec![load("f_out", vec![shifted(0, -1)])];
        let report = stencil_reach(&kernel(body, outputs));
        assert_eq!(report.per_field["f_for"], vec![AxisReach::Bounded(1)]);
        assert_eq!(report.per_field["f_then"], vec![AxisReach::Bounded(2)]);
        assert_eq!(report.per_field["f_scope"], vec![AxisReach::Bounded(3)]);
        assert_eq!(report.per_field["f_out"], vec![AxisReach::Bounded(1)]);
    }

    #[test]
    fn widened_stencil_is_named_by_field_and_axis() {
        // the bug-injection gate for the ghost-width law: widening one load
        // past the halo must surface as a nameable (field, axis, reach) fact.
        let narrow = stencil_reach(&kernel(
            vec![let_stmt(
                "q",
                load("prim_rho", vec![coord(0), shifted(1, 1)]),
            )],
            Vec::new(),
        ));
        let widened = stencil_reach(&kernel(
            vec![let_stmt(
                "q",
                load("prim_rho", vec![coord(0), shifted(1, 3)]),
            )],
            Vec::new(),
        ));
        let halo = 2u32;
        let violations = |r: &ReachReport| -> Vec<(String, usize, u32)> {
            r.per_field
                .iter()
                .flat_map(|(f, axes)| {
                    axes.iter()
                        .enumerate()
                        .filter_map(move |(ax, reach)| match reach {
                            AxisReach::Bounded(w) if *w > halo => Some((f.clone(), ax, *w)),
                            _ => None,
                        })
                })
                .collect()
        };
        assert!(violations(&narrow).is_empty());
        assert_eq!(violations(&widened), vec![("prim_rho".to_string(), 1, 3)]);
    }

    #[test]
    fn cse_let_bound_indices_resolve_through_the_environment() {
        // cse hoists index arithmetic into immutable lets (`__cse_0 =
        // _coord_0 - 2`) and the load component becomes a bare var; the
        // classification must see through the binding. a LetMut accumulator
        // of the same shape must not resolve — its value changes.
        let idx = ScalarExpr::BinOp(
            BinaryKind::Add,
            Box::new(coord(0)),
            Box::new(ScalarExpr::Const(ConstValue::I32(-2))),
        );
        let body = vec![
            ScalarStmt::Let {
                name: "__cse_0".to_string(),
                element: ElementTy::I32,
                value: idx.clone(),
            },
            let_stmt(
                "q",
                load(
                    "prim_rho",
                    vec![ScalarExpr::Var("__cse_0".to_string()), coord(1)],
                ),
            ),
            ScalarStmt::LetMut {
                name: "acc".to_string(),
                element: ElementTy::I32,
                init: idx,
            },
            let_stmt(
                "r",
                load(
                    "prim_pre",
                    vec![ScalarExpr::Var("acc".to_string()), coord(1)],
                ),
            ),
        ];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(
            report.per_field["prim_rho"],
            vec![AxisReach::Bounded(2), AxisReach::Bounded(0)],
        );
        assert_eq!(
            report.per_field["prim_pre"],
            vec![AxisReach::Unbounded, AxisReach::Bounded(0)],
        );
    }

    #[test]
    fn commuted_and_negated_offsets_normalize() {
        // `1 + ii`, `ii - 2`, and `-( -ii - 1 )` are all unit stencils.
        let one_plus = ScalarExpr::BinOp(
            BinaryKind::Add,
            Box::new(ScalarExpr::Const(ConstValue::F64(1.0))),
            Box::new(coord(0)),
        );
        let minus_two = ScalarExpr::BinOp(
            BinaryKind::Sub,
            Box::new(coord(0)),
            Box::new(ScalarExpr::Const(ConstValue::F64(2.0))),
        );
        let double_neg = ScalarExpr::UnaryOp(
            UnaryKind::Neg,
            Box::new(ScalarExpr::BinOp(
                BinaryKind::Sub,
                Box::new(ScalarExpr::UnaryOp(UnaryKind::Neg, Box::new(coord(0)))),
                Box::new(ScalarExpr::Const(ConstValue::F64(1.0))),
            )),
        );
        let body = vec![
            let_stmt("a", load("f", vec![one_plus])),
            let_stmt("b", load("f", vec![minus_two])),
            let_stmt("c", load("f", vec![double_neg])),
        ];
        let report = stencil_reach(&kernel(body, Vec::new()));
        assert_eq!(report.per_field["f"], vec![AxisReach::Bounded(2)]);
    }
}
