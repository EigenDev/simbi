// =============================================================================
// support_infer.rs
//
// support PROPAGATION over the traced graph: derive a kernel's output support
// automatically from builder-tagged mask nodes. a tag asserts "this node's
// value is exactly zero (f64) outside this
// ball, for every field input" — the saturation lemma lives where the mask is
// built. propagation classifies every node reachable from a write root:
// - Zero(b):  exactly zero outside b (None = zero everywhere)
// - One(b):   exactly one outside b  (None = one everywhere)
// - Eq(m, b): exactly equal to node m outside b
// - Unknown:  no statement (always sound)
// exact-arithmetic rules only (0*x = 0, x+0 = x, exp(0) = 1, 1-1 = 0, ... hold
// bit-exactly in f64), so a derived ball is as strong as a declared one. a
// write whose root is Zero contributes its ball; an in-place write whose root
// is Eq to its own field's offset-0 read is unchanged-valued outside the ball
// and contributes it too; anything else widens the kernel to Everywhere —
// a structural change that breaks the mask chain degrades the support
// FAIL-SAFE to a too-wide region; a stale narrow ball would be unsound.
//
// usage:
//   tag_support_ball(&chi, center_exprs, radius_expr);   // at the mask seam
//   let kernel = end_trace().with_derived_support(&writes);
// =============================================================================

use std::collections::HashMap;

use crate::FieldBind;
use crate::graph::{ConstValue, ElementWiseOp, Graph, NodeId, Op};
use crate::support::{ParamExpr, Support};
use crate::symbol::Symbol;

/// a symbolic ball over the kernel's scalar params. equality is structural:
/// propagation only ever joins balls descending from the same tag, so the
/// exprs are literally identical when the regions are.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportBall {
    pub center: Vec<ParamExpr>,
    pub radius: ParamExpr,
}

#[derive(Clone, PartialEq)]
enum Class {
    Zero(Option<SupportBall>),
    One(Option<SupportBall>),
    Eq(NodeId, SupportBall),
    Unknown,
}

// join two "outside" regions: the combined statement holds outside the union.
// None is the empty region (the statement holds everywhere). two DIFFERENT
// balls have no expressible enclosing ball in the ParamExpr language — the
// caller degrades to Unknown (sound).
fn join(a: &Option<SupportBall>, b: &Option<SupportBall>) -> Result<Option<SupportBall>, ()> {
    match (a, b) {
        (None, x) | (x, None) => Ok(x.clone()),
        (Some(x), Some(y)) if x == y => Ok(Some(x.clone())),
        _ => Err(()),
    }
}

fn const_f64(op: &Op) -> Option<f64> {
    match op {
        Op::Const(ConstValue::F64(v)) => Some(*v),
        Op::Const(ConstValue::F32(v)) => Some(*v as f64),
        _ => None,
    }
}

fn classify(
    g: &Graph,
    tags: &HashMap<NodeId, SupportBall>,
    memo: &mut HashMap<NodeId, Class>,
    id: NodeId,
) -> Class {
    if let Some(c) = memo.get(&id) {
        return c.clone();
    }
    // a tag overrides structure: the builder asserts zero outside the ball
    // (validated downstream by the compiled-kernel sampler).
    let class = if let Some(ball) = tags.get(&id) {
        Class::Zero(Some(ball.clone()))
    } else {
        classify_op(g, tags, memo, id)
    };
    memo.insert(id, class.clone());
    class
}

fn classify_op(
    g: &Graph,
    tags: &HashMap<NodeId, SupportBall>,
    memo: &mut HashMap<NodeId, Class>,
    id: NodeId,
) -> Class {
    use Class::*;
    let op = &g.node(id).op;
    if let Some(v) = const_f64(op) {
        if v == 0.0 {
            return Zero(None);
        }
        if v == 1.0 {
            return One(None);
        }
        return Unknown;
    }
    match op {
        Op::ElementWise(ew, ins) => {
            let bin = |memo: &mut HashMap<NodeId, Class>| {
                (
                    classify(g, tags, memo, ins[0]),
                    classify(g, tags, memo, ins[1]),
                )
            };
            match ew {
                ElementWiseOp::Mul => {
                    let (ca, cb) = bin(memo);
                    // zero annihilates; the tighter statement (zero everywhere) wins.
                    match (&ca, &cb) {
                        (Zero(None), _) | (_, Zero(None)) => return Zero(None),
                        (Zero(b), _) | (_, Zero(b)) => return Zero(b.clone()),
                        (One(a), One(b)) => {
                            return join(a, b).map(One).unwrap_or(Unknown);
                        }
                        // x * 1 == x: outside the one-ball the product IS the
                        // other operand; a one-everywhere factor is transparent.
                        (One(None), _) => return cb,
                        (_, One(None)) => return ca,
                        (One(Some(b)), _) => return Eq(ins[1], b.clone()),
                        (_, One(Some(b))) => return Eq(ins[0], b.clone()),
                        _ => return Unknown,
                    }
                }
                ElementWiseOp::Div => {
                    let (ca, cb) = bin(memo);
                    // 0 / x == 0 wherever x != 0 — the same everywhere-nonzero
                    // premise the kernel's own physics (a division by density)
                    // already relies on.
                    match (&ca, &cb) {
                        (Zero(b), _) => return Zero(b.clone()),
                        (_, One(None)) => return ca,
                        (_, One(Some(b))) => return Eq(ins[0], b.clone()),
                        _ => return Unknown,
                    }
                }
                ElementWiseOp::Add => {
                    let (ca, cb) = bin(memo);
                    match (&ca, &cb) {
                        (Zero(a), Zero(b)) => return join(a, b).map(Zero).unwrap_or(Unknown),
                        (Zero(a), One(b)) | (One(b), Zero(a)) => {
                            return join(a, b).map(One).unwrap_or(Unknown);
                        }
                        (Zero(a), Eq(m, b)) | (Eq(m, b), Zero(a)) => {
                            return join(a, &Some(b.clone()))
                                .map(|j| Eq(*m, j.expect("ball join with Some is Some")))
                                .unwrap_or(Unknown);
                        }
                        (Eq(m, a), Eq(n, b)) if m == n => {
                            return join(&Some(a.clone()), &Some(b.clone()))
                                .map(|j| Eq(*m, j.expect("ball join with Some is Some")))
                                .unwrap_or(Unknown);
                        }
                        // x + (zero outside b) == x outside b — the in-place
                        // increment pattern `field + masked-term`.
                        (_, Zero(Some(b))) => return Eq(ins[0], b.clone()),
                        (Zero(Some(b)), _) => return Eq(ins[1], b.clone()),
                        _ => return Unknown,
                    }
                }
                ElementWiseOp::Sub => {
                    let (ca, cb) = bin(memo);
                    match (&ca, &cb) {
                        (Zero(a), Zero(b)) => return join(a, b).map(Zero).unwrap_or(Unknown),
                        // 1 - 1 == 0 outside both regions.
                        (One(a), One(b)) => return join(a, b).map(Zero).unwrap_or(Unknown),
                        // x - (x outside b) == 0 outside b — the absorbed-delta
                        // pattern `field - field*factor`.
                        (_, Eq(m, b)) if *m == ins[0] => return Zero(Some(b.clone())),
                        (Eq(m, b), _) if *m == ins[1] => return Zero(Some(b.clone())),
                        (Eq(m, a), Eq(n, b)) if m == n => {
                            return join(&Some(a.clone()), &Some(b.clone()))
                                .map(Zero)
                                .unwrap_or(Unknown);
                        }
                        // x - 0 == x.
                        (_, Zero(None)) => return ca,
                        (Zero(a), Eq(m, b)) => {
                            let _ = (a, m, b);
                            return Unknown;
                        }
                        (_, Zero(Some(b))) => {
                            return match &ca {
                                Zero(a) => join(a, &Some(b.clone())).map(Zero).unwrap_or(Unknown),
                                One(a) => join(a, &Some(b.clone())).map(One).unwrap_or(Unknown),
                                Eq(m, a) => join(&Some(a.clone()), &Some(b.clone()))
                                    .map(|j| Eq(*m, j.expect("ball join with Some is Some")))
                                    .unwrap_or(Unknown),
                                _ => Eq(ins[0], b.clone()),
                            };
                        }
                        _ => return Unknown,
                    }
                }
                ElementWiseOp::Min | ElementWiseOp::Max => {
                    let (ca, cb) = bin(memo);
                    if let (Zero(a), Zero(b)) = (&ca, &cb) {
                        return join(a, b).map(Zero).unwrap_or(Unknown);
                    }
                    // min(f, c >= 1) == f and max(f, c <= 1) == f wherever
                    // f == 1 — the growth-cap pattern min(exp(..), cap).
                    let one_vs_const = |cone: &Class, other: NodeId| -> Class {
                        if let One(b) = cone {
                            if let Some(c) = const_f64(&g.node(other).op) {
                                let keeps = match ew {
                                    ElementWiseOp::Min => c >= 1.0,
                                    _ => c <= 1.0,
                                };
                                if keeps {
                                    return One(b.clone());
                                }
                            }
                        }
                        Unknown
                    };
                    let r = one_vs_const(&ca, ins[1]);
                    if r != Unknown {
                        return r;
                    }
                    return one_vs_const(&cb, ins[0]);
                }
                // f(0) = 0 exactly in f64 (sin/sinh/asinh trace as element-wise ops).
                // `Asinh` was classified only on the TRANSCENDENTAL twin of this table while
                // `.asinh()` traces through the element-wise op, so every traced asinh lost
                // zero-propagation and silently degraded its kernel's support to Everywhere.
                // two op enums carrying the same math is the root defect; until they merge,
                // the two tables must carry the same classes.
                ElementWiseOp::Neg
                | ElementWiseOp::Abs
                | ElementWiseOp::Sqrt
                | ElementWiseOp::Sin
                | ElementWiseOp::Tan
                | ElementWiseOp::Asin
                | ElementWiseOp::Atan
                | ElementWiseOp::Sinh
                | ElementWiseOp::Tanh
                | ElementWiseOp::Asinh
                | ElementWiseOp::Atanh => {
                    return match classify(g, tags, memo, ins[0]) {
                        Zero(b) => Zero(b),
                        One(b) if matches!(ew, ElementWiseOp::Abs | ElementWiseOp::Sqrt) => One(b),
                        _ => Unknown,
                    };
                }
                // f(0) = 1 exactly in f64.
                ElementWiseOp::Cos
                | ElementWiseOp::Cosh
                | ElementWiseOp::Exp
                | ElementWiseOp::Exp2 => {
                    return match classify(g, tags, memo, ins[0]) {
                        Zero(b) => One(b),
                        _ => Unknown,
                    };
                }
                ElementWiseOp::Cast(_) => return classify(g, tags, memo, ins[0]),
                _ => return Unknown,
            }
        }
        Op::Select(_, t, f) => {
            let (ct, cf) = (classify(g, tags, memo, *t), classify(g, tags, memo, *f));
            match (&ct, &cf) {
                (Class::Zero(a), Class::Zero(b)) => {
                    join(a, b).map(Class::Zero).unwrap_or(Class::Unknown)
                }
                (Class::Eq(m, a), Class::Eq(n, b)) if m == n => {
                    join(&Some(a.clone()), &Some(b.clone()))
                        .map(|j| Class::Eq(*m, j.expect("ball join with Some is Some")))
                        .unwrap_or(Class::Unknown)
                }
                _ => Class::Unknown,
            }
        }
        // the lazy branch (S::cond / cond_vec): result `index` takes one of the
        // two arm results — classify like an eager select over that pair.
        Op::Proj { source, index } => {
            if let Op::IfElse {
                then_results,
                else_results,
                ..
            } = &g.node(*source).op
            {
                let (t, f) = (then_results[*index as usize], else_results[*index as usize]);
                let (ct, cf) = (classify(g, tags, memo, t), classify(g, tags, memo, f));
                return match (&ct, &cf) {
                    (Class::Zero(a), Class::Zero(b)) => {
                        join(a, b).map(Class::Zero).unwrap_or(Class::Unknown)
                    }
                    (Class::Eq(m, a), Class::Eq(n, b)) if m == n => {
                        join(&Some(a.clone()), &Some(b.clone()))
                            .map(|j| Class::Eq(*m, j.expect("ball join with Some is Some")))
                            .unwrap_or(Class::Unknown)
                    }
                    _ => Class::Unknown,
                };
            }
            Class::Unknown
        }
        Op::IfElse {
            then_results,
            else_results,
            ..
        } if then_results.len() == 1 && else_results.len() == 1 => {
            let (ct, cf) = (
                classify(g, tags, memo, then_results[0]),
                classify(g, tags, memo, else_results[0]),
            );
            match (&ct, &cf) {
                (Class::Zero(a), Class::Zero(b)) => {
                    join(a, b).map(Class::Zero).unwrap_or(Class::Unknown)
                }
                (Class::Eq(m, a), Class::Eq(n, b)) if m == n => {
                    join(&Some(a.clone()), &Some(b.clone()))
                        .map(|j| Class::Eq(*m, j.expect("ball join with Some is Some")))
                        .unwrap_or(Class::Unknown)
                }
                _ => Class::Unknown,
            }
        }
        Op::Index(x, _) | Op::Broadcast(x, _) => classify(g, tags, memo, *x),
        Op::Construct(items) => {
            let mut acc: Option<SupportBall> = None;
            for it in items {
                match classify(g, tags, memo, *it) {
                    Class::Zero(b) => match join(&acc, &b) {
                        Ok(j) => acc = j,
                        Err(()) => return Class::Unknown,
                    },
                    _ => return Class::Unknown,
                }
            }
            Class::Zero(acc)
        }
        _ => Class::Unknown,
    }
}

/// derive the kernel-level output support from the tagged mask nodes:
/// - a write classified Zero contributes its ball (None contributes nothing);
/// - an in-place write classified Eq to its OWN field's offset-0 read is
///   unchanged-valued outside the ball and contributes it;
/// - anything else widens to Everywhere.
/// no tags -> Everywhere (nothing to propagate from).
pub fn derive_output_support(
    graph: &Graph,
    tags: &HashMap<NodeId, SupportBall>,
    field_inputs: &[(String, FieldBind)],
    writes: &[(String, FieldBind, NodeId)],
) -> Support {
    if tags.is_empty() || writes.is_empty() {
        return Support::Everywhere;
    }
    let mut memo: HashMap<NodeId, Class> = HashMap::new();
    let mut acc: Option<SupportBall> = None;
    let mut any_ball = false;
    for (_, bind, root) in writes {
        let contribution: Option<SupportBall> = match classify(graph, tags, &mut memo, *root) {
            Class::Zero(b) => b,
            Class::Eq(m, b) => {
                // the write must equal ITS OWN field outside the ball: find the
                // offset-0 read param bound to the same runtime field.
                let same_field_read = field_inputs
                    .iter()
                    .filter(|(_, fb)| fb == bind)
                    .filter_map(|(key, _)| graph.param(&Symbol::intern(key)))
                    .any(|pid| pid == m);
                if !same_field_read {
                    return Support::Everywhere;
                }
                Some(b)
            }
            _ => return Support::Everywhere,
        };
        if let Some(b) = contribution {
            any_ball = true;
            match join(&acc, &Some(b)) {
                Ok(j) => acc = j,
                Err(()) => return Support::Everywhere,
            }
        }
    }
    match (any_ball, acc) {
        (true, Some(b)) => Support::Ball {
            center: b.center,
            radius: b.radius,
        },
        // every write is zero everywhere: the kernel does nothing.
        (false, _) | (true, None) => Support::Empty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FieldRef;
    use crate::algebra::Scalar;
    use crate::gv::{Gv, Writes, begin_trace, end_trace, tag_support_ball};
    use symbi_algebra::algebra::Numeric;

    fn ball() -> SupportBall {
        SupportBall {
            center: vec![ParamExpr::param("body_0_pos_0")],
            radius: ParamExpr::param("body_0_racc"),
        }
    }

    // trace `f`, tagging the mask it returns, and derive the support of the
    // writes it produces.
    fn derive(f: impl FnOnce(Gv) -> Writes) -> Support {
        begin_trace();
        let den = Gv::field("den", FieldRef::cons_den());
        // the stand-in mask: a field-dependent value the builder ASSERTS is
        // ball-supported (the lemma is the tag here; the algebra carries no support fact).
        let chi = Gv::field("mask", FieldRef::PrimRho);
        tag_support_ball(&chi, ball().center, ball().radius);
        let _ = den;
        let writes = f(chi);
        let k = end_trace();
        derive_output_support(&k.graph, &k.node_supports, &k.field_inputs, &writes)
    }

    fn expect_ball(s: &Support) {
        match s {
            Support::Ball { center, radius } => {
                assert_eq!(center, &ball().center);
                assert_eq!(radius, &ball().radius);
            }
            other => panic!("expected the tagged ball, got {other:?}"),
        }
    }

    #[test]
    fn pure_delta_chain_inherits_the_ball() {
        // the drain-delta shape: (1 - exp(-chi*k)) * den * const — zero outside
        // the mask through the 1-exp lemma and multiplicative transparency.
        let s = derive(|chi| {
            let den = Gv::field("den2", FieldRef::cons_den());
            let g = Gv::ONE - (-(chi * Gv::scalar("k"))).exp();
            let delta = g * den * Gv::from_f64(0.5);
            vec![("d".into(), FieldRef::Scratch.into(), delta.node())]
        });
        expect_ball(&s);
    }

    #[test]
    fn in_place_write_unchanged_outside_the_ball() {
        // the cons-update shape: den_out = den * exp(-chi*k) + zero-supported
        // correction — equals the field's own read outside the mask.
        let s = derive(|chi| {
            let den = Gv::field("den2", FieldRef::cons_den());
            let f_rho = (-(chi * Gv::scalar("k"))).exp();
            let corr = chi * Gv::scalar("dt");
            let out = den * f_rho + corr;
            vec![("den_out".into(), FieldRef::cons_den().into(), out.node())]
        });
        expect_ball(&s);
    }

    #[test]
    fn absorbed_difference_is_ball_supported() {
        // delta = field - field*factor: the hash-consed field read makes the
        // subtraction structurally x - Eq(x, ball) -> Zero(ball).
        let s = derive(|chi| {
            let den = Gv::field("den2", FieldRef::cons_den());
            let f_rho = (-(chi * Gv::scalar("k"))).exp();
            let absorbed = den - den * f_rho;
            vec![("d".into(), FieldRef::Scratch.into(), absorbed.node())]
        });
        expect_ball(&s);
    }

    #[test]
    fn growth_cap_min_keeps_the_ball() {
        // the torque-free retention floor: min(exp(chi*k), cap>=1) stays one
        // outside the mask, so 1 - min(..) stays ball-supported.
        let s = derive(|chi| {
            let den = Gv::field("den2", FieldRef::cons_den());
            let b_t = (-(chi * Gv::scalar("k"))).exp().min(Gv::from_f64(1.0e12));
            let g_t = Gv::ONE - b_t;
            vec![("d".into(), FieldRef::Scratch.into(), (g_t * den).node())]
        });
        expect_ball(&s);
    }

    #[test]
    fn unmasked_contribution_widens_to_everywhere_fail_safe() {
        // a write mixing a masked term with a bare field term has no ball —
        // the derivation must refuse; keeping a stale narrow region would be unsound.
        let s = derive(|chi| {
            let den = Gv::field("den2", FieldRef::cons_den());
            let masked = chi * Gv::scalar("dt");
            let leak = den * Gv::from_f64(2.0);
            vec![("d".into(), FieldRef::Scratch.into(), (masked + leak).node())]
        });
        assert_eq!(s, Support::Everywhere);
    }

    #[test]
    fn in_place_write_against_a_different_field_widens() {
        // out bound to cons.nrg but equal to the DEN read outside the ball:
        // that is a genuine change to nrg everywhere — Everywhere.
        let s = derive(|chi| {
            let den = Gv::field("den2", FieldRef::cons_den());
            let out = den * (-(chi * Gv::scalar("k"))).exp();
            vec![("nrg_out".into(), FieldRef::cons_nrg().into(), out.node())]
        });
        assert_eq!(s, Support::Everywhere);
    }

    #[test]
    fn no_tags_derives_everywhere() {
        begin_trace();
        let den = Gv::field("den", FieldRef::cons_den());
        let writes: Writes = vec![("d".into(), FieldRef::Scratch.into(), (den * den).node())];
        let k = end_trace();
        let s = derive_output_support(&k.graph, &k.node_supports, &k.field_inputs, &writes);
        assert_eq!(s, Support::Everywhere);
    }
}
