// =============================================================================
// splice.rs
//
// graph splicing primitive. inserts a source Graph into a target Graph,
// binding each source param to a NodeId already present in target.
// the spliced subgraph is re-validated through target's public builders,
// so any shape error introduced by the substitution surfaces on target's
// error accumulator.
//
// enables regime-trait-method dispatch (R.6.f): a registered tensor IR
// fragment for `IsoNewtonian::to_primitive` is spliced into a kernel
// graph at the call site, with the fragment's params bound to the
// caller's gather() output members.
//
// usage:
//  let mut subst = HashMap::new();
//  subst.insert(Symbol::intern("cons_den"), den_nid);
//  subst.insert(Symbol::intern("cons_mom_0"), mom0_nid);
//  let outs = splice_graph(&mut target, &source, &[rho_out, vel_out], &subst)?;
//
// splice walks source once per call, so multiple outputs sharing common
// subexpressions (e.g., struct-return elementals where rho, vel, pre all
// depend on inv_rho) preserve common-subexpression sharing rather than
// duplicating intermediate ops.
// =============================================================================

use std::collections::HashMap;

use crate::einsum::{Atom, EinsumSpec};
use crate::graph::{Graph, NodeId, Op};
use crate::symbol::Symbol;

/// reason a splice could not be carried out structurally. shape errors
/// arising from the substitution itself accumulate on target via the
/// normal builder error path; splicing returns Ok in that case and the
/// caller drains target's errors as usual.
#[derive(Clone, Debug, PartialEq)]
pub enum SpliceError {
    /// source has `Op::Param(name)` but `param_subst` does not bind it.
    UnmappedParam {
        name: Symbol,
    },
    /// substituted NodeId has an element type or shape that does not
    /// match the source param's declared type. variance and detclass
    /// may differ.
    ParamTypeMismatch {
        name:           Symbol,
        expected_shape: Vec<crate::DimExpr>,
        found_shape:    Vec<crate::DimExpr>,
        expected_elem:  crate::ElementTy,
        found_elem:     crate::ElementTy,
    },
    /// source_output is not a valid NodeId for source.
    OutputOutOfRange,
}

impl std::fmt::Display for SpliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpliceError::UnmappedParam { name } => write!(
                f,
                "splice: source param `{}` has no binding in param_subst",
                name.as_str(),
            ),
            SpliceError::ParamTypeMismatch {
                name, expected_shape, found_shape, expected_elem, found_elem,
            } => write!(
                f,
                "splice: param `{}` type mismatch (source expects {:?}{:?}, target has {:?}{:?})",
                name.as_str(), expected_elem, expected_shape, found_elem, found_shape,
            ),
            SpliceError::OutputOutOfRange => write!(
                f, "splice: source_output is not a valid NodeId for source",
            ),
        }
    }
}

/// splice `source` into `target`. for each `Op::Param(sym)` in source,
/// `param_subst[sym]` provides the target-side NodeId to remap to. all
/// other ops are re-added to target via the public builders so that
/// shape inference re-runs against the substituted NodeIds.
///
/// returns the target NodeIds for each requested `source_outputs`,
/// preserving order. source is walked exactly once regardless of the
/// number of outputs — common subexpressions are inserted once and
/// shared across outputs.
pub fn splice_graph(
    target:         &mut Graph,
    source:         &Graph,
    source_outputs: &[NodeId],
    param_subst:    &HashMap<Symbol, NodeId>,
) -> Result<Vec<NodeId>, SpliceError> {
    for &out in source_outputs {
        if (out.0 as usize) >= source.len() {
            return Err(SpliceError::OutputOutOfRange);
        }
    }

    let mut remap: Vec<Option<NodeId>> = vec![None; source.len()];

    for (src_id, node, src_ty) in source.iter() {
        // Param + Lambda need cross-graph context (param substitution / FnDef
        // cloning) — handle them specially. every OTHER variant is "pure":
        // clone the op, remap its NodeId fields ONCE via the canonical
        // `Op::try_map_inputs`, then dispatch to the matching builder. the
        // remap+dispatch is now read like the IR shape, not duplicated per
        // variant (Phase 3).
        let new_id = match &node.op {
            Op::Param(sym) => {
                let target_id = param_subst.get(sym).copied().ok_or_else(|| {
                    SpliceError::UnmappedParam { name: sym.clone() }
                })?;
                let target_ty = target.ty(target_id);
                if target_ty.element != src_ty.element || target_ty.shape != src_ty.shape {
                    return Err(SpliceError::ParamTypeMismatch {
                        name:           sym.clone(),
                        expected_shape: src_ty.shape.clone(),
                        found_shape:    target_ty.shape.clone(),
                        expected_elem:  src_ty.element,
                        found_elem:     target_ty.element,
                    });
                }
                target_id
            }
            // F2.C: Lambda — clone the FnDef into target. the FnDef's body
            // lives in its own sub-graph; we copy it verbatim.
            Op::Lambda(fn_id) => {
                let fn_def = source.fn_defs()[fn_id.0 as usize].clone();
                target.add_lambda(fn_def, node.span)
            }
            _ => {
                // SINGLE remap call: `try_map_inputs` knows which
                // fields are NodeIds for each variant. lookup forwards into
                // `remap[..]`; a missing entry is an `OutputOutOfRange` error.
                let mut op = node.op.clone();
                op.try_map_inputs(|id| {
                    remap[id.0 as usize].ok_or(SpliceError::OutputOutOfRange)
                })?;
                // Phase 3 (continued): dispatch via `Op::dispatch_builder` —
                // the SINGLE per-variant Op->target-builder dispatcher in
                // graph.rs (alongside `try_map_inputs`). adding a new variant
                // touches `try_map_inputs` + `dispatch_builder` and nothing
                // else; this site stays variant-free. Param/Lambda are
                // handled in the explicit arms above (cross-graph context).
                op.dispatch_builder(target, node.span)
            }
        };
        remap[src_id.0 as usize] = Some(new_id);
    }

    // every visited source node now has a remap entry; outputs were
    // bounds-checked above so each lookup is populated.
    Ok(source_outputs.iter()
        .map(|out| remap[out.0 as usize].expect("output must be reachable in NodeId order"))
        .collect())
}

// `remap_one` / `remap_inputs` are gone — Phase 3's `Op::try_map_inputs`
// applies the same remap closure to every NodeId field in one call. forward
// references remain impossible (graph is bottom-up, we walk in NodeId order)
// so a missing entry still surfaces as `OutputOutOfRange` from the closure.

// reconstruct an einsum spec string from a parsed EinsumSpec. the
// grammar (einsum.rs) is closed: each atom is a single label char or
// "...", input atom-lists are comma-separated, output follows "->".
//
// `pub(crate)` so `Op::dispatch_builder` (the canonical Op->target-builder
// dispatcher in graph.rs) can rebuild the spec string when re-inserting
// `Op::Einsum` nodes — the public `Graph::einsum` builder reparses from
// the string form.
pub(crate) fn einsum_spec_to_string(spec: &EinsumSpec) -> String {
    let mut out = String::new();
    for (i, atoms) in spec.inputs.iter().enumerate() {
        if i > 0 { out.push(','); }
        append_atoms(&mut out, atoms);
    }
    out.push_str("->");
    append_atoms(&mut out, &spec.output);
    out
}

fn append_atoms(buf: &mut String, atoms: &[Atom]) {
    for a in atoms {
        match a {
            Atom::Label(c) => buf.push(*c),
            Atom::Ellipsis => buf.push_str("..."),
        }
    }
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConstValue, DimExpr, ElementTy, ElementWiseOp, Symbol, TensorTy,
    };

    fn lit(n: usize) -> DimExpr { DimExpr::Literal(n) }

    // build a 3-node source: param `x` (f64 scalar), const 2.0, x * 2.0.
    fn source_x_times_two() -> (Graph, NodeId) {
        let mut g = Graph::new();
        let x = g.add_scalar_param("x", ElementTy::F64);
        let c = g.add_const(ConstValue::F64(2.0), None);
        let r = g.element_wise(ElementWiseOp::Mul, vec![x, c], None);
        g.set_output(r);
        assert!(!g.has_errors(), "source build errors: {:?}", g.errors());
        (g, r)
    }

    #[test]
    fn splice_inlines_into_target_and_resolves_param() {
        let (src, src_out) = source_x_times_two();

        let mut target = Graph::new();
        let a = target.add_scalar_param("a", ElementTy::F64);
        let b = target.add_const(ConstValue::F64(3.0), None);
        let sum = target.element_wise(ElementWiseOp::Add, vec![a, b], None);

        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), sum);

        let results = splice_graph(&mut target, &src, &[src_out], &subst).unwrap();
        let result = results[0];

        assert!(!target.has_errors(), "target errors: {:?}", target.errors());
        // target now contains a, b, sum, (cloned const 2.0), result (mul).
        // result must depend on sum (the substituted node), not on a fresh param.
        let result_node = target.node(result);
        match &result_node.op {
            Op::ElementWise(ElementWiseOp::Mul, inputs) => {
                assert_eq!(inputs.len(), 2);
                assert_eq!(inputs[0], sum, "mul lhs should resolve to substituted NodeId");
                // rhs is the cloned const 2.0; verify by op shape.
                match &target.node(inputs[1]).op {
                    Op::Const(ConstValue::F64(v)) if (*v - 2.0).abs() < 1e-12 => {}
                    other => panic!("expected cloned Const(2.0), got {:?}", other),
                }
            }
            other => panic!("expected ElementWise(Mul), got {:?}", other),
        }
    }

    #[test]
    fn splice_does_not_register_source_params_on_target() {
        let (src, src_out) = source_x_times_two();
        let mut target = Graph::new();
        let a = target.add_scalar_param("a", ElementTy::F64);
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), a);
        let _ = splice_graph(&mut target, &src, &[src_out], &subst).unwrap();
        // target.params() should still only have `a`; the source's `x`
        // must not leak in.
        let names: Vec<&str> = target.params().iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(names, vec!["a"]);
    }

    #[test]
    fn splice_unmapped_param_returns_error() {
        let (src, src_out) = source_x_times_two();
        let mut target = Graph::new();
        let subst: HashMap<Symbol, NodeId> = HashMap::new();
        let err = splice_graph(&mut target, &src, &[src_out], &subst).unwrap_err();
        match err {
            SpliceError::UnmappedParam { name } => assert_eq!(name.as_str(), "x"),
            other => panic!("expected UnmappedParam, got {:?}", other),
        }
    }

    #[test]
    fn splice_param_shape_mismatch_returns_error() {
        let (src, src_out) = source_x_times_two();  // source x is scalar
        let mut target = Graph::new();
        // bind x to a rank-1 tensor; element matches, shape doesn't.
        let v = target.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), v);
        let err = splice_graph(&mut target, &src, &[src_out], &subst).unwrap_err();
        match err {
            SpliceError::ParamTypeMismatch { name, expected_shape, found_shape, .. } => {
                assert_eq!(name.as_str(), "x");
                assert_eq!(expected_shape, Vec::<DimExpr>::new());
                assert_eq!(found_shape, vec![lit(3)]);
            }
            other => panic!("expected ParamTypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn splice_param_element_mismatch_returns_error() {
        let (src, src_out) = source_x_times_two();  // source x is F64
        let mut target = Graph::new();
        let i = target.add_scalar_param("i", ElementTy::I32);
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), i);
        let err = splice_graph(&mut target, &src, &[src_out], &subst).unwrap_err();
        assert!(matches!(err, SpliceError::ParamTypeMismatch { .. }));
    }

    #[test]
    fn splice_output_out_of_range_returns_error() {
        let (src, _) = source_x_times_two();
        let mut target = Graph::new();
        let subst: HashMap<Symbol, NodeId> = HashMap::new();
        let err = splice_graph(&mut target, &src, &[NodeId(999)], &subst).unwrap_err();
        assert_eq!(err, SpliceError::OutputOutOfRange);
    }

    #[test]
    fn splice_einsum_node_roundtrips_spec() {
        // source: dot product v.w where v,w are rank-1.
        let mut src = Graph::new();
        let v = src.add_param(
            Symbol::intern("v"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let w = src.add_param(
            Symbol::intern("w"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let dot = src.einsum("i,i->", vec![v, w], None);
        src.set_output(dot);
        assert!(!src.has_errors(), "src errors: {:?}", src.errors());

        let mut target = Graph::new();
        let a = target.add_param(
            Symbol::intern("a"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let b = target.add_param(
            Symbol::intern("b"),
            TensorTy::from_shape(ElementTy::F64, vec![lit(3)]),
            None,
        );
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("v"), a);
        subst.insert(Symbol::intern("w"), b);

        let result = splice_graph(&mut target, &src, &[dot], &subst).unwrap()[0];
        assert!(!target.has_errors(), "target errors: {:?}", target.errors());
        assert_eq!(target.ty(result).rank, 0);
    }

    #[test]
    fn splice_chains_multiple_nodes_correctly() {
        // source: ((x + y) * z) — three params, two element-wise ops.
        let mut src = Graph::new();
        let x = src.add_scalar_param("x", ElementTy::F64);
        let y = src.add_scalar_param("y", ElementTy::F64);
        let z = src.add_scalar_param("z", ElementTy::F64);
        let s = src.element_wise(ElementWiseOp::Add, vec![x, y], None);
        let m = src.element_wise(ElementWiseOp::Mul, vec![s, z], None);
        src.set_output(m);
        assert!(!src.has_errors());

        let mut target = Graph::new();
        let p = target.add_scalar_param("p", ElementTy::F64);
        let q = target.add_scalar_param("q", ElementTy::F64);
        let r = target.add_scalar_param("r", ElementTy::F64);
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), p);
        subst.insert(Symbol::intern("y"), q);
        subst.insert(Symbol::intern("z"), r);

        let result = splice_graph(&mut target, &src, &[m], &subst).unwrap()[0];
        assert!(!target.has_errors(), "target errors: {:?}", target.errors());

        // result must be Mul(Add(p, q), r) structurally.
        let result_node = target.node(result);
        let (add_id, z_in) = match &result_node.op {
            Op::ElementWise(ElementWiseOp::Mul, inputs) => (inputs[0], inputs[1]),
            other => panic!("expected Mul, got {:?}", other),
        };
        assert_eq!(z_in, r);
        match &target.node(add_id).op {
            Op::ElementWise(ElementWiseOp::Add, inputs) => {
                assert_eq!(inputs[0], p);
                assert_eq!(inputs[1], q);
            }
            other => panic!("expected Add, got {:?}", other),
        }
    }

    #[test]
    fn splice_multi_output_shares_intermediate_nodes() {
        // source: inv = 1.0 / x; (a = inv * y, b = inv * z). two outputs
        // share `inv`. splicing both in one call must produce ONE inv
        // node in target, not two.
        let mut src = Graph::new();
        let x = src.add_scalar_param("x", ElementTy::F64);
        let y = src.add_scalar_param("y", ElementTy::F64);
        let z = src.add_scalar_param("z", ElementTy::F64);
        let one = src.add_const(ConstValue::F64(1.0), None);
        let inv = src.element_wise(ElementWiseOp::Div, vec![one, x], None);
        let a = src.element_wise(ElementWiseOp::Mul, vec![inv, y], None);
        let b = src.element_wise(ElementWiseOp::Mul, vec![inv, z], None);
        assert!(!src.has_errors());

        let mut target = Graph::new();
        let tx = target.add_scalar_param("tx", ElementTy::F64);
        let ty = target.add_scalar_param("ty", ElementTy::F64);
        let tz = target.add_scalar_param("tz", ElementTy::F64);
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), tx);
        subst.insert(Symbol::intern("y"), ty);
        subst.insert(Symbol::intern("z"), tz);

        let outs = splice_graph(&mut target, &src, &[a, b], &subst).unwrap();
        assert_eq!(outs.len(), 2);
        assert!(!target.has_errors());

        // both spliced outputs are Mul; their lhs (the inv node) must be
        // the same NodeId, proving the CSE-friendly single-walk property.
        let (a_inv, b_inv) = match (&target.node(outs[0]).op, &target.node(outs[1]).op) {
            (Op::ElementWise(ElementWiseOp::Mul, ai),
             Op::ElementWise(ElementWiseOp::Mul, bi)) => (ai[0], bi[0]),
            other => panic!("expected two Mul ops, got {:?}", other),
        };
        assert_eq!(a_inv, b_inv,
            "splice_graph must share the inv node between both outputs");

        // and exactly one Div node was inserted into target.
        let div_count = target.iter()
            .filter(|(_, n, _)| matches!(n.op, Op::ElementWise(ElementWiseOp::Div, _)))
            .count();
        assert_eq!(div_count, 1, "expected 1 Div in target, got {}", div_count);
    }

    #[test]
    fn splice_multi_output_preserves_ordering() {
        let (src, src_out) = source_x_times_two();
        let mut target = Graph::new();
        let a = target.add_scalar_param("a", ElementTy::F64);
        let mut subst = HashMap::new();
        subst.insert(Symbol::intern("x"), a);

        let outs = splice_graph(&mut target, &src, &[src_out, src_out, src_out], &subst).unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0], outs[1]);
        assert_eq!(outs[1], outs[2]);
    }

    #[test]
    fn spec_to_string_roundtrips_common_specs() {
        for s in ["i,i->", "ij,jk->ik", "...i,...i->...", "ij->ji", "i,j->ij"] {
            let parsed = crate::parse_einsum_spec(s).unwrap();
            let printed = einsum_spec_to_string(&parsed);
            let reparsed = crate::parse_einsum_spec(&printed).unwrap();
            assert_eq!(parsed, reparsed, "roundtrip failed for `{}` -> `{}`", s, printed);
        }
    }
}
