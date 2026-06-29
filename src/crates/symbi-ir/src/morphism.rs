// =============================================================================
// morphism.rs
//
// physics-aware morphisms in the tensor IR. each MorphismKind is an
// algebraic operation on Field-valued IR nodes that carries:
//   - a dimensionality constraint (DimConstraint) — the set of ndims at
//     which the morphism is meaningful
//   - an arity / structure (encoded in the variant + Vec<NodeId> args)
//   - lowering hooks to CPU + CUDA backends
//
// F4 introduces the morphism layer so that ndim restrictions live in the
// IR graph as facts the proc-macro reads, not as naming conventions
// (legacy `_3d` suffix) or per-author annotations. the proc-macro's per-
// ndim enumeration becomes:
//
//   for each ndim in 1..=3:
//       if graph_satisfies_at(graph, ndim):
//           emit CUDA for that ndim
//       else:
//           skip (no error — just no PTX at that ndim)
//
// where `graph_satisfies_at` walks the IR and checks that every Morphism
// node's DimConstraint admits this ndim.
//
// =============================================================================

/// dimensionality constraint a morphism imposes on the enclosing graph.
///
/// when an IR graph contains multiple morphisms, the effective constraint
/// is the intersection of all of them — the strictest one wins. e.g., a
/// graph with both `Curl { Exactly(3) }` and `Diff { AnyD }` is `Exactly(3)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimConstraint {
    /// the morphism makes sense at any ndim ≥ 1 (e.g., Diff, FaceAvg).
    AnyD,
    /// the morphism makes sense only at exactly this ndim
    /// (e.g., Curl is ThreeD-only; ε^{ijk} is rank-3).
    Exactly(u8),
    /// the morphism makes sense at this ndim or higher.
    /// reserved for future morphisms (e.g., Determinant<N> requires D ≥ N).
    AtLeast(u8),
}

impl DimConstraint {
    /// the strictest of two constraints. returns None when they are
    /// mutually unsatisfiable (e.g., `Exactly(2)` ∩ `Exactly(3)` = None).
    pub fn intersect(self, other: Self) -> Option<Self> {
        use DimConstraint::*;
        match (self, other) {
            (AnyD, c) | (c, AnyD) => Some(c),
            (Exactly(a), Exactly(b)) if a == b => Some(Exactly(a)),
            (Exactly(_), Exactly(_)) => None,
            (Exactly(n), AtLeast(m)) | (AtLeast(m), Exactly(n)) if n >= m => Some(Exactly(n)),
            (Exactly(_), AtLeast(_)) | (AtLeast(_), Exactly(_)) => None,
            (AtLeast(a), AtLeast(b)) => Some(AtLeast(a.max(b))),
        }
    }

    /// whether the given ndim satisfies this constraint.
    pub fn admits(&self, ndim: u8) -> bool {
        match self {
            Self::AnyD => ndim >= 1,
            Self::Exactly(n) => ndim == *n,
            Self::AtLeast(n) => ndim >= *n,
        }
    }
}

/// algebraic morphisms on the tensor IR. each variant denotes a specific
/// physics-aware operation; the proc-macro and IR lowering pipelines treat
/// these as IR primitives with known semantics (vs OpaqueCall, where the
/// semantics live in an external function body).
///
/// the variant choice and its `axis` payload determine both the stencil
/// pattern at lowering time and the DimConstraint for ndim inference.
///
/// kept axis-erased at the variant level (no `Diff<const AX>`) so the
/// hash-cons keys are values, not types. axis lives in the payload.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MorphismKind {
    /// `Diff<axis>(f) at coord = f[coord + e_axis] - f[coord]`. AnyD.
    /// args: [field_node]
    Diff { axis: u8 },

    /// `FaceAvg<axis>(f) at coord = 0.5 * (f[coord] + f[coord + e_axis])`. AnyD.
    /// args: [field_node]
    FaceAvg { axis: u8 },

    // F5.4-retire: `Curl { face_axis }` and `CtEdgeEmf { edge_axis }`
    // were RMHD-physics-specific morphisms that violated the SRP cut
    // between the math IR and physics. their dispatchers in
    // `symbi-macros::ir_builder` now emit the stencil inline as
    // ElementWise + OpaqueCall nodes, and the ndim constraint is
    // carried by `Graph::pin_ndim` instead of via these morphism kinds.
}

impl MorphismKind {
    /// dimensionality constraint this morphism imposes.
    pub fn dim_constraint(&self) -> DimConstraint {
        match self {
            Self::Diff { .. } => DimConstraint::AnyD,
            Self::FaceAvg { .. } => DimConstraint::AnyD,
        }
    }

    /// expected number of NodeId args. used by graph validation.
    pub fn arity(&self, _ndim: u8) -> usize {
        match self {
            Self::Diff { .. } => 1,
            Self::FaceAvg { .. } => 1,
        }
    }

    /// short debugging label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Diff { .. } => "Diff",
            Self::FaceAvg { .. } => "FaceAvg",
        }
    }
}

/// F4.0d: walk a tensor IR graph, intersect every Op::Morphism node's
/// DimConstraint, and return the resulting graph-level constraint.
///
/// returns:
///   - `Some(DimConstraint)` — at least one ndim satisfies every Morphism
///     in the graph; `.admits(n)` answers per-n queries.
///   - `Some(DimConstraint::AnyD)` — the graph contains no Morphism nodes
///     (or only AnyD morphisms); no restriction at the algebra level.
///     legacy `_nd` suffix and explicit `ndim = N` hints still apply.
///   - `None` — the graph contains two morphisms with mutually
///     unsatisfiable constraints (e.g., `Exactly(2)` ∩ `Exactly(3)`). this
///     is a graph-composition error; the proc-macro should surface it
///     at expansion time rather than silently emit no PTX.
///
/// the algebra `intersect` is associative and commutative, so walking
/// the graph in any order produces the same result. node visit order is
/// the natural NodeId order from `Graph::iter`.
pub fn graph_dim_constraint(graph: &crate::graph::Graph) -> Option<DimConstraint> {
    use crate::graph::Op;
    let mut constraint = DimConstraint::AnyD;
    for (_id, node, _ty) in graph.iter() {
        if let Op::Morphism { kind, .. } = &node.op {
            constraint = constraint.intersect(kind.dim_constraint())?;
        }
    }
    // F5.4-retire: graph-level pin from `Graph::pin_ndim` (e.g., set by
    // the macro dispatcher for kernels whose physics is ndim-restricted
    // beyond what the morphism kinds alone express).
    if let Some(n) = graph.pinned_ndim() {
        constraint = constraint.intersect(DimConstraint::Exactly(n))?;
    }
    Some(constraint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anyd_intersect_anyd() {
        assert_eq!(DimConstraint::AnyD.intersect(DimConstraint::AnyD), Some(DimConstraint::AnyD));
    }

    #[test]
    fn anyd_intersect_exactly() {
        assert_eq!(
            DimConstraint::AnyD.intersect(DimConstraint::Exactly(3)),
            Some(DimConstraint::Exactly(3))
        );
        assert_eq!(
            DimConstraint::Exactly(3).intersect(DimConstraint::AnyD),
            Some(DimConstraint::Exactly(3))
        );
    }

    #[test]
    fn exactly_clash() {
        assert_eq!(DimConstraint::Exactly(2).intersect(DimConstraint::Exactly(3)), None);
    }

    #[test]
    fn exactly_and_atleast() {
        // Exactly(3) ∩ AtLeast(2) = Exactly(3)
        assert_eq!(
            DimConstraint::Exactly(3).intersect(DimConstraint::AtLeast(2)),
            Some(DimConstraint::Exactly(3))
        );
        // Exactly(2) ∩ AtLeast(3) = None
        assert_eq!(DimConstraint::Exactly(2).intersect(DimConstraint::AtLeast(3)), None);
    }

    #[test]
    fn atleast_combines() {
        assert_eq!(
            DimConstraint::AtLeast(2).intersect(DimConstraint::AtLeast(3)),
            Some(DimConstraint::AtLeast(3))
        );
    }

    #[test]
    fn admits_ndim() {
        assert!(DimConstraint::AnyD.admits(1));
        assert!(DimConstraint::AnyD.admits(3));
        assert!(DimConstraint::Exactly(3).admits(3));
        assert!(!DimConstraint::Exactly(3).admits(2));
        assert!(DimConstraint::AtLeast(2).admits(2));
        assert!(DimConstraint::AtLeast(2).admits(3));
        assert!(!DimConstraint::AtLeast(2).admits(1));
    }

    #[test]
    fn morphism_constraints() {
        // F5.4-retire: Curl + CtEdgeEmf removed. only Diff + FaceAvg
        // remain, both AnyD.
        assert_eq!(MorphismKind::Diff { axis: 0 }.dim_constraint(), DimConstraint::AnyD);
        assert_eq!(MorphismKind::FaceAvg { axis: 1 }.dim_constraint(), DimConstraint::AnyD);
    }

    #[test]
    fn morphism_arity() {
        // Diff and FaceAvg take exactly one field, regardless of ndim.
        assert_eq!(MorphismKind::Diff { axis: 0 }.arity(1), 1);
        assert_eq!(MorphismKind::Diff { axis: 0 }.arity(3), 1);
    }

    // graph_dim_constraint inference: uses direct Graph construction
    // (bypassing the proc-macro) to exercise the intersection algebra
    // at the graph level.
    use crate::graph::Graph;
    use crate::element::ElementTy;
    use crate::symbol::Symbol;
    use crate::ty::TensorTy;

    fn fresh_graph_with_param() -> (Graph, crate::graph::NodeId) {
        let mut g = Graph::new();
        let p = g.add_param(
            Symbol::intern("field"),
            TensorTy::scalar(ElementTy::F64),
            None,
        );
        (g, p)
    }

    #[test]
    fn empty_graph_admits_anyd() {
        let g = Graph::new();
        assert_eq!(graph_dim_constraint(&g), Some(DimConstraint::AnyD));
    }

    #[test]
    fn graph_with_only_diff_is_anyd() {
        let (mut g, p) = fresh_graph_with_param();
        let _ = g.morphism(MorphismKind::Diff { axis: 0 }, vec![p, p], None);
        assert_eq!(graph_dim_constraint(&g), Some(DimConstraint::AnyD));
    }

    #[test]
    fn graph_pin_ndim_carries_exact_constraint() {
        // F5.4-retire: kernels whose physics is ndim-restricted (e.g.
        // the CT-MHD curl stencil's cyclic (J=I+1, K=I+2) is rank-3)
        // now call `Graph::pin_ndim(3)` in the macro dispatcher instead
        // of emitting `MorphismKind::Curl`. the constraint surfaces in
        // `graph_dim_constraint` the same way.
        let (mut g, _) = fresh_graph_with_param();
        g.pin_ndim(3);
        assert_eq!(graph_dim_constraint(&g), Some(DimConstraint::Exactly(3)));
    }

    #[test]
    fn graph_pin_ndim_intersects_with_morphism_constraints() {
        // pin_ndim(3) ∩ Diff (AnyD) = Exactly(3). diff doesn't loosen
        // the pinned constraint.
        let (mut g, p) = fresh_graph_with_param();
        g.pin_ndim(3);
        let _ = g.morphism(MorphismKind::Diff { axis: 1 }, vec![p, p], None);
        assert_eq!(graph_dim_constraint(&g), Some(DimConstraint::Exactly(3)));
    }
}
