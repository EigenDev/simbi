// =============================================================================
// ty.rs
//
// TensorTy: the fully-typed tensor type carried by every node in the
// tensor IR graph. covers element type, rank, shape (per-axis DimExpr),
// variance, and determinism class.
//
// scalars are rank-0 tensors (empty shape). this is what makes the
// "scalars are rank-0 tensors anyway" framing real at the IR level.
//
// the Det/Tainted lattice carries through the type — every tensor carries a
// `DetClass` (re-exported alias of `ClassWitness`). transcendentals taint; pure
// IEEE-mandated ops preserve. propagation rules implemented in ops (R.2).
// =============================================================================

use crate::{ElementTy, DimExpr, VarianceTag};
use crate::class::ClassWitness;

/// alias for the class witness; tensors carry the
/// same Det/Tainted lattice. one name in the spec; one type in the
/// implementation.
pub type DetClass = ClassWitness;

/// fully-typed tensor type. one of these is attached to every node in
/// a tensor IR graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorTy {
    pub element:  ElementTy,
    pub rank:     u32,
    pub shape:    Vec<DimExpr>,
    pub variance: VarianceTag,
    pub class:    DetClass,
}

impl TensorTy {
    /// build a rank-0 (scalar) tensor type. defaults to Untagged
    /// variance and Det class — both are the conservative starting
    /// points for literals and bit-equal parameters.
    pub fn scalar(element: ElementTy) -> Self {
        TensorTy {
            element,
            rank: 0,
            shape: vec![],
            variance: VarianceTag::Untagged,
            class: DetClass::Det,
        }
    }

    /// build a rank-N tensor type from a shape. variance and class
    /// default to Untagged / Det; callers override as needed.
    pub fn from_shape(element: ElementTy, shape: Vec<DimExpr>) -> Self {
        let rank = shape.len() as u32;
        TensorTy {
            element,
            rank,
            shape,
            variance: VarianceTag::Untagged,
            class: DetClass::Det,
        }
    }

    /// builder-style variance setter.
    pub fn with_variance(mut self, v: VarianceTag) -> Self {
        self.variance = v;
        self
    }

    /// builder-style class setter.
    pub fn with_class(mut self, c: DetClass) -> Self {
        self.class = c;
        self
    }

    /// is this a rank-0 (scalar) type?
    pub fn is_scalar(&self) -> bool {
        self.rank == 0
    }
}

/// join the determinism class of two inputs. `Det` joined with `Det`
/// stays `Det`; any `Tainted` operand produces a `Tainted` result.
/// re-export of the existing ClassWitness::join via a tensor-friendly name.
pub const fn join_class(a: DetClass, b: DetClass) -> DetClass {
    a.join(b)
}

impl std::fmt::Display for TensorTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Tensor<element, [d0, d1, …], variance, class>
        write!(f, "Tensor<{}, [", self.element)?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 { f.write_str(", ")?; }
            write!(f, "{}", d)?;
        }
        write!(f, "], {:?}, {}>", self.variance, self.class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::ClassWitness;

    fn lit(n: usize) -> DimExpr { DimExpr::Literal(n) }
    fn g(s: &str) -> DimExpr { DimExpr::generic(s) }

    #[test]
    fn scalar_constructor() {
        let t = TensorTy::scalar(ElementTy::F64);
        assert_eq!(t.element, ElementTy::F64);
        assert_eq!(t.rank, 0);
        assert!(t.shape.is_empty());
        assert_eq!(t.variance, VarianceTag::Untagged);
        assert_eq!(t.class, DetClass::Det);
        assert!(t.is_scalar());
    }

    #[test]
    fn from_shape_sets_rank_correctly() {
        let t = TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(4)]);
        assert_eq!(t.rank, 2);
        assert_eq!(t.shape, vec![lit(3), lit(4)]);
        assert!(!t.is_scalar());
    }

    #[test]
    fn from_shape_empty_is_scalar_shaped() {
        let t = TensorTy::from_shape(ElementTy::F32, vec![]);
        assert_eq!(t.rank, 0);
        assert!(t.is_scalar());
    }

    #[test]
    fn builder_setters_chain() {
        let t = TensorTy::from_shape(ElementTy::F64, vec![lit(3)])
            .with_variance(VarianceTag::Upper)
            .with_class(DetClass::Tainted);
        assert_eq!(t.variance, VarianceTag::Upper);
        assert_eq!(t.class, DetClass::Tainted);
    }

    #[test]
    fn equality_is_structural() {
        let a = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        let b = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        assert_eq!(a, b);
    }

    #[test]
    fn equality_distinguishes_variance() {
        let a = TensorTy::from_shape(ElementTy::F64, vec![lit(3)])
            .with_variance(VarianceTag::Upper);
        let b = TensorTy::from_shape(ElementTy::F64, vec![lit(3)])
            .with_variance(VarianceTag::Lower);
        assert_ne!(a, b);
    }

    #[test]
    fn equality_distinguishes_class() {
        let a = TensorTy::scalar(ElementTy::F64).with_class(DetClass::Det);
        let b = TensorTy::scalar(ElementTy::F64).with_class(DetClass::Tainted);
        assert_ne!(a, b);
    }

    #[test]
    fn equality_distinguishes_generic_dims() {
        let a = TensorTy::from_shape(ElementTy::F64, vec![g("D")]);
        let b = TensorTy::from_shape(ElementTy::F64, vec![g("N")]);
        assert_ne!(a, b);
    }

    // ---- class join lattice ----

    #[test]
    fn join_class_det_det_is_det() {
        assert_eq!(join_class(DetClass::Det, DetClass::Det), DetClass::Det);
    }

    #[test]
    fn join_class_with_tainted_is_tainted() {
        assert_eq!(join_class(DetClass::Det, DetClass::Tainted), DetClass::Tainted);
        assert_eq!(join_class(DetClass::Tainted, DetClass::Det), DetClass::Tainted);
        assert_eq!(join_class(DetClass::Tainted, DetClass::Tainted), DetClass::Tainted);
    }

    #[test]
    fn join_class_is_associative_and_commutative() {
        for a in [DetClass::Det, DetClass::Tainted] {
            for b in [DetClass::Det, DetClass::Tainted] {
                assert_eq!(join_class(a, b), join_class(b, a));
                for c in [DetClass::Det, DetClass::Tainted] {
                    let abc = join_class(join_class(a, b), c);
                    let a_bc = join_class(a, join_class(b, c));
                    assert_eq!(abc, a_bc);
                }
            }
        }
    }

    #[test]
    fn det_class_re_exports_class_witness() {
        // sanity that the alias is the same type as `ClassWitness`
        let _: DetClass = ClassWitness::Det;
        let _: ClassWitness = DetClass::Tainted;
    }

    // ---- display ----

    #[test]
    fn display_contains_element_and_rank() {
        let t = TensorTy::from_shape(ElementTy::F64, vec![lit(3), g("D")]);
        let s = format!("{}", t);
        assert!(s.contains("f64"), "{}", s);
        assert!(s.contains("3"), "{}", s);
        assert!(s.contains("D"), "{}", s);
    }
}
