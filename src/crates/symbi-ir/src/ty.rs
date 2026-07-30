// =============================================================================
// ty.rs
//
// TensorTy: the fully-typed tensor type carried by every node in the
// tensor IR graph. covers element type, rank, and shape (per-axis DimExpr).
//
// scalars are rank-0 tensors (empty shape). this is what makes the
// "scalars are rank-0 tensors anyway" framing real at the IR level.
// =============================================================================

use crate::{DimExpr, ElementTy};

/// fully-typed tensor type. one of these is attached to every node in
/// a tensor IR graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorTy {
    pub element: ElementTy,
    pub rank: u32,
    pub shape: Vec<DimExpr>,
}

impl TensorTy {
    /// build a rank-0 (scalar) tensor type.
    pub fn scalar(element: ElementTy) -> Self {
        TensorTy {
            element,
            rank: 0,
            shape: vec![],
        }
    }

    /// build a rank-N tensor type from a shape.
    pub fn from_shape(element: ElementTy, shape: Vec<DimExpr>) -> Self {
        let rank = shape.len() as u32;
        TensorTy {
            element,
            rank,
            shape,
        }
    }

    /// is this a rank-0 (scalar) type?
    pub fn is_scalar(&self) -> bool {
        self.rank == 0
    }
}

impl std::fmt::Display for TensorTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Tensor<element, [d0, d1, ...]>
        write!(f, "Tensor<{}, [", self.element)?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(n: usize) -> DimExpr {
        DimExpr::Literal(n)
    }

    #[test]
    fn scalar_constructor() {
        let t = TensorTy::scalar(ElementTy::F64);
        assert_eq!(t.element, ElementTy::F64);
        assert_eq!(t.rank, 0);
        assert!(t.shape.is_empty());
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
    fn equality_is_structural() {
        let a = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        let b = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        assert_eq!(a, b);
    }

    #[test]
    fn equality_distinguishes_literal_dims() {
        let a = TensorTy::from_shape(ElementTy::F64, vec![lit(3)]);
        let b = TensorTy::from_shape(ElementTy::F64, vec![lit(4)]);
        assert_ne!(a, b);
    }

    // ---- display ----

    #[test]
    fn display_contains_element_and_rank() {
        let t = TensorTy::from_shape(ElementTy::F64, vec![lit(3), lit(5)]);
        let s = format!("{}", t);
        assert!(s.contains("f64"), "{}", s);
        assert!(s.contains("3"), "{}", s);
        assert!(s.contains("5"), "{}", s);
    }
}
