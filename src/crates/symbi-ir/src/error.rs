// =============================================================================
// error.rs
//
// shape-error reporting for the tensor IR. follows the spec's § 12.4
// decision: a single ShapeError enum with rich variants, accumulated on
// the Graph and drained at the macro boundary. builders return NodeId
// unconditionally; on failure the error is pushed and the builder
// emits a "poison" node whose type is whatever makes downstream code
// fail gracefully (handled in graph.rs).
//
// each variant carries source spans where they're available. two-span
// variants (DimMismatch, VarianceMismatch, ElementMismatch) cite both
// conflicting nodes so diagnostics can point at both, per § 12.3.
// =============================================================================

use proc_macro2::Span;

use crate::{DimExpr, ElementTy, VarianceTag};

/// one shape-inference / IR-build error. variants carry the minimum
/// context needed to produce a diagnostic and (where applicable) one
/// or two source spans for the user-facing message.
#[derive(Clone, Debug)]
pub enum ShapeError {
    /// two dimensions that should have matched, don't. cites both
    /// occurrences so the diagnostic can point at each.
    DimMismatch {
        expected: DimExpr,
        found:    DimExpr,
        /// span of the node that introduced `expected`.
        span_a:   Option<Span>,
        /// span of the node that introduced `found`.
        span_b:   Option<Span>,
        /// short tag identifying the operation context (e.g. "einsum 'i'", ".dot").
        context:  String,
    },

    /// a tensor had the wrong rank for an op.
    RankMismatch {
        expected: u32,
        found:    u32,
        span:     Option<Span>,
        context:  String,
    },

    /// `.contract()` (or einsum contraction) saw operands whose variance
    /// doesn't pair (e.g. Upper paired with Upper, or anything paired
    /// with Untagged).
    VarianceMismatch {
        left:    VarianceTag,
        right:   VarianceTag,
        span_a:  Option<Span>,
        span_b:  Option<Span>,
        context: String,
    },

    /// element types disagree where the op requires equality.
    ElementMismatch {
        left:    ElementTy,
        right:   ElementTy,
        span_a:  Option<Span>,
        span_b:  Option<Span>,
        context: String,
    },

    /// shapes weren't broadcast-compatible (per § 1.5 broadcast rule).
    BroadcastIncompatible {
        left:    Vec<DimExpr>,
        right:   Vec<DimExpr>,
        span:    Option<Span>,
        context: String,
    },

    /// einsum spec exceeded the V1 max of 8 named labels.
    EinsumLabelOverLimit {
        count: usize,
        max:   usize,
        span:  Option<Span>,
    },

    /// einsum output references a label that doesn't appear in any input.
    EinsumOutputLabelNotInInputs {
        label: char,
        span:  Option<Span>,
    },

    /// einsum spec had more than one ellipsis on one side of the spec.
    EinsumMultipleEllipses {
        side: &'static str,        // "input" or "output"
        span: Option<Span>,
    },

    /// einsum input count didn't match the spec's input-spec count.
    EinsumInputArityMismatch {
        spec_inputs:    usize,
        actual_inputs:  usize,
        span:           Option<Span>,
    },

    /// einsum input rank doesn't match the input's labelled axes
    /// (after accounting for ellipsis).
    EinsumRankMismatch {
        input_index: usize,
        expected:    u32,
        found:       u32,
        span:        Option<Span>,
    },

    /// generic catch-all for builder preconditions (out-of-bounds index,
    /// empty Construct, etc.). use sparingly — prefer typed variants
    /// when the failure mode is fundamental.
    Other {
        message: String,
        span:    Option<Span>,
    },
}

impl ShapeError {
    /// the first span attached to this error (the "primary" location
    /// for IDE highlight), if any.
    pub fn primary_span(&self) -> Option<Span> {
        match self {
            ShapeError::DimMismatch { span_a, span_b, .. } => span_a.or(*span_b),
            ShapeError::RankMismatch { span, .. } => *span,
            ShapeError::VarianceMismatch { span_a, span_b, .. } => span_a.or(*span_b),
            ShapeError::ElementMismatch { span_a, span_b, .. } => span_a.or(*span_b),
            ShapeError::BroadcastIncompatible { span, .. } => *span,
            ShapeError::EinsumLabelOverLimit { span, .. } => *span,
            ShapeError::EinsumOutputLabelNotInInputs { span, .. } => *span,
            ShapeError::EinsumMultipleEllipses { span, .. } => *span,
            ShapeError::EinsumInputArityMismatch { span, .. } => *span,
            ShapeError::EinsumRankMismatch { span, .. } => *span,
            ShapeError::Other { span, .. } => *span,
        }
    }

    /// a one-line human-readable summary. used in fallback diagnostics
    /// and tests; the macro layer formats richer multi-line messages
    /// from the variant fields.
    pub fn summary(&self) -> String {
        match self {
            ShapeError::DimMismatch { expected, found, context, .. } => format!(
                "dim mismatch in {}: expected {}, found {}",
                context, expected, found,
            ),
            ShapeError::RankMismatch { expected, found, context, .. } => format!(
                "rank mismatch in {}: expected rank {}, found rank {}",
                context, expected, found,
            ),
            ShapeError::VarianceMismatch { left, right, context, .. } => format!(
                "variance mismatch in {}: {} paired with {}",
                context, left.label(), right.label(),
            ),
            ShapeError::ElementMismatch { left, right, context, .. } => format!(
                "element type mismatch in {}: {} vs {}",
                context, left, right,
            ),
            ShapeError::BroadcastIncompatible { left, right, context, .. } => format!(
                "incompatible broadcast in {}: [{}] vs [{}]",
                context,
                fmt_shape(left), fmt_shape(right),
            ),
            ShapeError::EinsumLabelOverLimit { count, max, .. } => format!(
                "einsum spec uses {} named labels, V1 limit is {}",
                count, max,
            ),
            ShapeError::EinsumOutputLabelNotInInputs { label, .. } => format!(
                "einsum output label '{}' does not appear in any input",
                label,
            ),
            ShapeError::EinsumMultipleEllipses { side, .. } => format!(
                "einsum spec has more than one ellipsis on the {} side",
                side,
            ),
            ShapeError::EinsumInputArityMismatch { spec_inputs, actual_inputs, .. } => format!(
                "einsum spec expects {} inputs, got {}",
                spec_inputs, actual_inputs,
            ),
            ShapeError::EinsumRankMismatch { input_index, expected, found, .. } => format!(
                "einsum input {}: expected rank {} per spec, got rank {}",
                input_index, expected, found,
            ),
            ShapeError::Other { message, .. } => message.clone(),
        }
    }
}

fn fmt_shape(s: &[DimExpr]) -> String {
    let mut out = String::new();
    for (i, d) in s.iter().enumerate() {
        if i > 0 { out.push_str(", "); }
        out.push_str(&format!("{}", d));
    }
    out
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DimExpr;

    fn lit(n: usize) -> DimExpr { DimExpr::Literal(n) }
    fn g(s: &str) -> DimExpr { DimExpr::generic(s) }

    #[test]
    fn dim_mismatch_summary_contains_both_dims() {
        let err = ShapeError::DimMismatch {
            expected: lit(3),
            found:    lit(4),
            span_a:   None,
            span_b:   None,
            context:  "einsum 'i'".to_string(),
        };
        let s = err.summary();
        assert!(s.contains("einsum"), "{}", s);
        assert!(s.contains("3"), "{}", s);
        assert!(s.contains("4"), "{}", s);
    }

    #[test]
    fn dim_mismatch_handles_generic() {
        let err = ShapeError::DimMismatch {
            expected: g("D"),
            found:    lit(3),
            span_a:   None,
            span_b:   None,
            context:  ".dot".to_string(),
        };
        let s = err.summary();
        assert!(s.contains("D"), "{}", s);
        assert!(s.contains("3"), "{}", s);
    }

    #[test]
    fn variance_mismatch_cites_both_sides() {
        let err = ShapeError::VarianceMismatch {
            left: VarianceTag::Upper,
            right: VarianceTag::Upper,
            span_a: None,
            span_b: None,
            context: ".contract".to_string(),
        };
        let s = err.summary();
        assert!(s.contains("Upper"), "{}", s);
        assert!(s.contains("variance"), "{}", s);
    }

    #[test]
    fn element_mismatch_summary() {
        let err = ShapeError::ElementMismatch {
            left:    ElementTy::F64,
            right:   ElementTy::F32,
            span_a:  None,
            span_b:  None,
            context: "ElementWise(Add)".to_string(),
        };
        let s = err.summary();
        assert!(s.contains("f64"), "{}", s);
        assert!(s.contains("f32"), "{}", s);
    }

    #[test]
    fn broadcast_incompatible_summary() {
        let err = ShapeError::BroadcastIncompatible {
            left:    vec![lit(3), lit(4)],
            right:   vec![lit(5)],
            span:    None,
            context: "ElementWise(Mul)".to_string(),
        };
        let s = err.summary();
        assert!(s.contains("incompatible"), "{}", s);
        assert!(s.contains("3"), "{}", s);
        assert!(s.contains("5"), "{}", s);
    }

    #[test]
    fn rank_mismatch_summary() {
        let err = ShapeError::RankMismatch {
            expected: 2,
            found:    1,
            span:     None,
            context:  "matmul".to_string(),
        };
        assert!(err.summary().contains("rank"));
        assert!(err.summary().contains("2"));
        assert!(err.summary().contains("1"));
    }

    #[test]
    fn einsum_overlimit_summary() {
        let err = ShapeError::EinsumLabelOverLimit { count: 9, max: 8, span: None };
        let s = err.summary();
        assert!(s.contains("9"), "{}", s);
        assert!(s.contains("8"), "{}", s);
    }

    #[test]
    fn einsum_output_label_missing_summary() {
        let err = ShapeError::EinsumOutputLabelNotInInputs { label: 'k', span: None };
        assert!(err.summary().contains("'k'"));
    }

    #[test]
    fn einsum_multiple_ellipses_summary() {
        let err = ShapeError::EinsumMultipleEllipses { side: "output", span: None };
        assert!(err.summary().contains("output"));
        assert!(err.summary().contains("ellipsis"));
    }

    #[test]
    fn einsum_arity_mismatch_summary() {
        let err = ShapeError::EinsumInputArityMismatch {
            spec_inputs: 2,
            actual_inputs: 3,
            span: None,
        };
        let s = err.summary();
        assert!(s.contains("2"), "{}", s);
        assert!(s.contains("3"), "{}", s);
    }

    #[test]
    fn einsum_input_rank_mismatch_summary() {
        let err = ShapeError::EinsumRankMismatch {
            input_index: 0,
            expected: 2,
            found: 3,
            span: None,
        };
        let s = err.summary();
        assert!(s.contains("0"), "{}", s);
        assert!(s.contains("rank 2"), "{}", s);
        assert!(s.contains("rank 3"), "{}", s);
    }

    #[test]
    fn other_summary_round_trips_message() {
        let err = ShapeError::Other {
            message: "construct with zero elements".to_string(),
            span:    None,
        };
        assert_eq!(err.summary(), "construct with zero elements");
    }

    #[test]
    fn primary_span_none_when_no_spans() {
        // none of the errors above have spans; primary_span returns None.
        let err = ShapeError::DimMismatch {
            expected: lit(1),
            found:    lit(2),
            span_a:   None,
            span_b:   None,
            context:  "x".to_string(),
        };
        assert!(err.primary_span().is_none());
    }

    #[test]
    fn display_matches_summary() {
        let err = ShapeError::RankMismatch {
            expected: 0,
            found:    1,
            span:     None,
            context:  "Index".to_string(),
        };
        assert_eq!(format!("{}", err), err.summary());
    }
}
