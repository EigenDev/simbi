// =============================================================================
// variance.rs
//
// index variance tag for a tensor value: indicates whether the index
// is contravariant (Upper, $v^i$), covariant (Lower, $v_i$), or
// untagged (plain Tensor, no GR meaning).
//
// V1 carries a single variance tag per tensor (matches the algebra
// crate's Indexed<V, S, D> single-axis design). V2 will turn this
// into a Vec<VarianceTag> of length `rank` for per-axis variance, the
// shape needed by general GR tensors with mixed Upper/Lower indices.
// the einsum spec validator already iterates over labels, so per-axis
// is mechanical when needed.
//
// used by:
//   - TensorTy.variance (one tag per tensor)
//   - einsum's contracted-index variance check (Upper pairs with Lower)
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarianceTag {
    /// plain tensor — no index-position meaning. cannot contract via
    /// `.contract()`; use `.dot()` for variance-agnostic inner products.
    Untagged,
    /// contravariant: $v^i$. produced by `Indexed<Upper, …>` and `Contravariant<…>`.
    Upper,
    /// covariant: $v_i$. produced by `Indexed<Lower, …>` and `Covariant<…>`.
    Lower,
}

impl VarianceTag {
    /// human-readable name for diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            VarianceTag::Untagged => "Untagged (plain Tensor)",
            VarianceTag::Upper => "Upper (contravariant)",
            VarianceTag::Lower => "Lower (covariant)",
        }
    }

    /// can `self` pair with `other` in a `.contract()`? requires
    /// opposite variance — one Upper and one Lower. Untagged operands
    /// are rejected (use `.dot()` for variance-agnostic contractions).
    pub fn contracts_with(self, other: VarianceTag) -> bool {
        matches!(
            (self, other),
            (VarianceTag::Upper, VarianceTag::Lower) | (VarianceTag::Lower, VarianceTag::Upper)
        )
    }
}

impl std::fmt::Display for VarianceTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_is_descriptive() {
        assert!(VarianceTag::Upper.label().contains("Upper"));
        assert!(VarianceTag::Lower.label().contains("Lower"));
        assert!(VarianceTag::Untagged.label().contains("Untagged"));
    }

    #[test]
    fn contraction_requires_opposite_variance() {
        assert!(VarianceTag::Upper.contracts_with(VarianceTag::Lower));
        assert!(VarianceTag::Lower.contracts_with(VarianceTag::Upper));
    }

    #[test]
    fn contraction_rejects_same_variance() {
        assert!(!VarianceTag::Upper.contracts_with(VarianceTag::Upper));
        assert!(!VarianceTag::Lower.contracts_with(VarianceTag::Lower));
    }

    #[test]
    fn contraction_rejects_untagged() {
        // Untagged can never contract; .dot() handles untagged inner products.
        assert!(!VarianceTag::Untagged.contracts_with(VarianceTag::Upper));
        assert!(!VarianceTag::Untagged.contracts_with(VarianceTag::Lower));
        assert!(!VarianceTag::Untagged.contracts_with(VarianceTag::Untagged));
        assert!(!VarianceTag::Upper.contracts_with(VarianceTag::Untagged));
        assert!(!VarianceTag::Lower.contracts_with(VarianceTag::Untagged));
    }

    #[test]
    fn display_matches_label() {
        assert_eq!(format!("{}", VarianceTag::Upper), "Upper (contravariant)");
        assert_eq!(format!("{}", VarianceTag::Lower), "Lower (covariant)");
        assert_eq!(format!("{}", VarianceTag::Untagged), "Untagged (plain Tensor)");
    }
}
