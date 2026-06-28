// =============================================================================
// dim.rs
//
// dimension expressions for tensor shapes. each axis is either a
// compile-time literal (e.g., 3, 4) or a const-generic identifier (e.g.
// "D", "N") that the backend resolves at monomorphization time.
//
// shape equality (§ 1.5): two `DimExpr` values are equal iff:
//   Literal(n) == Literal(n)            (same value)
//   Generic(s) == Generic(t)            (same interned symbol — O(1))
//   Literal(_) != Generic(_)            (always — bridged via broadcast)
//
// broadcast compatibility (§ 1.5): `S` broadcasts to `T` iff `S` has at
// most `T.len()` dims and, after right-aligning, each pair (s, t)
// satisfies `s == t || s == Literal(1)`. Generic dims never broadcast
// against Literal dims unless the Generic side is itself Literal(1).
// =============================================================================

use crate::symbol::Symbol;

/// one dimension of a tensor's shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DimExpr {
    /// known at macro expansion time. fully unrollable.
    Literal(usize),
    /// const-generic identifier from the surrounding context (e.g., "D").
    /// resolved at monomorphization; the backend either emits a fixed-
    /// bound loop or unrolls per monomorph value.
    Generic(Symbol),
}

impl DimExpr {
    /// convenience constructor: intern the name and wrap.
    pub fn generic(name: &str) -> Self {
        DimExpr::Generic(Symbol::intern(name))
    }

    /// is this a literal `1`? broadcastable along any axis it meets.
    pub fn is_one(&self) -> bool {
        matches!(self, DimExpr::Literal(1))
    }
}

impl std::fmt::Display for DimExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimExpr::Literal(n) => write!(f, "{}", n),
            DimExpr::Generic(s) => write!(f, "{}", s),
        }
    }
}

/// a tensor's full shape, ordered outermost-to-innermost.
pub type Shape = Vec<DimExpr>;

/// element-wise equality of two shapes.
pub fn shapes_equal(a: &[DimExpr], b: &[DimExpr]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// does shape `s` broadcast to shape `t`?
///
/// rules:
///   - `s.len() <= t.len()`.
///   - right-align `s` against `t`; each pair `(s_i, t_i)` must satisfy
///     `s_i == t_i || s_i == Literal(1)`.
///   - the leading dims of `t` that `s` doesn't cover are implicit
///     "1"s on `s`'s side (the standard NumPy-style rule).
pub fn broadcasts_to(s: &[DimExpr], t: &[DimExpr]) -> bool {
    if s.len() > t.len() {
        return false;
    }
    let offset = t.len() - s.len();
    for (i, s_dim) in s.iter().enumerate() {
        let t_dim = &t[offset + i];
        if s_dim != t_dim && !s_dim.is_one() {
            return false;
        }
    }
    true
}

/// the broadcast shape of `a` and `b`, if any. None on incompatible.
///
/// rule (symmetric): right-align the shorter against the longer. for
/// each axis, the broadcast dim is whichever side isn't Literal(1)
/// (mismatch unless they agree or one is 1). leading axes from only
/// the longer side pass through unchanged.
pub fn broadcast_shape(a: &[DimExpr], b: &[DimExpr]) -> Option<Shape> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let offset = long.len() - short.len();
    let mut out = Vec::with_capacity(long.len());
    for i in 0..long.len() {
        if i < offset {
            out.push(long[i].clone());
            continue;
        }
        let l = &long[i];
        let s = &short[i - offset];
        out.push(broadcast_axis(l, s)?);
    }
    Some(out)
}

/// pick the broadcast dim for one axis. None on incompatible.
fn broadcast_axis(a: &DimExpr, b: &DimExpr) -> Option<DimExpr> {
    if a == b {
        return Some(a.clone());
    }
    if a.is_one() {
        return Some(b.clone());
    }
    if b.is_one() {
        return Some(a.clone());
    }
    // both non-1 and unequal: incompatible (covers Literal vs Generic
    // mismatch as a special case).
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // helpers
    fn lit(n: usize) -> DimExpr { DimExpr::Literal(n) }
    fn g(s: &str) -> DimExpr { DimExpr::generic(s) }

    // ---- equality ----

    #[test]
    fn literal_equal_when_same_value() {
        assert_eq!(lit(3), lit(3));
        assert_ne!(lit(3), lit(4));
    }

    #[test]
    fn generic_equal_when_same_symbol() {
        assert_eq!(g("D"), g("D"));
        assert_eq!(g("D"), g(" D "));  // interner trims
        assert_ne!(g("D"), g("N"));
    }

    #[test]
    fn literal_never_equals_generic() {
        assert_ne!(lit(3), g("D"));
        assert_ne!(g("D"), lit(3));
    }

    #[test]
    fn shape_equality_element_wise() {
        assert!(shapes_equal(&[lit(3), lit(4)], &[lit(3), lit(4)]));
        assert!(shapes_equal(&[g("D")], &[g("D")]));
        assert!(!shapes_equal(&[lit(3)], &[lit(3), lit(4)]));
        assert!(!shapes_equal(&[lit(3), lit(4)], &[lit(4), lit(3)]));
    }

    // ---- broadcasts_to ----

    #[test]
    fn equal_shapes_broadcast() {
        assert!(broadcasts_to(&[lit(3)], &[lit(3)]));
        assert!(broadcasts_to(&[lit(3), lit(4)], &[lit(3), lit(4)]));
    }

    #[test]
    fn shorter_shape_right_aligns() {
        // [3] -> [4, 3] : ok (3 matches innermost of [4, 3])
        assert!(broadcasts_to(&[lit(3)], &[lit(4), lit(3)]));
        // [4] -> [4, 3] : not ok (4 doesn't match 3 innermost)
        assert!(!broadcasts_to(&[lit(4)], &[lit(4), lit(3)]));
    }

    #[test]
    fn literal_one_broadcasts_across_axis() {
        // [1, 3] -> [4, 3] : ok via broadcast on first axis
        assert!(broadcasts_to(&[lit(1), lit(3)], &[lit(4), lit(3)]));
        // [3, 1] -> [3, 5] : ok via broadcast on second axis
        assert!(broadcasts_to(&[lit(3), lit(1)], &[lit(3), lit(5)]));
        // [1] -> [4, 3] : ok (1 broadcasts to 3, leading 4 unmatched is fine)
        assert!(broadcasts_to(&[lit(1)], &[lit(4), lit(3)]));
    }

    #[test]
    fn longer_shape_does_not_broadcast_down() {
        assert!(!broadcasts_to(&[lit(4), lit(3)], &[lit(3)]));
    }

    #[test]
    fn generic_dim_matches_only_same_generic_or_one() {
        // [D] -> [D] : ok (same symbol)
        assert!(broadcasts_to(&[g("D")], &[g("D")]));
        // [D] -> [N] : not ok (different symbols)
        assert!(!broadcasts_to(&[g("D")], &[g("N")]));
        // [D] -> [3] : not ok (Generic vs Literal)
        assert!(!broadcasts_to(&[g("D")], &[lit(3)]));
        // [1] -> [D] : ok (Literal(1) broadcasts to anything)
        assert!(broadcasts_to(&[lit(1)], &[g("D")]));
        // [1, D] -> [4, D] : ok (1 broadcasts, D matches D)
        assert!(broadcasts_to(&[lit(1), g("D")], &[lit(4), g("D")]));
    }

    #[test]
    fn rank_0_broadcasts_to_anything() {
        // an empty shape (scalar) broadcasts to any shape.
        assert!(broadcasts_to(&[], &[]));
        assert!(broadcasts_to(&[], &[lit(3)]));
        assert!(broadcasts_to(&[], &[lit(4), lit(5), g("D")]));
    }

    // ---- broadcast_shape ----

    #[test]
    fn broadcast_shape_equal() {
        let r = broadcast_shape(&[lit(3)], &[lit(3)]).unwrap();
        assert_eq!(r, vec![lit(3)]);
    }

    #[test]
    fn broadcast_shape_picks_non_one_axis() {
        // [3, 1] vs [1, 4] -> [3, 4]
        let r = broadcast_shape(&[lit(3), lit(1)], &[lit(1), lit(4)]).unwrap();
        assert_eq!(r, vec![lit(3), lit(4)]);
    }

    #[test]
    fn broadcast_shape_extends_shorter_side() {
        // [3] vs [4, 3] -> [4, 3]
        let r = broadcast_shape(&[lit(3)], &[lit(4), lit(3)]).unwrap();
        assert_eq!(r, vec![lit(4), lit(3)]);
    }

    #[test]
    fn broadcast_shape_with_scalar() {
        // [] vs [3, 4] -> [3, 4]
        let r = broadcast_shape(&[], &[lit(3), lit(4)]).unwrap();
        assert_eq!(r, vec![lit(3), lit(4)]);
        // [3, 4] vs [] -> [3, 4] (symmetric)
        let r = broadcast_shape(&[lit(3), lit(4)], &[]).unwrap();
        assert_eq!(r, vec![lit(3), lit(4)]);
    }

    #[test]
    fn broadcast_shape_mismatch_returns_none() {
        // [3] vs [4] — neither is 1, not equal: incompatible.
        assert!(broadcast_shape(&[lit(3)], &[lit(4)]).is_none());
        // [D] vs [3] — Generic vs Literal: incompatible.
        assert!(broadcast_shape(&[g("D")], &[lit(3)]).is_none());
        // [D] vs [N] — different symbols: incompatible.
        assert!(broadcast_shape(&[g("D")], &[g("N")]).is_none());
    }

    #[test]
    fn broadcast_shape_symmetric() {
        let a = vec![lit(3), lit(1)];
        let b = vec![lit(1), lit(4)];
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        assert_eq!(ab, ba);
    }

    #[test]
    fn broadcast_shape_with_generic_dims() {
        // [D, 1] vs [1, N] -> [D, N]
        let r = broadcast_shape(&[g("D"), lit(1)], &[lit(1), g("N")]).unwrap();
        assert_eq!(r, vec![g("D"), g("N")]);
        // [D, D] vs [D, D] -> [D, D]
        let r = broadcast_shape(&[g("D"), g("D")], &[g("D"), g("D")]).unwrap();
        assert_eq!(r, vec![g("D"), g("D")]);
    }

    #[test]
    fn is_one_only_matches_literal_one() {
        assert!(lit(1).is_one());
        assert!(!lit(2).is_one());
        assert!(!g("D").is_one());
    }

    #[test]
    fn display_dim_expr() {
        assert_eq!(format!("{}", lit(7)), "7");
        assert_eq!(format!("{}", g("D")), "D");
    }
}
