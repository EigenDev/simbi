// =============================================================================
// class.rs
//
// the determinism class witness. distinguishes values that are bit-equal across
// CPU and GPU (Det — only IEEE-754-mandated ops in their ancestry) from values
// that may drift 1-2 ULP between libdevice and libm (Tainted — a transcendental
// somewhere upstream). the live tensor IR (`tensor/ty.rs`, aliased `DetClass`)
// tags every typed value with one of these and joins them along the dataflow.
// =============================================================================

use std::fmt;

/// determinism class of a scalar value. consumed by the tensor IR as `DetClass`:
/// every `TensorTy` carries one, and ops `join` their inputs' classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassWitness {
    Det,
    Tainted,
}

impl ClassWitness {
    /// the join of two classes — the class of an op whose inputs carry these.
    /// `Tainted` if either input is tainted; `Det` only when both are `Det`. a
    /// commutative, associative (semi)lattice with `Det` bottom, `Tainted` top.
    pub const fn join(self, other: ClassWitness) -> ClassWitness {
        match (self, other) {
            (ClassWitness::Det, ClassWitness::Det) => ClassWitness::Det,
            _                                      => ClassWitness::Tainted,
        }
    }

    pub const fn is_det(self)     -> bool { matches!(self, ClassWitness::Det) }
    pub const fn is_tainted(self) -> bool { matches!(self, ClassWitness::Tainted) }
}

impl fmt::Display for ClassWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ClassWitness::Det     => "Det",
            ClassWitness::Tainted => "Tainted",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_lattice() {
        use ClassWitness::*;
        assert_eq!(Det.join(Det),         Det);
        assert_eq!(Det.join(Tainted),     Tainted);
        assert_eq!(Tainted.join(Det),     Tainted);
        assert_eq!(Tainted.join(Tainted), Tainted);
    }

    #[test]
    fn join_is_associative() {
        use ClassWitness::*;
        for a in [Det, Tainted] {
            for b in [Det, Tainted] {
                for c in [Det, Tainted] {
                    assert_eq!(a.join(b).join(c), a.join(b.join(c)));
                }
            }
        }
    }

    #[test]
    fn join_is_commutative() {
        use ClassWitness::*;
        for a in [Det, Tainted] {
            for b in [Det, Tainted] {
                assert_eq!(a.join(b), b.join(a));
            }
        }
    }

    #[test]
    fn det_is_bottom_tainted_is_top() {
        use ClassWitness::*;
        assert_eq!(Det.join(Det),     Det);
        assert_eq!(Det.join(Tainted), Tainted);
        assert_eq!(Tainted.join(Det), Tainted);
    }
}
