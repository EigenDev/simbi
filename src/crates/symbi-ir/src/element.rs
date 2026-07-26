// =============================================================================
// element.rs
//
// element type of a tensor: the scalar primitive at every component.
// the supported types are F64, F32, I32, U32, Bool. promotions and casts
// are out of scope for V1 — mismatched
// elements at op-construction time are an IR build error.
//
// V2 will extend with BF16, F16, Complex64, Complex32; the enum's
// non-exhaustive layout keeps that addition source-compatible.
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ElementTy {
    F64,
    F32,
    I32,
    U32,
    Bool,
}

impl ElementTy {
    /// human-readable name used in diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            ElementTy::F64 => "f64",
            ElementTy::F32 => "f32",
            ElementTy::I32 => "i32",
            ElementTy::U32 => "u32",
            ElementTy::Bool => "bool",
        }
    }

    /// true for IEEE 754 floating-point types.
    pub fn is_float(self) -> bool {
        matches!(self, ElementTy::F64 | ElementTy::F32)
    }

    /// true for integer types (signed or unsigned).
    pub fn is_integer(self) -> bool {
        matches!(self, ElementTy::I32 | ElementTy::U32)
    }
}

impl std::fmt::Display for ElementTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_matches_rust_type_name() {
        assert_eq!(ElementTy::F64.label(), "f64");
        assert_eq!(ElementTy::F32.label(), "f32");
        assert_eq!(ElementTy::I32.label(), "i32");
        assert_eq!(ElementTy::U32.label(), "u32");
        assert_eq!(ElementTy::Bool.label(), "bool");
    }

    #[test]
    fn is_float_partition() {
        assert!(ElementTy::F64.is_float());
        assert!(ElementTy::F32.is_float());
        assert!(!ElementTy::I32.is_float());
        assert!(!ElementTy::U32.is_float());
        assert!(!ElementTy::Bool.is_float());
    }

    #[test]
    fn is_integer_partition() {
        assert!(ElementTy::I32.is_integer());
        assert!(ElementTy::U32.is_integer());
        assert!(!ElementTy::F64.is_integer());
        assert!(!ElementTy::Bool.is_integer());
    }

    #[test]
    fn display_matches_label() {
        assert_eq!(format!("{}", ElementTy::F64), "f64");
    }

    #[test]
    fn float_and_integer_are_disjoint() {
        for ty in [
            ElementTy::F64,
            ElementTy::F32,
            ElementTy::I32,
            ElementTy::U32,
            ElementTy::Bool,
        ] {
            assert!(!(ty.is_float() && ty.is_integer()), "{} cannot be both", ty);
        }
    }
}
