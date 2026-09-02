// =============================================================================
// c2p_result.rs
//
// the diagnostic-boundary error code and the carrier-generic relativistic
// c2p helpers: the bitflag `ErrorCode` the kernel status channel and the
// scratch-field scans speak, plus the cone/ceiling functions both
// relativistic recoveries share. the host recovery audits live in
// `recovery.rs` with the outcome algebra they certify.
//
// usage:
//   let ceiling = relativistic_velocity_ceiling_sq(r_sq);
//   let residual = relativistic_cone_residual(qq, r_sq);
// =============================================================================

/// the iteration cap every relativistic cons->prim solver shares -- the rhd newton,
/// the rmhd kkc false-position, and their metric-aware twins, on the host and in the
/// baked kernels alike. the solvers converge in far fewer steps on admissible states;
/// the cap only bounds pathological inputs, so one number serves them all.
use symbi_carrier::Scalar;

pub const C2P_MAX_ITER: usize = 100;

/// bitflag error code for cons-to-prim recovery.
/// zero = success. nonzero = the recovery audit flags for the cell.
/// flags can be combined via `merge` (bitwise or).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ErrorCode(pub u8);

impl ErrorCode {
    pub const NONE: Self = Self(0);
    pub const NEGATIVE_DENSITY: Self = Self(1 << 0);
    pub const NEGATIVE_PRESSURE: Self = Self(1 << 1);
    pub const NON_FINITE: Self = Self(1 << 2);
    pub const SUPERLUMINAL: Self = Self(1 << 3);
    pub const MAX_ITER: Self = Self(1 << 4);
    pub const NEGATIVE_ENERGY: Self = Self(1 << 5);
    pub const INVALID_PRIMITIVE: Self = Self(1 << 6);

    #[inline]
    pub fn is_ok(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_err(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }

    pub fn describe(self) -> &'static str {
        if self.0 == 0 {
            return "ok";
        }
        if self.contains(Self::NEGATIVE_DENSITY) {
            return "negative density";
        }
        if self.contains(Self::NEGATIVE_PRESSURE) {
            return "negative pressure";
        }
        if self.contains(Self::NON_FINITE) {
            return "non-finite value";
        }
        if self.contains(Self::SUPERLUMINAL) {
            return "superluminal velocity";
        }
        if self.contains(Self::MAX_ITER) {
            return "max iterations reached";
        }
        if self.contains(Self::NEGATIVE_ENERGY) {
            return "negative energy";
        }
        if self.contains(Self::INVALID_PRIMITIVE) {
            return "primitive outside the strict admissible interior";
        }
        "unknown error"
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        let flags = [
            (Self::NEGATIVE_DENSITY, "negative_density"),
            (Self::NEGATIVE_PRESSURE, "negative_pressure"),
            (Self::NON_FINITE, "non_finite"),
            (Self::SUPERLUMINAL, "superluminal"),
            (Self::MAX_ITER, "max_iter"),
            (Self::NEGATIVE_ENERGY, "negative_energy"),
            (Self::INVALID_PRIMITIVE, "invalid_primitive"),
        ];
        for (flag, name) in &flags {
            if self.contains(*flag) {
                if !first {
                    write!(f, "|")?;
                }
                write!(f, "{}", name)?;
                first = false;
            }
        }
        if first {
            write!(f, "ok")?;
        }
        Ok(())
    }
}

/// the pressure written when a relativistic recovery finds the conserved state
/// outside the physical cone. scaling the negative signal with `|D|` preserves
/// homogeneity under a change of density units while remaining finite and
/// non-positive for every valid conserved density.
#[inline]
pub fn c2p_cone_fail_pressure<S: Scalar>(den: S) -> S {
    S::ZERO - den.abs()
}

/// the shared relativistic-c2p velocity ceiling, squared: `v_limit^2 = r^2 / (1 + r^2)` with
/// `r = |S| / D` the rescaled conserved-momentum magnitude (enthalpy floor `h0 = 1`; KKC/Kastaun
/// 2021 Eq. 40). the true 3-velocity of any in-cone state with `p >= 0` satisfies `v <= v_limit`,
/// so clamping a recovered `v^2` to this leaves a valid recovery unchanged while keeping the
/// Lorentz factor / density finite for an out-of-cone input — no NaN. one source shared by
/// `rhd_recover` and `rmhd_recover` so the two regimes cannot drift. carrier-generic.
#[inline]
pub fn relativistic_velocity_ceiling_sq<S: Scalar>(r_sq: S) -> S {
    r_sq / (S::ONE + r_sq)
}

/// the shared relativistic-c2p admissibility residual `q(U)/D = tau/D + 1 - sqrt(1 + r^2)`,
/// `r = |S| / D` (Wu 2017; the B-free hydro limit of the RMHD KKC form — the magnetic terms do
/// not enter the cone bound). strictly positive iff a physical subluminal (`p > 0`, `v < 1`)
/// recovery exists; non-positive marks an out-of-cone conserved state whose pressure the caller
/// drives to [`c2p_cone_fail_pressure`]. one source shared by both recoveries.
#[inline]
pub fn relativistic_cone_residual<S: Scalar>(qq: S, r_sq: S) -> S {
    qq + S::ONE - (S::ONE + r_sq).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_none_is_ok() {
        assert!(ErrorCode::NONE.is_ok());
        assert!(!ErrorCode::NONE.is_err());
    }

    #[test]
    fn error_code_single_flag() {
        let code = ErrorCode::NEGATIVE_DENSITY;
        assert!(code.is_err());
        assert!(code.contains(ErrorCode::NEGATIVE_DENSITY));
        assert!(!code.contains(ErrorCode::NEGATIVE_PRESSURE));
    }

    #[test]
    fn error_code_merge() {
        let code = ErrorCode::NEGATIVE_DENSITY.merge(ErrorCode::NON_FINITE);
        assert!(code.contains(ErrorCode::NEGATIVE_DENSITY));
        assert!(code.contains(ErrorCode::NON_FINITE));
        assert!(!code.contains(ErrorCode::SUPERLUMINAL));
    }

    #[test]
    fn error_code_merge_with_none() {
        let code = ErrorCode::NONE.merge(ErrorCode::NEGATIVE_PRESSURE);
        assert_eq!(code, ErrorCode::NEGATIVE_PRESSURE);
    }

    #[test]
    fn error_code_display() {
        let code = ErrorCode::NEGATIVE_DENSITY.merge(ErrorCode::MAX_ITER);
        let s = format!("{}", code);
        assert!(s.contains("negative_density"));
        assert!(s.contains("max_iter"));
    }

    #[test]
    fn error_code_describe() {
        assert_eq!(ErrorCode::NONE.describe(), "ok");
        assert_eq!(ErrorCode::NEGATIVE_DENSITY.describe(), "negative density");
        assert_eq!(ErrorCode::SUPERLUMINAL.describe(), "superluminal velocity");
    }
}
