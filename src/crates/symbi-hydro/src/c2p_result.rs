// =============================================================================
// c2p_result.rs
//
// fallible conservative-to-primitive result type. bundles a usable (possibly
// floored) primitive value with a bitflag error code. avoids NaN cascades by
// always carrying a safe value, while recording what went wrong.
//
// usage:
//   let result = regime.to_primitive(&eos, &cons);
//   if result.is_ok() { /* clean */ }
//   let prim = result.value; // always safe to use
// =============================================================================

/// bitflag error code for cons-to-prim recovery.
/// zero = success. nonzero = something went wrong but the value is safe to use.
/// flags can be combined via `merge` (bitwise OR).
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

/// result of conservative-to-primitive inversion. always carries a usable
/// value (possibly floored), paired with an error code describing what
/// went wrong (if anything). Copy, no heap, GPU-safe.
#[derive(Clone, Copy, Debug)]
pub struct C2pResult<T: Copy> {
    pub value: T,
    pub error: ErrorCode,
}

impl<T: Copy> C2pResult<T> {
    #[inline]
    pub fn ok(value: T) -> Self {
        Self {
            value,
            error: ErrorCode::NONE,
        }
    }

    #[inline]
    pub fn err(value: T, error: ErrorCode) -> Self {
        Self { value, error }
    }

    #[inline]
    pub fn is_ok(&self) -> bool {
        self.error.is_ok()
    }

    #[inline]
    pub fn unwrap(self) -> T {
        if self.error.is_err() {
            panic!("c2p failed: {}", self.error);
        }
        self.value
    }
}

/// finite placeholder returned when relativistic c2p rejects its conserved density before
/// recovery. the error code is authoritative and the value must never enter evolution.
pub const C2P_FAILURE_SENTINEL: f64 = 1.0;

// the shared relativistic c2p diagnostic contract (RHD + RMHD). ONE source so the two
// regimes' threshold conventions cannot drift (tier-1 #5: the density-scaled-vs-absolute
// pressure floor, the superluminal margin, and the input-NaN check had all diverged).
//
// these are POST-HOC flags on the RAW recovered state — NO silent floor: the caller returns
// the raw recovery value, this only reports what is non-physical (feedback_no_silent_floors).
// thresholds are dimensionally clean:
//   * NON_FINITE       : rho or pressure is NaN.
//   * NEGATIVE_PRESSURE: pressure < 0 (strict). a near-zero
//                        positive pressure is the valid cold limit — so no
//                        arbitrary `1e-12` / `1e-12*rho` floor.
//   * SUPERLUMINAL     : v^2 >= 1 (the Lorentz factor is finite only for v^2 < 1) or v^2
//                        is NaN. no luminal margin.
pub fn relativistic_c2p_code<S: symbi_ir::algebra::Scalar + symbi_algebra::OrderedNumeric>(
    rho: S,
    pre: S,
    v_sq: S,
) -> ErrorCode {
    let mut code = ErrorCode::NONE;
    if !(rho == rho) || !(pre == pre) {
        code = code.merge(ErrorCode::NON_FINITE);
    }
    if pre < S::ZERO {
        code = code.merge(ErrorCode::NEGATIVE_PRESSURE);
    }
    if v_sq >= S::ONE || !(v_sq == v_sq) {
        code = code.merge(ErrorCode::SUPERLUMINAL);
    }
    code
}

// shared input-density guard for relativistic c2p (a host-only early-out before the kernel
// path). returns the failure code for a non-positive or non-finite conserved density, else
// None. the NaN branch was present in RHD but missing in RMHD before the unification.
pub fn relativistic_density_guard<S: symbi_ir::algebra::Scalar + symbi_algebra::OrderedNumeric>(
    dd: S,
) -> Option<ErrorCode> {
    if dd <= S::ZERO || !(dd == dd) {
        let mut code = ErrorCode::NEGATIVE_DENSITY;
        if !(dd == dd) {
            code = code.merge(ErrorCode::NON_FINITE);
        }
        Some(code)
    } else {
        None
    }
}

/// the pressure written when a relativistic recovery finds the conserved state
/// outside the physical cone. scaling the negative signal with `|D|` preserves
/// homogeneity under a change of density units while remaining finite and
/// non-positive for every valid conserved density.
#[inline]
pub fn c2p_cone_fail_pressure<S: symbi_ir::algebra::Scalar>(den: S) -> S {
    S::ZERO - den.abs()
}

/// the shared relativistic-c2p velocity ceiling, squared: `v_limit^2 = r^2 / (1 + r^2)` with
/// `r = |S| / D` the rescaled conserved-momentum magnitude (enthalpy floor `h0 = 1`; KKC/Kastaun
/// 2021 Eq. 40). the true 3-velocity of ANY in-cone state with `p >= 0` satisfies `v <= v_limit`,
/// so clamping a recovered `v^2` to this leaves a valid recovery unchanged while keeping the
/// Lorentz factor / density finite for an out-of-cone input — no NaN. ONE source shared by
/// `rhd_recover` and `rmhd_recover` so the two regimes cannot drift. carrier-generic.
#[inline]
pub fn relativistic_velocity_ceiling_sq<S: symbi_ir::algebra::Scalar>(r_sq: S) -> S {
    r_sq / (S::ONE + r_sq)
}

/// the shared relativistic-c2p admissibility residual `q(U)/D = tau/D + 1 - sqrt(1 + r^2)`,
/// `r = |S| / D` (Wu 2017; the B-free hydro limit of the RMHD KKC form — the magnetic terms do
/// not enter the cone bound). strictly positive iff a physical subluminal (`p > 0`, `v < 1`)
/// recovery exists; non-positive marks an out-of-cone conserved state whose pressure the caller
/// drives to [`c2p_cone_fail_pressure`]. one source shared by both recoveries.
#[inline]
pub fn relativistic_cone_residual<S: symbi_ir::algebra::Scalar>(qq: S, r_sq: S) -> S {
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
    fn c2p_result_ok() {
        let result = C2pResult::ok(42.0_f64);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42.0);
    }

    #[test]
    fn c2p_result_err_has_value() {
        let result = C2pResult::err(1e-12_f64, ErrorCode::NEGATIVE_DENSITY);
        assert!(!result.is_ok());
        assert_eq!(result.value, 1e-12);
    }

    #[test]
    #[should_panic(expected = "c2p failed")]
    fn c2p_result_unwrap_panics_on_error() {
        let result = C2pResult::err(0.0_f64, ErrorCode::SUPERLUMINAL);
        result.unwrap();
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
