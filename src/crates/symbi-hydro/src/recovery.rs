// =============================================================================
// recovery.rs
//
// the host recovery outcome algebra: cons->prim recovery returns Rust's
// closed sum, `Recovery<T> = Result<Recovered<T>, RecoveryFailure<T>>`.
// - `Recovered<T>` is minted inside this crate only, by the regime doors,
//   after their named recovery-interior predicate passes; `into_inner` is
//   the single transition back to an ordinary primitive.
// - `RecoveryFailure<T>` carries a nonempty closed `RecoveryIssues` set and
//   a diagnostic-only candidate: the failed state can be formatted and
//   inspected as text, and there is no typed path by which it re-enters
//   evolution.
// - `RecoveryIssues` is the closed seven-issue vocabulary over a nonzero
//   bitset; the empty set and unknown bits are unrepresentable. every
//   nonempty union of the seven issues is lawful (the recovery predicates
//   merge freely — a non-finite conserved density is negative-density and
//   non-finite at once). the serialized `u8` layout matches the diagnostic
//   `ErrorCode` bit-for-bit and crossing to it is an explicit boundary
//   operation.
//
// usage:
//   match regime.to_primitive(&eos, &cons) {
//       Ok(prim) => evolve(prim.into_inner()),
//       Err(failure) => log(failure.issues(), failure.candidate().snapshot()),
//   }
// =============================================================================

use std::num::NonZeroU8;

use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_carrier::Scalar;

use crate::c2p_result::ErrorCode;

/// the closed, nonempty recovery-issue set. bits mirror the diagnostic
/// `ErrorCode` layout; the seven named constants and `merge` are the whole
/// construction surface, so an unknown bit or an empty set cannot be minted.
///
/// external construction from raw bits is rejected —
///
/// ```compile_fail
/// let _ = symbi_hydro::RecoveryIssues(std::num::NonZeroU8::new(9).unwrap());
/// ```
///
/// and only the diagnostic boundary door admits serialized bits, refusing the
/// empty set and any bit outside the vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RecoveryIssues(NonZeroU8);

/// the bitmask spanned by the seven named issues.
const ISSUE_MASK: u8 = 0x7F;

const fn issue(bit: u8) -> RecoveryIssues {
    match NonZeroU8::new(bit) {
        Some(b) => RecoveryIssues(b),
        None => panic!("issue bit must be nonzero"),
    }
}

impl RecoveryIssues {
    pub const NEGATIVE_DENSITY: Self = issue(1 << 0);
    pub const NEGATIVE_PRESSURE: Self = issue(1 << 1);
    pub const NON_FINITE: Self = issue(1 << 2);
    pub const SUPERLUMINAL: Self = issue(1 << 3);
    pub const MAX_ITER: Self = issue(1 << 4);
    pub const NEGATIVE_ENERGY: Self = issue(1 << 5);
    pub const INVALID_PRIMITIVE: Self = issue(1 << 6);

    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self(self.0 | other.0.get())
    }

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0.get() & other.0.get()) == other.0.get()
    }

    /// fold an issue into an accumulator that starts clean; the clean state is
    /// the absence of a set, so the successful branch carries no issues at all.
    #[inline]
    pub fn note(acc: Option<Self>, issue: Self) -> Option<Self> {
        Some(match acc {
            Some(existing) => existing.merge(issue),
            None => issue,
        })
    }

    /// the serialized diagnostic byte — the explicit boundary operation toward
    /// the scratch-field / census representation.
    #[inline]
    pub fn to_diagnostic_u8(self) -> u8 {
        self.0.get()
    }

    /// admit serialized diagnostic bits back into the vocabulary: the empty
    /// set and any bit outside the seven named issues are refused.
    pub fn from_diagnostic_u8(bits: u8) -> Option<Self> {
        if bits & !ISSUE_MASK != 0 {
            return None;
        }
        NonZeroU8::new(bits).map(Self)
    }
}

/// the diagnostic-boundary crossing: the `u8` scratch/census representation.
impl From<RecoveryIssues> for ErrorCode {
    fn from(issues: RecoveryIssues) -> ErrorCode {
        ErrorCode(issues.to_diagnostic_u8())
    }
}

impl std::fmt::Display for RecoveryIssues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ErrorCode(self.to_diagnostic_u8()).fmt(f)
    }
}

/// a primitive state the named recovery-interior predicate accepted. minted by
/// the regime doors alone —
///
/// ```compile_fail
/// let prim: symbi_hydro::Prim<f64, 2> = todo!();
/// let _ = symbi_hydro::Recovered(prim);
/// ```
///
/// ```compile_fail
/// let prim: symbi_hydro::Prim<f64, 2> = todo!();
/// let _ = symbi_hydro::Recovered::mint(prim);
/// ```
///
/// `into_inner` is the single transition back to an ordinary primitive, taken
/// after the caller has handled the outcome. acceptance proves the recovery
/// interior only (finite, positive-density/pressure, subluminal where the
/// regime is relativistic); membership in any wider constraint family is a
/// separate, law-indexed proof.
#[derive(Clone, Copy, Debug)]
pub struct Recovered<T>(T);

impl<T> Recovered<T> {
    #[inline]
    fn mint(value: T) -> Self {
        Self(value)
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// a failed candidate held for diagnostics: it renders as text, and no
/// method returns `T` or `&T` —
///
/// ```compile_fail
/// fn steal<T>(d: symbi_hydro::DiagnosticOnly<T>) -> T { d.0 }
/// ```
///
/// a rejection that precedes any recovery iterate carries the placeholder
/// instead of a fabricated physical state.
#[derive(Clone, Copy)]
pub struct DiagnosticOnly<T>(Option<T>);

impl<T> DiagnosticOnly<T> {
    #[inline]
    pub(crate) fn of(candidate: T) -> Self {
        Self(Some(candidate))
    }

    #[inline]
    pub(crate) fn placeholder() -> Self {
        Self(None)
    }

    /// the formatted diagnostic snapshot of the rejected candidate.
    pub fn snapshot(&self) -> String
    where
        T: std::fmt::Debug,
    {
        match &self.0 {
            Some(candidate) => format!("{candidate:?}"),
            None => "rejected before recovery produced a candidate".to_string(),
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for DiagnosticOnly<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.snapshot())
    }
}

/// a rejected recovery: the nonempty issue set plus the diagnostic-only
/// candidate. the failed state informs a log line or a policy; it has no
/// typed path back into evolution.
#[derive(Clone, Copy, Debug)]
pub struct RecoveryFailure<T> {
    issues: RecoveryIssues,
    candidate: DiagnosticOnly<T>,
}

impl<T> RecoveryFailure<T> {
    #[inline]
    pub(crate) fn new(issues: RecoveryIssues, candidate: T) -> Self {
        Self {
            issues,
            candidate: DiagnosticOnly::of(candidate),
        }
    }

    #[inline]
    pub(crate) fn without_candidate(issues: RecoveryIssues) -> Self {
        Self {
            issues,
            candidate: DiagnosticOnly::placeholder(),
        }
    }

    #[inline]
    pub fn issues(&self) -> RecoveryIssues {
        self.issues
    }

    #[inline]
    pub fn candidate(&self) -> &DiagnosticOnly<T> {
        &self.candidate
    }
}

impl<T> std::fmt::Display for RecoveryFailure<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "c2p rejected: {}", self.issues)
    }
}

/// the host recovery outcome.
pub type Recovery<T> = Result<Recovered<T>, RecoveryFailure<T>>;

/// an evaluated recovery-interior audit: the opaque carrier between a named
/// predicate and `judge`. the field is private to this module and the named
/// predicates below are its only constructors, so certification is spelled
/// only by evaluating a predicate — `judge(candidate, None)` has no spelling.
/// the structural gate (`recovery_boundaries`) pins the construction sites.
pub(crate) struct RecoveryAudit(Option<RecoveryIssues>);

/// the one audit-to-outcome door the regime doors share: a clean audit mints
/// `Recovered`, a flagged one carries the candidate into the failure.
#[inline]
pub(crate) fn judge<T>(candidate: T, audit: RecoveryAudit) -> Recovery<T> {
    match audit.0 {
        None => Ok(Recovered::mint(candidate)),
        Some(issues) => Err(RecoveryFailure::new(issues, candidate)),
    }
}

/// host finiteness: NaN and both infinities fail (`inf - inf` is NaN).
#[inline]
fn finite<S: Scalar + OrderedNumeric>(x: S) -> bool {
    (x - x) == S::ZERO
}

/// componentwise host finiteness of a tensor.
#[inline]
fn tensor_finite<S: Scalar + OrderedNumeric, const D: usize>(x: &Tensor<S, D>) -> bool {
    let mut all = true;
    for k in 0..D {
        all = all && finite(x[k]);
    }
    all
}

/// the newtonian recovery-interior predicate: positive finite density and
/// pressure, finite velocity components.
pub(crate) fn newtonian_prim_audit<S: Scalar + OrderedNumeric, const D: usize>(
    rho: S,
    pre: S,
    vel: &Tensor<S, D>,
) -> RecoveryAudit {
    let mut audit = None;
    if rho <= S::ZERO {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NEGATIVE_DENSITY);
    }
    if pre <= S::ZERO {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NEGATIVE_PRESSURE);
    }
    if !finite(rho) || !finite(pre) || !tensor_finite(vel) {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NON_FINITE);
    }
    RecoveryAudit(audit)
}

/// the newtonian-MHD recovery-interior predicate: the newtonian interior on
/// the stripped hydro state plus finite cell-B components.
pub(crate) fn newtonian_mhd_prim_audit<S: Scalar + OrderedNumeric, const D: usize>(
    rho: S,
    pre: S,
    vel: &Tensor<S, D>,
    mag: &Tensor<S, D>,
) -> RecoveryAudit {
    let mut audit = newtonian_prim_audit(rho, pre, vel).0;
    if !tensor_finite(mag) {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NON_FINITE);
    }
    RecoveryAudit(audit)
}

/// the isothermal recovery-interior predicate: positive finite density and
/// finite velocity components. the algebraic `mom / den` inversion has no
/// iterate to diverge, and this is the interior it still owes: a
/// non-positive or non-finite density and the infinite velocity a zero
/// density produces are rejections, never certified recoveries.
pub(crate) fn isothermal_prim_audit<S: Scalar + OrderedNumeric, const D: usize>(
    rho: S,
    vel: &Tensor<S, D>,
) -> RecoveryAudit {
    let mut audit = None;
    if rho <= S::ZERO {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NEGATIVE_DENSITY);
    }
    if !finite(rho) || !tensor_finite(vel) {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NON_FINITE);
    }
    RecoveryAudit(audit)
}

/// the isothermal-MHD recovery-interior predicate: the isothermal interior
/// plus finite cell-B components.
pub(crate) fn isothermal_mhd_prim_audit<S: Scalar + OrderedNumeric, const D: usize>(
    rho: S,
    vel: &Tensor<S, D>,
    mag: &Tensor<S, D>,
) -> RecoveryAudit {
    let mut audit = isothermal_prim_audit(rho, vel).0;
    if !tensor_finite(mag) {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NON_FINITE);
    }
    RecoveryAudit(audit)
}

// the shared relativistic recovery-interior predicate (RHD + RMHD). one source
// so the two regimes' threshold conventions cannot drift (tier-1 #5: the
// density-scaled-vs-absolute pressure floor, the superluminal margin, and the
// input-NaN check had all diverged).
//
// post-hoc flags on the raw recovered state — no silent floor: the caller
// judges the raw recovery value against this audit, which only reports what
// is non-physical. thresholds are dimensionally clean:
//   * NON_FINITE       : rho or pressure is NaN or infinite.
//   * NEGATIVE_PRESSURE: pressure <= 0. a near-zero positive pressure is the
//                        valid cold limit; the zero-pressure boundary is not
//                        in the strict admissible interior used by the flux
//                        and fofc kernels.
//   * superluminal     : v^2 >= 1 (the Lorentz factor is finite only for
//                        v^2 < 1) or v^2 is NaN. no luminal margin.
pub(crate) fn relativistic_c2p_audit<S: Scalar + OrderedNumeric>(
    rho: S,
    pre: S,
    v_sq: S,
) -> RecoveryAudit {
    let mut audit = None;
    if !finite(rho) || !finite(pre) {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NON_FINITE);
    }
    if pre <= S::ZERO {
        audit = RecoveryIssues::note(audit, RecoveryIssues::NEGATIVE_PRESSURE);
    }
    if v_sq >= S::ONE || !(v_sq == v_sq) {
        audit = RecoveryIssues::note(audit, RecoveryIssues::SUPERLUMINAL);
    }
    RecoveryAudit(audit)
}

// shared input-density guard for relativistic c2p (a host-only early-out
// before the kernel path). returns the failure issues for a non-positive or
// non-finite conserved density, else None; the rejection path builds a
// `RecoveryFailure` directly and mints nothing. each flag names its own
// violation: a non-positive density is NEGATIVE_DENSITY, NaN and both
// infinities are NON_FINITE, and a negative infinity carries both.
pub(crate) fn relativistic_density_guard<S: Scalar + OrderedNumeric>(
    dd: S,
) -> Option<RecoveryIssues> {
    let mut issues = None;
    if dd <= S::ZERO {
        issues = RecoveryIssues::note(issues, RecoveryIssues::NEGATIVE_DENSITY);
    }
    if !finite(dd) {
        issues = RecoveryIssues::note(issues, RecoveryIssues::NON_FINITE);
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [RecoveryIssues; 7] = [
        RecoveryIssues::NEGATIVE_DENSITY,
        RecoveryIssues::NEGATIVE_PRESSURE,
        RecoveryIssues::NON_FINITE,
        RecoveryIssues::SUPERLUMINAL,
        RecoveryIssues::MAX_ITER,
        RecoveryIssues::NEGATIVE_ENERGY,
        RecoveryIssues::INVALID_PRIMITIVE,
    ];

    /// every nonempty union of the seven issues is lawful (the predicates
    /// merge freely), and each round-trips exactly through the diagnostic
    /// byte with membership intact.
    #[test]
    fn all_nonempty_unions_round_trip() {
        for bits in 1u8..=ISSUE_MASK {
            let issues =
                RecoveryIssues::from_diagnostic_u8(bits).expect("nonempty in-vocabulary bits");
            assert_eq!(
                issues.to_diagnostic_u8(),
                bits,
                "round-trip of {bits:#010b}"
            );
            for single in ALL {
                assert_eq!(
                    issues.contains(single),
                    bits & single.to_diagnostic_u8() != 0,
                    "membership of {single} in {bits:#010b}"
                );
            }
        }
    }

    /// merge is bitwise union and stays inside the vocabulary.
    #[test]
    fn merge_is_union() {
        for a in 1u8..=ISSUE_MASK {
            for b in 1u8..=ISSUE_MASK {
                let merged = RecoveryIssues::from_diagnostic_u8(a)
                    .unwrap()
                    .merge(RecoveryIssues::from_diagnostic_u8(b).unwrap());
                assert_eq!(merged.to_diagnostic_u8(), a | b);
            }
        }
    }

    /// the empty set and every out-of-vocabulary bit are refused at the
    /// diagnostic boundary.
    #[test]
    fn empty_and_unknown_bits_are_refused() {
        assert_eq!(RecoveryIssues::from_diagnostic_u8(0), None);
        for bits in 0u8..=u8::MAX {
            if bits & 0x80 != 0 {
                assert_eq!(
                    RecoveryIssues::from_diagnostic_u8(bits),
                    None,
                    "bit outside the vocabulary in {bits:#010b}"
                );
            }
        }
    }

    /// the diagnostic byte agrees with the `ErrorCode` layout bit-for-bit.
    #[test]
    fn diagnostic_layout_matches_error_code() {
        use crate::c2p_result::ErrorCode;
        let pairs = [
            (
                RecoveryIssues::NEGATIVE_DENSITY,
                ErrorCode::NEGATIVE_DENSITY,
            ),
            (
                RecoveryIssues::NEGATIVE_PRESSURE,
                ErrorCode::NEGATIVE_PRESSURE,
            ),
            (RecoveryIssues::NON_FINITE, ErrorCode::NON_FINITE),
            (RecoveryIssues::SUPERLUMINAL, ErrorCode::SUPERLUMINAL),
            (RecoveryIssues::MAX_ITER, ErrorCode::MAX_ITER),
            (RecoveryIssues::NEGATIVE_ENERGY, ErrorCode::NEGATIVE_ENERGY),
            (
                RecoveryIssues::INVALID_PRIMITIVE,
                ErrorCode::INVALID_PRIMITIVE,
            ),
        ];
        for (issue, code) in pairs {
            assert_eq!(issue.to_diagnostic_u8(), code.0);
            assert_eq!(ErrorCode::from(issue), code);
        }
    }

    /// `note` folds issues into a clean accumulator; success is the absence
    /// of a set.
    #[test]
    fn note_accumulates_from_clean() {
        let acc = None;
        let acc = RecoveryIssues::note(acc, RecoveryIssues::NEGATIVE_DENSITY);
        let acc = RecoveryIssues::note(acc, RecoveryIssues::NON_FINITE);
        let issues = acc.unwrap();
        assert!(issues.contains(RecoveryIssues::NEGATIVE_DENSITY));
        assert!(issues.contains(RecoveryIssues::NON_FINITE));
        assert!(!issues.contains(RecoveryIssues::SUPERLUMINAL));
    }

    /// a judged clean candidate is `Ok` and yields the value once; a flagged
    /// one carries its issues and only a textual candidate.
    #[test]
    fn judge_separates_the_outcomes() {
        let ok = judge(3.5f64, RecoveryAudit(None)).expect("clean audit mints Recovered");
        assert_eq!(ok.into_inner(), 3.5);
        let err = judge(
            -1.0f64,
            RecoveryAudit(Some(RecoveryIssues::NEGATIVE_PRESSURE)),
        )
        .unwrap_err();
        assert!(err.issues().contains(RecoveryIssues::NEGATIVE_PRESSURE));
        assert_eq!(err.candidate().snapshot(), "-1.0");
    }

    /// the predicates are the audit constructors: the interior passes clean,
    /// each violation carries its issue, and the isothermal audit rejects the
    /// infinite velocity a zero density produces.
    #[test]
    fn named_predicates_construct_the_audits() {
        let v2 = Tensor::new([0.5, -0.5]);
        assert!(judge((), newtonian_prim_audit(1.0f64, 1.0, &v2)).is_ok());
        let f = judge((), newtonian_prim_audit(-1.0f64, 1.0, &v2)).unwrap_err();
        assert!(f.issues().contains(RecoveryIssues::NEGATIVE_DENSITY));

        assert!(judge((), isothermal_prim_audit(1.0f64, &v2)).is_ok());
        let f = judge(
            (),
            isothermal_prim_audit(0.0f64, &Tensor::new([f64::INFINITY, 0.0])),
        )
        .unwrap_err();
        assert!(f.issues().contains(RecoveryIssues::NEGATIVE_DENSITY));
        assert!(f.issues().contains(RecoveryIssues::NON_FINITE));

        let f = judge(
            (),
            isothermal_mhd_prim_audit(
                1.0f64,
                &Tensor::new([0.0, 0.0]),
                &Tensor::new([f64::NAN, 0.0]),
            ),
        )
        .unwrap_err();
        assert!(f.issues().contains(RecoveryIssues::NON_FINITE));

        let f = judge((), relativistic_c2p_audit(1.0f64, 0.0, 0.0)).unwrap_err();
        assert!(f.issues().contains(RecoveryIssues::NEGATIVE_PRESSURE));
        assert!(judge((), relativistic_c2p_audit(1.0f64, f64::MIN_POSITIVE, 0.0)).is_ok());
    }

    /// the finiteness law is uniform: a positive infinity is NON_FINITE in
    /// every state family — scalar slots, velocity components, and cell-B —
    /// where a NaN-only check would certify it.
    #[test]
    fn positive_infinity_is_rejected_in_every_family() {
        let inf = f64::INFINITY;
        let v2 = Tensor::new([0.0, 0.0]);

        for f in [
            judge((), newtonian_prim_audit(inf, 1.0, &v2)).unwrap_err(),
            judge((), newtonian_prim_audit(1.0, inf, &v2)).unwrap_err(),
            judge((), newtonian_prim_audit(1.0, 1.0, &Tensor::new([inf, 0.0]))).unwrap_err(),
            judge(
                (),
                newtonian_mhd_prim_audit(1.0, 1.0, &v2, &Tensor::new([inf, 0.0])),
            )
            .unwrap_err(),
            judge((), isothermal_prim_audit(inf, &v2)).unwrap_err(),
            judge(
                (),
                isothermal_mhd_prim_audit(1.0, &v2, &Tensor::new([inf, 0.0])),
            )
            .unwrap_err(),
            judge((), relativistic_c2p_audit(inf, 1.0, 0.0)).unwrap_err(),
            judge((), relativistic_c2p_audit(1.0, inf, 0.0)).unwrap_err(),
        ] {
            assert!(f.issues().contains(RecoveryIssues::NON_FINITE));
        }
        // an infinite velocity norm exceeds one, so the superluminal flag
        // carries it; the subluminal branch is finite by construction.
        let f = judge((), relativistic_c2p_audit(1.0, 1.0, f64::INFINITY)).unwrap_err();
        assert!(f.issues().contains(RecoveryIssues::SUPERLUMINAL));
    }

    /// the density guard names each violation: a non-positive density is
    /// NEGATIVE_DENSITY, NaN and both infinities are NON_FINITE, a negative
    /// infinity carries both, and a positive finite density passes.
    #[test]
    fn density_guard_flags_each_violation() {
        let f = relativistic_density_guard(-1.0f64).expect("negative density is rejected");
        assert!(f.contains(RecoveryIssues::NEGATIVE_DENSITY));
        assert!(!f.contains(RecoveryIssues::NON_FINITE));

        let f = relativistic_density_guard(0.0f64).expect("zero density is rejected");
        assert!(f.contains(RecoveryIssues::NEGATIVE_DENSITY));

        let f = relativistic_density_guard(f64::NAN).expect("NaN density is rejected");
        assert!(f.contains(RecoveryIssues::NON_FINITE));
        assert!(!f.contains(RecoveryIssues::NEGATIVE_DENSITY));

        let f = relativistic_density_guard(f64::INFINITY).expect("+inf density is rejected");
        assert!(f.contains(RecoveryIssues::NON_FINITE));
        assert!(!f.contains(RecoveryIssues::NEGATIVE_DENSITY));

        let f = relativistic_density_guard(f64::NEG_INFINITY).expect("-inf density is rejected");
        assert!(f.contains(RecoveryIssues::NON_FINITE));
        assert!(f.contains(RecoveryIssues::NEGATIVE_DENSITY));

        assert!(relativistic_density_guard(1.0_f64).is_none());
    }

    /// the placeholder failure renders its own explanation.
    #[test]
    fn placeholder_snapshot_names_the_early_rejection() {
        let failure = RecoveryFailure::<f64>::without_candidate(RecoveryIssues::NEGATIVE_DENSITY);
        assert!(failure.candidate().snapshot().contains("before recovery"));
    }
}
