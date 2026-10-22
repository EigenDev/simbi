// =============================================================================
// retry_guard.rs
//
// the retry safeguards of a run: a bound on the rejections one root step may
// take and a bound on how many consecutive accepted root steps may be severely
// reduced. both are operational limits of the retry loop, not stability
// thresholds of the scheme; a run that crosses one ends with a report, and no
// step is ever accepted in place of a failure.
// - the candidate of a root step is the timestep its first attempt runs at,
//   after the cfl selection, the user cap and the final-time clamp; each
//   rejection halves it;
// - a step is severely reduced when accepted / candidate <= severe_ratio;
// - the rejection past `max_rejections` fails during the attempt loop; the
//   `sustained_steps`-th consecutive severely reduced step fails before the
//   accepted state is published;
// - each rejection records its evidence: the cells whose freeze acts counted
//   toward it, with the tile, level, stage and attempt that produced them.
//
// the step transaction (`step_transaction::take_root_step`) is the one caller
// on every march.
//
// usage:
//  let mut ledger = RetryLedger::new(candidate);
//  ledger.reject(&guard, dt, evidence)?;
//  accept(&guard, &mut streak, &ledger, accepted)?;
// =============================================================================

/// the retry safeguards of a run; `Default` is the built-in configuration, a run may override
/// each field
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryGuard {
    /// the rejections one root step may take; the next one ends the run
    pub max_rejections: u32,
    /// a root step accepted at or below this fraction of its candidate is severely reduced
    pub severe_ratio: f64,
    /// the consecutive severely reduced root steps that end the run
    pub sustained_steps: u32,
}

impl Default for RetryGuard {
    fn default() -> Self {
        Self { max_rejections: 20, severe_ratio: 1.0 / 16.0, sustained_steps: 8 }
    }
}

impl RetryGuard {
    /// whether a root step accepted at its full candidate counts as severely reduced, so the
    /// guard can refuse a step no rejection reduced
    pub fn refuses_unreduced_steps(&self) -> bool {
        1.0 <= self.severe_ratio
    }

    /// the thresholds in words, for the run's opening report and every failure
    pub fn describe(&self) -> String {
        format!(
            "retry guard: at most {} rejections per root step; a root step accepted at or below {} of its \
             candidate timestep is severely reduced; {} consecutive severely reduced root steps end the run",
            self.max_rejections, self.severe_ratio, self.sustained_steps
        )
    }
}

/// a run ended by the retry guard
#[derive(Clone, Debug, PartialEq)]
pub struct RetryGuardFailure {
    pub detail: String,
}

impl std::fmt::Display for RetryGuardFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for RetryGuardFailure {}

/// the rejections of one root step's attempts and the evidence each carried
#[derive(Clone, Debug, Default)]
pub struct RetryLedger {
    candidate: f64,
    rejections: u32,
    evidence: Vec<String>,
}

impl RetryLedger {
    /// a root step whose first attempt runs at `candidate`
    pub fn new(candidate: f64) -> Self {
        Self { candidate, rejections: 0, evidence: Vec::new() }
    }

    pub fn candidate(&self) -> f64 {
        self.candidate
    }

    pub fn rejections(&self) -> u32 {
        self.rejections
    }

    /// book the rejection of the attempt that ran at `dt`, with the evidence it carried; the
    /// rejection past the guard's limit fails with every rejection's evidence
    pub fn reject(&mut self, guard: &RetryGuard, dt: f64, evidence: Vec<String>) -> Result<(), RetryGuardFailure> {
        self.rejections += 1;
        if evidence.is_empty() {
            self.evidence.push(format!("rejection {} at dt={dt:e}: no freeze evidence held locally", self.rejections));
        } else {
            self.evidence.extend(evidence.into_iter().map(|e| format!("rejection {} at dt={dt:e}: {e}", self.rejections)));
        }
        if self.rejections > guard.max_rejections {
            return Err(RetryGuardFailure {
                detail: format!(
                    "retry guard: the root step rejected {} times, past the limit of {}; candidate dt={:e}, last \
                     attempt dt={dt:e}.\n{}\n{}",
                    self.rejections,
                    guard.max_rejections,
                    self.candidate,
                    guard.describe(),
                    self.evidence.join("\n")
                ),
            });
        }
        Ok(())
    }
}

/// the accepted root step: `streak` counts the consecutive severely reduced root steps and resets
/// on one that is not; the step that brings it to the guard's limit fails, before its state is
/// published, with the step's evidence
pub fn accept(guard: &RetryGuard, streak: &mut u32, ledger: &RetryLedger, accepted: f64) -> Result<(), RetryGuardFailure> {
    let ratio = accepted / ledger.candidate;
    if ratio <= guard.severe_ratio {
        *streak += 1;
    } else {
        *streak = 0;
    }
    if *streak >= guard.sustained_steps {
        return Err(RetryGuardFailure {
            detail: format!(
                "retry guard: {} consecutive root steps accepted at or below {} of their candidate timestep; this \
                 step's candidate dt={:e}, accepted dt={accepted:e} after {} rejections.\n{}\n{}",
                *streak,
                guard.severe_ratio,
                ledger.candidate,
                ledger.rejections,
                guard.describe(),
                ledger.evidence.join("\n")
            ),
        });
    }
    Ok(())
}

/// the guard's opening report: the thresholds in force, the streak a resumed run carries, and,
/// when the checkpoint recorded other thresholds, both sets with the rule that the configured ones
/// apply and the recorded streak carries over
pub fn opening_report(configured: &RetryGuard, recorded: Option<&RetryGuard>, streak: u32) -> String {
    let mut report = configured.describe();
    if streak > 0 {
        report.push_str(&format!("; resumed with {streak} consecutive severely reduced root steps"));
    }
    if let Some(recorded) = recorded.filter(|r| *r != configured) {
        report.push_str(&format!(
            "\nretry guard changed on restart: recorded max_rejections={} severe_ratio={} sustained_steps={}; \
             configured max_rejections={} severe_ratio={} sustained_steps={}; the configured thresholds apply \
             and the recorded streak of {streak} carries over",
            recorded.max_rejections,
            recorded.severe_ratio,
            recorded.sustained_steps,
            configured.max_rejections,
            configured.severe_ratio,
            configured.sustained_steps
        ));
    }
    report
}

/// the rejection evidence a store holds, taken, each line labeled with `site` (the worker, tile
/// and level that produced it) and the attempt; the slot is empty afterwards
pub fn take_evidence<const D: usize, const DOF: usize, Mem, Sc>(
    store: &crate::state::FieldStore<D, DOF, Mem, Sc>,
    site: &str,
    attempt: u32,
) -> Vec<String>
where
    Mem: symbi_xpu::MemoryStorage,
    Sc: symbi_carrier::Scalar + symbi_algebra::OrderedNumeric,
{
    let lines = std::mem::take(&mut *store.workspace.rejection_evidence.lock().expect("rejection evidence"));
    lines.into_iter().map(|line| format!("{site}, attempt {attempt}, {line}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn halved(candidate: f64, k: u32) -> f64 {
        candidate / 2f64.powi(k as i32)
    }

    #[test]
    fn the_limit_admits_twenty_rejections_and_fails_the_twenty_first() {
        let guard = RetryGuard::default();
        let mut ledger = RetryLedger::new(1.0);
        for k in 1..=20 {
            ledger.reject(&guard, halved(1.0, k - 1), vec![]).expect("within the limit");
        }
        let failure = ledger.reject(&guard, halved(1.0, 20), vec!["cell [3, 4]".into()]).unwrap_err();
        assert!(failure.detail.contains("rejected 21 times, past the limit of 20"), "{}", failure.detail);
        assert!(failure.detail.contains("cell [3, 4]") && failure.detail.contains("retry guard: at most 20"));
    }

    #[test]
    fn eight_consecutive_severe_steps_fail_and_seven_do_not() {
        let guard = RetryGuard::default();
        let mut streak = 0;
        // four halvings: accepted / candidate == 1/16 exactly, severe
        let ledger = RetryLedger::new(0.3);
        for _ in 0..7 {
            accept(&guard, &mut streak, &ledger, halved(0.3, 4)).expect("seven are within the limit");
        }
        assert_eq!(streak, 7);
        let failure = accept(&guard, &mut streak, &ledger, halved(0.3, 4)).unwrap_err();
        assert!(failure.detail.contains("8 consecutive"), "{}", failure.detail);
    }

    #[test]
    fn a_milder_step_resets_the_streak() {
        let guard = RetryGuard::default();
        let mut streak = 0;
        let ledger = RetryLedger::new(1.0);
        for _ in 0..7 {
            accept(&guard, &mut streak, &ledger, halved(1.0, 4)).unwrap();
        }
        // three halvings: 1/8 is above 1/16
        accept(&guard, &mut streak, &ledger, halved(1.0, 3)).unwrap();
        assert_eq!(streak, 0);
        for _ in 0..7 {
            accept(&guard, &mut streak, &ledger, halved(1.0, 4)).unwrap();
        }
        assert_eq!(streak, 7);
    }
}

/// refuse a run whose stores carry different retry exclusions: the decision of every tile and level
/// reads its own store, so stores that disagree would accept and reject different steps of one run
pub fn assert_one_retry_exclusion<'a>(exclusions: impl IntoIterator<Item = &'a crate::substrate_seam::RetryExclusion>) {
    let mut exclusions = exclusions.into_iter();
    let Some(first) = exclusions.next() else { return };
    for (ii, other) in exclusions.enumerate() {
        assert!(
            other == first,
            "the stores of one run carry different retry exclusions: store 0 {first}, store {} {other}; \
             set the run's policy on every tile and level",
            ii + 1
        );
    }
}
