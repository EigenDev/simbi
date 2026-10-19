// =============================================================================
// step_transaction.rs
//
// the acceptance of one root step, the same on every march. a driver supplies
// the mechanics of an attempt through `RootAttempt`; `take_root_step` owns every
// decision about it:
// - each attempt captures the step entry, then runs at its timestep; the
//   capture covers everything a restore reads whenever the step can end in a
//   restore: the attempt can be rejected, or the guard refuses a step accepted
//   at its full candidate;
// - a rejected attempt books its evidence in the retry ledger, discards its
//   pending ledger acts, restores the entry and replays at half the timestep;
// - a completed attempt meets the sustained-reduction guard before anything
//   commits; an admitted step then commits its pending ledger acts and the
//   streak, and the driver's accepted-step tail and publication follow;
// - a step the guard ends (a rejection past the limit, a sustained reduction)
//   or whose halved timestep can no longer advance time discards its pending
//   acts and restores the entry: the state it leaves is the last accepted one,
//   with that state's clock, and the halt carries the guard's evidence;
// - on a march whose attempt runs in this thread (`contains_panics`), an attempt
//   that panics (the freeze-streak halt, a failed assertion) ends the step the
//   same way: its pending acts are discarded, the entry restored, and the halt
//   carries the panic message, which names the failing cell and its state.
//
// usage:
//  match take_root_step(&mut attempt, &guard, candidate, time)? {
//      RootStep::Accepted { dt } => { /* accepted-step tail, publication */ }
//      RootStep::Halted(halt) => { /* the driver's halt report */ }
//  }
// =============================================================================

use crate::retry_guard::{RetryGuard, RetryGuardFailure, RetryLedger, accept};

/// the mechanics of one root step's attempts on one march
pub trait RootAttempt {
    type Error;
    /// whether an attempt can be rejected: a rejection source of this march is armed
    fn rejectable(&mut self) -> bool;
    /// capture the step entry for an attempt at `dt`, completely when `restorable`: a restore
    /// reads every field the capture keeps
    fn capture(&mut self, dt: f64, restorable: bool) -> Result<(), Self::Error>;
    /// run the attempt at `dt`: true when it completed, false when a rejection source fired
    fn run(&mut self, dt: f64) -> Result<bool, Self::Error>;
    /// the freeze evidence the rejected attempt left, numbered after `rejections` earlier ones
    fn evidence(&mut self, rejections: u32) -> Vec<String>;
    /// return every field, side-car, clock and halo to the captured step entry
    fn restore(&mut self) -> Result<(), Self::Error>;
    /// the consecutive severely reduced root steps the run carries
    fn severe_streak(&mut self) -> u32;
    /// whether a panic inside the attempt can end the step at its restored entry: every store
    /// lives in this thread and the restore covers its whole state. a march whose peers run
    /// elsewhere, or whose restore leaves state out, lets the panic propagate
    fn contains_panics(&self) -> bool {
        false
    }
    fn set_severe_streak(&mut self, streak: u32);
}

/// one operation of a root attempt, asked of a march whose attempt mechanics share one scope
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttemptOp {
    Rejectable,
    Capture { dt: f64, restorable: bool },
    Run { dt: f64 },
    Evidence { rejections: u32 },
    Restore,
    SevereStreak,
    SetSevereStreak(u32),
}

/// the answer to an `AttemptOp`: the flag of `Rejectable` and `Run`, the evidence lines, the
/// streak, or completion
#[derive(Clone, Debug, PartialEq)]
pub enum AttemptReply {
    Done,
    Flag(bool),
    Evidence(Vec<String>),
    Streak(u32),
}

/// a `RootAttempt` answered by one closure over the march's scope: the march's mechanics stay
/// beside the locals and macros they read, and every decision stays with `take_root_step`
pub struct ScopedAttempt<F>(pub F);

impl<E, F: FnMut(AttemptOp) -> Result<AttemptReply, E>> ScopedAttempt<F> {
    fn ask(&mut self, op: AttemptOp) -> Result<AttemptReply, E> {
        (self.0)(op)
    }
}

impl<E, F: FnMut(AttemptOp) -> Result<AttemptReply, E>> RootAttempt for ScopedAttempt<F> {
    type Error = E;

    fn rejectable(&mut self) -> bool {
        matches!(
            self.ask(AttemptOp::Rejectable),
            Ok(AttemptReply::Flag(true))
        )
    }

    fn capture(&mut self, dt: f64, restorable: bool) -> Result<(), E> {
        self.ask(AttemptOp::Capture { dt, restorable }).map(|_| ())
    }

    fn run(&mut self, dt: f64) -> Result<bool, E> {
        match self.ask(AttemptOp::Run { dt })? {
            AttemptReply::Flag(completed) => Ok(completed),
            other => panic!("a run answered {other:?}"),
        }
    }

    fn evidence(&mut self, rejections: u32) -> Vec<String> {
        match self.ask(AttemptOp::Evidence { rejections }) {
            Ok(AttemptReply::Evidence(lines)) => lines,
            Ok(other) => panic!("an evidence request answered {other:?}"),
            Err(_) => Vec::new(),
        }
    }

    fn restore(&mut self) -> Result<(), E> {
        self.ask(AttemptOp::Restore).map(|_| ())
    }

    fn severe_streak(&mut self) -> u32 {
        match self.ask(AttemptOp::SevereStreak) {
            Ok(AttemptReply::Streak(streak)) => streak,
            other => panic!("a streak request answered {:?}", other.ok()),
        }
    }

    fn set_severe_streak(&mut self, streak: u32) {
        let _ = self.ask(AttemptOp::SetSevereStreak(streak));
    }
}

/// why a root step ended the run
#[derive(Clone, Debug, PartialEq)]
pub enum StepHalt {
    /// the retry guard ended it: a rejection past the limit or a sustained reduction
    Guard(RetryGuardFailure),
    /// the rejected timestep halved can no longer advance the clock
    Timestep { dt: f64, detail: String },
    /// the attempt panicked: a failure no replay at this step resolves, with the panic message
    Unrecoverable { detail: String },
}

impl StepHalt {
    pub fn detail(&self) -> &str {
        match self {
            StepHalt::Guard(failure) => &failure.detail,
            StepHalt::Timestep { detail, .. } => detail,
            StepHalt::Unrecoverable { detail } => detail,
        }
    }
}

/// how a root step ended
#[derive(Clone, Debug, PartialEq)]
pub enum RootStep {
    /// the step advanced by `dt` and its pending acts are committed
    Accepted { dt: f64 },
    /// the run ends at the step entry
    Halted(StepHalt),
}

fn discard_pending() {
    if crate::projection_ledger::scope_is_open() {
        crate::projection_ledger::step_discard();
    }
    if crate::guard_ledger::scope_is_open() {
        crate::guard_ledger::step_discard();
    }
}

/// the text of a panic payload
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&'static str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic with a non-string payload".to_string())
}

fn commit_pending() {
    if crate::projection_ledger::scope_is_open() {
        crate::projection_ledger::step_commit();
    }
    if crate::guard_ledger::scope_is_open() {
        crate::guard_ledger::step_commit();
    }
}

/// one root step from its candidate timestep, the timestep its first attempt runs at after the
/// cfl agreement and every cap, at the step entry `time`
pub fn take_root_step<A: RootAttempt>(
    attempt: &mut A,
    guard: &RetryGuard,
    candidate: f64,
    time: f64,
) -> Result<RootStep, A::Error> {
    let mut ledger = RetryLedger::new(candidate);
    let mut dt = candidate;
    let restorable = attempt.rejectable() || guard.refuses_unreduced_steps();
    loop {
        attempt.capture(dt, restorable)?;
        let completed = if restorable && attempt.contains_panics() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| attempt.run(dt))) {
                Ok(completed) => completed?,
                Err(payload) => {
                    discard_pending();
                    attempt.restore()?;
                    return Ok(RootStep::Halted(StepHalt::Unrecoverable { detail: panic_message(&*payload) }));
                }
            }
        } else {
            attempt.run(dt)?
        };
        if completed {
            break;
        }
        let evidence = attempt.evidence(ledger.rejections());
        let refused = ledger.reject(guard, dt, evidence).err();
        discard_pending();
        attempt.restore()?;
        if let Some(failure) = refused {
            return Ok(RootStep::Halted(StepHalt::Guard(failure)));
        }
        dt = match crate::driver::retry_timestep(dt, time) {
            Ok(retry) => retry,
            Err(err) => {
                return Ok(RootStep::Halted(StepHalt::Timestep {
                    dt,
                    detail: err.detail,
                }));
            }
        };
    }
    let mut streak = attempt.severe_streak();
    if let Err(failure) = accept(guard, &mut streak, &ledger, dt) {
        discard_pending();
        attempt.restore()?;
        return Ok(RootStep::Halted(StepHalt::Guard(failure)));
    }
    attempt.set_severe_streak(streak);
    commit_pending();
    Ok(RootStep::Accepted { dt })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard_ledger::{self, GuardReceipt};

    /// a scripted march: one integer of state, each completed attempt adds one, and the attempts
    /// listed in `rejected` fire a rejection after booking a guard act and dirtying the state
    struct Scripted {
        entry: i64,
        state: i64,
        attempts: usize,
        rejected: Vec<usize>,
        streak: u32,
        log: Vec<String>,
        restorable: bool,
        /// the attempt that panics after dirtying the state, and whether the march contains it
        panics_at: Option<usize>,
        contained: bool,
    }

    impl Scripted {
        fn new(rejected: Vec<usize>) -> Self {
            Self {
                entry: 0,
                state: 0,
                attempts: 0,
                rejected,
                streak: 0,
                log: Vec::new(),
                restorable: false,
                panics_at: None,
                contained: false,
            }
        }
    }

    impl RootAttempt for Scripted {
        type Error = String;
        fn rejectable(&mut self) -> bool {
            !self.rejected.is_empty()
        }
        fn capture(&mut self, dt: f64, restorable: bool) -> Result<(), String> {
            self.restorable = restorable;
            self.entry = self.state;
            self.log.push(format!("capture {dt}"));
            Ok(())
        }
        fn run(&mut self, dt: f64) -> Result<bool, String> {
            self.attempts += 1;
            guard_ledger::record(&GuardReceipt::of_pass(1, 0, 0, 0));
            self.state += 100;
            self.log.push(format!("run {dt}"));
            if self.panics_at == Some(self.attempts) {
                panic!("attempt {} failed", self.attempts);
            }
            if self.rejected.contains(&self.attempts) {
                return Ok(false);
            }
            self.state = self.entry + 1;
            Ok(true)
        }
        fn contains_panics(&self) -> bool {
            self.contained
        }
        fn evidence(&mut self, rejections: u32) -> Vec<String> {
            vec![format!("attempt {} after {rejections}", self.attempts)]
        }
        fn restore(&mut self) -> Result<(), String> {
            assert!(self.restorable, "restored from an incomplete capture");
            self.state = self.entry;
            self.log.push("restore".into());
            Ok(())
        }
        fn severe_streak(&mut self) -> u32 {
            self.streak
        }
        fn set_severe_streak(&mut self, streak: u32) {
            self.streak = streak;
        }
    }

    const GUARD: RetryGuard = RetryGuard {
        max_rejections: 2,
        severe_ratio: 0.25,
        sustained_steps: 1,
    };

    #[test]
    fn a_rejected_attempt_replays_from_the_entry_and_only_the_accepted_attempt_commits() {
        let scope = guard_ledger::open_scope();
        let mut march = Scripted::new(vec![1]);
        let step = take_root_step(&mut march, &GUARD, 1.0, 0.0).unwrap();
        assert_eq!(step, RootStep::Accepted { dt: 0.5 });
        assert_eq!(
            march.state, 1,
            "the replay started from the dirtied attempt"
        );
        assert_eq!(
            march.log,
            ["capture 1", "run 1", "restore", "capture 0.5", "run 0.5"]
        );
        assert_eq!(
            march.streak, 0,
            "a step at half its candidate is not severely reduced under 1/4"
        );
        assert_eq!(
            scope.accepted().troubled_cells.total,
            1,
            "the rejected attempt's act was accepted"
        );
    }

    #[test]
    fn a_step_the_guard_refuses_restores_the_entry_and_commits_nothing() {
        let scope = guard_ledger::open_scope();
        let mut march = Scripted::new(vec![1, 2]);
        let step = take_root_step(&mut march, &GUARD, 1.0, 0.0).unwrap();
        let RootStep::Halted(StepHalt::Guard(failure)) = step else {
            panic!("the reduction to 1/4 was admitted: {step:?}")
        };
        assert!(
            failure
                .detail
                .contains("accepted dt=2.5e-1 after 2 rejections"),
            "{}",
            failure.detail
        );
        assert_eq!(march.state, 0, "the refused step left its completed state");
        assert_eq!(march.log.last().map(String::as_str), Some("restore"));
        assert_eq!(march.streak, 0, "a refused step advanced the streak");
        assert_eq!(
            scope.accepted().troubled_cells.total,
            0,
            "the refused step's acts were accepted"
        );
    }

    #[test]
    fn a_rejection_past_the_limit_restores_the_entry_and_halts_with_every_rejection() {
        let scope = guard_ledger::open_scope();
        let mut march = Scripted::new(vec![1, 2, 3]);
        let step = take_root_step(&mut march, &GUARD, 1.0, 0.0).unwrap();
        let RootStep::Halted(StepHalt::Guard(failure)) = step else {
            panic!("a third rejection was admitted: {step:?}")
        };
        for attempt in 1..=3 {
            assert!(
                failure
                    .detail
                    .contains(&format!("attempt {attempt} after {}", attempt - 1)),
                "{}",
                failure.detail
            );
        }
        assert_eq!(march.state, 0);
        assert_eq!(march.attempts, 3, "an attempt ran past the limit");
        assert_eq!(scope.accepted().troubled_cells.total, 0);
    }

    #[test]
    fn a_timestep_that_can_no_longer_advance_time_restores_the_entry_and_halts() {
        let scope = guard_ledger::open_scope();
        let mut march = Scripted::new(vec![1]);
        let step = take_root_step(&mut march, &RetryGuard::default(), 1.0e-20, 1.0).unwrap();
        let RootStep::Halted(StepHalt::Timestep { dt, .. }) = step else {
            panic!("the vanishing retry ran: {step:?}")
        };
        assert_eq!(dt, 1.0e-20);
        assert_eq!(march.state, 0);
        assert_eq!(march.attempts, 1);
        assert_eq!(scope.accepted().troubled_cells.total, 0);
    }

    #[test]
    fn a_guard_that_refuses_unreduced_steps_captures_a_march_without_a_rejection_source() {
        let scope = guard_ledger::open_scope();
        let mut march = Scripted::new(vec![]);
        let every_step = RetryGuard {
            max_rejections: 20,
            severe_ratio: 1.0,
            sustained_steps: 1,
        };
        let step = take_root_step(&mut march, &every_step, 1.0, 0.0).unwrap();
        assert!(
            matches!(step, RootStep::Halted(StepHalt::Guard(_))),
            "{step:?}"
        );
        assert_eq!(march.state, 0, "the refused step left its completed state");
        assert_eq!(scope.accepted().troubled_cells.total, 0);
    }

    #[test]
    fn a_contained_panic_restores_the_entry_and_halts_with_its_message() {
        let scope = guard_ledger::open_scope();
        let mut march = Scripted::new(vec![1]);
        march.panics_at = Some(2);
        march.contained = true;
        let step = take_root_step(&mut march, &GUARD, 1.0, 0.0).unwrap();
        assert_eq!(step, RootStep::Halted(StepHalt::Unrecoverable { detail: "attempt 2 failed".into() }));
        assert_eq!(march.state, 0, "the halt left the panicked attempt's state");
        assert_eq!(march.log.last().map(String::as_str), Some("restore"), "the entry was not restored");
        assert_eq!(scope.accepted().troubled_cells.total, 0, "the panicked step's acts were accepted");
    }

    #[test]
    fn a_panic_on_a_march_that_does_not_contain_it_propagates() {
        let mut march = Scripted::new(Vec::new());
        march.panics_at = Some(1);
        march.rejected = vec![2];
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| take_root_step(&mut march, &GUARD, 1.0, 0.0)));
        assert!(caught.is_err(), "a march whose peers run elsewhere had its panic contained");
    }
}
