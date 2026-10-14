// =============================================================================
// grant.rs
//
// the receiver-grant table: for every peer, the phases that peer has declared
// itself ready to receive from this worker, each consumed exactly once when
// this worker opens that phase. the table holds one fixed slot per axis per
// peer; a peer cannot be more than one exchange point ahead, so every tabled
// grant belongs to one exchange point and at most one per axis is live.
//
// usage:
//  table.table(peer, epoch, count)?;        // a Grant frame arrived
//  table.open(epoch, |peer| in_phase(peer))?; // this worker opens the phase
//  let count = table.consume(peer, epoch)?; // the grant is spent
// =============================================================================

use crate::error::FabricError;
use crate::ident::{Epoch, WorkerId};

/// the legal sequence of phases of a session, as the phase table states it, indexed by the
/// phase byte: `skippable` marks a phase a worker may pass without opening (the troubled
/// exchange of a clean stage, an audit when audits are off, the post-step when no recovery
/// wrote the interior), `init_only` a phase of the first attempt of the first step alone
/// (the prime and its audit), and the named bytes anchor the transitions: an attempt is
/// accepted once its sequence reaches `accept` (the last phase every worker opens), a
/// retry follows a `rollback`, the `rollback_audit` follows the rollback alone, and the
/// `checkpoint_audit` runs at the checkpoint epoch (attempt `u16::MAX`) of a step whose
/// attempt was accepted, or of the primed state before any step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseRules {
    pub skippable: Vec<bool>,
    pub init_only: Vec<bool>,
    /// the points whose exchanges repeat: each carries its pass (`round`) and the exchange within
    /// the pass (`op`), which rise through the point; every other point carries zero in both
    pub iterative: Vec<bool>,
    pub accept: usize,
    pub post_step: usize,
    pub rollback: usize,
    pub rollback_audit: usize,
    pub checkpoint_audit: usize,
}

/// the checkpoint epoch's attempt: the audit before a checkpoint sorts after every exchange
/// of its step and before the next step's
pub const CHECKPOINT_ATTEMPT: u16 = u16::MAX;

impl PhaseRules {
    pub fn len(&self) -> usize {
        self.skippable.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skippable.is_empty()
    }

    fn passable(&self, byte: usize, owed: &impl Fn(usize) -> bool) -> bool {
        self.skippable[byte] || self.init_only[byte] || !owed(byte)
    }

    /// every phase in `range` a worker must open and owes: none may be passed
    fn no_owed_phase_passed(
        &self,
        range: std::ops::Range<usize>,
        owed: &impl Fn(usize) -> bool,
        next: Epoch,
    ) -> Result<(), String> {
        for b in range {
            if !self.passable(b, owed) {
                return Err(format!("{next:?} passes the phase {b}, which every worker opens"));
            }
        }
        Ok(())
    }

    /// whether `next` may follow `last` in one worker's sequence. `owed(byte)` says whether
    /// the worker owes this observer a grant at that phase: a peer's sequence is observed
    /// through the grants it owes, so a phase in which it receives nothing leaves no trace
    /// and passes unjudged, while a worker's own opens are all owed. `strict_axes` names
    /// the axis count for a worker's own sequence: a phase opens its axes in order from the
    /// first and leaves for another phase from the last alone; a peer's grants name the
    /// axes it owes, so for a peer the axes rise and nothing more. the transitions: inside
    /// one attempt the phases rise, the rollback opens from any phase before the post-step,
    /// its audit from the rollback alone; a retry (the next attempt of the step) follows a
    /// rollback; the next step follows an accepted attempt; the checkpoint audit runs at
    /// the checkpoint epoch after an accepted attempt, and the next step follows it. the
    /// primed state is checkpointed without an audit (the prime audit certified it), so no
    /// transition leaves the prime except the first attempt of the first step.
    pub fn allows(
        &self,
        last: Option<Epoch>,
        next: Epoch,
        owed: impl Fn(usize) -> bool,
        strict_axes: Option<usize>,
    ) -> Result<(), String> {
        let byte = next.point as usize;
        if byte >= self.len() {
            return Err(format!("phase byte {} outside the table of {} phases", next.point, self.len()));
        }
        if !self.iterative[byte] && (next.round, next.op) != (0, 0) {
            return Err(format!("{next:?} carries a pass at a point that does not repeat"));
        }
        if self.init_only[byte] && (next.step, next.attempt) != (0, 0) {
            return Err(format!("{next:?} names a phase of the first attempt alone"));
        }
        if (byte == self.checkpoint_audit) != (next.attempt == CHECKPOINT_ATTEMPT) {
            return Err(format!("{next:?}: the checkpoint audit and the checkpoint epoch go together"));
        }
        let owed = &owed;
        // the phases of a fresh attempt before `upto`: a rollback opens from any phase, so
        // it may be the first phase observed of an attempt; its audit then owes the rollback
        let prefix = |upto: usize| {
            if upto == self.rollback {
                return Ok(());
            }
            if upto == self.rollback_audit {
                return if owed(self.rollback) {
                    Err(format!("{next:?} audits a rollback that was never opened"))
                } else {
                    Ok(())
                };
            }
            self.no_owed_phase_passed(0..upto, owed, next)
        };
        if let Some(axes) = strict_axes {
            let same_phase = last.is_some_and(|l| l.same_exchange(&next));
            if same_phase {
                let l = last.unwrap();
                if next.axis != l.axis + 1 {
                    return Err(format!("{next:?} skips an axis after {l:?}"));
                }
            } else {
                if let Some(l) = last {
                    if l.axis as usize + 1 != axes {
                        return Err(format!("{next:?} leaves {l:?} before its last axis"));
                    }
                }
                if next.axis != 0 {
                    return Err(format!("{next:?} opens a phase past its first axis"));
                }
            }
        }
        let Some(l) = last else {
            return prefix(byte);
        };
        let lb = l.point as usize;
        let same_attempt = (l.step, l.attempt) == (next.step, next.attempt);
        if same_attempt && lb == byte {
            // inside one point the exchanges rise by pass, then by exchange, then by axis
            return match (next.round, next.op, next.axis).cmp(&(l.round, l.op, l.axis)) {
                std::cmp::Ordering::Greater => Ok(()),
                _ if l.same_exchange(&next) => Err(format!("{next:?} repeats or reverses the axis of {l:?}")),
                _ => Err(format!("{next:?} moves backward from {l:?} inside its point")),
            };
        }
        // the previous attempt ended in acceptance: it left the primed state, nothing owed
        // between its last phase and the accept point was passed, and it did not roll back
        let accepted = |l: Epoch| -> Result<(), String> {
            if lb == self.rollback || lb == self.rollback_audit {
                return Err(format!("{next:?} follows the rollback {l:?} without a retry"));
            }
            if self.init_only[lb] {
                return Err(format!("{next:?} leaves the primed state {l:?} without the first attempt"));
            }
            self.no_owed_phase_passed(lb + 1..self.accept + 1, owed, next)
        };
        if next.attempt == CHECKPOINT_ATTEMPT {
            if l.attempt == CHECKPOINT_ATTEMPT || l.step != next.step {
                return Err(format!("{next:?} audits a checkpoint of another step than {l:?}"));
            }
            return accepted(l);
        }
        if same_attempt {
            if byte < lb {
                return Err(format!("{next:?} moves backward from {l:?}"));
            }
            if byte == self.rollback {
                return if lb < self.post_step {
                    Ok(())
                } else {
                    Err(format!("{next:?} rolls back after {l:?} accepted the step"))
                };
            }
            if byte == self.rollback_audit {
                return if lb == self.rollback || !owed(self.rollback) {
                    Ok(())
                } else {
                    Err(format!("{next:?} audits a rollback that {l:?} did not open"))
                };
            }
            if lb == self.rollback || lb == self.rollback_audit {
                return Err(format!("{next:?} continues the attempt after the rollback {l:?}"));
            }
            return self.no_owed_phase_passed(lb + 1..byte, owed, next);
        }
        if l.attempt == CHECKPOINT_ATTEMPT {
            return if next.step == l.step + 1 && next.attempt == 0 {
                prefix(byte)
            } else {
                Err(format!("{next:?} does not start the step after the checkpoint {l:?}"))
            };
        }
        if next.step == l.step && next.attempt == l.attempt + 1 {
            if lb == self.rollback || lb == self.rollback_audit || !owed(self.rollback) {
                return prefix(byte);
            }
            return Err(format!("{next:?} retries the step without a rollback after {l:?}"));
        }
        if next.step == l.step + 1 && next.attempt == 0 {
            accepted(l)?;
            return prefix(byte);
        }
        // a peer owing this observer no grant at the rollback or its audit leaves no trace of a
        // rejected attempt in which it owed nothing else: its next grant may belong to a later
        // attempt of the same step, or to a retry of the step after an accepted one
        let silent_rollback = !owed(self.rollback) && !owed(self.rollback_audit);
        if silent_rollback && next.step == l.step && next.attempt > l.attempt + 1 {
            return prefix(byte);
        }
        if silent_rollback && next.step == l.step + 1 && next.attempt > 0 {
            accepted(l)?;
            return prefix(byte);
        }
        Err(format!("{next:?} does not follow {l:?}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tabled {
    epoch: Epoch,
    count: u32,
}

#[derive(Debug)]
pub struct GrantTable {
    axes: usize,
    slots: Vec<Option<Tabled>>,
    /// the most recent epoch consumed from each peer
    consumed: Vec<Option<Epoch>>,
    /// the last grant tabled from each peer, the phase its sequence has reached
    tabled_last: Vec<Option<Epoch>>,
    /// the most grants held at once from each peer over the session
    high_water: Vec<usize>,
}

fn same_point(a: &Epoch, b: &Epoch) -> bool {
    a.same_exchange(b)
}

impl GrantTable {
    pub fn new(workers: usize, axes: usize) -> Self {
        Self {
            axes,
            slots: vec![None; workers * axes],
            consumed: vec![None; workers],
            tabled_last: vec![None; workers],
            high_water: vec![0; workers],
        }
    }

    /// the most grants held at once from `peer` over the session: the storage the peer's
    /// lead needed
    pub fn high_water(&self, peer: WorkerId) -> usize {
        self.high_water[peer.0 as usize]
    }

    /// the phase a peer's sequence has reached as this worker sees it: its last tabled
    /// grant, else its last consumed one
    pub fn reached(&self, peer: WorkerId) -> Option<Epoch> {
        self.tabled_last[peer.0 as usize].or(self.consumed[peer.0 as usize])
    }

    fn slots_of(&self, peer: WorkerId) -> &[Option<Tabled>] {
        let p = peer.0 as usize;
        &self.slots[p * self.axes..(p + 1) * self.axes]
    }

    fn slots_of_mut(&mut self, peer: WorkerId) -> &mut [Option<Tabled>] {
        let p = peer.0 as usize;
        let axes = self.axes;
        &mut self.slots[p * axes..(p + 1) * axes]
    }

    /// record a grant a peer sent. a grant at or before the last one consumed from that peer,
    /// a grant already tabled for the same epoch, or a grant finding no free slot is a
    /// protocol error; the table never grows.
    pub fn table(&mut self, peer: WorkerId, epoch: Epoch, count: u32) -> Result<(), FabricError> {
        if let Some(last) = self.consumed[peer.0 as usize] {
            if epoch <= last {
                return Err(FabricError::protocol(
                    peer,
                    format!("grant for {epoch:?} at or before the consumed phase {last:?}"),
                ));
            }
        }
        if self
            .slots_of(peer)
            .iter()
            .flatten()
            .any(|t| t.epoch == epoch)
        {
            return Err(FabricError::protocol(
                peer,
                format!("duplicate grant for {epoch:?}"),
            ));
        }
        let slots = self.slots_of_mut(peer);
        match slots.iter_mut().find(|s| s.is_none()) {
            Some(slot) => {
                *slot = Some(Tabled { epoch, count });
                self.tabled_last[peer.0 as usize] = Some(epoch);
                let held = self.tabled(peer);
                let hw = &mut self.high_water[peer.0 as usize];
                *hw = (*hw).max(held);
                Ok(())
            }
            None => Err(FabricError::protocol(
                peer,
                format!(
                    "grant table full ({} slots) on a grant for {epoch:?}",
                    slots.len()
                ),
            )),
        }
    }

    /// this worker opens `epoch`: every tabled grant from a peer this phase exchanges with
    /// must belong to the same exchange point. such a peer waits on this worker inside the
    /// point (a send leaves only on the receiver's grant), so its grant for another point
    /// means it took a different branch out of the last collective. a peer outside the phase
    /// may have opened the next point already; its grant waits for the phase it names.
    pub fn open(
        &self,
        epoch: Epoch,
        in_phase: impl Fn(WorkerId) -> bool,
    ) -> Result<(), FabricError> {
        for (p, chunk) in self.slots.chunks(self.axes).enumerate() {
            if !in_phase(WorkerId(p as u32)) {
                continue;
            }
            for t in chunk.iter().flatten() {
                if !same_point(&t.epoch, &epoch) {
                    return Err(FabricError::protocol(
                        WorkerId(p as u32),
                        format!(
                            "tabled grant for {:?} while this worker opens {epoch:?}",
                            t.epoch
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// take the grant `peer` issued for `epoch`, if it has arrived.
    pub fn consume(&mut self, peer: WorkerId, epoch: Epoch) -> Option<u32> {
        let slot = self
            .slots_of_mut(peer)
            .iter_mut()
            .find(|s| s.is_some_and(|t| t.epoch == epoch))?;
        let count = slot.take().map(|t| t.count);
        self.consumed[peer.0 as usize] = Some(epoch);
        count
    }

    pub fn tabled(&self, peer: WorkerId) -> usize {
        self.slots_of(peer).iter().flatten().count()
    }

    pub fn is_tabled(&self, peer: WorkerId, epoch: Epoch) -> bool {
        self.slots_of(peer)
            .iter()
            .flatten()
            .any(|t| t.epoch == epoch)
    }

    /// at session end nothing may remain tabled.
    pub fn assert_drained(&self) -> Result<(), FabricError> {
        for (p, chunk) in self.slots.chunks(self.axes).enumerate() {
            if let Some(t) = chunk.iter().flatten().next() {
                return Err(FabricError::protocol(
                    WorkerId(p as u32),
                    format!("grant for {:?} still tabled at shutdown", t.epoch),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(step: u64, attempt: u16, point: u8, axis: u8) -> Epoch {
        Epoch {
            round: 0,
            op: 0,
            step,
            attempt,
            point,
            axis,
        }
    }

    /// a flat two-stage table: prime 0, prime audit 1, troubled 2, stage 3, audit 4,
    /// troubled 5, stage 6, audit 7, post-step 8, rollback 9, rollback audit 10, checkpoint
    /// audit 11; the accept point is stage 6
    fn flat_rules() -> PhaseRules {
        let skippable = vec![false, true, true, false, true, true, false, true, true, true, true, true];
        let mut init_only = vec![false; 12];
        init_only[0] = true;
        init_only[1] = true;
        let iterative = vec![false; 12];
        PhaseRules { skippable, init_only, iterative, accept: 6, post_step: 8, rollback: 9, rollback_audit: 10, checkpoint_audit: 11 }
    }

    fn all(_: usize) -> bool {
        true
    }

    /// the flat table with the troubled phase 5 repeating
    fn iterative_rules() -> PhaseRules {
        let mut r = flat_rules();
        r.iterative[5] = true;
        r
    }

    fn pass(attempt: u16, point: u8, round: u32, op: u8, axis: u8) -> Epoch {
        Epoch { step: 0, attempt, point, round, op, axis }
    }

    #[test]
    fn an_iterative_point_rises_by_pass_then_exchange_then_axis_past_the_old_point_count() {
        let r = iterative_rules();
        // one worker's own sequence: 300 passes of three two-axis exchanges inside phase 5, more
        // phases than the 255 point bytes could name, then the stage after it
        let mut last = pass(0, 3, 0, 0, 1);
        for round in 0..300u32 {
            for op in 0..3u8 {
                for axis in 0..2u8 {
                    let next = pass(0, 5, round, op, axis);
                    r.allows(Some(last), next, all, Some(2)).unwrap_or_else(|e| panic!("{e}"));
                    last = next;
                }
            }
        }
        assert!(r.allows(Some(last), pass(0, 6, 0, 0, 0), all, Some(2)).is_ok(), "the stage after the solve");
        assert!(r.allows(Some(last), pass(0, 9, 0, 0, 0), all, Some(2)).is_ok(), "a rollback from inside the solve");
    }

    #[test]
    fn an_iterative_point_refuses_repeats_backward_steps_and_skipped_axes() {
        let r = iterative_rules();
        let at = pass(0, 5, 4, 2, 1);
        assert!(r.allows(Some(at), at, all, Some(2)).is_err(), "a repeated exchange");
        assert!(r.allows(Some(at), pass(0, 5, 4, 1, 0), all, Some(2)).is_err(), "an earlier exchange of the pass");
        assert!(r.allows(Some(at), pass(0, 5, 3, 7, 0), all, Some(2)).is_err(), "an earlier pass");
        assert!(r.allows(Some(pass(0, 5, 4, 2, 0)), pass(0, 5, 4, 3, 0), all, Some(2)).is_err(), "an exchange left before its last axis");
        assert!(r.allows(Some(at), pass(0, 5, 4, 3, 1), all, Some(2)).is_err(), "an exchange opened past its first axis");
        assert!(r.allows(Some(at), pass(0, 5, 5, 0, 0), all, Some(2)).is_ok(), "the next pass");
        // a peer's grants name only the exchanges it owes this observer: passes may be skipped
        assert!(r.allows(Some(at), pass(0, 5, 9, 0, 1), all, None).is_ok(), "a peer several passes on");
        assert!(r.allows(Some(at), pass(0, 5, 4, 0, 1), all, None).is_err(), "a peer moving backward");
    }

    #[test]
    fn a_point_that_does_not_repeat_refuses_a_pass() {
        let r = iterative_rules();
        assert!(r.allows(Some(pass(0, 3, 0, 0, 1)), pass(0, 6, 1, 0, 0), all, Some(2)).is_err(), "a round at a stage");
        assert!(r.allows(Some(pass(0, 3, 0, 0, 1)), pass(0, 6, 0, 1, 0), all, Some(2)).is_err(), "an op at a stage");
    }

    #[test]
    fn grants_of_another_exchange_of_the_same_point_are_told_apart() {
        let mut t = GrantTable::new(2, 2);
        let peer = WorkerId(1);
        t.table(peer, pass(0, 5, 7, 1, 0), 3).unwrap();
        assert!(t.open(pass(0, 5, 7, 1, 0), |_| true).is_ok(), "the same exchange");
        assert!(t.open(pass(0, 5, 7, 2, 0), |_| true).is_err(), "a grant of the previous exchange while this worker opens the next");
        assert_eq!(t.consume(peer, pass(0, 5, 7, 1, 0)), Some(3));
        assert!(t.table(peer, pass(0, 5, 7, 1, 0), 3).is_err(), "a stale grant at the consumed exchange");
        assert!(t.table(peer, pass(0, 5, 6, 9, 1), 3).is_err(), "a grant of an earlier pass");
        assert!(t.table(peer, pass(0, 5, 8, 0, 0), 3).is_ok(), "the next pass");
    }

    #[test]
    fn inside_an_attempt_the_phases_rise_and_owed_phases_are_never_passed() {
        let r = flat_rules();
        assert!(r.allows(None, e(0, 0, 0, 0), all, None).is_ok());
        assert!(r.allows(Some(e(0, 0, 0, 0)), e(0, 0, 0, 1), all, None).is_ok(), "the next axis of the same phase");
        assert!(r.allows(Some(e(0, 0, 0, 1)), e(0, 0, 0, 1), all, None).is_err(), "a repeated axis");
        assert!(r.allows(Some(e(0, 0, 0, 1)), e(0, 0, 0, 0), all, None).is_err(), "a reversed axis");
        assert!(r.allows(Some(e(0, 0, 0, 0)), e(0, 0, 3, 0), all, None).is_ok(), "the skippable prime audit and troubled passed");
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(0, 0, 8, 0), all, None).is_err(), "stage 1 passed");
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(0, 0, 8, 0), |b| b != 6, None).is_ok(), "stage 1 passed by a peer that owes nothing there");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 0, 3, 0), all, None).is_err(), "backward");
    }

    #[test]
    fn a_rollback_opens_before_the_post_step_and_its_audit_follows_it_alone() {
        let r = flat_rules();
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(0, 0, 9, 0), all, None).is_ok(), "rollback after stage 0");
        assert!(r.allows(Some(e(0, 0, 8, 0)), e(0, 0, 9, 0), all, None).is_err(), "rollback after the post-step");
        assert!(r.allows(Some(e(0, 0, 9, 0)), e(0, 0, 10, 0), all, None).is_ok(), "the rollback audit after the rollback");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 0, 10, 0), all, None).is_err(), "the rollback audit without a rollback");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 0, 10, 0), |b| b != 9, None).is_ok(), "a peer that owes no rollback grant");
        assert!(r.allows(Some(e(0, 0, 9, 0)), e(0, 0, 8, 0), all, None).is_err(), "the attempt continues after a rollback");
    }

    #[test]
    fn a_retry_follows_a_rollback_and_the_next_step_follows_an_accepted_attempt() {
        let r = flat_rules();
        assert!(r.allows(Some(e(0, 0, 9, 0)), e(0, 1, 3, 0), all, None).is_ok(), "the retry after the rollback");
        assert!(r.allows(Some(e(0, 0, 10, 0)), e(0, 1, 3, 0), all, None).is_ok(), "the retry after the rollback audit");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 1, 3, 0), all, None).is_err(), "a retry without a rollback");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 1, 3, 0), |b| b != 9, None).is_ok(), "a peer that owes no rollback grant");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(1, 0, 3, 0), all, None).is_ok(), "the next step after the accept point");
        assert!(r.allows(Some(e(0, 0, 8, 0)), e(1, 0, 3, 0), all, None).is_ok(), "the next step after the post-step");
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(1, 0, 3, 0), all, None).is_err(), "the next step with stage 1 unfinished");
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(1, 0, 3, 0), |b| b != 6, None).is_ok(), "unfinished only where nothing was owed");
        assert!(r.allows(Some(e(0, 0, 9, 0)), e(1, 0, 3, 0), all, None).is_err(), "the next step after a rollback");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(2, 0, 3, 0), all, None).is_err(), "a step skipped");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 2, 3, 0), all, None).is_err(), "an attempt skipped");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(1, 0, 0, 0), all, None).is_err(), "the prime after the first attempt");
        assert!(r.allows(Some(e(0, 0, 0, 0)), e(1, 0, 3, 0), all, None).is_err(), "step zero skipped from the prime");
        assert!(r.allows(Some(e(0, 0, 1, 0)), e(1, 0, 3, 0), all, None).is_err(), "step zero skipped from the prime audit");
        assert!(r.allows(Some(e(0, 0, 1, 0)), e(0, 0, 3, 0), all, None).is_ok(), "the first attempt after the prime audit");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(1, 0, 9, 0), all, None).is_ok(), "a rollback before any exchange of the step");
        assert!(r.allows(None, e(3, 0, 9, 0), all, None).is_ok(), "a rollback as the first phase observed");
        assert!(r.allows(None, e(3, 0, 10, 0), all, None).is_err(), "a rollback audit whose rollback was owed");
        assert!(r.allows(None, e(3, 0, 10, 0), |b| b != 9, None).is_ok(), "a rollback audit by a peer owing no rollback grant");
        // a diagonal peer owing this observer a grant at the last stage alone: an attempt rejected
        // before that stage leaves no trace of it
        let last_stage = |b: usize| b == 6;
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(1, 1, 6, 0), last_stage, None).is_ok(), "the next step's retry after a silent rejection");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(1, 1, 6, 0), |b| b == 6 || b == 9, None).is_err(), "the next step's retry past an owed rollback");
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(1, 1, 6, 0), |b| b == 3 || b == 6, None).is_err(), "the next step's retry with step 0 unaccepted");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(1, 1, 6, 0), |b| b == 3 || b == 6, None).is_err(), "the retry passes an owed stage");
        assert!(r.allows(Some(e(0, 0, 3, 0)), e(0, 2, 3, 0), |b| b == 3, None).is_ok(), "two silent rejections of one step");
    }

    #[test]
    fn the_checkpoint_audit_runs_at_the_checkpoint_epoch_of_an_accepted_or_primed_step() {
        let r = flat_rules();
        let cp = |step| e(step, CHECKPOINT_ATTEMPT, 11, 0);
        assert!(r.allows(Some(e(0, 0, 6, 0)), cp(0), all, None).is_ok(), "after the accept point");
        assert!(r.allows(Some(e(0, 0, 1, 0)), cp(0), all, None).is_err(), "the primed state is checkpointed without an audit");
        assert!(r.allows(Some(e(0, 0, 3, 0)), cp(0), all, None).is_err(), "mid-attempt");
        assert!(r.allows(Some(e(0, 0, 9, 0)), cp(0), all, None).is_err(), "after a rollback");
        assert!(r.allows(Some(e(0, 0, 6, 0)), cp(1), all, None).is_err(), "another step's checkpoint");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, 0, 11, 0), all, None).is_err(), "the audit outside the checkpoint epoch");
        assert!(r.allows(Some(e(0, 0, 6, 0)), e(0, CHECKPOINT_ATTEMPT, 6, 0), all, None).is_err(), "a stage at the checkpoint epoch");
        assert!(r.allows(Some(cp(0)), e(1, 0, 3, 0), all, None).is_ok(), "the next step after the checkpoint");
        assert!(r.allows(Some(cp(0)), e(0, 1, 3, 0), all, None).is_err(), "a retry after the checkpoint");
        assert!(r.allows(Some(cp(0)), cp(1), all, None).is_err(), "two checkpoints with no step between");
        assert!(r.allows(Some(cp(0)), cp(0), all, None).is_err(), "the checkpoint audit repeated");
        assert!(r.allows(Some(cp(0)), e(0, CHECKPOINT_ATTEMPT, 11, 1), all, None).is_ok(), "its next axis");
    }

    #[test]
    fn a_workers_own_axes_run_complete_and_in_order() {
        let r = flat_rules();
        let strict = Some(3);
        assert!(r.allows(None, e(0, 0, 0, 0), all, strict).is_ok());
        assert!(r.allows(None, e(0, 0, 0, 1), all, strict).is_err(), "a phase opened past its first axis");
        assert!(r.allows(Some(e(0, 0, 0, 0)), e(0, 0, 0, 1), all, strict).is_ok());
        assert!(r.allows(Some(e(0, 0, 0, 0)), e(0, 0, 0, 2), all, strict).is_err(), "the middle axis omitted");
        assert!(r.allows(Some(e(0, 0, 0, 1)), e(0, 0, 3, 0), all, strict).is_err(), "the last axis omitted before the next phase");
        assert!(r.allows(Some(e(0, 0, 0, 2)), e(0, 0, 3, 0), all, strict).is_ok(), "the next phase from the last axis");
        assert!(r.allows(Some(e(0, 0, 0, 2)), e(0, 0, 3, 1), all, strict).is_err(), "the next phase past its first axis");
        // a peer's grants name the axes it owes: gaps are no evidence
        assert!(r.allows(Some(e(0, 0, 0, 0)), e(0, 0, 0, 2), all, None).is_ok());
        assert!(r.allows(Some(e(0, 0, 0, 1)), e(0, 0, 3, 0), all, None).is_ok());
    }

    #[test]
    fn a_grant_is_consumed_exactly_once() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 3).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 0)), Some(3));
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 0)), None);
        assert!(
            t.table(WorkerId(1), e(0, 0, 1, 0), 3)
                .unwrap_err()
                .is_protocol()
        );
    }

    #[test]
    fn duplicate_tabled_grant_is_rejected() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 3).unwrap();
        assert!(
            t.table(WorkerId(1), e(0, 0, 1, 0), 3)
                .unwrap_err()
                .is_protocol()
        );
        assert_eq!(t.tabled(WorkerId(1)), 1);
    }

    #[test]
    fn the_table_holds_one_slot_per_axis_and_never_grows() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 1).unwrap();
        t.table(WorkerId(1), e(0, 0, 1, 1), 1).unwrap();
        let err = t.table(WorkerId(1), e(0, 0, 2, 0), 1).unwrap_err();
        assert!(err.is_protocol());
        assert_eq!(t.tabled(WorkerId(1)), 2);
    }

    #[test]
    fn a_future_axis_grant_survives_the_earlier_axis() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 1).unwrap();
        t.table(WorkerId(1), e(0, 0, 1, 1), 1).unwrap();
        t.open(e(0, 0, 1, 0), |_| true).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 0)), Some(1));
        assert!(t.is_tabled(WorkerId(1), e(0, 0, 1, 1)));
        t.open(e(0, 0, 1, 1), |_| true).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 1)), Some(1));
        t.assert_drained().unwrap();
    }

    #[test]
    fn a_stale_grant_is_rejected() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 1), 1).unwrap();
        t.consume(WorkerId(1), e(0, 0, 1, 1));
        assert!(
            t.table(WorkerId(1), e(0, 0, 1, 0), 1)
                .unwrap_err()
                .is_protocol()
        );
    }

    #[test]
    fn a_grant_for_another_branch_is_rejected_at_open() {
        let mut t = GrantTable::new(2, 2);
        // the peer took the accept branch into stage 1; this worker's result said rollback.
        t.table(WorkerId(1), e(3, 0, 2, 0), 1).unwrap();
        let err = t.open(e(3, 0, 0xF0, 0), |_| true).unwrap_err();
        assert!(err.is_protocol());
        // and the converse: the peer rolled back, this worker opens stage 1.
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(3, 0, 0xF0, 0), 1).unwrap();
        assert!(t.open(e(3, 0, 2, 0), |_| true).unwrap_err().is_protocol());
    }

    /// a peer with no transfer in the phase being opened may have finished the point and
    /// opened the next one; its grant for that next point stays tabled through the open
    #[test]
    fn a_next_point_grant_from_a_peer_outside_the_phase_survives_the_open() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(3, 0, 3, 0), 2).unwrap();
        t.open(e(3, 0, 2, 1), |_| false).unwrap();
        assert!(t.open(e(3, 0, 2, 1), |_| true).unwrap_err().is_protocol());
        t.open(e(3, 0, 3, 0), |_| true).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(3, 0, 3, 0)), Some(2));
    }

    #[test]
    fn a_grant_for_the_next_point_is_tabled_before_the_result_and_consumed_after() {
        let mut t = GrantTable::new(2, 2);
        // the peer already holds Result(k) and opens stage 1; this worker is still waiting.
        t.table(WorkerId(1), e(3, 0, 2, 0), 4).unwrap();
        // this worker's result arrives and it opens the same phase.
        t.open(e(3, 0, 2, 0), |_| true).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(3, 0, 2, 0)), Some(4));
    }
}
