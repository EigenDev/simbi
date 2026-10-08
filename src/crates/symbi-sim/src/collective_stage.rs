// =============================================================================
// collective_stage.rs
//
// the agreement of one stage of one level across every tile that folds it, the
// same on every march. a driver supplies the tiles' mechanics and its transport
// through `StageCollective`; `collective_stage` owns the sequence:
// - every local tile folds the stage through the troubled-cell mark (through
//   the edge EMF, the axis-ring reconciliation and the curl first when the
//   tiles share an axis ring), and a test fault may raise further trouble;
// - the participants agree whether any tile is troubled; only then do the
//   troubled flags cross the cuts, and each tile learns whether a neighbor's
//   trouble reached its cut ghosts;
// - every local tile corrects with that knowledge;
// - the participants agree the level's freeze streak from the count the tiles
//   held before the pass (the first local tile's, zero on a participant
//   holding none) and whether any tile froze, and every local tile holds it;
// - a test fault may force a rejection, and the participants agree the retry
//   before any halo of the stage moves.
// a participant holding no tile at the level issues every agreement with
// neutral inputs, so every participant issues the same collective sequence.
// what follows an agreed stage (excision, halos, tracers, audits) stays with
// the driver.
//
// usage:
//  let agreed = collective_stage(&mut level_stage)?;
//  if agreed.retry { /* the step is rejected */ }
// =============================================================================

/// the mechanics of one stage of one level on one participant's tiles
pub trait StageCollective {
    type Error;
    /// the participant's tiles at this level
    fn tiles(&self) -> usize;
    /// fold tile `k` through the troubled-cell mark; the troubled interior cells it marked
    fn mark(&mut self, k: usize) -> Result<u64, Self::Error>;
    /// whether the tiles share an axis ring, whose edges are reconciled between the stage's edge
    /// EMF and its curl
    fn axis_rings(&self) -> bool {
        false
    }
    /// fold tile `k` through the stage's complete edge EMF
    fn mark_through_emf(&mut self, _k: usize) -> Result<(), Self::Error> {
        unreachable!("a stage without axis rings folds through the mark in one span")
    }
    /// reconcile the axis rings across every tile holding a copy
    fn reconcile_rings(&mut self) -> Result<(), Self::Error> {
        unreachable!("a stage without axis rings reconciles none")
    }
    /// fold tile `k` from the curl through the troubled-cell mark; the troubled interior cells
    fn mark_from_curl(&mut self, _k: usize) -> Result<u64, Self::Error> {
        unreachable!("a stage without axis rings folds through the mark in one span")
    }
    /// the cells a test fault raises troubled on tile `k` at this stage
    fn injected_trouble(&mut self, _k: usize) -> u64 {
        0
    }
    /// whether `local` holds on any participant
    fn any(&mut self, local: bool) -> Result<bool, Self::Error>;
    /// carry the troubled flags across the level's cuts; per tile, whether a neighbor's trouble
    /// reached its cut ghosts
    fn exchange_trouble(&mut self) -> Result<Vec<bool>, Self::Error>;
    /// fold tile `k` from the correction on; true on a retry request
    fn correct(&mut self, k: usize, cut_trouble: bool) -> Result<bool, Self::Error>;
    /// the freeze streak tile `k`'s kernel holds
    fn freeze_streak(&self, k: usize) -> u32;
    fn set_freeze_streak(&mut self, k: usize, streak: u32);
    /// whether a test fault forces this stage's rejection
    fn forced_rejection(&mut self) -> bool {
        false
    }
}

/// what the participants agreed for one stage
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageAgreement {
    /// a tile somewhere requested a retry: the step is rejected
    pub retry: bool,
    /// a tile somewhere was troubled, so the flags crossed the cuts
    pub exchanged: bool,
    /// local tiles with no trouble of their own that corrected for a neighbor's trouble at a cut
    pub neighbor_corrections: u64,
}

/// one stage of one level, marked, exchanged, corrected and agreed across every participant
pub fn collective_stage<C: StageCollective>(cx: &mut C) -> Result<StageAgreement, C::Error> {
    let n = cx.tiles();
    let mut own = vec![0u64; n];
    if cx.axis_rings() {
        for k in 0..n {
            cx.mark_through_emf(k)?;
        }
        cx.reconcile_rings()?;
        for k in 0..n {
            own[k] = cx.mark_from_curl(k)?;
        }
    } else {
        for k in 0..n {
            own[k] = cx.mark(k)?;
        }
    }
    for k in 0..n {
        own[k] += cx.injected_trouble(k);
    }
    let exchanged = cx.any(own.iter().any(|&count| count > 0))?;
    let cut_trouble = if exchanged {
        cx.exchange_trouble()?
    } else {
        vec![false; n]
    };
    // every tile holds the level's agreed freeze count between passes: zero when a session opens,
    // the agreed value after every pass, a rejected attempt's included, and untouched by the
    // equilibrium probes, which put back the counts they found
    let prior = if n > 0 { cx.freeze_streak(0) } else { 0 };
    debug_assert!(
        (0..n).all(|k| cx.freeze_streak(k) == prior),
        "the tiles of a level entered a stage holding different freeze counts"
    );
    let mut retry = false;
    let mut neighbor_corrections = 0;
    for k in 0..n {
        neighbor_corrections += u64::from(cut_trouble[k] && own[k] == 0);
        retry |= cx.correct(k, cut_trouble[k])?;
    }
    let froze = (0..n).any(|k| cx.freeze_streak(k) > 0);
    let streak = crate::stage::agreed_freeze_streak(prior, cx.any(froze)?);
    for k in 0..n {
        cx.set_freeze_streak(k, streak);
    }
    retry |= cx.forced_rejection();
    // the rejection is decided before this stage's halos move, so no halo is computed from a
    // rejected state
    let retry = cx.any(retry)?;
    Ok(StageAgreement {
        retry,
        exchanged,
        neighbor_corrections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a scripted participant: per tile its marked count, freeze outcome and retry request; the
    /// other participants' votes are `others`, consumed in order
    struct Scripted {
        marked: Vec<u64>,
        froze: Vec<bool>,
        retries: Vec<bool>,
        cut: Vec<bool>,
        streaks: Vec<u32>,
        others: Vec<bool>,
        forced: bool,
        injected: Vec<u64>,
        log: Vec<String>,
    }

    impl Scripted {
        fn new(marked: Vec<u64>, others: Vec<bool>) -> Self {
            let n = marked.len();
            Self {
                marked,
                froze: vec![false; n],
                retries: vec![false; n],
                cut: vec![false; n],
                streaks: vec![3; n],
                others,
                forced: false,
                injected: vec![0; n],
                log: Vec::new(),
            }
        }
    }

    impl StageCollective for Scripted {
        type Error = String;
        fn tiles(&self) -> usize {
            self.marked.len()
        }
        fn mark(&mut self, k: usize) -> Result<u64, String> {
            self.log.push(format!("mark {k}"));
            Ok(self.marked[k])
        }
        fn injected_trouble(&mut self, k: usize) -> u64 {
            self.injected[k]
        }
        fn any(&mut self, local: bool) -> Result<bool, String> {
            let other = if self.others.is_empty() {
                false
            } else {
                self.others.remove(0)
            };
            self.log.push(format!("any {local}"));
            Ok(local || other)
        }
        fn exchange_trouble(&mut self) -> Result<Vec<bool>, String> {
            self.log.push("exchange".into());
            Ok(self.cut.clone())
        }
        fn correct(&mut self, k: usize, cut_trouble: bool) -> Result<bool, String> {
            self.log.push(format!("correct {k} {cut_trouble}"));
            self.streaks[k] = if self.froze[k] {
                self.streaks[k] + 1
            } else {
                0
            };
            Ok(self.retries[k])
        }
        fn freeze_streak(&self, k: usize) -> u32 {
            self.streaks[k]
        }
        fn set_freeze_streak(&mut self, k: usize, streak: u32) {
            self.log.push(format!("streak {k} {streak}"));
            self.streaks[k] = streak;
        }
        fn forced_rejection(&mut self) -> bool {
            self.forced
        }
    }

    #[test]
    fn a_clean_stage_exchanges_nothing_and_agrees_three_times() {
        let mut cx = Scripted::new(vec![0, 0], vec![]);
        let agreed = collective_stage(&mut cx).unwrap();
        assert_eq!(agreed, StageAgreement::default());
        assert_eq!(
            cx.log,
            [
                "mark 0",
                "mark 1",
                "any false",
                "correct 0 false",
                "correct 1 false",
                "any false",
                "streak 0 0",
                "streak 1 0",
                "any false"
            ]
        );
    }

    #[test]
    fn trouble_elsewhere_exchanges_the_flags_and_a_clean_tile_corrects_for_its_neighbor() {
        let mut cx = Scripted::new(vec![0, 0], vec![true]);
        cx.cut = vec![false, true];
        let agreed = collective_stage(&mut cx).unwrap();
        assert!(agreed.exchanged && !agreed.retry);
        assert_eq!(agreed.neighbor_corrections, 1);
        assert_eq!(
            &cx.log[..6],
            [
                "mark 0",
                "mark 1",
                "any false",
                "exchange",
                "correct 0 false",
                "correct 1 true"
            ]
        );
    }

    #[test]
    fn the_streak_advances_from_the_count_held_before_the_pass_when_any_tile_froze() {
        let mut cx = Scripted::new(vec![1, 0], vec![]);
        cx.froze = vec![true, false];
        collective_stage(&mut cx).unwrap();
        assert_eq!(
            cx.streaks,
            [4, 4],
            "every tile holds the prior count plus one"
        );

        let mut cx = Scripted::new(vec![0, 0], vec![false, true]);
        collective_stage(&mut cx).unwrap();
        assert_eq!(
            cx.streaks,
            [4, 4],
            "a freeze on another participant advances this one's tiles"
        );
    }

    #[test]
    fn a_participant_without_tiles_issues_every_agreement() {
        let mut cx = Scripted::new(vec![], vec![true, true, true]);
        let agreed = collective_stage(&mut cx).unwrap();
        assert_eq!(
            agreed,
            StageAgreement {
                retry: true,
                exchanged: true,
                neighbor_corrections: 0
            }
        );
        assert_eq!(cx.log, ["any false", "exchange", "any false", "any false"]);
    }

    #[test]
    fn a_forced_rejection_and_a_tile_retry_both_reach_the_agreement_after_the_streak() {
        let mut cx = Scripted::new(vec![0], vec![]);
        cx.forced = true;
        let agreed = collective_stage(&mut cx).unwrap();
        assert!(agreed.retry);
        assert_eq!(cx.log.last().map(String::as_str), Some("any true"));
        assert_eq!(
            cx.log[cx.log.len() - 2],
            "streak 0 0",
            "the retry was agreed before the streak"
        );

        let mut cx = Scripted::new(vec![0, 0], vec![]);
        cx.retries = vec![false, true];
        assert!(collective_stage(&mut cx).unwrap().retry);
    }

    // a neighbor correction is a tile with no troubled cell of its own at the stage, marked or
    // raised by a test fault, that corrects because a neighbor's flag reached its cut ghosts
    #[test]
    fn a_tile_holding_injected_trouble_is_no_neighbor_correction() {
        let mut cx = Scripted::new(vec![0, 0], vec![]);
        cx.injected = vec![1, 0];
        cx.cut = vec![true, true];
        let agreed = collective_stage(&mut cx).unwrap();
        assert!(agreed.exchanged, "the injected cell alone carries the stage's trouble");
        assert_eq!(agreed.neighbor_corrections, 1, "only the tile without trouble of its own counts");
    }
}
