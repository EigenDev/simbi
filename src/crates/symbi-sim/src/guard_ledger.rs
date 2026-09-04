// =============================================================================
// guard_ledger.rs
//
// the run-owned accounting for the FOFC guard acts: the troubled cells the
// recovery flagged and the cells the correcting select froze, booked over the
// accepted solution. a diagnostic ledger, not a correction — it records acts the
// ladder already performed and changes no physics.
//
// each substage's guard acts arrive as a [`GuardReceipt`] the substrate mints at
// the sites that perform them: `troubled_cells` from the recovery flag at the
// c2p-status decode, `frozen_cells` from the `FreezeApplied` mask the correcting
// select writes. the sim owns the transaction: every substage's receipt books
// into the attempted totals and accumulates into a pending-step bucket; the
// driver commits the complete bucket into the accepted totals only when the step
// is accepted, and discards it on retry or crash — so the accepted totals count
// only acts on states that survived into the solution.
//
// `troubled_cells` is a flag count, never a fallback-applied count: the fallback
// is applied per face in the splice, so a per-cell applied-fallback event has no
// clean mint site. counting the recovery flag reports what was flagged, at the
// site that flagged it.
//
// ownership: the book is thread-local, so each run's execution thread owns its
// own ledger. the shared tile driver runs every shape — single grid, decomposed,
// refined — sequentially on one host thread, so all tiles' and levels' substages
// book into that thread's ledger; concurrent runs on other threads keep separate
// books and never collide. the driver opens a [`GuardScope`] for the whole run,
// which the accepted totals are read back through.
// =============================================================================

use std::cell::Cell;
use std::marker::PhantomData;

/// a count of cells and the subset of them inside a configured horizon.
/// `inside_horizon <= total` holds by construction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellCount {
    pub total: u64,
    pub inside_horizon: u64,
}

impl CellCount {
    /// a cell count whose horizon subset is at most the total.
    pub fn new(total: u64, inside_horizon: u64) -> Self {
        assert!(
            inside_horizon <= total,
            "guard ledger: horizon subset ({inside_horizon}) exceeds the total ({total})"
        );
        Self {
            total,
            inside_horizon,
        }
    }
    fn add(&mut self, o: &CellCount) {
        self.total += o.total;
        self.inside_horizon += o.inside_horizon;
    }
}

/// the guard acts of one substage: the cells the recovery flagged as troubled and
/// the cells the correcting select froze, each with its horizon subset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GuardReceipt {
    /// cells the recovery flagged troubled at the c2p-status decode. a flag
    /// count, not a fallback-applied count.
    pub troubled_cells: CellCount,
    /// cells the correcting select held at stage input (the `FreezeApplied`
    /// mask): the freeze act performed.
    pub frozen_cells: CellCount,
}

impl GuardReceipt {
    /// the receipt of one substage from its measured cell counts.
    pub fn of_pass(
        troubled: u64,
        troubled_inside_horizon: u64,
        frozen: u64,
        frozen_inside_horizon: u64,
    ) -> Self {
        Self {
            troubled_cells: CellCount::new(troubled, troubled_inside_horizon),
            frozen_cells: CellCount::new(frozen, frozen_inside_horizon),
        }
    }
    /// a pass that performed no guard act.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// the accumulated guard totals — troubled and frozen cell counts summed over
/// substages, tiles, and levels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GuardTotals {
    pub troubled_cells: CellCount,
    pub frozen_cells: CellCount,
}

impl GuardTotals {
    fn fold_pass(&mut self, r: &GuardReceipt) {
        self.troubled_cells.add(&r.troubled_cells);
        self.frozen_cells.add(&r.frozen_cells);
    }
    fn fold_totals(&mut self, o: &GuardTotals) {
        self.troubled_cells.add(&o.troubled_cells);
        self.frozen_cells.add(&o.frozen_cells);
    }
}

const ZERO: GuardTotals = GuardTotals {
    troubled_cells: CellCount {
        total: 0,
        inside_horizon: 0,
    },
    frozen_cells: CellCount {
        total: 0,
        inside_horizon: 0,
    },
};

#[derive(Clone, Copy)]
struct Book {
    open: bool,
    attempted: GuardTotals,
    accepted: GuardTotals,
    pending: GuardTotals,
}

const CLEARED: Book = Book {
    open: false,
    attempted: ZERO,
    accepted: ZERO,
    pending: ZERO,
};

thread_local! {
    static BOOK: Cell<Book> = const { Cell::new(CLEARED) };
}

fn with_book<R>(f: impl FnOnce(&mut Book) -> R) -> R {
    BOOK.with(|cell| {
        let mut b = cell.get();
        let r = f(&mut b);
        cell.set(b);
        r
    })
}

fn require_open(b: &Book, what: &str) {
    assert!(
        b.open,
        "guard ledger {what} outside a scope: the tile driver opens one guard scope \
         per run before any substage books"
    );
}

/// whether this thread's guard scope is open. the stage routine consults this to
/// book only while the driver's run scope is live.
pub fn scope_is_open() -> bool {
    BOOK.with(|cell| cell.get().open)
}

/// the open run scope. dropping it closes the scope and leaves the totals
/// queryable. thread-bound: it stays on the thread that opened it.
pub struct GuardScope {
    _single_thread: PhantomData<*const ()>,
}

/// open this thread's run scope: clear the ledger and enable booking. a scope
/// already open panics — one run per thread book at a time.
pub fn open_scope() -> GuardScope {
    with_book(|b| {
        assert!(
            !b.open,
            "guard ledger scope already open on this thread: the previous scope must \
             close before the next opens"
        );
        *b = CLEARED;
        b.open = true;
    });
    GuardScope {
        _single_thread: PhantomData,
    }
}

impl GuardScope {
    /// the accepted guard totals booked under this run's scope. reading requires
    /// the open scope, so only the run that owns the booking window can extract
    /// its own evidence.
    pub fn accepted(&self) -> GuardTotals {
        BOOK.with(|cell| cell.get().accepted)
    }
}

impl Drop for GuardScope {
    fn drop(&mut self) {
        let unresolved = with_book(|b| {
            let unresolved = b.pending;
            b.pending = ZERO;
            b.open = false;
            unresolved
        });
        if !std::thread::panicking() {
            assert_eq!(
                unresolved, ZERO,
                "guard ledger scope closed with an unresolved timestep: pending guard \
                 acts neither committed nor discarded"
            );
        }
    }
}

/// book one substage's guard receipt: into the attempted totals immediately, and
/// into the pending-step bucket the driver later commits or discards.
pub fn record(receipt: &GuardReceipt) {
    with_book(|b| {
        require_open(b, "record");
        b.attempted.fold_pass(receipt);
        b.pending.fold_pass(receipt);
    });
}

/// the accepted-step transaction boundary: promote the complete pending-step
/// bucket into the accepted totals. a guard-free step is a cheap no-op.
pub fn step_commit() {
    with_book(|b| {
        require_open(b, "commit");
        if b.pending == ZERO {
            return;
        }
        let pending = b.pending;
        b.accepted.fold_totals(&pending);
        b.pending = ZERO;
    });
}

/// the retry/rollback boundary: the pending-step bucket describes states the
/// rollback discarded, so it empties without touching the accepted totals.
pub fn step_discard() {
    with_book(|b| {
        require_open(b, "discard");
        b.pending = ZERO;
    });
}

/// the (attempted, accepted) totals for this thread — queryable during a run and
/// after the scope closes, until the next open clears the book. the attempted
/// totals count every substage, including rejected attempts, so they compare
/// like-for-like with the legacy process-global `guard_census()`; the accepted
/// totals count only acts on the surviving solution.
pub fn report() -> (GuardTotals, GuardTotals) {
    BOOK.with(|cell| {
        let b = cell.get();
        (b.attempted, b.accepted)
    })
}

/// clear this thread's book between runs. a scope must be closed first.
pub fn reset() {
    with_book(|b| {
        assert!(!b.open, "guard ledger reset inside an open scope");
        *b = CLEARED;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // each test runs on its own thread, so the thread-local book is fresh and
    // independent — no cross-test serialization is needed.

    fn pass(troubled: u64, frozen: u64) -> GuardReceipt {
        GuardReceipt::of_pass(troubled, 0, frozen, 0)
    }

    /// a retried step books attempted-only; nothing enters accepted.
    #[test]
    fn a_retried_step_leaves_no_accepted_acts() {
        let s = open_scope();
        record(&pass(3, 1));
        record(&pass(2, 0));
        step_discard();
        let (attempted, accepted) = report();
        assert_eq!(attempted.troubled_cells.total, 5);
        assert_eq!(attempted.frozen_cells.total, 1);
        assert_eq!(accepted, ZERO);
        drop(s);
    }

    /// an accepted multi-substage step promotes its accumulated bucket once.
    #[test]
    fn an_accepted_step_promotes_its_whole_bucket_once() {
        let s = open_scope();
        record(&pass(4, 1));
        record(&pass(2, 3));
        step_commit();
        step_commit(); // act-free: a second commit moves nothing
        let (_, accepted) = report();
        assert_eq!(accepted.troubled_cells.total, 6);
        assert_eq!(accepted.frozen_cells.total, 4);
        drop(s);
    }

    /// retry then accepted replay: only the replay's acts enter accepted.
    #[test]
    fn only_the_accepted_replay_promotes() {
        let s = open_scope();
        record(&pass(10, 5));
        step_discard();
        record(&pass(2, 1));
        step_commit();
        let (attempted, accepted) = report();
        assert_eq!(attempted.troubled_cells.total, 12);
        assert_eq!(accepted.troubled_cells.total, 2);
        assert_eq!(accepted.frozen_cells.total, 1);
        drop(s);
    }

    /// the horizon subset accumulates alongside the total and never exceeds it.
    #[test]
    fn the_horizon_subset_accumulates_within_the_total() {
        let s = open_scope();
        record(&GuardReceipt::of_pass(5, 2, 3, 1));
        record(&GuardReceipt::of_pass(4, 1, 2, 2));
        step_commit();
        let (_, accepted) = report();
        assert_eq!(accepted.troubled_cells.total, 9);
        assert_eq!(accepted.troubled_cells.inside_horizon, 3);
        assert_eq!(accepted.frozen_cells.total, 5);
        assert_eq!(accepted.frozen_cells.inside_horizon, 3);
        assert!(accepted.frozen_cells.inside_horizon <= accepted.frozen_cells.total);
        drop(s);
    }

    /// a horizon subset exceeding the total is rejected at the mint.
    #[test]
    fn a_horizon_subset_over_the_total_fails_loudly() {
        assert!(std::panic::catch_unwind(|| GuardReceipt::of_pass(2, 3, 0, 0)).is_err());
    }

    /// a fresh thread carries its own book with no scope open, so a stray record
    /// there fails loudly rather than corrupting another run.
    #[test]
    fn a_thread_without_a_scope_panics_on_record() {
        let s = open_scope();
        record(&pass(1, 0));
        let stray = std::thread::spawn(|| record(&pass(1, 0)));
        assert!(stray.join().is_err());
        step_discard();
        drop(s);
    }

    /// a rejected nested open leaves the outer scope's booking intact.
    #[test]
    fn a_rejected_nested_open_leaves_the_outer_scope_intact() {
        let s = open_scope();
        record(&pass(3, 1));
        let nested = std::panic::catch_unwind(open_scope);
        assert!(nested.is_err(), "the nested open must reject");
        step_commit();
        let (_, accepted) = report();
        assert_eq!(accepted.troubled_cells.total, 3);
        assert_eq!(accepted.frozen_cells.total, 1);
        drop(s);
    }

    /// booking without a scope fails loudly.
    #[test]
    fn a_missing_scope_fails_loudly() {
        reset();
        assert!(std::panic::catch_unwind(|| record(&pass(1, 0))).is_err());
    }
}
