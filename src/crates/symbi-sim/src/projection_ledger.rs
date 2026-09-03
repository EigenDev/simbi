// =============================================================================
// projection_ledger.rs
//
// the production accounting for the GRMHD admissible-boundary projection: the
// evidence of what the live projection did to the conserved state, booked over
// the accepted solution. this is a diagnostic ledger, not a correction — it
// records interventions the projection already performed and changes no
// physics.
//
// the projection kernel emits a per-cell receipt (theta and the (D, tau) state
// deltas) that the substrate reduces into a [`ProjectionReceipt`] and returns
// through the [`FofcReport`]. the sim owns the transaction: every substage's
// receipt books into the attempted totals and accumulates into a pending-step
// bucket; the driver commits the complete bucket into the accepted totals only
// when the step is accepted, and discards it on retry — so the accepted
// [`ProjectionLedger`] describes only states that survived into the solution.
//
// two accumulations per bucket. the intervention totals sum every raw per-pass
// delta at weight one — the magnitude of every correction the projection made.
// the injected totals scale each pass by its downstream shu-osher propagation
// weight (euler 1; rk2 1/2, 1; rk3 1/6, 2/3, 1) — the factor by which a delta
// added to that substage's output reaches the accepted step. each carries a
// signed net and a gross absolute: the injected `signed` is the exact direct
// projection contribution to the accepted conserved total, since summation
// commutes with the linear ssp propagation; the `abs` is the scheme-weighted
// gross (L1) budget `sum_stage w sum_cell |delta|`, which bounds the absolute
// net defect from above and reports total projection activity.
//
// ownership: the book is thread-local, so each run's execution thread owns its
// own ledger. a single-grid run executes fold-and-commit on one thread, so its
// projection books there; parallel runs on other threads keep separate books
// and never collide. the single-grid driver opens a [`LedgerScope`] for the
// run — the open clears the thread's ledger and enables booking, the close
// disables it and leaves the totals queryable. the shared step routine books
// only while a scope is open, so the decomposed and refined runners, whose step
// transaction is not modeled here, record nothing.
// =============================================================================

use std::cell::Cell;
use std::marker::PhantomData;

/// a signed sum and the sum of absolute values of the same per-cell quantity.
/// `abs` is a gross (L1) magnitude built from per-term absolute values, so a
/// weighted fold reports `sum w|x|`, not `|sum w x|` — it bounds the absolute
/// net from above and never cancels across terms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SignedAbs {
    pub signed: f64,
    pub abs: f64,
}

impl SignedAbs {
    fn add(&mut self, v: f64) {
        self.signed += v;
        self.abs += v.abs();
    }
    fn fold(&mut self, o: &SignedAbs) {
        self.signed += o.signed;
        self.abs += o.abs;
    }
    fn fold_scaled(&mut self, o: &SignedAbs, w: f64) {
        self.signed += w * o.signed;
        self.abs += w * o.abs;
    }
}

/// the receipt of one projection pass: the raw per-pass conserved deltas the
/// projection wrote, at weight one. `den` is the mass slot delta, `nrg` the
/// total energy slot delta (the segment blend plus the anchor-energy raise);
/// both match the projection's state writes by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionReceipt {
    pub fired: bool,
    pub min_theta: f64,
    pub projected_cells: u64,
    pub den: SignedAbs,
    pub nrg: SignedAbs,
}

impl ProjectionReceipt {
    /// the empty-pass identities: no cell projected, min theta one, zero
    /// counts and deltas.
    pub fn empty() -> Self {
        Self {
            fired: false,
            min_theta: 1.0,
            projected_cells: 0,
            den: SignedAbs::default(),
            nrg: SignedAbs::default(),
        }
    }
}

/// the accumulated projection ledger — the run's booked constraint-injection
/// evidence for the admissible-boundary projection.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProjectionLedger {
    pub passes: u64,
    pub passes_fired: u64,
    pub projected_cells: u64,
    pub min_theta: f64,
    /// raw intervention: every per-pass delta at weight one.
    pub intervention_den: SignedAbs,
    pub intervention_nrg: SignedAbs,
    /// scheme-effective injection into the accepted step: each pass scaled by
    /// its downstream shu-osher propagation weight. `signed` is the exact
    /// conservation contribution; `abs` is the gross (L1) budget.
    pub injected_den: SignedAbs,
    pub injected_nrg: SignedAbs,
}

impl ProjectionLedger {
    fn identity() -> Self {
        IDENTITY
    }
    fn fold_pass(&mut self, r: &ProjectionReceipt, injection_weight: f64) {
        self.passes += 1;
        if r.fired {
            self.passes_fired += 1;
        }
        self.projected_cells += r.projected_cells;
        if r.min_theta < self.min_theta {
            self.min_theta = r.min_theta;
        }
        self.intervention_den.fold(&r.den);
        self.intervention_nrg.fold(&r.nrg);
        self.injected_den.fold_scaled(&r.den, injection_weight);
        self.injected_nrg.fold_scaled(&r.nrg, injection_weight);
    }
    fn fold_totals(&mut self, o: &ProjectionLedger) {
        self.passes += o.passes;
        self.passes_fired += o.passes_fired;
        self.projected_cells += o.projected_cells;
        if o.min_theta < self.min_theta {
            self.min_theta = o.min_theta;
        }
        self.intervention_den.fold(&o.intervention_den);
        self.intervention_nrg.fold(&o.intervention_nrg);
        self.injected_den.fold(&o.injected_den);
        self.injected_nrg.fold(&o.injected_nrg);
    }
}

#[derive(Clone, Copy)]
struct Book {
    open: bool,
    attempted: ProjectionLedger,
    accepted: ProjectionLedger,
    pending: ProjectionLedger,
}

/// the empty ledger: min theta one, every count and sum zero.
const IDENTITY: ProjectionLedger = ProjectionLedger {
    passes: 0,
    passes_fired: 0,
    projected_cells: 0,
    min_theta: 1.0,
    intervention_den: SignedAbs { signed: 0.0, abs: 0.0 },
    intervention_nrg: SignedAbs { signed: 0.0, abs: 0.0 },
    injected_den: SignedAbs { signed: 0.0, abs: 0.0 },
    injected_nrg: SignedAbs { signed: 0.0, abs: 0.0 },
};

const CLEARED: Book = Book {
    open: false,
    attempted: IDENTITY,
    accepted: IDENTITY,
    pending: IDENTITY,
};

impl Book {
    fn cleared() -> Self {
        CLEARED
    }
}

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
        "projection ledger {what} outside a scope: the single-grid driver opens one \
         ledger scope per run before any pass books"
    );
}

/// whether this thread's ledger scope is open. the shared step routine consults
/// this to book only on the single-grid path that owns the transaction.
pub fn scope_is_open() -> bool {
    BOOK.with(|cell| cell.get().open)
}

/// the open run scope. dropping it closes the scope and leaves the totals
/// queryable. thread-bound: it stays on the thread that opened it.
pub struct LedgerScope {
    _single_thread: PhantomData<*const ()>,
}

/// open this thread's run scope: clear the ledger and enable booking. a scope
/// already open panics — one run per thread book at a time.
pub fn open_scope() -> LedgerScope {
    with_book(|b| {
        assert!(
            !b.open,
            "projection ledger scope already open on this thread: the previous scope must \
             close before the next opens"
        );
        *b = Book::cleared();
        b.open = true;
    });
    LedgerScope {
        _single_thread: PhantomData,
    }
}

impl Drop for LedgerScope {
    fn drop(&mut self) {
        let unresolved = with_book(|b| {
            let unresolved = b.pending.passes;
            b.pending = ProjectionLedger::identity();
            b.open = false;
            unresolved
        });
        if !std::thread::panicking() {
            assert_eq!(
                unresolved, 0,
                "projection ledger scope closed with an unresolved timestep: \
                 {unresolved} pending pass(es) neither committed nor discarded"
            );
        }
    }
}

/// reduce the projection kernel's per-cell diagnostic channels into a receipt,
/// in interior iteration order. the four slices cover the same interior; a
/// non-finite or out-of-range theta panics, since `NaN < 1` is false and would
/// read as "did not fire". the energy delta is the segment blend plus the
/// anchor-energy raise, summed per cell so the receipt matches the projection's
/// energy-slot write.
pub fn receipt_from_diagnostics(
    theta: &[f64],
    d_den: &[f64],
    d_nrg_seg: &[f64],
    d_nrg_raise: &[f64],
) -> ProjectionReceipt {
    assert!(
        d_den.len() == theta.len()
            && d_nrg_seg.len() == theta.len()
            && d_nrg_raise.len() == theta.len(),
        "projection ledger: the diagnostic slices cover the same interior \
         (theta {}, d_den {}, d_nrg_seg {}, d_nrg_raise {})",
        theta.len(),
        d_den.len(),
        d_nrg_seg.len(),
        d_nrg_raise.len()
    );
    let mut r = ProjectionReceipt::empty();
    for i in 0..theta.len() {
        assert!(
            theta[i].is_finite() && (0.0..=1.0).contains(&theta[i]),
            "projection ledger: unclassifiable theta {} at interior index {i}",
            theta[i]
        );
        if theta[i] < 1.0 {
            r.projected_cells += 1;
            if theta[i] < r.min_theta {
                r.min_theta = theta[i];
            }
            r.den.add(d_den[i]);
            r.nrg.add(d_nrg_seg[i] + d_nrg_raise[i]);
        }
    }
    r.fired = r.projected_cells > 0;
    r
}

/// book one projection pass: into the attempted totals immediately, and into
/// the pending-step bucket the driver later commits or discards.
/// `injection_weight` is the pass's downstream shu-osher propagation weight
/// (`driver::downstream_injection_weight`), a convex-coefficient product in
/// (0, 1].
pub fn record(receipt: &ProjectionReceipt, injection_weight: f64) {
    assert!(
        injection_weight.is_finite() && injection_weight > 0.0 && injection_weight <= 1.0,
        "projection ledger: injection weight {injection_weight} outside the convex range (0, 1]"
    );
    with_book(|b| {
        require_open(b, "record");
        b.attempted.fold_pass(receipt, injection_weight);
        b.pending.fold_pass(receipt, injection_weight);
    });
}

/// the accepted-step transaction boundary: promote the complete pending-step
/// bucket into the accepted totals. a pass-free step is a cheap no-op.
pub fn step_commit() {
    with_book(|b| {
        require_open(b, "commit");
        if b.pending.passes == 0 {
            return;
        }
        let pending = b.pending;
        b.accepted.fold_totals(&pending);
        b.pending = ProjectionLedger::identity();
    });
}

/// the retry/rollback boundary: the pending-step bucket describes states the
/// rollback discarded, so it empties without touching the accepted totals.
pub fn step_discard() {
    with_book(|b| {
        require_open(b, "discard");
        if b.pending.passes == 0 {
            return;
        }
        b.pending = ProjectionLedger::identity();
    });
}

/// the (attempted, accepted) ledgers for this thread — queryable during a run
/// and after the scope closes, until the next open clears the book.
pub fn ledger_report() -> (ProjectionLedger, ProjectionLedger) {
    BOOK.with(|cell| {
        let b = cell.get();
        (b.attempted, b.accepted)
    })
}

/// clear this thread's book between runs. a scope must be closed first.
pub fn ledger_reset() {
    with_book(|b| {
        assert!(!b.open, "projection ledger reset inside an open scope");
        *b = Book::cleared();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // each test runs on its own thread, so the thread-local book is fresh and
    // independent — no cross-test serialization is needed.

    fn fired(den: f64, nrg: f64) -> ProjectionReceipt {
        let mut r = ProjectionReceipt::empty();
        r.fired = true;
        r.projected_cells = 1;
        r.min_theta = 0.5;
        r.den.add(den);
        r.nrg.add(nrg);
        r
    }

    /// a retried step books attempted-only; nothing enters accepted.
    #[test]
    fn a_retried_step_leaves_no_accepted_receipts() {
        let s = open_scope();
        record(&fired(1.0, 0.5), 1.0);
        record(&fired(2.0, 0.5), 1.0);
        step_discard();
        let (attempted, accepted) = ledger_report();
        assert_eq!(attempted.passes, 2);
        assert_eq!(attempted.intervention_den.signed, 3.0);
        assert_eq!(accepted.passes, 0);
        drop(s);
    }

    /// an accepted multi-substage step promotes its accumulated bucket once;
    /// the rk2 weights separate the ledgers.
    #[test]
    fn an_accepted_step_promotes_its_whole_bucket_once() {
        let s = open_scope();
        record(&fired(1.0, 0.0), 0.5);
        record(&fired(2.0, 0.0), 1.0);
        step_commit();
        step_commit(); // pass-free: a second commit moves nothing
        let (_, accepted) = ledger_report();
        assert_eq!(accepted.passes, 2);
        assert_eq!(accepted.intervention_den.signed, 3.0);
        assert_eq!(accepted.injected_den.signed, 0.5 * 1.0 + 1.0 * 2.0);
        assert_eq!(accepted.injected_den.abs, 2.5);
        drop(s);
    }

    /// retry then accepted replay: only the replay's passes enter accepted.
    #[test]
    fn only_the_accepted_replay_promotes() {
        let s = open_scope();
        record(&fired(10.0, 0.0), 1.0);
        step_discard();
        record(&fired(2.0, 0.0), 1.0);
        step_commit();
        let (attempted, accepted) = ledger_report();
        assert_eq!(attempted.passes, 2);
        assert_eq!(accepted.passes, 1);
        assert_eq!(accepted.intervention_den.signed, 2.0);
        drop(s);
    }

    /// a fresh thread carries its own book with no scope open, so a stray
    /// projection there fails loudly rather than corrupting another run.
    #[test]
    fn a_thread_without_a_scope_panics_on_record() {
        let s = open_scope();
        record(&fired(1.0, 0.0), 1.0);
        let stray = std::thread::spawn(|| record(&ProjectionReceipt::empty(), 1.0));
        assert!(stray.join().is_err());
        step_discard();
        drop(s);
    }

    /// a same-thread overlap fails loudly.
    #[test]
    fn a_same_thread_overlap_panics() {
        let s = open_scope();
        let overlap = std::panic::catch_unwind(open_scope);
        assert!(overlap.is_err());
        drop(s);
    }

    /// closing a scope with a timestep still pending fails loudly and leaves
    /// the book clean for the next run.
    #[test]
    fn an_unresolved_timestep_at_close_fails_loudly() {
        let bad = std::panic::catch_unwind(|| {
            let s = open_scope();
            record(&fired(1.0, 0.0), 1.0);
            drop(s);
        });
        assert!(bad.is_err());
        let s = open_scope();
        drop(s);
    }

    /// booking without a scope fails loudly — the production guard that a run
    /// with no scope reaches no transaction hook.
    #[test]
    fn a_missing_scope_fails_loudly() {
        ledger_reset();
        assert!(std::panic::catch_unwind(|| record(&fired(1.0, 0.0), 1.0)).is_err());
    }

    /// an unclassifiable theta panics rather than reading as "did not fire",
    /// and mismatched slices fail as a named invariant. the energy delta is the
    /// per-cell sum of the segment and raise channels.
    #[test]
    fn diagnostics_reduce_and_fail_loudly() {
        let r = receipt_from_diagnostics(&[0.5, 1.0], &[3.0, 0.0], &[0.2, 0.0], &[0.1, 0.0]);
        assert_eq!(r.projected_cells, 1);
        assert_eq!(r.den.signed, 3.0);
        assert_eq!(r.nrg.signed, 0.2 + 0.1); // the per-cell segment + raise sum
        assert!(std::panic::catch_unwind(|| {
            receipt_from_diagnostics(&[f64::NAN], &[0.0], &[0.0], &[0.0])
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            receipt_from_diagnostics(&[1.0], &[0.0], &[], &[0.0])
        })
        .is_err());
    }
}
