// =============================================================================
// projection_experiment.rs
//
// the bookkeeping half of the projection-anchor measurement apparatus: the
// per-pass receipts, the step-scoped transaction, and the run session. the
// kernel/dispatch half lives with the substrate; this book is homed here so
// every runner can reach the transaction boundary and the guards.
//
// sessions: a run opens an [`ExperimentSession`] before its first step; the
// open clears the book and claims it, a second open while one is active
// panics (even on the same thread — two interleaved simulations must fail
// loudly instead of merging their ledgers), and the close at the end of the
// run leaves the completed totals queryable until the next open. every
// record/commit/discard requires the active session and its owning thread.
//
// the transaction follows the whole timestep: every projection pass books
// into the attempted totals immediately and accumulates into the pending-step
// bucket (multiple RK substages accumulate); the driver commits the complete
// pending bucket into the accepted totals only when the step is accepted, and
// discards it on retry/rollback — so accepted totals describe only states
// that survived into the solution, and a step accepted on its early substages
// but retried on a later one leaves nothing behind.
//
// two injection ledgers per bucket: the intervention totals sum every raw
// per-pass projection delta (the magnitude of every correction applied), and
// the injected totals scale each pass's deltas by its downstream shu-osher
// propagation weight — the factor by which a conserved delta added to that
// substage's output reaches the accepted step (euler 1; rk2 1/2, 1; rk3 1/6,
// 2/3, 1). the intervention totals measure how hard the projection worked.
//
// signed vs abs claim strength. each ledger's `signed` is a sum of signed,
// weighted deltas: the injected `signed` is exactly the direct projection
// contribution to the accepted conserved total, since summation commutes with
// the linear ssp propagation. the `abs` is `sum_stage w_stage sum_cell
// |delta|` — a scheme-weighted gross (L1) intervention budget, which is NOT
// the absolute net conservation defect `sum_cell |sum_stage w_stage delta|`:
// opposite-signed corrections across substages or cells cancel in the latter
// and not here. `abs` bounds that net defect from above and reports total
// projection activity; only `signed` carries the exact-defect claim.
//
// firsts: the attempted-first fired pass (by global pass index) and the
// accepted-first fire — the pass index plus the simulation time and iteration
// of the state the accepted step produced — stay distinct: a rejected attempt
// may project before the first projection that survives.
// =============================================================================

use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

/// the arm-selection environment variable. value validation (and the arm
/// enum) live with the substrate dispatch; presence alone gates the runner
/// guards and the driver's session + transaction hooks.
pub const ANCHOR_EXPERIMENT_ENV: &str = "SIMBI_ANCHOR_EXPERIMENT";

/// whether a run named an experiment arm (any value; the dispatch validates).
pub fn experiment_named() -> bool {
    static NAMED: OnceLock<bool> = OnceLock::new();
    *NAMED.get_or_init(|| std::env::var_os(ANCHOR_EXPERIMENT_ENV).is_some())
}

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
    /// fold `o` scaled by a nonnegative weight (`|w v| = w |v|`).
    fn fold_scaled(&mut self, o: &SignedAbs, w: f64) {
        self.signed += w * o.signed;
        self.abs += w * o.abs;
    }
}

/// the receipts of one projection pass: raw per-pass deltas at weight one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionReceipts {
    pub fired: bool,
    pub min_theta: f64,
    pub projected_cells: u64,
    pub mass: SignedAbs,
    pub energy_segment: SignedAbs,
    pub energy_raise: SignedAbs,
}

impl ProjectionReceipts {
    /// the empty-pass identities: no cell projected, min theta one, zero
    /// counts and injections.
    pub fn empty() -> Self {
        Self {
            fired: false,
            min_theta: 1.0,
            projected_cells: 0,
            mass: SignedAbs::default(),
            energy_segment: SignedAbs::default(),
            energy_raise: SignedAbs::default(),
        }
    }
}

/// the running totals of one bookkeeping bucket.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExperimentTotals {
    pub passes: u64,
    pub passes_fired: u64,
    pub projected_cells: u64,
    pub min_theta: f64,
    /// raw intervention sums: every per-pass projection delta at weight one.
    pub intervention_mass: SignedAbs,
    pub intervention_energy_segment: SignedAbs,
    pub intervention_energy_raise: SignedAbs,
    /// scheme-effective injection into the step output: each pass's deltas
    /// scaled by its downstream shu-osher propagation weight. `signed` is the
    /// exact direct projection contribution to the accepted conserved total;
    /// `abs` is the scheme-weighted gross (L1) injection budget, an upper
    /// bound on the absolute net defect rather than the defect itself.
    pub injected_mass: SignedAbs,
    pub injected_energy_segment: SignedAbs,
    pub injected_energy_raise: SignedAbs,
}

impl ExperimentTotals {
    fn identity() -> Self {
        Self {
            min_theta: 1.0,
            ..Self::default()
        }
    }
    fn fold_pass(&mut self, r: &ProjectionReceipts, injection_weight: f64) {
        self.passes += 1;
        if r.fired {
            self.passes_fired += 1;
        }
        self.projected_cells += r.projected_cells;
        if r.min_theta < self.min_theta {
            self.min_theta = r.min_theta;
        }
        self.intervention_mass.fold(&r.mass);
        self.intervention_energy_segment.fold(&r.energy_segment);
        self.intervention_energy_raise.fold(&r.energy_raise);
        self.injected_mass.fold_scaled(&r.mass, injection_weight);
        self.injected_energy_segment
            .fold_scaled(&r.energy_segment, injection_weight);
        self.injected_energy_raise
            .fold_scaled(&r.energy_raise, injection_weight);
    }
    fn fold_totals(&mut self, o: &ExperimentTotals) {
        self.passes += o.passes;
        self.passes_fired += o.passes_fired;
        self.projected_cells += o.projected_cells;
        if o.min_theta < self.min_theta {
            self.min_theta = o.min_theta;
        }
        self.intervention_mass.fold(&o.intervention_mass);
        self.intervention_energy_segment
            .fold(&o.intervention_energy_segment);
        self.intervention_energy_raise
            .fold(&o.intervention_energy_raise);
        self.injected_mass.fold(&o.injected_mass);
        self.injected_energy_segment
            .fold(&o.injected_energy_segment);
        self.injected_energy_raise.fold(&o.injected_energy_raise);
    }
}

/// the first fire that survived into the solution: the global pass index and
/// the accepted step's post-step simulation time and iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirstFire {
    pub pass: u64,
    pub time: f64,
    pub iteration: u64,
}

enum Session {
    Idle,
    Open(ThreadId),
}

struct Book {
    session: Session,
    passes_total: u64,
    attempted: ExperimentTotals,
    accepted: ExperimentTotals,
    first_attempted_pass: Option<u64>,
    first_accepted: Option<FirstFire>,
    pending: ExperimentTotals,
    pending_first_pass: Option<u64>,
}

impl Book {
    fn cleared() -> Self {
        Self {
            session: Session::Idle,
            passes_total: 0,
            attempted: ExperimentTotals::identity(),
            accepted: ExperimentTotals::identity(),
            first_attempted_pass: None,
            first_accepted: None,
            pending: ExperimentTotals::identity(),
            pending_first_pass: None,
        }
    }
    fn require_session(&self, what: &str) {
        let me = std::thread::current().id();
        match self.session {
            Session::Idle => panic!(
                "anchor-experiment {what} outside a session: the driver opens one \
                 experiment session per run (open_session) before any pass books"
            ),
            Session::Open(owner) => assert!(
                owner == me,
                "anchor-experiment {what} from a second owner: the session belongs to \
                 {owner:?}, this call came from {me:?} — one experiment session per book"
            ),
        }
    }
}

/// a panic inside a locked scope stays a deliberate loud failure; the book
/// itself is mutated only after every check passes, so a poisoned guard still
/// wraps consistent state.
fn book() -> MutexGuard<'static, Book> {
    static BOOK: OnceLock<Mutex<Book>> = OnceLock::new();
    BOOK.get_or_init(|| Mutex::new(Book::cleared()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// the open run session. dropping it closes the session and leaves the
/// completed totals queryable until the next open. single-thread by
/// construction: the handle stays on the thread that opened it.
pub struct ExperimentSession {
    _single_thread: PhantomData<*const ()>,
}

/// open the run session: clear the book and claim it. a session already open
/// panics — one experiment run per book at a time, even on one thread.
pub fn open_session() -> ExperimentSession {
    let mut b = book();
    if let Session::Open(owner) = b.session {
        drop(b);
        panic!(
            "anchor-experiment session already open (owner {owner:?}): one run per book — \
             the previous session must close before the next opens"
        );
    }
    *b = Book::cleared();
    b.session = Session::Open(std::thread::current().id());
    ExperimentSession {
        _single_thread: PhantomData,
    }
}

impl Drop for ExperimentSession {
    fn drop(&mut self) {
        // release the session and empty the pending bucket first, so the book
        // is clean for the next open even when the guard below fails.
        let unresolved = {
            let mut b = book();
            let unresolved = b.pending.passes;
            b.pending = ExperimentTotals::identity();
            b.pending_first_pass = None;
            b.session = Session::Idle;
            unresolved
        };
        // a nonempty pending bucket at an ordinary close means the driver left
        // a timestep's projection receipts unresolved — neither committed into
        // the accepted totals nor discarded — so the accepted/rejected
        // classification of that step is silently lost. fail loudly, except
        // while already unwinding (the pending state is expected mid-panic and
        // a second panic would abort).
        if !std::thread::panicking() {
            assert_eq!(
                unresolved, 0,
                "anchor experiment session closed with an unresolved timestep: \
                 {unresolved} pending projection pass(es) neither committed nor discarded"
            );
        }
    }
}

/// the deterministic receipts reduction over the diagnostic channels, in
/// interior iteration order. the four slices must cover the same interior; a
/// non-finite or out-of-range theta panics: the apparatus exists to expose
/// exactly the failure a NaN would silently erase (`NaN < 1` is false and
/// would read as "did not fire").
pub fn receipts_from_diagnostics(
    theta: &[f64],
    d_den: &[f64],
    d_nrg_seg: &[f64],
    d_nrg_raise: &[f64],
) -> ProjectionReceipts {
    assert!(
        d_den.len() == theta.len()
            && d_nrg_seg.len() == theta.len()
            && d_nrg_raise.len() == theta.len(),
        "anchor experiment: the four diagnostic slices cover the same interior \
         (theta {}, d_den {}, d_nrg_seg {}, d_nrg_raise {})",
        theta.len(),
        d_den.len(),
        d_nrg_seg.len(),
        d_nrg_raise.len()
    );
    let mut r = ProjectionReceipts::empty();
    for i in 0..theta.len() {
        assert!(
            theta[i].is_finite() && (0.0..=1.0).contains(&theta[i]),
            "anchor experiment: unclassifiable theta {} at interior index {i}",
            theta[i]
        );
        if theta[i] < 1.0 {
            r.projected_cells += 1;
            if theta[i] < r.min_theta {
                r.min_theta = theta[i];
            }
            r.mass.add(d_den[i]);
            r.energy_segment.add(d_nrg_seg[i]);
            r.energy_raise.add(d_nrg_raise[i]);
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
pub fn record_pass(receipts: &ProjectionReceipts, injection_weight: f64) {
    assert!(
        injection_weight.is_finite() && injection_weight > 0.0 && injection_weight <= 1.0,
        "anchor experiment: injection weight {injection_weight} outside the convex range (0, 1]"
    );
    let mut b = book();
    b.require_session("record");
    b.passes_total += 1;
    let pass = b.passes_total;
    if receipts.fired && b.first_attempted_pass.is_none() {
        b.first_attempted_pass = Some(pass);
    }
    if receipts.fired && b.pending_first_pass.is_none() {
        b.pending_first_pass = Some(pass);
    }
    b.attempted.fold_pass(receipts, injection_weight);
    b.pending.fold_pass(receipts, injection_weight);
}

/// the accepted-step transaction boundary: promote the complete pending-step
/// bucket into the accepted totals, stamping the accepted-first fire with the
/// post-step simulation time and iteration of the state the step produced. a
/// pass-free step is a cheap no-op.
pub fn step_commit(time: f64, iteration: u64) {
    let mut b = book();
    b.require_session("commit");
    if b.pending.passes == 0 {
        return;
    }
    let pending = b.pending;
    b.accepted.fold_totals(&pending);
    if b.first_accepted.is_none() {
        if let Some(pass) = b.pending_first_pass {
            b.first_accepted = Some(FirstFire {
                pass,
                time,
                iteration,
            });
        }
    }
    b.pending = ExperimentTotals::identity();
    b.pending_first_pass = None;
}

/// the retry/rollback boundary: the pending-step bucket describes states the
/// rollback discarded, so it empties without touching the accepted totals.
pub fn step_discard() {
    let mut b = book();
    b.require_session("discard");
    if b.pending.passes == 0 {
        return;
    }
    b.pending = ExperimentTotals::identity();
    b.pending_first_pass = None;
}

/// the (attempted, accepted) totals — queryable during a session and after it
/// closes, until the next open clears the book.
pub fn experiment_report() -> (ExperimentTotals, ExperimentTotals) {
    let b = book();
    (b.attempted, b.accepted)
}

/// the first-fire records: the attempted-first global pass index and the
/// accepted-first fire with its post-step time and iteration.
pub fn experiment_first_report() -> (Option<u64>, Option<FirstFire>) {
    let b = book();
    (b.first_attempted_pass, b.first_accepted)
}

/// clear the book between runs. a session must be closed first.
pub fn experiment_reset() {
    let mut b = book();
    if let Session::Open(owner) = b.session {
        drop(b);
        panic!("anchor-experiment reset inside an open session (owner {owner:?})");
    }
    *b = Book::cleared();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the global book serializes its tests.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static GATE: OnceLock<Mutex<()>> = OnceLock::new();
        GATE.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn fired(mass: f64) -> ProjectionReceipts {
        let mut r = ProjectionReceipts::empty();
        r.fired = true;
        r.projected_cells = 1;
        r.min_theta = 0.5;
        r.mass.add(mass);
        r
    }

    /// two accepted-looking substages followed by a step retry: everything
    /// stays attempted, nothing enters accepted.
    #[test]
    fn a_retried_step_leaves_no_accepted_receipts() {
        let _g = lock();
        let s = open_session();
        record_pass(&fired(1.0), 1.0);
        record_pass(&fired(2.0), 1.0);
        step_discard();
        let (attempted, accepted) = experiment_report();
        assert_eq!(attempted.passes, 2);
        assert_eq!(attempted.intervention_mass.signed, 3.0);
        assert_eq!(accepted.passes, 0);
        let (fa, facc) = experiment_first_report();
        assert_eq!(fa, Some(1), "the rejected attempt still recorded first");
        assert_eq!(facc, None);
        drop(s);
    }

    /// a fully accepted multi-substage step promotes its accumulated pending
    /// bucket exactly once, stamping the accepted-first fire. the rk2 weights
    /// separate the ledgers: the predictor pass folds at 1/2 into the injected
    /// totals while the intervention totals keep the raw sum.
    #[test]
    fn an_accepted_step_promotes_its_whole_bucket_once() {
        let _g = lock();
        let s = open_session();
        record_pass(&fired(1.0), 0.5);
        record_pass(&fired(2.0), 1.0);
        step_commit(3.25, 7);
        step_commit(4.0, 8); // pass-free: a second commit moves nothing
        let (attempted, accepted) = experiment_report();
        assert_eq!(attempted.passes, 2);
        assert_eq!(accepted.passes, 2);
        assert_eq!(accepted.intervention_mass.signed, 3.0);
        assert_eq!(accepted.injected_mass.signed, 0.5 * 1.0 + 1.0 * 2.0);
        assert_eq!(accepted.injected_mass.abs, 2.5);
        let (_, facc) = experiment_first_report();
        assert_eq!(
            facc,
            Some(FirstFire {
                pass: 1,
                time: 3.25,
                iteration: 7
            })
        );
        drop(s);
    }

    /// retry followed by an accepted replay: only the replayed step's passes
    /// enter the accepted totals, and the accepted-first fire is the replay's.
    #[test]
    fn only_the_accepted_replay_promotes() {
        let _g = lock();
        let s = open_session();
        record_pass(&fired(10.0), 1.0);
        step_discard();
        record_pass(&fired(2.0), 1.0);
        step_commit(1.5, 3);
        let (attempted, accepted) = experiment_report();
        assert_eq!(attempted.passes, 2);
        assert_eq!(accepted.passes, 1);
        assert_eq!(accepted.intervention_mass.signed, 2.0);
        let (fa, facc) = experiment_first_report();
        assert_eq!(fa, Some(1));
        assert_eq!(
            facc,
            Some(FirstFire {
                pass: 2,
                time: 1.5,
                iteration: 3
            })
        );
        drop(s);
    }

    /// closing a session while a timestep's receipts are still pending —
    /// neither committed nor discarded — fails loudly, and leaves the book
    /// clean for the next run.
    #[test]
    fn an_unresolved_timestep_at_close_fails_loudly() {
        let _g = lock();
        let unresolved = std::panic::catch_unwind(|| {
            let s = open_session();
            record_pass(&fired(1.0), 1.0);
            drop(s); // no commit, no discard: the classification is lost
        });
        assert!(
            unresolved.is_err(),
            "a session dropped mid-timestep must panic"
        );
        // the drop released the session despite the panic: the next run opens.
        let s = open_session();
        drop(s);
    }

    /// a second thread fails loudly instead of interleaving the transaction.
    #[test]
    fn a_second_owner_panics() {
        let _g = lock();
        let s = open_session();
        record_pass(&fired(1.0), 1.0);
        let stray = std::thread::spawn(|| record_pass(&ProjectionReceipts::empty(), 1.0));
        assert!(
            stray.join().is_err(),
            "a second thread must not enter the session"
        );
        step_discard(); // resolve the owning-thread pass before the session closes
        drop(s);
    }

    /// a second open on the same thread fails loudly: two interleaved runs
    /// must never merge their ledgers.
    #[test]
    fn a_same_thread_overlap_panics() {
        let _g = lock();
        let s = open_session();
        let overlap = std::panic::catch_unwind(|| open_session());
        assert!(overlap.is_err(), "an overlapping open must panic");
        drop(s);
    }

    /// booking or closing the transaction with no session open fails loudly —
    /// which also pins the driver contract that a production run (no
    /// experiment named, no session) reaches no transaction hook.
    #[test]
    fn a_forgotten_session_fails_loudly() {
        let _g = lock();
        experiment_reset();
        assert!(std::panic::catch_unwind(|| record_pass(&fired(1.0), 1.0)).is_err());
        assert!(std::panic::catch_unwind(|| step_commit(0.0, 0)).is_err());
        assert!(std::panic::catch_unwind(step_discard).is_err());
    }

    /// completed totals stay queryable after the session closes; the next
    /// open starts a cleared book.
    #[test]
    fn results_stay_queryable_after_the_session_closes() {
        let _g = lock();
        let s = open_session();
        record_pass(&fired(4.0), 1.0);
        step_commit(2.0, 5);
        drop(s);
        let (attempted, accepted) = experiment_report();
        assert_eq!(attempted.passes, 1);
        assert_eq!(accepted.intervention_mass.signed, 4.0);
        let s = open_session();
        let (attempted, _) = experiment_report();
        assert_eq!(attempted.passes, 0, "the next open clears the book");
        drop(s);
    }

    /// an unclassifiable theta panics rather than reading as "did not fire",
    /// and mismatched diagnostic slices fail as a named invariant.
    #[test]
    fn nan_theta_fails_loudly() {
        let _g = lock();
        let bad = std::panic::catch_unwind(|| {
            receipts_from_diagnostics(&[f64::NAN], &[0.0], &[0.0], &[0.0])
        });
        assert!(bad.is_err());
        let big =
            std::panic::catch_unwind(|| receipts_from_diagnostics(&[1.5], &[0.0], &[0.0], &[0.0]));
        assert!(big.is_err());
        let short =
            std::panic::catch_unwind(|| receipts_from_diagnostics(&[1.0], &[0.0], &[], &[0.0]));
        assert!(
            short.is_err(),
            "a short diagnostic slice is a named failure"
        );
    }
}
