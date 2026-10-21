// =============================================================================
// dual_energy_ledger.rs
//
// the run-owned accounting of the dual-energy synchronization: the total energy
// each stage's entropy-path cells were given or lost when their thermal energy
// was replaced by the entropy-derived one. this is energy the conservative
// update did not produce, so it is booked signed, per sign, never netted.
//
// each stage's synchronization arrives as a [`CorrectionReceipt`] keyed by the
// stage's SSP coefficients. a first-order redo of the same stage books under the
// same key and replaces the stage's earlier receipt, because the redo replaces
// the state that receipt described. the sim owns the transaction exactly as for
// the guard ledger: receipts accumulate into a pending-step bucket the step
// transaction commits on acceptance and discards on rejection or a halt.
//
// ownership: thread-local, one book per run thread; the caller opens a scope for
// the run before any stage books.
//
// usage:
//  let scope = dual_energy_ledger::open_scope();
//  dual_energy_ledger::record_stage(key, &receipt);   // per stage
//  dual_energy_ledger::step_commit();                 // the step transaction
//  let totals = scope.accepted();
// =============================================================================

use std::cell::RefCell;
use std::marker::PhantomData;

/// one stage's synchronization: the energy it added and removed, where, and against which
/// thermal energies
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectionReceipt {
    /// the summed positive energy corrections, (p_S - p_E) dV / (gamma - 1) > 0
    pub added: f64,
    /// the summed negative corrections, kept as a negative number
    pub removed: f64,
    /// cells whose thermal energy came from the entropy
    pub entropy_cells: u64,
    /// cells the switch would have put on the entropy that a detected shock kept on the energy
    pub shock_cells: u64,
    /// cells inside a detected shock whose conservative energy was not admissible; the
    /// correction ladder decides them
    pub invalid_shock_cells: u64,
    /// the largest correction against a cell's own thermal energy, |p_S - p_E| / p_S
    pub worst_relative: f64,
    /// where the largest relative correction sat (grid index, per axis)
    pub worst_cell: Vec<isize>,
    /// cells whose relative correction exceeds 0.1 and 1
    pub above_tenth: u64,
    pub above_one: u64,
    /// the thermal energy of the whole interior and of the entropy-path cells after the stage
    pub thermal_domain: f64,
    pub thermal_entropy_cells: f64,
}

/// the accumulated corrections of a set of stages: per-sign sums, counts and the worst
/// relative correction among them
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectionTotals {
    pub added: f64,
    pub removed: f64,
    pub entropy_cells: u64,
    pub shock_cells: u64,
    pub invalid_shock_cells: u64,
    pub worst_relative: f64,
    pub above_tenth: u64,
    pub above_one: u64,
    pub stages: u64,
}

impl CorrectionTotals {
    fn fold(&mut self, r: &CorrectionReceipt) {
        self.added += r.added;
        self.removed += r.removed;
        self.entropy_cells += r.entropy_cells;
        self.shock_cells += r.shock_cells;
        self.invalid_shock_cells += r.invalid_shock_cells;
        self.worst_relative = self.worst_relative.max(r.worst_relative);
        self.above_tenth += r.above_tenth;
        self.above_one += r.above_one;
        self.stages += 1;
    }

    fn merge(&mut self, o: &CorrectionTotals) {
        self.added += o.added;
        self.removed += o.removed;
        self.entropy_cells += o.entropy_cells;
        self.shock_cells += o.shock_cells;
        self.invalid_shock_cells += o.invalid_shock_cells;
        self.worst_relative = self.worst_relative.max(o.worst_relative);
        self.above_tenth += o.above_tenth;
        self.above_one += o.above_one;
        self.stages += o.stages;
    }
}

/// a stage's identity within one step attempt: the bits of its SSP coefficients (a0, ac)
pub type StageKey = (u64, u64);

#[derive(Default)]
struct Book {
    open: bool,
    accepted: CorrectionTotals,
    pending: Vec<(StageKey, CorrectionReceipt)>,
}

thread_local! {
    static BOOK: RefCell<Book> = RefCell::new(Book::default());
}

fn require_open(b: &Book, what: &str) {
    assert!(
        b.open,
        "dual-energy ledger {what} outside a scope: the run opens one dual-energy scope before any \
         stage synchronizes"
    );
}

/// whether this thread's dual-energy scope is open
pub fn scope_is_open() -> bool {
    BOOK.with(|b| b.borrow().open)
}

/// the open run scope; dropping it closes the scope
pub struct DualEnergyScope {
    _single_thread: PhantomData<*const ()>,
}

/// open this thread's run scope with an empty book. a scope already open panics.
pub fn open_scope() -> DualEnergyScope {
    BOOK.with(|b| {
        let mut b = b.borrow_mut();
        assert!(!b.open, "dual-energy ledger scope already open on this thread");
        *b = Book { open: true, ..Book::default() };
    });
    DualEnergyScope { _single_thread: PhantomData }
}

impl DualEnergyScope {
    /// the corrections of the accepted steps
    pub fn accepted(&self) -> CorrectionTotals {
        accepted()
    }
}

impl Drop for DualEnergyScope {
    fn drop(&mut self) {
        let unresolved = BOOK.with(|b| {
            let mut b = b.borrow_mut();
            b.open = false;
            std::mem::take(&mut b.pending)
        });
        if !std::thread::panicking() {
            assert!(
                unresolved.is_empty(),
                "dual-energy ledger scope closed with an unresolved timestep: pending corrections \
                 neither committed nor discarded"
            );
        }
    }
}

/// book one stage's synchronization into the pending step, replacing an earlier receipt of the
/// same stage (a first-order redo of it)
pub fn record_stage(key: StageKey, receipt: &CorrectionReceipt) {
    BOOK.with(|b| {
        let mut b = b.borrow_mut();
        require_open(&b, "record");
        match b.pending.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = receipt.clone(),
            None => b.pending.push((key, receipt.clone())),
        }
    });
}

/// the corrections of the steps accepted so far
pub fn accepted() -> CorrectionTotals {
    BOOK.with(|b| b.borrow().accepted.clone())
}

/// the corrections of the attempt in progress
pub fn pending() -> CorrectionTotals {
    BOOK.with(|b| {
        let mut t = CorrectionTotals::default();
        for (_, r) in &b.borrow().pending {
            t.fold(r);
        }
        t
    })
}

/// the accepted-step boundary: the pending stages join the accepted totals
pub fn step_commit() {
    BOOK.with(|b| {
        let mut b = b.borrow_mut();
        require_open(&b, "commit");
        let mut t = CorrectionTotals::default();
        for (_, r) in std::mem::take(&mut b.pending) {
            t.fold(&r);
        }
        b.accepted.merge(&t);
    });
}

/// the rejection boundary: the pending stages described a discarded attempt
pub fn step_discard() {
    BOOK.with(|b| {
        let mut b = b.borrow_mut();
        require_open(&b, "discard");
        b.pending.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(added: f64, removed: f64) -> CorrectionReceipt {
        CorrectionReceipt { added, removed, entropy_cells: 1, ..Default::default() }
    }

    #[test]
    fn a_redo_replaces_its_stage_and_a_rejection_books_nothing() {
        let scope = open_scope();
        record_stage((0, 1), &receipt(2.0, -1.0));
        record_stage((0, 1), &receipt(0.5, -0.25));
        record_stage((5, 5), &receipt(1.0, 0.0));
        assert_eq!(pending().added, 1.5);
        assert_eq!(pending().removed, -0.25);
        step_commit();
        record_stage((0, 1), &receipt(9.0, -9.0));
        step_discard();
        let t = scope.accepted();
        assert_eq!((t.added, t.removed, t.stages), (1.5, -0.25, 2));
    }

    #[test]
    fn positive_and_negative_corrections_never_cancel() {
        let scope = open_scope();
        record_stage((0, 1), &receipt(3.0, -3.0));
        step_commit();
        let t = scope.accepted();
        assert_eq!((t.added, t.removed), (3.0, -3.0));
    }
}
