// =============================================================================
// conservation_ledger.rs
//
// the production receipts of a run's conserved totals: what crossed each domain wall, what the
// viscous operator moved and heated, and the lowest density and specific internal energy any stage
// produced. the shared stage and post-step folds book them; the step transaction commits a step's
// receipts when the step is accepted and discards them on retry, so the accepted totals describe
// only the solution. a run's change in mass, angular momentum and E + rho Phi then closes against
// these receipts:
//   mass:              change + outflow = 0
//   angular momentum:  change + advective outflow = viscous torque in
//   E + rho Phi:       change + advective outflow = viscous stress work in
// the first-order flux correction books the cells it corrected: their count summed over stages,
// the largest count in one stage, and the count per cell (`take_corrections`), so a reader sees how
// often and where the high-order update left the admissible set.
// the viscous heating is internal conversion: the operator's change of internal energy, its total
// energy change less its kinetic energy change. the operator's own total energy change equals the
// stress work through the walls, and the ledger keeps both so a reader checks the identity.
//
// each stage's wall transport is a rate the kernel set reports from the stage's stored fluxes;
// the fold books it times dt times the stage's weight ac times its downstream shu-osher weight,
// the factor by which the stage's flux reaches the accepted step (rk2: 1/2 and 1/2 overall).
//
// ownership follows the projection ledger: the book is thread-local, a run opens a scope on its
// execution thread, and the folds book only while a scope is open. a single grid is the shape the
// step transaction models; other run shapes open no scope and book nothing.
//
// usage:
//   let scope = conservation_ledger::open_scope();
//   ... run ...
//   let (attempted, accepted) = conservation_ledger::ledger_report();
// =============================================================================

use std::cell::Cell;
use std::marker::PhantomData;

/// one quantity set crossing a surface: the net outward mass, angular momentum about the symmetry
/// axis, and total energy with the potential energy the mass carries, and the gross mass entering
/// through the faces whose mass flux points inward, a nonnegative amount
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Transport {
    pub mass: f64,
    pub angular_momentum: f64,
    pub energy: f64,
    pub mass_in: f64,
}

impl Transport {
    fn add_scaled(&mut self, o: &Transport, w: f64) {
        self.mass += w * o.mass;
        self.angular_momentum += w * o.angular_momentum;
        self.energy += w * o.energy;
        self.mass_in += w * o.mass_in;
    }
}

/// an extreme value and the interior cell holding it
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extreme {
    pub value: f64,
    pub cell: [isize; 3],
}

impl Default for Extreme {
    fn default() -> Self {
        Extreme { value: f64::INFINITY, cell: [0; 3] }
    }
}

impl Extreme {
    fn lower(&mut self, o: &Extreme) {
        // a NaN is lower than everything: it reports the cell that holds it
        if o.value.is_nan() || o.value < self.value {
            *self = *o;
        }
    }
}

/// the viscous operator's receipt for one application
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViscousReceipt {
    /// the torque and the stress work the walls passed into the domain, time-integrated
    pub torque_in: f64,
    pub work_in: f64,
    /// the operator's change of total energy, of kinetic energy and of angular momentum
    pub energy_change: f64,
    pub kinetic_change: f64,
    pub angular_momentum_change: f64,
}

/// the totals of a run's receipts
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConservationLedger {
    /// outward transport through each wall, indexed by axis and side (0 low, 1 high)
    pub outflow: [[Transport; 2]; 3],
    pub viscous_torque_in: f64,
    pub viscous_work_in: f64,
    /// internal energy the viscous operator created: its total less its kinetic energy change
    pub viscous_heating: f64,
    pub viscous_energy_change: f64,
    pub viscous_angular_momentum_change: f64,
    pub lowest_density: Extreme,
    pub lowest_specific_internal: Extreme,
    pub stages: u64,
    pub viscous_steps: u64,
    /// the cells the first-order flux correction corrected, summed over stages
    pub corrected_cell_stages: u64,
    /// the largest number of cells the correction corrected in one stage
    pub max_corrected_in_stage: u64,
}

impl ConservationLedger {
    /// the sum over every wall
    pub fn total_outflow(&self) -> Transport {
        let mut t = Transport::default();
        for side in self.outflow.iter().flatten() {
            t.add_scaled(side, 1.0);
        }
        t
    }

    fn fold(&mut self, o: &ConservationLedger) {
        for (mine, theirs) in self.outflow.iter_mut().flatten().zip(o.outflow.iter().flatten()) {
            mine.add_scaled(theirs, 1.0);
        }
        self.viscous_torque_in += o.viscous_torque_in;
        self.viscous_work_in += o.viscous_work_in;
        self.viscous_heating += o.viscous_heating;
        self.viscous_energy_change += o.viscous_energy_change;
        self.viscous_angular_momentum_change += o.viscous_angular_momentum_change;
        self.lowest_density.lower(&o.lowest_density);
        self.lowest_specific_internal.lower(&o.lowest_specific_internal);
        self.stages += o.stages;
        self.viscous_steps += o.viscous_steps;
        self.corrected_cell_stages += o.corrected_cell_stages;
        self.max_corrected_in_stage = self.max_corrected_in_stage.max(o.max_corrected_in_stage);
    }

    fn is_identity(&self) -> bool {
        self.stages == 0 && self.viscous_steps == 0 && self.corrected_cell_stages == 0
    }
}

#[derive(Clone, Copy, Default)]
struct Book {
    open: bool,
    /// a scope opened on this thread: its totals describe this thread's run after the scope closes
    opened: bool,
    attempted: ConservationLedger,
    pending: ConservationLedger,
    accepted: ConservationLedger,
    /// the accepted minima since the last [`take_window`]
    window: (Extreme, Extreme),
}

/// the per-cell correction counts, pending for the open step and accepted for the run
#[derive(Default)]
struct CorrectionMap {
    pending: std::collections::HashMap<[isize; 3], u64>,
    accepted: std::collections::HashMap<[isize; 3], u64>,
}

thread_local! {
    static CORRECTIONS: std::cell::RefCell<CorrectionMap> = std::cell::RefCell::new(CorrectionMap::default());
    static BOOK: Cell<Book> = Cell::new(Book::default());
    static RESTART_SEED: Cell<Option<ConservationLedger>> = const { Cell::new(None) };
}

/// the accepted totals a restarted run continues, set on the run's thread before its driver
/// opens the scope
pub fn set_restart_seed(accepted: ConservationLedger) {
    RESTART_SEED.with(|cell| cell.set(Some(accepted)));
}

/// the restart seed, once
pub fn take_restart_seed() -> Option<ConservationLedger> {
    RESTART_SEED.with(|cell| cell.take())
}

/// the accepted lowest density and specific internal energy since the previous call, which a
/// checkpoint writer records beside the cumulative totals
pub fn take_window() -> (Extreme, Extreme) {
    with_book(|b| std::mem::take(&mut b.window))
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
    assert!(b.open, "conservation ledger {what} outside a scope: the run opens one before any fold books");
}

/// whether this thread's scope is open; the folds book only then
pub fn scope_is_open() -> bool {
    BOOK.with(|cell| cell.get().open)
}

/// whether a scope opened on this thread; its run's totals stay readable after it closes, for the
/// final checkpoint the run writes after its march
pub fn scope_was_opened() -> bool {
    BOOK.with(|cell| cell.get().opened)
}

/// the open run scope. dropping it closes the scope and leaves the totals queryable
pub struct LedgerScope {
    _single_thread: PhantomData<*const ()>,
}

/// open this thread's scope with a cleared book
pub fn open_scope() -> LedgerScope {
    CORRECTIONS.with(|c| *c.borrow_mut() = CorrectionMap::default());
    open_scope_from(ConservationLedger::default())
}

/// open this thread's scope with accepted totals carried from a restart checkpoint
pub fn open_scope_from(accepted: ConservationLedger) -> LedgerScope {
    with_book(|b| {
        assert!(!b.open, "conservation ledger scope already open on this thread");
        *b = Book { open: true, opened: true, accepted, ..Book::default() };
    });
    // the accepted per-cell counts are the restart seed's, set before the scope opens
    CORRECTIONS.with(|c| c.borrow_mut().pending.clear());
    LedgerScope { _single_thread: PhantomData }
}

impl LedgerScope {
    /// the accepted totals booked under this scope
    pub fn accepted(&self) -> ConservationLedger {
        BOOK.with(|cell| cell.get().accepted)
    }
}

impl Drop for LedgerScope {
    fn drop(&mut self) {
        let unresolved = with_book(|b| {
            let unresolved = !b.pending.is_identity();
            b.pending = ConservationLedger::default();
            b.open = false;
            unresolved
        });
        if !std::thread::panicking() {
            assert!(!unresolved, "conservation ledger scope closed with a step neither committed nor discarded");
        }
    }
}

/// book one stage's wall transport rates, at `weight` = dt x ac x the stage's downstream weight
pub fn record_stage(rates: &[[Transport; 2]; 3], weight: f64) {
    assert!(weight.is_finite() && weight >= 0.0, "conservation ledger: stage weight {weight}");
    with_book(|b| {
        require_open(b, "record_stage");
        for target in [&mut b.attempted, &mut b.pending] {
            for (mine, rate) in target.outflow.iter_mut().flatten().zip(rates.iter().flatten()) {
                mine.add_scaled(rate, weight);
            }
            target.stages += 1;
        }
    });
}

/// the cells one stage's first-order flux correction corrected, each by its whole-grid index
pub fn record_correction(cells: &[[isize; 3]]) {
    let count = cells.len() as u64;
    with_book(|b| {
        require_open(b, "record_correction");
        for target in [&mut b.attempted, &mut b.pending] {
            target.corrected_cell_stages += count;
            target.max_corrected_in_stage = target.max_corrected_in_stage.max(count);
        }
    });
    CORRECTIONS.with(|c| {
        let mut c = c.borrow_mut();
        for cell in cells {
            *c.pending.entry(*cell).or_insert(0) += 1;
        }
    });
}

/// the accepted correction count per cell, sorted by cell
pub fn corrections_report() -> Vec<([isize; 3], u64)> {
    CORRECTIONS.with(|c| {
        let mut v: Vec<_> = c.borrow().accepted.iter().map(|(k, n)| (*k, *n)).collect();
        v.sort();
        v
    })
}

/// the accepted per-cell correction counts a restarted run continues, set before its scope opens
pub fn set_corrections_seed(counts: Vec<([isize; 3], u64)>) {
    CORRECTIONS.with(|c| {
        let mut c = c.borrow_mut();
        c.pending.clear();
        c.accepted = counts.into_iter().collect();
    });
}

/// book the lowest density and specific internal energy one stage's output holds
pub fn record_extremes(density: Extreme, specific_internal: Extreme) {
    with_book(|b| {
        require_open(b, "record_extremes");
        for target in [&mut b.attempted, &mut b.pending] {
            target.lowest_density.lower(&density);
            target.lowest_specific_internal.lower(&specific_internal);
        }
    });
}

/// book one application of the viscous operator
pub fn record_viscous(receipt: &ViscousReceipt) {
    with_book(|b| {
        require_open(b, "record_viscous");
        for target in [&mut b.attempted, &mut b.pending] {
            target.viscous_torque_in += receipt.torque_in;
            target.viscous_work_in += receipt.work_in;
            target.viscous_heating += receipt.energy_change - receipt.kinetic_change;
            target.viscous_energy_change += receipt.energy_change;
            target.viscous_angular_momentum_change += receipt.angular_momentum_change;
            target.viscous_steps += 1;
        }
    });
}

/// the accepted step: its pending receipts join the accepted totals
pub fn step_commit() {
    with_book(|b| {
        require_open(b, "commit");
        let pending = b.pending;
        b.accepted.fold(&pending);
        b.window.0.lower(&pending.lowest_density);
        b.window.1.lower(&pending.lowest_specific_internal);
        b.pending = ConservationLedger::default();
    });
    CORRECTIONS.with(|c| {
        let mut c = c.borrow_mut();
        let pending = std::mem::take(&mut c.pending);
        for (cell, n) in pending {
            *c.accepted.entry(cell).or_insert(0) += n;
        }
    });
}

/// the retried step: its pending receipts describe a discarded state and leave the totals alone
pub fn step_discard() {
    with_book(|b| {
        require_open(b, "discard");
        b.pending = ConservationLedger::default();
    });
    CORRECTIONS.with(|c| c.borrow_mut().pending.clear());
}

/// this thread's (attempted, accepted) totals, queryable during a run and after its scope closes
pub fn ledger_report() -> (ConservationLedger, ConservationLedger) {
    BOOK.with(|cell| {
        let b = cell.get();
        (b.attempted, b.accepted)
    })
}

/// the totals of a run executed on another thread, handed to this one
pub fn ledger_adopt(attempted: ConservationLedger, accepted: ConservationLedger) {
    with_book(|b| {
        assert!(!b.open, "conservation ledger adopted inside an open scope");
        b.attempted = attempted;
        b.accepted = accepted;
    });
}

/// the totals as a flat list of numbers, for checkpoint metadata: the outflow by axis, side and
/// quantity, the viscous terms, the extremes with their cells, and the counts
pub fn to_values(l: &ConservationLedger) -> Vec<f64> {
    let mut out = Vec::new();
    for t in l.outflow.iter().flatten() {
        out.extend([t.mass, t.angular_momentum, t.energy, t.mass_in]);
    }
    out.extend([l.viscous_torque_in, l.viscous_work_in, l.viscous_heating, l.viscous_energy_change, l.viscous_angular_momentum_change]);
    for e in [l.lowest_density, l.lowest_specific_internal] {
        out.push(e.value);
        out.extend(e.cell.map(|c| c as f64));
    }
    out.extend([l.stages as f64, l.viscous_steps as f64, l.corrected_cell_stages as f64, l.max_corrected_in_stage as f64]);
    out
}

/// the inverse of [`to_values`]
pub fn from_values(v: &[f64]) -> ConservationLedger {
    // a ledger written before the correction counts holds the 39 values that precede them
    assert!(v.len() == VALUES || v.len() == VALUES - 2, "a conservation ledger holds {VALUES} values; {} given", v.len());
    let mut l = ConservationLedger::default();
    for (k, t) in l.outflow.iter_mut().flatten().enumerate() {
        *t = Transport { mass: v[4 * k], angular_momentum: v[4 * k + 1], energy: v[4 * k + 2], mass_in: v[4 * k + 3] };
    }
    let rest = &v[24..];
    l.viscous_torque_in = rest[0];
    l.viscous_work_in = rest[1];
    l.viscous_heating = rest[2];
    l.viscous_energy_change = rest[3];
    l.viscous_angular_momentum_change = rest[4];
    let extreme = |at: usize| Extreme { value: rest[at], cell: [rest[at + 1] as isize, rest[at + 2] as isize, rest[at + 3] as isize] };
    l.lowest_density = extreme(5);
    l.lowest_specific_internal = extreme(9);
    l.stages = rest[13] as u64;
    l.viscous_steps = rest[14] as u64;
    if v.len() == VALUES {
        l.corrected_cell_stages = rest[15] as u64;
        l.max_corrected_in_stage = rest[16] as u64;
    }
    l
}

/// the length of [`to_values`]
pub const VALUES: usize = 24 + 5 + 8 + 4;

#[cfg(test)]
mod tests {
    use super::*;

    fn rates(mass: f64) -> [[Transport; 2]; 3] {
        let mut r = [[Transport::default(); 2]; 3];
        r[0][1] = Transport { mass, angular_momentum: 2.0 * mass, energy: 3.0 * mass, mass_in: 4.0 * mass };
        r
    }

    #[test]
    fn a_retried_step_leaves_the_accepted_totals_and_enters_the_attempted() {
        let scope = open_scope();
        record_stage(&rates(1.0), 0.5);
        record_extremes(Extreme { value: 2.0, cell: [1, 2, 0] }, Extreme { value: -1.0, cell: [3, 4, 0] });
        step_discard();
        record_stage(&rates(1.0), 0.25);
        record_viscous(&ViscousReceipt { torque_in: 1.0, work_in: 2.0, energy_change: 2.0, kinetic_change: 0.5, angular_momentum_change: 1.0 });
        step_commit();
        let accepted = scope.accepted();
        drop(scope);
        let (attempted, _) = ledger_report();
        assert_eq!(accepted.outflow[0][1], Transport { mass: 0.25, angular_momentum: 0.5, energy: 0.75, mass_in: 1.0 });
        assert_eq!(attempted.outflow[0][1].mass, 0.75);
        assert_eq!(accepted.stages, 1);
        assert_eq!(attempted.stages, 2);
        assert_eq!(accepted.lowest_specific_internal, Extreme::default(), "a discarded stage's extreme reached the accepted totals");
        assert_eq!(attempted.lowest_specific_internal.cell, [3, 4, 0]);
        assert_eq!(accepted.viscous_heating, 1.5);
        assert_eq!(accepted.viscous_work_in, 2.0);
    }

    #[test]
    fn the_totals_survive_their_flat_form_and_a_restart_continues_them() {
        let scope = open_scope();
        record_stage(&rates(0.125), 1.0);
        record_extremes(Extreme { value: 0.5, cell: [7, 8, 0] }, Extreme { value: f64::NAN, cell: [9, 1, 0] });
        record_viscous(&ViscousReceipt { torque_in: -1.0, work_in: 0.25, energy_change: 0.25, kinetic_change: -0.5, angular_momentum_change: -1.0 });
        step_commit();
        let first = scope.accepted();
        drop(scope);
        assert!(scope_was_opened() && !scope_is_open(), "a closed scope leaves its run's totals readable");
        // the bit patterns: the totals hold a NaN extreme, which compares unequal to itself
        let bits = |l: &ConservationLedger| to_values(l).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&ledger_report().1), bits(&first), "the totals after the scope closed");
        let values = to_values(&first);
        assert_eq!(values.len(), VALUES);
        let back = from_values(&values);
        assert_eq!(back.outflow, first.outflow);
        assert_eq!(back.lowest_density, first.lowest_density);
        assert!(back.lowest_specific_internal.value.is_nan() && back.lowest_specific_internal.cell == [9, 1, 0]);
        set_restart_seed(back);
        let resumed = open_scope_from(take_restart_seed().expect("the seed"));
        assert!(take_restart_seed().is_none(), "the seed is taken once");
        record_stage(&rates(0.125), 1.0);
        step_commit();
        assert_eq!(resumed.accepted().outflow[0][1].mass, 0.25);
        assert_eq!(resumed.accepted().stages, 2);
        drop(resumed);
    }

    #[test]
    fn a_retried_step_leaves_the_corrections_and_the_accepted_step_books_them() {
        let scope = open_scope();
        record_correction(&[[3, 4, 0], [5, 6, 0]]);
        step_discard();
        assert!(corrections_report().is_empty(), "a discarded step's corrections reached the accepted map");
        record_correction(&[[3, 4, 0]]);
        step_commit();
        record_correction(&[[3, 4, 0], [7, 1, 0], [8, 1, 0]]);
        step_commit();
        let accepted = scope.accepted();
        drop(scope);
        assert_eq!((accepted.corrected_cell_stages, accepted.max_corrected_in_stage), (4, 3));
        assert_eq!(corrections_report(), vec![([3, 4, 0], 2), ([7, 1, 0], 1), ([8, 1, 0], 1)]);
        let (attempted, _) = ledger_report();
        assert_eq!(attempted.corrected_cell_stages, 6, "the attempted totals hold the discarded correction");
        // the flat form carries the counts, and a ledger written before them reads as zero
        let back = from_values(&to_values(&accepted));
        assert_eq!((back.corrected_cell_stages, back.max_corrected_in_stage), (4, 3));
        let older = from_values(&to_values(&accepted)[..VALUES - 2]);
        assert_eq!((older.corrected_cell_stages, older.stages), (0, accepted.stages));
    }

    #[test]
    fn the_window_holds_the_accepted_minima_since_it_was_last_taken() {
        let scope = open_scope();
        record_extremes(Extreme { value: 1.0, cell: [1, 0, 0] }, Extreme { value: 0.5, cell: [2, 0, 0] });
        step_commit();
        assert_eq!(take_window().1.value, 0.5);
        record_extremes(Extreme { value: 3.0, cell: [3, 0, 0] }, Extreme { value: 0.75, cell: [4, 0, 0] });
        step_discard();
        assert_eq!(take_window(), (Extreme::default(), Extreme::default()), "a discarded stage reached the window");
        record_extremes(Extreme { value: 3.0, cell: [3, 0, 0] }, Extreme { value: 0.75, cell: [4, 0, 0] });
        step_commit();
        let (density, specific) = take_window();
        assert_eq!((density.value, specific.value, specific.cell), (3.0, 0.75, [4, 0, 0]));
        assert_eq!(scope.accepted().lowest_specific_internal.value, 0.5, "the cumulative minimum");
    }
}
