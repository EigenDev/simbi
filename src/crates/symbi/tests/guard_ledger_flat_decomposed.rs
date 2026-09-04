// =============================================================================
// guard_ledger_flat_decomposed.rs
//
// run-owned guard diagnostics on the FLAT-decomposed driver (`evolve_scheduled` /
// `evolve_decomposed` in decomp.rs), the gpus>1-without-refinement path. an
// earlier decomposition test covered the refined-decomposed driver
// (`evolve_hierarchy_decomposed`); this closes the flat-decomposed driver, which
// runs its own guard scope and commits/discards at its own accepted-step boundary.
//
// a deterministic mock kernel set mints a chosen guard receipt on every substage
// and can reject a chosen substage, so the driver's accepted-step commit and
// retry discard are exercised without depending on physics tripping a real guard.
//
// gates:
// - two tiles accepting with the same acts sum into one run-owned total
//   (flat-decomposed aggregation, nonzero);
// - a rejected attempt's acts book into attempted but never into the accepted
//   totals the driver returns (retry discard);
// - two sequential runs report their own accepted totals;
// - two concurrent runs on separate threads report their own;
// - an observer break returns the accepted-so-far totals (an early-exit path the
//   driver actually supports).
// =============================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::guard_ledger::{self, GuardReceipt};
use symbi_sim::run_diagnostics::RunDiagnostics;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::{FofcDecision, FofcReport, KernelSet, SourceReplayOutcome};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const ACCEPT_TROUBLED: u64 = 2;
const REJECT_SENTINEL: u64 = 1000;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

struct GuardMock {
    calls: Arc<AtomicUsize>,
    reject_at: Option<usize>,
}

impl GuardMock {
    fn new(calls: &Arc<AtomicUsize>, reject_at: Option<usize>) -> Self {
        Self {
            calls: calls.clone(),
            reject_at,
        }
    }
}

impl KernelSet<1, 1, HostMemory, f64> for GuardMock {
    fn reconstruction_reach(&self) -> u8 {
        1
    }
    fn flux(&self, _store: &FieldStore<1, 1, HostMemory>, _dir: usize) {}
    fn c2p(&self, store: &FieldStore<1, 1, HostMemory>) {
        for cell in store.geom.interior.iter() {
            let rho = *store.fields.cons.den.view().at(cell);
            let mom = *store.fields.cons.mom[0].view().at(cell);
            let nrg = *store.fields.cons.nrg_field().unwrap().view().at(cell);
            store.fields.prim.rho.view_mut().set(cell, rho);
            store.fields.prim.vel[0].view_mut().set(cell, mom / rho);
            store
                .fields
                .prim
                .pre_field()
                .unwrap()
                .view_mut()
                .set(cell, (GAMMA - 1.0) * (nrg - 0.5 * mom * mom / rho));
        }
        store.mark_primitives_recovered();
    }
    fn godunov_stage(&self, _store: &FieldStore<1, 1, HostMemory>, _dt: f64, _a0: f64, _ac: f64) {}
    fn cfl(&self, _store: &FieldStore<1, 1, HostMemory>) -> f64 {
        0.1
    }
    fn ghost_fill(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn snapshot(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn fofc_active(&self) -> bool {
        true
    }
    fn fofc(
        &self,
        _store: &FieldStore<1, 1, HostMemory>,
        _dt: f64,
        _a0: f64,
        _ac: f64,
        _stage: u8,
    ) -> FofcReport {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.reject_at == Some(call) {
            return FofcReport::of_pass(
                REJECT_SENTINEL,
                1,
                1,
                SourceReplayOutcome::SharedRedo,
                FofcDecision::RetryStep,
            )
            .with_guards(GuardReceipt::of_pass(REJECT_SENTINEL, 0, 1, 0));
        }
        FofcReport::of_pass(
            ACCEPT_TROUBLED,
            0,
            0,
            SourceReplayOutcome::SharedRedo,
            FofcDecision::Accept,
        )
        .with_guards(GuardReceipt::of_pass(ACCEPT_TROUBLED, 0, 0, 0))
    }
    fn horizon_accretion(
        &self,
        _store: &FieldStore<1, 1, HostMemory>,
        _diagnostic_radius: f64,
    ) -> (f64, f64) {
        (0.0, 0.0)
    }
}

fn tile(origin: f64, lo: BoundaryType, hi: BoundaryType) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([8])
        .origin([origin])
        .spacing([1.0 / 16.0])
        .boundaries(Boundaries::per_axis([[lo, hi]]))
        .timestepping(Timestepping::Euler)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build()
}

/// two flat-decomposed tiles, each driven by a mock, marched one step with a
/// break-controlled observer. returns the driver's run-owned diagnostics.
fn run_two_tiles(
    calls: &Arc<AtomicUsize>,
    reject_at: Option<usize>,
    t_final: f64,
    mut observe: impl FnMut(u64) -> std::ops::ControlFlow<()>,
) -> RunDiagnostics {
    let mut tiles = vec![
        (
            tile(0.0, BoundaryType::Outflow, BoundaryType::CoarseFine),
            GuardMock::new(calls, reject_at),
        ),
        (
            tile(0.5, BoundaryType::CoarseFine, BoundaryType::Outflow),
            GuardMock::new(calls, reject_at),
        ),
    ];
    let devices: Vec<i32> = vec![0; tiles.len()];
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(
        &mut stores,
        &kernels,
        [2],
        &devices,
        Timestepping::Euler,
        0.0,
        t_final,
        1,
        &LocalCopy,
        |iter, _, _| observe(iter),
    )
}

/// two tiles accepting the same acts each substage sum into one run-owned total.
#[test]
fn two_tiles_sum_into_one_run_owned_total() {
    let calls = Arc::new(AtomicUsize::new(0));
    let diag = run_two_tiles(&calls, None, 0.05, |_| std::ops::ControlFlow::Continue(()));
    assert!(
        diag.guards.troubled_cells.total >= 2 * ACCEPT_TROUBLED,
        "the flat-decomposed accept boundary dropped a tile's acts (total {})",
        diag.guards.troubled_cells.total
    );
    assert_eq!(
        diag.guards.troubled_cells.total % ACCEPT_TROUBLED,
        0,
        "the accepted total is not a whole number of per-substage acts"
    );
    // the flat-decomposed driver opens no projection scope, so its projection is empty.
    assert_eq!(diag.projection.passes, 0);
}

/// a rejected attempt's acts book into attempted but never into the accepted
/// totals the driver returns: the collective rollback discards the pending bucket.
#[test]
fn a_rejected_attempt_is_absent_from_the_accepted_guards() {
    let calls = Arc::new(AtomicUsize::new(0));
    let diag = run_two_tiles(&calls, Some(1), 0.05, |_| {
        std::ops::ControlFlow::Continue(())
    });
    let (attempted, _) = guard_ledger::report();
    assert!(
        attempted.troubled_cells.total >= REJECT_SENTINEL,
        "the rejected attempt never booked into the attempted totals"
    );
    assert!(
        diag.guards.troubled_cells.total > 0,
        "no accepted guard act booked; the gate is vacuous"
    );
    assert!(
        diag.guards.troubled_cells.total < REJECT_SENTINEL,
        "the rejected attempt's acts survived into the accepted totals ({})",
        diag.guards.troubled_cells.total
    );
}

/// two sequential flat-decomposed runs report their own accepted totals.
#[test]
fn sequential_runs_report_their_own() {
    let a = run_two_tiles(&Arc::new(AtomicUsize::new(0)), None, 0.05, |_| {
        std::ops::ControlFlow::Continue(())
    });
    let b = run_two_tiles(&Arc::new(AtomicUsize::new(0)), None, 0.05, |_| {
        std::ops::ControlFlow::Continue(())
    });
    assert!(a.guards.troubled_cells.total > 0);
    assert_eq!(a.guards.troubled_cells.total, b.guards.troubled_cells.total);
}

/// two concurrent flat-decomposed runs on separate threads report their own: the
/// thread-local guard book keeps the runs from colliding.
#[test]
fn concurrent_runs_report_their_own() {
    let solo = run_two_tiles(&Arc::new(AtomicUsize::new(0)), None, 0.05, |_| {
        std::ops::ControlFlow::Continue(())
    });
    let left = std::thread::spawn(|| {
        run_two_tiles(&Arc::new(AtomicUsize::new(0)), None, 0.05, |_| {
            std::ops::ControlFlow::Continue(())
        })
    });
    let right = std::thread::spawn(|| {
        run_two_tiles(&Arc::new(AtomicUsize::new(0)), None, 0.05, |_| {
            std::ops::ControlFlow::Continue(())
        })
    });
    let a = left.join().expect("left run panicked");
    let b = right.join().expect("right run panicked");
    assert_eq!(
        a.guards.troubled_cells.total,
        solo.guards.troubled_cells.total
    );
    assert_eq!(
        b.guards.troubled_cells.total,
        solo.guards.troubled_cells.total
    );
}

/// an observer break returns the accepted-so-far totals: an early-exit path the
/// flat-decomposed driver actually supports, whose accepted totals are settled.
#[test]
fn an_observer_break_returns_the_accepted_totals() {
    let calls = Arc::new(AtomicUsize::new(0));
    // break at the first observer callback, after at least one accepted step.
    let diag = run_two_tiles(&calls, None, 1.0, |_iter| std::ops::ControlFlow::Break(()));
    assert!(
        diag.guards.troubled_cells.total > 0,
        "the break exit returned no accepted acts; the early-exit path is untested"
    );
}
