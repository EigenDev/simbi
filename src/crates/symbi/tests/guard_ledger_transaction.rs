// =============================================================================
// guard_ledger_transaction.rs
//
// the guard ledger's step transaction, pinned on the shared tile driver with a
// deterministic mock kernel set. the mock mints a chosen guard receipt on each
// substage and can reject or crash a chosen substage, so the driver's
// accepted-step commit and retry/crash discard are exercised without depending on
// the physics tripping a real guard.
//
// gates:
// - a rejected attempt's guard acts book into the attempted totals but never into
//   the accepted totals the run returns (retry discard);
// - a crashed step's pending guard acts are discarded, so the accepted totals hold
//   only steps that survived (crash discard);
// - a decomposed run sums every tile's accepted guard acts into one run-owned
//   total (decomposed aggregation);
// - a refined run sums every level's accepted guard acts (refined aggregation).
// =============================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use symbi::sim::decomp::LocalCopy;
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::guard_ledger::{self, GuardReceipt};
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::{FofcDecision, FofcReport, KernelSet, SourceReplayOutcome};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
// the accepted-substage troubled count; the rejected-substage sentinel is large
// so its presence in a total is unmistakable.
const ACCEPT_TROUBLED: u64 = 2;
const REJECT_SENTINEL: u64 = 1000;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

/// a kernel set that mints a chosen guard receipt on every substage and, on the
/// nth fofc call, either rejects the step (a coherent retry) or panics (a crash).
struct GuardMock {
    calls: Arc<AtomicUsize>,
    reject_at: Option<usize>,
    panic_at: Option<usize>,
}

impl GuardMock {
    fn new(calls: &Arc<AtomicUsize>, reject_at: Option<usize>, panic_at: Option<usize>) -> Self {
        Self {
            calls: calls.clone(),
            reject_at,
            panic_at,
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
        if self.panic_at == Some(call) {
            panic!("guard mock crash on fofc call {call}");
        }
        if self.reject_at == Some(call) {
            // a coherent rejected pass: an exterior freeze act is the retry
            // evidence. its guard acts book into pending, then get discarded.
            return FofcReport::of_pass(
                REJECT_SENTINEL,
                1,
                1,
                SourceReplayOutcome::SharedRedo,
                FofcDecision::RetryStep,
            )
            .with_guards(GuardReceipt::of_pass(REJECT_SENTINEL, 0, 1, 0));
        }
        // an accepted pass with a small, fixed troubled count.
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

fn uniform_sim(cells: usize, origin: f64, spacing: f64, boundaries: Boundaries<1>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .origin([origin])
        .spacing([spacing])
        .boundaries(boundaries)
        .timestepping(Timestepping::Euler)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build()
}

/// a rejected attempt's guard acts reach the attempted totals but never the
/// accepted totals the run returns: the driver discards the pending bucket on
/// retry, exactly as it does for projection evidence.
#[test]
fn a_rejected_attempt_is_absent_from_the_accepted_guards() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sim = uniform_sim(
        16,
        0.0,
        1.0 / 16.0,
        Boundaries::uniform(BoundaryType::Periodic),
    );
    // reject the first fofc call; every later call accepts.
    let mut hier = Hierarchy::single(sim, GuardMock::new(&calls, Some(1), None));
    let diag = hier
        .evolve_with_callback(0.05, 1, |_| std::ops::ControlFlow::Continue(()))
        .expect("the mock run completes");

    let (attempted, _) = guard_ledger::report();
    // the rejected sentinel booked into attempted...
    assert!(
        attempted.troubled_cells.total >= REJECT_SENTINEL,
        "the rejected attempt never booked into the attempted totals"
    );
    // ...but the accepted totals hold only the accepted substages, well under it.
    assert!(
        diag.guards.troubled_cells.total > 0,
        "no accepted guard act booked; the gate is vacuous"
    );
    assert!(
        diag.guards.troubled_cells.total < REJECT_SENTINEL,
        "the rejected attempt's guard acts survived into the accepted totals \
         ({} >= sentinel {REJECT_SENTINEL})",
        diag.guards.troubled_cells.total
    );
}

/// a crashed step's pending guard acts are discarded: a two-stage step books the
/// first substage's acts, then the second substage crashes, and the run returns
/// no accepted guard acts for the step that did not survive.
#[test]
fn a_crashed_step_discards_its_pending_guards() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([16])
        .spacing([1.0 / 16.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .timestepping(Timestepping::Rk2)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build();
    // stage one books its acts (call 1, accept), stage two crashes (call 2).
    let mut hier = Hierarchy::single(sim, GuardMock::new(&calls, None, Some(2)));
    let diag = hier
        .evolve_with_callback(0.05, 1, |_| std::ops::ControlFlow::Continue(()))
        .expect("the driver catches the crash and ends the march");

    let (attempted, _) = guard_ledger::report();
    assert!(
        attempted.troubled_cells.total >= ACCEPT_TROUBLED,
        "the first substage never booked into the attempted totals"
    );
    assert_eq!(
        diag.guards.troubled_cells.total, 0,
        "the crashed step's guard acts survived into the accepted totals"
    );
}

/// a decomposed run sums every tile's accepted guard acts into one run-owned
/// total: two tiles, each accepting with the same acts, commit at the collective
/// accept boundary and the returned total carries both.
#[test]
fn a_decomposed_run_sums_every_tile() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut tiles = Vec::new();
    for tile in 0..2 {
        let boundaries = Boundaries::per_axis([[
            if tile == 0 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            },
            if tile == 1 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            },
        ]]);
        let root = uniform_sim(16, tile as f64 * 0.5, 1.0 / 32.0, boundaries);
        tiles.push(Hierarchy::single(root, GuardMock::new(&calls, None, None)));
    }
    let devices = [0, 0];
    let diag = evolve_hierarchy_decomposed(
        &mut tiles,
        [2],
        &devices,
        &LocalCopy,
        Timestepping::Euler,
        0.0,
        0.025,
        1,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
    // both tiles accept the same acts each root substage, so the accepted total is
    // an exact multiple of the per-tile per-substage count and strictly exceeds
    // one tile's contribution.
    assert!(
        diag.guards.troubled_cells.total >= 2 * ACCEPT_TROUBLED,
        "the decomposed accept boundary dropped a tile's guard acts (total {})",
        diag.guards.troubled_cells.total
    );
    assert_eq!(
        diag.guards.troubled_cells.total % ACCEPT_TROUBLED,
        0,
        "the accepted total is not a whole number of per-substage acts"
    );
}

/// a refined run sums every level's accepted guard acts: a two-level hierarchy
/// books the root and the fine level, and the returned total carries both.
#[test]
fn a_refined_run_sums_every_level() {
    let calls = Arc::new(AtomicUsize::new(0));
    let coarse = uniform_sim(
        16,
        0.0,
        1.0 / 16.0,
        Boundaries::uniform(BoundaryType::Periodic),
    );
    let mut hier = Hierarchy::with_refinement(
        coarse,
        GuardMock::new(&calls, None, None),
        &[RefinementRegion {
            x_lo: [0.25],
            x_hi: [0.75],
        }],
        ProlongOrder::Plm,
        |_| GuardMock::new(&calls, None, None),
    )
    .unwrap();
    let diag = hier
        .evolve_with_callback(0.025, 1, |_| std::ops::ControlFlow::Continue(()))
        .expect("the refined mock run completes");
    // the root and the fine subcycle each book their accepting substages, so the
    // accepted total exceeds one level's single-substage contribution.
    assert!(
        diag.guards.troubled_cells.total > ACCEPT_TROUBLED,
        "the refined accept boundary booked only one level (total {})",
        diag.guards.troubled_cells.total
    );
}
