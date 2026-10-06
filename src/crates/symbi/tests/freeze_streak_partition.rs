// =============================================================================
// freeze_streak_partition.rs
//
// the FOFC halt counts consecutive substages that end with a freeze outside any
// excision region, over the whole level, wherever on it the freeze lands. a
// mock kernel set freezes on a schedule and advances and halts the count by the
// substrate's rule (one more on a substage it froze, zero on one it did not,
// the halt at sixteen); the drivers are the production ones. the scheduled
// freezes alternate between the two tiles of a cut on every forward-Euler step,
// so every substage freezes somewhere and no tile freezes twice in a row. the
// single grid halts at the sixteenth step; the two-tile march, which agrees one
// level-wide count before each correction pass, halts at the same step with
// the same message, in process and on a launched worker holding both tiles.
// the same freezes kept inside one tile halt both alike.
//
// run: cargo test -p symbi --test freeze_streak_partition
// =============================================================================

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU32, Ordering};

use std::path::PathBuf;
use std::time::Duration;

use symbi::sim::decomp::{LocalCopy, Partition, Topology, evolve_decomposed, plan_schema};
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::{FofcDecision, FofcReport, KernelSet, SourceReplayOutcome};
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const HALT_STREAK: u32 = 16;
const HALT: &str = "freeze fired on 16 consecutive substages";

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

/// which tile freezes at an iteration: the two tiles in turn, or the first alone
#[derive(Clone, Copy)]
enum Schedule {
    Alternating,
    FirstTile,
}

impl Schedule {
    fn freezing_tile(self, iteration: u64) -> usize {
        match self {
            Schedule::Alternating => (iteration % 2) as usize,
            Schedule::FirstTile => 0,
        }
    }
}

/// a kernel set whose correction pass freezes when the schedule names its tile (`None`: the
/// whole grid, every scheduled freeze), counting by the substrate's rule
struct StreakMock {
    tile: Option<usize>,
    schedule: Schedule,
    streak: AtomicU32,
}

impl StreakMock {
    fn new(tile: Option<usize>, schedule: Schedule) -> Self {
        Self { tile, schedule, streak: AtomicU32::new(0) }
    }
}

impl KernelSet<1, 1, HostMemory, f64> for StreakMock {
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
            store.fields.prim.pre_field().unwrap().view_mut().set(cell, (GAMMA - 1.0) * (nrg - 0.5 * mom * mom / rho));
        }
        store.mark_primitives_recovered();
    }
    fn godunov_stage(&self, _store: &FieldStore<1, 1, HostMemory>, _dt: f64, _a0: f64, _ac: f64) {}
    fn cfl(&self, _store: &FieldStore<1, 1, HostMemory>) -> f64 {
        0.01
    }
    fn ghost_fill(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn snapshot(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn fofc_active(&self) -> bool {
        true
    }
    fn freeze_streak(&self) -> u32 {
        self.streak.load(Ordering::Relaxed)
    }
    fn set_freeze_streak(&self, streak: u32) {
        self.streak.store(streak, Ordering::Relaxed);
    }
    fn fofc(
        &self,
        store: &FieldStore<1, 1, HostMemory>,
        _dt: f64,
        _a0: f64,
        _ac: f64,
        _stage: u8,
        _redo_update: &mut dyn FnMut(),
    ) -> FofcReport {
        let scheduled = self.schedule.freezing_tile(store.iteration);
        let froze = self.tile.is_none_or(|tile| tile == scheduled);
        let streak = if froze {
            self.streak.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            self.streak.store(0, Ordering::Relaxed);
            0
        };
        if streak >= HALT_STREAK {
            panic!("FOFC last-resort {HALT} at iteration {}", store.iteration);
        }
        let n = u64::from(froze);
        FofcReport::of_pass(n, n, n, SourceReplayOutcome::SharedRedo, FofcDecision::Accept)
    }
    fn horizon_accretion(&self, _store: &FieldStore<1, 1, HostMemory>, _diagnostic_radius: f64) -> (f64, f64) {
        (0.0, 0.0)
    }
}

fn store(cells: usize, origin: f64, lo: BoundaryType, hi: BoundaryType) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
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

/// the halt message of a single-grid run (its driver catches a step panic into the crash report)
fn single(schedule: Schedule) -> Option<String> {
    let sim = store(16, 0.0, BoundaryType::Outflow, BoundaryType::Outflow);
    let mut hier = Hierarchy::single(sim, StreakMock::new(None, schedule));
    hier.evolve_with_callback(1.0, 1, |_| ControlFlow::Continue(())).expect("single-grid run");
    hier.crash.as_ref().and_then(|c| c.panic.clone())
}

/// the halt message of a two-tile run
fn tiled(schedule: Schedule) -> Option<String> {
    let mut a = store(8, 0.0, BoundaryType::Outflow, BoundaryType::CoarseFine);
    let mut b = store(8, 0.5, BoundaryType::CoarseFine, BoundaryType::Outflow);
    let (ka, kb) = (StreakMock::new(Some(0), schedule), StreakMock::new(Some(1), schedule));
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let mut stores = vec![&mut *a, &mut *b];
        evolve_decomposed(&mut stores, &[&ka, &kb], [2], &[0, 0], Timestepping::Euler, 0.0, 1.0, 1, &LocalCopy, |_, _, _| {
            ControlFlow::Continue(())
        });
    }));
    caught.err().map(|payload| {
        payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
    })
}

/// the halt message of a launched worker holding both tiles: the worker's own correction loop,
/// its agreement over the tiles it holds and its freeze vote across the (one-worker) fabric
fn launched(schedule: Schedule) -> Option<String> {
    const DEADLINE: Duration = Duration::from_secs(8);
    let mut a = store(8, 0.0, BoundaryType::Outflow, BoundaryType::CoarseFine);
    let mut b = store(8, 0.5, BoundaryType::CoarseFine, BoundaryType::Outflow);
    let (ka, kb) = (StreakMock::new(Some(0), schedule), StreakMock::new(Some(1), schedule));
    let partition = Partition::explicit([16], [vec![8]]).expect("the cut lies inside the grid");
    let me = WorkerId(0);
    let placement = Placement::new(1, vec![me, me]).expect("placement");
    let plan = ExchangePlan::compile(&partition, &Topology::open(), &plan_schema(&*a), a.geom.ng).expect("plan");
    let table = PhaseTable::flat(1).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, me, table.clone());
    let announce = std::env::temp_dir().join(format!("freeze-streak-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(&announce)),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0x5EED,
            build_id: "freeze_streak_partition".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: table.digest(),
            ng: a.geom.ng as u32,
            workers: 1,
            block_credit: 4096,
        },
        startup: DEADLINE,
    };
    let max_payload = exchange.lens().iter().max().copied().unwrap_or(0) * 8;
    let (link, session) = connect(&r, max_payload.max(64)).expect("the one-worker session");
    let mut fabric = exchange.fabric(link, session);
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 1, Newtonian>(),
        timestepping: Timestepping::Euler,
        start_time: 0.0,
        t_final: 1.0,
        max_steps: 64,
        deadline: DEADLINE,
        injection: Injection::default(),
        audit_every_point: false,
    };
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let mut stores = vec![&mut *a, &mut *b];
        let _ = evolve_worker(&[TileId(0), TileId(1)], &mut stores, &[&ka, &kb], &[0, 0], &exchange, &LocalCopy, &mut fabric, &cfg, |_, _, _, _| {
            Ok(ControlFlow::Continue(()))
        });
    }));
    let _ = std::fs::remove_file(&announce);
    caught.err().map(|payload| {
        payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
    })
}

#[test]
fn a_streak_that_moves_between_tiles_halts_a_launched_worker_where_it_halts_the_single_grid() {
    let one = single(Schedule::Alternating).expect("the single grid never halted");
    let worker = launched(Schedule::Alternating).expect("the worker never halted on a streak whose freezes alternate between its tiles");
    assert_eq!(worker, one, "the worker halted elsewhere than the single grid");
}

#[test]
fn a_streak_that_moves_between_tiles_halts_the_tiled_march_where_it_halts_the_single_grid() {
    let one = single(Schedule::Alternating).expect("the single grid never halted; the schedule froze fewer than sixteen consecutive substages");
    assert!(one.contains(HALT) && one.contains("at iteration 15"), "the single grid halted elsewhere: {one}");
    let many = tiled(Schedule::Alternating).expect("the two-tile march never halted on a streak whose freezes alternate between its tiles");
    assert_eq!(many, one, "the two-tile march halted elsewhere than the single grid");
}

#[test]
fn a_streak_inside_one_tile_halts_both_alike() {
    let one = single(Schedule::FirstTile).expect("the single grid never halted");
    assert!(one.contains(HALT) && one.contains("at iteration 15"), "the single grid halted elsewhere: {one}");
    assert_eq!(tiled(Schedule::FirstTile).as_deref(), Some(one.as_str()), "the two-tile march halted elsewhere than the single grid");
}
