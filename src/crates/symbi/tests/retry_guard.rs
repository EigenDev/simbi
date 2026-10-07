// =============================================================================
// retry_guard.rs
//
// the retry guard at its exact boundaries, through the drivers that take root
// steps. a test kernel set requests a scheduled number of retries per root
// step, each carrying one line of freeze evidence, so the rejection count and
// the accepted / candidate ratio are known exactly (a retry halves):
// - twenty rejections of one root step are admitted, the twenty-first ends the
//   run during the attempt loop, naming the evidence of every rejection;
// - seven consecutive root steps accepted at exactly 1/16 of their candidate
//   are admitted, the eighth ends the run before the observer sees it;
// - a step at 1/8 resets the streak; a timestep capped below the cfl step
//   and a final step shortened to the end time are candidates of their own
//   and never count;
// - a rejection in the second fine substep of a refined hierarchy counts once
//   against the enclosing root step;
// - a checkpoint carries the streak and the thresholds: a resumed run ends at
//   the eighth consecutive step counted across the restart, and a changed
//   configuration is named in the opening report;
// - two worker processes, only one of which supplies the rejecting evidence,
//   both end at the same attempt through the coordinated failure path, and
//   the sustained guard ends both before either publishes the eighth step;
// - workers resuming different guard states refuse the session before a step;
// - the worker holding the evidence held back before the guard's collective:
//   the other waits there for it, both end, and the evidence survives; the
//   worker holding the evidence lost at its decision: the other still ends
//   with its own failure within the collective's deadline.
//
// run: cargo test -p symbi --test retry_guard
// =============================================================================

use std::collections::HashMap;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::checkpoint::{load_checkpoint, write_checkpoint};
use symbi::sim::decomp::{LocalCopy, Partition, Topology, plan_schema, unflatten};
use symbi::sim::evolve::KernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{FofcDecision, FofcReport, SourceReplayOutcome};
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_io::Metadata;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::retry_guard::{RetryGuard, opening_report};
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Store = FieldStore<2, 2, HostMemory, f64>;

/// the cfl step of the test kernel set: every root step's candidate before a cap or the end time
const CFL_DT: f64 = 1.0e-3;
const EVIDENCE: &str = "frozen cell [3, 4]";

/// a kernel set that touches no field and requests `schedule[step]` retries of root step `step`
/// at its first stage (stage tag 1, the rk2 predictor), each with one line of freeze evidence on
/// the store
#[derive(Clone)]
struct Rejector {
    schedule: Arc<Vec<u32>>,
    done: Arc<Mutex<HashMap<u64, u32>>>,
}

impl Rejector {
    fn new(schedule: Vec<u32>) -> Self {
        Self { schedule: Arc::new(schedule), done: Arc::new(Mutex::new(HashMap::new())) }
    }
}

impl KernelSet<2, 2, HostMemory, f64> for Rejector {
    fn flux(&self, _s: &Store, _dir: usize) {}
    fn c2p(&self, _s: &Store) {}
    fn godunov_stage(&self, _s: &Store, _dt: f64, _a0: f64, _ac: f64) {}
    fn cfl(&self, _s: &Store) -> f64 {
        CFL_DT
    }
    fn ghost_fill(&self, _s: &Store) {}
    fn snapshot(&self, _s: &Store) {}
    fn fofc_active(&self) -> bool {
        true
    }
    fn rollback_snapshot_is_complete(&self, _s: &Store) -> bool {
        true
    }
    fn fofc(&self, s: &Store, _dt: f64, _a0: f64, _ac: f64, stage: u8, _redo_update: &mut dyn FnMut()) -> FofcReport {
        if stage != 1 {
            return FofcReport::inactive();
        }
        let mut done = self.done.lock().unwrap();
        let taken = done.entry(s.iteration).or_insert(0);
        if *taken < self.schedule.get(s.iteration as usize).copied().unwrap_or(0) {
            *taken += 1;
            s.workspace.rejection_evidence.lock().unwrap().push(format!("stage tag {stage}: {EVIDENCE}"));
            return FofcReport::of_pass(1, 1, 1, SourceReplayOutcome::SharedRedo, FofcDecision::RetryStep);
        }
        FofcReport::inactive()
    }
}

/// a grid of `cells` placed as production places a tile: the whole grid's origin and the index
/// offset of its first cell
fn tiny(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells(cells)
        .origin([0.0, 0.0])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .spacing([0.125; 2])
        .boundaries(bnd)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
}

fn whole() -> Sim {
    tiny([8, 8], [0, 0], Boundaries::uniform(BoundaryType::Outflow))
}

/// the hierarchy driven to `t_final`: the accepted steps its observer saw before any failure, and
/// the failure message the driver recorded
fn run_hierarchy(sim: Sim, schedule: Vec<u32>, t_final: f64) -> (Vec<f64>, Option<String>) {
    let mut h = Hierarchy::single(sim, Rejector::new(schedule));
    let mut seen: Vec<(u64, f64)> = Vec::new();
    h.evolve_with_callback(t_final, 1, |h| {
        let s = &h.levels[0].state;
        // the observer's last call after the loop reports the final state again
        if h.crash.is_none() && seen.last().is_none_or(|&(it, _)| it != s.iteration) {
            seen.push((s.iteration, s.dt));
        }
        ControlFlow::Continue(())
    })
    .expect("evolve");
    let failure = h.crash.as_ref().and_then(|c| c.halt.as_ref()).map(|halt| halt.detail().to_string());
    (seen.into_iter().map(|(_, dt)| dt).collect(), failure)
}

#[test]
fn twenty_rejections_are_admitted_and_the_twenty_first_ends_the_run() {
    if in_worker_role() {
        worker_main();
    }
    let (seen, failure) = run_hierarchy(whole(), vec![20], CFL_DT / 2f64.powi(20) + 2.0 * CFL_DT);
    assert!(failure.is_none(), "twenty rejections ended the run: {failure:?}");
    assert_eq!(seen.first().copied(), Some(CFL_DT / 2f64.powi(20)), "the step was not accepted after twenty halvings");

    let (seen, failure) = run_hierarchy(whole(), vec![21], 1.0);
    let failure = failure.expect("the twenty-first rejection did not end the run");
    assert!(seen.is_empty(), "the observer saw a step the guard rejected");
    assert!(failure.contains("rejected 21 times, past the limit of 20"), "{failure}");
    assert!(failure.contains(&format!("level 0, attempt 20, stage tag 1: {EVIDENCE}")), "the last rejection's evidence is missing:\n{failure}");
    assert!(failure.contains("retry guard: at most 20 rejections"), "the thresholds are missing:\n{failure}");
    // the uni-grid driver ends the run at the same attempt
    let mut sim = whole();
    let err = symbi::sim::evolve::evolve(&mut sim, &Rejector::new(vec![21]), 1.0).unwrap_err();
    assert!(err.detail.contains("rejected 21 times") && err.detail.contains(&format!("grid, attempt 20, stage tag 1: {EVIDENCE}")), "{}", err.detail);
    assert_eq!(sim.iteration, 0, "the uni-grid driver accepted a step the guard rejected");
}

#[test]
fn the_eighth_consecutive_severe_step_ends_the_run_before_it_is_published() {
    if in_worker_role() {
        worker_main();
    }
    let severe = CFL_DT / 16.0;
    // seven steps at exactly 1/16, then ordinary steps to the end time
    let (seen, failure) = run_hierarchy(whole(), vec![4; 7], 7.0 * severe + 3.0 * CFL_DT);
    assert!(failure.is_none(), "seven severe steps ended the run: {failure:?}");
    assert_eq!(seen.iter().filter(|&&dt| dt == severe).count(), 7);

    let (seen, failure) = run_hierarchy(whole(), vec![4; 8], 1.0);
    let failure = failure.expect("the eighth severe step did not end the run");
    assert_eq!(seen, vec![severe; 7], "the observer saw other than the seven admitted steps: {seen:?}");
    assert!(failure.contains("8 consecutive root steps accepted at or below 0.0625"), "{failure}");
    assert!(failure.contains("after 4 rejections") && failure.contains(EVIDENCE), "{failure}");
}

#[test]
fn a_milder_step_resets_and_ordinary_reductions_never_count() {
    if in_worker_role() {
        worker_main();
    }
    // seven at 1/16, one at 1/8, seven at 1/16
    let mut schedule = vec![4; 7];
    schedule.push(3);
    schedule.extend(vec![4; 7]);
    let t_final = 14.0 * CFL_DT / 16.0 + CFL_DT / 8.0 + 2.0 * CFL_DT;
    let (_, failure) = run_hierarchy(whole(), schedule, t_final);
    assert!(failure.is_none(), "a step at 1/8 did not reset the streak: {failure:?}");

    // a cap far below the cfl step, and an end time that shortens the last step: twenty steps
    // accepted at their own candidates, no rejection
    let mut capped = whole();
    capped.max_dt = CFL_DT / 64.0;
    let t_final = 19.5 * CFL_DT / 64.0;
    let mut h = Hierarchy::single(capped, Rejector::new(vec![]));
    h.evolve_with_callback(t_final, 1, |_| ControlFlow::Continue(())).expect("evolve");
    assert!(h.crash.is_none(), "an ordinary cap counted as a retry reduction: {:?}", h.crash.as_ref().and_then(|c| c.panic.clone()));
    assert_eq!(h.levels[0].state.iteration, 20, "the capped run did not take its twenty steps");
    assert_eq!(h.levels[0].state.severe_streak, 0, "an ordinary reduction entered the streak");
}

#[test]
fn a_late_fine_substep_rejection_counts_once_against_the_root_step() {
    if in_worker_role() {
        worker_main();
    }
    type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
    let dx = 1.0 / 32.0;
    let root = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([32, 32])
        .spacing([dx; 2])
        .origin([0.0, 0.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root")
        .set_initial(|[x, y]| {
            let b = 0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp();
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build();
    let kset = |s: &Sim| Kern::new(1.4, 0.4, &s.geom.allocated);
    let k = kset(&root);
    let patch = RefinementRegion { x_lo: [12.0 * dx, 12.0 * dx], x_hi: [20.0 * dx, 20.0 * dx] };
    let mut h = Hierarchy::with_refinement(root, k, &[patch], ProlongOrder::Ppm, kset).expect("hierarchy");
    h.seed_fine_from_coarse().expect("seed");
    h.prime();
    // one halving ends the run: a step accepted at or below 1/2 of its candidate, once
    h.levels[0].state.retry_guard = RetryGuard { max_rejections: 20, severe_ratio: 0.5, sustained_steps: 1 };
    // the last stage of the second fine substep of step 3
    h.reject_fine_once_at(3, 1, 1);
    let mut failure = None;
    for _ in 0..5 {
        let dt = h.root_cfl_dt();
        h.step_root_with_dt(dt);
        if let Some(crash) = h.crash.as_ref() {
            failure = crash.halt.as_ref().map(|halt| halt.detail().to_string());
            break;
        }
    }
    assert!(!h.forced_fine_rejection_pending(), "the fine rejection never fired; the gate is vacuous");
    let failure = failure.expect("the fine rejection did not count against the root step");
    assert_eq!(h.levels[0].state.iteration, 3, "the guard ended a different root step");
    assert!(failure.contains("1 consecutive") && failure.contains("after 1 rejections"), "{failure}");
    assert!(failure.contains("rejection 1") && failure.contains("no freeze evidence held locally"), "{failure}");
}

#[test]
fn a_restart_carries_the_streak_and_names_a_changed_configuration() {
    if in_worker_role() {
        worker_main();
    }
    let dir = std::env::temp_dir().join(format!("symbi_retry_guard_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("five.h5");
    let path = path.to_str().unwrap();
    // root steps at the cfl candidate through the uni-grid step, the clock advanced here
    let step = |sim: &mut Sim, k: &Rejector| {
        symbi::sim::evolve::step_once(sim, k, CFL_DT);
        sim.time += sim.dt;
        sim.iteration += 1;
    };
    // five severe steps, then a checkpoint
    let mut sim = whole();
    let k = Rejector::new(vec![4; 5]);
    for _ in 0..5 {
        step(&mut sim, &k);
    }
    assert_eq!(sim.severe_streak, 5);
    write_checkpoint(&sim, path, &Metadata::new()).expect("checkpoint");

    // resumed under the same thresholds: the streak carries, the eighth step counted across the
    // restart ends the run
    let mut resumed = whole();
    load_checkpoint(&mut resumed, path).expect("load");
    assert_eq!(resumed.severe_streak, 5, "the streak did not survive the restart");
    assert_eq!(resumed.retry_guard_recorded, Some(RetryGuard::default()));
    let report = opening_report(&resumed.retry_guard, resumed.retry_guard_recorded.as_ref(), resumed.severe_streak);
    assert!(report.contains("resumed with 5 consecutive") && !report.contains("changed on restart"), "{report}");
    let k = Rejector::new((0..8).map(|n| if n >= 5 { 4 } else { 0 }).collect());
    let mut failure = None;
    for _ in 0..3 {
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| step(&mut resumed, &k))) {
            failure = payload.downcast_ref::<String>().cloned();
            break;
        }
    }
    let failure = failure.expect("the eighth step across the restart did not end the run");
    assert!(failure.contains("8 consecutive"), "{failure}");
    assert_eq!(resumed.iteration, 7, "the resumed run ended at a different step");

    // a changed configuration is named, the recorded streak still carries
    let mut changed = whole();
    changed.retry_guard = RetryGuard { max_rejections: 30, severe_ratio: 1.0 / 32.0, sustained_steps: 12 };
    load_checkpoint(&mut changed, path).expect("load");
    let report = opening_report(&changed.retry_guard, changed.retry_guard_recorded.as_ref(), changed.severe_streak);
    // the change names both sets of thresholds and the streak that carries
    assert!(
        report.contains("recorded max_rejections=20 severe_ratio=0.0625 sustained_steps=8")
            && report.contains("configured max_rejections=30 severe_ratio=0.03125 sustained_steps=12"),
        "{report}"
    );
    assert!(report.contains("the recorded streak of 5 carries over") && changed.severe_streak == 5, "{report}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- two worker processes -----------------------------------------------------------------

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(60);

fn in_worker_role() -> bool {
    std::env::var(ROLE).is_ok()
}

fn env_list(name: &str) -> Vec<u32> {
    std::env::var(name).unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect()
}

/// one worker of the pair on a 16 x 8 grid cut at 8: the rejecting worker's kernel set requests
/// the scheduled retries, the other's none; the vote makes every worker reject
fn run_worker(me: WorkerId) -> Result<(), WorkerError> {
    let partition = Partition::explicit([16, 8], [vec![8], vec![]]).unwrap();
    let placement = Placement::new(2, vec![WorkerId(0), WorkerId(1)]).expect("placement");
    let tile = me.0 as usize;
    let tc = unflatten(tile, partition.counts());
    let bnd = Boundaries(std::array::from_fn(|a| {
        let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
        [face(tc[a] == 0), face(tc[a] + 1 == partition.counts()[a])]
    }));
    let mut sim = tiny([8, 8], [tc[0] * 8, 0], bnd);
    // a resumed streak on this worker alone: the session's guard state disagrees
    if let Some(streak) = std::env::var(format!("W_STREAK_{}", me.0)).ok() {
        sim.severe_streak = streak.parse().unwrap();
    }
    let rejecting = std::env::var("W_REJECTING").unwrap().parse::<u32>().unwrap() == me.0;
    let mut injection = Injection::default();
    if rejecting {
        injection.hold_before_guard = std::env::var("W_HOLD_MS").ok().map(|ms| Duration::from_millis(ms.parse().unwrap()));
        injection.vanish_before_guard = std::env::var_os("W_VANISH").is_some();
    }
    let kernels = Rejector::new(if rejecting { env_list("W_SCHEDULE") } else { vec![] });
    let schema = plan_schema(&*sim);
    let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, sim.geom.ng).expect("plan");
    let table = PhaseTable::flat(2).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, me, table.clone());
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0x9E7,
            build_id: "retry_guard".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: table.digest(),
            ng: sim.geom.ng as u32,
            workers: 2,
            block_credit: 4096,
        },
        startup: DEADLINE,
    };
    let max_payload = exchange.lens().iter().max().copied().unwrap_or(0) * 8;
    let (link, session) = connect(&r, max_payload.max(64))?;
    let mut fabric = exchange.fabric(link, session);
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 2, Newtonian>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 1.0,
        max_steps: 40,
        deadline: DEADLINE,
        injection,
        audit_every_point: false,
    };
    let mut stores: Vec<&mut Store> = vec![&mut *sim];
    let devices = vec![0i32; 2];
    evolve_worker(&[TileId(tile as u32)], &mut stores, &[&kernels], &devices, &exchange, &LocalCopy, &mut fabric, &cfg, |accepted, _, _, _| {
        println!("PUBLISHED {accepted}");
        Ok(ControlFlow::Continue(()))
    })?;
    fabric.finish(DEADLINE)?;
    Ok(())
}

fn worker_main() -> ! {
    let me = WorkerId(std::env::var("W_ME").unwrap().parse().unwrap());
    match run_worker(me) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("worker {}: {e}", me.0);
            std::process::exit(2);
        }
    }
}

struct Outcome {
    code: Option<i32>,
    stdout: String,
    stderr: String,
    /// when the worker's exit was seen, from the spawn of the pair
    exited: Duration,
}

/// both workers of an arm under the watchdog
fn run_pair(test: &str, tag: &str, schedule: &[u32], extra: &[(&str, &str)]) -> Vec<Outcome> {
    let dir = std::env::temp_dir().join(format!("symbi_retry_guard_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let list = schedule.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let mut children: Vec<Child> = (0..2)
        .map(|w| {
            Command::new(std::env::current_exe().unwrap())
                .args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(ROLE, "worker")
                .env("W_ME", w.to_string())
                .env("W_COORD", &coord)
                .env("W_REJECTING", "1")
                .env("W_SCHEDULE", &list)
                .envs(extra.iter().copied())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let start = Instant::now();
    let mut exited = [None; 2];
    while exited.iter().any(Option::is_none) {
        for (w, c) in children.iter_mut().enumerate() {
            if exited[w].is_none() && c.try_wait().unwrap().is_some() {
                exited[w] = Some(start.elapsed());
            }
        }
        if start.elapsed() > WATCHDOG {
            for c in children.iter_mut() {
                let _ = c.kill();
            }
            panic!("the watchdog had to end the workers");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let outcomes = children
        .into_iter()
        .zip(exited)
        .map(|(mut c, exited)| {
            let code = c.wait().unwrap().code();
            let (mut stdout, mut stderr) = (String::new(), String::new());
            c.stdout.take().unwrap().read_to_string(&mut stdout).unwrap();
            c.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
            Outcome { code, stdout, stderr, exited: exited.unwrap() }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

#[test]
fn workers_end_together_when_one_supplies_the_rejecting_evidence() {
    if in_worker_role() {
        worker_main();
    }
    let test = "workers_end_together_when_one_supplies_the_rejecting_evidence";
    let pair = run_pair(test, "limit", &[21], &[]);
    for (w, o) in pair.iter().enumerate() {
        assert_eq!(o.code, Some(2), "worker {w} did not end through the failure path:\n{}", o.stderr);
        assert!(o.stderr.contains("rejected 21 times, past the limit of 20"), "worker {w}:\n{}", o.stderr);
        assert!(!o.stdout.contains("PUBLISHED 1"), "worker {w} published a rejected step");
    }
    assert!(
        pair[1].stderr.contains(&format!("worker 1, tile 1, level 0, attempt 20, stage tag 1: {EVIDENCE}")),
        "the rejecting worker did not name its evidence:\n{}",
        pair[1].stderr
    );
    assert!(
        pair[0].stderr.contains("no freeze evidence held locally") && !pair[0].stderr.contains(EVIDENCE),
        "the other worker named evidence it does not hold:\n{}",
        pair[0].stderr
    );

    let pair = run_pair(test, "sustained", &[4; 8], &[]);
    for (w, o) in pair.iter().enumerate() {
        assert_eq!(o.code, Some(2), "worker {w} did not end through the failure path:\n{}", o.stderr);
        assert!(o.stderr.contains("8 consecutive root steps"), "worker {w}:\n{}", o.stderr);
        assert!(o.stdout.contains("PUBLISHED 7") && !o.stdout.contains("PUBLISHED 8"), "worker {w} published the eighth step:\n{}", o.stdout);
    }
}

/// the hold of the worker holding the evidence before the guard's collective
const HOLD: Duration = Duration::from_millis(2000);

#[test]
fn the_guard_waits_for_the_worker_holding_the_evidence_and_ends_without_it() {
    if in_worker_role() {
        worker_main();
    }
    let test = "the_guard_waits_for_the_worker_holding_the_evidence_and_ends_without_it";
    // the worker holding the evidence arrives last: the other waits at the collective for it
    let hold = HOLD.as_millis().to_string();
    let pair = run_pair(test, "held", &[21], &[("W_HOLD_MS", &hold)]);
    for (w, o) in pair.iter().enumerate() {
        assert_eq!(o.code, Some(2), "worker {w} did not end through the failure path:\n{}", o.stderr);
        assert!(o.stderr.contains("rejected 21 times, past the limit of 20"), "worker {w}:\n{}", o.stderr);
    }
    assert!(
        pair[1].stderr.contains(&format!("worker 1, tile 1, level 0, attempt 20, stage tag 1: {EVIDENCE}")),
        "the held worker's evidence did not survive:\n{}",
        pair[1].stderr
    );
    assert!(
        pair[0].exited >= HOLD,
        "the other worker ended at {:?}, before the held worker reached the guard's collective",
        pair[0].exited
    );

    // the worker holding the evidence is lost at its decision: the other ends with its own
    // failure, bounded by the collective's deadline
    let pair = run_pair(test, "vanished", &[21], &[("W_VANISH", "1")]);
    assert_eq!(pair[1].code, Some(3), "the vanishing worker did not vanish:\n{}", pair[1].stderr);
    assert_eq!(pair[0].code, Some(2), "the other worker did not end through the failure path:\n{}", pair[0].stderr);
    assert!(pair[0].stderr.contains("rejected 21 times, past the limit of 20"), "{}", pair[0].stderr);
    assert!(pair[0].exited <= DEADLINE + Duration::from_secs(5), "the other worker ended at {:?}, past the deadline", pair[0].exited);
}

#[test]
fn workers_resuming_different_guard_states_refuse_the_session() {
    if in_worker_role() {
        worker_main();
    }
    let test = "workers_resuming_different_guard_states_refuse_the_session";
    let pair = run_pair(test, "disagree", &[], &[("W_STREAK_1", "5")]);
    for (w, o) in pair.iter().enumerate() {
        assert_eq!(o.code, Some(2), "worker {w} did not refuse the session:\n{}", o.stderr);
        assert!(o.stderr.contains("the retry guard or retry exclusion differs across the workers"), "worker {w}:\n{}", o.stderr);
        assert!(!o.stdout.contains("PUBLISHED 1"), "worker {w} took a step before refusing");
    }
    assert!(pair[1].stderr.contains("here streak 5") && pair[1].stderr.contains("the minimum over the workers streak 0"), "{}", pair[1].stderr);
    // the same guard state on both workers starts the session
    let pair = run_pair(test, "agree", &[], &[("W_STREAK_0", "5"), ("W_STREAK_1", "5")]);
    for (w, o) in pair.iter().enumerate() {
        assert_eq!(o.code, Some(0), "worker {w} refused an agreed guard state:\n{}", o.stderr);
    }
}
