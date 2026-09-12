// =============================================================================
// worker_processes.rs
//
// the distributed evolution loop across real processes on the seed fixture:
// each worker process builds the tiles it holds, evolves them with the fabric
// carrying the timestep minimum, the rejection vote, and the halo plan, and
// writes every cell of every field. the parent evolves the whole tile set in
// one process with the decomposed loop and compares each worker's tiles
// bitwise. a rejection forced on one worker rolls every worker back to the
// same halved timestep, and the result equals a single worker driven through
// the same rejection. an invalid timestep candidate on one worker ends every
// worker within the deadline.
//
// run: cargo test -p symbi --test worker_processes
// =============================================================================

use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{
    LocalCopy, Partition, Schedule, Topology, evolve_scheduled, plan_fields, plan_schema, unflatten,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, Fabric, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Store = FieldStore<2, 2, HostMemory, f64>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const STEPS: u64 = 12;
const T_FINAL: f64 = 10.0;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(8);
const STARTUP_DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(60);

fn seed(x: f64, y: f64) -> f64 {
    1.0 + 0.3 * (7.0 * x).sin() * (5.0 * y).cos() + 0.1 * x * y
}

#[derive(Clone)]
struct Arm {
    cuts0: Vec<usize>,
    cuts1: Vec<usize>,
    periodic: [bool; 2],
    owner: Vec<u32>,
}

impl Arm {
    fn env(&self) -> Vec<(&'static str, String)> {
        let list = |v: &[usize]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        vec![
            ("W_CUTS0", list(&self.cuts0)),
            ("W_CUTS1", list(&self.cuts1)),
            (
                "W_PERIODIC",
                format!(
                    "{},{}",
                    u8::from(self.periodic[0]),
                    u8::from(self.periodic[1])
                ),
            ),
            (
                "W_OWNER",
                self.owner
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ]
    }

    fn from_env() -> Self {
        let list = |name: &str| -> Vec<usize> {
            let v = std::env::var(name).unwrap_or_default();
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect()
        };
        let periodic = list("W_PERIODIC");
        Self {
            cuts0: list("W_CUTS0"),
            cuts1: list("W_CUTS1"),
            periodic: [periodic[0] == 1, periodic[1] == 1],
            owner: list("W_OWNER").into_iter().map(|x| x as u32).collect(),
        }
    }

    fn partition(&self) -> Partition<2> {
        Partition::explicit([N, N], [self.cuts0.clone(), self.cuts1.clone()]).unwrap()
    }

    fn topology(&self) -> Topology<2> {
        Topology::wrapping(self.periodic)
    }

    fn workers(&self) -> u32 {
        self.owner.iter().max().unwrap() + 1
    }
}

fn tile(arm: &Arm, flat: usize) -> (Sim, Kern) {
    let p = arm.partition();
    let counts = p.counts();
    let tc = unflatten(flat, counts);
    let ext = p.tile_extents(tc);
    // a cut face is filled by the exchange; a domain face on a wrapped axis is a cut too
    let bnd = Boundaries(std::array::from_fn(|a| {
        let edge = |at_edge: bool| {
            if at_edge && !arm.periodic[a] {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            }
        };
        [edge(tc[a] == 0), edge(tc[a] == counts[a] - 1)]
    }));
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([DX; 2])
        .origin([ext[0].0 as f64 * DX, ext[1].0 as f64 * DX])
        .boundaries(bnd)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(seed(x, y)),
                Tensor::new([0.3 * seed(x, y), -0.2 * seed(y, x)]),
                Pressure(0.5 + 0.25 * seed(y, x)),
            )
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

/// every primitive over the allocated domain, since the stencil reads the halos, and every
/// conserved field over the interior, since conserved ghosts are never exchanged or read.
fn tile_values(store: &Store) -> Vec<f64> {
    let mut out = Vec::new();
    for field in plan_fields(store) {
        let view = field.view();
        for c in field.domain().iter() {
            out.push(*view.at(c));
        }
    }
    for field in store.fields.cons.exchange_fields() {
        let view = field.view();
        for c in store.geom.interior.iter() {
            out.push(*view.at(c));
        }
    }
    out
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn injection_from_env(me: WorkerId) -> Injection {
    let mut injection = Injection::default();
    if env_u64("W_REJECT_WORKER") == Some(u64::from(me.0)) {
        injection.reject_at = Some((
            env_u64("W_REJECT_STEP").unwrap(),
            env_u64("W_REJECT_STAGE").unwrap() as usize,
        ));
    }
    if env_u64("W_BAD_CFL_WORKER") == Some(u64::from(me.0)) {
        injection.invalid_cfl_at = env_u64("W_BAD_CFL_STEP");
    }
    injection
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    let mine: Vec<usize> = (0..n)
        .filter(|&f| placement.owner_of(TileId(f as u32)) == me)
        .collect();
    let mut built: Vec<(Sim, Kern)> = mine.iter().map(|&f| tile(arm, f)).collect();
    let tiles: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let schema = plan_schema(&*built[0].0);
    let plan = ExchangePlan::compile(&partition, &arm.topology(), &schema, built[0].0.geom.ng)
        .expect("plan");
    let exchange = PlanExchange::new(&plan, &placement, me);
    let devices = vec![0i32; n];
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        identity: Identity {
            credential: 0xBEEF,
            build_id: "worker_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            ng: built[0].0.geom.ng as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    };
    let lens = exchange.lens();
    let max_payload = lens.iter().max().copied().unwrap_or(0) * 8;
    let (link, session) = connect(&r, max_payload.max(64))?;
    let mut fabric = Fabric::new(
        link,
        me,
        session,
        arm.workers() as usize,
        2,
        &lens,
        exchange.sends_per_peer_axis(),
    );
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 2, Newtonian>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: T_FINAL,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection: injection_from_env(me),
    };
    let mut stores: Vec<&mut Store> = Vec::new();
    let mut kernels: Vec<&Kern> = Vec::new();
    for (s, k) in built.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    let report = evolve_worker(
        &tiles,
        &mut stores,
        &kernels,
        &devices,
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )?;
    fabric.finish(TRANSFER_DEADLINE)?;
    let mut bytes = Vec::new();
    for (flat, (sim, _)) in mine.iter().zip(&built) {
        bytes.extend_from_slice(&(*flat as u64).to_le_bytes());
        let values = tile_values(sim);
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    std::fs::write(out, bytes).unwrap();
    println!(
        "REPORT steps={} rejections={} time={} dts={}",
        report.steps,
        report.rejections,
        report.time.to_bits(),
        report
            .dt_sequence
            .iter()
            .map(|d| d.to_bits().to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}

fn worker_main() -> ! {
    let arm = Arm::from_env();
    let me = WorkerId(env_u64("W_ME").unwrap() as u32);
    let out = PathBuf::from(std::env::var("W_OUT").unwrap());
    match run_worker(&arm, me, out) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("worker {me:?}: {e}");
            std::process::exit(2);
        }
    }
}

fn in_worker_role() -> bool {
    std::env::var(ROLE).is_ok()
}

struct Spawned {
    child: Child,
    stderr: std::process::ChildStderr,
    stdout: std::process::ChildStdout,
    out: PathBuf,
}

fn spawn(
    test: &str,
    me: u32,
    arm: &Arm,
    coord: &std::path::Path,
    dir: &std::path::Path,
    extra: &[(&str, String)],
) -> Spawned {
    let out = dir.join(format!("worker_{me}.f64"));
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(ROLE, "worker")
        .env("W_ME", me.to_string())
        .env("W_COORD", coord)
        .env("W_OUT", &out)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in arm.env() {
        cmd.env(k, v);
    }
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    Spawned {
        child,
        stderr,
        stdout,
        out,
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Report {
    steps: u64,
    rejections: u64,
    time: u64,
    dts: Vec<u64>,
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    tiles: Vec<(usize, Vec<f64>)>,
}

fn parse_report(stdout: &str) -> Option<Report> {
    let line = stdout.lines().find(|l| l.contains("REPORT "))?;
    let line = &line[line.find("REPORT ")?..];
    let field = |name: &str| -> Option<String> {
        line.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
    };
    Some(Report {
        steps: field("steps")?.parse().ok()?,
        rejections: field("rejections")?.parse().ok()?,
        time: field("time")?.parse().ok()?,
        dts: field("dts")?
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect(),
    })
}

/// run every worker of an arm under the watchdog and gather exit codes, reports, and tiles.
fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir =
        std::env::temp_dir().join(format!("symbi_worker_{}_{test}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let mut spawned: Vec<Spawned> = (0..arm.workers())
        .map(|w| spawn(test, w, arm, &coord, &dir, extra))
        .collect();
    let start = Instant::now();
    loop {
        if spawned
            .iter_mut()
            .all(|s| s.child.try_wait().unwrap().is_some())
        {
            break;
        }
        if start.elapsed() > WATCHDOG {
            for s in spawned.iter_mut() {
                let _ = s.child.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let outcomes = spawned
        .iter_mut()
        .map(|s| {
            let status = s.child.wait().unwrap();
            let mut stderr = String::new();
            s.stderr.read_to_string(&mut stderr).unwrap();
            let mut stdout = String::new();
            s.stdout.read_to_string(&mut stdout).unwrap();
            let mut tiles = Vec::new();
            if let Ok(bytes) = std::fs::read(&s.out) {
                let mut pos = 0;
                while pos < bytes.len() {
                    let flat = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
                    let len =
                        u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap()) as usize;
                    pos += 16;
                    let values = (0..len)
                        .map(|i| {
                            f64::from_bits(u64::from_le_bytes(
                                bytes[pos + 8 * i..pos + 8 * i + 8].try_into().unwrap(),
                            ))
                        })
                        .collect();
                    pos += 8 * len;
                    tiles.push((flat, values));
                }
            }
            Outcome {
                code: status.code(),
                stderr,
                report: parse_report(&stdout),
                tiles,
            }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

/// the single-process reference: every tile evolved by the decomposed loop for `STEPS`.
fn reference(arm: &Arm) -> Vec<Vec<f64>> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let mut built: Vec<(Sim, Kern)> = (0..n).map(|f| tile(arm, f)).collect();
    let mut stores: Vec<&mut Store> = Vec::new();
    let mut kernels: Vec<&Kern> = Vec::new();
    for (s, k) in built.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    let schedule = Schedule::derive(partition.counts(), stores[0].geom.ng, &arm.topology());
    evolve_scheduled(
        &mut stores,
        &kernels,
        &schedule,
        &vec![0; n],
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        1,
        &LocalCopy,
        |iter, _, _| {
            if iter >= STEPS {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    built.iter().map(|(s, _)| tile_values(s)).collect()
}

fn assert_all_ok(outcomes: &[Outcome]) {
    for (i, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "worker {i} failed:\n{}", o.stderr);
    }
}

fn assert_tiles_match(outcomes: &[Outcome], want: &[Vec<f64>], label: &str) {
    let mut compared = 0;
    for o in outcomes {
        for (flat, got) in &o.tiles {
            assert_eq!(
                got.len(),
                want[*flat].len(),
                "{label}: tile {flat} value count"
            );
            for (i, (g, w)) in got.iter().zip(&want[*flat]).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits(),
                    "{label}: tile {flat} value {i}: worker {g:e} != reference {w:e}"
                );
            }
            compared += 1;
        }
    }
    assert_eq!(compared, want.len(), "{label}: every tile compared once");
}

fn even() -> Arm {
    Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [false, false],
        owner: vec![0, 0, 1, 1],
    }
}

#[test]
fn two_workers_match_the_single_process_reference() {
    if in_worker_role() {
        worker_main();
    }
    let test = "two_workers_match_the_single_process_reference";
    let arm = even();
    let outcomes = run_arm(test, "even", &arm, &[]);
    assert_all_ok(&outcomes);
    assert_tiles_match(&outcomes, &reference(&arm), "even");
    let ragged = Arm {
        cuts0: vec![19, 37],
        cuts1: vec![27],
        periodic: [false, false],
        owner: vec![0, 1, 0, 1, 1, 0],
    };
    let outcomes = run_arm(test, "ragged", &ragged, &[]);
    assert_all_ok(&outcomes);
    assert_tiles_match(&outcomes, &reference(&ragged), "ragged");
    for o in &outcomes {
        let r = o.report.as_ref().expect("report");
        assert_eq!(r.steps, STEPS);
        assert_eq!(r.rejections, 0);
    }
}

#[test]
fn four_workers_with_distinct_neighbors_match_the_reference() {
    if in_worker_role() {
        worker_main();
    }
    let test = "four_workers_with_distinct_neighbors_match_the_reference";
    let arm = Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [true, false],
        owner: vec![0, 1, 2, 3],
    };
    let outcomes = run_arm(test, "grid", &arm, &[]);
    assert_all_ok(&outcomes);
    assert_tiles_match(&outcomes, &reference(&arm), "grid");
    let reports: Vec<Report> = outcomes
        .iter()
        .map(|o| o.report.clone().expect("report"))
        .collect();
    assert!(
        reports.iter().all(|r| *r == reports[0]),
        "every worker reports the same clock and dt sequence"
    );
}

/// a rejection forced on one worker at step 5, stage 1: every worker rolls back, halves the
/// step, and lands on the state a single worker reaches through the same rejection.
#[test]
fn a_rejection_on_one_worker_rolls_back_all() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_rejection_on_one_worker_rolls_back_all";
    let inject = |worker: u32| {
        vec![
            ("W_REJECT_STEP", "5".to_string()),
            ("W_REJECT_STAGE", "1".to_string()),
            ("W_REJECT_WORKER", worker.to_string()),
        ]
    };
    let single = Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [false, false],
        owner: vec![0, 0, 0, 0],
    };
    let one = run_arm(test, "single", &single, &inject(0));
    assert_all_ok(&one);
    let want: Vec<Vec<f64>> = {
        let mut tiles = one[0].tiles.clone();
        tiles.sort_by_key(|(flat, _)| *flat);
        tiles.into_iter().map(|(_, v)| v).collect()
    };
    let single_report = one[0].report.clone().expect("report");
    assert_eq!(single_report.rejections, 1, "the injected rejection fired");
    assert_eq!(single_report.steps, STEPS);
    // the halved step: the accepted dt at step 5 is half of what the candidate gave
    assert!(f64::from_bits(single_report.dts[5]) < f64::from_bits(single_report.dts[4]) * 0.75);
    let clean = run_arm(test, "clean", &single, &[]);
    assert_all_ok(&clean);
    let clean_report = clean[0].report.clone().expect("report");
    assert_ne!(
        clean_report.time, single_report.time,
        "the rejection changed the clock"
    );
    for (workers, arm, injected) in [
        (2u32, even(), 1u32),
        (
            4,
            Arm {
                cuts0: vec![32],
                cuts1: vec![32],
                periodic: [false, false],
                owner: vec![0, 1, 2, 3],
            },
            2,
        ),
    ] {
        let outcomes = run_arm(test, &format!("w{workers}"), &arm, &inject(injected));
        assert_all_ok(&outcomes);
        assert_tiles_match(&outcomes, &want, &format!("{workers} workers"));
        for o in &outcomes {
            assert_eq!(
                o.report.as_ref().expect("report"),
                &single_report,
                "{workers} workers: report"
            );
        }
    }
}

/// a NaN timestep candidate on one worker: it aborts, every peer learns of it, and every
/// worker exits nonzero within the deadline.
#[test]
fn an_invalid_candidate_aborts_every_worker() {
    if in_worker_role() {
        worker_main();
    }
    let test = "an_invalid_candidate_aborts_every_worker";
    let arm = Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [false, false],
        owner: vec![0, 1, 2, 3],
    };
    let started = Instant::now();
    let outcomes = run_arm(
        test,
        "nan",
        &arm,
        &[
            ("W_BAD_CFL_STEP", "3".to_string()),
            ("W_BAD_CFL_WORKER", "2".to_string()),
        ],
    );
    assert!(
        started.elapsed() < TRANSFER_DEADLINE * 3,
        "termination took {:?}",
        started.elapsed()
    );
    for (i, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {i} exited zero");
    }
    assert!(
        outcomes[2].stderr.contains("invalid CFL candidate"),
        "{}",
        outcomes[2].stderr
    );
    // the coordinator reads worker 2's abort before anything else on that stream; a worker
    // may instead read the coordinator's end of file first, which is the same termination
    assert!(
        outcomes[0].stderr.contains("aborted"),
        "coordinator:\n{}",
        outcomes[0].stderr
    );
    for i in [1usize, 3] {
        let e = &outcomes[i].stderr;
        assert!(
            e.contains("aborted") || e.contains("disconnected"),
            "worker {i}:\n{e}"
        );
    }
}

/// a worker holding two tiles, one of which reports a NaN candidate while the other is
/// finite: the bad tile is caught before the fold and every worker aborts.
#[test]
fn a_bad_tile_behind_a_good_one_still_aborts() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_bad_tile_behind_a_good_one_still_aborts";
    let arm = even();
    let outcomes = run_arm(
        test,
        "two_tiles",
        &arm,
        &[
            ("W_BAD_CFL_STEP", "3".to_string()),
            ("W_BAD_CFL_WORKER", "1".to_string()),
        ],
    );
    for (i, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {i} exited zero");
    }
    assert!(
        outcomes[1].stderr.contains("invalid CFL candidate"),
        "{}",
        outcomes[1].stderr
    );
    assert!(
        outcomes[1].stderr.contains("TileId(2)"),
        "the first held tile is named:\n{}",
        outcomes[1].stderr
    );
    assert!(
        outcomes[0].stderr.contains("aborted"),
        "{}",
        outcomes[0].stderr
    );
}
