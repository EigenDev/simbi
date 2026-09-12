// =============================================================================
// checkpoint_processes.rs
//
// the distributed checkpoint across real processes on the seed fixture: two
// workers write one file through the coordinator at a named step, and the
// file equals the single-process partitioned writer's file for the reference
// state at that step, attribute for attribute and value for value. a restart
// from that file onto three ragged tiles continues to the same final state as
// the uninterrupted reference. a withheld block leaves the ledger incomplete
// and no file published; a repeated block is a protocol error; a worker killed
// mid-write leaves no published file.
//
// run: cargo test -p symbi --test checkpoint_processes
// =============================================================================

use std::io::{BufRead, BufReader, Read};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
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
use symbi_io::{Hdf5Backend, IoBackend, Metadata, TreeBuf};
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::checkpoint::{
    GlobalGrid, LevelTiles, PhysicsIdentity, TileView, load_partitioned_level,
    read_checkpoint_meta, write_partitioned_checkpoint_with_budget,
};
use symbi_sim::distributed_checkpoint::{
    CheckpointRequest, block_credit_for, distributed_checkpoint,
};
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
const CKPT_STEP: u64 = 6;
const BUDGET: usize = 300;
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
        Self {
            cuts0: list("W_CUTS0"),
            cuts1: list("W_CUTS1"),
            owner: list("W_OWNER").into_iter().map(|x| x as u32).collect(),
        }
    }

    fn partition(&self) -> Partition<2> {
        Partition::explicit([N, N], [self.cuts0.clone(), self.cuts1.clone()]).unwrap()
    }

    fn workers(&self) -> u32 {
        self.owner.iter().max().unwrap() + 1
    }
}

fn tile(p: &Partition<2>, flat: usize) -> (Sim, Kern) {
    let counts = p.counts();
    let tc = unflatten(flat, counts);
    let ext = p.tile_extents(tc);
    let bnd = Boundaries(std::array::from_fn(|a| {
        let edge = |at_edge: bool| {
            if at_edge {
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

fn whole_grid() -> GlobalGrid<2> {
    let p = Partition::uniform([N, N], [1, 1]).unwrap();
    let (sim, _) = tile(&p, 0);
    GlobalGrid::of_state(&sim)
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

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    let mine: Vec<usize> = (0..n)
        .filter(|&f| placement.owner_of(TileId(f as u32)) == me)
        .collect();
    let mut built: Vec<(Sim, Kern)> = mine.iter().map(|&f| tile(&partition, f)).collect();
    let tiles: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let offsets: Vec<[isize; 2]> = mine
        .iter()
        .map(|&f| {
            let ext = partition.tile_extents(unflatten(f, partition.counts()));
            [ext[0].0 as isize, ext[1].0 as isize]
        })
        .collect();
    let grid = whole_grid();
    let mut start_time = 0.0;
    let mut max_steps = STEPS;
    if let Ok(from) = std::env::var("W_RESTART_FROM") {
        let meta =
            read_checkpoint_meta(&from).map_err(|e| WorkerError::Checkpoint(e.to_string()))?;
        for ((sim, _), offset) in built.iter_mut().zip(&offsets) {
            load_partitioned_level(sim, &from, 0, *offset, &grid)
                .map_err(|e| WorkerError::Checkpoint(e.to_string()))?;
        }
        start_time = meta.time;
        max_steps = env_u64("W_RESTART_STEPS").unwrap();
    }
    let identity = PhysicsIdentity::of(&built[0].0);
    let schema = plan_schema(&*built[0].0);
    let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, built[0].0.geom.ng)
        .expect("plan");
    let exchange = PlanExchange::new(&plan, &placement, me);
    let devices = vec![0i32; n];
    let credit = block_credit_for(BUDGET, 2);
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        identity: Identity {
            credential: 0xC4EC,
            build_id: "checkpoint_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            ng: built[0].0.geom.ng as u32,
            workers: arm.workers(),
            block_credit: credit,
        },
        startup: STARTUP_DEADLINE,
    };
    let lens = exchange.lens();
    let max_payload = (lens.iter().max().copied().unwrap_or(0) * 8).max(credit as usize);
    let (link, session) = connect(&r, max_payload)?;
    let mut fabric = Fabric::new(
        link,
        me,
        session,
        arm.workers() as usize,
        2,
        &lens,
        exchange.sends_per_peer_axis(),
    )
    .with_block_credit(credit);
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 2, Newtonian>(),
        timestepping: Timestepping::Rk2,
        start_time,
        t_final: T_FINAL,
        max_steps,
        deadline: TRANSFER_DEADLINE,
        injection: Injection::default(),
    };
    let ckpt_step = env_u64("W_CKPT_STEP");
    let ckpt_path = std::env::var("W_CKPT_PATH").unwrap_or_default();
    let ckpt_deadline = Duration::from_millis(env_u64("W_CKPT_DEADLINE_MS").unwrap_or(8000));
    let drop_block = if env_u64("W_DROP_WORKER") == Some(u64::from(me.0)) {
        env_u64("W_DROP_BLOCK").map(|b| b as usize)
    } else {
        None
    };
    let duplicate = env_u64("W_DUP_WORKER") == Some(u64::from(me.0));
    let vote_no = env_u64("W_VOTE_NO_WORKER") == Some(u64::from(me.0));
    let stall = env_u64("W_STALL_WORKER") == Some(u64::from(me.0));
    let mut stores: Vec<&mut Store> = Vec::new();
    let mut kernels: Vec<&Kern> = Vec::new();
    for (s, k) in built.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    let extras = Metadata::new();
    let report = evolve_worker(
        &tiles,
        &mut stores,
        &kernels,
        &devices,
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |iter, _, sh, fabric| {
            if Some(iter) != ckpt_step {
                return Ok(ControlFlow::Continue(()));
            }
            if stall {
                println!("CKPT");
                std::thread::sleep(Duration::from_secs(60));
            }
            let level = LevelTiles {
                tiles: sh
                    .iter()
                    .zip(&offsets)
                    .map(|(s, &offset)| TileView { state: *s, offset })
                    .collect(),
                grid: grid.clone(),
            };
            let req = CheckpointRequest {
                path: &ckpt_path,
                extras: &extras,
                budget: BUDGET,
                deadline: ckpt_deadline,
                drop_block,
                duplicate_first_block: duplicate,
                vote_no,
            };
            let before = fabric.buffer_bytes();
            distributed_checkpoint::<Newtonian, 2, 2, HostMemory, _>(
                fabric, &plan, &placement, &identity, &level, &tiles, &req,
            )
            .map_err(|e| WorkerError::Checkpoint(e.to_string()))?;
            if fabric.buffer_bytes() != before {
                return Err(WorkerError::Checkpoint(
                    "buffers grew during the checkpoint".into(),
                ));
            }
            Ok(ControlFlow::Continue(()))
        },
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
        "REPORT steps={} time={}",
        report.steps,
        report.time.to_bits()
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
    stdout: BufReader<std::process::ChildStdout>,
    out: PathBuf,
}

fn spawn(
    test: &str,
    me: u32,
    arm: &Arm,
    coord: &Path,
    dir: &Path,
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
    let stdout = BufReader::new(child.stdout.take().unwrap());
    Spawned {
        child,
        stderr,
        stdout,
        out,
    }
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    tiles: Vec<(usize, Vec<f64>)>,
}

fn collect(mut spawned: Vec<Spawned>) -> Vec<Outcome> {
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
    spawned
        .iter_mut()
        .map(|s| {
            let status = s.child.wait().unwrap();
            let mut stderr = String::new();
            s.stderr.read_to_string(&mut stderr).unwrap();
            let mut rest = String::new();
            s.stdout.read_to_string(&mut rest).unwrap();
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
                tiles,
            }
        })
        .collect()
}

fn scratch(test: &str, tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("symbi_ckpt_{}_{test}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_arm(test: &str, dir: &Path, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let coord = dir.join("rendezvous");
    let _ = std::fs::remove_file(&coord);
    let spawned: Vec<Spawned> = (0..arm.workers())
        .map(|w| spawn(test, w, arm, &coord, dir, extra))
        .collect();
    collect(spawned)
}

/// the reference: every tile of `arm` evolved for `steps` in one process.
fn reference(arm: &Arm, steps: u64) -> Vec<(Sim, Kern)> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let mut built: Vec<(Sim, Kern)> = (0..n).map(|f| tile(&partition, f)).collect();
    let mut stores: Vec<&mut Store> = Vec::new();
    let mut kernels: Vec<&Kern> = Vec::new();
    for (s, k) in built.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    let schedule = Schedule::derive(partition.counts(), stores[0].geom.ng, &Topology::open());
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
            if iter >= steps {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    built
}

fn assert_trees_equal(a: &TreeBuf, b: &TreeBuf, path: &str) {
    assert_eq!(a.attrs.len(), b.attrs.len(), "{path}: attribute count");
    for (x, y) in a.attrs.iter().zip(&b.attrs) {
        assert_eq!(x.0, y.0, "{path}: attribute name");
        assert_eq!(
            format!("{:?}", x.1),
            format!("{:?}", y.1),
            "{path}: attribute {}",
            x.0
        );
    }
    assert_eq!(a.datasets.len(), b.datasets.len(), "{path}: dataset count");
    for (x, y) in a.datasets.iter().zip(&b.datasets) {
        assert_eq!(x.name, y.name, "{path}: dataset name");
        assert_eq!(x.shape, y.shape, "{path}/{}: shape", x.name);
        match (x.data.as_f64(), y.data.as_f64()) {
            (Some(p), Some(q)) => {
                assert_eq!(p.len(), q.len(), "{path}/{}: length", x.name);
                for (i, (u, v)) in p.iter().zip(q).enumerate() {
                    assert_eq!(
                        u.to_bits(),
                        v.to_bits(),
                        "{path}/{}[{i}]: {u} vs {v}",
                        x.name
                    );
                }
            }
            _ => assert_eq!(
                format!("{:?}", x.data),
                format!("{:?}", y.data),
                "{path}/{}",
                x.name
            ),
        }
    }
    assert_eq!(
        a.groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
        b.groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
        "{path}: groups"
    );
    for (x, y) in a.groups.iter().zip(&b.groups) {
        assert_trees_equal(x, y, &format!("{path}/{}", x.name));
    }
}

fn assert_all_ok(outcomes: &[Outcome]) {
    for (i, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "worker {i} failed:\n{}", o.stderr);
    }
}

fn assert_tiles_match(outcomes: &[Outcome], want: &[(Sim, Kern)], label: &str) {
    let mut compared = 0;
    for o in outcomes {
        for (flat, got) in &o.tiles {
            let w = tile_values(&want[*flat].0);
            assert_eq!(got.len(), w.len(), "{label}: tile {flat} value count");
            for (i, (g, w)) in got.iter().zip(&w).enumerate() {
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
        owner: vec![0, 0, 1, 1],
    }
}

fn ragged_three() -> Arm {
    Arm {
        cuts0: vec![21, 43],
        cuts1: vec![32],
        owner: vec![0, 1, 2, 0, 1, 2],
    }
}

fn ckpt_env(dir: &Path) -> Vec<(&'static str, String)> {
    vec![
        ("W_CKPT_STEP", CKPT_STEP.to_string()),
        (
            "W_CKPT_PATH",
            dir.join("ckpt.h5").to_string_lossy().into_owned(),
        ),
    ]
}

/// two workers write the checkpoint at step 6; it equals the partitioned writer's file for
/// the reference state, and the workers finish the run bitwise against the reference.
#[test]
fn two_workers_write_the_reference_checkpoint() {
    if in_worker_role() {
        worker_main();
    }
    let test = "two_workers_write_the_reference_checkpoint";
    let dir = scratch(test, "write");
    let arm = even();
    let outcomes = run_arm(test, &dir, &arm, &ckpt_env(&dir));
    assert_all_ok(&outcomes);
    let at_six = reference(&arm, CKPT_STEP);
    let partition = arm.partition();
    let level = LevelTiles {
        tiles: (0..partition.n_tiles())
            .map(|f| {
                let ext = partition.tile_extents(unflatten(f, partition.counts()));
                TileView {
                    state: &*at_six[f].0,
                    offset: [ext[0].0 as isize, ext[1].0 as isize],
                }
            })
            .collect(),
        grid: whole_grid(),
    };
    let ref_path = dir.join("reference.h5");
    write_partitioned_checkpoint_with_budget::<Newtonian, 2, 2, HostMemory>(
        &PhysicsIdentity::of(&at_six[0].0),
        &[level],
        ref_path.to_str().unwrap(),
        &Metadata::new(),
        BUDGET,
    )
    .unwrap();
    let got = Hdf5Backend.read(&dir.join("ckpt.h5")).unwrap();
    let want = Hdf5Backend.read(&ref_path).unwrap();
    assert_trees_equal(&got, &want, "");
    assert_tiles_match(&outcomes, &reference(&arm, STEPS), "after the checkpoint");
    let _ = std::fs::remove_dir_all(&dir);
}

/// the two-worker file restarted onto three ragged tiles continues to the state the
/// uninterrupted reference reaches at step 12.
#[test]
fn restart_onto_a_new_partition_continues_bitwise() {
    if in_worker_role() {
        worker_main();
    }
    let test = "restart_onto_a_new_partition_continues_bitwise";
    let dir = scratch(test, "restart");
    let writers = run_arm(test, &dir, &even(), &ckpt_env(&dir));
    assert_all_ok(&writers);
    let file = dir.join("ckpt.h5");
    assert!(file.exists());
    let restarted = run_arm(
        test,
        &dir,
        &ragged_three(),
        &[
            ("W_RESTART_FROM", file.to_string_lossy().into_owned()),
            ("W_RESTART_STEPS", (STEPS - CKPT_STEP).to_string()),
        ],
    );
    assert_all_ok(&restarted);
    assert_tiles_match(&restarted, &reference(&ragged_three(), STEPS), "restarted");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_block_leaves_no_file() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_missing_block_leaves_no_file";
    let dir = scratch(test, "drop");
    let mut env = ckpt_env(&dir);
    env.push(("W_DROP_WORKER", "1".into()));
    env.push(("W_DROP_BLOCK", "2".into()));
    let outcomes = run_arm(test, &dir, &even(), &env);
    for (i, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {i} exited zero:\n{}", o.stderr);
    }
    assert!(
        outcomes[0].stderr.contains("ledger is incomplete"),
        "{}",
        outcomes[0].stderr
    );
    assert!(!dir.join("ckpt.h5").exists(), "a file was published");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_overlapping_block_is_a_protocol_error() {
    if in_worker_role() {
        worker_main();
    }
    let test = "an_overlapping_block_is_a_protocol_error";
    let dir = scratch(test, "dup");
    let mut env = ckpt_env(&dir);
    env.push(("W_DUP_WORKER", "1".into()));
    let outcomes = run_arm(test, &dir, &even(), &env);
    for (i, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {i} exited zero:\n{}", o.stderr);
    }
    assert!(
        outcomes[0].stderr.contains("overlaps"),
        "{}",
        outcomes[0].stderr
    );
    assert!(!dir.join("ckpt.h5").exists(), "a file was published");
    let _ = std::fs::remove_dir_all(&dir);
}

/// worker 1 is killed once it reaches the checkpoint step; the coordinator's write hits the
/// checkpoint deadline and publishes nothing.
#[test]
fn a_worker_killed_during_the_write_publishes_nothing() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_worker_killed_during_the_write_publishes_nothing";
    let dir = scratch(test, "kill");
    let mut env = ckpt_env(&dir);
    env.push(("W_STALL_WORKER", "1".into()));
    env.push(("W_CKPT_DEADLINE_MS", "2000".into()));
    let arm = even();
    let coord = dir.join("rendezvous");
    let a = spawn(test, 0, &arm, &coord, &dir, &env);
    let mut b = spawn(test, 1, &arm, &coord, &dir, &env);
    let mut line = String::new();
    loop {
        line.clear();
        let n = b.stdout.read_line(&mut line).unwrap();
        if n == 0 || line.trim_end().ends_with("CKPT") {
            break;
        }
    }
    b.child.kill().unwrap();
    let outcomes = collect(vec![a, b]);
    assert_ne!(
        outcomes[0].code,
        Some(0),
        "the coordinator exited zero:\n{}",
        outcomes[0].stderr
    );
    assert!(!dir.join("ckpt.h5").exists(), "a file was published");
    let _ = std::fs::remove_dir_all(&dir);
}

/// every block arrives, then worker 1 votes against the close: the coordinator publishes
/// nothing and every worker reports the refusal.
#[test]
fn a_negative_close_vote_after_complete_data_publishes_nothing() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_negative_close_vote_after_complete_data_publishes_nothing";
    let dir = scratch(test, "vote");
    let mut env = ckpt_env(&dir);
    env.push(("W_VOTE_NO_WORKER", "1".into()));
    let outcomes = run_arm(test, &dir, &even(), &env);
    for (i, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {i} exited zero:\n{}", o.stderr);
    }
    assert!(
        outcomes[0]
            .stderr
            .contains("a worker reported a failed write"),
        "{}",
        outcomes[0].stderr
    );
    assert!(!dir.join("ckpt.h5").exists(), "a file was published");
    let _ = std::fs::remove_dir_all(&dir);
}
