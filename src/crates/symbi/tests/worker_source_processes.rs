// =============================================================================
// worker_source_processes.rs
//
// the distributed evolution with user sources across real processes. two
// independently parameterized forces sum to a = [x, 0] with the position leaf
// reading the cell's global x, so a tile that evaluated its source at a
// tile-local coordinate would depart from the one-worker run at every cut.
// tiles are placed as production places them (the whole grid's origin and an
// index offset) on 60 cells a side, a spacing no power of two divides, so a
// tile-local coordinate or a per-tile origin lands an ulp or a cut away. two
// workers, four, and three tiles owned unevenly reach the one-worker state bit
// for bit, and the sources changed the run.
//
// run: cargo test -p symbi --test worker_source_processes
// =============================================================================

use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{
    LocalCopy, Partition, Topology, plan_fields, plan_schema, unflatten,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::newtonian::Newtonian;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_user_sources;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Store = FieldStore<2, 2, HostMemory, f64>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
/// cells per axis: no power of two divides the spacing 1/60
const N: usize = 60;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const STEPS: u64 = 12;
const T_FINAL: f64 = 10.0;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(8);
const STARTUP_DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(60);

const SOURCE_JSON_1: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [2, 3], "params": [0.25],
    "vocabulary":{"reads":["x_0"],"params":[0]},
    "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "VARIABLE_X1"},
               {"op": "MULTIPLY", "left": 0, "right": 1},
               {"op": "CONSTANT", "value": 0.0} ]
}"#;

const SOURCE_JSON_2: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [2, 3], "params": [0.75],
    "vocabulary":{"reads":["x_0"],"params":[0]},
    "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "VARIABLE_X1"},
               {"op": "MULTIPLY", "left": 0, "right": 1},
               {"op": "CONSTANT", "value": 0.0} ]
}"#;

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
    // a cut face is filled by the exchange; a domain face on a wrapped axis is a cut too when
    // the axis is cut, and the tile's own periodic fill when it is not
    let bnd = Boundaries(std::array::from_fn(|a| {
        let edge = |at_edge: bool| {
            if at_edge && !arm.periodic[a] {
                BoundaryType::Outflow
            } else if arm.periodic[a] && counts[a] == 1 {
                BoundaryType::Periodic
            } else {
                BoundaryType::CoarseFine
            }
        };
        [edge(tc[a] == 0), edge(tc[a] == counts[a] - 1)]
    }));
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([DX; 2])
        // the production placement: the whole grid's origin and the tile's index offset, so every
        // cell center is the single grid's to the bit on a spacing no power of two divides
        .origin([0.0, 0.0])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
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
    let k = if std::env::var("W_SOURCES").is_ok() {
        let configs = [SOURCE_JSON_1, SOURCE_JSON_2]
            .map(|json| SourceConfig::from_json(json).expect("parse source config"));
        let (built, params) = build_user_sources(&configs, &NEWTONIAN_SPEC).expect("lower sources");
        k.with_runtime_source(built, params)
    } else {
        k
    };
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
    // W_REJECT_STEP and W_REJECT_STAGE list the (step, stage) pairs, comma separated
    if env_u64("W_REJECT_WORKER") == Some(u64::from(me.0)) {
        let list = |name: &str| -> Vec<u64> {
            std::env::var(name)
                .unwrap()
                .split(',')
                .map(|s| s.parse().unwrap())
                .collect()
        };
        injection.reject_at = list("W_REJECT_STEP")
            .into_iter()
            .zip(list("W_REJECT_STAGE"))
            .map(|(s, g)| (s, g as usize))
            .collect();
    }
    if let (Some(step), Some(bits)) = (env_u64("W_CAP_DT_STEP"), env_u64("W_CAP_DT_BITS")) {
        injection.cap_dt_at = Some((step, bits));
    }
    if env_u64("W_BAD_CFL_WORKER") == Some(u64::from(me.0)) {
        injection.invalid_cfl_at = env_u64("W_BAD_CFL_STEP");
    }
    // every worker carries the troubled cell; the one holding it raises the flag
    if let Some(step) = env_u64("W_TROUBLE_STEP") {
        injection.trouble_at = Some((
            step,
            env_u64("W_TROUBLE_STAGE").unwrap() as usize,
            [
                env_u64("W_TROUBLE_I").unwrap() as isize,
                env_u64("W_TROUBLE_J").unwrap() as isize,
                0,
            ],
        ));
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
    let exchange = PlanExchange::new(&plan, &placement, me, PhaseTable::flat(2).unwrap());
    let devices = vec![0i32; n];
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0xBEEF,
            build_id: "worker_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: PhaseTable::flat(2).unwrap().digest(),
            ng: built[0].0.geom.ng as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    };
    let lens = exchange.lens();
    let max_payload = lens.iter().max().copied().unwrap_or(0) * 8;
    let (link, session) = connect(&r, max_payload.max(64))?;
    let mut fabric = exchange.fabric(link, session);
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 2, Newtonian>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: T_FINAL,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection: injection_from_env(me),
        audit_every_point: false,
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
    std::fs::write(&out, bytes).unwrap();
    // the conserved density of every owned cell under its global index, so the parent can
    // assemble one grid from any partition
    let mut den = Vec::new();
    for (flat, (sim, _)) in mine.iter().zip(&built) {
        let ext = partition.tile_extents(unflatten(*flat, partition.counts()));
        let lo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        let view = sim.fields.cons.den.view();
        for c in sim.geom.interior.iter() {
            let g: [u64; 2] = std::array::from_fn(|a| (ext[a].0 as isize + c[a] - lo[a]) as u64);
            den.extend_from_slice(&g[0].to_le_bytes());
            den.extend_from_slice(&g[1].to_le_bytes());
            den.extend_from_slice(&view.at(c).to_bits().to_le_bytes());
        }
    }
    std::fs::write(out.with_extension("den"), den).unwrap();
    println!(
        "REPORT steps={} rejections={} time={} exchanges={} neighbor={} troubled={} frozen={} dts={}",
        report.steps,
        report.rejections,
        report.time.to_bits(),
        report.troubled_exchanges,
        report.neighbor_corrections,
        report.troubled_cells,
        report.frozen_cells,
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


struct Outcome {
    code: Option<i32>,
    stderr: String,
    /// (global i, global j, conserved density bits) of every cell this worker holds
    density: Vec<(usize, usize, u64)>,
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
            let density = std::fs::read(s.out.with_extension("den"))
                .map(|bytes| {
                    bytes
                        .chunks_exact(24)
                        .map(|c| {
                            let word = |k: usize| {
                                u64::from_le_bytes(c[8 * k..8 * k + 8].try_into().unwrap())
                            };
                            (word(0) as usize, word(1) as usize, word(2))
                        })
                        .collect()
                })
                .unwrap_or_default();
            Outcome {
                code: status.code(),
                stderr,
                density,
            }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

fn assert_all_ok(outcomes: &[Outcome]) {
    for (i, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "worker {i} failed:\n{}", o.stderr);
    }
}

fn even() -> Arm {
    Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [false, false],
        owner: vec![0, 0, 1, 1],
    }
}

/// the conserved density of an arm on the global grid, every cell written exactly once.
fn global_density(outcomes: &[Outcome]) -> Vec<u64> {
    let mut grid = vec![None; N * N];
    for o in outcomes {
        for &(i, j, bits) in &o.density {
            assert!(grid[i * N + j].replace(bits).is_none(), "cell ({i},{j}) written twice");
        }
    }
    grid.into_iter()
        .map(|c| c.expect("a cell no worker wrote"))
        .collect()
}

fn uncut() -> Arm {
    Arm { cuts0: vec![], cuts1: vec![], periodic: [false, false], owner: vec![0] }
}

#[test]
fn sources_evaluated_at_global_coordinates_match_one_worker() {
    if in_worker_role() {
        worker_main();
    }
    let test = "sources_evaluated_at_global_coordinates_match_one_worker";
    let sourced = vec![("W_SOURCES", "1".to_string())];
    let plain = run_arm(test, "plain", &uncut(), &[]);
    assert_all_ok(&plain);
    let one = run_arm(test, "one", &uncut(), &sourced);
    assert_all_ok(&one);
    let want = global_density(&one);
    assert_ne!(global_density(&plain), want, "the sources left the run unchanged; the gate is vacuous");
    for (label, arm) in [
        ("two workers", even()),
        ("four workers", Arm { cuts0: vec![32], cuts1: vec![32], periodic: [false, false], owner: vec![0, 1, 2, 3] }),
        ("three tiles on two workers", Arm { cuts0: vec![20, 44], cuts1: vec![], periodic: [false, false], owner: vec![0, 0, 1] }),
    ] {
        let outcomes = run_arm(test, label.split(' ').next().unwrap(), &arm, &sourced);
        assert_all_ok(&outcomes);
        assert_eq!(global_density(&outcomes), want, "{label}: density differs from one worker");
    }
}
