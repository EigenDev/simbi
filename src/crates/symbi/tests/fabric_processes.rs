// =============================================================================
// fabric_processes.rs
//
// the plan-driven halo exchange across real processes on the seed fixture:
// each worker process builds only the tiles it holds, compiles the plan from
// the partition, rendezvouses over TCP, runs the two axis phases under grants,
// and writes every cell and face of its tiles to a file. the parent builds the
// whole tile set, runs the fused whole-ownership exchange, and compares each
// worker's tiles bitwise. the same exchange contract must survive independent
// process timing: a worker that arrives late at a phase, and a four-worker grid
// whose axis-0 neighbor differs from its axis-1 neighbor.
//
// run: cargo test -p symbi --test fabric_processes
// =============================================================================

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::sim::decomp::{
    LocalCopy, Partition, Schedule, Topology, exchange_grid, plan_fields, plan_schema, unflatten,
};
use symbi::sim::state::{Boundaries, BoundaryType, FieldStore, SimStateGeneric, Timestepping};
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, Fabric, FabricError, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_sim::atlas::{ExchangePlan, ExchangePoint, Placement};
use symbi_sim::plan_exchange::PlanExchange;
use symbi_xpu::{CpuSpace, HostMemory};

type Hydro = SimStateGeneric<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Store = FieldStore<2, 2, HostMemory, f64>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 64;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(5);
const STARTUP_DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(40);

fn tile(p: &Partition<2>, flat: usize, ng: usize) -> Hydro {
    let ext = p.tile_extents(unflatten(flat, p.counts()));
    Hydro::new_at(
        Newtonian,
        IdealGas { gamma: 1.4 },
        Cartesian,
        [0, 0],
        [ext[0].1, ext[1].1],
        [ext[0].0 as f64, ext[1].0 as f64],
        [1.0, 1.0],
        ng,
        Boundaries::uniform(BoundaryType::Outflow),
        0.4,
        Timestepping::Rk2,
        0,
    )
    .unwrap()
}

fn seed(flat: usize, tag: usize, c: [isize; 2]) -> f64 {
    1.0 + 0.01 * tag as f64 + 0.1 * flat as f64 + 0.001 * c[0] as f64 + 0.0007 * c[1] as f64
}

fn fill(store: &Store, flat: usize) {
    for (tag, field) in plan_fields(store).into_iter().enumerate() {
        let mut view = field.view_mut();
        for c in field.domain().iter() {
            *view.at_mut(c) = seed(flat, tag, c);
        }
    }
}

fn tile_values(store: &Store) -> Vec<f64> {
    let mut out = Vec::new();
    for field in plan_fields(store) {
        let view = field.view();
        for c in field.domain().iter() {
            out.push(*view.at(c));
        }
    }
    out
}

/// an arm: the cuts, the topology, the owner map, and the halo width, all passed to the
/// workers through the environment so parent and worker compile the same plan.
#[derive(Clone)]
struct Arm {
    cuts0: Vec<usize>,
    cuts1: Vec<usize>,
    periodic: [bool; 2],
    owner: Vec<u32>,
    ng: usize,
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
            ("W_NG", self.ng.to_string()),
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
        let periodic: Vec<usize> = list("W_PERIODIC");
        Self {
            cuts0: list("W_CUTS0"),
            cuts1: list("W_CUTS1"),
            periodic: [periodic[0] == 1, periodic[1] == 1],
            owner: list("W_OWNER").into_iter().map(|x| x as u32).collect(),
            ng: list("W_NG")[0],
        }
    }

    fn partition(&self) -> Partition<2> {
        Partition::explicit([N, N], [self.cuts0.clone(), self.cuts1.clone()]).unwrap()
    }

    fn workers(&self) -> u32 {
        self.owner.iter().max().unwrap() + 1
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), FabricError> {
    let partition = arm.partition();
    let topology = Topology::wrapping(arm.periodic);
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    // this worker builds its own tiles only; the schema comes from one of them.
    let mine: Vec<(usize, Hydro)> = (0..n)
        .filter(|&f| placement.owner_of(symbi_sim::atlas::TileId(f as u32)) == me)
        .map(|f| (f, tile(&partition, f, arm.ng)))
        .collect();
    for (flat, sim) in &mine {
        fill(sim, *flat);
    }
    let schema = plan_schema(&*mine[0].1);
    let plan = ExchangePlan::compile(&partition, &topology, &schema, arm.ng).expect("plan");
    let exchange = PlanExchange::new(&plan, &placement, me);
    let mut fields: Vec<Vec<&symbi_grid::Field<f64, 2, HostMemory>>> =
        (0..n).map(|_| Vec::new()).collect();
    for (flat, sim) in &mine {
        fields[*flat] = plan_fields(sim);
    }
    let devices = vec![0i32; n];
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        identity: Identity {
            credential: 0xF00D,
            build_id: "fabric_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            ng: arm.ng as u32,
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
    let result = run_phases(&exchange, &fields, &devices, &mut fabric, me);
    if let Err(e) = &result {
        fabric.abort(&e.to_string());
    }
    result?;
    let mut bytes = Vec::new();
    for (flat, sim) in &mine {
        bytes.extend_from_slice(&(*flat as u64).to_le_bytes());
        let values = tile_values(sim);
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    std::fs::write(out, bytes).unwrap();
    Ok(())
}

fn run_phases(
    exchange: &PlanExchange<'_, 2>,
    fields: &[Vec<&symbi_grid::Field<f64, 2, HostMemory>>],
    devices: &[i32],
    fabric: &mut Fabric<symbi_fabric::TcpLink>,
    me: WorkerId,
) -> Result<(), FabricError> {
    let slow_ms = env_u64("W_SLOW_MS", 0);
    let slow_worker = env_u64("W_SLOW_WORKER", u64::MAX);
    for axis in 0..2 {
        if slow_ms > 0 && u64::from(me.0) == slow_worker {
            // this worker arrives late at every phase, servicing the link as it waits
            fabric.service_for(Duration::from_millis(slow_ms))?;
        }
        let epoch = PlanExchange::<2>::epoch(ExchangePoint::Prime, 0, 0, axis);
        exchange.open_axis(fields, devices, &LocalCopy, fabric, epoch)?;
        fabric.wait(TRANSFER_DEADLINE)?;
        exchange.finish_axis(fields, fabric, axis)?;
    }
    fabric.finish(TRANSFER_DEADLINE)
}

fn worker_main() -> ! {
    let arm = Arm::from_env();
    let me = WorkerId(env_u64("W_ME", 0) as u32);
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
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    for (k, v) in arm.env() {
        cmd.env(k, v);
    }
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    let stderr = child.stderr.take().unwrap();
    Spawned { child, stderr, out }
}

/// the fused reference: every tile built, filled, and exchanged in one process.
fn reference(arm: &Arm) -> Vec<Vec<f64>> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let tiles: Vec<Hydro> = (0..n).map(|f| tile(&partition, f, arm.ng)).collect();
    let stores: Vec<&Store> = tiles.iter().map(|s| &**s).collect();
    for (flat, s) in stores.iter().enumerate() {
        fill(s, flat);
    }
    let before: Vec<Vec<f64>> = stores.iter().map(|s| tile_values(s)).collect();
    let schedule = Schedule::derive(
        partition.counts(),
        arm.ng,
        &Topology::wrapping(arm.periodic),
    );
    exchange_grid(&stores, &schedule, &vec![0; n], &LocalCopy);
    let after: Vec<Vec<f64>> = stores.iter().map(|s| tile_values(s)).collect();
    let touched = before
        .iter()
        .flatten()
        .zip(after.iter().flatten())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        touched > 0,
        "the exchange wrote no cells; the seam is not under test"
    );
    after
}

fn run_arm(test: &str, arm: &Arm, extra: &[(&str, String)]) {
    let dir = std::env::temp_dir().join(format!("symbi_fabric_{}_{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let workers = arm.workers();
    let mut spawned: Vec<Spawned> = (0..workers)
        .map(|w| spawn(test, w, arm, &coord, &dir, extra))
        .collect();
    let start = Instant::now();
    loop {
        let mut all = true;
        for s in spawned.iter_mut() {
            if s.child.try_wait().unwrap().is_none() {
                all = false;
            }
        }
        if all {
            break;
        }
        if start.elapsed() > WATCHDOG {
            for s in spawned.iter_mut() {
                let _ = s.child.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let want = reference(arm);
    let mut compared = 0;
    for s in spawned.iter_mut() {
        let status = s.child.wait().unwrap();
        let mut stderr = String::new();
        s.stderr.read_to_string(&mut stderr).unwrap();
        assert_eq!(status.code(), Some(0), "worker failed:\n{stderr}");
        let bytes = std::fs::read(&s.out).unwrap();
        let mut pos = 0;
        while pos < bytes.len() {
            let flat = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap()) as usize;
            pos += 16;
            assert_eq!(len, want[flat].len(), "tile {flat}: value count");
            for (i, w) in want[flat].iter().enumerate() {
                let got = f64::from_bits(u64::from_le_bytes(
                    bytes[pos + 8 * i..pos + 8 * i + 8].try_into().unwrap(),
                ));
                assert!(
                    got.to_bits() == w.to_bits(),
                    "tile {flat} value {i}: process exchange {got:e} != fused {w:e}"
                );
            }
            pos += 8 * len;
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        arm.partition().n_tiles(),
        "every tile was compared exactly once"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn two_processes_even_cut() {
    if in_worker_role() {
        worker_main();
    }
    let arm = Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [false, false],
        owner: vec![0, 0, 1, 1],
        ng: 2,
    };
    run_arm("two_processes_even_cut", &arm, &[]);
}

#[test]
fn two_processes_ragged_cuts() {
    if in_worker_role() {
        worker_main();
    }
    let arm = Arm {
        cuts0: vec![19, 37],
        cuts1: vec![27],
        periodic: [false, false],
        owner: vec![0, 1, 0, 1, 1, 0],
        ng: 3,
    };
    run_arm("two_processes_ragged_cuts", &arm, &[]);
}

#[test]
fn two_processes_periodic_wrap() {
    if in_worker_role() {
        worker_main();
    }
    let arm = Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [true, true],
        owner: vec![0, 0, 1, 1],
        ng: 2,
    };
    run_arm("two_processes_periodic_wrap", &arm, &[]);
}

/// one tile per worker on a 2x2 grid: every worker's axis-0 peer differs from its axis-1 peer.
#[test]
fn four_processes_distinct_neighbors() {
    if in_worker_role() {
        worker_main();
    }
    for ng in [2usize, 3] {
        let arm = Arm {
            cuts0: vec![32],
            cuts1: vec![32],
            periodic: [false, false],
            owner: vec![0, 1, 2, 3],
            ng,
        };
        run_arm("four_processes_distinct_neighbors", &arm, &[]);
    }
}

/// the same grid with worker 2 arriving late at every phase.
#[test]
fn four_processes_with_one_slow_worker() {
    if in_worker_role() {
        worker_main();
    }
    let arm = Arm {
        cuts0: vec![32],
        cuts1: vec![32],
        periodic: [true, false],
        owner: vec![0, 1, 2, 3],
        ng: 2,
    };
    run_arm(
        "four_processes_with_one_slow_worker",
        &arm,
        &[("W_SLOW_MS", "400".into()), ("W_SLOW_WORKER", "2".into())],
    );
}
