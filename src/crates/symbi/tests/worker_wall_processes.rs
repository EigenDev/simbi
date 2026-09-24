// =============================================================================
// worker_wall_processes.rs
//
// the domain walls of a distributed run across real processes. a tile at the
// domain edge fills its wall ghosts itself, after the cut halos arrive, so the
// corner ghosts where a cut meets a wall are the wall's image of the
// neighbor's cells. a smooth pulse spinning inside a closed box reaches every
// wall, and the first-order correction is fired on the cell where a cut meets
// a wall, so its redo reads wall ghosts and cut halos in one stencil. the
// walls: reflect, outflow, and the prescribed-gradient walls (Neumann, Robin)
// for adiabatic hydro; reflect and outflow for ideal MHD with a uniform field
// crossing the walls. two workers with the cell against the low wall and four
// workers with the cell against the high wall each reproduce the one-worker
// run bit for bit, cells and faces, with every shared face's copies equal and
// div B at round-off; each wall is shown to change the run against an outflow
// box, and the flag to change the one-worker run.
//
// run: cargo test -p symbi --test worker_wall_processes
// =============================================================================

use std::collections::BTreeMap;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::prelude::SimSubstrate;
use symbi::regimes::substrate_kernels::GradientBc;
use symbi::sim::decomp::{LocalCopy, Partition, Topology, plan_schema, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type GasSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type MhdSim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 64;
const H: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const STEPS: u64 = 8;
const TROUBLE_STEP: u64 = 3;
const DIV_TOL: f64 = 1.0e-10;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(8);
const STARTUP_DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(120);

#[derive(Clone)]
struct Arm {
    cuts0: Vec<usize>,
    cuts1: Vec<usize>,
    owner: Vec<u32>,
}

impl Arm {
    fn env(&self) -> Vec<(&'static str, String)> {
        let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        vec![
            ("W_CUTS0", list(&self.cuts0)),
            ("W_CUTS1", list(&self.cuts1)),
            ("W_OWNER", self.owner.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
        ]
    }

    fn from_env() -> Self {
        let list = |name: &str| -> Vec<usize> {
            std::env::var(name)
                .unwrap_or_default()
                .split(',')
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

/// the wall kind under test, from W_WALL; the gradient walls register their coefficient
/// table on the kernel set as id 0, which every wall face of the tile names
fn wall_type(wall: &str) -> BoundaryType {
    match wall {
        "reflect" => BoundaryType::Reflect,
        "outflow" => BoundaryType::Outflow,
        "neumann" => BoundaryType::Neumann(0),
        "robin" => BoundaryType::Robin(0),
        other => panic!("unknown wall {other}"),
    }
}

/// the prescribed gradients per primitive, in order rho, vx, vy, pre: outward derivatives for
/// the neumann wall, (a, b, c) of a*U + b*dU/dn = c for the robin wall
fn gradient_table(wall: &str) -> Option<GradientBc> {
    match wall {
        "neumann" => Some(GradientBc::Neumann(vec![0.2, 0.0, 0.1, -0.3])),
        "robin" => Some(GradientBc::Robin(vec![
            [1.0, 0.2, 1.1],
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.3, 1.2],
        ])),
        _ => None,
    }
}

/// a cut face is filled by the exchange; a domain face is the wall
fn boundaries(arm: &Arm, flat: usize, wall: BoundaryType) -> Boundaries<2> {
    let p = arm.partition();
    let counts = p.counts();
    let tc = unflatten(flat, counts);
    Boundaries(std::array::from_fn(|a| {
        let edge = |at_edge: bool| if at_edge { wall } else { BoundaryType::CoarseFine };
        [edge(tc[a] == 0), edge(tc[a] == counts[a] - 1)]
    }))
}

fn extents(arm: &Arm, flat: usize) -> [(usize, usize); 2] {
    let p = arm.partition();
    p.tile_extents(unflatten(flat, p.counts()))
}

/// a dense, over-pressured pulse off the box center, spun by a solid-body rotation about the
/// center so the flow reaches every wall within the run
fn pulse(x: f64, y: f64) -> (f64, [f64; 2], f64) {
    let g = (-((x - 0.4) * (x - 0.4) + (y - 0.6) * (y - 0.6)) / 0.02).exp();
    (1.0 + 0.5 * g, [-0.3 * (y - 0.5), 0.3 * (x - 0.5)], 1.0 + g)
}

fn gas_tile(arm: &Arm, flat: usize, wall: BoundaryType) -> GasSim {
    let ext = extents(arm, flat);
    GasSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([H, H])
        .origin([ext[0].0 as f64 * H, ext[1].0 as f64 * H])
        .boundaries(boundaries(arm, flat, wall))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let (rho, v, p) = pulse(x, y);
            Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p))
        })
        .build()
}

/// the pulse in a uniform field crossing both walls at an angle
const B: [f64; 3] = [0.5, 0.3, 0.0];

fn mhd_tile(arm: &Arm, flat: usize, wall: BoundaryType) -> MhdSim {
    let ext = extents(arm, flat);
    MhdSim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([H, H])
        .origin([ext[0].0 as f64 * H, ext[1].0 as f64 * H])
        .boundaries(boundaries(arm, flat, wall))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let (rho, v, p) = pulse(x, y);
            MhdPrim::new(
                Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], 0.0]), Pressure(p)),
                Tensor::new(B),
            )
        })
        .seed_faces(|axis, _| B[axis])
        .build()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// a record of what a worker holds: (kind, global i, global j, bits). kind 0 is the
/// conserved density of an owned cell; kind 1 + d is a face of component d, the tile's
/// closing face included, so a face on a cut is recorded by both of its tiles
fn push_record(out: &mut Vec<u8>, kind: u64, g: [usize; 2], v: f64) {
    for w in [kind, g[0] as u64, g[1] as u64, v.to_bits()] {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

fn dump_density<const DOF: usize>(out: &mut Vec<u8>, sim: &FieldStore<2, DOF, HostMemory, f64>, ext: [(usize, usize); 2]) {
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    let den = sim.fields.cons.den.view();
    for c in sim.geom.interior.iter() {
        let g = [ext[0].0 + (c[0] - lo[0]) as usize, ext[1].0 + (c[1] - lo[1]) as usize];
        push_record(out, 0, g, *den.at(c));
    }
}

fn dump_faces(out: &mut Vec<u8>, sim: &MhdSim, ext: [(usize, usize); 2]) {
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    let mhd = sim.fields.mhd.as_ref().unwrap();
    for d in 0..2 {
        let view = mhd.bface.b[d].view();
        for i in 0..=ext[0].1 {
            for j in 0..=ext[1].1 {
                if (d == 0 && j == ext[1].1) || (d == 1 && i == ext[0].1) {
                    continue;
                }
                let g = [ext[0].0 + i, ext[1].0 + j];
                push_record(out, 1 + d as u64, g, *view.at([lo[0] + i as isize, lo[1] + j as isize]));
            }
        }
    }
}

fn tile_div(sim: &MhdSim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let (bx, by) = (mhd.bface.b[0].view(), mhd.bface.b[1].view());
    let mut worst = 0.0f64;
    for c in sim.geom.interior.iter() {
        let div = (*bx.at([c[0] + 1, c[1]]) - *bx.at(c)) / H + (*by.at([c[0], c[1] + 1]) - *by.at(c)) / H;
        worst = worst.max(div.abs());
    }
    worst
}

struct Session {
    partition: Partition<2>,
    placement: Placement,
    mine: Vec<usize>,
    tiles: Vec<TileId>,
    devices: Vec<i32>,
    me: WorkerId,
}

fn session(arm: &Arm, me: WorkerId) -> Session {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    let mine: Vec<usize> = (0..n).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    let tiles = mine.iter().map(|&f| TileId(f as u32)).collect();
    Session { partition, placement, mine, tiles, devices: vec![0i32; n], me }
}

fn rendezvous(s: &Session, arm: &Arm, plan_digest: Digest, ng: usize) -> Rendezvous {
    Rendezvous {
        me: s.me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0x3A11,
            build_id: "worker_wall_processes".into(),
            config_digest: Digest(1),
            plan_digest,
            placement_digest: s.placement.digest(),
            ng: ng as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    }
}

fn config(regime: RegimeKind) -> WorkerConfig {
    let mut injection = Injection::default();
    if let Some(stage) = env_u64("W_TROUBLE_STAGE") {
        injection.trouble_at = Some((
            TROUBLE_STEP,
            stage as usize,
            [env_u64("W_TROUBLE_I").unwrap() as isize, env_u64("W_TROUBLE_J").unwrap() as isize, 0],
        ));
    }
    WorkerConfig {
        regime,
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 10.0,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection,
        audit_every_point: true,
    }
}

fn print_report(report: &symbi_sim::worker::WorkerReport, div: f64) {
    println!(
        "REPORT steps={} exchanges={} neighbor={} troubled={} frozen={} audits={} div={}",
        report.steps,
        report.troubled_exchanges,
        report.neighbor_corrections,
        report.troubled_cells,
        report.frozen_cells,
        report.audits,
        div.to_bits(),
    );
}

fn run_gas(arm: &Arm, me: WorkerId, out: PathBuf, wall: &str) -> Result<(), WorkerError> {
    let s = session(arm, me);
    let kind = wall_type(wall);
    let mut built: Vec<GasSim> = s.mine.iter().map(|&f| gas_tile(arm, f, kind)).collect();
    let kernels: Vec<_> = built
        .iter()
        .map(|sim| {
            let k = sim.substrate();
            match gradient_table(wall) {
                Some(table) => k.with_gradient_boundary(table).0,
                None => k,
            }
        })
        .collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(&s.partition, &Topology::open(), &schema, built[0].geom.ng).expect("plan");
    let exchange = PlanExchange::new(&plan, &s.placement, me);
    let r = rendezvous(&s, arm, plan.digest, built[0].geom.ng);
    let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
    let (link, session_id) = connect(&r, max_payload)?;
    let mut fabric = exchange.fabric(link, session_id);
    let cfg = config(RegimeKind::of::<f64, 2, Newtonian>());
    let mut stores: Vec<&mut FieldStore<2, 2, HostMemory, f64>> = built.iter_mut().map(|sim| &mut **sim).collect();
    let kernel_refs: Vec<_> = kernels.iter().collect();
    let report = evolve_worker(
        &s.tiles,
        &mut stores,
        &kernel_refs,
        &s.devices,
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )?;
    fabric.finish(TRANSFER_DEADLINE)?;
    drop(stores);
    let mut bytes = Vec::new();
    for (sim, &flat) in built.iter().zip(&s.mine) {
        dump_density(&mut bytes, sim, extents(arm, flat));
    }
    std::fs::write(&out, bytes).unwrap();
    print_report(&report, 0.0);
    Ok(())
}

fn run_mhd(arm: &Arm, me: WorkerId, out: PathBuf, wall: &str) -> Result<(), WorkerError> {
    let s = session(arm, me);
    let kind = wall_type(wall);
    let mut built: Vec<MhdSim> = s.mine.iter().map(|&f| mhd_tile(arm, f, kind)).collect();
    let kernels: Vec<_> = built.iter().map(|sim| sim.substrate()).collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(&s.partition, &Topology::open(), &schema, built[0].geom.ng).expect("plan");
    let exchange = PlanExchange::new(&plan, &s.placement, me);
    let r = rendezvous(&s, arm, plan.digest, built[0].geom.ng);
    let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
    let (link, session_id) = connect(&r, max_payload)?;
    let mut fabric = exchange.fabric(link, session_id);
    let cfg = config(RegimeKind::of::<f64, 2, NewtonianMhd>());
    let mut stores: Vec<&mut FieldStore<2, 3, HostMemory, f64>> = built.iter_mut().map(|sim| &mut **sim).collect();
    let kernel_refs: Vec<_> = kernels.iter().collect();
    let report = evolve_worker(
        &s.tiles,
        &mut stores,
        &kernel_refs,
        &s.devices,
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )?;
    fabric.finish(TRANSFER_DEADLINE)?;
    drop(stores);
    let mut bytes = Vec::new();
    for (sim, &flat) in built.iter().zip(&s.mine) {
        let ext = extents(arm, flat);
        dump_density(&mut bytes, sim, ext);
        dump_faces(&mut bytes, sim, ext);
    }
    std::fs::write(&out, bytes).unwrap();
    let div = built.iter().map(tile_div).fold(0.0, f64::max);
    print_report(&report, div);
    Ok(())
}

fn worker_main() -> ! {
    let arm = Arm::from_env();
    let me = WorkerId(env_u64("W_ME").unwrap() as u32);
    let out = PathBuf::from(std::env::var("W_OUT").unwrap());
    let wall = std::env::var("W_WALL").unwrap();
    let magnetized = std::env::var("W_REGIME").is_ok_and(|v| v == "nmhd");
    let result = if magnetized { run_mhd(&arm, me, out, &wall) } else { run_gas(&arm, me, out, &wall) };
    match result {
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

#[derive(Debug, Clone, PartialEq)]
struct Report {
    steps: u64,
    exchanges: u64,
    neighbor: u64,
    troubled: u64,
    frozen: u64,
    audits: u64,
    div: f64,
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    records: Vec<(u64, usize, usize, u64)>,
}

fn parse_report(stdout: &str) -> Option<Report> {
    let line = stdout.lines().find(|l| l.contains("REPORT "))?;
    let line = &line[line.find("REPORT ")?..];
    let field = |name: &str| -> Option<String> {
        line.split_whitespace().find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
    };
    Some(Report {
        steps: field("steps")?.parse().ok()?,
        exchanges: field("exchanges")?.parse().ok()?,
        neighbor: field("neighbor")?.parse().ok()?,
        troubled: field("troubled")?.parse().ok()?,
        frozen: field("frozen")?.parse().ok()?,
        audits: field("audits")?.parse().ok()?,
        div: f64::from_bits(field("div")?.parse().ok()?),
    })
}

fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir = std::env::temp_dir().join(format!("symbi_wall_worker_{}_{test}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let mut children: Vec<(Child, PathBuf)> = (0..arm.workers())
        .map(|w| {
            let out = dir.join(format!("worker_{w}.rec"));
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(ROLE, "worker")
                .env("W_ME", w.to_string())
                .env("W_COORD", &coord)
                .env("W_OUT", &out)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for (k, v) in arm.env() {
                cmd.env(k, v);
            }
            for (k, v) in extra {
                cmd.env(k, v);
            }
            (cmd.spawn().unwrap(), out)
        })
        .collect();
    let start = Instant::now();
    while children.iter_mut().any(|(c, _)| c.try_wait().unwrap().is_none()) {
        if start.elapsed() > WATCHDOG {
            for (c, _) in children.iter_mut() {
                let _ = c.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let outcomes = children
        .iter_mut()
        .map(|(c, out)| {
            let status = c.wait().unwrap();
            let (mut stdout, mut stderr) = (String::new(), String::new());
            c.stdout.take().unwrap().read_to_string(&mut stdout).unwrap();
            c.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
            let records = std::fs::read(out)
                .map(|bytes| {
                    bytes
                        .chunks_exact(32)
                        .map(|r| {
                            let word = |k: usize| u64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
                            (word(0), word(1) as usize, word(2) as usize, word(3))
                        })
                        .collect()
                })
                .unwrap_or_default();
            Outcome { code: status.code(), stderr, report: parse_report(&stdout), records }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

/// the assembled grid of an arm: density per cell and, for the magnetized regime, each face
/// component per distinct global face, every face on a cut asserted equal in its two copies
struct Assembled {
    den: Vec<u64>,
    faces: Option<BTreeMap<(u64, usize, usize), u64>>,
}

fn assemble(outcomes: &[Outcome], label: &str, magnetized: bool) -> Assembled {
    let mut den = vec![None; N * N];
    let mut copies: BTreeMap<(u64, usize, usize), Vec<u64>> = BTreeMap::new();
    for o in outcomes {
        for &(kind, i, j, bits) in &o.records {
            match kind {
                0 => assert!(den[i * N + j].replace(bits).is_none(), "{label}: cell ({i},{j}) written twice"),
                d => copies.entry((d, i, j)).or_default().push(bits),
            }
        }
    }
    let den = den.into_iter().map(|c| c.expect("a cell no worker wrote")).collect();
    if !magnetized {
        assert!(copies.is_empty(), "{label}: faces recorded for a regime without a field");
        return Assembled { den, faces: None };
    }
    let mut shared = 0;
    let mut faces = BTreeMap::new();
    for ((d, i, j), values) in &copies {
        if values.len() > 1 {
            shared += 1;
            assert!(
                values.iter().all(|v| v == &values[0]),
                "{label}: the copies of face {d} at ({i},{j}) differ: {values:?}"
            );
        }
        faces.insert((*d, *i, *j), values[0]);
    }
    assert_eq!(faces.len(), 2 * N * (N + 1), "{label}: the assembled faces do not tile the box");
    assert_eq!(shared > 0, outcomes.len() > 1, "{label}: shared faces against the worker count");
    Assembled { den, faces: Some(faces) }
}

fn assembled_div(faces: &BTreeMap<(u64, usize, usize), u64>) -> f64 {
    let at = |d: usize, i: usize, j: usize| f64::from_bits(faces[&(1 + d as u64, i, j)]);
    let mut worst = 0.0f64;
    for i in 0..N {
        for j in 0..N {
            let div = (at(0, i + 1, j) - at(0, i, j)) / H + (at(1, i, j + 1) - at(1, i, j)) / H;
            worst = worst.max(div.abs());
        }
    }
    worst
}

fn differing(a: &Assembled, b: &Assembled) -> usize {
    let cells = a.den.iter().zip(&b.den).filter(|(x, y)| x != y).count();
    let faces = match (&a.faces, &b.faces) {
        (Some(fa), Some(fb)) => fa.iter().filter(|(k, v)| fb.get(k) != Some(v)).count(),
        _ => 0,
    };
    cells + faces
}

fn assert_all_ok(outcomes: &[Outcome], label: &str) {
    for (w, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "{label}: worker {w} failed:\n{}", o.stderr);
    }
}

fn uncut() -> Arm {
    Arm { cuts0: vec![], cuts1: vec![], owner: vec![0] }
}

fn two_workers() -> Arm {
    Arm { cuts0: vec![32], cuts1: vec![], owner: vec![0, 1] }
}

fn four_workers() -> Arm {
    Arm { cuts0: vec![32], cuts1: vec![32], owner: vec![0, 1, 2, 3] }
}

struct Case {
    arm: Arm,
    /// the troubled cell, where a cut meets a wall
    cell: (usize, usize),
    /// the worker holding the cell
    holder: usize,
    /// the workers that hold no troubled cell and must still correct
    neighbors: Vec<usize>,
    stage: usize,
}

/// the pulse against the low wall on the cut, two workers, the flag in the predictor
fn low_wall_two_workers() -> Case {
    Case { arm: two_workers(), cell: (31, 0), holder: 0, neighbors: vec![1], stage: 0 }
}

/// the pulse against the high wall on the cut, four workers, the flag in the corrector; the
/// tiles are numbered with the second axis fastest, so the cell's tile (1, 1) is worker 3 and
/// its neighbor across the cut, tile (0, 1), is worker 1
fn high_wall_four_workers() -> Case {
    Case { arm: four_workers(), cell: (32, 63), holder: 3, neighbors: vec![1], stage: 1 }
}

fn assert_cut_matches_uncut(test: &str, regime: &str, wall: &str, case: Case) {
    let magnetized = regime == "nmhd";
    let label = format!("{regime} {wall} wall, stage {}, {} workers", case.stage, case.arm.workers());
    let base = |w: &str| vec![("W_REGIME", regime.to_string()), ("W_WALL", w.to_string())];
    let mut inject = base(wall);
    inject.extend([
        ("W_TROUBLE_STAGE", case.stage.to_string()),
        ("W_TROUBLE_I", case.cell.0.to_string()),
        ("W_TROUBLE_J", case.cell.1.to_string()),
    ]);
    let clean = run_arm(test, "clean", &uncut(), &base(wall));
    assert_all_ok(&clean, "clean");
    let clean_report = clean[0].report.clone().expect("report");
    assert_eq!((clean_report.troubled, clean_report.exchanges), (0, 0), "{label}: the pulse corrected by itself");
    let clean_grid = assemble(&clean, "clean", magnetized);
    if wall != "outflow" {
        let open = run_arm(test, "open", &uncut(), &base("outflow"));
        assert_all_ok(&open, "open");
        let open_grid = assemble(&open, "open", magnetized);
        assert!(differing(&clean_grid, &open_grid) > 0, "{label}: the wall left the run equal to an outflow box");
    }
    let reference = run_arm(test, "uncut", &uncut(), &inject);
    assert_all_ok(&reference, "uncut");
    let want = assemble(&reference, "uncut", magnetized);
    assert!(differing(&want, &clean_grid) > 0, "{label}: the flag left the one-worker run unchanged; the gate is vacuous");
    let reference_report = reference[0].report.clone().expect("report");
    assert_eq!((reference_report.troubled, reference_report.frozen), (1, 0), "{label}: uncut census");
    if magnetized {
        assert!(reference_report.div < DIV_TOL, "{label}: one-worker div B {:e}", reference_report.div);
    }

    let outcomes = run_arm(test, "cut", &case.arm, &inject);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().expect("report");
        assert_eq!(r.steps, STEPS, "{label}: worker {w} steps");
        assert_eq!(r.frozen, 0, "{label}: worker {w} froze a cell");
        assert_eq!(r.exchanges, 1, "{label}: worker {w} flag exchanges");
        assert_eq!(r.troubled, u64::from(w == case.holder), "{label}: worker {w} troubled cells");
        assert_eq!(r.neighbor, u64::from(case.neighbors.contains(&w)), "{label}: worker {w} clean-tile corrections");
        // the audit covers the faces on a cut and follows every state exchange, prime plus two
        // stages per step; a regime without a field has no interface to audit
        let audits = if magnetized { 1 + 2 * STEPS } else { 0 };
        assert_eq!(r.audits, audits, "{label}: worker {w} audits");
        if magnetized {
            assert!(r.div < DIV_TOL, "{label}: worker {w} tile-local div B {:e}", r.div);
        }
    }
    let got = assemble(&outcomes, &label, magnetized);
    if let Some(faces) = &got.faces {
        assert!(assembled_div(faces) < DIV_TOL, "{label}: assembled div B {:e}", assembled_div(faces));
    }
    let n = differing(&got, &want);
    assert_eq!(n, 0, "{label}: {n} cells or faces differ from the one-worker run");
}

macro_rules! wall_gate {
    ($name:ident, $regime:literal, $wall:literal, $case:ident) => {
        #[test]
        fn $name() {
            if in_worker_role() {
                worker_main();
            }
            assert_cut_matches_uncut(stringify!($name), $regime, $wall, $case());
        }
    };
}

wall_gate!(hydro_reflect_wall_low_corner_two_workers, "newtonian", "reflect", low_wall_two_workers);
wall_gate!(hydro_reflect_wall_high_corner_four_workers, "newtonian", "reflect", high_wall_four_workers);
wall_gate!(hydro_outflow_wall_low_corner_two_workers, "newtonian", "outflow", low_wall_two_workers);
wall_gate!(hydro_outflow_wall_high_corner_four_workers, "newtonian", "outflow", high_wall_four_workers);
wall_gate!(hydro_neumann_wall_low_corner_two_workers, "newtonian", "neumann", low_wall_two_workers);
wall_gate!(hydro_neumann_wall_high_corner_four_workers, "newtonian", "neumann", high_wall_four_workers);
wall_gate!(hydro_robin_wall_low_corner_two_workers, "newtonian", "robin", low_wall_two_workers);
wall_gate!(hydro_robin_wall_high_corner_four_workers, "newtonian", "robin", high_wall_four_workers);
wall_gate!(mhd_reflect_wall_low_corner_two_workers, "nmhd", "reflect", low_wall_two_workers);
wall_gate!(mhd_reflect_wall_high_corner_four_workers, "nmhd", "reflect", high_wall_four_workers);
wall_gate!(mhd_outflow_wall_low_corner_two_workers, "nmhd", "outflow", low_wall_two_workers);
wall_gate!(mhd_outflow_wall_high_corner_four_workers, "nmhd", "outflow", high_wall_four_workers);
