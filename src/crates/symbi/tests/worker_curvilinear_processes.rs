// =============================================================================
// worker_curvilinear_processes.rs
//
// the curvilinear charts across real processes. on a spherical (r, theta) or
// cylindrical (R, z) grid the wave speeds, the geometric source and the face
// areas vary with radius, so a tile is placed by its own radial origin and,
// on a log radial axis, by the global map advanced to its first cell; the
// halo exchange feeds each cut the neighbor's interior as on a cartesian
// grid. the spherical shell spans both poles on a log radial axis, its polar
// faces the coordinate axis, and the cylinder holds the R = 0 axis on its low
// radial face with reflecting ends in z. a smooth pulse launches waves across
// the cuts, and the first-order correction is fired against a cut and, on
// four workers, against the polar or reflecting end. two and four workers
// reproduce the one-worker run bit for bit, with no cell frozen and the flag
// shown to change the one-worker run.
//
// run: cargo test -p symbi --test worker_curvilinear_processes
// =============================================================================

use std::f64::consts::PI;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::prelude::SimSubstrate;
use symbi::sim::decomp::{LocalCopy, Partition, Topology, plan_schema, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::{AxisMap, Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type SphSim = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type CylSim = SimState<Newtonian, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
type Store = FieldStore<2, 2, HostMemory, f64>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 64;
const GAMMA: f64 = 1.4;
const STEPS: u64 = 8;
const TROUBLE_STEP: u64 = 3;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(8);
const STARTUP_DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(120);

/// the spherical shell: r in [1, 2] on a log axis, theta over both poles
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const DR: f64 = (R_HI - R_LO) / N as f64;
const DTHETA: f64 = PI / N as f64;
/// the cylinder: R in [0, 1] from the axis, z in [0, 1] between reflecting ends
const DRC: f64 = 1.0 / N as f64;
const DZ: f64 = 1.0 / N as f64;

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

fn extents(arm: &Arm, flat: usize) -> [(usize, usize); 2] {
    let p = arm.partition();
    p.tile_extents(unflatten(flat, p.counts()))
}

/// the tile's faces: a cut is filled by the exchange, a domain face takes the chart's own
/// condition on that side
fn boundaries(arm: &Arm, flat: usize, domain: [[BoundaryType; 2]; 2]) -> Boundaries<2> {
    let p = arm.partition();
    let counts = p.counts();
    let tc = unflatten(flat, counts);
    Boundaries(std::array::from_fn(|a| {
        [
            if tc[a] == 0 { domain[a][0] } else { BoundaryType::CoarseFine },
            if tc[a] == counts[a] - 1 { domain[a][1] } else { BoundaryType::CoarseFine },
        ]
    }))
}

/// the log radial slope spanning [R_LO, R_HI] over N cells: face(i) = R_LO * 10^(i * slope)
fn log_slope() -> f64 {
    (R_HI / R_LO).log10() / N as f64
}

fn pulse(r: f64) -> f64 {
    0.2 * (-((r - 1.5) / 0.08).powi(2)).exp()
}

/// a radial pulse shaded in latitude, with a meridional flow vanishing at the poles
fn sph_tile(arm: &Arm, flat: usize) -> SphSim {
    let ext = extents(arm, flat);
    // the whole shell's origin and maps, with the tile placed by the index of its first cell;
    // the radial axis uniform or log
    let maps = std::env::var("W_RADIAL").is_ok_and(|v| v == "log").then(|| {
        [AxisMap::Log { start: R_LO, log_slope: log_slope() }, AxisMap::Uniform { start: 0.0, dx: DTHETA }]
    });
    let domain = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
    SphSim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([ext[0].1, ext[1].1])
        .origin([R_LO, 0.0])
        .spacing([DR, DTHETA])
        .coord_maps(maps)
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
        .boundaries(boundaries(arm, flat, domain))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[r, theta]| {
            let b = pulse(r);
            Prim::adiabatic(
                Density(1.0 + b * (1.0 + 0.5 * theta.cos())),
                Tensor::new([0.0, 0.05 * (2.0 * theta).sin()]),
                Pressure(1.0 + b),
            )
        })
        .build()
}

/// a pulse off the axis in a flow that vanishes on the axis and turns at the ends
fn cyl_tile(arm: &Arm, flat: usize) -> CylSim {
    let ext = extents(arm, flat);
    let domain = [[BoundaryType::Axis, BoundaryType::Outflow], [BoundaryType::Reflect, BoundaryType::Reflect]];
    CylSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([ext[0].1, ext[1].1])
        .origin([0.0, 0.0])
        .spacing([DRC, DZ])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
        .boundaries(boundaries(arm, flat, domain))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[rr, z]| {
            let g = (-((rr - 0.4) * (rr - 0.4) + (z - 0.6) * (z - 0.6)) / 0.02).exp();
            Prim::adiabatic(
                Density(1.0 + 0.5 * g),
                Tensor::new([0.1 * rr * (1.0 - rr) * (PI * z).sin(), 0.1 * (0.5 * PI * z).cos() * (1.0 - rr)]),
                Pressure(1.0 + g),
            )
        })
        .build()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// (global i, global j, bits) of the conserved density of every owned cell
fn dump_density(out: &mut Vec<u8>, sim: &Store, ext: [(usize, usize); 2]) {
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    let den = sim.fields.cons.den.view();
    for c in sim.geom.interior.iter() {
        let g = [ext[0].0 + (c[0] - lo[0]) as usize, ext[1].0 + (c[1] - lo[1]) as usize];
        for w in [g[0] as u64, g[1] as u64, den.at(c).to_bits()] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
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
            credential: 0xC0A7,
            build_id: "worker_curvilinear_processes".into(),
            config_digest: Digest(1),
            plan_digest,
            placement_digest: s.placement.digest(),
            phase_table_digest: PhaseTable::flat(2).unwrap().digest(),
            ng: ng as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    }
}

fn config() -> WorkerConfig {
    let mut injection = Injection::default();
    if let Some(stage) = env_u64("W_TROUBLE_STAGE") {
        injection.trouble_at = Some((
            TROUBLE_STEP,
            stage as usize,
            [env_u64("W_TROUBLE_I").unwrap() as isize, env_u64("W_TROUBLE_J").unwrap() as isize, 0],
        ));
    }
    WorkerConfig {
        regime: RegimeKind::of::<f64, 2, Newtonian>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 10.0,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection,
        audit_every_point: true,
    }
}

macro_rules! chart_runner {
    ($name:ident, $sim:ty, $tile:ident) => {
        fn $name(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
            let s = session(arm, me);
            let mut built: Vec<$sim> = s.mine.iter().map(|&f| $tile(arm, f)).collect();
            let kernels: Vec<_> = built.iter().map(|sim| sim.substrate()).collect();
            let schema = plan_schema(&*built[0]);
            let plan = ExchangePlan::compile(&s.partition, &Topology::open(), &schema, built[0].geom.ng)
                .expect("plan");
            let exchange = PlanExchange::new(&plan, &s.placement, me, PhaseTable::flat(2).unwrap());
            let r = rendezvous(&s, arm, plan.digest, built[0].geom.ng);
            let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
            let (link, session_id) = connect(&r, max_payload)?;
            let mut fabric = exchange.fabric(link, session_id);
            let cfg = config();
            let mut stores: Vec<&mut Store> = built.iter_mut().map(|sim| &mut **sim).collect();
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
            println!(
                "REPORT steps={} exchanges={} neighbor={} troubled={} frozen={} audits={}",
                report.steps,
                report.troubled_exchanges,
                report.neighbor_corrections,
                report.troubled_cells,
                report.frozen_cells,
                report.audits,
            );
            Ok(())
        }
    };
}

chart_runner!(run_sph, SphSim, sph_tile);
chart_runner!(run_cyl, CylSim, cyl_tile);

fn worker_main() -> ! {
    let arm = Arm::from_env();
    let me = WorkerId(env_u64("W_ME").unwrap() as u32);
    let out = PathBuf::from(std::env::var("W_OUT").unwrap());
    let result = match std::env::var("W_CHART").unwrap().as_str() {
        "spherical" => run_sph(&arm, me, out),
        "cylindrical" => run_cyl(&arm, me, out),
        other => panic!("unknown chart {other}"),
    };
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
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    density: Vec<(usize, usize, u64)>,
}

fn parse_report(stdout: &str) -> Option<Report> {
    let line = stdout.lines().find(|l| l.contains("REPORT "))?;
    let line = &line[line.find("REPORT ")?..];
    let field = |name: &str| -> Option<u64> {
        line.split_whitespace().find_map(|kv| kv.strip_prefix(&format!("{name}=")).and_then(|v| v.parse().ok()))
    };
    Some(Report {
        steps: field("steps")?,
        exchanges: field("exchanges")?,
        neighbor: field("neighbor")?,
        troubled: field("troubled")?,
        frozen: field("frozen")?,
        audits: field("audits")?,
    })
}

fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir = std::env::temp_dir().join(format!("symbi_curv_worker_{}_{test}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let mut children: Vec<(Child, PathBuf)> = (0..arm.workers())
        .map(|w| {
            let out = dir.join(format!("worker_{w}.den"));
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
            let density = std::fs::read(out)
                .map(|bytes| {
                    bytes
                        .chunks_exact(24)
                        .map(|r| {
                            let word = |k: usize| u64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
                            (word(0) as usize, word(1) as usize, word(2))
                        })
                        .collect()
                })
                .unwrap_or_default();
            Outcome { code: status.code(), stderr, report: parse_report(&stdout), density }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

/// the conserved density of an arm on the global grid, every cell written exactly once
fn global_density(outcomes: &[Outcome], label: &str) -> Vec<u64> {
    let mut grid = vec![None; N * N];
    for o in outcomes {
        for &(i, j, bits) in &o.density {
            assert!(grid[i * N + j].replace(bits).is_none(), "{label}: cell ({i},{j}) written twice");
        }
    }
    grid.into_iter().map(|c| c.expect("a cell no worker wrote")).collect()
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
    cell: (usize, usize),
    holder: usize,
    neighbors: Vec<usize>,
    stage: usize,
}

/// the flag against the radial cut, two workers, in the predictor
fn radial_cut_two_workers() -> Case {
    Case { arm: two_workers(), cell: (31, 10), holder: 0, neighbors: vec![1], stage: 0 }
}

/// the flag against the radial cut and the high transverse end, four workers, in the
/// corrector; the tiles are numbered with the second axis fastest, so the cell's tile (1, 1)
/// is worker 3 and its neighbor across the radial cut, tile (0, 1), is worker 1
fn radial_cut_at_the_end_four_workers() -> Case {
    Case { arm: four_workers(), cell: (32, 63), holder: 3, neighbors: vec![1], stage: 1 }
}

fn assert_cut_matches_uncut(test: &str, chart: &str, radial: &str, case: Case) {
    let label = format!("{chart} {radial}, stage {}, {} workers", case.stage, case.arm.workers());
    let base = vec![("W_CHART", chart.to_string()), ("W_RADIAL", radial.to_string())];
    let mut inject = base.clone();
    inject.extend([
        ("W_TROUBLE_STAGE", case.stage.to_string()),
        ("W_TROUBLE_I", case.cell.0.to_string()),
        ("W_TROUBLE_J", case.cell.1.to_string()),
    ]);
    let clean = run_arm(test, "clean", &uncut(), &base);
    assert_all_ok(&clean, "clean");
    let clean_report = clean[0].report.clone().expect("report");
    assert_eq!((clean_report.troubled, clean_report.exchanges), (0, 0), "{label}: the pulse corrected by itself");
    let clean_grid = global_density(&clean, "clean");
    let reference = run_arm(test, "uncut", &uncut(), &inject);
    assert_all_ok(&reference, "uncut");
    let want = global_density(&reference, "uncut");
    assert_ne!(want, clean_grid, "{label}: the flag left the one-worker run unchanged; the gate is vacuous");
    let reference_report = reference[0].report.clone().expect("report");
    assert_eq!((reference_report.troubled, reference_report.frozen), (1, 0), "{label}: uncut census");

    let outcomes = run_arm(test, "cut", &case.arm, &inject);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().expect("report");
        assert_eq!(r.steps, STEPS, "{label}: worker {w} steps");
        assert_eq!(r.frozen, 0, "{label}: worker {w} froze a cell");
        assert_eq!(r.exchanges, 1, "{label}: worker {w} flag exchanges");
        assert_eq!(r.troubled, u64::from(w == case.holder), "{label}: worker {w} troubled cells");
        assert_eq!(r.neighbor, u64::from(case.neighbors.contains(&w)), "{label}: worker {w} clean-tile corrections");
        assert_eq!(r.audits, 0, "{label}: worker {w} audits; a regime without a field has no interface to audit");
    }
    let got = global_density(&outcomes, &label);
    let n = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| ((f64::from_bits(*a) - f64::from_bits(*b)) / f64::from_bits(*b)).abs())
        .fold(0.0, f64::max);
    assert_eq!(n, 0, "{label}: {n} cells differ from the one-worker run, the worst by {worst:e} relative");
}

macro_rules! chart_gate {
    ($name:ident, $chart:literal, $radial:literal, $case:ident) => {
        #[test]
        fn $name() {
            if in_worker_role() {
                worker_main();
            }
            assert_cut_matches_uncut(stringify!($name), $chart, $radial, $case());
        }
    };
}

chart_gate!(spherical_shell_radial_cut_two_workers, "spherical", "uniform", radial_cut_two_workers);
chart_gate!(spherical_shell_cut_at_the_pole_four_workers, "spherical", "uniform", radial_cut_at_the_end_four_workers);
chart_gate!(spherical_log_shell_radial_cut_two_workers, "spherical", "log", radial_cut_two_workers);
chart_gate!(spherical_log_shell_cut_at_the_pole_four_workers, "spherical", "log", radial_cut_at_the_end_four_workers);
chart_gate!(cylindrical_axis_radial_cut_two_workers, "cylindrical", "uniform", radial_cut_two_workers);
chart_gate!(cylindrical_axis_cut_at_the_end_four_workers, "cylindrical", "uniform", radial_cut_at_the_end_four_workers);
