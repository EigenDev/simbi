// =============================================================================
// worker_polar_processes.rs
//
// the polar seam across real processes: a spherical shell gridded over both
// poles with a periodic azimuth cut into tiles. a tile's ghost band beyond a
// pole is the interior half a turn away in azimuth, which lives on other
// tiles, so the plan carries antipodal transfers for it and the worker
// negates the band's odd components after every state exchange; a single
// tile fills the same band by its own half-turn. the azimuth is cut into
// two equal slabs and into three uneven ones whose half-turn images
// straddle tiles, with the polar axis cut as well on four workers; the
// first-order correction is fired on a cell at a pole against an azimuth
// cut, so its redo reads the antipodal band and the cut halo at once. every
// layout reproduces the one-worker run bit for bit, with no cell frozen and
// the flag shown to change the one-worker run.
//
// run: cargo test -p symbi --test worker_polar_processes
// =============================================================================

use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::prelude::SimSubstrate;
use std::f64::consts::PI;
use symbi::sim::decomp::{LocalCopy, Partition, PolarSeam, Topology, plan_schema, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Spherical;
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

type Sim = SimState<Newtonian, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Store = FieldStore<3, 3, HostMemory, f64>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
/// the shell: r in [1, 2], theta over both poles, phi around the full circle
const R_LO: f64 = 1.0;
const DR: f64 = 1.0 / N as f64;
const DTHETA: f64 = PI / N as f64;
const DPHI: f64 = 2.0 * PI / N as f64;
const GAMMA: f64 = 1.4;
const STEPS: u64 = 6;
const TROUBLE_STEP: u64 = 2;
const TRANSFER_DEADLINE: Duration = Duration::from_secs(8);
const STARTUP_DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(180);

#[derive(Clone)]
struct Arm {
    cuts: [Vec<usize>; 3],
    owner: Vec<u32>,
}

impl Arm {
    fn env(&self) -> Vec<(&'static str, String)> {
        let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        vec![
            ("W_CUTS0", list(&self.cuts[0])),
            ("W_CUTS1", list(&self.cuts[1])),
            ("W_CUTS2", list(&self.cuts[2])),
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
            cuts: [list("W_CUTS0"), list("W_CUTS1"), list("W_CUTS2")],
            owner: list("W_OWNER").into_iter().map(|x| x as u32).collect(),
        }
    }

    fn partition(&self) -> Partition<3> {
        Partition::explicit([N; 3], self.cuts.clone()).unwrap()
    }

    fn workers(&self) -> u32 {
        self.owner.iter().max().unwrap() + 1
    }
}

fn extents(arm: &Arm, flat: usize) -> [(usize, usize); 3] {
    let p = arm.partition();
    p.tile_extents(unflatten(flat, p.counts()))
}

/// the polar seam of an arm: present once the azimuth is cut, with one slab width per
/// azimuth tile
fn seam(arm: &Arm) -> Option<PolarSeam> {
    let p = arm.partition();
    let counts = p.counts();
    (counts[2] > 1).then(|| PolarSeam {
        mirror: 1,
        sides: [true, true],
        azimuth: 2,
        slabs: (0..counts[2]).map(|i| p.tile_range(2, i).1).collect(),
    })
}

/// a cut face is filled by the exchange. the radial ends flow out; a pole face is the axis
/// when the tile holds the whole azimuth and fills its own half-turn, and a cut filled by
/// the antipodal transfers once the azimuth is cut; the azimuth wraps on an uncut axis
fn tile(arm: &Arm, flat: usize) -> Sim {
    let p = arm.partition();
    let counts = p.counts();
    let tc = unflatten(flat, counts);
    let ext = p.tile_extents(tc);
    let pole = if counts[2] > 1 { BoundaryType::CoarseFine } else { BoundaryType::Axis };
    let at = |a: usize, side: usize, kind: BoundaryType| {
        let edge = if side == 0 { tc[a] == 0 } else { tc[a] + 1 == counts[a] };
        if edge { kind } else { BoundaryType::CoarseFine }
    };
    let phi = if counts[2] > 1 { BoundaryType::CoarseFine } else { BoundaryType::Periodic };
    let bnd = Boundaries([
        [at(0, 0, BoundaryType::Outflow), at(0, 1, BoundaryType::Outflow)],
        [at(1, 0, pole), at(1, 1, pole)],
        [phi, phi],
    ]);
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([ext[0].1, ext[1].1, ext[2].1])
        .origin([R_LO, 0.0, 0.0])
        .spacing([DR, DTHETA, DPHI])
        .index_offset(std::array::from_fn(|a| ext[a].0 as isize))
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[r, theta, phi]| {
            // a radial pulse shaded in latitude and longitude, with meridional and azimuthal
            // flows that vanish at the poles
            let b = 0.2 * (-((r - 1.5) / 0.08).powi(2)).exp();
            Prim::adiabatic(
                Density(1.0 + b * (1.0 + 0.5 * theta.cos()) * (1.0 + 0.3 * phi.cos())),
                Tensor::new([0.0, 0.05 * (2.0 * theta).sin(), 0.05 * theta.sin() * phi.cos()]),
                Pressure(1.0 + b),
            )
        })
        .build()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// (global i, j, k, bits) of the conserved density of every owned cell
fn dump_density(out: &mut Vec<u8>, sim: &Store, ext: [(usize, usize); 3]) {
    let lo: [isize; 3] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    let den = sim.fields.cons.den.view();
    for c in sim.geom.interior.iter() {
        for a in 0..3 {
            out.extend_from_slice(&((ext[a].0 as isize + c[a] - lo[a]) as u64).to_le_bytes());
        }
        out.extend_from_slice(&den.at(c).to_bits().to_le_bytes());
    }
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    let mine: Vec<usize> = (0..n).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    let mut built: Vec<Sim> = mine.iter().map(|&f| tile(arm, f)).collect();
    let kernels: Vec<_> = built.iter().map(|sim| sim.substrate()).collect();
    let tiles: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile_with_seam(
        &partition,
        &Topology::wrapping([false, false, true]),
        &schema,
        built[0].geom.ng,
        seam(arm),
    )
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
            credential: 0x9013,
            build_id: "worker_polar_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: PhaseTable::flat(2).unwrap().digest(),
            ng: built[0].geom.ng as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    };
    let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
    let (link, session) = connect(&r, max_payload)?;
    let mut fabric = exchange.fabric(link, session);
    let mut injection = Injection::default();
    if let Some(stage) = env_u64("W_TROUBLE_STAGE") {
        injection.trouble_at = Some((
            TROUBLE_STEP,
            stage as usize,
            [
                env_u64("W_TROUBLE_I").unwrap() as isize,
                env_u64("W_TROUBLE_J").unwrap() as isize,
                env_u64("W_TROUBLE_K").unwrap() as isize,
            ],
        ));
    }
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 3, Newtonian>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 10.0,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection,
        audit_every_point: true,
    };
    let mut stores: Vec<&mut Store> = built.iter_mut().map(|s| &mut **s).collect();
    let kernel_refs: Vec<_> = kernels.iter().collect();
    let report = evolve_worker(
        &tiles,
        &mut stores,
        &kernel_refs,
        &devices,
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )?;
    fabric.finish(TRANSFER_DEADLINE)?;
    drop(stores);
    let mut bytes = Vec::new();
    for (sim, &flat) in built.iter().zip(&mine) {
        dump_density(&mut bytes, sim, extents(arm, flat));
    }
    std::fs::write(&out, bytes).unwrap();
    println!(
        "REPORT steps={} exchanges={} neighbor={} troubled={} frozen={}",
        report.steps, report.troubled_exchanges, report.neighbor_corrections, report.troubled_cells, report.frozen_cells,
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

#[derive(Debug, Clone, PartialEq)]
struct Report {
    steps: u64,
    exchanges: u64,
    neighbor: u64,
    troubled: u64,
    frozen: u64,
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    density: Vec<([usize; 3], u64)>,
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
    })
}

fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir = std::env::temp_dir().join(format!("symbi_polar_worker_{}_{test}_{tag}", std::process::id()));
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
                        .chunks_exact(32)
                        .map(|r| {
                            let word = |k: usize| u64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
                            ([word(0) as usize, word(1) as usize, word(2) as usize], word(3))
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

fn global_density(outcomes: &[Outcome], label: &str) -> Vec<u64> {
    let mut grid = vec![None; N * N * N];
    for o in outcomes {
        for &(g, bits) in &o.density {
            let at = (g[0] * N + g[1]) * N + g[2];
            assert!(grid[at].replace(bits).is_none(), "{label}: cell {g:?} written twice");
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
    Arm { cuts: [vec![], vec![], vec![]], owner: vec![0] }
}

struct Case {
    arm: Arm,
    cell: [usize; 3],
    holder: usize,
    neighbors: Vec<usize>,
    stage: usize,
}

/// the azimuth cut into two equal slabs; the cell at the theta = 0 pole against the azimuth
/// cut, in the predictor: its band beyond the pole is the other slab's interior
fn azimuth_halves_two_workers() -> Case {
    Case {
        arm: Arm { cuts: [vec![], vec![], vec![16]], owner: vec![0, 1] },
        cell: [10, 0, 15],
        holder: 0,
        neighbors: vec![1],
        stage: 0,
    }
}

/// the azimuth cut into three uneven slabs [0, 12), [12, 20), [20, 32), whose half-turn images
/// straddle tiles; the cell at the theta = pi pole against the first cut, in the corrector: the
/// flag reaches the slab across the cut and, through the antipodal band, the slab holding the
/// half-turn image of the cell
fn azimuth_uneven_three_workers() -> Case {
    Case {
        arm: Arm { cuts: [vec![], vec![], vec![12, 20]], owner: vec![0, 1, 2] },
        cell: [10, 31, 11],
        holder: 0,
        neighbors: vec![1, 2],
        stage: 1,
    }
}

/// the polar axis and the azimuth both cut, four workers; the cell at the theta = 0 pole against
/// the azimuth cut, in the predictor; the tiles are numbered with the last axis fastest, so the
/// cell's tile (0, 0, 0) is worker 0 and the slab across the cut, tile (0, 0, 1), is worker 1
fn polar_and_azimuth_cuts_four_workers() -> Case {
    Case {
        arm: Arm { cuts: [vec![], vec![16], vec![16]], owner: vec![0, 1, 2, 3] },
        cell: [10, 0, 15],
        holder: 0,
        neighbors: vec![1],
        stage: 0,
    }
}

fn assert_cut_matches_uncut(test: &str, case: Case) {
    let label = format!("{} workers, {} tiles, stage {}", case.arm.workers(), case.arm.owner.len(), case.stage);
    let inject = vec![
        ("W_TROUBLE_STAGE", case.stage.to_string()),
        ("W_TROUBLE_I", case.cell[0].to_string()),
        ("W_TROUBLE_J", case.cell[1].to_string()),
        ("W_TROUBLE_K", case.cell[2].to_string()),
    ];
    let clean = run_arm(test, "clean", &uncut(), &[]);
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
        assert_eq!(r.neighbor > 0, case.neighbors.contains(&w), "{label}: worker {w} clean-tile corrections {}", r.neighbor);
    }
    let got = global_density(&outcomes, &label);
    let n = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(n, 0, "{label}: {n} cells differ from the one-worker run");
}

macro_rules! gate {
    ($name:ident, $case:ident) => {
        #[test]
        fn $name() {
            if in_worker_role() {
                worker_main();
            }
            assert_cut_matches_uncut(stringify!($name), $case());
        }
    };
}

gate!(azimuth_halves_on_two_workers_match_one_worker, azimuth_halves_two_workers);
gate!(uneven_azimuth_slabs_on_three_workers_match_one_worker, azimuth_uneven_three_workers);
gate!(polar_and_azimuth_cuts_on_four_workers_match_one_worker, polar_and_azimuth_cuts_four_workers);
