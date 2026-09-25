// =============================================================================
// worker_refined_processes.rs
//
// the refined hydro transaction across worker processes: a root grid with one
// refined patch of ratio 2 spanning tile cuts, each worker holding its root
// tiles and the fine tiles under them, advanced through the fabric by
// `evolve_refined_worker`, against the single hierarchy driven with the same
// timesteps, interior cell for interior cell over the composite (the uncovered
// root and the fine level). arms: two ragged layouts with cuts off the patch's
// center over several root steps; a layout where two of three workers hold no
// fine tile; a rejection forced in the second fine substep on one worker
// against a clean run capped to the accepted smaller timestep. every worker
// reports its grant high-water per peer.
//
// run: cargo test -p symbi --test worker_refined_processes
// =============================================================================
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi::sim::decomp::{FinePartition, LocalCopy, Partition, Topology, flatten, plan_cons_schema, plan_schema, unflatten};
use symbi::sim::refinement::launched::{Held, RefinedConfig, RefinedPlans, evolve_refined_worker};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, Fnv, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::{PlanExchange, PlanRole, fabric_for};
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const NG: usize = 2;
const RATIO: usize = 2;
const STEPS: u64 = 8;
const T_FINAL: f64 = 10.0;
// the refined patch: root cells 12..20 on both axes
const PATCH: [(usize, usize); 2] = [(12, 20), (12, 20)];
const TRANSFER_DEADLINE: Duration = Duration::from_secs(10);
const STARTUP_DEADLINE: Duration = Duration::from_secs(10);
const WATCHDOG: Duration = Duration::from_secs(120);

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

fn build_root(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .ghosts(NG)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build()
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

fn patch_region() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [PATCH[0].0 as f64 * DX, PATCH[1].0 as f64 * DX],
        x_hi: [PATCH[0].1 as f64 * DX, PATCH[1].1 as f64 * DX],
    }
}

/// the single hierarchy: the whole root with the whole patch
fn build_mono() -> Hier {
    let root = build_root([N, N], [0.0, 0.0], Boundaries::uniform(BoundaryType::Outflow));
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[patch_region()], ProlongOrder::Ppm, kset).expect("mono hier");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

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

/// one root tile's hierarchy: the patch clipped to the tile with the whole patch beside it
fn tile(partition: &Partition<2>, flat: usize) -> Hier {
    let counts = partition.counts();
    let tc = unflatten(flat, counts);
    let ext = partition.tile_extents(tc);
    let m: [usize; 2] = [ext[0].1, ext[1].1];
    let origin: [f64; 2] = std::array::from_fn(|a| ext[a].0 as f64 * DX);
    let hi: [f64; 2] = std::array::from_fn(|a| origin[a] + m[a] as f64 * DX);
    let bnd = Boundaries(std::array::from_fn(|a| {
        let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
        [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
    }));
    let root = build_root(m, origin, bnd);
    let whole = patch_region();
    let cx = [whole.x_lo[0].max(origin[0]), whole.x_hi[0].min(hi[0])];
    let cy = [whole.x_lo[1].max(origin[1]), whole.x_hi[1].min(hi[1])];
    let k = kset(&root);
    if cx[0] < cx[1] && cy[0] < cy[1] {
        let clipped = RefinementRegion { x_lo: [cx[0], cy[0]], x_hi: [cx[1], cy[1]] };
        Hier::with_refinement_of(root, k, &[clipped], &[whole], ProlongOrder::Ppm, kset).expect("tile hier")
    } else {
        Hier::single(root, k)
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn injection_from_env(me: WorkerId) -> Injection {
    let mut injection = Injection::default();
    if let Some(worker) = env_u64("W_REJECT_ROOT_WORKER") {
        if worker == u64::from(me.0) {
            let step = env_u64("W_REJECT_ROOT_STEP").unwrap();
            let stage = env_u64("W_REJECT_ROOT_STAGE").unwrap() as usize;
            injection.reject_at = vec![(step, stage)];
        }
    }
    if let Some(worker) = env_u64("W_CAP_WORKER") {
        if worker == u64::from(me.0) || worker == u64::MAX {
            let step = env_u64("W_CAP_STEP").unwrap();
            let bits = env_u64("W_CAP_BITS").unwrap();
            injection.cap_dt_at = Some((step, bits));
        }
    }
    injection
}

fn reject_fine_from_env(me: WorkerId) -> Option<(u64, u8, usize)> {
    let worker = env_u64("W_REJECT_FINE_WORKER")?;
    if worker != u64::from(me.0) {
        return None;
    }
    let spec = std::env::var("W_REJECT_FINE").ok()?;
    let parts: Vec<u64> = spec.split(',').map(|s| s.parse().unwrap()).collect();
    Some((parts[0], parts[1] as u8, parts[2] as usize))
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let counts = partition.counts();
    let n_root = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner.clone()).expect("placement");
    let fine_partition = FinePartition::of(&partition, PATCH, RATIO).expect("fine partition");
    let fine_counts = fine_partition.partition.counts();
    let n_fine = fine_partition.partition.n_tiles();
    let fine_owner: Vec<WorkerId> = (0..n_fine)
        .map(|f| owner[flatten(fine_partition.root_tile_of(unflatten(f, fine_counts)), counts)])
        .collect();
    let fine_placement = Placement::with_idle_workers(arm.workers(), fine_owner).expect("fine placement");
    let mine: Vec<usize> = (0..n_root).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    let mut tiles: Vec<Hier> = mine.iter().map(|&f| tile(&partition, f)).collect();
    let root_ids: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let fine_ids: Vec<Option<TileId>> = mine
        .iter()
        .map(|&f| fine_partition.fine_tile_of(unflatten(f, counts)).map(|tc| TileId(flatten(tc, fine_counts) as u32)))
        .collect();
    let root_store = &*tiles[0].levels[0].state;
    let schema = plan_schema(root_store);
    let cons_schema = plan_cons_schema(root_store);
    let root_plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, NG).expect("root plan");
    let seed_plan = ExchangePlan::compile(&partition, &Topology::open(), &cons_schema, NG).expect("seed plan");
    // a fine store indexes its cells in its parent's frame, whose interior starts at zero:
    // its first interior cell sits at the ratio times the patch's offset inside the tile
    let fine_lo = |t: TileId| -> [isize; 2] {
        let tc = unflatten(t.0 as usize, fine_counts);
        let root_ext = partition.tile_extents(fine_partition.root_tile_of(tc));
        std::array::from_fn(|a| (RATIO * PATCH[a].0.saturating_sub(root_ext[a].0)) as isize)
    };
    let fine_plan = ExchangePlan::compile_shifted(&fine_partition.partition, &Topology::open(), &schema, NG + 1, &fine_lo)
        .expect("fine plan");
    let table = PhaseTable::refined(2, RATIO as u8, 2).expect("table");
    let root = PlanExchange::new(&root_plan, &placement, me, table.clone());
    let seed = PlanExchange::new(&seed_plan, &placement, me, table.clone())
        .with_role(PlanRole::RootSeed, root.lens().len() as u32);
    let fine = PlanExchange::new(&fine_plan, &fine_placement, me, table.clone())
        .with_role(PlanRole::Fine, (root.lens().len() + seed.lens().len()) as u32);
    let plan_digest = {
        let mut h = Fnv::new();
        for d in [root_plan.digest, seed_plan.digest, fine_plan.digest] {
            h.u32((d.0 >> 32) as u32);
            h.u32(d.0 as u32);
        }
        h.finish()
    };
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0xBEEF,
            build_id: "worker_refined_processes".into(),
            config_digest: Digest(1),
            plan_digest,
            placement_digest: placement.digest(),
            phase_table_digest: table.digest(),
            ng: NG as u32,
            workers: arm.workers(),
            block_credit: 4096,
        },
        startup: STARTUP_DEADLINE,
    };
    let max_payload = [&root, &seed, &fine]
        .iter()
        .flat_map(|p| p.lens())
        .max()
        .unwrap_or(0)
        * 8;
    let (link, session) = connect(&r, max_payload.max(64))?;
    let mut fabric = fabric_for(&[&root, &seed, &fine], link, session);
    let plans = RefinedPlans { root, seed, fine };
    let cfg = RefinedConfig {
        base: WorkerConfig {
            regime: RegimeKind::of::<f64, 2, Newtonian>(),
            timestepping: Timestepping::Rk2,
            start_time: 0.0,
            t_final: T_FINAL,
            max_steps: STEPS,
            deadline: TRANSFER_DEADLINE,
            injection: injection_from_env(me),
            audit_every_point: false,
        },
        reject_fine_at: reject_fine_from_env(me),
    };
    let root_devices = vec![0i32; n_root];
    let fine_devices = vec![0i32; n_fine];
    let held = Held {
        root_ids: &root_ids,
        fine_ids: &fine_ids,
        n_root,
        n_fine,
        root_devices: &root_devices,
        fine_devices: &fine_devices,
    };
    let report = evolve_refined_worker(&mut tiles, &held, &plans, &LocalCopy, &mut fabric, &cfg, |_, _, _| {
        Ok(ControlFlow::Continue(()))
    })?;
    fabric.finish(TRANSFER_DEADLINE)?;
    // the composite this worker holds: every uncovered root interior cell as its 2x2 fine
    // block and every fine interior cell, by global fine index, conserved density bits
    let mut cells: Vec<u8> = Vec::new();
    let mut put = |i: usize, j: usize, bits: u64| {
        cells.extend_from_slice(&(i as u64).to_le_bytes());
        cells.extend_from_slice(&(j as u64).to_le_bytes());
        cells.extend_from_slice(&bits.to_le_bytes());
    };
    for (&flat, h) in mine.iter().zip(&tiles) {
        let ext = partition.tile_extents(unflatten(flat, counts));
        let root = &h.levels[0].state;
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        let cov = h.levels[0].coverage.as_ref();
        let den = root.fields.cons.den.view();
        for c in root.geom.interior.iter() {
            if cov.is_some_and(|cov| cov.contains(c)) {
                continue;
            }
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - rlo[a]) as usize);
            let bits = den.at(c).to_bits();
            for sy in 0..RATIO {
                for sx in 0..RATIO {
                    put(RATIO * g[0] + sx, RATIO * g[1] + sy, bits);
                }
            }
        }
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
            let cov = cov.unwrap();
            let clo: [isize; 2] = std::array::from_fn(|a| cov.spaces[a].lo);
            let den = fine.fields.cons.den.view();
            for c in fine.geom.interior.iter() {
                let gf: [usize; 2] = std::array::from_fn(|a| {
                    let cov_global_root = ext[a].0 + (clo[a] - rlo[a]) as usize;
                    cov_global_root * RATIO + (c[a] - flo[a]) as usize
                });
                put(gf[0], gf[1], den.at(c).to_bits());
            }
        }
    }
    std::fs::write(&out, cells).unwrap();
    // every cut halo cell this worker holds at the end of the run, by level and global index
    // on that level, primitive density bits: the exchange moves primitives, so the root
    // halos stand as the post-restriction exchange left them and the fine halos as the last
    // post-tail exchange left them
    let mut halos: Vec<u8> = Vec::new();
    let mut put_halo = |level: u64, i: isize, j: isize, bits: u64| {
        halos.extend_from_slice(&level.to_le_bytes());
        halos.extend_from_slice(&(i as u64).to_le_bytes());
        halos.extend_from_slice(&(j as u64).to_le_bytes());
        halos.extend_from_slice(&bits.to_le_bytes());
    };
    let inside = |g: [isize; 2], lo: [isize; 2], hi: [isize; 2]| (0..2).all(|a| lo[a] <= g[a] && g[a] < hi[a]);
    for (&flat, h) in mine.iter().zip(&tiles) {
        let ext = partition.tile_extents(unflatten(flat, counts));
        let root = &h.levels[0].state;
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        let den = root.fields.prim.rho.view();
        for c in root.geom.allocated.iter() {
            if root.geom.interior.contains(c) {
                continue;
            }
            let g: [isize; 2] = std::array::from_fn(|a| ext[a].0 as isize + c[a] - rlo[a]);
            if inside(g, [0, 0], [N as isize, N as isize]) {
                put_halo(0, g[0], g[1], den.at(c).to_bits());
            }
        }
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
            let clo: [isize; 2] = std::array::from_fn(|a| h.levels[0].coverage.as_ref().unwrap().spaces[a].lo);
            let den = fine.fields.prim.rho.view();
            let patch_lo: [isize; 2] = std::array::from_fn(|a| (PATCH[a].0 * RATIO) as isize);
            let patch_hi: [isize; 2] = std::array::from_fn(|a| (PATCH[a].1 * RATIO) as isize);
            for c in fine.geom.allocated.iter() {
                if fine.geom.interior.contains(c) {
                    continue;
                }
                let gf: [isize; 2] =
                    std::array::from_fn(|a| (ext[a].0 as isize + clo[a] - rlo[a]) * RATIO as isize + c[a] - flo[a]);
                if inside(gf, patch_lo, patch_hi) {
                    put_halo(1, gf[0], gf[1], den.at(c).to_bits());
                }
            }
        }
    }
    std::fs::write(out.with_extension("halos"), halos).unwrap();
    println!(
        "REPORT steps={} rejections={} time={} fine_troubled={} hw={} dts={}",
        report.steps,
        report.rejections,
        report.time.to_bits(),
        report.fine_troubled_exchanges,
        report.grant_high_water.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","),
        report.dt_sequence.iter().map(|d| d.to_bits().to_string()).collect::<Vec<_>>().join(",")
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
    out: PathBuf,
}

fn spawn(test: &str, me: u32, arm: &Arm, coord: &std::path::Path, dir: &std::path::Path, extra: &[(&str, String)]) -> Spawned {
    let out = dir.join(format!("worker_{me}.cells"));
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
    let child = cmd.spawn().unwrap();
    Spawned { child, out }
}

#[derive(Debug, Clone, PartialEq)]
struct Report {
    steps: u64,
    rejections: u64,
    time: u64,
    fine_troubled: u64,
    high_water: Vec<usize>,
    dts: Vec<u64>,
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    /// (global fine i, global fine j, conserved density bits) of every composite cell held
    cells: Vec<(usize, usize, u64)>,
    /// (level, global i, global j, primitive density bits) of every cut halo cell held
    halos: Vec<(u8, usize, usize, u64)>,
}

fn parse_report(stdout: &str) -> Option<Report> {
    let line = stdout.lines().find(|l| l.contains("REPORT "))?;
    let line = &line[line.find("REPORT ")?..];
    let field = |name: &str| -> Option<String> {
        line.split_whitespace().find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
    };
    let list = |name: &str| -> Option<Vec<u64>> {
        Some(field(name)?.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect())
    };
    Some(Report {
        steps: field("steps")?.parse().ok()?,
        rejections: field("rejections")?.parse().ok()?,
        time: field("time")?.parse().ok()?,
        fine_troubled: field("fine_troubled")?.parse().ok()?,
        high_water: list("hw")?.into_iter().map(|h| h as usize).collect(),
        dts: list("dts")?,
    })
}

fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir = std::env::temp_dir().join(format!("symbi_refined_worker_{}_{test}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let mut spawned: Vec<Spawned> = (0..arm.workers()).map(|w| spawn(test, w, arm, &coord, &dir, extra)).collect();
    let start = Instant::now();
    loop {
        if spawned.iter_mut().all(|s| s.child.try_wait().unwrap().is_some()) {
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
        .into_iter()
        .map(|mut s| {
            let status = s.child.wait().unwrap();
            let mut stdout = String::new();
            s.child.stdout.take().unwrap().read_to_string(&mut stdout).unwrap();
            let mut stderr = String::new();
            s.child.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
            let cells = std::fs::read(&s.out)
                .map(|bytes| {
                    bytes
                        .chunks_exact(24)
                        .map(|c| {
                            let u = |k: usize| u64::from_le_bytes(c[8 * k..8 * k + 8].try_into().unwrap());
                            (u(0) as usize, u(1) as usize, u(2))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let halos = std::fs::read(s.out.with_extension("halos"))
                .map(|bytes| {
                    bytes
                        .chunks_exact(32)
                        .map(|c| {
                            let u = |k: usize| u64::from_le_bytes(c[8 * k..8 * k + 8].try_into().unwrap());
                            (u(0) as u8, u(1) as usize, u(2) as usize, u(3))
                        })
                        .collect()
                })
                .unwrap_or_default();
            Outcome { code: status.code(), stderr, report: parse_report(&stdout), cells, halos }
        })
        .collect()
}

fn assert_all_ok(outcomes: &[Outcome], label: &str) {
    for (w, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "{label}: worker {w} failed:\n{}", o.stderr);
    }
}

/// the composite of every worker's cells on the 2N x 2N fine grid; every cell written once
fn composite(outcomes: &[Outcome]) -> Vec<u64> {
    let fn_n = RATIO * N;
    let mut out = vec![u64::MAX; fn_n * fn_n];
    let mut seen = vec![false; fn_n * fn_n];
    for o in outcomes {
        for &(i, j, bits) in &o.cells {
            let at = j * fn_n + i;
            assert!(!seen[at], "composite cell ({i}, {j}) held by two workers");
            seen[at] = true;
            out[at] = bits;
        }
    }
    assert!(seen.iter().all(|&s| s), "a composite cell was never written");
    out
}

/// the single hierarchy driven with the launched run's timesteps
struct Reference {
    /// the same composite on the 2N x 2N fine grid
    composite: Vec<u64>,
    /// primitive density of every root interior cell on the N x N root grid, the covered
    /// ones restricted
    root_rho: Vec<u64>,
    /// primitive density of every fine interior cell on the 2N x 2N fine grid
    fine_rho: Vec<u64>,
}

fn reference(dts: &[u64]) -> Reference {
    let mut mono = build_mono();
    for &bits in dts {
        mono.step_root_with_dt(f64::from_bits(bits));
    }
    let fn_n = RATIO * N;
    let mut composite = vec![u64::MAX; fn_n * fn_n];
    let mut root_rho = vec![u64::MAX; N * N];
    let mut fine_rho = vec![u64::MAX; fn_n * fn_n];
    let root = &mono.levels[0].state;
    let cov = mono.levels[0].coverage.as_ref().unwrap();
    let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
    let den = root.fields.cons.den.view();
    let rho = root.fields.prim.rho.view();
    for c in root.geom.interior.iter() {
        let g: [usize; 2] = std::array::from_fn(|a| (c[a] - rlo[a]) as usize);
        root_rho[g[1] * N + g[0]] = rho.at(c).to_bits();
        if cov.contains(c) {
            continue;
        }
        for sy in 0..RATIO {
            for sx in 0..RATIO {
                composite[(RATIO * g[1] + sy) * fn_n + RATIO * g[0] + sx] = den.at(c).to_bits();
            }
        }
    }
    let fine = &mono.levels[1].state;
    let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
    let fden = fine.fields.cons.den.view();
    let frho = fine.fields.prim.rho.view();
    for c in fine.geom.interior.iter() {
        let gf: [usize; 2] = std::array::from_fn(|a| PATCH[a].0 * RATIO + (c[a] - flo[a]) as usize);
        composite[gf[1] * fn_n + gf[0]] = fden.at(c).to_bits();
        fine_rho[gf[1] * fn_n + gf[0]] = frho.at(c).to_bits();
    }
    Reference { composite, root_rho, fine_rho }
}

/// every cut halo a worker holds equals the single hierarchy's value at that cell: the root
/// halos as the post-restriction exchange left them, the fine halos as the last post-tail
/// exchange left them
fn assert_halos_fresh(outcomes: &[Outcome], expected: &Reference, label: &str) {
    let fn_n = RATIO * N;
    let (mut root_halos, mut fine_halos, mut stale) = (0usize, 0usize, 0usize);
    for (w, o) in outcomes.iter().enumerate() {
        for &(level, i, j, bits) in &o.halos {
            let want = match level {
                0 => expected.root_rho[j * N + i],
                _ => expected.fine_rho[j * fn_n + i],
            };
            if level == 0 {
                root_halos += 1;
            } else {
                fine_halos += 1;
            }
            if want != bits {
                stale += 1;
                eprintln!("{label}: worker {w} level {level} halo ({i}, {j}) is stale");
            }
        }
    }
    assert!(root_halos > 0 && fine_halos > 0, "{label}: no cut halo on both levels; the freshness gate is vacuous");
    assert_eq!(stale, 0, "{label}: {stale} of {} halo cells are stale", root_halos + fine_halos);
}

fn assert_matches_reference(outcomes: &[Outcome], label: &str) {
    assert_all_ok(outcomes, label);
    let reports: Vec<&Report> = outcomes.iter().map(|o| o.report.as_ref().expect("report")).collect();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.dts, reports[0].dts, "{label}: worker {w} took other timesteps");
        assert_eq!(r.steps, STEPS, "{label}: worker {w} took {} steps", r.steps);
        assert!(
            r.high_water.iter().all(|&h| h <= 2),
            "{label}: worker {w} held more grants than one per axis per peer: {:?}",
            r.high_water
        );
        eprintln!("{label}: worker {w} grant high-water {:?}", r.high_water);
    }
    let launched = composite(outcomes);
    let expected = reference(&reports[0].dts);
    let differing = launched.iter().zip(&expected.composite).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{label}: {differing} composite cells differ from the single hierarchy");
    assert_halos_fresh(outcomes, &expected, label);
}

fn ragged_17_13() -> Arm {
    Arm { cuts0: vec![17], cuts1: vec![13], owner: vec![0, 1, 2, 3] }
}

fn ragged_18_14() -> Arm {
    Arm { cuts0: vec![18], cuts1: vec![14], owner: vec![0, 1, 2, 3] }
}

/// the patch lies under the middle column of tiles and spans the y cut; the outer columns'
/// workers hold no fine tile
fn no_fine_on_two_workers() -> Arm {
    Arm { cuts0: vec![10, 22], cuts1: vec![16], owner: vec![0, 0, 1, 1, 2, 2] }
}

#[test]
fn a_ragged_layout_matches_the_single_hierarchy_over_several_root_steps() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_ragged_layout_matches_the_single_hierarchy_over_several_root_steps";
    assert_matches_reference(&run_arm(test, "17_13", &ragged_17_13(), &[]), "cuts 17,13");
    assert_matches_reference(&run_arm(test, "18_14", &ragged_18_14(), &[]), "cuts 18,14");
}

#[test]
fn workers_without_fine_tiles_walk_the_transaction_with_the_others() {
    if in_worker_role() {
        worker_main();
    }
    let test = "workers_without_fine_tiles_walk_the_transaction_with_the_others";
    let outcomes = run_arm(test, "no_fine", &no_fine_on_two_workers(), &[]);
    assert_matches_reference(&outcomes, "two of three workers without fine tiles");
}

/// where two composites differ: the cells inside the fine patch and outside it
fn diff_report(a: &[u64], b: &[u64]) -> String {
    let fn_n = RATIO * N;
    let (mut fine, mut root) = (0usize, 0usize);
    let mut first: Option<(usize, usize)> = None;
    for j in 0..fn_n {
        for i in 0..fn_n {
            if a[j * fn_n + i] != b[j * fn_n + i] {
                let inside = (RATIO * PATCH[0].0..RATIO * PATCH[0].1).contains(&i) && (RATIO * PATCH[1].0..RATIO * PATCH[1].1).contains(&j);
                if inside { fine += 1 } else { root += 1 }
                first.get_or_insert((i, j));
            }
        }
    }
    format!("{fine} fine cells and {root} root cells differ, first at {first:?}")
}

/// the run whose timestep at `step` is capped to `bits`, on every worker
fn capped(test: &str, tag: &str, arm: &Arm, step: u64, bits: u64) -> Vec<Outcome> {
    run_arm(
        test,
        tag,
        arm,
        &[
            ("W_CAP_WORKER", u64::MAX.to_string()),
            ("W_CAP_STEP", step.to_string()),
            ("W_CAP_BITS", bits.to_string()),
        ],
    )
}

/// a rejection forced on worker 1 at root stage 1 of step 3: the control for the fine
/// rollback, on the path the flat gates already hold
#[test]
fn a_rejection_at_a_root_stage_rolls_every_worker_back() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_rejection_at_a_root_stage_rolls_every_worker_back";
    let arm = ragged_17_13();
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[
            ("W_REJECT_ROOT_WORKER", "1".into()),
            ("W_REJECT_ROOT_STEP", "3".into()),
            ("W_REJECT_ROOT_STAGE", "1".into()),
        ],
    );
    assert_all_ok(&rejecting, "the root-rejecting run");
    let reports: Vec<&Report> = rejecting.iter().map(|o| o.report.as_ref().unwrap()).collect();
    assert!(reports.iter().all(|r| r.rejections == 1), "every worker rejects once");
    let clean = capped(test, "clean", &arm, 3, reports[0].dts[3]);
    assert_matches_reference(&clean, "the clean run capped to the accepted timestep");
    let (a, b) = (composite(&rejecting), composite(&clean));
    assert!(a == b, "the root-rejecting run differs from the clean run: {}", diff_report(&a, &b));
}

/// worker 1 rejects the last stage of the second fine substep of step 3; every worker rolls
/// the root step back and replays it at the halved timestep, and the run equals a clean run
/// whose timestep at that step is capped to the accepted one
#[test]
fn a_rejection_in_the_second_fine_substep_rolls_every_worker_back() {
    if in_worker_role() {
        worker_main();
    }
    let test = "a_rejection_in_the_second_fine_substep_rolls_every_worker_back";
    let arm = ragged_17_13();
    let rejecting = run_arm(
        test,
        "rejecting",
        &arm,
        &[("W_REJECT_FINE_WORKER", "1".into()), ("W_REJECT_FINE", "3,1,1".into())],
    );
    assert_all_ok(&rejecting, "the rejecting run");
    let reports: Vec<&Report> = rejecting.iter().map(|o| o.report.as_ref().unwrap()).collect();
    for (w, r) in reports.iter().enumerate() {
        assert_eq!(r.rejections, 1, "worker {w} reports {} rejections", r.rejections);
        assert_eq!(r.dts, reports[0].dts, "worker {w} took other timesteps");
    }
    let clean = capped(test, "clean", &arm, 3, reports[0].dts[3]);
    assert_matches_reference(&clean, "the clean run capped to the accepted timestep");
    let clean_reports: Vec<&Report> = clean.iter().map(|o| o.report.as_ref().unwrap()).collect();
    assert_eq!(clean_reports[0].dts, reports[0].dts, "the capped clean run took other timesteps than the rejecting one");
    assert!(clean_reports.iter().all(|r| r.rejections == 0));
    let (a, b) = (composite(&rejecting), composite(&clean));
    assert!(a == b, "the rejecting run differs from the clean run at the accepted timestep: {}", diff_report(&a, &b));
}
