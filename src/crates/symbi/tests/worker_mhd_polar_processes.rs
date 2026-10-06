// =============================================================================
// worker_mhd_polar_processes.rs
//
// the polar seam under ideal newtonian MHD across real processes: a
// spherical shell gridded over both poles with a periodic azimuth cut into
// tiles, so the ghost band beyond a pole, cells and transverse faces alike,
// is the interior half a turn away on other tiles and takes its odd
// components' sign after every state exchange. the azimuth is cut into two
// equal slabs and into three uneven ones whose half-turn images straddle
// tiles; the flag is raised on a cell at a pole against an azimuth cut.
// every arm compares its assembled cells and faces bit for bit with the
// one-worker run, whose pole band is its own half-turn, asserts the two
// copies of every shared face agree, that every cell's magnetic flux under the
// constrained-transport face weights is the seed's to round-off on every tile,
// that no cell froze, and that every worker audited its shared faces after
// every exchange.
//
// run: cargo test -p symbi --test worker_mhd_polar_processes
// =============================================================================

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::prelude::SimSubstrate;
use symbi::sim::decomp::{LocalCopy, Partition, PolarSeam, Topology, plan_schema, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
/// the shell: r in [1, 2], theta over both poles, phi around the full circle
const R_LO: f64 = 1.0;
const DR: f64 = 1.0 / N as f64;
const DTHETA: f64 = PI / N as f64;
const DPHI: f64 = 2.0 * PI / N as f64;
const GAMMA: f64 = 5.0 / 3.0;
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

/// the seeded field: a uniform field along the polar axis in spherical components,
/// B_r = cos(theta), B_theta = -sin(theta), so the field threads both poles and every
/// component the half-turn reverses is present
fn field(axis: usize, [_r, theta, _phi]: [f64; 3]) -> f64 {
    match axis {
        0 => theta.cos(),
        1 => -theta.sin(),
        _ => 0.0,
    }
}

/// the polar seam of an arm: present once the azimuth is cut
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
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells([ext[0].1, ext[1].1, ext[2].1])
        .origin([R_LO, 0.0, 0.0])
        .spacing([DR, DTHETA, DPHI])
        .index_offset(std::array::from_fn(|a| ext[a].0 as isize))
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            let [r, theta, phi] = x;
            let b = 0.2 * (-((r - 1.5) / 0.08).powi(2)).exp();
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0 + b * (1.0 + 0.5 * theta.cos()) * (1.0 + 0.3 * phi.cos())),
                    Tensor::new([0.0, 0.05 * (2.0 * theta).sin(), 0.05 * theta.sin() * phi.cos()]),
                    Pressure(1.0 + b),
                ),
                Tensor::new([field(0, x), field(1, x), field(2, x)]),
            )
        })
        .seed_faces(field)
        .build()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// a record of what a worker holds: (kind, global i, j, k, bits). kind 0 is the conserved
/// density of an owned cell; kind 1 + d is a face of component d at the tile's own local
/// index, its closing copy on a cut included, so a shared face is recorded by both tiles
fn dump(sims: &[Sim], mine: &[usize], arm: &Arm) -> Vec<u8> {
    let mut out = Vec::new();
    let mut push = |kind: u64, g: [usize; 3], v: f64| {
        for w in [kind, g[0] as u64, g[1] as u64, g[2] as u64, v.to_bits()] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    };
    for (sim, &flat) in sims.iter().zip(mine) {
        let ext = extents(arm, flat);
        let lo: [isize; 3] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        let den = sim.fields.cons.den.view();
        for c in sim.geom.interior.iter() {
            let g: [usize; 3] = std::array::from_fn(|a| ext[a].0 + (c[a] - lo[a]) as usize);
            push(0, g, *den.at(c));
        }
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for d in 0..3 {
            let view = mhd.bface.b[d].view();
            let hi: [usize; 3] = std::array::from_fn(|a| ext[a].1 + usize::from(a == d));
            for i in 0..hi[0] {
                for j in 0..hi[1] {
                    for k in 0..hi[2] {
                        let g = [ext[0].0 + i, ext[1].0 + j, (ext[2].0 + k) % N];
                        push(1 + d as u64, g, *view.at([lo[0] + i as isize, lo[1] + j as isize, lo[2] + k as isize]));
                    }
                }
            }
        }
    }
    out
}

/// the unnormalized magnetic flux out of every interior cell with the constrained-transport face
/// weights, the quantity the CT curl preserves cell by cell
fn cell_flux(sim: &Sim) -> Vec<f64> {
    let bg = sim.geom.block_geometry(sim.physics.metric);
    let m = sim.fields.mhd.as_ref().unwrap();
    sim.geom
        .interior
        .iter()
        .map(|c| {
            (0..3)
                .map(|d| {
                    let mut up = c;
                    up[d] += 1;
                    bg.ct_face_weight(up, d) * *m.bface.b[d].view().at(up) - bg.ct_face_weight(c, d) * *m.bface.b[d].view().at(c)
                })
                .sum()
        })
        .collect()
}

/// the largest weighted face flux of a tile, the scale a flux change is round-off against
fn flux_scale(sim: &Sim) -> f64 {
    let bg = sim.geom.block_geometry(sim.physics.metric);
    let m = sim.fields.mhd.as_ref().unwrap();
    sim.geom.interior.iter().flat_map(|c| (0..3).map(move |d| (c, d))).map(|(c, d)| (bg.ct_face_weight(c, d) * *m.bface.b[d].view().at(c)).abs()).fold(0.0, f64::max)
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    let mine: Vec<usize> = (0..n).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    let mut built: Vec<Sim> = mine.iter().map(|&f| tile(arm, f)).collect();
    let kernels: Vec<_> = built.iter().map(|s| s.substrate()).collect();
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
            credential: 0x90B0,
            build_id: "worker_mhd_polar_processes".into(),
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
        regime: RegimeKind::of::<f64, 3, NewtonianMhd>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 10.0,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection,
        audit_every_point: true,
    };
    let seed: Vec<(Vec<f64>, f64)> = built.iter().map(|s| (cell_flux(s), flux_scale(s))).collect();
    let mut stores: Vec<&mut FieldStore<3, 3, HostMemory, f64>> = built.iter_mut().map(|s| &mut **s).collect();
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
    std::fs::write(&out, dump(&built, &mine, arm)).unwrap();
    // the largest change of a cell's weighted flux from the seed's, relative to the tile's flux scale
    let div = built
        .iter()
        .zip(&seed)
        .map(|(s, (flux, scale))| cell_flux(s).iter().zip(flux).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale)
        .fold(0.0, f64::max);
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
    audits: u64,
    div: f64,
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    report: Option<Report>,
    records: Vec<(u64, [usize; 3], u64)>,
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
    let dir = std::env::temp_dir().join(format!("symbi_mhdpolar_worker_{}_{test}_{tag}", std::process::id()));
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
                        .chunks_exact(40)
                        .map(|r| {
                            let word = |k: usize| u64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap());
                            (word(0), [word(1) as usize, word(2) as usize, word(3) as usize], word(4))
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

struct Assembled {
    den: Vec<u64>,
    faces: [Vec<u64>; 3],
}

/// the assembled index: the radial and polar axes carry their closing faces, the azimuth wraps
fn at(g: [usize; 3]) -> usize {
    (g[0] * (N + 1) + g[1]) * N + g[2] % N
}
const CELLS: usize = (N + 1) * (N + 1) * N;

fn assemble(outcomes: &[Outcome], label: &str) -> Assembled {
    let mut den = vec![None; CELLS];
    let mut faces: [Vec<Option<u64>>; 3] = std::array::from_fn(|_| vec![None; CELLS]);
    let mut copies: BTreeMap<(u64, [usize; 3]), Vec<u64>> = BTreeMap::new();
    for o in outcomes {
        for &(kind, g, bits) in &o.records {
            match kind {
                0 => assert!(den[at(g)].replace(bits).is_none(), "{label}: cell {g:?} written twice"),
                d => copies.entry((d, g)).or_default().push(bits),
            }
        }
    }
    let mut shared = 0;
    for ((d, g), values) in &copies {
        if values.len() > 1 {
            shared += 1;
            assert!(values.iter().all(|v| v == &values[0]), "{label}: the copies of face {d} at {g:?} differ: {values:?}");
        }
        faces[*d as usize - 1][at(*g)] = Some(values[0]);
    }
    // a tile's closing face on a periodic axis is its opening face, so even one worker records
    // every wrapped face twice, and a cut adds the faces its two tiles share
    assert!(shared > 0, "{label}: no face was recorded twice");
    // the closing rows carry cells and faces only where a field has them
    Assembled {
        den: den.into_iter().map(|c| c.unwrap_or(0)).collect(),
        faces: faces.map(|f| f.into_iter().map(|v| v.unwrap_or(0)).collect()),
    }
}

fn differing(a: &Assembled, b: &Assembled) -> usize {
    a.den.iter().zip(&b.den).filter(|(x, y)| x != y).count()
        + (0..3).map(|d| a.faces[d].iter().zip(&b.faces[d]).filter(|(x, y)| x != y).count()).sum::<usize>()
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
/// cut, in the predictor
fn azimuth_halves_two_workers() -> Case {
    Case { arm: Arm { cuts: [vec![], vec![], vec![16]], owner: vec![0, 1] }, cell: [10, 0, 15], holder: 0, neighbors: vec![1], stage: 0 }
}

/// the azimuth cut into three uneven slabs whose half-turn images straddle tiles; the cell at
/// the theta = pi pole against the first cut, in the corrector
fn azimuth_uneven_three_workers() -> Case {
    Case {
        arm: Arm { cuts: [vec![], vec![], vec![12, 20]], owner: vec![0, 1, 2] },
        cell: [10, 31, 11],
        holder: 0,
        neighbors: vec![1, 2],
        stage: 1,
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
    assert_eq!((clean_report.troubled, clean_report.exchanges), (0, 0), "{label}: the vortex corrected by itself");
    let reference = run_arm(test, "uncut", &uncut(), &inject);
    assert_all_ok(&reference, "uncut");
    let want = assemble(&reference, "uncut");
    assert!(differing(&want, &assemble(&clean, "clean")) > 0, "{label}: the flag left the one-worker run unchanged; the gate is vacuous");
    let reference_report = reference[0].report.clone().expect("report");
    assert_eq!((reference_report.troubled, reference_report.frozen), (1, 0), "{label}: uncut census");
    // the CT curl keeps every cell's weighted flux: STEPS x 2 stages = 12 curl updates, each moving
    // a cell's six-face sum by at most six round-offs of the flux scale, bound the change by 72 u
    // (measured 5.4 u to 6.0 u over the pole band and both cuts)
    let bound = 72.0 * 0.5 * f64::EPSILON;
    assert!(reference_report.div <= bound, "{label}: one worker's cell flux moved by {:e} of its scale", reference_report.div);

    let outcomes = run_arm(test, "cut", &case.arm, &inject);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().expect("report");
        assert_eq!(r.steps, STEPS, "{label}: worker {w} steps");
        assert_eq!(r.frozen, 0, "{label}: worker {w} froze a cell");
        assert_eq!(r.exchanges, 1, "{label}: worker {w} flag exchanges");
        assert_eq!(r.troubled, u64::from(w == case.holder), "{label}: worker {w} troubled cells");
        assert_eq!(r.neighbor > 0, case.neighbors.contains(&w), "{label}: worker {w} clean-tile corrections {}", r.neighbor);
        assert_eq!(r.audits, 1 + 2 * STEPS, "{label}: worker {w} audits");
        assert!(r.div <= bound, "{label}: worker {w} moved a cell's weighted flux by {:e} of its scale", r.div);
    }
    let got = assemble(&outcomes, &label);
    let n = differing(&got, &want);
    assert_eq!(n, 0, "{label}: {n} cells or faces differ from the one-worker run");
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
