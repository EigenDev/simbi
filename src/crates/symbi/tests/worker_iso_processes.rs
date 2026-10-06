// =============================================================================
// worker_iso_processes.rs
//
// the distributed evolution of the isothermal closures across real processes:
// isothermal newtonian hydro and isothermal MHD. the closure holds one sound
// speed per run, so a cut ghost continued from a neighbor carries the same
// temperature as the tile's own cells and the halo exchange of density and
// velocity is the whole dependency. a flag raised on one physical cell against
// a tile cut at a named step and stage fires the first-order correction on one
// side, the flag crosses the cut, and the clean neighbor redoes its side. two
// workers with the cell against a cut and four workers with the cell in a tile
// corner each reproduce the one-worker run under the same flag bit for bit:
// every cell, and for the magnetized closure every face of the staggered field,
// with the two copies of each shared face equal, div B at round-off per tile and
// on the assembled grid, no cell frozen, the interface audited after every
// exchange, and the flag shown to have changed the one-worker run.
//
// run: cargo test -p symbi --test worker_iso_processes
// =============================================================================

use std::collections::BTreeMap;
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
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::RegimeKind;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type IsoSim = SimStateGeneric<IsoNewtonian, 2, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type ImhdSim = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 64;
const H: f64 = 1.0 / N as f64;
const CS: f64 = 1.0;
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

/// a cut face is filled by the exchange; a domain face on an uncut axis is the tile's own
/// periodic fill
fn boundaries(arm: &Arm) -> Boundaries<2> {
    let counts = arm.partition().counts();
    Boundaries(std::array::from_fn(|a| {
        let kind = if counts[a] > 1 { BoundaryType::CoarseFine } else { BoundaryType::Periodic };
        [kind, kind]
    }))
}

fn extents(arm: &Arm, flat: usize) -> [(usize, usize); 2] {
    let p = arm.partition();
    p.tile_extents(unflatten(flat, p.counts()))
}

/// a smooth periodic shear with a density ripple; the isothermal flux depends on density
/// and velocity alone, so this is the whole state
fn iso_tile(arm: &Arm, flat: usize) -> IsoSim {
    let ext = extents(arm, flat);
    IsoSim::build(IsoNewtonian, Isothermal { cs: CS }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([H, H])
        .origin([ext[0].0 as f64 * H, ext[1].0 as f64 * H])
        .boundaries(boundaries(arm))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            PrimG::<f64, 2, IsoModel>::isothermal(
                Density(1.0 + 0.2 * (2.0 * PI * x).cos() * (2.0 * PI * y).sin()),
                Tensor::new([0.5 * (2.0 * PI * y).sin(), 0.5 * (2.0 * PI * x).sin()]),
            )
        })
        .build()
}

/// the orszag-tang vortex under the isothermal closure: the face field is one function of
/// the global face position, so every tile seeds the same value on a shared face
fn imhd_tile(arm: &Arm, flat: usize) -> ImhdSim {
    let ext = extents(arm, flat);
    ImhdSim::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([H, H])
        .origin([ext[0].0 as f64 * H, ext[1].0 as f64 * H])
        .boundaries(boundaries(arm))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            MhdPrimG::<f64, 3, IsoModel>::new(
                PrimG::isothermal(
                    Density(25.0 / 9.0),
                    Tensor::new([-(2.0 * PI * y).sin(), (2.0 * PI * x).sin(), 0.0]),
                ),
                Tensor::new([-(2.0 * PI * y).sin(), (4.0 * PI * x).sin(), 0.0]),
            )
        })
        .seed_faces(|axis, [x, y]| match axis {
            0 => -(2.0 * PI * y).sin(),
            _ => (4.0 * PI * x).sin(),
        })
        .build()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// a record of what a worker holds: (kind, global i, global j, bits). kind 0 is the
/// conserved density of an owned cell; kind 1 + d is a face of component d at the tile's
/// own local index, its closing copy on a cut included, so a shared face is recorded by
/// both of its tiles
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

fn dump_faces(out: &mut Vec<u8>, sim: &ImhdSim, ext: [(usize, usize); 2]) {
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    let mhd = sim.fields.mhd.as_ref().unwrap();
    for d in 0..2 {
        let view = mhd.bface.b[d].view();
        for i in 0..=ext[0].1 {
            for j in 0..=ext[1].1 {
                if (d == 0 && j == ext[1].1) || (d == 1 && i == ext[0].1) {
                    continue;
                }
                let g = [(ext[0].0 + i) % N, (ext[1].0 + j) % N];
                push_record(out, 1 + d as u64, g, *view.at([lo[0] + i as isize, lo[1] + j as isize]));
            }
        }
    }
}

/// the largest |div B| over a tile's interior cells from that tile's own faces
fn tile_div(sim: &ImhdSim) -> f64 {
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
            credential: 0x150,
            build_id: "worker_iso_processes".into(),
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

fn run_iso(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let s = session(arm, me);
    let mut built: Vec<IsoSim> = s.mine.iter().map(|&f| iso_tile(arm, f)).collect();
    let kernels: Vec<_> = built.iter().map(|sim| sim.substrate()).collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(&s.partition, &Topology::wrapping([true; 2]), &schema, built[0].geom.ng)
        .expect("plan");
    let exchange = PlanExchange::new(&plan, &s.placement, me, PhaseTable::flat(2).unwrap());
    let r = rendezvous(&s, arm, plan.digest, built[0].geom.ng);
    let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
    let (link, session_id) = connect(&r, max_payload)?;
    let mut fabric = exchange.fabric(link, session_id);
    let cfg = config(RegimeKind::of::<f64, 2, IsoNewtonian>());
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

fn run_imhd(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let s = session(arm, me);
    let mut built: Vec<ImhdSim> = s.mine.iter().map(|&f| imhd_tile(arm, f)).collect();
    // the edge EMF: Contact, or UCT under W_CT=uct
    let uct = std::env::var("W_CT").is_ok_and(|v| v == "uct");
    let kernels: Vec<_> = built
        .iter()
        .map(|sim| {
            let k = sim.substrate();
            if uct { k.ct_method(CtMethod::Uct) } else { k }
        })
        .collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(&s.partition, &Topology::wrapping([true; 2]), &schema, built[0].geom.ng)
        .expect("plan");
    let exchange = PlanExchange::new(&plan, &s.placement, me, PhaseTable::flat(2).unwrap());
    let r = rendezvous(&s, arm, plan.digest, built[0].geom.ng);
    let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
    let (link, session_id) = connect(&r, max_payload)?;
    let mut fabric = exchange.fabric(link, session_id);
    let cfg = config(RegimeKind::of::<f64, 2, IsothermalMhd>());
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
    let magnetized = std::env::var("W_REGIME").is_ok_and(|v| v == "imhd");
    let result = if magnetized { run_imhd(&arm, me, out) } else { run_iso(&arm, me, out) };
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
    let dir = std::env::temp_dir().join(format!("symbi_iso_worker_{}_{test}_{tag}", std::process::id()));
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

/// the assembled grid of an arm: density per cell and, when faces were recorded, each face
/// component per distinct global face with every shared face's copies asserted equal
struct Assembled {
    den: Vec<u64>,
    faces: Option<[Vec<u64>; 2]>,
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
        assert!(copies.is_empty(), "{label}: faces recorded for a closure without a field");
        return Assembled { den, faces: None };
    }
    let mut faces: [Vec<Option<u64>>; 2] = [vec![None; N * N], vec![None; N * N]];
    let mut shared = 0;
    for ((d, i, j), values) in &copies {
        if values.len() > 1 {
            shared += 1;
            assert!(
                values.iter().all(|v| v == &values[0]),
                "{label}: the copies of face {d} at ({i},{j}) differ: {values:?}"
            );
        }
        faces[*d as usize - 1][i * N + j] = Some(values[0]);
    }
    assert!(shared > 0, "{label}: no face was held by two tiles");
    Assembled {
        den,
        faces: Some(faces.map(|f| f.into_iter().map(|v| v.expect("a face no worker wrote")).collect())),
    }
}

fn assembled_div(faces: &[Vec<u64>; 2]) -> f64 {
    let at = |d: usize, i: usize, j: usize| f64::from_bits(faces[d][(i % N) * N + (j % N)]);
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
        (Some(fa), Some(fb)) => (0..2).map(|d| fa[d].iter().zip(&fb[d]).filter(|(x, y)| x != y).count()).sum(),
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

/// `neighbors` are the workers that hold no troubled cell and must still correct
fn assert_cut_matches_uncut(
    test: &str,
    regime: &str,
    ct: &str,
    arm: Arm,
    cell: (usize, usize),
    neighbors: &[usize],
    stage: usize,
) {
    let magnetized = regime == "imhd";
    let label = format!("{regime} {ct}, stage {stage}, {} workers", arm.workers());
    let base = vec![("W_REGIME", regime.to_string()), ("W_CT", ct.to_string())];
    let mut inject = base.clone();
    inject.extend([
        ("W_TROUBLE_STAGE", stage.to_string()),
        ("W_TROUBLE_I", cell.0.to_string()),
        ("W_TROUBLE_J", cell.1.to_string()),
    ]);
    let clean = run_arm(test, &format!("clean_{stage}"), &uncut(), &base);
    assert_all_ok(&clean, "clean");
    let clean_report = clean[0].report.clone().expect("report");
    assert_eq!((clean_report.troubled, clean_report.exchanges), (0, 0), "{label}: the flow corrected by itself");
    let reference = run_arm(test, &format!("uncut_{stage}"), &uncut(), &inject);
    assert_all_ok(&reference, "uncut");
    let want = assemble(&reference, "uncut", magnetized);
    let clean_grid = assemble(&clean, "clean", magnetized);
    if magnetized && ct != "contact" {
        // premise: the edge EMF under test changes the run against contact, so a silent fallback
        // to contact cannot pass for it
        let contact = vec![("W_REGIME", regime.to_string()), ("W_CT", "contact".to_string())];
        let contact_run = run_arm(test, &format!("clean_contact_against_{ct}_{stage}"), &uncut(), &contact);
        assert_all_ok(&contact_run, "clean contact");
        assert!(differing(&clean_grid, &assemble(&contact_run, "clean contact", magnetized)) > 0, "{label}: {ct} and contact evolve the same run; the {ct} arms test nothing contact does not");
    }
    assert!(
        differing(&want, &clean_grid) > 0,
        "{label}: the flag left the one-worker run unchanged; the gate is vacuous"
    );
    let reference_report = reference[0].report.clone().expect("report");
    assert_eq!((reference_report.troubled, reference_report.frozen), (1, 0), "{label}: uncut census");
    if magnetized {
        assert!(reference_report.div < DIV_TOL, "{label}: one-worker div B {:e}", reference_report.div);
    }

    let outcomes = run_arm(test, &format!("cut_{stage}"), &arm, &inject);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().expect("report");
        assert_eq!(r.steps, STEPS, "{label}: worker {w} steps");
        assert_eq!(r.frozen, 0, "{label}: worker {w} froze a cell");
        assert_eq!(r.exchanges, 1, "{label}: worker {w} flag exchanges");
        assert_eq!(r.troubled, u64::from(w == 0), "{label}: worker {w} troubled cells");
        assert_eq!(r.neighbor, u64::from(neighbors.contains(&w)), "{label}: worker {w} clean-tile corrections");
        // the audit covers the shared faces of the staggered field and follows every state
        // exchange, prime plus two stages per step; a closure without a field has no
        // interface to audit
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

#[test]
fn isothermal_trouble_against_the_cut_in_the_predictor_two_workers() {
    if in_worker_role() {
        worker_main();
    }
    let test = "isothermal_trouble_against_the_cut_in_the_predictor_two_workers";
    assert_cut_matches_uncut(test, "iso", "contact", two_workers(), (31, 10), &[1], 0);
}

#[test]
fn isothermal_trouble_in_a_tile_corner_in_the_corrector_four_workers() {
    if in_worker_role() {
        worker_main();
    }
    let test = "isothermal_trouble_in_a_tile_corner_in_the_corrector_four_workers";
    assert_cut_matches_uncut(test, "iso", "contact", four_workers(), (31, 31), &[1, 2, 3], 1);
}

#[test]
fn isothermal_mhd_contact_trouble_against_the_cut_in_the_predictor_two_workers() {
    if in_worker_role() {
        worker_main();
    }
    let test = "isothermal_mhd_contact_trouble_against_the_cut_in_the_predictor_two_workers";
    assert_cut_matches_uncut(test, "imhd", "contact", two_workers(), (31, 10), &[1], 0);
}

#[test]
fn isothermal_mhd_uct_trouble_in_a_tile_corner_in_the_corrector_four_workers() {
    if in_worker_role() {
        worker_main();
    }
    let test = "isothermal_mhd_uct_trouble_in_a_tile_corner_in_the_corrector_four_workers";
    assert_cut_matches_uncut(test, "imhd", "uct", four_workers(), (31, 31), &[1, 2, 3], 1);
}
