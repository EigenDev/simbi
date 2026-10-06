// =============================================================================
// worker_mhd_cyl_rz_processes.rs
//
// the distributed evolution of ideal newtonian MHD on the cylindrical r-z
// plane across real processes, with the first-order correction fired on one
// side of a tile cut. the corner EMF E_phi pairs the z-face fluxes across r
// and the r-face fluxes across z; a face on a cut belongs to two tiles and
// each writes its own copy from its own curl of the edge EMFs, so the copies
// agree only when both tiles form and correct the cut's corner EMFs
// identically. every worker audits the shared faces after every exchange
// point.
//
// the flow is a smooth poloidal circulation between reflecting radial walls
// with a periodic z, which corrects nothing by itself; the flag is raised on
// one physical cell against a cut at a named step and stage, on whichever
// worker holds it. arms: a cut along z, a cut along r, and a two-by-two
// layout with the cell in a tile corner, each with the flag in the RK2
// predictor and in the corrector, under the Contact and under the UCT corner
// EMF. every arm compares its assembled cells and
// faces bit for bit with the one-worker run under the same flag, asserts that
// the two copies of every shared face agree, that div B in the chart's
// r-weighted form is at round-off per tile and on the assembled grid, that no
// cell froze, that the worker holding the cell booked one troubled cell and
// every neighbor ran one correction on a clean tile, and that the flag changed
// the one-worker run.
//
// run: cargo test -p symbi --test worker_mhd_cyl_rz_processes
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
use symbi_geometry::Cylindrical;
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

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
const H: f64 = 1.0 / N as f64;
const R_LO: f64 = 1.0;
const GAMMA: f64 = 5.0 / 3.0;
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

/// the radius of global r-face `i`
fn r_face(i: usize) -> f64 {
    R_LO + i as f64 * H
}

/// the axial field B_z(r): z-uniform with B_r = 0, so the seed is divergence-free on the faces
fn bz(r: f64) -> f64 {
    0.3 + 0.2 * (2.0 * PI * (r - R_LO)).sin()
}

/// a smooth poloidal circulation between reflecting radial walls: v_r vanishes on both walls,
/// and every tile seeds the whole grid's value at a shared face
fn tile(arm: &Arm, flat: usize) -> Sim {
    let p = arm.partition();
    let counts = p.counts();
    let tc = unflatten(flat, counts);
    let ext = p.tile_extents(tc);
    let bnd = Boundaries(std::array::from_fn(|a| {
        if a == 1 {
            let kind = if counts[1] > 1 {
                BoundaryType::CoarseFine
            } else {
                BoundaryType::Periodic
            };
            [kind, kind]
        } else {
            [
                if tc[0] == 0 {
                    BoundaryType::Reflect
                } else {
                    BoundaryType::CoarseFine
                },
                if tc[0] == counts[0] - 1 {
                    BoundaryType::Reflect
                } else {
                    BoundaryType::CoarseFine
                },
            ]
        }
    }));
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([ext[0].1, ext[1].1])
        .origin([R_LO, 0.0])
        .spacing([H, H])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[r, z]| {
            let s = (PI * (r - R_LO)).sin();
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0 + 0.2 * s * (2.0 * PI * z).sin()),
                    Tensor::new([
                        0.2 * s * (2.0 * PI * z).sin(),
                        0.1 * s,
                        0.2 * (PI * (r - R_LO)).cos() * (2.0 * PI * z).cos(),
                    ]),
                    Pressure(1.0),
                ),
                Tensor::new([0.0, 0.1 * s * (2.0 * PI * z).cos(), bz(r)]),
            )
        })
        .seed_faces(|axis, [r, _z]| match axis {
            0 => 0.0,
            _ => bz(r),
        })
        .build()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// a worker's record of what it holds: (kind, global i, global j, bits). kind 0 is the
/// conserved density of an owned cell; kind 1 + d is a face of component d at the tile's
/// own local index, its closing copy on a cut included, so a shared face is recorded by both
/// of its tiles
fn dump(sims: &[Sim], mine: &[usize], partition: &Partition<2>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut push = |kind: u64, g: [usize; 2], v: f64| {
        for w in [kind, g[0] as u64, g[1] as u64, v.to_bits()] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    };
    for (sim, &flat) in sims.iter().zip(mine) {
        let ext = partition.tile_extents(unflatten(flat, partition.counts()));
        let lo = [
            sim.geom.interior.spaces[0].lo,
            sim.geom.interior.spaces[1].lo,
        ];
        let den = sim.fields.cons.den.view();
        for c in sim.geom.interior.iter() {
            let g = [
                ext[0].0 + (c[0] - lo[0]) as usize,
                ext[1].0 + (c[1] - lo[1]) as usize,
            ];
            push(0, g, *den.at(c));
        }
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for d in 0..2 {
            let view = mhd.bface.b[d].view();
            for i in 0..=ext[0].1 {
                for j in 0..=ext[1].1 {
                    if (d == 0 && j == ext[1].1) || (d == 1 && i == ext[0].1) {
                        continue;
                    }
                    let g = [ext[0].0 + i, (ext[1].0 + j) % N];
                    push(
                        1 + d as u64,
                        g,
                        *view.at([lo[0] + i as isize, lo[1] + j as isize]),
                    );
                }
            }
        }
    }
    out
}

/// the largest |div B| over a tile's interior cells from that tile's own faces, in the chart's
/// form (1/r) d_r(r B_r) + d_z B_z with r at the faces and the cell's mid-radius
fn tile_div(sim: &Sim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let (br, bzf) = (mhd.bface.b[0].view(), mhd.bface.b[1].view());
    let lo = sim.geom.interior.spaces[0].lo;
    let off = sim.geom.index_offset[0];
    let mut worst = 0.0f64;
    for c in sim.geom.interior.iter() {
        let gi = (c[0] - lo + off) as usize;
        let (rl, rh) = (r_face(gi), r_face(gi + 1));
        let div = (rh * *br.at([c[0] + 1, c[1]]) - rl * *br.at(c)) / (0.5 * (rl + rh) * H)
            + (*bzf.at([c[0], c[1] + 1]) - *bzf.at(c)) / H;
        worst = worst.max(div.abs());
    }
    worst
}

fn run_worker(arm: &Arm, me: WorkerId, out: PathBuf) -> Result<(), WorkerError> {
    let partition = arm.partition();
    let n = partition.n_tiles();
    let owner: Vec<WorkerId> = arm.owner.iter().map(|&w| WorkerId(w)).collect();
    let placement = Placement::new(arm.workers(), owner).expect("placement");
    let mine: Vec<usize> = (0..n)
        .filter(|&f| placement.owner_of(TileId(f as u32)) == me)
        .collect();
    let mut built: Vec<Sim> = mine.iter().map(|&f| tile(arm, f)).collect();
    // the corner EMF the arm names: Contact or UCT, both built for the r-z plane's E_phi
    let ct = match std::env::var("W_CT").as_deref() {
        Ok("uct") => CtMethod::Uct,
        Ok("contact") => CtMethod::Contact,
        other => panic!("W_CT names no constrained transport: {other:?}"),
    };
    let kernels: Vec<_> = built.iter().map(|s| s.substrate().ct_method(ct)).collect();
    let tiles: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(
        &partition,
        &Topology::wrapping([false, true]),
        &schema,
        built[0].geom.ng,
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
            credential: 0x0D7A,
            build_id: "worker_mhd_cyl_rz_processes".into(),
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
    let lens = exchange.lens();
    let max_payload = (lens.iter().max().copied().unwrap_or(0) * 8).max(64);
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
                0,
            ],
        ));
    }
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 2, NewtonianMhd>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 10.0,
        max_steps: STEPS,
        deadline: TRANSFER_DEADLINE,
        injection,
        audit_every_point: true,
    };
    let mut stores: Vec<&mut FieldStore<2, 3, HostMemory, f64>> =
        built.iter_mut().map(|s| &mut **s).collect();
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
    std::fs::write(&out, dump(&built, &mine, &partition)).unwrap();
    let div = built.iter().map(tile_div).fold(0.0, f64::max);
    println!(
        "REPORT steps={} exchanges={} neighbor={} troubled={} frozen={} audits={} div={} \
         audit_faces={} audit_bytes={} audit_us={} exchange_us={}",
        report.steps,
        report.troubled_exchanges,
        report.neighbor_corrections,
        report.troubled_cells,
        report.frozen_cells,
        report.audits,
        div.to_bits(),
        report.audit_cost.faces,
        report.audit_cost.wire_bytes,
        report.audit_cost.elapsed.as_micros(),
        report.timing.exchange.as_micros()
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
    audit_faces: u64,
    audit_bytes: u64,
    audit_us: u64,
    exchange_us: u64,
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
        line.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
    };
    Some(Report {
        steps: field("steps")?.parse().ok()?,
        exchanges: field("exchanges")?.parse().ok()?,
        neighbor: field("neighbor")?.parse().ok()?,
        troubled: field("troubled")?.parse().ok()?,
        frozen: field("frozen")?.parse().ok()?,
        audits: field("audits")?.parse().ok()?,
        div: f64::from_bits(field("div")?.parse().ok()?),
        audit_faces: field("audit_faces")?.parse().ok()?,
        audit_bytes: field("audit_bytes")?.parse().ok()?,
        audit_us: field("audit_us")?.parse().ok()?,
        exchange_us: field("exchange_us")?.parse().ok()?,
    })
}

fn run_arm(test: &str, tag: &str, arm: &Arm, extra: &[(&str, String)]) -> Vec<Outcome> {
    let dir = std::env::temp_dir().join(format!(
        "symbi_mhd_cyl_rz_worker_{}_{test}_{tag}",
        std::process::id()
    ));
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
    while children
        .iter_mut()
        .any(|(c, _)| c.try_wait().unwrap().is_none())
    {
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
            c.stdout
                .take()
                .unwrap()
                .read_to_string(&mut stdout)
                .unwrap();
            c.stderr
                .take()
                .unwrap()
                .read_to_string(&mut stderr)
                .unwrap();
            let records = std::fs::read(out)
                .map(|bytes| {
                    bytes
                        .chunks_exact(32)
                        .map(|r| {
                            let word = |k: usize| {
                                u64::from_le_bytes(r[8 * k..8 * k + 8].try_into().unwrap())
                            };
                            (word(0), word(1) as usize, word(2) as usize, word(3))
                        })
                        .collect()
                })
                .unwrap_or_default();
            Outcome {
                code: status.code(),
                stderr,
                report: parse_report(&stdout),
                records,
            }
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

/// the assembled grid of an arm: density per cell and each face component per distinct
/// global face (radial faces run 0..=N, axial faces wrap); every shared face's copies are
/// asserted equal
struct Assembled {
    den: Vec<u64>,
    faces: BTreeMap<(u64, usize, usize), u64>,
}

fn assemble(outcomes: &[Outcome], label: &str) -> Assembled {
    let mut den = vec![None; N * N];
    let mut copies: BTreeMap<(u64, usize, usize), Vec<u64>> = BTreeMap::new();
    for o in outcomes {
        for &(kind, i, j, bits) in &o.records {
            match kind {
                0 => assert!(
                    den[i * N + j].replace(bits).is_none(),
                    "{label}: cell ({i},{j}) written twice"
                ),
                d => copies.entry((d, i, j)).or_default().push(bits),
            }
        }
    }
    let mut shared = 0;
    for ((d, i, j), values) in &copies {
        if values.len() > 1 {
            shared += 1;
            assert!(
                values.iter().all(|v| v == &values[0]),
                "{label}: the copies of face {d} at ({i},{j}) differ: {values:?}"
            );
        }
    }
    assert!(shared > 0, "{label}: no face was held by two tiles");
    assert_eq!(
        copies.len(),
        (N + 1) * N + N * N,
        "{label}: faces missing from the assembly"
    );
    Assembled {
        den: den
            .into_iter()
            .map(|c| c.expect("a cell no worker wrote"))
            .collect(),
        faces: copies.into_iter().map(|(k, v)| (k, v[0])).collect(),
    }
}

fn assembled_div(a: &Assembled) -> f64 {
    let at = |d: u64, i: usize, j: usize| f64::from_bits(a.faces[&(d, i, j % N)]);
    let mut worst = 0.0f64;
    for i in 0..N {
        let (rl, rh) = (r_face(i), r_face(i + 1));
        for j in 0..N {
            let div = (rh * at(1, i + 1, j) - rl * at(1, i, j)) / (0.5 * (rl + rh) * H)
                + (at(2, i, j + 1) - at(2, i, j)) / H;
            worst = worst.max(div.abs());
        }
    }
    worst
}

fn differing_faces(a: &Assembled, b: &Assembled) -> usize {
    a.faces
        .iter()
        .filter(|(k, v)| b.faces.get(k) != Some(v))
        .count()
}

fn assert_all_ok(outcomes: &[Outcome], label: &str) {
    for (w, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "{label}: worker {w} failed:\n{}", o.stderr);
    }
}

fn uncut() -> Arm {
    Arm {
        cuts0: vec![],
        cuts1: vec![],
        owner: vec![0],
    }
}

/// `neighbors` are the workers that hold no troubled cell and must still correct
/// `extra` is the numerics every arm shares beyond the edge EMF, the resistivity among them
fn assert_cut_matches_uncut(
    test: &str,
    arm: Arm,
    cell: (usize, usize),
    neighbors: &[usize],
    stage: usize,
    ct: &str,
    extra: &[(&str, String)],
) {
    let label = format!("{ct}, stage {stage}, {} workers", arm.workers());
    let mut base = vec![("W_CT", ct.to_string())];
    base.extend(extra.iter().cloned());
    let mut inject = base.clone();
    inject.extend([
        ("W_TROUBLE_STAGE", stage.to_string()),
        ("W_TROUBLE_I", cell.0.to_string()),
        ("W_TROUBLE_J", cell.1.to_string()),
    ]);
    let clean = run_arm(test, &format!("clean_{ct}_{stage}"), &uncut(), &base);
    assert_all_ok(&clean, "clean");
    let clean_report = clean[0].report.clone().expect("report");
    assert_eq!(
        (clean_report.troubled, clean_report.exchanges),
        (0, 0),
        "the circulation corrected by itself"
    );
    let reference = run_arm(test, &format!("uncut_{ct}_{stage}"), &uncut(), &inject);
    assert_all_ok(&reference, "uncut");
    let want = assemble(&reference, "uncut");
    let clean_grid = assemble(&clean, "clean");
    if ct != "contact" {
        // premise: the edge EMF under test changes the run against contact, so a silent fallback
        // to contact cannot pass for it
        let mut contact = vec![("W_CT", "contact".to_string())];
        contact.extend(extra.iter().cloned());
        let contact_run = run_arm(test, &format!("clean_contact_against_{ct}_{stage}"), &uncut(), &contact);
        assert_all_ok(&contact_run, "clean contact");
        let contact_grid = assemble(&contact_run, "clean contact");
        let apart = clean_grid.den.iter().zip(&contact_grid.den).filter(|(a, b)| a != b).count() + differing_faces(&clean_grid, &contact_grid);
        assert!(apart > 0, "{label}: {ct} and contact evolve the same run; the {ct} arms test nothing contact does not");
    }
    let changed = want
        .den
        .iter()
        .zip(&clean_grid.den)
        .filter(|(a, b)| a != b)
        .count()
        + differing_faces(&want, &clean_grid);
    assert!(
        changed > 0,
        "{label}: the flag left the one-worker run unchanged; the gate is vacuous"
    );
    let reference_report = reference[0].report.clone().expect("report");
    assert_eq!(
        (reference_report.troubled, reference_report.frozen),
        (1, 0),
        "{label}: uncut census"
    );
    assert!(
        reference_report.div < DIV_TOL,
        "{label}: one-worker div B {:e}",
        reference_report.div
    );

    let outcomes = run_arm(test, &format!("cut_{ct}_{stage}"), &arm, &inject);
    assert_all_ok(&outcomes, &label);
    for (w, o) in outcomes.iter().enumerate() {
        let r = o.report.as_ref().expect("report");
        assert_eq!(r.steps, STEPS, "{label}: worker {w} steps");
        assert_eq!(r.frozen, 0, "{label}: worker {w} froze a cell");
        assert_eq!(r.exchanges, 1, "{label}: worker {w} flag exchanges");
        assert_eq!(
            r.troubled,
            u64::from(w == 0),
            "{label}: worker {w} troubled cells"
        );
        assert_eq!(
            r.neighbor,
            u64::from(neighbors.contains(&w)),
            "{label}: worker {w} clean-tile corrections"
        );
        // the prime audit, one per stage exchange and one per stage flag... the audit follows
        // every state exchange: prime plus two stages per step
        assert_eq!(r.audits, 1 + 2 * STEPS, "{label}: worker {w} audits");
        assert!(
            r.div < DIV_TOL,
            "{label}: worker {w} tile-local div B {:e}",
            r.div
        );
        // the audit's cost beside the halo exchange it follows, for the record
        eprintln!(
            "AUDIT {label}: worker {w} audits {} faces {} bytes {} audit {} us, halo exchange {} us",
            r.audits, r.audit_faces, r.audit_bytes, r.audit_us, r.exchange_us
        );
    }
    let got = assemble(&outcomes, &label);
    assert!(
        assembled_div(&got) < DIV_TOL,
        "{label}: assembled div B {:e}",
        assembled_div(&got)
    );
    let differing = got
        .den
        .iter()
        .zip(&want.den)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "{label}: {differing} cells differ from the one-worker run"
    );
    let differing = differing_faces(&got, &want);
    assert_eq!(
        differing, 0,
        "{label}: {differing} faces differ from the one-worker run"
    );
}

fn z_cut_two_workers() -> Arm {
    Arm {
        cuts0: vec![],
        cuts1: vec![16],
        owner: vec![0, 1],
    }
}

fn r_cut_two_workers() -> Arm {
    Arm {
        cuts0: vec![16],
        cuts1: vec![],
        owner: vec![0, 1],
    }
}

fn four_workers() -> Arm {
    Arm {
        cuts0: vec![16],
        cuts1: vec![16],
        owner: vec![0, 1, 2, 3],
    }
}

macro_rules! cut_arm {
    ($name:ident, $arm:expr, $cell:expr, $neighbors:expr, $stage:expr, $ct:expr) => {
        #[test]
        fn $name() {
            if in_worker_role() {
                worker_main();
            }
            assert_cut_matches_uncut(
                stringify!($name),
                $arm,
                $cell,
                $neighbors,
                $stage,
                $ct,
                &[],
            );
        }
    };
}

cut_arm!(
    trouble_against_a_z_cut_in_the_predictor,
    z_cut_two_workers(),
    (10, 15),
    &[1],
    0,
    "contact"
);
cut_arm!(
    uct_trouble_against_a_z_cut_in_the_predictor,
    z_cut_two_workers(),
    (10, 15),
    &[1],
    0,
    "uct"
);
cut_arm!(
    trouble_against_a_z_cut_in_the_corrector,
    z_cut_two_workers(),
    (10, 15),
    &[1],
    1,
    "contact"
);
cut_arm!(
    uct_trouble_against_a_z_cut_in_the_corrector,
    z_cut_two_workers(),
    (10, 15),
    &[1],
    1,
    "uct"
);
cut_arm!(
    trouble_against_an_r_cut_in_the_predictor,
    r_cut_two_workers(),
    (15, 10),
    &[1],
    0,
    "contact"
);
cut_arm!(
    uct_trouble_against_an_r_cut_in_the_predictor,
    r_cut_two_workers(),
    (15, 10),
    &[1],
    0,
    "uct"
);
cut_arm!(
    trouble_against_an_r_cut_in_the_corrector,
    r_cut_two_workers(),
    (15, 10),
    &[1],
    1,
    "contact"
);
cut_arm!(
    uct_trouble_against_an_r_cut_in_the_corrector,
    r_cut_two_workers(),
    (15, 10),
    &[1],
    1,
    "uct"
);
cut_arm!(
    trouble_in_a_tile_corner_in_the_predictor,
    four_workers(),
    (15, 15),
    &[1, 2, 3],
    0,
    "contact"
);
cut_arm!(
    uct_trouble_in_a_tile_corner_in_the_predictor,
    four_workers(),
    (15, 15),
    &[1, 2, 3],
    0,
    "uct"
);
cut_arm!(
    trouble_in_a_tile_corner_in_the_corrector,
    four_workers(),
    (15, 15),
    &[1, 2, 3],
    1,
    "contact"
);
cut_arm!(
    uct_trouble_in_a_tile_corner_in_the_corrector,
    four_workers(),
    (15, 15),
    &[1, 2, 3],
    1,
    "uct"
);
