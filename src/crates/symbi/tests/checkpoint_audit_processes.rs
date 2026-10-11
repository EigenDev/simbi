// =============================================================================
// checkpoint_audit_processes.rs
//
// the interface audit that precedes a distributed checkpoint of a magnetized
// state, across real processes. two workers hold two magnetized tiles each,
// prime the halos through the plan, and write one checkpoint through the
// coordinator with the audit in front of it. with equal copies of every shared
// face the file is published, both workers compare faces, and the lower worker
// alone puts bytes on the wire. with one face of one worker's opening copy off
// by one unit in the last place, the audit names the face, every worker exits
// nonzero, and nothing is published: no file and no temporary, because the
// audit runs before the checkpoint transaction opens.
//
// run: cargo test -p symbi --test checkpoint_audit_processes
// =============================================================================

use std::f64::consts::PI;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::prelude::SimSubstrate;
use symbi::sim::decomp::{LocalCopy, Partition, Topology, plan_schema, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_io::Metadata;
use symbi_sim::atlas::{ExchangePlan, ExchangePoint, Placement, TileId};
use symbi_sim::checkpoint::{GlobalGrid, LevelTiles, PhysicsIdentity, TileView};
use symbi_sim::distributed_checkpoint::{
    AuditBasis, CheckpointError, CheckpointRequest, audited_checkpoint, block_credit_for,
};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::worker::field_lists;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 32;
const H: f64 = 1.0 / N as f64;
const GAMMA: f64 = 5.0 / 3.0;
const BUDGET: usize = 200;
const DEADLINE: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(60);
const WORKERS: u32 = 2;
/// the low-x tiles belong to worker 0, the high-x tiles to worker 1, so the x cut crosses
/// workers and the y cut sits inside each
const OWNER: [u32; 4] = [0, 0, 1, 1];

fn partition() -> Partition<2> {
    Partition::explicit([N, N], [vec![N / 2], vec![N / 2]]).unwrap()
}

/// a magnetized tile whose face field is one function of the global face index, so the two
/// copies of every shared face agree by construction
fn tile(p: &Partition<2>, flat: usize) -> Sim {
    let counts = p.counts();
    let ext = p.tile_extents(unflatten(flat, counts));
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([H, H])
        .origin([ext[0].0 as f64 * H, ext[1].0 as f64 * H])
        .boundaries(Boundaries::uniform(BoundaryType::CoarseFine))
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0 + 0.1 * (2.0 * PI * x).sin()),
                    Tensor::new([0.1, -0.2 * (2.0 * PI * y).cos(), 0.0]),
                    Pressure(1.0),
                ),
                Tensor::new([0.0, 0.0, 0.0]),
            )
        })
        .seed_faces(|axis, [x, y]| match axis {
            0 => -(2.0 * PI * y).sin(),
            _ => (4.0 * PI * x).sin(),
        })
        .build();
    sim
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn run_worker(me: WorkerId, ckpt: &Path) -> Result<(), String> {
    let p = partition();
    let n = p.n_tiles();
    let placement = Placement::new(WORKERS, OWNER.iter().map(|&w| WorkerId(w)).collect()).unwrap();
    let mine: Vec<usize> = (0..n).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    let built: Vec<Sim> = mine.iter().map(|&f| tile(&p, f)).collect();
    let kernels: Vec<_> = built.iter().map(|s| s.substrate()).collect();
    let tiles: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let offsets: Vec<[isize; 2]> = mine
        .iter()
        .map(|&f| {
            let ext = p.tile_extents(unflatten(f, p.counts()));
            [ext[0].0 as isize, ext[1].0 as isize]
        })
        .collect();
    let grid = {
        let whole = Partition::uniform([N, N], [1, 1]).unwrap();
        GlobalGrid::of_state(&tile(&whole, 0))
    };
    let identity = PhysicsIdentity::of(&built[0]);
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(&p, &Topology::open(), &schema, built[0].geom.ng).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, me, PhaseTable::flat(2).unwrap());
    let credit = block_credit_for(BUDGET, 2);
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0xA0D1,
            build_id: "checkpoint_audit_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: PhaseTable::flat(2).unwrap().digest(),
            ng: built[0].geom.ng as u32,
            workers: WORKERS,
            block_credit: credit,
        },
        startup: DEADLINE,
    };
    let lens = exchange.lens();
    let max_payload = (lens.iter().max().copied().unwrap_or(0) * 8).max(credit as usize);
    let (link, session) = connect(&r, max_payload).map_err(|e| e.to_string())?;
    let mut fabric = exchange.fabric(link, session).with_block_credit(credit);
    let stores: Vec<&FieldStore<2, 3, HostMemory, f64>> = built.iter().map(|s| &**s).collect();
    let result = (|| -> Result<(), String> {
        for (s, k) in stores.iter().zip(&kernels) {
            k.c2p(s);
            k.ghost_fill(s);
        }
        let fields = field_lists(&tiles, &stores, n);
        // the prime, then the stage exchanges of one accepted attempt of step 0 (the state
        // itself is unchanged): the audited checkpoint of step 0 follows an accepted attempt
        for point in [ExchangePoint::Prime, ExchangePoint::Stage(0), ExchangePoint::Stage(1)] {
            for axis in 0..2 {
                let epoch = exchange.epoch(point, 0, 0, axis);
                exchange
                    .open_axis(&fields, &vec![0; n], &LocalCopy, &mut fabric, epoch)
                    .map_err(|e| e.to_string())?;
                fabric.wait(DEADLINE).map_err(|e| e.to_string())?;
                exchange
                    .finish_axis(&fields, &mut fabric, axis)
                    .map_err(|e| e.to_string())?;
            }
        }
        for (s, k) in stores.iter().zip(&kernels) {
            k.ghost_fill(s);
        }
        // the injected disagreement: this worker's opening copy of the x-cut interface of
        // its first tile, one row, one unit in the last place
        if env_u64("W_PERTURB_WORKER") == Some(u64::from(me.0)) {
            let s = stores[0];
            let field = &s.fields.mhd.as_ref().unwrap().bface.b[0];
            let at = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo + 3];
            let old = *field.view().at(at);
            *field.view_mut().at_mut(at) = f64::from_bits(old.to_bits() + 1);
        }
        let level = LevelTiles {
            tiles: stores
                .iter()
                .zip(&offsets)
                .map(|(s, &offset)| TileView { state: *s, offset })
                .collect(),
            grid: grid.clone(),
            header: None,
        };
        let extras = Metadata::new();
        let req = CheckpointRequest {
            path: &ckpt.to_string_lossy(),
            extras: &extras,
            budget: BUDGET,
            deadline: DEADLINE,
            drop_block: None,
            duplicate_first_block: false,
            vote_no: false,
        };
        let cost = audited_checkpoint::<NewtonianMhd, 2, 3, HostMemory, _>(
            &mut fabric, &exchange, &identity, &level, &tiles, &req, AuditBasis::AfterStep(0),
        )
        .map_err(|e| match e {
            CheckpointError::Interface(m) => format!("interface: {m}"),
            other => other.to_string(),
        })?;
        println!(
            "COST faces={} bytes={} audit_us={} checkpoint_us={}",
            cost.audit.faces,
            cost.audit.wire_bytes,
            cost.audit.elapsed.as_micros(),
            cost.checkpoint.as_micros()
        );
        fabric.finish(DEADLINE).map_err(|e| e.to_string())
    })();
    if let Err(e) = &result {
        fabric.abort(e);
    }
    result
}

fn worker_main(test: &str) -> ! {
    std::fs::write(std::env::var("W_RAN").unwrap(), test).unwrap();
    let me = WorkerId(env_u64("W_ME").unwrap() as u32);
    let ckpt = PathBuf::from(std::env::var("W_CKPT").unwrap());
    match run_worker(me, &ckpt) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("worker {me:?}: {e}");
            std::process::exit(2);
        }
    }
}

/// the name libtest gives the test function `func`: its module path below the test crate's
/// root, then the function. the root module is the crate itself, so a test file built as its
/// own target names `func` bare and the same file built as a module of a larger target
/// prefixes the module path.
fn exact_name(func: &str) -> String {
    match module_path!().split_once("::") {
        Some((_, below_root)) => format!("{below_root}::{func}"),
        None => func.to_string(),
    }
}

/// the record a worker child writes on entering its test: a sibling of the scratch directory,
/// so the directory holds only what the protocol publishes.
fn ran_record(dir: &Path, me: u32) -> PathBuf {
    PathBuf::from(format!("{}.ran{me}", dir.display()))
}

/// the child ran the test function `test`: its record names it. a child whose filter matched
/// no test exits zero without a record.
fn assert_ran(record: &Path, test: &str, me: u32) {
    let ran = std::fs::read_to_string(record).unwrap_or_default();
    let _ = std::fs::remove_file(record);
    assert_eq!(
        ran, test,
        "worker {me} was launched for {test} but ran {ran:?}; its test filter {} selects a \
         different test or none",
        exact_name(test)
    );
}

fn in_worker_role() -> bool {
    std::env::var(ROLE).is_ok()
}

struct Outcome {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run_arm(test: &str, dir: &Path, extra: &[(&str, String)]) -> Vec<Outcome> {
    let coord = dir.join("rendezvous");
    let ckpt = dir.join("ckpt.h5");
    let mut children: Vec<Child> = (0..WORKERS)
        .map(|w| {
            let _ = std::fs::remove_file(ran_record(dir, w));
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.args([exact_name(test).as_str(), "--exact", "--nocapture", "--test-threads=1"])
                .env(ROLE, "worker")
                .env("W_RAN", ran_record(dir, w))
                .env("W_ME", w.to_string())
                .env("W_COORD", &coord)
                .env("W_CKPT", &ckpt)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for (k, v) in extra {
                cmd.env(k, v);
            }
            cmd.spawn().unwrap()
        })
        .collect();
    let start = Instant::now();
    while children.iter_mut().any(|c| c.try_wait().unwrap().is_none()) {
        if start.elapsed() > WATCHDOG {
            for c in children.iter_mut() {
                let _ = c.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    children
        .iter_mut()
        .enumerate()
        .map(|(w, c)| {
            let status = c.wait().unwrap();
            assert_ran(&ran_record(dir, w as u32), test, w as u32);
            let (mut stdout, mut stderr) = (String::new(), String::new());
            c.stdout.take().unwrap().read_to_string(&mut stdout).unwrap();
            c.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
            Outcome {
                code: status.code(),
                stdout,
                stderr,
            }
        })
        .collect()
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("symbi_ckpt_audit_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cost_field(stdout: &str, name: &str) -> u64 {
    let line = stdout
        .lines()
        .find(|l| l.contains("COST "))
        .unwrap_or_else(|| panic!("no cost line in the worker's stdout:\n{stdout}"));
    let line = &line[line.find("COST ").unwrap()..];
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
        .expect("the field")
        .parse()
        .unwrap()
}

/// the files in `dir` other than the rendezvous record
fn published(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|f| f != "rendezvous")
        .collect()
}

#[test]
fn equal_copies_pass_the_audit_and_the_checkpoint_is_published() {
    if in_worker_role() {
        worker_main("equal_copies_pass_the_audit_and_the_checkpoint_is_published");
    }
    let dir = scratch("clean");
    let outcomes = run_arm("equal_copies_pass_the_audit_and_the_checkpoint_is_published", &dir, &[]);
    for (w, o) in outcomes.iter().enumerate() {
        assert_eq!(o.code, Some(0), "worker {w} failed:\n{}", o.stderr);
    }
    assert!(dir.join("ckpt.h5").exists(), "no checkpoint published");
    // one x cut of 32 faces crosses workers: the lower worker sends it, the upper compares
    // it; the y cut inside each worker is compared in place: 16 faces per tile pair
    assert_eq!(cost_field(&outcomes[0].stdout, "faces"), 16);
    assert_eq!(cost_field(&outcomes[1].stdout, "faces"), 32 + 16);
    assert_eq!(cost_field(&outcomes[0].stdout, "bytes"), 32 * 8);
    assert_eq!(cost_field(&outcomes[1].stdout, "bytes"), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// one face off by a unit in the last place on the upper worker's opening copy of the x
/// cut: the audit names it, every worker exits nonzero, and nothing is on disk
#[test]
fn a_one_bit_face_disagreement_aborts_every_worker_and_publishes_nothing() {
    if in_worker_role() {
        worker_main("a_one_bit_face_disagreement_aborts_every_worker_and_publishes_nothing");
    }
    let dir = scratch("perturbed");
    let outcomes = run_arm(
        "a_one_bit_face_disagreement_aborts_every_worker_and_publishes_nothing",
        &dir,
        &[("W_PERTURB_WORKER", "1".into())],
    );
    for (w, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {w} exited zero:\n{}", o.stderr);
    }
    assert!(
        outcomes[1].stderr.contains("shared-face mismatch on bface0"),
        "{}",
        outcomes[1].stderr
    );
    assert!(
        outcomes[0].stderr.contains("aborted") || outcomes[0].stderr.contains("mismatch"),
        "worker 0 did not learn of the abort:\n{}",
        outcomes[0].stderr
    );
    assert_eq!(published(&dir), Vec::<String>::new(), "something was published");
    assert!(
        outcomes.iter().all(|o| !o.stdout.contains("COST ")),
        "a worker reached the checkpoint"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
