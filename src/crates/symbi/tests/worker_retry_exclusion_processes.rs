// =============================================================================
// worker_retry_exclusion_processes.rs
//
// the retry exclusion is part of the retry state every worker of a session
// agrees on before the first step, beside the retry guard: workers configured
// with different policies refuse to start, each naming the disagreement, and
// workers configured alike run. the fixture is a flat 2D newtonian MHD box cut
// once across two workers; the workers admit flat charts alone, where the
// policy counts freeze acts toward the freeze-streak halt.
//
// run: cargo test -p symbi --test worker_retry_exclusion_processes
// =============================================================================

use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symbi::prelude::SimSubstrate;
use symbi::sim::decomp::{LocalCopy, Partition, Topology, plan_schema, unflatten};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::RetryExclusion;
use symbi_algebra::Tensor;
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, WorkerId};
use symbi_geometry::Cartesian;
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

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const N: usize = 16;
const H: f64 = 1.0 / N as f64;
const STEPS: u64 = 2;
const DEADLINE: Duration = Duration::from_secs(20);
const WATCHDOG: Duration = Duration::from_secs(120);

fn policy(name: &str) -> RetryExclusion {
    match name {
        "none" => RetryExclusion::None,
        "sphere" => RetryExclusion::LegacyEuclideanSphere { radius: 0.25 },
        other => panic!("unknown policy {other}"),
    }
}

fn partition() -> Partition<2> {
    Partition::explicit([N, N], [vec![N / 2], Vec::new()]).unwrap()
}

fn tile(flat: usize) -> Sim {
    let p = partition();
    let ext = p.tile_extents(unflatten(flat, p.counts()));
    Sim::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([H, H])
        .origin([ext[0].0 as f64 * H, ext[1].0 as f64 * H])
        .boundaries(Boundaries([
            [BoundaryType::CoarseFine; 2],
            [BoundaryType::Periodic; 2],
        ]))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[_x, y]| {
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0),
                    Tensor::new([0.1 * (6.0 * y).sin(), 0.0, 0.0]),
                    Pressure(1.0),
                ),
                Tensor::new([0.0, 0.2, 0.0]),
            )
        })
        .seed_faces_uniform([0.0, 0.2])
        .build()
}

fn run_worker(me: WorkerId) -> Result<(), WorkerError> {
    let p = partition();
    let n = p.n_tiles();
    let placement = Placement::new(2, vec![WorkerId(0), WorkerId(1)]).expect("placement");
    let mine: Vec<usize> = (0..n)
        .filter(|&f| placement.owner_of(TileId(f as u32)) == me)
        .collect();
    let mut built: Vec<Sim> = mine.iter().map(|&f| tile(f)).collect();
    // the run's configured policy on every store; W_POLICY_W1 gives worker 1 its own
    let name = if me == WorkerId(1) {
        std::env::var("W_POLICY_W1").ok()
    } else {
        None
    }
    .unwrap_or_else(|| std::env::var("W_POLICY").unwrap());
    for s in built.iter_mut() {
        s.retry_exclusion = policy(&name);
    }
    let kernels: Vec<_> = built.iter().map(|s| s.substrate()).collect();
    let tiles: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let schema = plan_schema(&*built[0]);
    let plan = ExchangePlan::compile(
        &p,
        &Topology::wrapping([true; 2]),
        &schema,
        built[0].geom.ng,
    )
    .expect("plan");
    let exchange = PlanExchange::new(&plan, &placement, me, PhaseTable::flat(2).unwrap());
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().unwrap(),
        announce: Some(PathBuf::from(std::env::var("W_COORD").unwrap())),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0x0D7C,
            build_id: "worker_retry_exclusion_processes".into(),
            config_digest: Digest(1),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: PhaseTable::flat(2).unwrap().digest(),
            ng: built[0].geom.ng as u32,
            workers: 2,
            block_credit: 4096,
        },
        startup: DEADLINE,
    };
    let max_payload = (exchange.lens().iter().max().copied().unwrap_or(0) * 8).max(64);
    let (link, session) = connect(&r, max_payload)?;
    let mut fabric = exchange.fabric(link, session);
    let cfg = WorkerConfig {
        regime: RegimeKind::of::<f64, 2, NewtonianMhd>(),
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 10.0,
        max_steps: STEPS,
        deadline: DEADLINE,
        injection: Injection::default(),
        audit_every_point: false,
    };
    let mut stores: Vec<&mut FieldStore<2, 3, HostMemory, f64>> =
        built.iter_mut().map(|s| &mut **s).collect();
    let kernel_refs: Vec<_> = kernels.iter().collect();
    let report = evolve_worker(
        &tiles,
        &mut stores,
        &kernel_refs,
        &vec![0i32; n],
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )?;
    fabric.finish(DEADLINE)?;
    println!("REPORT steps={}", report.steps);
    Ok(())
}

fn worker_main() -> ! {
    let me = WorkerId(std::env::var("W_ME").unwrap().parse::<u32>().unwrap());
    match run_worker(me) {
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

/// each worker's exit code, stdout and stderr
fn run_arm(test: &str, tag: &str, env: &[(&str, &str)]) -> Vec<(Option<i32>, String, String)> {
    let dir = std::env::temp_dir().join(format!(
        "symbi_retry_exclusion_{}_{test}_{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let coord = dir.join("rendezvous");
    let mut children: Vec<Child> = (0..2)
        .map(|w| {
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(ROLE, "worker")
                .env("W_ME", w.to_string())
                .env("W_COORD", &coord)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for (k, v) in env {
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
    let outcomes = children
        .iter_mut()
        .map(|c| {
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
            (status.code(), stdout, stderr)
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    outcomes
}

#[test]
fn workers_configured_with_different_policies_refuse_to_start() {
    if in_worker_role() {
        worker_main();
    }
    let test = "workers_configured_with_different_policies_refuse_to_start";
    for policy in ["none", "sphere"] {
        for (w, (code, stdout, stderr)) in run_arm(test, policy, &[("W_POLICY", policy)])
            .iter()
            .enumerate()
        {
            assert_eq!(
                *code,
                Some(0),
                "{policy}: worker {w} of an agreeing session failed:\n{stderr}"
            );
            assert!(
                stdout.contains(&format!("REPORT steps={STEPS}")),
                "{policy}: worker {w} did not run its steps"
            );
        }
    }
    for (w, (code, _, stderr)) in run_arm(
        test,
        "split",
        &[("W_POLICY", "sphere"), ("W_POLICY_W1", "none")],
    )
    .iter()
    .enumerate()
    {
        assert_ne!(
            *code,
            Some(0),
            "worker {w} ran a session whose workers carry different policies"
        );
        assert!(
            stderr.contains("retry exclusion differs across the workers"),
            "worker {w} failed for another reason:\n{stderr}"
        );
    }
}
