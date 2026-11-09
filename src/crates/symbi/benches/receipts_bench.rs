// =============================================================================
// receipts_bench.rs
//
// the cost of the conservation receipts on the lifted (r, theta) plane with the azimuthal
// velocity, a point mass and the outward-only outer wall, through the production single-hierarchy
// step on host memory. per grid size the same root steps run twice from the same state: once with
// the receipt scope open, so every stage books its wall transport and extremes, and once with it
// closed. each run takes warm-up steps, resets the profile, then times a window of steps. the
// report per run: end-to-end wall time per step, the receipts bucket with its wall-transport and
// stage-extremes parts, the time inside any profiled phase and the time in none, and the receipt
// scratch memory the store holds.
//
// run: SYMBI_PROFILE=1 cargo bench -p symbi --bench receipts_bench [-- 84x44,168x88,336x176 300]
// =============================================================================

use std::f64::consts::PI;

use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_xpu::{CpuSpace, HostMemory};

const R_LO: f64 = 0.01;
const R_HI: f64 = 4.0;
const WARMUP: u64 = 50;

type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

fn build(ni: usize, nj: usize) -> Hierarchy<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory, <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet> {
    let slope = (R_HI / R_LO).log10() / ni as f64;
    let dtheta = PI / nj as f64;
    let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
    let sim: Sim = Sim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Spherical)
        .cells([ni, nj])
        .origin([R_LO, 0.0])
        .spacing([(R_HI - R_LO) / ni as f64, dtheta])
        .coord_maps(Some([AxisMap::Log { start: R_LO, log_slope: slope }, AxisMap::Uniform { start: 0.0, dx: dtheta }]))
        .boundaries(Boundaries::per_axis(walls))
        .cfl(0.1)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("the plane")
        .set_initial(|x: [f64; 2]| {
            // a dense ring near r = 1 in an ambient at rest, rotating at the keplerian rate
            let (r, theta) = (x[0], x[1]);
            let ring = (-((r - 1.0).powi(2) + (theta - 0.5 * PI).powi(2)) / 0.05).exp();
            let rho = 1e-4 + ring;
            Prim::adiabatic(Density(rho), Tensor::new([0.0, 0.0, ring / (r * theta.sin()).sqrt().max(1e-3)]), Pressure(1e-4 / r + 0.01 * ring))
        })
        .build();
    let kernels = sim.substrate().with_outward_walls([[false, true], [false, false], [false, false]]);
    let mass = BodyCollection::new()
        .add(Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.0, 0.9 * R_LO).with_softening_kind(SofteningKind::Compact));
    let mut hier = Hierarchy::single(sim, kernels).with_bodies(mass);
    hier.prime();
    hier
}

fn steps(hier: &mut Hierarchy<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory, <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet>, n: u64) {
    for _ in 0..n {
        let dt = hier.root_cfl_dt();
        hier.step_root_with_dt(dt);
        assert!(hier.crash.is_none(), "the bench run stopped: {:?}", hier.crash);
    }
}

fn row(name: &str) -> f64 {
    symbi::sim::evolve::report_profile().iter().find(|(k, _)| *k == name).map_or(0.0, |(_, ms)| *ms)
}

fn main() {
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let grids: Vec<(usize, usize)> = args
        .get(1)
        .map_or("84x44,168x88,336x176", String::as_str)
        .split(',')
        .map(|g| {
            let (a, b) = g.split_once('x').expect("grids as NIxNJ");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let window: u64 = args.get(2).map_or(300, |s| s.parse().unwrap());
    assert!(std::env::var("SYMBI_PROFILE").is_ok(), "set SYMBI_PROFILE=1: the buckets are the measurement");
    println!(
        "revision {} · host memory · {} rayon threads · warm-up {WARMUP} steps, window {window} steps",
        option_env!("SIMBI_GIT_SHA").unwrap_or("unknown"),
        rayon::current_num_threads()
    );
    println!("| grid | receipts | ms/step | receipts ms/step | wall ms/step | extremes ms/step | covered ms/step | unattributed ms/step | receipt scratch |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (ni, nj) in grids {
        for receipts in [true, false] {
            let mut hier = build(ni, nj);
            let scope = receipts.then(symbi_sim::conservation_ledger::open_scope);
            steps(&mut hier, WARMUP);
            symbi::sim::evolve::reset_profile();
            let start = std::time::Instant::now();
            steps(&mut hier, window);
            let wall = start.elapsed().as_secs_f64() * 1e3;
            let covered = symbi::sim::evolve::report_covered();
            let per = |ms: f64| ms / window as f64;
            let scratch = symbi::regimes::wall_ledger::scratch_bytes(&hier.levels[0].state);
            println!(
                "| {ni}x{nj} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} kB |",
                if receipts { "on" } else { "off" },
                per(wall),
                per(row("receipts")),
                per(row("receipts_wall")),
                per(row("receipts_extremes")),
                per(covered),
                per(wall - covered),
                scratch / 1024
            );
            drop(scope);
        }
    }
}
