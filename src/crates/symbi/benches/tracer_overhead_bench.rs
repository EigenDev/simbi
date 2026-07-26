// =============================================================================
// tracer_overhead_bench.rs
//
// cpu throughput comparison for the same two-dimensional kelvin-helmholtz
// evolution without tracers, with ito-2 tracers, and with ito-3 tracers.
// every mode uses the same grid, initial state, warmup interval, timed interval,
// solver, and particle seeding rule.
//
// usage:
//  cargo bench -p symbi --bench tracer_overhead_bench
//  SYMBI_BENCH_N=256 SYMBI_BENCH_TRACERS=262144 \
//    cargo bench -p symbi --bench tracer_overhead_bench
// =============================================================================

use std::f64::consts::PI;
use std::time::Instant;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::mass_transport::ItoOrder;
use symbi_sim::tracers::{ContinuousTracerSet, seed_mass_weighted};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const WARM_TIME: f64 = 0.02;
const TIMED_TIME: f64 = 0.10;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn environment_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn build(n: usize) -> Sim {
    let dx = 1.0 / n as f64;
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("tracer benchmark simulation construction failed")
        .set_initial(|[x, y]| {
            let smooth = 0.02;
            let inside =
                0.5 * (1.0 + ((y - 0.25) / smooth).tanh() * ((0.75 - y) / smooth).tanh());
            Prim {
                rho: 1.0 + inside,
                vel: Tensor::new([-0.5 + inside, 0.01 * (4.0 * PI * x).sin()]),
                pre: 2.5,
            }
        })
        .build()
}

fn run(n: usize, particles: usize, order: Option<ItoOrder>) -> (f64, u64) {
    let mut sim = build(n);
    if let Some(order) = order {
        let seed = seed_mass_weighted(&sim, particles);
        sim.continuous_tracers =
            Some(ContinuousTracerSet::from_discrete(&seed, order).unwrap());
    }
    let kernels = Kern::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::Hllc)
        .expect("hllc is valid for newtonian hydro");
    evolve(&mut sim, &kernels, WARM_TIME).expect("tracer benchmark warmup failed");
    let first_iteration = sim.iteration;
    let start = Instant::now();
    evolve(&mut sim, &kernels, WARM_TIME + TIMED_TIME)
        .expect("tracer benchmark timed evolution failed");
    (start.elapsed().as_secs_f64(), sim.iteration - first_iteration)
}

fn main() {
    let n = environment_usize("SYMBI_BENCH_N", 128);
    let particles = environment_usize("SYMBI_BENCH_TRACERS", n * n);
    let cells = n * n;
    let (plain_time, plain_steps) = run(n, particles, None);
    let (ito2_time, ito2_steps) = run(n, particles, Some(ItoOrder::Two));
    let (ito3_time, ito3_steps) = run(n, particles, Some(ItoOrder::Three));
    assert_eq!(ito2_steps, plain_steps, "ito-2 changed the hydro step count");
    assert_eq!(ito3_steps, plain_steps, "ito-3 changed the hydro step count");

    let zone_cycles = cells as f64 * plain_steps as f64;
    let particle_updates = particles as f64 * plain_steps as f64;
    let report_traced = |name: &str, elapsed: f64| {
        println!(
            "{name:<10} {elapsed:>8.4} s  {:>8.2} Mzcps  {:>8.2} Mparticle-updates/s  \
             {:>8.2} incremental ns/particle-update  overhead={:>7.2}%",
            zone_cycles / elapsed / 1.0e6,
            particle_updates / elapsed / 1.0e6,
            (elapsed - plain_time) * 1.0e9 / particle_updates,
            100.0 * (elapsed / plain_time - 1.0),
        );
    };

    println!(
        "=== continuous tracer overhead: {n}^2 cells, {particles} particles, \
         {plain_steps} rk2 steps ==="
    );
    println!(
        "{:<10} {:>8.4} s  {:>8.2} Mzcps",
        "tracerless",
        plain_time,
        zone_cycles / plain_time / 1.0e6,
    );
    report_traced("ito-2", ito2_time);
    report_traced("ito-3", ito3_time);
}
