// =============================================================================
// dye_overhead_bench.rs
//
// cpu throughput comparison for the same two-dimensional kelvin-helmholtz
// evolution without a passive scalar and with one. both modes use the same grid,
// initial state, warmup interval, timed interval, solver, and step count, so the
// difference is the dye stage alone.
//
// the dye costs one extra godunov-form pass per substage. this measures that
// pass against the gas it rides along with, which is the number that decides
// whether the dye flux is worth materializing unconditionally or only when a
// refinement hierarchy needs it for reflux.
//
// usage:
//  cargo bench -p symbi --bench dye_overhead_bench
//  SYMBI_BENCH_N=256 cargo bench -p symbi --bench dye_overhead_bench
//  SYMBI_PROFILE=1 cargo bench -p symbi --bench dye_overhead_bench   # per-phase
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
        .expect("dye benchmark simulation construction failed")
        .set_initial(|[x, y]| {
            let smooth = 0.02;
            let inside = 0.5 * (1.0 + ((y - 0.25) / smooth).tanh() * ((0.75 - y) / smooth).tanh());
            Prim {
                rho: 1.0 + inside,
                vel: Tensor::new([-0.5 + inside, 0.01 * (4.0 * PI * x).sin()]),
                pre: 2.5,
            }
        })
        .build()
}

fn run(n: usize, dyed: bool) -> (f64, u64) {
    let mut sim = build(n);
    if dyed {
        sim = sim.with_passive_scalar().expect("dye allocation failed");
        // a sheared dye step: the upwind selection flips sign across the shear layer, so the
        // branchy half of the dye stage is exercised rather than a uniform field that upwinds
        // one way everywhere.
        let dx = 1.0 / n as f64;
        let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
        let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
        for c in sim.geom.allocated.clone().iter() {
            let y = (c[1] as f64 + 0.5) * dx;
            let chi = if (0.25..0.75).contains(&y) { 1.0 } else { 0.0 };
            let rho = *sim.fields.cons.den.view().at(c);
            cons_chi.view_mut().set(c, rho * chi);
            prim_chi.view_mut().set(c, chi);
        }
    }
    let kernels = Kern::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::Hllc)
        .expect("hllc is valid for newtonian hydro");
    evolve(&mut sim, &kernels, WARM_TIME).expect("dye benchmark warmup failed");
    let first_iteration = sim.iteration;
    let start = Instant::now();
    evolve(&mut sim, &kernels, WARM_TIME + TIMED_TIME).expect("dye benchmark timed evolution failed");
    (
        start.elapsed().as_secs_f64(),
        sim.iteration - first_iteration,
    )
}

fn main() {
    let n = environment_usize("SYMBI_BENCH_N", 128);
    let cells = n * n;
    let (plain_time, plain_steps) = run(n, false);
    let (dyed_time, dyed_steps) = run(n, true);
    assert_eq!(
        dyed_steps, plain_steps,
        "the dye changed the hydro step count; the two modes are no longer comparable"
    );

    let zone_cycles = cells as f64 * plain_steps as f64;
    println!("=== passive scalar overhead: {n}^2 cells, {plain_steps} rk2 steps ===");
    println!(
        "{:<10} {:>8.4} s  {:>8.2} Mzcps",
        "undyed",
        plain_time,
        zone_cycles / plain_time / 1.0e6,
    );
    println!(
        "{:<10} {:>8.4} s  {:>8.2} Mzcps  {:>8.2} incremental ns/zone-cycle  overhead={:>7.2}%",
        "dyed",
        dyed_time,
        zone_cycles / dyed_time / 1.0e6,
        (dyed_time - plain_time) * 1.0e9 / zone_cycles,
        100.0 * (dyed_time / plain_time - 1.0),
    );
}
