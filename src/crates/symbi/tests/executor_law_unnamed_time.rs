// =============================================================================
// executor_law_unnamed_time.rs
//
// the no-unnamed-time executor law: the sum of the per-phase profiler
// accumulators (`symbi_sim::driver::prof`) must account for nearly all of the
// step loop's wall time. a dispatch or reduction added to the loop without a
// `prof` wrapper is invisible to every profile — an unwrapped `body_feedback`
// can silently swallow a third of a run's wall time — and fails this law instead.
//
// the profiler is env-gated through a process-global OnceLock, so this binary
// holds ONLY this law (the flag must be set before the first `prof` call).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_sim::driver::{report_profile, reset_profile};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 64;
const L: f64 = 1.0;

#[test]
fn e1_every_step_phase_is_named() {
    // the profiler latches SYMBI_PROFILE once per process; set it before any
    // prof call. sound here because this binary runs only this test.
    unsafe { std::env::set_var("SYMBI_PROFILE", "1") };

    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    // a central accreting body drives the widest loop surface: cfl, flux,
    // godunov, source, ghost fill, body source, body feedback, body motion.
    let bodies = BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.0, 0.0]),
        Tensor::zeros(),
        1.0,
        0.1,
        0.2,
        0.5,
        0.0,
        0.15,
    ));
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(bodies);
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);

    // warmup: jit builds and lazy allocations happen outside the measured window.
    evolve(&mut sim, &sub, 0.02).expect("warmup evolve failed");
    let warm_iter = sim.iteration;

    reset_profile();
    let t0 = std::time::Instant::now();
    evolve(&mut sim, &sub, 0.20).expect("measured evolve failed");
    let wall = t0.elapsed().as_secs_f64();
    let steps = sim.iteration - warm_iter;
    assert!(steps >= 4, "measurement window too short ({steps} steps)");

    let phases = report_profile();
    let named: f64 = phases.iter().map(|(_, ms)| ms / 1e3).sum();
    let coverage = named / wall;
    println!(
        "e1: {:.1}% of {wall:.3}s named across {} phases, {steps} steps",
        coverage * 100.0,
        phases.len()
    );
    for (name, ms) in &phases {
        println!("  {name}: {ms:.2} ms");
    }
    assert!(
        coverage >= 0.95,
        "e1 violated: only {:.1}% of the step loop's wall time is inside named prof \
         phases — something in the loop runs unwrapped and is invisible to every profile",
        coverage * 100.0,
    );
}
