// =============================================================================
// wb_cost_gpu.rs
//
// measurement instrument, not a gate: the device wall-clock cost of the balanced
// (hydrostatic-departure) reconstruction relative to plain, on the same sealed
// plummer column the cpu probe times (sealed_column_unclamped::wb_cost_probe).
// the balanced arm pays a powf per face side (the isentrope ratio); how that
// prices out on device is a measurement, never an op count -- the cpu numbers
// (1.34-1.43x flux-stage at n = 65536, noise at n = 128) do not transfer, since
// the gpu amortizes transcendental latency differently and the kernel may be
// bandwidth-bound. one correctness assertion rides along: the balanced arm must
// hold the column stagnant on device exactly as it does on host.
//
// run on a cuda host (nvrtc, no nvcc needed):
//   cargo test -p symbi --release --features cuda --test wb_cost_gpu -- --ignored --nocapture
// =============================================================================
#![cfg(feature = "cuda")]

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the gravitating mass sits one domain width left of x = 0, so the gas at x
/// feels the mass at radius x + 1 and the domain covers r in [1, 2] with no
/// singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;
/// plummer softening of the body's field; the column is built from the same
/// softened potential, so state, gravity and balance are one field.
const SOFT: f64 = 1.0e-3;
/// wide enough that the face-flux kernels dominate the step on device; the
/// per-step launch overhead that buries the cpu ratio at small n is fixed-cost
/// on gpu too.
const PROBE_N: usize = 1 << 20;
const PROBE_STEPS: u64 = 200;

fn phi(x: f64) -> f64 {
    let r = x + G_OFFSET;
    -GM / (r * r + SOFT * SOFT).sqrt()
}

/// the isentropic atmosphere in hydrostatic balance against the softened field,
/// normalized to rho = 1 at the outer edge.
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(1.0);
    let rho = (a * (c - phi(x[0]))).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CudaSpace, UnifiedMemory>;
type Kset = AdiabaticSubstrateKernelSet<UnifiedMemory, f64, 1>;
type Hier =
    Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CudaSpace, UnifiedMemory, Kset>;

fn build(balanced: bool, n: usize) -> Hier {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([1.0 / n as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::HllcLm)
        .expect("solver/regime mismatch")
        .well_balanced_reconstruction(balanced);
    Hierarchy::single(sim, kernels).with_bodies(symbi_ib::BodyCollection::new().add(
        symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            1.0e-6,
            SOFT,
        ),
    ))
}

#[test]
#[ignore = "measurement instrument: device wall-clock of balanced vs plain reconstruction"]
fn wb_cost_probe_gpu() {
    let time = |balanced: bool| {
        // one warm-up build absorbs nvrtc render + alloc costs outside the timed
        // window; the short warm evolve settles the device clocks.
        let mut hier = build(balanced, PROBE_N);
        hier.evolve_steps(8).unwrap();
        let mut hier = build(balanced, PROBE_N);
        let t0 = std::time::Instant::now();
        hier.evolve_steps(PROBE_STEPS).unwrap();
        t0.elapsed().as_secs_f64()
    };
    let (t_plain, t_wb) = (time(false), time(true));
    println!(
        "device sealed column, {PROBE_STEPS} steps, n = {PROBE_N}: plain {t_plain:.3}s, \
         balanced {t_wb:.3}s, ratio {:.3}",
        t_wb / t_plain
    );

    // the correctness rider: the balanced arm holds the device column stagnant.
    let mut hier = build(true, 4096);
    hier.evolve_steps(400).unwrap();
    let st = &hier.levels[0].state;
    let vel = st.fields.prim.vel[0].view();
    let mut vmax = 0.0_f64;
    for ii in st.geom.interior.spaces[0].lo..st.geom.interior.spaces[0].hi {
        vmax = vmax.max(vel.at([ii]).abs());
    }
    assert!(
        vmax < 1.0e-11,
        "the balanced column drifts on device: max |v| = {vmax:.3e} after 400 steps \
         (host holds 3.5e-15)"
    );
}
