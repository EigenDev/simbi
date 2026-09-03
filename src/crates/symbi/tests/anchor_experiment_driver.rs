// =============================================================================
// anchor_experiment_driver.rs
//
// the projection-anchor experiment's driver transaction, pinned on a real
// evolve run: a violently magnetized near-horizon kerr-schild box whose
// first step trips the admissible-boundary projection, driven through the
// tile driver the production backend uses.
//
// the run goes through `Hierarchy::single(...).evolve_with_callback`, which is
// exactly the path `run_simulation` takes for every grid — the single-grid case
// is a one-level hierarchy, and the shared `evolve_tiles` loop owns the session
// and the accepted-step transaction. a transaction wired only into the
// standalone `sim::evolve` driver would leave the session unopened here and the
// projection's first `record_pass` would panic, so this pins that the backend's
// driver carries the hooks.
//
// gates:
// - the accepted-first fire is stamped with the post-step clock of the state
//   the accepted step produced: its (time, iteration) equal the simulation
//   clock the first post-step callback observes, with iteration = 1 and
//   time > 0 (the step-entry clock is (0, 0.0) and can never appear);
// - the run session opened by the driver closes at return and leaves the
//   totals queryable;
// - the intervention ledger bounds the injected ledger: every downstream
//   shu-osher propagation weight lies in (0, 1].
//
// the projection firing inside the first accepted step is asserted as a
// precondition, so a setup that stops exercising the tier reports its own
// vacuity instead of passing empty.
// =============================================================================

use std::f64::consts::PI;
use symbi_hydro::quantity::{Density, Pressure};

use std::ops::ControlFlow;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 6;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const MASS: f64 = 0.8;
const B0: f64 = 8.0;
const X_LO: f64 = 0.9;

type KerrSim =
    SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;

/// a truncation-style contrast: a dense, cold, swirling slab against a
/// magnetized near-vacuum, all of it strongly curved. the sharp candidate at
/// the slab edge leaves the admissible set, so the fallback ladder descends
/// to the boundary projection.
fn violent_prim(x: f64, y: f64, z: f64) -> MhdPrim<f64, 3> {
    let s = 2.0 * PI;
    let dense = y < 1.4;
    let rho = if dense { 1.0e2 } else { 1.0e-6 };
    let vel = if dense {
        Tensor::new([
            0.4 * (s * y).sin(),
            0.4 * (s * z).sin(),
            0.4 * (s * x).sin(),
        ])
    } else {
        Tensor::new([0.0, 0.0, 0.0])
    };
    MhdPrim::new(
        Prim::adiabatic(Density(rho), vel, Pressure(1.0e-8)),
        Tensor::new([B0, 0.0, 0.0]),
    )
}

fn kerr_sim() -> KerrSim {
    let dx = 1.0 / N as f64;
    let metric = SchwarzschildKSCartesian { mass: MASS };
    KerrSim::build(Rmhd, IdealGas { gamma: GAMMA }, metric)
        .cells([N; 3])
        .origin([X_LO; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("kerr sim")
        .set_initial(|[x, y, z]| violent_prim(x, y, z))
        .seed_faces(|axis, [x, y, z]| {
            if axis == 0 {
                B0 / symbi_geometry::Metric::<f64, 3>::sqrt_det_gamma(
                    &metric,
                    Tensor::new([x, y, z]),
                )
            } else {
                0.0
            }
        })
        .build()
}

#[test]
fn the_accepted_first_fire_carries_the_post_step_clock() {
    // the arm must be named before any experiment gate caches the
    // environment; this test binary is its own process.
    unsafe { std::env::set_var("SIMBI_ANCHOR_EXPERIMENT", "eulerian_rebuilt") };

    let sim = kerr_sim();
    let kern =
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);

    // the single-grid run is a one-level hierarchy driven through the shared
    // tile driver — the exact path the backend takes, where the experiment
    // session opens and the step transaction commits.
    let mut hier = Hierarchy::single(sim, kern);

    // the post-step clock of the first accepted step, and the accepted fired
    // passes visible at that boundary — observed through the per-step
    // callback, which the driver invokes after the canonical clock advance.
    let mut first_post_step: Option<(f64, u64, u64)> = None;
    hier.evolve_with_callback(2.0e-2, 1, |h| {
        if first_post_step.is_none() {
            let (_, accepted) = symbi::regimes::projection_experiment::experiment_report();
            let st = &h.levels[0].state;
            first_post_step = Some((st.time, st.iteration, accepted.passes_fired));
        }
        ControlFlow::Continue(())
    })
    .expect("the violent box must survive to t_final");

    let (time_1, iteration_1, fired_by_step_1) =
        first_post_step.expect("the run never completed a step");
    let (attempted_first, accepted_first) =
        symbi::regimes::projection_experiment::experiment_first_report();
    assert!(
        attempted_first.is_some(),
        "setup never tripped the boundary projection; the timestamp pin is vacuous"
    );
    assert!(
        fired_by_step_1 > 0,
        "the projection must fire inside the first accepted step for the stamp \
         to be discriminating; re-sharpen the setup"
    );

    // the stamp is the clock of the state the accepted step produced: the
    // step-entry clock is (0, 0.0), so equality with the first post-step
    // observation is exact and one-sided.
    let first = accepted_first.expect("a fired accepted step records its first fire");
    assert_eq!(first.iteration, iteration_1);
    assert_eq!(first.time, time_1);
    assert_eq!(iteration_1, 1);
    assert!(time_1 > 0.0);

    // the session the driver opened has closed; the totals stay queryable
    // and the injected ledger is the intervention ledger scaled by convex
    // weights in (0, 1].
    let (attempted, accepted) = symbi::regimes::projection_experiment::experiment_report();
    assert!(
        accepted.passes > 0,
        "the accepted-step commit never fired through the hierarchy driver; \
         the transaction hooks are not on the backend's path"
    );
    assert!(attempted.passes >= accepted.passes);
    assert!(accepted.injected_mass.abs <= accepted.intervention_mass.abs);
    assert!(accepted.injected_energy_segment.abs <= accepted.intervention_energy_segment.abs);
    assert!(accepted.injected_energy_raise.abs <= accepted.intervention_energy_raise.abs);
}
