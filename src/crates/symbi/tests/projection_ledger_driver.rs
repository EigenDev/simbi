// =============================================================================
// projection_ledger_driver.rs
//
// the projection ledger's step transaction, pinned on a real evolve run: a
// violently magnetized near-horizon kerr-schild box whose first step trips the
// admissible-boundary projection, driven through the tile driver the production
// backend uses.
//
// the run goes through `Hierarchy::single(...).evolve_with_callback`, which is
// exactly the path `run_simulation` takes for every grid — the single-grid case
// is a one-level hierarchy, and the shared `evolve_tiles` loop opens the ledger
// scope and owns the accepted-step transaction. a transaction wired only into
// the standalone `sim::evolve` driver would leave the scope unopened here and
// the projection's first `record` would panic, so this pins that the backend's
// driver carries the hooks.
//
// gates:
// - the accepted ledger booked (`accepted.passes > 0`), and the count is
//   visible at the first post-step callback, so the accepted-step commit ran
//   through the hierarchy rather than a driver the backend never calls;
// - the injected ledger never exceeds the intervention ledger, since every
//   downstream shu-osher propagation weight lies in (0, 1].
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
fn the_projection_ledger_books_through_the_backend_driver() {
    let sim = kerr_sim();
    let kern =
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);

    // the single-grid run is a one-level hierarchy driven through the shared
    // tile driver — the exact path the backend takes, where the projection
    // ledger scope opens and the step transaction commits.
    let mut hier = Hierarchy::single(sim, kern);

    // the accepted-fired passes visible at the first post-step boundary,
    // observed through the per-step callback the driver invokes after the
    // canonical clock advance: a nonzero count proves the accepted-step commit
    // ran through the hierarchy, not a driver the backend never calls.
    let mut fired_by_step_1: Option<u64> = None;
    hier.evolve_with_callback(2.0e-2, 1, |_h| {
        if fired_by_step_1.is_none() {
            let (_, accepted) = symbi_sim::projection_ledger::ledger_report();
            fired_by_step_1 = Some(accepted.passes_fired);
        }
        ControlFlow::Continue(())
    })
    .expect("the violent box must survive to t_final");

    assert!(
        fired_by_step_1.expect("the run never completed a step") > 0,
        "the projection never fired inside the first accepted step; re-sharpen the setup so \
         the ledger booking is discriminating"
    );

    // the scope the driver opened has closed; the totals stay queryable. the
    // accepted ledger booked (the commit fired through the hierarchy), and the
    // injected ledger is the intervention ledger scaled by convex weights in
    // (0, 1], so it never exceeds it.
    let (attempted, accepted) = symbi_sim::projection_ledger::ledger_report();
    assert!(
        accepted.passes > 0,
        "the accepted-step commit never fired through the hierarchy driver; \
         the transaction hooks are not on the backend's path"
    );
    assert!(attempted.passes >= accepted.passes);
    assert!(accepted.injected_den.abs <= accepted.intervention_den.abs);
    assert!(accepted.injected_nrg.abs <= accepted.intervention_nrg.abs);
}
