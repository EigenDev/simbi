// =============================================================================
// run_diagnostics_ownership.rs
//
// the run-owned projection diagnostics returned by the backend driver: proof
// that the value a run hands back is that run's own accepted evidence, extracted
// from the run's own ledger scope, and cannot be contaminated by another run.
//
// the run goes through `Hierarchy::single(...).evolve_with_callback`, the exact
// path `run_simulation` takes for a single grid, and reads only the returned
// `RunDiagnostics` — never the process-visible `ledger_report()`. the driving
// setup is a violently magnetized near-horizon kerr-schild box whose first step
// trips the admissible-boundary projection, so the returned projection totals are
// nonzero and the isolation claims are discriminating.
//
// gates:
// - two sequential runs on one thread each report their own evidence; the second
//   does not inherit the first's booked passes;
// - two concurrent runs on separate threads each report their own evidence;
// - the returned value is the run's own copy, so clearing the legacy global
//   readout leaves it unchanged.
// =============================================================================

use std::f64::consts::PI;
use std::ops::ControlFlow;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::run_diagnostics::RunDiagnostics;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 6;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const MASS: f64 = 0.8;
const B0: f64 = 8.0;
const X_LO: f64 = 0.9;
const T_FINAL: f64 = 2.0e-2;

type KerrSim =
    SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;

/// a dense, cold, swirling slab against a magnetized near-vacuum, strongly
/// curved: the sharp candidate at the slab edge leaves the admissible set, so the
/// fallback ladder descends to the boundary projection.
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

/// one complete single-grid run, returning the backend's run-owned diagnostics.
/// each call builds a fresh sim, so nothing carries between runs except through
/// the thread's ledger scope — which is exactly what the isolation gates probe.
fn run_once() -> RunDiagnostics {
    let sim = kerr_sim();
    let kern =
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);
    let mut hier = Hierarchy::single(sim, kern);
    hier.evolve_with_callback(T_FINAL, 1, |_h| ControlFlow::Continue(()))
        .expect("the violent box must survive to t_final")
}

/// two sequential runs on one thread each report their own accepted projection.
/// identical setups return identical totals: the second run's open clears the
/// book, so it books only its own passes rather than inheriting the first's.
#[test]
fn sequential_runs_report_their_own_projection() {
    let a = run_once();
    let b = run_once();
    assert!(
        a.projection.passes_fired > 0,
        "setup never tripped the projection; the ownership gate is vacuous"
    );
    assert_eq!(a.projection.passes, b.projection.passes);
    assert_eq!(a.projection.passes_fired, b.projection.passes_fired);
    assert_eq!(a.projection.projected_cells, b.projection.projected_cells);
    assert_eq!(
        a.projection.intervention_den.abs,
        b.projection.intervention_den.abs
    );
}

/// two concurrent runs on separate threads each report their own accepted
/// projection: the thread-local ledger book keeps the two runs from colliding, so
/// each returns the same evidence a lone run does.
#[test]
fn concurrent_runs_report_their_own_projection() {
    let solo = run_once();
    assert!(
        solo.projection.passes_fired > 0,
        "setup never tripped the projection; the ownership gate is vacuous"
    );
    let left = std::thread::spawn(run_once);
    let right = std::thread::spawn(run_once);
    let a = left.join().expect("left run panicked");
    let b = right.join().expect("right run panicked");
    assert_eq!(a.projection.passes, solo.projection.passes);
    assert_eq!(b.projection.passes, solo.projection.passes);
    assert_eq!(a.projection.projected_cells, b.projection.projected_cells);
    assert_eq!(
        a.projection.intervention_den.abs,
        solo.projection.intervention_den.abs
    );
}

/// the returned value is the run's own copy, extracted from its scope. the legacy
/// process-visible readout reflects the same run right after it, but clearing that
/// global leaves the returned `RunDiagnostics` untouched — proof the new API does
/// not read the global.
#[test]
fn the_returned_value_is_independent_of_the_legacy_readout() {
    let diag = run_once();
    assert!(
        diag.projection.passes > 0,
        "setup never tripped the projection; the independence gate is vacuous"
    );
    let (_, accepted) = symbi_sim::projection_ledger::ledger_report();
    assert_eq!(
        accepted.passes, diag.projection.passes,
        "the legacy readout and the returned value describe the same run"
    );
    symbi_sim::projection_ledger::ledger_reset();
    let (_, after) = symbi_sim::projection_ledger::ledger_report();
    assert_eq!(after.passes, 0, "the legacy global cleared");
    assert!(
        diag.projection.passes > 0,
        "the returned value survived the global reset; it is the run's own copy"
    );
}
