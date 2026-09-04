// =============================================================================
// guard_ledger_ownership.rs
//
// the run-owned FOFC guard diagnostics returned by the backend driver: proof that
// the accepted guard acts a run hands back are that run's own, extracted from the
// run's own scope, and cannot be contaminated by another run. the driving setup
// is a violently magnetized near-horizon kerr-schild box whose recovery floods
// the troubled-cell flag, so the returned guard totals are nonzero and the
// isolation claims are discriminating.
//
// gates:
// - two sequential runs on one thread report their own accepted guard acts;
// - two concurrent runs on separate threads report their own.
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

/// two sequential runs on one thread report their own accepted guard acts: the
/// second run's scope open clears the book, so identical setups return identical
/// totals rather than the second inheriting the first.
#[test]
fn sequential_runs_report_their_own_guards() {
    let a = run_once();
    let b = run_once();
    assert!(
        a.guards.troubled_cells.total > 0,
        "setup never tripped a guard; the ownership gate is vacuous"
    );
    assert_eq!(a.guards.troubled_cells.total, b.guards.troubled_cells.total);
    assert_eq!(a.guards.frozen_cells.total, b.guards.frozen_cells.total);
    assert_eq!(
        a.guards.troubled_cells.inside_horizon,
        b.guards.troubled_cells.inside_horizon
    );
    assert!(a.guards.frozen_cells.inside_horizon <= a.guards.frozen_cells.total);
}

/// two concurrent runs on separate threads report their own accepted guard acts:
/// the thread-local guard book keeps the runs from colliding.
#[test]
fn concurrent_runs_report_their_own_guards() {
    let solo = run_once();
    assert!(
        solo.guards.troubled_cells.total > 0,
        "setup never tripped a guard; the ownership gate is vacuous"
    );
    let left = std::thread::spawn(run_once);
    let right = std::thread::spawn(run_once);
    let a = left.join().expect("left run panicked");
    let b = right.join().expect("right run panicked");
    assert_eq!(
        a.guards.troubled_cells.total,
        solo.guards.troubled_cells.total
    );
    assert_eq!(
        b.guards.troubled_cells.total,
        solo.guards.troubled_cells.total
    );
    assert_eq!(a.guards.frozen_cells.total, b.guards.frozen_cells.total);
}
