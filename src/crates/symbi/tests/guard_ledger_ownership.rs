// =============================================================================
// guard_ledger_ownership.rs
//
// the run-owned FOFC guard diagnostics returned by the backend driver: proof that
// the accepted guard acts a run hands back are that run's own, extracted from the
// run's own scope, and cannot be contaminated by another run. the driving setup
// is a cold atmosphere in prescribed mach-173 radial infall onto an excised
// cartesian kerr-schild hole: the supersonic stream drives the high-order
// recovery out of the physical set at the excision rim from the first step,
// inside the horizon, so the returned guard totals and their horizon subsets
// are nonzero and the isolation claims are discriminating. the horizon
// reporting radius is 2M.
//
// gates:
// - two sequential runs on one thread report their own accepted guard acts;
// - two concurrent runs on separate threads report their own.
// =============================================================================

use std::ops::ControlFlow;

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::run_diagnostics::RunDiagnostics;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 32;
const L: f64 = 1.2;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 0.3;
const R_EXC: f64 = 0.35;
const V_INFALL: f64 = 0.2;
const T_FINAL: f64 = 5.0e-4;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn run_once() -> RunDiagnostics {
    symbi::regimes::fofc::fofc_set_horizon_radius(2.0 * MASS);
    let sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, SchwarzschildKSCartesian { mass: MASS })
        .cells([N; 2])
        .spacing([DX; 2])
        .origin([-L; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Rk2)
        .cfl(CFL)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            let r = (x[0] * x[0] + x[1] * x[1]).sqrt().max(1.0e-12);
            Prim::adiabatic(Density(1.0), Tensor::new([-V_INFALL * x[0] / r, -V_INFALL * x[1] / r]), Pressure(1.0e-6))
        })
        .build();
    let kern = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0);
    let mut hier = Hierarchy::single(sim, kern);
    hier.evolve_with_callback(T_FINAL, 1, |_h| ControlFlow::Continue(()))
        .expect("the infall must survive to t_final")
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
    assert!(
        a.guards.troubled_cells.inside_horizon > 0,
        "no guard fired inside the horizon; the horizon subset comparison is vacuous"
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
