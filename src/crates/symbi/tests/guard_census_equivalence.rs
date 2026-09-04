// =============================================================================
// guard_census_equivalence.rs
//
// the run-owned guard ledger's ATTEMPTED totals equal the legacy process-global
// guard census, like-for-like. both count every substage's troubled and frozen
// cells, including substages later rejected, so the attempted totals reproduce
// the legacy `(fallback, freeze, fallback_inside_horizon, freeze_inside_horizon)`
// counters that gate acceptance thresholds. the accepted totals are a different,
// solution-restricted quantity and are deliberately NOT compared here.
//
// the legacy counters are process-global, so this is the only test in its binary;
// cargo runs each test binary as its own process, so nothing else races the
// counters. the setup is a violently magnetized near-horizon kerr-schild box that
// floods the troubled-cell flag.
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
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 6;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const MASS: f64 = 0.8;
const B0: f64 = 8.0;
const X_LO: f64 = 0.9;

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

/// the guard ledger's attempted totals equal the legacy process-global census,
/// field for field, for the same run.
#[test]
fn the_attempted_totals_equal_the_legacy_census() {
    symbi::regimes::fofc::fofc_reset_stats();

    let sim = kerr_sim();
    let kern =
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);
    let mut hier = Hierarchy::single(sim, kern);
    hier.evolve_with_callback(2.0e-2, 1, |_h| ControlFlow::Continue(()))
        .expect("the violent box must survive to t_final");

    let (attempted, _) = symbi_sim::guard_ledger::report();
    let (fb, fz) = symbi::regimes::fofc::fofc_stats();
    let (fb_h, fz_h) = symbi::regimes::fofc::fofc_horizon_stats();

    assert!(
        fb > 0 || fz > 0,
        "setup never tripped a guard; the equivalence gate is vacuous"
    );
    assert_eq!(
        attempted.troubled_cells.total, fb,
        "attempted troubled cells disagree with the legacy fallback census"
    );
    assert_eq!(
        attempted.frozen_cells.total, fz,
        "attempted frozen cells disagree with the legacy freeze census"
    );
    assert_eq!(
        attempted.troubled_cells.inside_horizon, fb_h,
        "attempted troubled horizon subset disagrees with the legacy census"
    );
    assert_eq!(
        attempted.frozen_cells.inside_horizon, fz_h,
        "attempted frozen horizon subset disagrees with the legacy census"
    );
}
