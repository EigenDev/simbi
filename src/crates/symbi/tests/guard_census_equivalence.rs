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
// counters. the setup is a cold atmosphere in prescribed mach-173 radial infall
// onto an excised cartesian kerr-schild hole: the supersonic stream meets the
// excision rim's vacuum-floor cells and the high-order recovery there leaves
// the physical set from the first step, at a rate set by the initial data. the
// rim lies inside the horizon r = 2M, so the horizon subsets are exercised too.
// every guard the equivalence compares is asserted to have fired.
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

fn infall_sim() -> (Sim, Kern) {
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
            Prim::adiabatic(
                Density(1.0),
                Tensor::new([-V_INFALL * x[0] / r, -V_INFALL * x[1] / r]),
                Pressure(1.0e-6),
            )
        })
        .build();
    let kern = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0);
    (sim, kern)
}

/// the guard ledger's attempted totals equal the legacy process-global census,
/// field for field, for the same run.
#[test]
fn the_attempted_totals_equal_the_legacy_census() {
    symbi::regimes::fofc::fofc_reset_stats();
    symbi::regimes::fofc::fofc_set_horizon_radius(2.0 * MASS);

    let (sim, kern) = infall_sim();
    let mut hier = Hierarchy::single(sim, kern);
    hier.evolve_with_callback(T_FINAL, 1, |_h| ControlFlow::Continue(()))
        .expect("the infall must survive to t_final");

    let (attempted, _) = symbi_sim::guard_ledger::report();
    let (fb, fz) = symbi::regimes::fofc::fofc_stats();
    let (fb_h, fz_h) = symbi::regimes::fofc::fofc_horizon_stats();

    assert!(
        fb > 0 || fz > 0,
        "setup never tripped a guard; the equivalence gate is vacuous"
    );
    assert!(
        fb_h > 0,
        "no correction fired inside the horizon; the horizon subsets are vacuous"
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
