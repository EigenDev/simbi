// =============================================================================
// mhd_energy_probe.rs
//
// the base-scheme CT energy-conservation gate at cargo level: the periodic 2d
// magnetized relativistic shock (colliding v = +0.95 / -0.94 streams, p = 1e-4,
// uniform transverse By) whose total tau must hold to roundoff — with the FOFC
// counters read alongside, so a drift is immediately attributed: fofc firing
// (redo-path leakage / the freeze waiver) vs the silent base CT scheme.
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::Solver;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const B0Y: f64 = 0.2;
const NY: usize = 8;

type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn tau_sum(s: &Sim) -> f64 {
    let mut e = 0.0;
    for c in s.geom.interior.iter() {
        e += *s.fields.cons.nrg_field().unwrap().view().at(c);
    }
    e
}

fn run_case(nx: usize, solver: Solver) -> (f64, u64, u64, u64) {
    fofc_reset_stats();
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([nx, NY])
        .origin([0.0, 0.0])
        .spacing([1.0 / nx as f64, 1.0 / NY as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.1)
        .allocate()
        .expect("sim")
        .set_initial(|[x, _y]| {
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0),
                    Tensor::new([if x <= 0.5 { 0.95 } else { -0.94 }, 0.0, 0.0]),
                    Pressure(1e-4),
                ),
                Tensor::new([0.0, B0Y, 0.0]),
            )
        })
        .seed_faces_uniform([0.0, B0Y])
        .build();
    let mut kset = sim.substrate().with_solver(solver).expect("solver");
    kset.theta = 1.5;
    let tau0 = tau_sum(&sim);
    let mut hier = Hierarchy::single(sim, kset);
    hier.evolve(0.1).expect("hierarchy evolve");
    let s = &hier.levels[0].state;
    let drift = ((tau_sum(s) - tau0) / tau0).abs();
    let (fired, froze) = fofc_stats();
    (drift, fired, froze, s.iteration)
}

#[test]
fn magnetized_shock_conserves_tau_with_silent_fofc() {
    for (nx, solver) in [
        (128, Solver::Hlle),
        (128, Solver::Hlld),
        (256, Solver::Hlld),
    ] {
        let (drift, fired, froze, steps) = run_case(nx, solver);
        eprintln!(
            "DBG nx={nx} solver={solver:?}: fallback={fired} freeze={froze} dtau/tau={drift:.3e} steps={steps}"
        );
        assert_eq!(froze, 0, "nx={nx} {solver:?}: froze {froze} cell-substages");
        assert!(
            drift < 1e-11,
            "nx={nx} {solver:?}: tau drifted {drift:.3e} (fofc fallbacks: {fired})"
        );
    }
}
