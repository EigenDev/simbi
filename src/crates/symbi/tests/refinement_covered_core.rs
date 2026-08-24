// =============================================================================
// refinement_covered_core.rs
//
// a compact central force resolved on the finest grid but not on its covered
// ancestors must not poison those inactive coarse cores. only the covered shell
// that supplies coarse-fine donor data participates in the composite update;
// restriction owns the deep core.
//
// run: cargo test -p symbi --test refinement_covered_core -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const LEVELS: usize = 4;
const FINEST_DX: f64 = 1.0 / (N * (1 << (LEVELS - 1))) as f64;
const R_ACC: f64 = 4.0 * FINEST_DX;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

fn uniform(_: [f64; 3]) -> Prim<f64, 3> {
    Prim {
        rho: 1.0,
        vel: Tensor::zeros(),
        pre: 0.6,
    }
}

#[test]
fn unresolved_covered_coarse_core_never_enters_fofc() {
    let make = |s: &Sim| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .reconstruction(Recon::Plm)
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([1.0 / N as f64; 3])
        .ghosts(3)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(uniform)
        .build();
    let ck = make(&coarse);
    let regions: Vec<_> = (1..LEVELS)
        .map(|ll| {
            let half = 0.5 / (1 << ll) as f64;
            RefinementRegion {
                x_lo: [-half; 3],
                x_hi: [half; 3],
            }
        })
        .collect();
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, make)
        .expect("hierarchy construction failed")
        .with_bodies(
            BodyCollection::new().add(
                Body::black_hole(
                    0,
                    Tensor::zeros(),
                    Tensor::zeros(),
                    1.0,
                    R_ACC,
                    R_ACC,
                    0.0,
                    1.0,
                    R_ACC,
                )
                .with_surface(SurfaceSpec::Porous {
                    porosity: 0.0,
                    k_eta_n: 50.0,
                    k_eta_t: 0.0,
                }),
            ),
        );
    for level in 1..hier.levels.len() {
        hier.levels[level].state.seed_cells(uniform);
    }

    fofc_reset_stats();
    hier.evolve_steps(5).expect("covered-core regression run");
    let (fallback, freeze) = fofc_stats();
    assert_eq!(
        (fallback, freeze),
        (0, 0),
        "an inactive covered coarse cell entered FOFC"
    );
}
