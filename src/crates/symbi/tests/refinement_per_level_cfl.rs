// =============================================================================
// refine_per_level_cfl.rs
//
// per-level cfl gate: the root step must be limited by EVERY level's cfl
// (dt = min over levels of 2^l * dt_l), not the root's alone. covered coarse
// cells are conservative AVERAGES of fine data, so a fast feature resolved
// only on the fine level — the very thing refinement exists for — is diluted
// on the coarse grid and invisible to a root-only cfl. here a hot blob
// (sound speed ~14x ambient) spans 2^3 fine cells inside the nested box: the
// restriction spreads its energy over 1/8 the coarse cell, the root-only dt
// overdrives the fine level by ~4x its stability limit, and the run blows up
// within a few steps. the per-level minimum keeps every level inside its own
// cfl and the run bounded.
//
// usage:
//  cargo test -p symbi --release --test refine_per_level_cfl
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 16;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

/// quiet ambient gas with a hot blob spanning [0, 2)^3 FINE cells at the box
/// center: pressure 200x ambient over one coarse cell's worth of fine cells.
fn fill(sim: &Sim) {
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let fine = sim.geom.dx[0] < 0.9 / N as f64;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        let in_blob = (0..3).all(|ax| x[ax].abs() < 1.0 / N as f64);
        let pre = if fine && in_blob { 200.0 } else { 1.0 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre };
        let cons = Regime::to_conserved(&sim.physics.regime, &sim.physics.eos, &prim);
        sim.fields.cons.den.view_mut().set(c, cons.den);
        for dd in 0..3 {
            sim.fields.cons.mom[dd].view_mut().set(c, cons.mom[dd]);
        }
        cnrg.view_mut().set(c, cons.nrg);
    }
}

#[test]
fn root_step_respects_the_fine_level_cfl() {
    let dx = 1.0 / N as f64;
    // on the coarse level fill() carries no blob (the blob is fine-only), so the
    // coarse ic is uniform quiet gas: rho=1, vel=0, pre=1.
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|_x: [f64; 3]| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion { x_lo: [-0.25; 3], x_hi: [0.25; 3] }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap();
    fill(&hier.levels[1].state);

    hier.evolve_steps(20).unwrap();

    for (ll, lvl) in hier.levels.iter().enumerate() {
        let pre = lvl.state.fields.prim.pre_field().unwrap();
        for c in lvl.state.geom.interior.iter() {
            let den = *lvl.state.fields.cons.den.view().at(c);
            let p = *pre.view().at(c);
            assert!(
                den.is_finite() && den > 0.0 && p.is_finite() && p > 0.0,
                "L{ll}: fine-level cfl violated by the root-only timestep — \
                 state blew up at {c:?} (den {den:e}, p {p:e})"
            );
        }
    }
}
