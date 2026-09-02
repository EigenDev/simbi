// =============================================================================
// refined_ppm_order.rs
//
// the coarse-fine order gate: a smooth entropy wave advected one period across
// a two-level hierarchy whose refinement boundaries sit in the wave's path, so
// every cell's history crosses the interface. the composite L1 error (uncovered
// root cells + fine cells, each against the exact cell-average solution at its
// own spacing) must converge at the interior scheme's rate — a prolongation
// whose ghost averages carry a lower-order error than the interior truncation
// shows up here as a rate capped below the uniform-grid measurement.
//
// two pairings:
// - plm evolution + ppm prolongation (the production default): ~2nd order;
// - ppm evolution + quartic prolongation: must hold the rate well above the
//   plm pairing's — the whole point of the degree-4 transfer.
//
// run: cargo test -p symbi --test refined_ppm_order -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const AMP: f64 = 0.2;
const V: f64 = 1.0;
const P: f64 = 1.0;
const RATIO: usize = 2;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

/// exact cell average of the entropy-wave density over a width-`h` cell at `x`
/// (the scheme's own variable; a center-point sample differs at O(h^2) and
/// would cap any measured rate at two).
fn rho_exact_avg(x: f64, h: f64) -> f64 {
    let sinc = (std::f64::consts::PI * h).sin() / (std::f64::consts::PI * h);
    1.0 + AMP * (std::f64::consts::TAU * x).sin() * sinc
}

/// composite L1 density error after one period at root resolution `n`.
fn l1_refined(n: usize, recon: Recon, prolong: ProlongOrder, ng: usize, cfl: f64) -> f64 {
    let dx = 1.0 / n as f64;
    let ic_root = move |[x]: [f64; 1]| {
        Prim::adiabatic(Density(rho_exact_avg(x, dx)), Tensor::new([V]), Pressure(P))
    };
    let root = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .ghosts(ng)
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cfl)
        .timestepping(Timestepping::Rk3)
        .allocate()
        .expect("root construction failed")
        .set_initial(ic_root)
        .build();
    let kern = |s: &Sim| {
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, cfl, &s.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .reconstruction(recon)
    };
    let ck = kern(&root);
    let regions = [RefinementRegion {
        x_lo: [0.25],
        x_hi: [0.75],
    }];
    let mut hier = Hierarchy::with_refinement(root, ck, &regions, prolong, kern)
        .expect("hierarchy construction failed");
    let dxf = dx / RATIO as f64;
    hier.levels[1].state.seed_cells(move |[x]: [f64; 1]| {
        Prim::adiabatic(
            Density(rho_exact_avg(x, dxf)),
            Tensor::new([V]),
            Pressure(P),
        )
    });
    hier.evolve(1.0).expect("refined evolve failed");

    // the covered root index band, read off the fine level's own extent so the
    // composite sum never double-counts a cell at a rounded region edge.
    let fine = &hier.levels[1].state;
    let fspan = &fine.geom.interior.spaces[0];
    let (cov_lo, cov_hi) = (
        fspan.lo.div_euclid(RATIO as isize),
        fspan.hi.div_euclid(RATIO as isize),
    );
    let mut l1 = 0.0;
    let root = &hier.levels[0].state;
    for c in root.geom.interior.iter() {
        if c[0] >= cov_lo && c[0] < cov_hi {
            continue;
        }
        let x = (c[0] as f64 + 0.5) * dx;
        l1 += (*root.fields.prim.rho.view().at(c) - rho_exact_avg(x, dx)).abs() * dx;
    }
    for c in fine.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dxf;
        l1 += (*fine.fields.prim.rho.view().at(c) - rho_exact_avg(x, dxf)).abs() * dxf;
    }
    // a full period needs hundreds of root steps; a run that halted
    // early (crash detector, dt collapse) leaves a near-IC state whose tiny L1
    // would fake convergence — the gate must fail loudly instead.
    let (t_end, iters) = (hier.levels[0].state.time, hier.levels[0].state.iteration);
    eprintln!("  refined run: t = {t_end:.6}, {iters} root steps");
    assert!(
        (t_end - 1.0).abs() < 1e-12 && iters > 100,
        "the refined run halted early (t = {t_end:.6}, {iters} steps); the gate is vacuous"
    );
    l1
}

#[test]
fn plm_rate_holds_across_the_refinement_boundary() {
    let e1 = l1_refined(64, Recon::Plm, ProlongOrder::Ppm, 2, CFL);
    let e2 = l1_refined(128, Recon::Plm, ProlongOrder::Ppm, 2, CFL);
    let ratio = e1 / e2;
    eprintln!("refined plm+ppm: L1(64)={e1:.3e} L1(128)={e2:.3e} ratio {ratio:.2}");
    assert!(
        e2 < e1 && ratio > 3.0,
        "plm evolution lost second order across the refinement boundary: ratio {ratio:.2}"
    );
}

/// the spatial transfer property, isolated: the coarse ghost fill interpolates
/// linearly in time between coarse steps, an O(dt_c^2) boundary term that no
/// spatial prolongation can remove — it hides below plm's own O(h^2) interior
/// but shows once the interior is fourth order. at cfl 0.1 that term recedes
/// ~16x and the quartic transfer's spatial rate is exposed: measured ratio
/// 14.7 per halving (L1 8.9e-8 at root 128). a rate near 4 here means the
/// spatial prolongation, not the boundary, sets the composite order.
#[test]
fn ppm_quartic_spatial_rate_holds_across_the_refinement_boundary() {
    let e1 = l1_refined(64, Recon::Ppm, ProlongOrder::Quartic, 3, 0.1);
    let e2 = l1_refined(128, Recon::Ppm, ProlongOrder::Quartic, 3, 0.1);
    let ratio = e1 / e2;
    eprintln!("refined ppm+quartic (cfl 0.1): L1(64)={e1:.3e} L1(128)={e2:.3e} ratio {ratio:.2}");
    assert!(
        e2 < e1 && ratio > 10.0,
        "the quartic coarse-fine transfer lost its spatial order: ratio {ratio:.2} \
         (measured ~14.7; ~4 means an O(h^2) boundary term returned, ~8 means the \
         ghost averages dropped an order)"
    );
}

/// the production-cfl pin: at cfl 0.4 the linear-in-time ghost interpolation
/// contributes an O(dt_c^2) composite term (measured ratio ~3.9 there), but its
/// magnitude is what production feels — the composite error must stay far below
/// the plm pairing's at the same resolution (measured 3.3e-6 vs 2.5e-3, 770x),
/// and the rate must not drop below the plm family's own.
#[test]
fn ppm_quartic_beats_plm_outright_at_production_cfl() {
    let e_ppm = l1_refined(64, Recon::Ppm, ProlongOrder::Quartic, 3, CFL);
    let e2_ppm = l1_refined(128, Recon::Ppm, ProlongOrder::Quartic, 3, CFL);
    let e_plm = l1_refined(64, Recon::Plm, ProlongOrder::Ppm, 2, CFL);
    let ratio = e_ppm / e2_ppm;
    eprintln!(
        "refined at cfl 0.4: ppm+quartic L1(64)={e_ppm:.3e} (ratio {ratio:.2}), \
         plm+ppm L1(64)={e_plm:.3e}"
    );
    assert!(
        e_ppm * 100.0 < e_plm,
        "ppm+quartic composite error {e_ppm:.3e} is not far below plm's {e_plm:.3e} \
         at production cfl; the time-interpolation boundary term has grown"
    );
    assert!(
        ratio > 3.0,
        "ppm+quartic composite rate {ratio:.2} fell below the plm family's at cfl 0.4"
    );
}
