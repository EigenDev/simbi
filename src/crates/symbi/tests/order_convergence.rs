// =============================================================================
// order_convergence.rs
//
// a measured spatial order-of-accuracy gate. every other order-sensitive check in the suite
// passes at first order (positivity / finiteness / edge smoke), so a silent drop to first order —
// an unapplied reconstruction slope limiter, say — sails through unnoticed while quietly halving
// the convergence rate.
//
// vehicle: a smooth uniform-pressure, uniform-velocity entropy wave. `rho = 1 + A sin(2 pi x)`,
// `v = V`, `p = P` (both constant) is an exact solution of the Euler equations — with no pressure
// gradient the momentum/energy equations are trivially satisfied and the density advects passively
// at V. advected one full period on a periodic unit domain it returns to its initial profile, so the
// final L1 density error is the scheme's accumulated truncation error. halving dx must cut it by
// ~2^p; PLM + SSP-RK2 is second order, so the ratio is ~4. a ratio below 3 means the reconstruction
// collapsed to first order.
//
// run: cargo test -p symbi --test order_convergence
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
const GAMMA: f64 = 5.0 / 3.0;
const AMP: f64 = 0.2;
const V: f64 = 1.0; // advection velocity; one period over the unit domain at t = 1
const P: f64 = 1.0;

fn rho_exact(x: f64) -> f64 {
    1.0 + AMP * (std::f64::consts::TAU * x).sin()
}

/// L1 density error after advecting the entropy wave one full period (t = 1, domain [0,1], v = 1),
/// at resolution `n` with the given solver.
fn l1_after_one_period(n: usize, solver: Solver) -> f64 {
    let dx = 1.0 / n as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x]| Prim::adiabatic(Density(rho_exact(x)), Tensor::new([V]), Pressure(P)))
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated)
            .with_solver(solver)
            .expect("solver/regime mismatch");
    evolve(&mut sim, &sub, 1.0).expect("evolve failed");
    let rho = &sim.fields.prim.rho;
    let mut l1 = 0.0;
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        l1 += (*rho.view().at(c) - rho_exact(x)).abs() * dx;
    }
    l1
}

fn assert_second_order(solver: Solver, tag: &str) {
    let e1 = l1_after_one_period(64, solver);
    let e2 = l1_after_one_period(128, solver);
    let ratio = e1 / e2;
    eprintln!(
        "{tag}: L1(64)={e1:.3e}  L1(128)={e2:.3e}  order~{:.2} (ratio {ratio:.2})",
        ratio.log2()
    );
    // 2nd order -> ratio ~4; 1st order -> ~2. the > 3 threshold cleanly separates them and leaves
    // margin for the smooth-problem constant. a non-decreasing error (ratio <= 1) means the scheme
    // is not even converging.
    assert!(
        e2 < e1 && ratio > 3.0,
        "{tag}: adiabatic PLM+RK2 is not ~2nd order — L1(64)={e1:.3e}, L1(128)={e2:.3e}, \
         ratio={ratio:.2} (expect ~4; a ratio < 3 means reconstruction dropped to first order)"
    );
}

#[test]
fn adiabatic_plm_rk2_second_order_hllc() {
    assert_second_order(Solver::Hllc, "HLLC");
}

#[test]
fn adiabatic_plm_rk2_second_order_hlle() {
    assert_second_order(Solver::Hlle, "HLLE");
}

// --- PPM: the same entropy wave under SSP-RK3. the extremum-preserving interfaces --------------
// (colella & sekora 2008) are 4th-order on smooth data including at the sine crest and trough —
// both the flatten-at-extremum form and the plain neighbor-range interface clamp inject O(h^2)
// there (the interface point value legitimately exceeds both adjacent cell averages when the
// extremum sits near a face), and either one drags the whole scheme's L1 to second order. on
// this pure-advection exact solution the L1 error tracks the interface order: measured ratios
// 15.7-15.9 per halving at 32..256, cfl-independent (identical at cfl 0.05).
/// the exact cell average of the entropy-wave density over a width-`h` cell centered at
/// `x`: averaging `1 + A sin(2 pi x)` gives `1 + A sin(2 pi x) * sinc(pi h)`. the scheme
/// evolves cell averages, and average vs center-point sample differ at O(h^2) — sampling
/// the profile at centers would cap any measured convergence at second order, hiding a
/// third-order scheme behind the initialization. momentum and energy are linear in rho at
/// constant v and p, so building cons from this averaged rho is exact in every field.
fn rho_exact_avg(x: f64, h: f64) -> f64 {
    let sinc = (std::f64::consts::PI * h).sin() / (std::f64::consts::PI * h);
    1.0 + AMP * (std::f64::consts::TAU * x).sin() * sinc
}

fn l1_after_one_period_ppm(n: usize, solver: Solver) -> f64 {
    l1_ppm_at_cfl(n, solver, 0.4)
}

fn l1_ppm_at_cfl(n: usize, solver: Solver, cfl: f64) -> f64 {
    let dx = 1.0 / n as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .ghosts(3)
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(cfl)
        .timestepping(Timestepping::Rk3)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x]| {
            Prim::adiabatic(Density(rho_exact_avg(x, dx)), Tensor::new([V]), Pressure(P))
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, cfl, &sim.geom.allocated)
            .with_solver(solver)
            .expect("solver/regime mismatch")
            .reconstruction(symbi_discretize::Recon::Ppm);
    evolve(&mut sim, &sub, 1.0).expect("evolve failed");
    let rho = &sim.fields.prim.rho;
    let mut l1 = 0.0;
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        l1 += (*rho.view().at(c) - rho_exact_avg(x, dx)).abs() * dx;
    }
    l1
}

#[test]
fn adiabatic_ppm_rk3_beats_second_order() {
    let errs: Vec<f64> = [32, 64, 128, 256]
        .iter()
        .map(|&n| l1_after_one_period_ppm(n, Solver::Hllc))
        .collect();
    for w in errs.windows(2) {
        eprintln!(
            "PPM: ratio {:.2} (order ~{:.2})",
            w[0] / w[1],
            (w[0] / w[1]).log2()
        );
    }
    let ratio = errs[1] / errs[2]; // 64 -> 128, matching the plm gate's pair
    // the bound pins the measured 15.7-15.9: a drop to ~8 means the interfaces lost
    // their 4th order at extrema (a re-introduced clamp or flatten), a drop to ~4
    // means the parabola collapsed to the linear fan. both are defects this gate
    // exists to catch, so the bound sits just under the measurement, not at the
    // generic third-order floor.
    assert!(
        errs[2] < errs[1] && ratio > 12.0,
        "ppm+rk3 lost interface order on the smooth wave — L1(64)={:.3e}, L1(128)={:.3e}, \
         ratio={ratio:.2} (measured ~15.8; ~8 = extremum interfaces degraded; ~4 = \
         parabola collapsed to plm)",
        errs[1],
        errs[2]
    );
    // ppm at n must also beat plm at n outright — the whole point of the wider stencil.
    let plm = l1_after_one_period(128, Solver::Hllc);
    assert!(
        errs[2] < plm,
        "ppm L1(128)={:.3e} is not below plm L1(128)={plm:.3e}",
        errs[2]
    );
}

// --- RHD (special-relativistic) — the same exact-solution trick, subluminal. -----------------
// a uniform-p, uniform-v smooth density wave is an exact SRHD solution too: with no pressure
// gradient the momentum/energy equations reduce to advection, and the conserved D = rho W (W const
// at v const) advects at v, so the primitive rho does. v = 0.5 is subluminal; one period is t = 2.
type RhdSim = SimState<Rhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
const V_REL: f64 = 0.5;

fn l1_after_one_period_rhd(n: usize, solver: Solver) -> f64 {
    let dx = 1.0 / n as f64;
    let mut sim = RhdSim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("rhd sim construction failed")
        .set_initial(|[x]| {
            Prim::adiabatic(Density(rho_exact(x)), Tensor::new([V_REL]), Pressure(P))
        })
        .build();
    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated)
        .with_solver(solver)
        .expect("solver/regime mismatch");
    evolve(&mut sim, &sub, 1.0 / V_REL).expect("rhd evolve failed"); // one period at v = 0.5 -> t = 2
    let rho = &sim.fields.prim.rho;
    let mut l1 = 0.0;
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        l1 += (*rho.view().at(c) - rho_exact(x)).abs() * dx;
    }
    l1
}

#[test]
fn rhd_plm_rk2_second_order_hllc() {
    let e1 = l1_after_one_period_rhd(64, Solver::Hllc);
    let e2 = l1_after_one_period_rhd(128, Solver::Hllc);
    let ratio = e1 / e2;
    eprintln!(
        "RHD HLLC: L1(64)={e1:.3e}  L1(128)={e2:.3e}  order~{:.2} (ratio {ratio:.2})",
        ratio.log2()
    );
    assert!(
        e2 < e1 && ratio > 3.0,
        "RHD PLM+RK2 is not ~2nd order — L1(64)={e1:.3e}, L1(128)={e2:.3e}, ratio={ratio:.2} \
         (expect ~4; < 3 means reconstruction dropped to first order)"
    );
}
