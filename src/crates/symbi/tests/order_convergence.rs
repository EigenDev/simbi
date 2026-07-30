// =============================================================================
// order_convergence.rs
//
// a MEASURED spatial order-of-accuracy gate. every other order-sensitive check in the suite
// passes at first order (positivity / finiteness / edge smoke), so a silent drop to first order —
// an unapplied reconstruction slope limiter, say — sails through unnoticed while quietly halving
// the convergence rate.
//
// vehicle: a smooth uniform-pressure, uniform-velocity ENTROPY WAVE. `rho = 1 + A sin(2 pi x)`,
// `v = V`, `p = P` (both constant) is an EXACT solution of the Euler equations — with no pressure
// gradient the momentum/energy equations are trivially satisfied and the density advects passively
// at V. advected one full period on a periodic unit domain it returns to its initial profile, so the
// final L1 density error IS the scheme's accumulated truncation error. halving dx must cut it by
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
        .set_initial(|[x]| Prim {
            rho: rho_exact(x),
            vel: Tensor::new([V]),
            pre: P,
        })
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
    // margin for the smooth-problem constant. a NON-decreasing error (ratio <= 1) means the scheme
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

// --- RHD (special-relativistic) — the SAME exact-solution trick, subluminal. -----------------
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
        .set_initial(|[x]| Prim {
            rho: rho_exact(x),
            vel: Tensor::new([V_REL]),
            pre: P,
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
