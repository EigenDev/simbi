// =============================================================================
// gravity_spatial_convergence.rs
//
// the observed spatial order of the gravitational source.
//
// the hydrostatic equilibrium cannot measure this: the discrete flux gradient and the discrete
// gravity source cancel there exactly, so the error is identically zero at every resolution and
// the ratio is 0/0. an order measurement needs a flow that is smooth (so the reconstruction runs
// at its design order rather than clipping on an extremum) and not stationary (so gravity is
// doing something whose discretization can be wrong).
//
// the setup is a small isentropic density perturbation released on the hydrostatic background.
// it launches acoustic waves that propagate through the stratification, and the run stops well
// before they reach either wall, so neither the reflecting boundary nor any patch edge enters
// the measurement — both of which are known to carry their own errors that would otherwise
// dominate the norm.
//
// with no analytic solution available, the order comes from self-convergence: run at N, 2N and
// 4N, conservatively restrict each finer solution onto the next coarser grid, and take
//
//   p = log2( |u_N - R u_2N|_1 / |u_2N - R u_4N|_1 ).
//
// the gravity-free control runs the identical perturbation with no body attached, so the two
// numbers are directly comparable: gravity's own order is only meaningful against the order the
// rest of the scheme achieves on the same problem.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const K0: f64 = 0.6;
/// the mass sits one domain-width left of `x = 0`, so the gas spans `r` in `[1, 2]` and the bare
/// potential needs no softening.
const R_OFFSET: f64 = 1.0;
/// strong enough that the background is genuinely stratified (a density contrast of about 9
/// across the domain) and gravity is a leading term rather than a perturbation.
const GM: f64 = 10.0;
/// the released perturbation: a narrow gaussian in density, isentropic so no shock forms and the
/// exact evolution stays smooth.
const AMP: f64 = 0.05;
const WIDTH: f64 = 0.06;
/// short enough that the launched waves stay clear of both walls. the fastest sound speed on this
/// background is about 2.7, so the disturbance spreads roughly 0.14 either way from the center.
const T_END: f64 = 0.05;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

/// the isentropic hydrostatic background, from the bernoulli invariant
/// `gamma K0/(gamma-1) rho^(gamma-1) - GM/r = const`, normalized to `rho = 1` at the outer edge.
fn background(r: f64, gm: f64) -> f64 {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a - gm / (1.0 + R_OFFSET);
    (a * (gm / r + c)).powf(1.0 / (GAMMA - 1.0))
}

fn initial(gm: f64) -> impl Fn([f64; 1]) -> Prim<f64, 1> + Copy {
    move |x: [f64; 1]| {
        let d = x[0] - 0.5;
        let bump = AMP * (-(d * d) / (WIDTH * WIDTH)).exp();
        let rho = background(x[0] + R_OFFSET, gm) * (1.0 + bump);
        Prim::adiabatic(
            Density(rho),
            Tensor::new([0.0]),
            Pressure(K0 * rho.powf(GAMMA)),
        )
    }
}

/// the interior density profile after the run, one value per cell.
fn run(cells: usize, gm: Option<f64>) -> Vec<f64> {
    let ic = initial(gm.unwrap_or(0.0));
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .spacing([1.0 / cells as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(ic)
        .build();
    let mut sim = match gm {
        Some(g) => sim.with_bodies(BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([-R_OFFSET]),
            Tensor::zeros(),
            g,
            1.0e-3,
            0.0,
        ))),
        None => sim,
    };
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
    evolve(&mut sim, &kernels, T_END).expect("evolve failed");
    let rho = sim.fields.prim.rho.view();
    sim.geom.interior.iter().map(|c| *rho.at(c)).collect()
}

/// conservative 2:1 restriction onto the next coarser grid: the coarse cell average is the mean
/// of the two fine cells it contains, which is the projection that makes the norm below a
/// comparison of the same quantity rather than of two different samplings.
fn restrict(fine: &[f64]) -> Vec<f64> {
    fine.chunks_exact(2).map(|p| 0.5 * (p[0] + p[1])).collect()
}

/// the discrete `L1` distance between a coarse solution and a restricted finer one.
fn l1(coarse: &[f64], restricted: &[f64]) -> f64 {
    assert_eq!(coarse.len(), restricted.len(), "grid mismatch");
    let dx = 1.0 / coarse.len() as f64;
    coarse
        .iter()
        .zip(restricted)
        .map(|(a, b)| (a - b).abs() * dx)
        .sum()
}

/// the self-convergence order over the triple `(N, 2N, 4N)`, and the two norms it came from.
fn observed_order(n: usize, gm: Option<f64>) -> (f64, f64, f64) {
    let (u1, u2, u4) = (run(n, gm), run(2 * n, gm), run(4 * n, gm));
    let e1 = l1(&u1, &restrict(&u2));
    let e2 = l1(&u2, &restrict(&u4));
    ((e1 / e2).log2(), e1, e2)
}

#[test]
fn the_gravitational_source_converges_at_the_order_of_the_scheme() {
    const N: usize = 128;

    // the control: the identical perturbation with no body attached. this is the order the
    // reconstruction, the riemann solver and the time integrator reach on this problem, and it is
    // the only meaningful yardstick for the gravitational number — an order below it would mean
    // gravity is the limiting term, which is the thing under test.
    let (p_free, f1, f2) = observed_order(N, None);
    let (p_grav, g1, g2) = observed_order(N, Some(GM));
    println!("no gravity: L1 {f1:.4e} -> {f2:.4e}   observed order {p_free:.3}");
    println!("gravity:    L1 {g1:.4e} -> {g2:.4e}   observed order {p_grav:.3}");

    // non-vacuity: gravity must actually be shaping the solution, or the two orders would agree
    // because they are measuring the same run. the stratified background alone is a factor of
    // nine in density across the domain.
    let contrast = background(1.0, GM) / background(1.0 + R_OFFSET, GM);
    assert!(
        contrast > 5.0,
        "the background is nearly uniform (contrast {contrast:.3}); gravity is not shaping this \
         flow and the comparison is vacuous"
    );
    // and the errors must be resolved, not round-off: a norm at machine epsilon would make the
    // ratio meaningless whatever it came out as.
    assert!(
        g2 > 1.0e-13 && f2 > 1.0e-13,
        "the finer error norms are at round-off (gravity {g2:.3e}, control {f2:.3e}); the ratio \
         carries no information"
    );

    assert!(
        p_grav > 1.5,
        "the gravitational source converges at order {p_grav:.3} on a smooth flow (L1 \
         {g1:.4e} -> {g2:.4e}). the source is folded into the same convex stage update as the \
         flux divergence, so it should not be limiting the scheme"
    );
    // the real statement: gravity does not cost order. a source evaluated alongside the flux
    // inherits the scheme's accuracy; one composed sequentially with it caps the whole update at
    // first order however accurately either half is integrated.
    assert!(
        p_grav > p_free - 0.5,
        "gravity dropped the observed order from {p_free:.3} to {p_grav:.3}: the source is the \
         limiting term on a smooth flow"
    );
}
