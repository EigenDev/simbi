// =============================================================================
// spherical_axisymmetric_rotation.rs
//
// the newtonian gas on an axisymmetric spherical (r, theta) grid carrying the
// three velocity components (v_r, v_theta, v_phi): DOF = 3 on an NDIM = 2 grid,
// through the production kernel set and the driver. each case compares against a
// closed form that shares no code with the kernels:
// - a uniform gas with uniform v_phi accelerates at v_phi^2 / r radially and
//   v_phi^2 cot(theta) / r polar: the centrifugal and the polar hoop terms
// - rigid rotation v_phi = omega r sin(theta) with p = p0 + (omega r sin(theta))^2 / 2
//   is a steady state: the meridional velocity it develops converges to zero
// - a sealed spherical shell keeps the angular momentum about the axis,
//   sum rho v_phi r sin(theta) dV, with every wall and the poles closed, to a
//   truncation that converges at second order
//
// run: cargo test -p symbi --test spherical_axisymmetric_rotation
// =============================================================================

use std::f64::consts::{FRAC_PI_4, PI};

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

const GAMMA: f64 = 5.0 / 3.0;

fn grid(n: [usize; 2], lo: [f64; 2], hi: [f64; 2], walls: [[BoundaryType; 2]; 2], state: impl Fn([f64; 2]) -> Prim<f64, 3>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells(n)
        .origin(lo)
        .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
        .boundaries(Boundaries::per_axis(walls))
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("the (r, theta) grid with three velocity components")
        .set_initial(state)
        .build()
}

fn evolved(mut sim: Sim, t: f64) -> Sim {
    let kernels = Kern::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &kernels, t).expect("the axisymmetric evolution");
    assert!(sim.time >= t, "the run stopped at {} short of {t}", sim.time);
    sim
}

fn center(sim: &Sim, c: [isize; 2]) -> [f64; 2] {
    std::array::from_fn(|a| sim.geom.x_lo[a] + (c[a] as f64 + 0.5) * sim.geom.dx[a])
}

#[test]
fn uniform_swirl_accelerates_by_the_centrifugal_and_polar_hoop_terms() {
    const V0: f64 = 0.5;
    const T: f64 = 0.01;
    let open = [BoundaryType::Outflow, BoundaryType::Outflow];
    let sim = evolved(
        grid([64, 48], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [open, open], |_| {
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, V0]), Pressure(1.0))
        }),
        T,
    );
    let mut checked = 0;
    for c in sim.geom.interior.iter() {
        if c[0] < 8 || c[0] >= 56 || c[1] < 8 || c[1] >= 40 {
            continue;
        }
        let [r, theta] = center(&sim, c);
        let want = [V0 * V0 / r * sim.time, V0 * V0 / (theta.tan() * r) * sim.time];
        let got = [*sim.fields.prim.vel[0].view().at(c), *sim.fields.prim.vel[1].view().at(c)];
        assert!((got[0] - want[0]).abs() < 0.02 * want[0], "cell {c:?}: v_r {} against {}", got[0], want[0]);
        assert!((got[1] - want[1]).abs() < 0.02 * want[0], "cell {c:?}: v_theta {} against {}", got[1], want[1]);
        checked += 1;
    }
    assert!(checked > 500, "{checked} cells checked");
}

/// the largest meridional speed rigid rotation develops in the shell's middle by t = 0.01
fn rigid_rotation_residual(nr: usize) -> f64 {
    const OMEGA: f64 = 0.5;
    let open = [BoundaryType::Outflow, BoundaryType::Outflow];
    let sim = evolved(
        grid([nr, nr], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [open, open], |[r, theta]: [f64; 2]| {
            let big_r = r * theta.sin();
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, OMEGA * big_r]), Pressure(1.0 + 0.5 * OMEGA * OMEGA * big_r * big_r))
        }),
        0.01,
    );
    sim.geom
        .interior
        .iter()
        .filter(|c| {
            let [r, theta] = center(&sim, *c);
            (1.25..1.75).contains(&r) && (1.0..PI - 1.0).contains(&theta)
        })
        .map(|c| sim.fields.prim.vel[0].view().at(c).abs().max(sim.fields.prim.vel[1].view().at(c).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn rigid_rotation_is_steady_up_to_a_converging_truncation() {
    let residuals: Vec<f64> = [32, 64, 128].into_iter().map(rigid_rotation_residual).collect();
    eprintln!("rigid rotation meridional speed by resolution: {residuals:?}");
    assert!(residuals[0] > 0.0, "the residual is exactly zero; the state never exercised the source");
    assert!(residuals[1] < 0.5 * residuals[0] && residuals[2] < 0.5 * residuals[1], "the residual does not converge: {residuals:?}");
}

fn angular_momentum(sim: &Sim) -> f64 {
    let (dr, dtheta) = (sim.geom.dx[0], sim.geom.dx[1]);
    sim.geom
        .interior
        .iter()
        .map(|c| {
            let [r, theta] = center(sim, c);
            let (r_lo, r_hi) = (r - 0.5 * dr, r + 0.5 * dr);
            let volume = (r_hi.powi(3) - r_lo.powi(3)) / 3.0 * ((theta - 0.5 * dtheta).cos() - (theta + 0.5 * dtheta).cos());
            *sim.fields.cons.mom[2].view().at(c) * r * theta.sin() * volume
        })
        .sum()
}

/// the relative angular-momentum change of a sealed shell, poles on the grid, by t = 0.25
fn sealed_shell_drift(n: usize) -> f64 {
    let wall = [BoundaryType::Reflect, BoundaryType::Reflect];
    let axis = [BoundaryType::Axis, BoundaryType::Axis];
    let sim = grid([n, n], [1.0, 0.0], [2.0, PI], [wall, axis], |[r, theta]: [f64; 2]| {
        let bump = (-((r - 1.5) / 0.15).powi(2)).exp() * (-((theta - 1.2) / 0.4).powi(2)).exp();
        Prim::adiabatic(Density(1.0 + bump), Tensor::new([0.0, 0.0, 0.5 * r * theta.sin()]), Pressure(1.0 + bump))
    });
    let before = angular_momentum(&sim);
    assert!(before > 0.0);
    let sim = evolved(sim, 0.25);
    let radial: f64 = sim.geom.interior.iter().map(|c| sim.fields.prim.vel[0].view().at(c).abs()).fold(0.0, f64::max);
    assert!(radial > 1e-3, "the meridional flow never moved the swirl (max |v_r| = {radial}); the budget is vacuous");
    (angular_momentum(&sim) - before).abs() / before
}

#[test]
fn a_sealed_shell_keeps_its_angular_momentum() {
    let drift: Vec<f64> = [32, 64, 128].into_iter().map(sealed_shell_drift).collect();
    eprintln!("sealed shell relative angular-momentum change by resolution: {drift:?}");
    // the godunov stage's azimuthal source form conserves the angular momentum to its truncation:
    // measured 4.3e-4, 1.1e-4, 2.8e-5 at n = 32, 64, 128, second order
    assert!(drift[0] > 0.0, "the shell holds its angular momentum exactly; the bound below tests nothing");
    assert!(drift[1] < 0.3 * drift[0] && drift[2] < 0.3 * drift[1], "the drift does not converge at second order: {drift:?}");
}
