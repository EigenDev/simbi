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
// the sealed shell's angular momentum is checked with the other closures in
// lifted_newtonian_physics.rs
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
