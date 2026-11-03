// =============================================================================
// lifted_newtonian_physics.rs
//
// the newtonian gas with three velocity components on a 2-axis grid, across the
// layouts and closures, through the production kernel sets and the driver. each case
// compares against a closed form sharing no code with the kernels:
// - cartesian plane, adiabatic and isothermal: an out-of-plane velocity w(x) in a
//   uniform x flow is advected exactly, w(x - u t), converging; the in-plane transverse
//   velocity stays zero; mass and out-of-plane momentum are conserved to rounding,
//   and total energy too on the adiabatic closure
// - (r, theta) and (R, z) planes under the isothermal closure, and (R, z) adiabatic:
//   rigid rotation with the matching hydrostatic state is steady up to a truncation that
//   converges
// - (r, theta) isothermal: uniform swirl accelerates by the centrifugal and polar hoop terms
// - sealed shells on both axisymmetric planes, both closures: the angular momentum
//   about the axis, sum M_phi R_c V, holds to rounding; the torque form telescopes
// - a radial drift carries a uniform specific angular momentum v_phi R, up to a truncation
//   that converges
// - the locally isothermal closure, cs^2(x) prescribed per cell: uniform density under
//   cs^2 = 1 + (omega R)^2 / 2 is the rigid-rotation hydrostatic state on both planes, and the
//   cartesian advection converges through a density and cs^2 varying across the flow at uniform
//   pressure, which only a closure read per cell keeps in balance
//
// the adiabatic (r, theta) cases live in spherical_axisymmetric_rotation.rs.
//
// run: cargo test -p symbi --test lifted_newtonian_physics
// =============================================================================

use std::f64::consts::{FRAC_PI_4, PI};

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Cylindrical, Spherical};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CS: f64 = 1.0;
const OMEGA: f64 = 0.5;
const CFL: f64 = 0.4;

const OPEN: [BoundaryType; 2] = [BoundaryType::Outflow, BoundaryType::Outflow];
const RING: [BoundaryType; 2] = [BoundaryType::Periodic, BoundaryType::Periodic];
const WALL: [BoundaryType; 2] = [BoundaryType::Reflect, BoundaryType::Reflect];
const AXIS: [BoundaryType; 2] = [BoundaryType::Axis, BoundaryType::Axis];

/// builds a closure's evolved grid and hands back cell-indexed views of the fields a check reads
macro_rules! adiabatic_run {
    ($metric:expr, $metric_ty:ty, $n:expr, $lo:expr, $hi:expr, $walls:expr, $t:expr, $state:expr) => {{
        type Sim = SimStateGeneric<Newtonian, 2, 3, $metric_ty, IdealGas<f64>, CpuSpace, HostMemory>;
        let (n, lo, hi): ([usize; 2], [f64; 2], [f64; 2]) = ($n, $lo, $hi);
        let state = $state;
        let mut sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $metric)
            .cells(n)
            .origin(lo)
            .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
            .boundaries(Boundaries::per_axis($walls))
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the plane with three velocity components")
            .set_initial(|x: [f64; 2]| {
                let (rho, v, p): (f64, [f64; 3], f64) = state(x);
                Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p))
            })
            .build();
        let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated);
        let opening = snapshot(&sim);
        if $t > 0.0 {
            evolve(&mut sim, &kernels, $t).expect("the adiabatic evolution");
        }
        (opening, snapshot(&sim))
    }};
}

macro_rules! isothermal_run {
    ($metric:expr, $metric_ty:ty, $n:expr, $lo:expr, $hi:expr, $walls:expr, $t:expr, $state:expr) => {
        isothermal_run!($metric, $metric_ty, $n, $lo, $hi, $walls, $t, $state, |_x: [f64; 2]| CS * CS)
    };
    ($metric:expr, $metric_ty:ty, $n:expr, $lo:expr, $hi:expr, $walls:expr, $t:expr, $state:expr, $cs2:expr) => {{
        type Sim = SimStateGeneric<IsoNewtonian, 2, 3, $metric_ty, Isothermal<f64>, CpuSpace, HostMemory>;
        let (n, lo, hi): ([usize; 2], [f64; 2], [f64; 2]) = ($n, $lo, $hi);
        let state = $state;
        let mut sim: Sim = Sim::build(IsoNewtonian, Isothermal { cs: CS }, $metric)
            .cells(n)
            .origin(lo)
            .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
            .boundaries(Boundaries::per_axis($walls))
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the plane with three velocity components")
            .set_initial(|x: [f64; 2]| {
                let (rho, v, _p): (f64, [f64; 3], f64) = state(x);
                PrimG::<f64, 3, IsoModel>::isothermal(Density(rho), Tensor::new(v))
            })
            .build();
        let kernels = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(CS, CFL, &sim.geom.allocated);
        // the prescribed closure on every allocated cell, ghosts held at their own coordinates
        let closure = $cs2;
        for c in sim.geom.allocated.iter() {
            let x: [f64; 2] = std::array::from_fn(|a| lo[a] + (c[a] as f64 + 0.5) * (hi[a] - lo[a]) / n[a] as f64);
            kernels.cs2.set(c, closure(x));
        }
        let opening = snapshot(&sim);
        if $t > 0.0 {
            evolve(&mut sim, &kernels, $t).expect("the isothermal evolution");
        }
        (opening, snapshot(&sim))
    }};
}

/// one grid's interior at one time: cell centers, density, the three momenta, velocity and energy
struct State {
    time: f64,
    dx: [f64; 2],
    x: Vec<[f64; 2]>,
    den: Vec<f64>,
    mom: [Vec<f64>; 3],
    vel: [Vec<f64>; 3],
    nrg: Option<Vec<f64>>,
}

fn snapshot(sim: &FieldStore<2, 3, HostMemory, f64>) -> State {
    let cells: Vec<[isize; 2]> = sim.geom.interior.iter().collect();
    let dx = [sim.geom.dx[0], sim.geom.dx[1]];
    let x = cells.iter().map(|c| std::array::from_fn(|a| sim.geom.x_lo[a] + (c[a] as f64 + 0.5) * dx[a])).collect();
    let read = |f: &symbi_grid::Field<f64, 2, HostMemory>| -> Vec<f64> { cells.iter().map(|c| *f.view().at(*c)).collect() };
    State {
        time: sim.time,
        dx,
        x,
        den: read(&sim.fields.cons.den),
        mom: std::array::from_fn(|k| read(&sim.fields.cons.mom[k])),
        vel: std::array::from_fn(|k| read(&sim.fields.prim.vel[k])),
        nrg: sim.fields.cons.nrg_field().map(read),
    }
}

fn converges(label: &str, measured: &[f64], factor: f64) {
    eprintln!("{label} by resolution: {measured:?}");
    assert!(measured[0] > 0.0, "{label}: exactly zero; the setup never exercised the term");
    for pair in measured.windows(2) {
        assert!(pair[1] < factor * pair[0], "{label} does not converge by {factor}: {measured:?}");
    }
}

fn largest(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::max)
}

// --------------------- cartesian plane: out-of-plane advection ---------------------

const U: f64 = 1.0;
const T_ADVECT: f64 = 0.25;

fn sheared_flow([x, _y]: [f64; 2]) -> (f64, [f64; 3], f64) {
    (1.0, [U, 0.0, (2.0 * PI * x).sin()], 1.0)
}

fn advection_checks(label: &str, opening: &State, end: &State) -> f64 {
    transverse_advection_checks(label, opening, end, 0.0)
}

fn transverse_advection_checks(label: &str, opening: &State, end: &State, transverse: f64) -> f64 {
    assert!(end.time >= T_ADVECT, "{label}: stopped at {}", end.time);
    let moved = largest(end.vel[1].iter().map(|v| v.abs()));
    assert!(moved <= transverse, "{label}: the in-plane transverse velocity reached {moved}");
    let total = |f: &[f64]| f.iter().sum::<f64>();
    // the conservative update telescopes on a ring, leaving the rounding of a sum over the cells:
    // a few units of 2^-53 per cell per step, scaled by the largest magnitude summed
    let rounding = |f: &[f64]| 1e-13 * f.len() as f64 * largest(f.iter().map(|v| v.abs()));
    assert!((total(&end.den) - total(&opening.den)).abs() < rounding(&end.den), "{label}: mass moved");
    assert!((total(&end.mom[2]) - total(&opening.mom[2])).abs() < rounding(&end.den), "{label}: out-of-plane momentum moved");
    if let (Some(a), Some(b)) = (&end.nrg, &opening.nrg) {
        assert!((total(a) - total(b)).abs() < rounding(a), "{label}: total energy moved");
    }
    let err: f64 = end.x.iter().zip(&end.vel[2]).map(|(x, w)| (w - (2.0 * PI * (x[0] - U * end.time)).sin()).abs()).sum();
    err / end.x.len() as f64
}

#[test]
fn the_cartesian_plane_advects_the_out_of_plane_velocity_adiabatic() {
    let errors: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (a, b) = adiabatic_run!(Cartesian, Cartesian, [n, 4], [0.0, 0.0], [1.0, 1.0], [RING, RING], T_ADVECT, sheared_flow);
            advection_checks("cartesian adiabatic", &a, &b)
        })
        .collect();
    converges("cartesian adiabatic advection L1", &errors, 0.5);
}

#[test]
fn the_cartesian_plane_advects_the_out_of_plane_velocity_isothermal() {
    let errors: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (a, b) = isothermal_run!(Cartesian, Cartesian, [n, 4], [0.0, 0.0], [1.0, 1.0], [RING, RING], T_ADVECT, sheared_flow);
            advection_checks("cartesian isothermal", &a, &b)
        })
        .collect();
    converges("cartesian isothermal advection L1", &errors, 0.5);
}

// --------------------- (r, theta), isothermal: hoop terms ---------------------

#[test]
fn uniform_isothermal_swirl_accelerates_by_the_centrifugal_and_polar_hoop_terms() {
    const V0: f64 = 0.5;
    let (_, end) = isothermal_run!(Spherical, Spherical, [64, 48], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [OPEN, OPEN], 0.01, |_x: [f64; 2]| (1.0, [0.0, 0.0, V0], 1.0));
    let (mut checked, width) = (0, 8.0 * end.dx[0]);
    for (kk, x) in end.x.iter().enumerate() {
        let [r, theta] = *x;
        if r < 1.0 + width || r > 2.0 - width || (theta - FRAC_PI_4) < 8.0 * end.dx[1] || (3.0 * FRAC_PI_4 - theta) < 8.0 * end.dx[1] {
            continue;
        }
        let want_r = V0 * V0 / r * end.time;
        let want_t = V0 * V0 / (theta.tan() * r) * end.time;
        assert!((end.vel[0][kk] - want_r).abs() < 0.02 * want_r, "({r}, {theta}): v_r {} against {want_r}", end.vel[0][kk]);
        assert!((end.vel[1][kk] - want_t).abs() < 0.02 * want_r, "({r}, {theta}): v_theta {} against {want_t}", end.vel[1][kk]);
        checked += 1;
    }
    assert!(checked > 500, "{checked} cells checked");
}

// --------------------- rigid rotation ---------------------

/// rigid rotation about the axis at cylindrical radius `big_r`, with the state holding it up
fn rigid_adiabatic(big_r: f64) -> (f64, f64, f64) {
    (1.0, OMEGA * big_r, 1.0 + 0.5 * OMEGA * OMEGA * big_r * big_r)
}

fn rigid_isothermal(big_r: f64) -> (f64, f64, f64) {
    ((0.5 * OMEGA * OMEGA * big_r * big_r / (CS * CS)).exp(), OMEGA * big_r, 0.0)
}

fn meridional_residual(end: &State, inside: impl Fn([f64; 2]) -> bool, slots: [usize; 2]) -> f64 {
    assert!(end.time >= 0.01, "stopped at {}", end.time);
    largest(end.x.iter().enumerate().filter(|(_, x)| inside(**x)).map(|(k, _)| end.vel[slots[0]][k].abs().max(end.vel[slots[1]][k].abs())))
}

#[test]
fn rigid_rotation_is_steady_on_the_rz_plane_adiabatic() {
    let residual: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = adiabatic_run!(Cylindrical, Cylindrical, [n, 8], [1.0, 0.0], [2.0, 0.5], [OPEN, RING], 0.01, |[r, _z]: [f64; 2]| {
                let (rho, v, p) = rigid_adiabatic(r);
                (rho, [0.0, v, 0.0], p)
            });
            meridional_residual(&end, |[r, _z]| (1.25..1.75).contains(&r), [0, 2])
        })
        .collect();
    converges("rz adiabatic rigid rotation", &residual, 0.6);
}

#[test]
fn rigid_rotation_is_steady_on_the_rz_plane_isothermal() {
    let residual: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = isothermal_run!(Cylindrical, Cylindrical, [n, 8], [1.0, 0.0], [2.0, 0.5], [OPEN, RING], 0.01, |[r, _z]: [f64; 2]| {
                let (rho, v, p) = rigid_isothermal(r);
                (rho, [0.0, v, 0.0], p)
            });
            meridional_residual(&end, |[r, _z]| (1.25..1.75).contains(&r), [0, 2])
        })
        .collect();
    converges("rz isothermal rigid rotation", &residual, 0.6);
}

#[test]
fn rigid_rotation_is_steady_on_the_spherical_plane_isothermal() {
    let residual: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = isothermal_run!(Spherical, Spherical, [n, n], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [OPEN, OPEN], 0.01, |[r, theta]: [f64; 2]| {
                let (rho, v, p) = rigid_isothermal(r * theta.sin());
                (rho, [0.0, 0.0, v], p)
            });
            meridional_residual(&end, |[r, theta]| (1.25..1.75).contains(&r) && (1.0..PI - 1.0).contains(&theta), [0, 1])
        })
        .collect();
    converges("spherical isothermal rigid rotation", &residual, 0.6);
}

// --------------------- sealed shells: angular momentum ---------------------

const T_SHELL: f64 = 0.25;

fn bump(a: f64, a0: f64, b: f64, b0: f64) -> f64 {
    (-((a - a0) / 0.15).powi(2)).exp() * (-((b - b0) / 0.25).powi(2)).exp()
}

/// the angular momentum about the axis, sum M_phi R_c V, with the centroid's distance from the axis
fn sphere_l(s: &State) -> f64 {
    let [dr, dtheta] = s.dx;
    s.x.iter()
        .zip(&s.mom[2])
        .map(|([r, t], m)| {
            let (r_lo, r_hi, t_lo, t_hi) = (r - 0.5 * dr, r + 0.5 * dr, t - 0.5 * dtheta, t + 0.5 * dtheta);
            let i_r2 = (r_hi.powi(3) - r_lo.powi(3)) / 3.0;
            let i_sin = t_lo.cos() - t_hi.cos();
            let r_c = (r_hi.powi(4) - r_lo.powi(4)) / 4.0 / i_r2;
            let t_c = ((t_hi.sin() - t_hi * t_hi.cos()) - (t_lo.sin() - t_lo * t_lo.cos())) / i_sin;
            m * r_c * t_c.sin() * i_r2 * i_sin
        })
        .sum()
}

fn rz_l(s: &State) -> f64 {
    let [dr, dz] = s.dx;
    s.x.iter()
        .zip(&s.mom[1])
        .map(|([r, _z], m)| {
            let (lo, hi) = (r - 0.5 * dr, r + 0.5 * dr);
            let i_r = (hi * hi - lo * lo) / 2.0;
            m * ((hi.powi(3) - lo.powi(3)) / 3.0 / i_r) * i_r * dz
        })
        .sum()
}

/// the relative change a sealed domain's torque form leaves: the rounding of the sum over the
/// cells across the steps, measured at 2e-14 and below on these grids
const SEALED_BOUND: f64 = 1e-12;

fn exactly_held(label: &str, a: f64, b: f64, speed: f64) {
    assert!(a > 0.0 && speed > 1e-3, "{label}: the budget is vacuous (L = {a}, max meridional speed {speed})");
    let change = (b - a).abs() / a;
    eprintln!("{label} relative L change: {change:e}");
    assert!(change < SEALED_BOUND, "{label}: L changed by {change:e}");
}

#[test]
fn a_sealed_spherical_shell_holds_its_angular_momentum_adiabatic() {
    for n in [32, 64, 128] {
        let (a, b) = adiabatic_run!(Spherical, Spherical, [n, n], [1.0, 0.0], [2.0, PI], [WALL, AXIS], T_SHELL, |[r, t]: [f64; 2]| {
            let extra = bump(r, 1.5, t, 1.2);
            (1.0 + extra, [0.0, 0.0, 0.5 * r * t.sin()], 1.0 + extra)
        });
        exactly_held("spherical adiabatic", sphere_l(&a), sphere_l(&b), largest(b.vel[0].iter().map(|v| v.abs())));
    }
}

#[test]
fn a_sealed_spherical_shell_holds_its_angular_momentum_isothermal() {
    for n in [32, 64, 128] {
        let (a, b) = isothermal_run!(Spherical, Spherical, [n, n], [1.0, 0.0], [2.0, PI], [WALL, AXIS], T_SHELL, |[r, t]: [f64; 2]| {
            (1.0 + bump(r, 1.5, t, 1.2), [0.0, 0.0, 0.5 * r * t.sin()], 0.0)
        });
        exactly_held("spherical isothermal", sphere_l(&a), sphere_l(&b), largest(b.vel[0].iter().map(|v| v.abs())));
    }
}

#[test]
fn a_sealed_rz_annulus_holds_its_angular_momentum_adiabatic() {
    for n in [32, 64, 128] {
        let (a, b) = adiabatic_run!(Cylindrical, Cylindrical, [n, n], [1.0, 0.0], [2.0, 1.0], [WALL, WALL], T_SHELL, |[r, z]: [f64; 2]| {
            let extra = bump(r, 1.5, z, 0.5);
            (1.0 + extra, [0.0, 0.5 * r, 0.0], 1.0 + extra)
        });
        exactly_held("rz adiabatic", rz_l(&a), rz_l(&b), largest(b.vel[0].iter().map(|v| v.abs())));
    }
}

#[test]
fn a_sealed_rz_annulus_holds_its_angular_momentum_isothermal() {
    for n in [32, 64, 128] {
        let (a, b) = isothermal_run!(Cylindrical, Cylindrical, [n, n], [1.0, 0.0], [2.0, 1.0], [WALL, WALL], T_SHELL, |[r, z]: [f64; 2]| {
            (1.0 + bump(r, 1.5, z, 0.5), [0.0, 0.5 * r, 0.0], 0.0)
        });
        exactly_held("rz isothermal", rz_l(&a), rz_l(&b), largest(b.vel[0].iter().map(|v| v.abs())));
    }
}

// --------------------- specific angular momentum carried by a radial flow ---------------------

/// a uniform specific angular momentum ell = v_phi R in a radial drift stays uniform: the worst
/// departure in the domain's middle, by t = 0.05. the azimuthal flux reconstructs v_phi, so the
/// torque form holds it to a truncation
const ELL: f64 = 0.5;
const DRIFT: f64 = 0.1;

fn ell_departure(end: &State, slot: usize, radius: impl Fn([f64; 2]) -> f64, inside: impl Fn([f64; 2]) -> bool) -> f64 {
    assert!(end.time >= 0.05, "stopped at {}", end.time);
    assert!(largest(end.vel[0].iter().map(|v| v.abs())) > 0.5 * DRIFT, "the radial drift vanished; nothing carries ell");
    largest(end.x.iter().enumerate().filter(|(_, x)| inside(**x)).map(|(k, x)| (end.vel[slot][k] * radius(*x) / ELL - 1.0).abs()))
}

#[test]
fn a_radial_flow_carries_uniform_specific_angular_momentum_rz_adiabatic() {
    let measured: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = adiabatic_run!(Cylindrical, Cylindrical, [n, 8], [1.0, 0.0], [2.0, 0.5], [OPEN, RING], 0.05, |[r, _z]: [f64; 2]| (1.0, [DRIFT, ELL / r, 0.0], 1.0));
            ell_departure(&end, 1, |[r, _z]| r, |[r, _z]| (1.25..1.75).contains(&r))
        })
        .collect();
    converges("rz adiabatic ell departure", &measured, 0.6);
}

#[test]
fn a_radial_flow_carries_uniform_specific_angular_momentum_rz_isothermal() {
    let measured: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = isothermal_run!(Cylindrical, Cylindrical, [n, 8], [1.0, 0.0], [2.0, 0.5], [OPEN, RING], 0.05, |[r, _z]: [f64; 2]| (1.0, [DRIFT, ELL / r, 0.0], 0.0));
            ell_departure(&end, 1, |[r, _z]| r, |[r, _z]| (1.25..1.75).contains(&r))
        })
        .collect();
    converges("rz isothermal ell departure", &measured, 0.6);
}

#[test]
fn a_radial_flow_carries_uniform_specific_angular_momentum_spherical_adiabatic() {
    let measured: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = adiabatic_run!(Spherical, Spherical, [n, n], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [OPEN, OPEN], 0.05, |[r, t]: [f64; 2]| (1.0, [DRIFT, 0.0, ELL / (r * t.sin())], 1.0));
            ell_departure(&end, 2, |[r, t]| r * t.sin(), |[r, t]| (1.25..1.75).contains(&r) && (1.0..PI - 1.0).contains(&t))
        })
        .collect();
    converges("spherical adiabatic ell departure", &measured, 0.6);
}

#[test]
fn a_radial_flow_carries_uniform_specific_angular_momentum_spherical_isothermal() {
    let measured: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = isothermal_run!(Spherical, Spherical, [n, n], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [OPEN, OPEN], 0.05, |[r, t]: [f64; 2]| (1.0, [DRIFT, 0.0, ELL / (r * t.sin())], 0.0));
            ell_departure(&end, 2, |[r, t]| r * t.sin(), |[r, t]| (1.25..1.75).contains(&r) && (1.0..PI - 1.0).contains(&t))
        })
        .collect();
    converges("spherical isothermal ell departure", &measured, 0.6);
}

// --------------------- locally isothermal ---------------------

/// cs^2 holding uniform density in rigid rotation at cylindrical radius `big_r`
fn rigid_local_cs2(big_r: f64) -> f64 {
    1.0 + 0.5 * OMEGA * OMEGA * big_r * big_r
}

#[test]
fn rigid_rotation_is_steady_on_the_rz_plane_locally_isothermal() {
    let residual: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = isothermal_run!(
                Cylindrical, Cylindrical, [n, 8], [1.0, 0.0], [2.0, 0.5], [OPEN, RING], 0.01,
                |[r, _z]: [f64; 2]| (1.0, [0.0, OMEGA * r, 0.0], 0.0),
                |[r, _z]: [f64; 2]| rigid_local_cs2(r)
            );
            meridional_residual(&end, |[r, _z]| (1.25..1.75).contains(&r), [0, 2])
        })
        .collect();
    converges("rz locally isothermal rigid rotation", &residual, 0.6);
}

#[test]
fn rigid_rotation_is_steady_on_the_spherical_plane_locally_isothermal() {
    let residual: Vec<f64> = [32, 64, 128]
        .into_iter()
        .map(|n| {
            let (_, end) = isothermal_run!(
                Spherical, Spherical, [n, n], [1.0, FRAC_PI_4], [2.0, 3.0 * FRAC_PI_4], [OPEN, OPEN], 0.01,
                |[r, theta]: [f64; 2]| (1.0, [0.0, 0.0, OMEGA * r * theta.sin()], 0.0),
                |[r, theta]: [f64; 2]| rigid_local_cs2(r * theta.sin())
            );
            meridional_residual(&end, |[r, theta]| (1.25..1.75).contains(&r) && (1.0..PI - 1.0).contains(&theta), [0, 1])
        })
        .collect();
    converges("spherical locally isothermal rigid rotation", &residual, 0.6);
}

/// cs^2 varying across the x flow; density 1 / cs^2 makes the pressure uniform
fn striped_cs2(y: f64) -> f64 {
    1.0 + 0.3 * (2.0 * PI * y).sin()
}

#[test]
fn the_cartesian_plane_advects_the_out_of_plane_velocity_locally_isothermal() {
    let (mut errors, mut transverse) = (Vec::new(), Vec::new());
    for n in [32, 64, 128] {
        let (a, b) = isothermal_run!(
            Cartesian, Cartesian, [n, n], [0.0, 0.0], [1.0, 1.0], [RING, RING], T_ADVECT,
            |[x, y]: [f64; 2]| (1.0 / striped_cs2(y), [U, 0.0, (2.0 * PI * x).sin()], 0.0),
            |[_x, y]: [f64; 2]| striped_cs2(y)
        );
        // the riemann fan diffuses the standing density stripe while cs^2 stays put, so the pressure
        // balance holds to a truncation the transverse speed reports
        errors.push(transverse_advection_checks("cartesian locally isothermal", &a, &b, f64::INFINITY));
        transverse.push(largest(b.vel[1].iter().map(|v| v.abs())));
    }
    converges("cartesian locally isothermal advection L1", &errors, 0.5);
    converges("cartesian locally isothermal transverse speed", &transverse, 0.6);
}
