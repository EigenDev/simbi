// =============================================================================
// magnetized_body_gravity.rs
//
// softened newtonian body gravity on magnetized gas (adiabatic nmhd and isothermal
// imhd, 2D cartesian). a point mass M at x_b with plummer length h pulls every cell with
// g = -M r / (r^2 + h^2)^{3/2}, r = x_c - x_b, and one euler step adds
// dt rho g to the momentum and dt m.g to the total energy, both read at the
// stage input (the isothermal gas has no energy and receives the momentum alone):
// - on a uniform state the flux divergence is exactly zero, so the step changes
//   the gas by the body source alone, on either gas; every cell matches the plummer law
//   written out here, density and out-of-plane momentum stay bitwise, and every
//   face and cell field stays bitwise;
// - on a nonuniform magnetized state of either gas, one euler step with the body leaves every
//   face and cell field bitwise equal to the step without it, while the momentum
//   changes: gravity acts on the gas alone and constrained transport alone moves
//   the field;
// - at zero field the body's contribution to one euler step (the step with the
//   body less the step without it) equals the hydro gas's bit for bit, and the
//   isothermal magnetized gas's equals the isothermal hydro gas's;
// - a cell flagged troubled in the step sends the stage through the first-order
//   redo, which rebuilds every cell from the stage input; the redone step still
//   carries the body source in every cell;
// - five RK2 steps of a uniform gas in the near-uniform field of a distant mass
//   reproduce the exact free-fall momentum and energy, the energy's t^2 work term
//   included, up to the field's order-1/R departure from uniform.
//
// run: cargo test -p symbi --test magnetized_body_gravity
// =============================================================================

use std::f64::consts::TAU;
use std::ops::ControlFlow;
use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::Newtonian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::{MhdPrim, MhdPrimG};
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

type Mhd = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Hydro = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type IsoMhd = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type IsoHydro = SimStateGeneric<IsoNewtonian, 2, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 32;
const DX: f64 = 2.0 / N as f64;
const GAMMA: f64 = 5.0 / 3.0;
const MASS: f64 = 0.05;
const SOFT: f64 = 0.1;
const BODY: [f64; 2] = [0.13, -0.07];
const T_END: f64 = 1e-3;

/// the gas state at a point: density, velocity, pressure and cell field
type State = (f64, [f64; 3], f64, [f64; 3]);

fn uniform(_: [f64; 2]) -> State {
    (1.3, [0.3, -0.2, 0.1], 1.0, [0.5, 0.3, 0.2])
}

/// a smooth flow threaded by a divergence-free field: B_x varies along y, B_y along x
fn wave([x, y]: [f64; 2]) -> State {
    let rho = 1.0 + 0.2 * (0.5 * TAU * x).sin() * (0.5 * TAU * y).sin();
    let v = [0.3 * (0.5 * TAU * y).sin(), 0.3 * (0.5 * TAU * x).sin(), 0.1];
    (rho, v, 1.0, [face_x(y), face_y(x), 0.2])
}

fn face_x(y: f64) -> f64 {
    0.5 + 0.1 * (0.5 * TAU * y).cos()
}

fn face_y(x: f64) -> f64 {
    0.3 + 0.1 * (0.5 * TAU * x).cos()
}

/// the conserved and magnetic state of one interior cell, in grid order
#[derive(Clone, Debug)]
struct Cell {
    den: f64,
    mom: Vec<f64>,
    nrg: f64,
    field: Vec<f64>,
}

/// the cell-centered state over the interior and the bits of every face, closing faces included
struct Snapshot {
    cells: Vec<([usize; 2], Cell)>,
    faces: Vec<u64>,
}

fn body() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::gravitational(0, Tensor::new(BODY), Tensor::zeros(), MASS, 0.0, SOFT))
}

fn snapshot<const DOF: usize>(s: &FieldStore<2, DOF, HostMemory, f64>) -> Snapshot {
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let c = &s.fields.cons;
    let cells = s
        .geom
        .interior
        .iter()
        .map(|x| {
            let field = s.fields.mhd.as_ref().map_or(Vec::new(), |m| m.bcell.b.iter().map(|f| *f.view().at(x)).collect());
            let cell = Cell {
                den: *c.den.view().at(x),
                mom: c.mom.iter().map(|f| *f.view().at(x)).collect(),
                // the isothermal gas carries no energy; its slot reads zero
                nrg: c.nrg_field().map_or(0.0, |f| *f.view().at(x)),
                field,
            };
            ([(x[0] - lo[0]) as usize, (x[1] - lo[1]) as usize], cell)
        })
        .collect();
    let mut faces = Vec::new();
    if let Some(m) = s.fields.mhd.as_ref() {
        for (d, f) in m.bface.b.iter().enumerate() {
            for ii in 0..N + usize::from(d == 0) {
                for jj in 0..N + usize::from(d == 1) {
                    faces.push(f.view().at([lo[0] + ii as isize, lo[1] + jj as isize]).to_bits());
                }
            }
        }
    }
    Snapshot { cells, faces }
}

fn build_mhd(state: fn([f64; 2]) -> State, with_body: bool) -> Mhd {
    build_mhd_walled(state, with_body, BoundaryType::Outflow)
}

fn build_mhd_walled(state: fn([f64; 2]) -> State, with_body: bool, walls: BoundaryType) -> Mhd {
    let mut s = Mhd::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([DX, DX])
        .boundaries(Boundaries::uniform(walls))
        .cfl(0.3)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim")
        .set_initial(move |x| {
            let (rho, v, p, b) = state(x);
            MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p)), Tensor::new(b))
        })
        .seed_faces(move |d, x| state(x).3[d])
        .build();
    if with_body {
        s = s.with_bodies(body());
    }
    s
}

fn build_hydro(state: fn([f64; 2]) -> State, with_body: bool) -> Hydro {
    let mut s = Hydro::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([DX, DX])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim")
        .set_initial(move |x| {
            let (rho, v, p, _) = state(x);
            Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1]]), Pressure(p))
        })
        .build();
    if with_body {
        s = s.with_bodies(body());
    }
    s
}

/// one accepted euler step on the single-grid driver: the step's dt, the troubled cells it
/// booked, the state before and the state after
fn step_mhd(s: Mhd, troubled: Option<[usize; 2]>) -> (f64, u64, Snapshot, Snapshot) {
    if let Some(cell) = troubled {
        let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
        let local = [lo[0] + cell[0] as isize, lo[1] + cell[1] as isize];
        *s.workspace.troubled_once.lock().unwrap() = Some((0, 0, local, false));
    }
    let k = s.substrate();
    let mut hier = Hierarchy::single(s, k);
    let before = snapshot(&hier.levels[0].state);
    let d = hier.evolve_with_callback(T_END, 1, |_| ControlFlow::Break(())).expect("one step");
    let state = &hier.levels[0].state;
    (state.time, d.guards.troubled_cells.total, before, snapshot(state))
}

fn step_hydro(s: Hydro) -> (f64, Snapshot) {
    let k = s.substrate();
    let mut hier = Hierarchy::single(s, k);
    hier.evolve_with_callback(T_END, 1, |_| ControlFlow::Break(())).expect("one step");
    let state = &hier.levels[0].state;
    (state.time, snapshot(state))
}

/// the plummer acceleration at the center of cell `idx`, written out from the law
fn plummer(idx: [usize; 2]) -> [f64; 2] {
    let r = [-1.0 + (idx[0] as f64 + 0.5) * DX - BODY[0], -1.0 + (idx[1] as f64 + 0.5) * DX - BODY[1]];
    let r_eff = (r[0] * r[0] + r[1] * r[1] + SOFT * SOFT).sqrt();
    [-MASS * r[0] / r_eff.powi(3), -MASS * r[1] / r_eff.powi(3)]
}

/// the largest departure of a uniform-state step from the plummer source, over every cell's
/// momentum and energy change, and the largest momentum change it was measured against
fn source_departure(dt: f64, before: &Snapshot, after: &Snapshot) -> (f64, f64) {
    let (mut worst, mut largest) = (0.0f64, 0.0f64);
    for ((idx, b), (_, a)) in before.cells.iter().zip(&after.cells) {
        let g = plummer(*idx);
        assert_eq!(a.den.to_bits(), b.den.to_bits(), "cell {idx:?}: gravity changed the density");
        assert_eq!(a.mom[2].to_bits(), b.mom[2].to_bits(), "cell {idx:?}: in-plane gravity changed the out-of-plane momentum");
        for ax in 0..2 {
            let want = dt * b.den * g[ax];
            worst = worst.max((a.mom[ax] - b.mom[ax] - want).abs());
            largest = largest.max(want.abs());
        }
        let work = dt * (b.mom[0] * g[0] + b.mom[1] * g[1]);
        worst = worst.max((a.nrg - b.nrg - work).abs());
    }
    (worst, largest)
}

#[test]
fn gravity_accelerates_uniform_magnetized_gas_by_the_plummer_law() {
    let (dt, _, before, after) = step_mhd(build_mhd(uniform, true), None);
    assert!(dt > 0.0, "the step advanced no time");
    let (worst, largest) = source_departure(dt, &before, &after);
    assert!(largest > 1e-5, "the largest momentum change is {largest:.3e}; the gate is vacuous");
    // each change is the difference of conserved values of order one, and the penalization
    // removes and restores the magnetic energy around the drain, so round-off enters at the ulp
    // of the conserved values (measured 1.1e-16); the change it is measured against reaches 2e-3
    assert!(worst < 1e-15, "the step departs from the plummer source by {worst:.3e} (largest change {largest:.3e})");
    for ((idx, b), (_, a)) in before.cells.iter().zip(&after.cells) {
        for (cc, (fb, fa)) in b.field.iter().zip(&a.field).enumerate() {
            assert_eq!(fa.to_bits(), fb.to_bits(), "cell {idx:?}: the cell field {cc} changed");
        }
    }
    assert_eq!(after.faces, before.faces, "gravity changed a face field on a uniform state");
}

#[test]
fn gravity_leaves_the_magnetic_field_to_constrained_transport() {
    let (dt_with, _, _, with) = step_mhd(build_mhd(wave, true), None);
    let (dt_without, _, _, without) = step_mhd(build_mhd(wave, false), None);
    assert_eq!(dt_with.to_bits(), dt_without.to_bits(), "the body changed the timestep; the arms are not comparable");
    let moved = with.cells.iter().zip(&without.cells).filter(|((_, a), (_, b))| a.mom[0] != b.mom[0] || a.mom[1] != b.mom[1]).count();
    assert!(moved > N * N / 2, "the body changed the momentum of {moved} cells; the gate is vacuous");
    assert_eq!(with.faces, without.faces, "the body changed a face field");
    for ((idx, a), (_, b)) in with.cells.iter().zip(&without.cells) {
        assert_eq!(a.field, b.field, "cell {idx:?}: the body changed the cell field");
    }
}

/// the step with the body less the step without it, per cell: density, momentum, energy
fn body_part(with: &Snapshot, without: &Snapshot) -> Vec<[f64; 4]> {
    with.cells
        .iter()
        .zip(&without.cells)
        .map(|((_, a), (_, b))| [a.den - b.den, a.mom[0] - b.mom[0], a.mom[1] - b.mom[1], a.nrg - b.nrg])
        .collect()
}

fn zero_field(x: [f64; 2]) -> State {
    let (rho, v, p, _) = wave(x);
    (rho, [v[0], v[1], 0.0], p, [0.0; 3])
}

#[test]
fn at_zero_field_the_body_acts_on_magnetized_gas_as_on_hydro_gas() {
    let (dt_mhd, _, _, mhd_with) = step_mhd(build_mhd(zero_field, true), None);
    let (_, _, _, mhd_without) = step_mhd(build_mhd(zero_field, false), None);
    let (dt_hydro, hydro_with) = step_hydro(build_hydro(zero_field, true));
    let (_, hydro_without) = step_hydro(build_hydro(zero_field, false));
    // the two regimes' fluxes differ, and each body part cancels its own regime's flux, so the
    // parts compare at one shared dt
    assert_eq!(dt_mhd.to_bits(), dt_hydro.to_bits(), "the regimes took different timesteps");
    let mhd = body_part(&mhd_with, &mhd_without);
    let hydro = body_part(&hydro_with, &hydro_without);
    let largest = mhd.iter().map(|p| p[1].abs().max(p[2].abs())).fold(0.0f64, f64::max);
    assert!(largest > 1e-5, "the magnetized body part peaks at {largest:.3e}; the gate is vacuous");
    let worst = mhd.iter().zip(&hydro).flat_map(|(a, b)| (0..4).map(move |q| (a[q] - b[q]).abs())).fold(0.0f64, f64::max);
    // both regimes apply the one body kernel to the stage input after the flux update, so the
    // parts agree bit for bit
    assert!(worst == 0.0, "the body part differs between the regimes by {worst:.3e} (dt {dt_mhd:.3e}, peak {largest:.3e})");
}

#[test]
fn the_first_order_redo_keeps_the_body_source() {
    let (dt, troubled, before, after) = step_mhd(build_mhd(uniform, true), Some([18, 15]));
    assert!(troubled >= 1, "the flagged cell booked no troubled census; the redo never ran");
    let (worst, largest) = source_departure(dt, &before, &after);
    assert!(largest > 1e-5, "the largest momentum change is {largest:.3e}; the gate is vacuous");
    // the uniform state's first-order flux divergence is also exactly zero, so the bound is the
    // uniform step's (measured 1.1e-16)
    assert!(worst < 1e-15, "the redone step departs from the plummer source by {worst:.3e} (largest change {largest:.3e})");
}



/// a point mass at distance `FAR` along x, softened over the same length, pulling with unit
/// acceleration at the box center: across the box its field departs from uniform by order 1/FAR
const FAR: f64 = 4.0e3;
const FAR_MASS: f64 = 2.828_427_124_746_19 * FAR * FAR;

fn far_plummer(idx: [usize; 2]) -> [f64; 2] {
    let r = [-1.0 + (idx[0] as f64 + 0.5) * DX - FAR, -1.0 + (idx[1] as f64 + 0.5) * DX];
    let r_eff = (r[0] * r[0] + r[1] * r[1] + FAR * FAR).sqrt();
    [-FAR_MASS * r[0] / r_eff.powi(3), -FAR_MASS * r[1] / r_eff.powi(3)]
}

/// the uniform gas of `uniform` pulled by the distant mass after `steps` RK2 steps on the
/// single-grid driver, with the time reached
fn rk2_far(steps: usize) -> (f64, Snapshot, Snapshot) {
    // periodic walls keep a uniformly accelerating gas uniform; an outflow wall treats the side
    // the gas enters from differently from the side it leaves
    let mut s = build_mhd_walled(uniform, false, BoundaryType::Periodic);
    s.timestepping = Timestepping::Rk2;
    let mass = Body::gravitational(0, Tensor::new([FAR, 0.0]), Tensor::zeros(), FAR_MASS, 0.0, FAR);
    let s = s.with_bodies(BodyCollection::new().add(mass));
    let k = s.substrate();
    let mut hier = Hierarchy::single(s, k);
    // the ghost layer is filled before the first step stands on it
    hier.prime();
    let before = snapshot(&hier.levels[0].state);
    for _ in 0..steps {
        hier.step_root_with_dt(T_END);
    }
    let state = &hier.levels[0].state;
    (state.time, before, snapshot(state))
}

#[test]
fn rk2_integrates_the_free_fall_work_to_second_order() {
    // in a uniform field a uniform gas stays uniform, its flux divergence stays zero, and each cell
    // evolves by m' = rho g, E' = m.g at fixed rho: m = m0 + rho g t and
    // E = E0 + m0.g t + rho |g|^2 t^2 / 2. the source evaluated at each stage input and summed with
    // the stage weights reproduces the t^2 term; n steps of a first-order update miss 1/n of it,
    // 20% here. the distant mass's field departs from uniform by order 1/FAR across the periodic
    // box, which leaves a flux divergence of that relative order in the later stages: the
    // departure measures 4.3%, 2.2% and 1.1% of the t^2 term at FAR = 1e3, 2e3 and 4e3
    let (t, before, after) = rk2_far(5);
    assert!((t - 5.0 * T_END).abs() < 1e-15, "the run reached t = {t}; the timestep was clipped");
    let (mut worst, mut quadratic) = (0.0f64, 0.0f64);
    for ((idx, b), (_, a)) in before.cells.iter().zip(&after.cells) {
        let g = far_plummer(*idx);
        for ax in 0..2 {
            worst = worst.max((a.mom[ax] - (b.mom[ax] + t * b.den * g[ax])).abs());
        }
        let g2 = g[0] * g[0] + g[1] * g[1];
        let exact = b.nrg + t * (b.mom[0] * g[0] + b.mom[1] * g[1]) + 0.5 * t * t * b.den * g2;
        worst = worst.max((a.nrg - exact).abs());
        quadratic = quadratic.max(0.5 * t * t * b.den * g2);
    }
    assert!(quadratic > 1e-5, "the t^2 work term peaks at {quadratic:.3e}; the gate cannot see it");
    assert!(worst < 5e-2 * quadratic, "five RK2 steps depart from the exact free fall by {worst:.3e}, against a t^2 work term of {quadratic:.3e}");
}

fn build_iso_mhd(state: fn([f64; 2]) -> State, with_body: bool) -> IsoMhd {
    let mut s = IsoMhd::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([DX, DX])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim")
        .set_initial(move |x| {
            let (rho, v, _, b) = state(x);
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(rho), Tensor::new(v)), Tensor::new(b))
        })
        .seed_faces(move |d, x| state(x).3[d])
        .build();
    if with_body {
        s = s.with_bodies(body());
    }
    s
}

fn build_iso_hydro(state: fn([f64; 2]) -> State, with_body: bool) -> IsoHydro {
    let mut s = IsoHydro::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N])
        .origin([-1.0, -1.0])
        .spacing([DX, DX])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim")
        .set_initial(move |x| {
            let (rho, v, _, _) = state(x);
            PrimG::<f64, 2, IsoModel>::isothermal(Density(rho), Tensor::new([v[0], v[1]]))
        })
        .build();
    if with_body {
        s = s.with_bodies(body());
    }
    s
}

/// one accepted euler step of the isothermal magnetized gas: the step's dt, the state before and
/// the state after
fn step_iso_mhd(s: IsoMhd) -> (f64, Snapshot, Snapshot) {
    let k = s.substrate();
    let mut hier = Hierarchy::single(s, k);
    let before = snapshot(&hier.levels[0].state);
    hier.evolve_with_callback(T_END, 1, |_| ControlFlow::Break(())).expect("one step");
    let state = &hier.levels[0].state;
    (state.time, before, snapshot(state))
}

fn step_iso_hydro(s: IsoHydro) -> (f64, Snapshot) {
    let k = s.substrate();
    let mut hier = Hierarchy::single(s, k);
    hier.evolve_with_callback(T_END, 1, |_| ControlFlow::Break(())).expect("one step");
    let state = &hier.levels[0].state;
    (state.time, snapshot(state))
}

#[test]
fn gravity_accelerates_uniform_isothermal_magnetized_gas_by_the_plummer_law() {
    let (dt, before, after) = step_iso_mhd(build_iso_mhd(uniform, true));
    let (mut worst, mut largest) = (0.0f64, 0.0f64);
    for ((idx, b), (_, a)) in before.cells.iter().zip(&after.cells) {
        let g = plummer(*idx);
        assert_eq!(a.den.to_bits(), b.den.to_bits(), "cell {idx:?}: gravity changed the density");
        assert_eq!(a.mom[2].to_bits(), b.mom[2].to_bits(), "cell {idx:?}: in-plane gravity changed the out-of-plane momentum");
        for ax in 0..2 {
            let want = dt * b.den * g[ax];
            worst = worst.max((a.mom[ax] - b.mom[ax] - want).abs());
            largest = largest.max(want.abs());
        }
        assert_eq!(a.field, b.field, "cell {idx:?}: gravity changed the cell field");
    }
    assert!(largest > 1e-5, "the largest momentum change is {largest:.3e}; the gate is vacuous");
    // the momentum change is the difference of order-one values, so round-off enters at their ulp
    assert!(worst < 1e-15, "the step departs from the plummer source by {worst:.3e} (largest change {largest:.3e})");
    assert_eq!(after.faces, before.faces, "gravity changed a face field on a uniform state");
}

#[test]
fn gravity_leaves_the_isothermal_magnetic_field_to_constrained_transport() {
    let (dt_with, _, with) = step_iso_mhd(build_iso_mhd(wave, true));
    let (dt_without, _, without) = step_iso_mhd(build_iso_mhd(wave, false));
    assert_eq!(dt_with.to_bits(), dt_without.to_bits(), "the body changed the timestep; the arms are not comparable");
    let moved = with.cells.iter().zip(&without.cells).filter(|((_, a), (_, b))| a.mom[0] != b.mom[0] || a.mom[1] != b.mom[1]).count();
    assert!(moved > N * N / 2, "the body changed the momentum of {moved} cells; the gate is vacuous");
    assert_eq!(with.faces, without.faces, "the body changed a face field");
    for ((idx, a), (_, b)) in with.cells.iter().zip(&without.cells) {
        assert_eq!(a.field, b.field, "cell {idx:?}: the body changed the cell field");
    }
}

#[test]
fn at_zero_field_the_body_acts_on_isothermal_magnetized_gas_as_on_isothermal_hydro_gas() {
    let (dt_mhd, _, mhd_with) = step_iso_mhd(build_iso_mhd(zero_field, true));
    let (_, _, mhd_without) = step_iso_mhd(build_iso_mhd(zero_field, false));
    let (dt_hydro, hydro_with) = step_iso_hydro(build_iso_hydro(zero_field, true));
    let (_, hydro_without) = step_iso_hydro(build_iso_hydro(zero_field, false));
    assert_eq!(dt_mhd.to_bits(), dt_hydro.to_bits(), "the regimes took different timesteps");
    let mhd = body_part(&mhd_with, &mhd_without);
    let hydro = body_part(&hydro_with, &hydro_without);
    let largest = mhd.iter().map(|p| p[1].abs().max(p[2].abs())).fold(0.0f64, f64::max);
    assert!(largest > 1e-5, "the magnetized body part peaks at {largest:.3e}; the gate is vacuous");
    let worst = mhd.iter().zip(&hydro).flat_map(|(a, b)| (0..3).map(move |q| (a[q] - b[q]).abs())).fold(0.0f64, f64::max);
    // the closure-field kernel forms the same momentum source from the same stage-input density
    assert!(worst == 0.0, "the body part differs between the isothermal regimes by {worst:.3e} (peak {largest:.3e})");
}
