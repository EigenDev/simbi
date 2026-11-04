// =============================================================================
// cold_surface_stage_replay.rs
//
// the production stage on the captured pre-failure state of the cold surface torus
// (simbi/simulation/tests/fixtures/cold_surface_torus.py, captured at step 2000 by
// fixtures/capture_cold_surface.py). a replay runs fold_stage itself, so the kernels,
// the order and the stage weights are the run's.
// - the regression: the replay reaches the end time with every interior cell holding a positive
//   density and internal energy at every stage
// - the audit: at the first inadmissible stage, an independent recomputation of the update from the
//   stage's stored fluxes and the closed-form geometry must reproduce the kernel's cells, and the
//   accounting of density, momenta, angular momentum, total and kinetic energy prints per face and
//   per source. the same operands then give the source-form update and the donor-work candidate
// - the sweeps: the failing step at first order and at smaller timesteps, then the whole window at
//   smaller timesteps, first order and the source-form control
//
// run: cargo test -p symbi --test cold_surface_stage_replay -- --nocapture
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_sim::driver::{downstream_injection_weight, needs_step_snapshot, stage_exit_fraction, stage_schedule};
use symbi_sim::stage::{HookPoint, StageArgs, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

const FIXTURE: &[u8] = include_bytes!("fixtures/cold_surface_torus_84x44_step2000.bin");
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.1;
const THETA: f64 = 1.5;
const NI: usize = 84;
const NJ: usize = 44;
const R_LO: f64 = 0.01;
const R_HI: f64 = 4.0;
const STEP0: u64 = 2000;
const SOFTENING: f64 = 0.009;
/// past the frontend run's crash at t = 0.0110636
const T_END: f64 = 0.0115;

fn field_of(index: usize) -> impl Iterator<Item = f64> {
    let start = 8 + index * NI * NJ * 8;
    FIXTURE[start..start + NI * NJ * 8].chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()))
}

fn captured() -> Sim {
    assert_eq!(FIXTURE.len(), 8 + 5 * NI * NJ * 8, "the fixture's size");
    let slope = (R_HI / R_LO).log10() / NI as f64;
    let dtheta = PI / NJ as f64;
    let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
    let mut sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([NI, NJ])
        .origin([R_LO, 0.0])
        .spacing([(R_HI - R_LO) / NI as f64, dtheta])
        .coord_maps(Some([AxisMap::Log { start: R_LO, log_slope: slope }, AxisMap::Uniform { start: 0.0, dx: dtheta }]))
        .boundaries(Boundaries::per_axis(walls))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("the torus grid")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(1.0)))
        .build()
        .with_bodies(BodyCollection::new().add(
            Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.0, SOFTENING).with_softening_kind(SofteningKind::Compact),
        ));
    let time = f64::from_le_bytes(FIXTURE[..8].try_into().unwrap());
    let cells: Vec<[isize; 2]> = (0..NJ as isize).flat_map(|j| (0..NI as isize).map(move |i| [i, j])).collect();
    let fill = |f: &symbi_grid::Field<f64, 2, HostMemory>, index| {
        for (c, v) in cells.iter().zip(field_of(index)) {
            f.view_mut().set(*c, v);
        }
    };
    fill(&sim.fields.cons.den, 0);
    for k in 0..3 {
        fill(&sim.fields.cons.mom[k], k + 1);
    }
    fill(sim.fields.cons.nrg_field().expect("cons.nrg"), 4);
    sim.time = time;
    sim.iteration = STEP0;
    sim
}

/// one interior or face-indexed array, i fastest
#[derive(Clone)]
struct Store {
    ni: usize,
    v: Vec<f64>,
}

impl Store {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.v[i + self.ni * j]
    }
}

fn store(f: &symbi_grid::Field<f64, 2, HostMemory>, ni: usize, nj: usize) -> Store {
    let view = f.view();
    Store { ni, v: (0..nj as isize).flat_map(|j| (0..ni as isize).map(move |i| [i, j])).map(|c| *view.at(c)).collect() }
}

#[derive(Clone)]
struct Cons {
    den: Store,
    mom: [Store; 3],
    nrg: Store,
}

fn cons_of(c: &ConsFieldsGeneric<2, 3, HostMemory, f64>, ni: usize, nj: usize) -> Cons {
    Cons {
        den: store(&c.den, ni, nj),
        mom: std::array::from_fn(|k| store(&c.mom[k], ni, nj)),
        nrg: store(c.nrg_field().expect("nrg"), ni, nj),
    }
}

/// everything one stage read and wrote
#[derive(Clone)]
struct Frame {
    step: u64,
    stage: usize,
    time: f64,
    dt: f64,
    a0: f64,
    ac: f64,
    input: Cons,
    u_n: Cons,
    rho: Store,
    vel: [Store; 3],
    pre: Store,
    flux: [Cons; 2],
    output: Cons,
}

#[derive(Clone, Copy)]
struct Options {
    theta: f64,
    dt_factor: f64,
    control: bool,
    /// the first step taking the options; earlier steps run the production settings
    from_step: u64,
    t_end: f64,
    /// an energy flux evaluated from the stage's operands and added before the recovery
    energy: EnergyTerm,
}

/// the energy updates the replay can evaluate on a stage's own operands, outside the kernels
#[derive(Clone, Copy, PartialEq)]
enum EnergyTerm {
    Kernel,
    /// the single-valued flux that leaves the mass flux's receiver free of lever work
    DonorWork,
    /// the point mass's work from the face mass fluxes: E + rho Phi holds a flux form
    PotentialFlux,
}

const PRODUCTION: Options =
    Options { theta: THETA, dt_factor: 1.0, control: false, from_step: STEP0, t_end: T_END, energy: EnergyTerm::Kernel };

struct Outcome {
    /// the frame of the first inadmissible stage, with the cell holding the lowest internal energy
    first: Option<(Frame, [usize; 2], f64)>,
    time: f64,
    steps: u64,
    /// (time, lowest specific internal energy over the interior) at each step
    history: Vec<(f64, f64)>,
    /// the accumulated departure of (angular momentum, E + rho Phi) from their boundary fluxes,
    /// each over the initial magnitude
    drift: [f64; 2],
}

fn lowest_internal(c: &Cons) -> ([usize; 2], f64) {
    let mut worst = ([0, 0], f64::INFINITY);
    for j in 0..NJ {
        for i in 0..NI {
            let den = c.den.at(i, j);
            let ke = 0.5 * (0..3).map(|k| c.mom[k].at(i, j).powi(2)).sum::<f64>() / den;
            let specific = (c.nrg.at(i, j) - ke) / den;
            if !(den > 0.0) || specific < worst.1 {
                worst = ([i, j], if den > 0.0 { specific } else { f64::NEG_INFINITY });
            }
        }
    }
    worst
}

fn kernels_for(sim: &Sim, theta: f64, control: bool) -> Kern {
    let mut k = Kern::new(GAMMA, CFL, &sim.geom.allocated).theta(theta);
    k.azimuthal_source_control = control;
    k
}

fn replay(options: Options) -> Outcome {
    let mut sim = captured();
    let production = kernels_for(&sim, THETA, false);
    let chosen = kernels_for(&sim, options.theta, options.control);
    KernelSet::c2p(&production, &sim);
    KernelSet::ghost_fill(&production, &sim);
    let mut history = Vec::new();
    let mut steps = 0;
    let start = totals(&cons_of(&sim.fields.cons, NI, NJ));
    let mut wall = [0.0, 0.0];
    while sim.time < options.t_end {
        let active = sim.iteration >= options.from_step;
        let k: &Kern = if active { &chosen } else { &production };
        let cfl_dt = KernelSet::cfl(k, &sim);
        assert!(cfl_dt > 0.0, "the timestep at step {} is {cfl_dt}", sim.iteration);
        let dt = if active { cfl_dt * options.dt_factor } else { cfl_dt };
        sim.dt = dt;
        let stages = sim.timestepping.stages();
        if needs_step_snapshot(stages) {
            KernelSet::snapshot(k, &sim);
        }
        for spec in stage_schedule(stages) {
            assert_eq!(stages.len(), 2, "the boundary ledger weights each of the two stages by one half");
            let input = cons_of(&sim.fields.cons, NI, NJ);
            let u_n = cons_of(&sim.workspace.u_n, NI, NJ);
            let rho = store(&sim.fields.prim.rho, NI, NJ);
            let vel = std::array::from_fn(|k| store(&sim.fields.prim.vel[k], NI, NJ));
            let pre = store(sim.fields.prim.pre_field().expect("pre"), NI, NJ);
            let mut flux = None;
            let mut output = None;
            let store_ref: &FieldStore<2, 3, HostMemory, f64> = &sim;
            let stage_outcome = fold_stage(
                store_ref,
                k,
                StageArgs {
                    dt,
                    a0: spec.a0,
                    ac: spec.ac,
                    stage: spec.index,
                    n_stages: stages.len(),
                    injection_weight: downstream_injection_weight(stages, spec.index),
                    allow_elision: true,
                    exit_fraction: stage_exit_fraction(stages, spec.index),
                },
                &mut |point| match point {
                    HookPoint::AfterFlux => {
                        flux = Some([cons_of(&store_ref.fields.flux[0], NI + 1, NJ), cons_of(&store_ref.fields.flux[1], NI, NJ + 1)]);
                    }
                    HookPoint::BeforeC2p => {
                        if active && options.energy != EnergyTerm::Kernel {
                            let probe = Frame {
                                step: 0,
                                stage: spec.index,
                                time: 0.0,
                                dt,
                                a0: spec.a0,
                                ac: spec.ac,
                                input: input.clone(),
                                u_n: u_n.clone(),
                                rho: rho.clone(),
                                vel: vel.clone(),
                                pre: pre.clone(),
                                flux: flux.clone().expect("the stage's fluxes"),
                                output: cons_of(&store_ref.fields.cons, NI, NJ),
                            };
                            let nrg = store_ref.fields.cons.nrg_field().expect("cons.nrg");
                            for j in 0..NJ {
                                for i in 0..NI {
                                    let added = match options.energy {
                                        EnergyTerm::DonorWork => donor_work_update(&probe, i, j),
                                        _ => potential_flux_update(&probe, i, j),
                                    };
                                    let cell = [i as isize, j as isize];
                                    let held = *nrg.view().at(cell);
                                    nrg.view_mut().set(cell, held + added);
                                }
                            }
                        }
                        output = Some(cons_of(&store_ref.fields.cons, NI, NJ));
                    }
                    _ => {}
                },
            );
            std::hint::black_box(&stage_outcome);
            let output = output.expect("the stage reached its recovery");
            let (cell, specific) = lowest_internal(&output);
            let through_walls = boundary_rates(flux.as_ref().expect("the stage's fluxes"));
            wall[0] += 0.5 * dt * through_walls[0];
            wall[1] += 0.5 * dt * through_walls[1];
            if !(specific > 0.0) {
                let frame = Frame {
                    step: sim.iteration,
                    stage: spec.index,
                    time: sim.time,
                    dt,
                    a0: spec.a0,
                    ac: spec.ac,
                    input,
                    u_n,
                    rho,
                    vel,
                    pre,
                    flux: flux.expect("the stage's fluxes"),
                    output,
                };
                return Outcome { first: Some((frame, cell, specific)), time: sim.time, steps, history, drift: [f64::NAN; 2] };
            }
        }
        sim.time += dt;
        sim.iteration += 1;
        steps += 1;
        let (_, specific) = lowest_internal(&cons_of(&sim.fields.cons, NI, NJ));
        history.push((sim.time, specific));
    }
    let end = totals(&cons_of(&sim.fields.cons, NI, NJ));
    let drift = std::array::from_fn(|k| (end[k] - start[k] + wall[k]) / start[k].abs());
    Outcome { first: None, time: sim.time, steps, history, drift }
}

// ------------------------------- closed-form geometry -------------------------------

struct Geometry {
    r: [f64; 2],
    t: [f64; 2],
    r_c: f64,
    t_c: f64,
    volume: f64,
    /// area on [axis][side]
    area: [[f64; 2]; 2],
    /// the distance from the axis at each face center
    lever: [[f64; 2]; 2],
    lever_c: f64,
}

fn geometry(i: usize, j: usize) -> Geometry {
    let slope = (R_HI / R_LO).log10() / NI as f64;
    let r = [R_LO * 10f64.powf(i as f64 * slope), R_LO * 10f64.powf((i + 1) as f64 * slope)];
    let dt = PI / NJ as f64;
    let t = [j as f64 * dt, (j + 1) as f64 * dt];
    let i_r2 = (r[1].powi(3) - r[0].powi(3)) / 3.0;
    let i_r = (r[1] * r[1] - r[0] * r[0]) / 2.0;
    let i_sin = t[0].cos() - t[1].cos();
    let two_pi = 2.0 * PI;
    let r_c = (r[1].powi(4) - r[0].powi(4)) / 4.0 / i_r2;
    let t_c = ((t[1].sin() - t[1] * t[1].cos()) - (t[0].sin() - t[0] * t[0].cos())) / i_sin;
    Geometry {
        r,
        t,
        r_c,
        t_c,
        volume: i_r2 * i_sin * two_pi,
        area: [[r[0] * r[0] * i_sin * two_pi, r[1] * r[1] * i_sin * two_pi], [i_r * t[0].sin() * two_pi, i_r * t[1].sin() * two_pi]],
        lever: [[r[0] * t_c.sin(), r[1] * t_c.sin()], [r_c * t[0].sin(), r_c * t[1].sin()]],
        lever_c: r_c * t_c.sin(),
    }
}

/// the flux of a component on one face of a cell
fn face_flux(frame: &Frame, comp: usize, d: usize, side: usize, i: usize, j: usize) -> f64 {
    let f = &frame.flux[d];
    let (fi, fj) = if d == 0 { (i + side, j) } else { (i, j + side) };
    match comp {
        0 => f.den.at(fi, fj),
        1..=3 => f.mom[comp - 1].at(fi, fj),
        _ => f.nrg.at(fi, fj),
    }
}

/// the geometric sources of (M_r, M_theta) and the source-form azimuthal term, with each piece apart
struct Sources {
    pressure: [f64; 2],
    inertial: [f64; 2],
    azimuthal_source_form: f64,
    gravity: f64,
}

fn sources(frame: &Frame, g: &Geometry, i: usize, j: usize) -> Sources {
    let p = frame.pre.at(i, j);
    let m: [f64; 3] = std::array::from_fn(|k| frame.input.mom[k].at(i, j));
    let v: [f64; 3] = std::array::from_fn(|k| frame.vel[k].at(i, j));
    let cot = g.t_c.cos() / g.t_c.sin();
    Sources {
        pressure: [p * (g.area[0][1] - g.area[0][0]) / g.volume, p * (g.area[1][1] - g.area[1][0]) / g.volume],
        inertial: [(m[1] * v[1] + m[2] * v[2]) / g.r_c, (m[2] * v[2] * cot - m[0] * v[1]) / g.r_c],
        azimuthal_source_form: -m[2] * (v[0] + v[1] * cot) / g.r_c,
        gravity: -1.0 / (g.r_c * g.r_c),
    }
}

/// the azimuthal momentum's rate on a cell, in the torque form or in the source form
fn azimuthal_rate(frame: &Frame, g: &Geometry, s: &Sources, i: usize, j: usize, torque: bool) -> f64 {
    let mut rate = if torque { 0.0 } else { s.azimuthal_source_form };
    for d in 0..2 {
        for side in 0..2 {
            let sign = if side == 1 { -1.0 } else { 1.0 };
            let weight = if torque { g.lever[d][side] / g.lever_c } else { 1.0 };
            rate += sign * g.area[d][side] * weight * face_flux(frame, 3, d, side, i, j) / g.volume;
        }
    }
    rate
}

/// the stage's conserved update of one cell, recomputed from the stored fluxes: [den, m_r, m_t, m_p, nrg]
fn recomputed(frame: &Frame, i: usize, j: usize, torque: bool) -> [f64; 5] {
    let g = geometry(i, j);
    let s = sources(frame, &g, i, j);
    let div = |comp: usize| -> f64 {
        (0..2).map(|d| (g.area[d][1] * face_flux(frame, comp, d, 1, i, j) - g.area[d][0] * face_flux(frame, comp, d, 0, i, j)) / g.volume).sum()
    };
    let (rho, dt, ac, a0) = (frame.input.den.at(i, j), frame.dt, frame.ac, frame.a0);
    let stage = |un: f64, u: f64, rate: f64| a0 * un + ac * (u + dt * rate);
    let body = ac * dt * rho * s.gravity;
    let work = ac * dt * frame.input.mom[0].at(i, j) * s.gravity;
    [
        stage(frame.u_n.den.at(i, j), rho, -div(0)),
        stage(frame.u_n.mom[0].at(i, j), frame.input.mom[0].at(i, j), -div(1) + s.pressure[0] + s.inertial[0]) + body,
        stage(frame.u_n.mom[1].at(i, j), frame.input.mom[1].at(i, j), -div(2) + s.pressure[1] + s.inertial[1]),
        stage(frame.u_n.mom[2].at(i, j), frame.input.mom[2].at(i, j), azimuthal_rate(frame, &g, &s, i, j, torque)),
        stage(frame.u_n.nrg.at(i, j), frame.input.nrg.at(i, j), -div(4)) + work,
    ]
}

fn internal(c: &[f64; 5]) -> f64 {
    c[4] - 0.5 * (c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / c[0]
}

/// the single-valued energy flux correction that leaves the receiving cell's azimuthal kinetic
/// energy paid: the receiver of the mass flux takes no lever term, the donor pays it
fn donor_work(frame: &Frame, d: usize, fi: usize, fj: usize) -> f64 {
    let (lo, hi) = if d == 0 { (fi.wrapping_sub(1), fi) } else { (fj.wrapping_sub(1), fj) };
    let limit = if d == 0 { NI } else { NJ };
    if hi >= limit || lo >= limit {
        return 0.0;
    }
    let cell = |k: usize| if d == 0 { (k, fj) } else { (fi, k) };
    let mass = frame.flux[d].den.at(fi, fj);
    let f_phi = frame.flux[d].mom[2].at(fi, fj);
    let receiver = if mass >= 0.0 { hi } else { lo };
    let (ci, cj) = cell(receiver);
    let g = geometry(ci, cj);
    let side = usize::from(receiver == lo);
    f_phi * frame.vel[2].at(ci, cj) * (g.lever[d][side] / g.lever_c - 1.0)
}

/// a cell's stage update in the torque form with the candidate's energy flux
fn candidate(frame: &Frame, i: usize, j: usize) -> [f64; 5] {
    let mut u = recomputed(frame, i, j, true);
    let g = geometry(i, j);
    for d in 0..2 {
        for side in 0..2 {
            let (fi, fj) = if d == 0 { (i + side, j) } else { (i, j + side) };
            let sign = if side == 1 { -1.0 } else { 1.0 };
            u[4] += frame.ac * frame.dt * sign * g.area[d][side] * donor_work(frame, d, fi, fj) / g.volume;
        }
    }
    u
}

/// the point mass's potential, outside the softening length at every cell and face of this grid
fn potential(r: f64) -> f64 {
    assert!(r > SOFTENING, "the softened interior needs its own potential");
    -1.0 / r
}

/// (rate of E) from the work of the face mass fluxes: \sum_f w_f F_rho,f (Phi_f - Phi_c). the
/// polar faces sit at the centroid radius, so only the radial faces work on the point mass's field
fn potential_flux_rate(frame: &Frame, g: &Geometry, i: usize, j: usize) -> f64 {
    let phi_c = potential(g.r_c);
    (0..2)
        .map(|side| {
            let sign = if side == 1 { -1.0 } else { 1.0 };
            sign * g.area[0][side] / g.volume * face_flux(frame, 0, 0, side, i, j) * (potential(g.r[side]) - phi_c)
        })
        .sum()
}

/// a stage's change of E when the face work replaces the kernel's work M_r g
fn potential_flux_update(frame: &Frame, i: usize, j: usize) -> f64 {
    let g = geometry(i, j);
    let kernel_work = frame.input.mom[0].at(i, j) * sources(frame, &g, i, j).gravity;
    frame.ac * frame.dt * (potential_flux_rate(frame, &g, i, j) - kernel_work)
}

/// a stage's change of E from the donor-work energy flux
fn donor_work_update(frame: &Frame, i: usize, j: usize) -> f64 {
    let g = geometry(i, j);
    let mut added = 0.0;
    for d in 0..2 {
        for side in 0..2 {
            let (fi, fj) = if d == 0 { (i + side, j) } else { (i, j + side) };
            let sign = if side == 1 { -1.0 } else { 1.0 };
            added += frame.ac * frame.dt * sign * g.area[d][side] * donor_work(frame, d, fi, fj) / g.volume;
        }
    }
    added
}

/// the volume integrals of (angular momentum, E + rho Phi)
fn totals(c: &Cons) -> [f64; 2] {
    let mut sum = [0.0, 0.0];
    for j in 0..NJ {
        for i in 0..NI {
            let g = geometry(i, j);
            sum[0] += c.mom[2].at(i, j) * g.lever_c * g.volume;
            sum[1] += (c.nrg.at(i, j) + c.den.at(i, j) * potential(g.r_c)) * g.volume;
        }
    }
    sum
}

/// the outward rates of (angular momentum, E + rho Phi) through the radial walls
fn boundary_rates(flux: &[Cons; 2]) -> [f64; 2] {
    let mut rate = [0.0, 0.0];
    for j in 0..NJ {
        let (inner, outer) = (geometry(0, j), geometry(NI - 1, j));
        let f = &flux[0];
        rate[0] += outer.area[0][1] * outer.lever[0][1] * f.mom[2].at(NI, j) - inner.area[0][0] * inner.lever[0][0] * f.mom[2].at(0, j);
        rate[1] += outer.area[0][1] * (f.nrg.at(NI, j) + f.den.at(NI, j) * potential(outer.r[1]))
            - inner.area[0][0] * (f.nrg.at(0, j) + f.den.at(0, j) * potential(inner.r[0]));
    }
    rate
}

/// the internal-energy rate of a cell per the rates of its azimuthal momentum and gravity choices
fn internal_rate(frame: &Frame, i: usize, j: usize, torque: bool, face_work: bool) -> f64 {
    let g = geometry(i, j);
    let s = sources(frame, &g, i, j);
    let v: [f64; 3] = std::array::from_fn(|k| frame.vel[k].at(i, j));
    let vsq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let div = |comp: usize| -> f64 {
        (0..2).map(|d| (g.area[d][1] * face_flux(frame, comp, d, 1, i, j) - g.area[d][0] * face_flux(frame, comp, d, 0, i, j)) / g.volume).sum()
    };
    let mut rate = -div(4) - 0.5 * vsq * div(0);
    rate -= v[0] * (-div(1) + s.pressure[0] + s.inertial[0]);
    rate -= v[1] * (-div(2) + s.pressure[1] + s.inertial[1]);
    rate -= v[2] * azimuthal_rate(frame, &g, &s, i, j, torque);
    if face_work {
        rate += potential_flux_rate(frame, &g, i, j) - frame.input.mom[0].at(i, j) * s.gravity;
    }
    rate
}

// ------------------------------- the regression -------------------------------

#[test]
fn the_cold_torus_surface_stays_admissible_through_the_captured_window() {
    let run = replay(PRODUCTION);
    if let Some((frame, cell, specific)) = &run.first {
        panic!(
            "step {} stage {} t = {}: cell {cell:?} holds specific internal energy {specific:e}",
            frame.step, frame.stage, frame.time
        );
    }
    assert!(run.time >= T_END, "the replay stopped at {}", run.time);
}

// ------------------------------- the audit -------------------------------

fn sci(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| format!("{v:+.6e}")).collect();
    format!("[{}]", parts.join(", "))
}

fn print_cell(label: &str, frame: &Frame, i: usize, j: usize) {
    let g = geometry(i, j);
    let s = sources(frame, &g, i, j);
    let (den, m) = (frame.input.den.at(i, j), [frame.input.mom[0].at(i, j), frame.input.mom[1].at(i, j), frame.input.mom[2].at(i, j)]);
    let e = frame.input.nrg.at(i, j);
    let ke = 0.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / den;
    let v: [f64; 3] = std::array::from_fn(|k| frame.vel[k].at(i, j));
    let out = [frame.output.den.at(i, j), frame.output.mom[0].at(i, j), frame.output.mom[1].at(i, j), frame.output.mom[2].at(i, j), frame.output.nrg.at(i, j)];
    eprintln!("\n=== {label}: cell [{i}, {j}], r [{:.6}, {:.6}] theta [{:.6}, {:.6}], r_c {:.6} theta_c {:.6} R_c {:.6} V {:.6e}", g.r[0], g.r[1], g.t[0], g.t[1], g.r_c, g.t_c, g.lever_c, g.volume);
    eprintln!("  stage input: rho {den:.6e} M {} v {} L/V {:.6e} E {e:.6e} KE {ke:.6e} e {:.6e} p {:.6e}", sci(&m), sci(&v), m[2] * g.lever_c, e - ke, frame.pre.at(i, j));
    eprintln!("  step entry:  rho {:.6e} M_phi {:.6e} E {:.6e}", frame.u_n.den.at(i, j), frame.u_n.mom[2].at(i, j), frame.u_n.nrg.at(i, j));
    eprintln!("  stage output: rho {:.6e} M {} E {:.6e} KE {:.6e} e {:.6e}", out[0], sci(&out[1..4]), out[4], out[4] - internal(&out), internal(&out));
    eprintln!("  sources: pressure {} inertial {} azimuthal source form {:.6e} gravity g_r {:.6e} (rho g {:.6e}, work {:.6e})", sci(&s.pressure), sci(&s.inertial), s.azimuthal_source_form, s.gravity, den * s.gravity, m[0] * s.gravity);
    let vsq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let (mut rate_mass, mut rate_e, mut rate_r, mut rate_t, mut rate_torque, mut rate_plain) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for d in 0..2 {
        for side in 0..2 {
            let sign = if side == 1 { -1.0 } else { 1.0 };
            let w = sign * g.area[d][side] / g.volume;
            let lever = g.lever[d][side] / g.lever_c;
            let f: [f64; 5] = std::array::from_fn(|c| face_flux(frame, c, d, side, i, j));
            // the neighbor across this face, for the advective estimate of the mass flux
            let nb = match (d, side) {
                (0, 0) if i > 0 => Some((i - 1, j)),
                (0, 1) if i + 1 < NI => Some((i + 1, j)),
                (1, 0) if j > 0 => Some((i, j - 1)),
                (1, 1) if j + 1 < NJ => Some((i, j + 1)),
                _ => None,
            };
            let advective = nb.map(|(a, b)| 0.5 * (frame.rho.at(i, j) * frame.vel[d].at(i, j) + frame.rho.at(a, b) * frame.vel[d].at(a, b)));
            eprintln!(
                "  axis {d} side {side}: A {:.5e} R_f {:.6} R_f/R_c {:.6} F_rho {:+.5e} (cell-mean rho v {}) F_r {:+.5e} F_t {:+.5e} F_phi {:+.5e} F_E {:+.5e}",
                g.area[d][side], g.lever[d][side], lever, f[0],
                advective.map_or("-".to_string(), |a| format!("{a:+.3e}")), f[1], f[2], f[3], f[4]
            );
            eprintln!(
                "      rates: rho {:+.5e} E {:+.5e} M_phi torque {:+.5e} plain {:+.5e} | e rate: mass {:+.5e} E {:+.5e} M_phi torque {:+.5e} plain {:+.5e}",
                w * f[0], w * f[4], w * lever * f[3], w * f[3], 0.5 * vsq * w * f[0], w * f[4], -v[2] * w * lever * f[3], -v[2] * w * f[3]
            );
            rate_mass += w * f[0];
            rate_e += w * f[4];
            rate_r += w * f[1];
            rate_t += w * f[2];
            rate_torque += w * lever * f[3];
            rate_plain += w * f[3];
        }
    }
    let flux_part = 0.5 * vsq * rate_mass + rate_e;
    let meridional = -v[0] * (rate_r + s.pressure[0] + s.inertial[0]) - v[1] * (rate_t + s.pressure[1] + s.inertial[1]);
    let new = flux_part + meridional - v[2] * rate_torque;
    let old = flux_part + meridional - v[2] * (rate_plain + s.azimuthal_source_form);
    eprintln!("  rate of e at the stage input (gravity's kinetic and total parts cancel):");
    eprintln!("    mass + energy flux {flux_part:+.6e}; meridional momenta and their sources {meridional:+.6e}");
    eprintln!("    of which v_r times the centrifugal source {:+.6e}; v_theta times the hoop {:+.6e}", -v[0] * s.inertial[0], -v[1] * s.inertial[1]);
    eprintln!("    azimuthal: torque {:+.6e}; source form flux {:+.6e} source {:+.6e}", -v[2] * rate_torque, -v[2] * rate_plain, -v[2] * s.azimuthal_source_form);
    eprintln!("    total e rate: torque {new:+.6e}; source form {old:+.6e}; the difference {:+.6e}", new - old);
    let (a, b) = (rate_torque, rate_plain + s.azimuthal_source_form);
    eprintln!("    M_phi rate: torque {a:+.6e}; source form {b:+.6e}");
    let lever_as_source = rate_torque - rate_plain;
    eprintln!("    the torque form's implied source {lever_as_source:+.6e} against the source form's {:+.6e}", s.azimuthal_source_form);
    let frozen: [f64; 5] = recomputed(frame, i, j, true);
    let old_u: [f64; 5] = recomputed(frame, i, j, false);
    let cand: [f64; 5] = candidate(frame, i, j);
    let mut with_face = frozen;
    with_face[4] += potential_flux_update(frame, i, j);
    eprintln!("  stage e with the face mass fluxes doing gravity's work: {:.6e}; its e rate {:+.6e}", internal(&with_face), internal_rate(frame, i, j, true, true));
    eprintln!("  stage e on identical operands: kernel {:.6e} torque {:.6e} source form {:.6e} donor-work candidate {:.6e}", internal(&out), internal(&frozen), internal(&old_u), internal(&cand));
}

/// each inflow face's parcel on the failing cell, per unit of received mass:
/// F_E / F_rho - v.v_c + v_c^2 / 2, split so the lever and gravity terms stand alone
fn print_inflow_budget(frame: &Frame, i: usize, j: usize) {
    let g = geometry(i, j);
    let v: [f64; 3] = std::array::from_fn(|k| frame.vel[k].at(i, j));
    eprintln!("\nheat per unit received mass on the cell's inflow faces");
    for d in 0..2 {
        for side in 0..2 {
            let f: [f64; 5] = std::array::from_fn(|c| face_flux(frame, c, d, side, i, j));
            let inflow = if side == 1 { f[0] < 0.0 } else { f[0] > 0.0 };
            if !inflow {
                continue;
            }
            let lever = g.lever[d][side] / g.lever_c;
            let parcel: [f64; 3] = std::array::from_fn(|k| f[k + 1] / f[0]);
            let v_l = parcel[2] * lever;
            let mixing = 0.5 * (v[2] - v_l).powi(2);
            let swirl_barrier = 0.5 * parcel[2] * parcel[2] - 0.5 * v_l * v_l;
            let specific_flux = f[4] / f[0];
            let residual = specific_flux - 0.5 * parcel[2] * parcel[2];
            let meridional = -v[0] * parcel[0] - v[1] * parcel[1] + 0.5 * (v[0] * v[0] + v[1] * v[1]);
            let work = if d == 0 { potential(g.r[side]) - potential(g.r_c) } else { 0.0 };
            let specific_ang = parcel[2] * g.lever[d][side];
            let neighbor = match (d, side) {
                (0, 0) => (i - 1, j),
                (0, 1) => (i + 1, j),
                (1, 0) => (i, j - 1),
                _ => (i, j + 1),
            };
            let n = geometry(neighbor.0, neighbor.1);
            eprintln!(
                "  axis {d} side {side}: v_phi at the face {:.6}, l_f {:.6} (cell {:.6}, neighbor {:.6}); F_E/F_rho {:.6}, of which beyond v_f^2/2 {:.6}",
                parcel[2], specific_ang, v[2] * g.lever_c, frame.vel[2].at(neighbor.0, neighbor.1) * n.lever_c, specific_flux, residual
            );
            eprintln!(
                "    mixing {mixing:+.6e} + flux beyond swirl {residual:+.6e} + lever barrier {swirl_barrier:+.6e} + meridional {meridional:+.6e} = torque heat {:+.6e}; face gravity {work:+.6e}; total {:+.6e}",
                mixing + residual + swirl_barrier + meridional,
                mixing + residual + swirl_barrier + meridional + work
            );
        }
    }
}

/// the sign of every interior face's pair heat and of every cell's rate on the failing stage
fn print_grid_budget(frame: &Frame) {
    eprintln!("\nthe stage's internal-energy budget over the grid (volume-integrated rates):");
    let (mut torque, mut source, mut face) = (0.0, 0.0, 0.0);
    let (mut cold_t, mut cold_s, mut cold_f) = (0usize, 0usize, 0usize);
    for j in 0..NJ {
        for i in 0..NI {
            let v = geometry(i, j).volume;
            let (a, b, c) = (internal_rate(frame, i, j, true, false), internal_rate(frame, i, j, false, false), internal_rate(frame, i, j, true, true));
            torque += a * v;
            source += b * v;
            face += c * v;
            cold_t += usize::from(a < 0.0);
            cold_s += usize::from(b < 0.0);
            cold_f += usize::from(c < 0.0);
        }
    }
    eprintln!("  total: torque {torque:+.6e} ({cold_t} cooling cells); source form {source:+.6e} ({cold_s}); torque with face gravity work {face:+.6e} ({cold_f})");
    let dense = 0.01;
    let (mut in_bracket, mut out_bracket, mut worst) = (0usize, 0usize, 0.0f64);
    let (mut neg_torque, mut neg_source, mut neg_gravity, mut faces) = (0usize, 0usize, 0usize, 0usize);
    let (mut sum_torque, mut sum_source, mut sum_gravity) = (0.0, 0.0, 0.0);
    for d in 0..2 {
        for j in 0..NJ - usize::from(d == 1) {
            for i in 0..NI - usize::from(d == 0) {
                let (ia, ja, ib, jb) = if d == 0 { (i, j, i + 1, j) } else { (i, j, i, j + 1) };
                let (ga, gb) = (geometry(ia, ja), geometry(ib, jb));
                let (fi, fj) = (ib, jb);
                let (mass, f_phi) = (frame.flux[d].den.at(fi, fj), frame.flux[d].mom[2].at(fi, fj));
                let (a, r_f) = (ga.area[d][1], ga.lever[d][1]);
                let (va, vb) = (frame.vel[2].at(ia, ja), frame.vel[2].at(ib, jb));
                let (oa, ob) = (va / ga.lever_c, vb / gb.lever_c);
                let kin = 0.5 * a * (vb * vb - va * va) * mass;
                let pair_torque = (oa - ob) * a * r_f * f_phi + kin;
                let pair_source = (va - vb) * a * f_phi + kin;
                let pair_gravity = pair_torque + a * mass * (potential(ga.r_c) - potential(gb.r_c));
                faces += 1;
                neg_torque += usize::from(pair_torque < 0.0);
                neg_source += usize::from(pair_source < 0.0);
                neg_gravity += usize::from(pair_gravity < 0.0);
                sum_torque += pair_torque;
                sum_source += pair_source;
                sum_gravity += pair_gravity;
                let (la, lb) = (va * ga.lever_c, vb * gb.lever_c);
                if mass.abs() > 1e-12 && frame.rho.at(ia, ja).max(frame.rho.at(ib, jb)) > dense {
                    let l_f = f_phi / mass * r_f;
                    if l_f >= la.min(lb) && l_f <= la.max(lb) {
                        in_bracket += 1;
                    } else {
                        out_bracket += 1;
                        worst = worst.max((l_f - la.max(lb)).max(la.min(lb) - l_f));
                    }
                }
            }
        }
    }
    eprintln!("  interior face pairs ({faces}), the azimuthal pair heat: torque {neg_torque} negative, sum {sum_torque:+.6e}; source form {neg_source} negative, sum {sum_source:+.6e}; torque with face gravity {neg_gravity} negative, sum {sum_gravity:+.6e}");
    eprintln!("  faces beside gas with rho > {dense}: face specific angular momentum inside the two cells' values {in_bracket}, outside {out_bracket}, furthest {worst:.3e}");
}

/// the discrete lever source against its continuum coefficients on the failing cell
fn print_lever_geometry(i: usize, j: usize) {
    let g = geometry(i, j);
    let radial = (g.area[0][1] * (g.lever[0][1] / g.lever_c - 1.0) - g.area[0][0] * (g.lever[0][0] / g.lever_c - 1.0)) / g.volume;
    let polar = (g.area[1][1] * (g.lever[1][1] / g.lever_c - 1.0) - g.area[1][0] * (g.lever[1][0] / g.lever_c - 1.0)) / g.volume;
    let mean_sin = (0.5 * (g.t[1] - g.t[0]) - 0.25 * ((2.0 * g.t[1]).sin() - (2.0 * g.t[0]).sin())) / (g.t[0].cos() - g.t[1].cos());
    eprintln!("\nlever weights: radial sum {radial:.6} against 1/r_c {:.6}; polar sum {polar:.6} against cot(theta_c)/r_c {:.6}", 1.0 / g.r_c, g.t_c.cos() / g.t_c.sin() / g.r_c);
    eprintln!("the cell's lever R_c {:.8} against the volume mean of R {:.8}", g.lever_c, g.r_c * mean_sin);
}

#[test]
fn the_first_inadmissible_stage_accounts_for_every_term() {
    let run = replay(PRODUCTION);
    let (frame, cell, specific) = run.first.as_ref().expect("the captured state no longer reaches an inadmissible stage; the audit is vacuous");
    eprintln!(
        "first inadmissible: step {} stage {} of the torque form, t = {:.17e}, dt = {:.6e}, a0 {} ac {}, cell {cell:?}, specific e {specific:e}",
        frame.step, frame.stage, frame.time, frame.dt, frame.a0, frame.ac
    );
    // the discrete meaning: the independent recomputation reproduces the kernel
    let mut worst = [0.0f64; 5];
    let mut l = [0.0, 0.0];
    let mut energy = [0.0, 0.0];
    for j in 0..NJ {
        for i in 0..NI {
            let u = recomputed(frame, i, j, true);
            let kernel = [frame.output.den.at(i, j), frame.output.mom[0].at(i, j), frame.output.mom[1].at(i, j), frame.output.mom[2].at(i, j), frame.output.nrg.at(i, j)];
            let scale = [frame.input.den.at(i, j), kernel[0] * 1.0, kernel[0] * 1.0, kernel[0] * 1.0, frame.input.nrg.at(i, j)];
            for c in 0..5 {
                worst[c] = worst[c].max((u[c] - kernel[c]).abs() / scale[c].abs().max(f64::MIN_POSITIVE));
            }
            let g = geometry(i, j);
            l[1] += kernel[3] * g.lever_c * g.volume;
            l[0] += frame.input.mom[2].at(i, j) * g.lever_c * g.volume;
            energy[1] += kernel[4] * g.volume;
            energy[0] += frame.input.nrg.at(i, j) * g.volume;
        }
    }
    eprintln!("recomputation against the kernel, worst relative difference [den, m_r, m_t, m_p, nrg]: {}", sci(&worst));
    eprintln!("the crash clock: t + dt = {:.17e}, against the frontend run's 1.1063582370435884e-2", frame.time + frame.dt);
    assert_eq!(frame.time + frame.dt, 1.1063582370435884e-2, "the replay's clock left the frontend run's");
    eprintln!("stage ledger: angular momentum {:.9e} -> {:.9e}, E {:.9e} -> {:.9e}", l[0], l[1], energy[0], energy[1]);
    assert!(worst.iter().all(|w| *w < 1e-9), "the recomputation does not reproduce the kernel: {worst:?}");
    let [i, j] = *cell;
    print_cell("the inadmissible cell", frame, i, j);
    print_cell("its inner radial neighbor", frame, i - 1, j);
    print_cell("its outer radial neighbor (the dense donor)", frame, i + 1, j);
    print_cell("its polar neighbor toward the pole", frame, i, j - 1);
    print_cell("its polar neighbor toward the equator", frame, i, j + 1);
    let (mut torque, mut source, mut cand, mut face) = (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut bad_t, mut bad_s, mut bad_c, mut bad_f) = (0, 0, 0, 0);
    for j in 0..NJ {
        for i in 0..NI {
            let per = |u: [f64; 5]| internal(&u) / u[0];
            let (a, b, c) = (per(recomputed(frame, i, j, true)), per(recomputed(frame, i, j, false)), per(candidate(frame, i, j)));
            let mut w = recomputed(frame, i, j, true);
            w[4] += potential_flux_update(frame, i, j);
            let d = per(w);
            face = face.min(d);
            bad_f += usize::from(d <= 0.0);
            torque = torque.min(a);
            source = source.min(b);
            cand = cand.min(c);
            bad_t += usize::from(a <= 0.0);
            bad_s += usize::from(b <= 0.0);
            bad_c += usize::from(c <= 0.0);
        }
    }
    print_inflow_budget(frame, i, j);
    print_grid_budget(frame);
    print_lever_geometry(i, j);
    eprintln!("\nthe stage on identical operands, over the grid: lowest specific e, cells at or below zero");
    eprintln!("  torque form {torque:.6e}, {bad_t}; source form {source:.6e}, {bad_s}; donor-work candidate {cand:.6e}, {bad_c}; face gravity work {face:.6e}, {bad_f}");
}

// ------------------------------- the sweeps -------------------------------

fn report(label: &str, run: &Outcome) {
    let low = run.history.iter().map(|h| h.1).fold(f64::INFINITY, f64::min);
    match &run.first {
        Some((frame, cell, specific)) => eprintln!(
            "{label}: inadmissible at t = {:.8e} (step {}, stage {}, {} steps), cell {cell:?}, specific e {specific:.3e}; lowest before {low:.3e}",
            frame.time, frame.step, frame.stage, run.steps
        ),
        None => eprintln!(
            "{label}: admissible to t = {:.8e} ({} steps), lowest specific e {low:.3e}, ledger drift [L, E + rho Phi] {}",
            run.time,
            run.steps,
            sci(&run.drift)
        ),
    }
}

/// the failing step from the captured state: at first order, then at smaller timesteps
#[test]
fn the_failing_step_at_first_order_and_at_smaller_timesteps() {
    let production = replay(PRODUCTION);
    let (frame, ..) = production.first.as_ref().expect("the captured state no longer fails; the sweep is vacuous");
    let failing = frame.step;
    report("production", &production);
    let t_after = frame.time + frame.dt;
    let window = Options { from_step: failing, t_end: t_after, ..PRODUCTION };
    report("the failing step at first order", &replay(Options { theta: 0.0, ..window }));
    for factor in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625] {
        report(&format!("the failing step at dt x {factor}"), &replay(Options { dt_factor: factor, ..window }));
    }
    report("the failing step, source form", &replay(Options { control: true, ..window }));
}

/// the failing step's input with first-order fluxes on every face: the failing cell's update here
/// is the one a first-order flux correction on its four faces produces from the same stage input
#[test]
fn the_first_order_correction_of_the_failing_cell() {
    let production = replay(PRODUCTION);
    let (frame, ..) = production.first.as_ref().expect("the captured state no longer fails; the audit is vacuous");
    let run = replay(Options { theta: 0.0, from_step: frame.step, t_end: frame.time + frame.dt, ..PRODUCTION });
    let (first_order, [i, j], specific) = run.first.as_ref().expect("the first-order step stays admissible");
    assert_eq!((first_order.step, first_order.stage), (frame.step, frame.stage), "the first-order stage left the failing stage's input");
    for (field, kept) in [(&first_order.input.den, &frame.input.den), (&first_order.input.nrg, &frame.input.nrg), (&first_order.input.mom[2], &frame.input.mom[2])] {
        assert_eq!(field.v, kept.v, "the first-order stage read a different input");
    }
    eprintln!("first-order stage on the failing input: cell [{i}, {j}], specific e {specific:e}");
    print_cell("first order, the inadmissible cell", first_order, *i, *j);
    print_inflow_budget(first_order, *i, *j);
    print_grid_budget(first_order);
}

/// the whole captured window, from step 2000: does the failure persist as the timestep decreases
#[test]
fn the_window_at_smaller_timesteps_first_order_and_the_control() {
    let whole = Options { from_step: STEP0, ..PRODUCTION };
    for factor in [1.0, 0.5, 0.25, 0.125, 0.0625] {
        let run = replay(Options { dt_factor: factor, ..whole });
        report(&format!("torque form, window at dt x {factor}"), &run);
    }
    report("torque form, window at first order", &replay(Options { theta: 0.0, ..whole }));
    report("torque form, window at first order, dt x 0.25", &replay(Options { theta: 0.0, dt_factor: 0.25, ..whole }));
    let control = replay(Options { control: true, ..whole });
    report("source form, window", &control);
    assert!(control.first.is_none(), "the source-form control left the admissible set; it no longer separates the two updates");
    report("source form, window at dt x 0.25", &replay(Options { control: true, dt_factor: 0.25, ..whole }));
}

/// each energy update on the captured window and past the frontend's end time, the correction
/// evaluated outside the kernels on the stage's operands
#[test]
fn the_candidate_energy_updates_over_the_window() {
    let far = Options { t_end: 0.02, ..PRODUCTION };
    report("source form to t = 0.02", &replay(Options { control: true, ..far }));
    report("donor work to t = 0.02", &replay(Options { energy: EnergyTerm::DonorWork, ..far }));
    report("face gravity work to t = 0.02", &replay(Options { energy: EnergyTerm::PotentialFlux, ..far }));
    report("face gravity work to t = 0.02, dt x 0.25", &replay(Options { energy: EnergyTerm::PotentialFlux, dt_factor: 0.25, ..far }));
    report("face gravity work to t = 0.02, first order", &replay(Options { energy: EnergyTerm::PotentialFlux, theta: 0.0, ..far }));
    report("face gravity work to t = 0.05", &replay(Options { energy: EnergyTerm::PotentialFlux, t_end: 0.05, ..far }));
}

/// the first-order replay's face flux from the two cell states alone: the slope vanishes at zero theta
#[test]
fn zero_theta_reconstructs_the_cell_states() {
    let run = replay(Options { theta: 0.0, ..PRODUCTION });
    let (frame, [i, j], _) = run.first.as_ref().expect("the first-order replay stays admissible; the premise is vacuous");
    let (i, j) = (*i, *j);
    let side = |ii: usize| {
        let (rho, v, p) = (frame.rho.at(ii, j), frame.vel[0].at(ii, j), frame.pre.at(ii, j));
        let c = (GAMMA * p / rho).sqrt();
        (rho, v, p, c)
    };
    let (l, r) = (side(i), side(i + 1));
    let (plus, minus) = ((l.1 + l.3).max(r.1 + r.3).max(0.0), (l.1 - l.3).min(r.1 - r.3).min(0.0));
    let hlle = (plus * l.0 * l.1 - minus * r.0 * r.1 + plus * minus * (r.0 - l.0)) / (plus - minus);
    let kernel = frame.flux[0].den.at(i + 1, j);
    eprintln!("first order mass flux at the failing cell's outer face: kernel {kernel:.9e}, HLLE of the cell states {hlle:.9e}");
    assert!((kernel - hlle).abs() < 1e-3 * hlle.abs(), "kernel {kernel:e} against the cell-state flux {hlle:e}");
}
