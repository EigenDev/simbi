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
//   per source. the same operands then give the source-form update and the donor-work candidate.
//   the audit runs the cell momentum's gravity work, the update that fails on this state
// - the sweeps: the failing step at first order and at smaller timesteps, then the whole window at
//   smaller timesteps, first order and the source-form control
// - the face work: the kernel's update against its evaluation from the stage's operands, and the
//   flux form of E + rho Phi with its boundary fluxes, for Phi and for Phi plus a constant
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
use symbi_sim::stage::{HookPoint, StageArgs, StageOutcome, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

/// a captured state the replay starts from: the interior conserved fields, the step and the walls
#[derive(Clone, Copy)]
struct Fixture {
    bytes: &'static [u8],
    step: u64,
    /// the layout: the time then the interior fields, or a header (time, step, halo, rows,
    /// columns) then the haloed conserved and primitive fields
    haloed: bool,
    /// the outer radial wall passes outflow and refuses inflow
    outward_outer: bool,
}

/// the cold surface torus at step 2000 (fixtures/capture_cold_surface.py)
const COLD: Fixture = Fixture { bytes: include_bytes!("fixtures/cold_surface_torus_84x44_step2000.bin"), step: 2000, haloed: false, outward_outer: false };
/// the inviscid SPB torus with the outward-only outer wall at step 2061947
/// (fixtures/capture_outward_wall_surface.py)
const OUTWARD: Fixture =
    Fixture { bytes: include_bytes!("fixtures/outward_wall_surface_n84_step2061947.bin"), step: 2061947, haloed: true, outward_outer: true };
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

fn word(bytes: &[u8], k: usize) -> f64 {
    f64::from_le_bytes(bytes[8 * k..8 * k + 8].try_into().unwrap())
}

/// conserved field `index` (den, m1, m2, m3, nrg) of the fixture's interior, i fastest
fn field_of(fx: &Fixture, index: usize) -> Vec<f64> {
    if !fx.haloed {
        return (0..NI * NJ).map(|c| word(fx.bytes, 1 + index * NI * NJ + c)).collect();
    }
    let (ng, rows, cols) = (word(fx.bytes, 2) as usize, word(fx.bytes, 3) as usize, word(fx.bytes, 4) as usize);
    assert_eq!((rows, cols), (NJ + 2 * ng, NI + 2 * ng), "the fixture's haloed shape");
    (0..NJ).flat_map(|j| (0..NI).map(move |i| (j, i))).map(|(j, i)| word(fx.bytes, 5 + index * rows * cols + (j + ng) * cols + i + ng)).collect()
}

fn captured(fx: &Fixture) -> Sim {
    if fx.haloed {
        let (ng, rows, cols) = (word(fx.bytes, 2) as usize, word(fx.bytes, 3) as usize, word(fx.bytes, 4) as usize);
        assert_eq!(fx.bytes.len(), 8 * (5 + 10 * rows * cols), "the fixture's size");
        assert_eq!((word(fx.bytes, 1) as u64, ng), (fx.step, 2), "the fixture's step and halo");
    } else {
        assert_eq!(fx.bytes.len(), 8 + 5 * NI * NJ * 8, "the fixture's size");
    }
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
    let time = word(fx.bytes, 0);
    let cells: Vec<[isize; 2]> = (0..NJ as isize).flat_map(|j| (0..NI as isize).map(move |i| [i, j])).collect();
    let fill = |f: &symbi_grid::Field<f64, 2, HostMemory>, index| {
        for (c, v) in cells.iter().zip(field_of(fx, index)) {
            f.view_mut().set(*c, v);
        }
    };
    fill(&sim.fields.cons.den, 0);
    for k in 0..3 {
        fill(&sim.fields.cons.mom[k], k + 1);
    }
    fill(sim.fields.cons.nrg_field().expect("cons.nrg"), 4);
    sim.time = time;
    sim.iteration = fx.step;
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
    fixture: Fixture,
    theta: f64,
    dt_factor: f64,
    control: bool,
    /// the first step taking the options; earlier steps run the production settings
    from_step: u64,
    t_end: f64,
    /// an energy flux evaluated from the stage's operands and added before the recovery
    energy: EnergyTerm,
    /// the cell momentum's gravity work in place of the admitted face mass flux's
    cell_work: bool,
    /// past a stage whose output the first-order correction repaired, keep marching; otherwise
    /// the replay ends at the first inadmissible stage output
    continue_corrected: bool,
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

/// the update that fails on the captured state
const PRODUCTION: Options =
    Options {
        fixture: COLD,
        theta: THETA,
        dt_factor: 1.0,
        control: false,
        from_step: STEP0,
        t_end: T_END,
        energy: EnergyTerm::Kernel,
        cell_work: true,
        continue_corrected: false,
    };
/// the kernels the failing configuration selects
const ADMITTED: Options = Options { cell_work: false, ..PRODUCTION };

struct Outcome {
    /// the frame of the first inadmissible stage, with the cell holding the lowest internal energy
    first: Option<(Frame, [usize; 2], f64)>,
    time: f64,
    steps: u64,
    /// (time, lowest specific internal energy over the interior) after each completed step
    history: Vec<(f64, f64)>,
    /// the lowest specific internal energy over the interior at every stage's output
    lowest_stage: f64,
    /// the frames of the last stages run, one per stage of the last step
    last_frames: Vec<Frame>,
    /// the conserved state at the replay's end
    last_state: Cons,
    /// the accumulated departure of (angular momentum, E + rho Phi) from their boundary fluxes,
    /// each over the initial magnitude
    drift: [f64; 2],
    /// the stages whose inadmissible output the first-order correction repaired
    corrected: u64,
    /// a stage the production stage asked to retry, which this replay does not take
    retried: Option<(u64, usize)>,
    /// the first accepted stage whose corrected state stayed outside the admissible set
    failed_after: Option<(u64, usize, [usize; 2], f64)>,
}

/// the cell with the lowest specific internal energy. a cell with a non-finite conserved value, a
/// non-finite internal energy or a density at or below zero is the answer at once, reported as
/// NaN or negative infinity, so a caller's `specific > 0` rejects it
fn lowest_internal(c: &Cons) -> ([usize; 2], f64) {
    let mut worst = ([0, 0], f64::INFINITY);
    for j in 0..NJ {
        for i in 0..NI {
            let den = c.den.at(i, j);
            let mom: [f64; 3] = std::array::from_fn(|k| c.mom[k].at(i, j));
            let nrg = c.nrg.at(i, j);
            let ke = 0.5 * mom.iter().map(|m| m * m).sum::<f64>() / den;
            let specific = (nrg - ke) / den;
            if !den.is_finite() || !nrg.is_finite() || mom.iter().any(|m| !m.is_finite()) || !specific.is_finite() {
                return ([i, j], f64::NAN);
            }
            if den <= 0.0 {
                return ([i, j], f64::NEG_INFINITY);
            }
            if specific < worst.1 {
                worst = ([i, j], specific);
            }
        }
    }
    worst
}

fn kernels_for(sim: &Sim, theta: f64, control: bool, cell_work: bool, outward_outer: bool) -> Kern {
    let mut k = Kern::new(GAMMA, CFL, &sim.geom.allocated).theta(theta).with_outward_walls([[false, outward_outer], [false, false], [false, false]]);
    k.azimuthal_source_control = control;
    k.cell_momentum_gravity_work = cell_work;
    k
}

fn replay(options: Options) -> Outcome {
    let mut sim = captured(&options.fixture);
    let outward = options.fixture.outward_outer;
    let production = kernels_for(&sim, THETA, false, options.cell_work, outward);
    let chosen = kernels_for(&sim, options.theta, options.control, options.cell_work, outward);
    KernelSet::c2p(&production, &sim);
    KernelSet::ghost_fill(&production, &sim);
    let mut history = Vec::new();
    let mut steps = 0;
    let start = totals(&cons_of(&sim.fields.cons, NI, NJ));
    let mut wall = [0.0, 0.0];
    let mut lowest_stage = f64::INFINITY;
    let mut last_frames = Vec::new();
    let mut first = None;
    let mut corrected = 0;
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
        last_frames.clear();
        for spec in stage_schedule(stages) {
            assert_eq!(stages.len(), 2, "the boundary ledger weights each of the two stages by one half");
            let input = cons_of(&sim.fields.cons, NI, NJ);
            let u_n = cons_of(&sim.workspace.u_n, NI, NJ);
            let rho = store(&sim.fields.prim.rho, NI, NJ);
            let vel = std::array::from_fn(|k| store(&sim.fields.prim.vel[k], NI, NJ));
            let pre = store(sim.fields.prim.pre_field().expect("pre"), NI, NJ);
            let mut flux = None;
            let mut output = None;
            let mut redone = false;
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
                        // the first call precedes the stage's recovery; a second is the first-order
                        // correction's redo, whose state the stage then recovers
                        if output.is_none() {
                            output = Some(cons_of(&store_ref.fields.cons, NI, NJ));
                        } else {
                            redone = true;
                        }
                    }
                    _ => {}
                },
            );
            let output = output.expect("the stage reached its recovery");
            let (cell, specific) = lowest_internal(&output);
            let rejected = stage_outcome == StageOutcome::RetryStep;
            if !KernelSet::<2, 3, HostMemory, f64>::fofc_active(k) {
                // without the first-order correction the recovery rejects exactly the inadmissible outputs
                assert_eq!(
                    rejected,
                    !(specific > 0.0),
                    "step {} stage {}: outcome {stage_outcome:?} against the output's lowest specific internal energy {specific:e}",
                    sim.iteration,
                    spec.index
                );
            }
            // the state the accepted stage leaves, after any correction
            let after = cons_of(&sim.fields.cons, NI, NJ);
            lowest_stage = lowest_stage.min(specific);
            let through_walls = boundary_rates(flux.as_ref().expect("the stage's fluxes"));
            wall[0] += 0.5 * dt * through_walls[0];
            wall[1] += 0.5 * dt * through_walls[1];
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
            macro_rules! stop {
                ($first:expr, $retried:expr, $failed_after:expr, $last_state:expr) => {
                    Outcome {
                        first: $first,
                        time: sim.time,
                        steps,
                        history,
                        lowest_stage,
                        last_frames,
                        last_state: $last_state,
                        drift: [f64::NAN; 2],
                        corrected,
                        retried: $retried,
                        failed_after: $failed_after,
                    }
                };
            }
            if rejected {
                let last_state = frame.output.clone();
                let at = (frame.step, frame.stage);
                return stop!(first.or(Some((frame, cell, specific))), Some(at), None, last_state);
            }
            if !(specific > 0.0) {
                assert!(redone, "step {} stage {}: an inadmissible output was accepted without the correction's redo", frame.step, frame.stage);
                corrected += 1;
                if first.is_none() {
                    first = Some((frame.clone(), cell, specific));
                }
                if !options.continue_corrected {
                    return stop!(first, None, None, frame.output.clone());
                }
            }
            let (after_cell, after_specific) = lowest_internal(&after);
            if !(after_specific > 0.0) {
                let at = (frame.step, frame.stage, after_cell, after_specific);
                return stop!(first, None, Some(at), after);
            }
            last_frames.push(frame);
        }
        sim.time += dt;
        sim.iteration += 1;
        steps += 1;
        let (_, specific) = lowest_internal(&cons_of(&sim.fields.cons, NI, NJ));
        history.push((sim.time, specific));
    }
    let last_state = cons_of(&sim.fields.cons, NI, NJ);
    let end = totals(&last_state);
    let drift = std::array::from_fn(|k| (end[k] - start[k] + wall[k]) / start[k].abs());
    Outcome { first, time: sim.time, steps, history, lowest_stage, last_frames, last_state, drift, corrected, retried: None, failed_after: None }
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

/// the same rate with a constant added to the potential at the cell and at every face
fn potential_flux_rate_shifted(frame: &Frame, g: &Geometry, i: usize, j: usize, constant: f64) -> f64 {
    let phi_c = potential(g.r_c) + constant;
    (0..2)
        .map(|side| {
            let sign = if side == 1 { -1.0 } else { 1.0 };
            let phi_f = potential(g.r[side]) + constant;
            sign * g.area[0][side] / g.volume * face_flux(frame, 0, 0, side, i, j) * (phi_f - phi_c)
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
    let run = replay(ADMITTED);
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
    eprintln!("  faces beside gas with rho > {dense}: the flux ratio R_f F_phi / F_rho inside the two cells' l {in_bracket}, outside {out_bracket}, furthest {worst:.3e}");
}

/// one theta-limited slope's two face values of a primitive at cell `c`, from its neighbors
fn limited_faces(lo: f64, c: f64, hi: f64) -> [f64; 2] {
    let (a, b, d) = (THETA * (c - lo), 0.5 * (hi - lo), THETA * (hi - c));
    let slope = if a > 0.0 && b > 0.0 && d > 0.0 {
        a.min(b).min(d)
    } else if a < 0.0 && b < 0.0 && d < 0.0 {
        a.max(b).max(d)
    } else {
        0.0
    };
    [c - 0.5 * slope, c + 0.5 * slope]
}

/// the left and right reconstructed (rho, v_n, p, v_phi) at the interior radial face below cell `i`
fn radial_face_states(frame: &Frame, i: usize, j: usize) -> [[f64; 4]; 2] {
    let at = |field: &Store, ii: usize| field.at(ii, j);
    let side = |field: &Store| -> [f64; 2] {
        [limited_faces(at(field, i - 2), at(field, i - 1), at(field, i))[1], limited_faces(at(field, i - 1), at(field, i), at(field, i + 1))[0]]
    };
    let (rho, vn, p, vp) = (side(&frame.rho), side(&frame.vel[0]), side(&frame.pre), side(&frame.vel[2]));
    [[rho[0], vn[0], p[0], vp[0]], [rho[1], vn[1], p[1], vp[1]]]
}

/// the kernel's radial fluxes against the HLLE of the reconstructed states, then the bracket of the
/// reconstructed angular momentum beside the flux-derived ratio
fn print_reconstructed_states(frame: &Frame) {
    let (mut worst_mass, mut worst_phi) = (0.0f64, 0.0f64);
    let (mut state_inside, mut state_outside, mut ratio_inside, mut ratio_outside, mut both_inside_ratio_outside) = (0, 0, 0, 0, 0);
    for j in 0..NJ {
        for i in 2..NI - 1 {
            if frame.rho.at(i - 1, j).max(frame.rho.at(i, j)) <= 0.01 {
                continue;
            }
            let [l, r] = radial_face_states(frame, i, j);
            let sound = |s: &[f64; 4]| (GAMMA * s[2] / s[0]).sqrt();
            let plus = (l[1] + sound(&l)).max(r[1] + sound(&r)).max(0.0);
            let minus = (l[1] - sound(&l)).min(r[1] - sound(&r)).min(0.0);
            let hlle = |f_l: f64, f_r: f64, u_l: f64, u_r: f64| (plus * f_l - minus * f_r + plus * minus * (u_r - u_l)) / (plus - minus);
            let mass = hlle(l[0] * l[1], r[0] * r[1], l[0], r[0]);
            let phi = hlle(l[0] * l[1] * l[3], r[0] * r[1] * r[3], l[0] * l[3], r[0] * r[3]);
            let (kernel_mass, kernel_phi) = (frame.flux[0].den.at(i, j), frame.flux[0].mom[2].at(i, j));
            worst_mass = worst_mass.max((mass - kernel_mass).abs() / kernel_mass.abs().max(1e-300));
            worst_phi = worst_phi.max((phi - kernel_phi).abs() / kernel_phi.abs().max(1e-300));
            let (ga, gb) = (geometry(i - 1, j), geometry(i, j));
            let (la, lb) = (frame.vel[2].at(i - 1, j) * ga.lever_c, frame.vel[2].at(i, j) * gb.lever_c);
            let inside = |l: f64| l >= la.min(lb) && l <= la.max(lb);
            let r_f = ga.lever[0][1];
            let states_in = inside(l[3] * r_f) && inside(r[3] * r_f);
            let ratio_in = inside(r_f * kernel_phi / kernel_mass);
            if states_in { state_inside += 1 } else { state_outside += 1 }
            if ratio_in { ratio_inside += 1 } else { ratio_outside += 1 }
            if states_in && !ratio_in {
                both_inside_ratio_outside += 1;
            }
        }
    }
    eprintln!("\nreconstructed radial face states beside gas with rho > 0.01:");
    eprintln!("  HLLE of the reconstructed states against the kernel, worst relative: mass {worst_mass:.3e}, azimuthal {worst_phi:.3e}");
    eprintln!("  R_f v_phi of both states inside the two cells' l: {state_inside}, outside {state_outside}");
    eprintln!("  flux ratio inside: {ratio_inside}, outside {ratio_outside}; states inside with the ratio outside: {both_inside_ratio_outside}");
    assert!(worst_mass < 1e-9 && worst_phi < 1e-9, "the reconstruction model leaves the kernel's flux: mass {worst_mass:e}, azimuthal {worst_phi:e}");
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
    print_reconstructed_states(frame);
    eprintln!("\nthe stage on identical operands, over the grid: lowest specific e, cells at or below zero");
    eprintln!("  torque form {torque:.6e}, {bad_t}; source form {source:.6e}, {bad_s}; donor-work candidate {cand:.6e}, {bad_c}; face gravity work {face:.6e}, {bad_f}");
}

// ------------------------------- the sweeps -------------------------------

fn report(label: &str, run: &Outcome) {
    let low = run.history.iter().map(|h| h.1).fold(f64::INFINITY, f64::min);
    let stage_low = run.lowest_stage;
    match &run.first {
        Some((frame, cell, specific)) => eprintln!(
            "{label}: inadmissible at t = {:.8e} (step {}, stage {}, {} steps), cell {cell:?}, specific e {specific:.3e}; lowest after a completed step {low:.3e}",
            frame.time, frame.step, frame.stage, run.steps
        ),
        None => eprintln!(
            "{label}: admissible to t = {:.8e} ({} steps), lowest specific e over stages {stage_low:.3e}, after steps {low:.3e}, ledger drift [L, E + rho Phi] {}",
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

/// the admitted kernels' run: the target reached, every stage admissible, and angular momentum
/// and E + rho Phi closing against their wall fluxes at roundoff. a run of N steps accumulates at
/// most about N * 2.2e-16 relative: 8090 steps at dt / 4 give 1.8e-12, the bound asserted
fn assert_admissible(label: &str, run: &Outcome, t_end: f64) {
    report(label, run);
    assert!(run.first.is_none(), "{label}: left the admissible set");
    assert!(run.time >= t_end, "{label}: stopped at t = {}", run.time);
    assert!(run.lowest_stage > 0.0, "{label}: lowest stage value {}", run.lowest_stage);
    assert!(run.drift.iter().all(|d| d.abs() < 2e-12), "{label}: ledger drift {:?}", run.drift);
}

#[test]
fn the_face_work_kernels_stay_admissible_past_the_frontend_end_time() {
    let far = Options { t_end: 0.02, ..ADMITTED };
    assert_admissible("kernel face work to t = 0.02", &replay(far), 0.02);
    assert_admissible("kernel face work to t = 0.02, dt x 0.25", &replay(Options { dt_factor: 0.25, ..far }), 0.02);
    assert_admissible("kernel face work to t = 0.02, first order", &replay(Options { theta: 0.0, ..far }), 0.02);
    assert_admissible("kernel face work to t = 0.05", &replay(Options { t_end: 0.05, ..far }), 0.05);
}

/// the donor-work energy update and the source form, measured beside the kernels and asserted on
/// nothing but the control's separation: these rows inform the choice of energy update
#[test]
fn the_energy_updates_beside_the_kernels_are_measured() {
    let far = Options { t_end: 0.02, ..PRODUCTION };
    report("source form to t = 0.02", &replay(Options { control: true, ..far }));
    report("donor work to t = 0.02", &replay(Options { energy: EnergyTerm::DonorWork, ..far }));
}

/// the kernel's face work against its evaluation from the stage's operands: one step from the
/// captured state, then the whole window. the two arms differ only in the order of roundoff per step
#[test]
fn the_face_work_kernel_matches_its_evaluation_from_the_operands() {
    let offline = Options { energy: EnergyTerm::PotentialFlux, ..PRODUCTION };
    let one_step = |o: Options| replay(Options { t_end: word(COLD.bytes, 0) + 1e-12, ..o });
    let (kernel, evaluated) = (one_step(ADMITTED), one_step(offline));
    assert_eq!(kernel.steps, 1, "the comparison window holds one step");
    let closeness = |a: &Cons, b: &Cons| -> f64 {
        let mut worst = 0.0f64;
        for j in 0..NJ {
            for i in 0..NI {
                worst = worst.max((a.nrg.at(i, j) - b.nrg.at(i, j)).abs() / a.nrg.at(i, j).abs());
                worst = worst.max((a.den.at(i, j) - b.den.at(i, j)).abs() / a.den.at(i, j).abs());
                for k in 0..3 {
                    let scale = a.mom[k].at(i, j).abs().max(a.den.at(i, j));
                    worst = worst.max((a.mom[k].at(i, j) - b.mom[k].at(i, j)).abs() / scale);
                }
            }
        }
        worst
    };
    let step = closeness(&kernel.last_state, &evaluated.last_state);
    let (kernel, evaluated) = (replay(ADMITTED), replay(offline));
    assert!(kernel.first.is_none() && evaluated.first.is_none(), "an arm left the admissible set");
    assert_eq!(kernel.steps, evaluated.steps, "the two arms took different step counts");
    let window = closeness(&kernel.last_state, &evaluated.last_state);
    eprintln!("kernel against its evaluation: one step {step:.3e}; {} steps {window:.3e}", kernel.steps);
    // a stage's closed-form potential and the kernel's cartesian distance differ by ulps
    assert!(step < 1e-13, "one step: {step:e}");
    assert!(window < 1e-11, "the window: {window:e}");
}

/// the conserved flux form on every stage of the kernel's last step, cell by cell and over the grid
/// with its wall fluxes, for the potential and for the potential plus a constant. a stage's residual
/// is roundoff of the terms it sums: a cell at 1e-13 of its largest term, the grid at 1e-13 of the
/// grid's summed magnitude
#[test]
fn e_plus_rho_phi_closes_in_flux_form_for_any_constant_in_the_potential() {
    let run = replay(ADMITTED);
    assert!(run.first.is_none(), "the kernels left the admissible set; the identity reads a failed state");
    assert_eq!(run.last_frames.len(), 2, "the two stages of the last step");
    for frame in &run.last_frames {
        for constant in [0.0, 7.0, -1e3] {
            let (mut grid, mut grid_scale, mut worst) = (0.0, 0.0, 0.0f64);
            for j in 0..NJ {
                for i in 0..NI {
                    let g = geometry(i, j);
                    let phi_c = potential(g.r_c) + constant;
                    let tot = |c: &Cons| c.nrg.at(i, j) + c.den.at(i, j) * phi_c;
                    let mut divergence = 0.0;
                    let mut magnitude: f64 = [&frame.input, &frame.u_n, &frame.output]
                        .iter()
                        .map(|c| c.nrg.at(i, j).abs() + c.den.at(i, j).abs() * phi_c.abs())
                        .sum();
                    for d in 0..2 {
                        for side in 0..2 {
                            let sign = if side == 1 { 1.0 } else { -1.0 };
                            let phi_f = constant + if d == 0 { potential(g.r[side]) } else { potential(g.r_c) };
                            let (f_e, f_rho) = (face_flux(frame, 4, d, side, i, j), face_flux(frame, 0, d, side, i, j));
                            let weight = g.area[d][side] / g.volume;
                            divergence += sign * weight * (f_e + phi_f * f_rho);
                            magnitude += frame.dt * weight * (f_e.abs() + (phi_f * f_rho).abs());
                        }
                    }
                    let residual = tot(&frame.output) - (frame.a0 * tot(&frame.u_n) + frame.ac * (tot(&frame.input) - frame.dt * divergence));
                    worst = worst.max(residual.abs() / magnitude);
                    let swept = (tot(&frame.output) - frame.a0 * tot(&frame.u_n) - frame.ac * tot(&frame.input)) * g.volume;
                    grid += swept;
                    grid_scale += magnitude * g.volume;
                }
            }
            let mut walls = 0.0;
            for j in 0..NJ {
                let (inner, outer) = (geometry(0, j), geometry(NI - 1, j));
                let f = &frame.flux[0];
                walls += outer.area[0][1] * (f.nrg.at(NI, j) + f.den.at(NI, j) * (potential(outer.r[1]) + constant));
                walls -= inner.area[0][0] * (f.nrg.at(0, j) + f.den.at(0, j) * (potential(inner.r[0]) + constant));
            }
            let closure = (grid + frame.ac * frame.dt * walls).abs() / grid_scale;
            eprintln!("stage {}, constant {constant}: cell residual {worst:.3e}, grid closure {closure:.3e}", frame.stage);
            assert!(worst < 1e-13, "stage {} constant {constant}: cell residual {worst:e}", frame.stage);
            assert!(closure < 1e-13, "stage {} constant {constant}: grid closure {closure:e}", frame.stage);
        }
        for j in 0..NJ {
            for i in 0..NI {
                let g = geometry(i, j);
                let (a, b) = (potential_flux_rate(frame, &g, i, j), potential_flux_rate_shifted(frame, &g, i, j, 7.0));
                assert!((a - b).abs() <= 1e-13 * a.abs().max(1e-30) + 1e-14 * frame.flux[0].den.at(i, j).abs(), "the constant moved the rate at [{i}, {j}]: {a:e} against {b:e}");
            }
        }
    }
}

/// a planted non-finite or non-positive value is rejected wherever it sits, and a clean state is not
#[test]
fn the_admissibility_scan_rejects_planted_values() {
    let clean = || {
        let held = |v: f64, ni: usize, nj: usize| Store { ni, v: vec![v; ni * nj] };
        Cons { den: held(1.0, NI, NJ), mom: std::array::from_fn(|_| held(0.0, NI, NJ)), nrg: held(1.0, NI, NJ) }
    };
    assert_eq!(lowest_internal(&clean()).1, 1.0, "a uniform state's specific internal energy");
    let cell = 7 + NI * 11;
    let mut c = clean();
    c.nrg.v[cell] = f64::NAN;
    assert!(lowest_internal(&c).1.is_nan(), "NaN energy");
    let mut c = clean();
    c.mom[2].v[cell] = f64::NAN;
    assert!(lowest_internal(&c).1.is_nan(), "NaN momentum");
    let mut c = clean();
    c.nrg.v[cell] = f64::INFINITY;
    assert!(lowest_internal(&c).1.is_nan(), "infinite energy");
    let mut c = clean();
    c.den.v[cell] = f64::NAN;
    assert!(lowest_internal(&c).1.is_nan(), "NaN density");
    let mut c = clean();
    c.den.v[cell] = 0.0;
    assert!(!(lowest_internal(&c).1 > 0.0), "zero density");
    let mut c = clean();
    c.den.v[cell] = -1.0;
    assert!(!(lowest_internal(&c).1 > 0.0), "negative density");
    let mut c = clean();
    c.nrg.v[cell] = -1e-30;
    assert!(!(lowest_internal(&c).1 > 0.0), "negative internal energy");
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

// ------------------------------- the outward-wall surface -------------------------------

/// the frontend run's failure: the step whose stage left the admissible set ends at this clock,
/// and the next step's cfl timestep is not positive
const OUTWARD_FAILED_STEP: u64 = 2062990;
const OUTWARD_FAILED_CLOCK: f64 = 9.605410191424152;
/// the inviscid SPB torus with the outward-only outer wall, its face gravity work, from step
/// 2061947 through t = 9.62
const OUTWARD_RUN: Options = Options { fixture: OUTWARD, from_step: 2061947, t_end: 9.62, cell_work: false, ..PRODUCTION };

/// a stage's update of one cell with the face gravity work, recomputed from the stored fluxes
fn recomputed_face_work(frame: &Frame, i: usize, j: usize) -> [f64; 5] {
    let mut u = recomputed(frame, i, j, true);
    u[4] += potential_flux_update(frame, i, j);
    u
}

/// the production stage through the window with its first-order correction: every accepted stage's
/// corrected state admissible, no retry requested, the correction firing where the high-order
/// output left the admissible set, and angular momentum and E + rho Phi closing against the wall
/// fluxes. the window is about 4400 steps, each rounding the totals at relative 2.2e-16, so the
/// drift stays below 2e-12
#[test]
fn the_outward_wall_surface_stays_admissible_through_its_window() {
    let run = replay(Options { continue_corrected: true, ..OUTWARD_RUN });
    if let Some((step, stage, cell, specific)) = run.failed_after {
        panic!("step {step} stage {stage}: the corrected state leaves cell {cell:?} with specific internal energy {specific:e}");
    }
    assert!(run.retried.is_none(), "the stage asked to retry at {:?}; the corrected path did not hold", run.retried);
    assert!(run.time >= OUTWARD_RUN.t_end, "the replay stopped at {}", run.time);
    let (frame, ..) = run.first.as_ref().expect("no stage output left the admissible set; the correction was never exercised");
    assert_eq!(frame.step + 1, OUTWARD_FAILED_STEP, "the first repaired stage is not the recorded failure's");
    assert!(run.corrected > 0, "the correction never fired");
    eprintln!("{} corrected stages; drift [L, E + rho Phi] {}", run.corrected, sci(&run.drift));
    assert!(run.drift.iter().all(|d| d.abs() < 2e-12), "ledger drift {:?}", run.drift);
}

#[test]
fn the_outward_wall_surface_failure_accounts_for_every_term() {
    let run = replay(OUTWARD_RUN);
    let (frame, cell, specific) = run.first.as_ref().expect("the captured state no longer reaches an inadmissible stage; the audit is vacuous");
    eprintln!(
        "first inadmissible: step {} stage {}, t = {:.17e}, dt = {:.6e}, a0 {} ac {}, cell {cell:?}, specific e {specific:e}",
        frame.step, frame.stage, frame.time, frame.dt, frame.a0, frame.ac
    );
    // the replay is the frontend run: the failing step and its clock
    assert_eq!((frame.step + 1, frame.time + frame.dt), (OUTWARD_FAILED_STEP, OUTWARD_FAILED_CLOCK), "the replay left the frontend run");
    // the independent recomputation from the stage's stored fluxes and the closed-form sources
    let mut worst = [0.0f64; 5];
    for j in 0..NJ {
        for i in 0..NI {
            let u = recomputed_face_work(frame, i, j);
            let kernel = [frame.output.den.at(i, j), frame.output.mom[0].at(i, j), frame.output.mom[1].at(i, j), frame.output.mom[2].at(i, j), frame.output.nrg.at(i, j)];
            let scale = [frame.input.den.at(i, j), kernel[0], kernel[0], kernel[0], frame.input.nrg.at(i, j)];
            for c in 0..5 {
                worst[c] = worst[c].max((u[c] - kernel[c]).abs() / scale[c].abs().max(f64::MIN_POSITIVE));
            }
        }
    }
    eprintln!("recomputation against the kernel, worst relative difference [den, m_r, m_t, m_p, nrg]: {}", sci(&worst));
    assert!(worst.iter().all(|w| *w < 1e-9), "the recomputation does not reproduce the kernel: {worst:?}");
    let [i, j] = *cell;
    // the failing cell's stage: density, angular momentum, kinetic, total and internal energy,
    // each from the stage input to the kernel's output and to the recomputation
    let g = geometry(i, j);
    let state = |c: [f64; 5]| {
        let ke = 0.5 * (c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / c[0];
        [c[0], c[3] * g.lever_c, ke, c[4], c[4] - ke]
    };
    let input = [frame.input.den.at(i, j), frame.input.mom[0].at(i, j), frame.input.mom[1].at(i, j), frame.input.mom[2].at(i, j), frame.input.nrg.at(i, j)];
    let u_n = [frame.u_n.den.at(i, j), frame.u_n.mom[0].at(i, j), frame.u_n.mom[1].at(i, j), frame.u_n.mom[2].at(i, j), frame.u_n.nrg.at(i, j)];
    let output = [frame.output.den.at(i, j), frame.output.mom[0].at(i, j), frame.output.mom[1].at(i, j), frame.output.mom[2].at(i, j), frame.output.nrg.at(i, j)];
    eprintln!("cell [{i}, {j}] [rho, R m_phi, KE, E, E - KE]:");
    eprintln!("  step entry   {}", sci(&state(u_n)));
    eprintln!("  stage input  {}", sci(&state(input)));
    eprintln!("  kernel       {}", sci(&state(output)));
    eprintln!("  recomputed   {}", sci(&state(recomputed_face_work(frame, i, j))));
    eprintln!(
        "  internal-energy rate from the stored fluxes, torque form with face work {:.6e}; times ac dt {:.6e}",
        internal_rate(frame, i, j, true, true),
        frame.ac * frame.dt * internal_rate(frame, i, j, true, true)
    );
    print_cell("the inadmissible cell", frame, i, j);
    print_cell("its inner radial neighbor", frame, i - 1, j);
    print_cell("its outer radial neighbor", frame, i + 1, j);
    print_cell("its polar neighbor toward the pole", frame, i, j - 1);
    print_cell("its polar neighbor toward the equator", frame, i, j + 1);
    print_inflow_budget(frame, i, j);
    print_grid_budget(frame);
    print_reconstructed_states(frame);
}

/// the failing step from the captured state at first order and at smaller timesteps, and the
/// whole captured window at smaller timesteps and first order: whether the failure depends on the
/// timestep a retry would take
#[test]
fn the_outward_wall_surface_failure_at_first_order_and_smaller_timesteps() {
    let production = replay(OUTWARD_RUN);
    let (frame, ..) = production.first.as_ref().expect("the captured state no longer fails; the sweep is vacuous");
    report("production", &production);
    let window = Options { from_step: frame.step, t_end: frame.time + frame.dt, ..OUTWARD_RUN };
    report("the failing step at first order", &replay(Options { theta: 0.0, ..window }));
    for factor in [0.5, 0.25, 0.125, 0.0625] {
        report(&format!("the failing step at dt x {factor}"), &replay(Options { dt_factor: factor, ..window }));
    }
    let whole = Options { t_end: OUTWARD_FAILED_CLOCK + 1e-3, ..OUTWARD_RUN };
    for factor in [0.5, 0.25] {
        report(&format!("the window at dt x {factor}"), &replay(Options { dt_factor: factor, ..whole }));
    }
    report("the window at first order", &replay(Options { theta: 0.0, ..whole }));
}
