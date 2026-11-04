// =============================================================================
// face_gravity_work.rs
//
// gravity's energy work from the face mass fluxes, on the two axisymmetric planes with three
// velocity components under a static central point mass, plummer or compact softening:
// - the kernel's update holds d(E + rho Phi)/dt = -(1/V) \sum_f s_f A_f (F_E + Phi_f F_rho) cell by
//   cell and over the grid with its wall fluxes, for Phi and for Phi plus a constant
// - density and momenta equal the cell-momentum arm's bits; energy differs by the two work rates,
//   the cell momentum's written against g = -d Phi / dr in closed form
// - the two work rates agree at second order on a smooth flow, inside, outside and across the
//   softening length of either kind
// - the kernel's momentum source is rho g with g = -grad Phi of the production force and potential
//   pair, which stay continuous through the softening length and reduce to the bare point mass
//   at zero softening
// - the admission holds only on these planes under one bare point mass at rest at the origin, and
//   every refused body runs the cell momentum's work bit for bit
//
// run: cargo test -p symbi --test face_gravity_work -- --nocapture
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newton::{AdiabaticSubstrateKernelSet, admits_face_gravity_work};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Geometry, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_sim::driver::{downstream_injection_weight, needs_step_snapshot, stage_exit_fraction, stage_schedule};
use symbi_sim::stage::{HookPoint, StageArgs, StageOutcome, fold_stage};
use symbi_discretize::ibm::{body_gravity, body_potential};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.2;
const MASS: f64 = 1.0;
const SOFTENING: f64 = 0.5;

/// a conserved field set by interior or face index, axis 0 fastest
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

#[derive(Clone)]
struct Cons {
    den: Store,
    mom: [Store; 3],
    nrg: Store,
}

fn store(f: &symbi_grid::Field<f64, 2, HostMemory>, ni: usize, nj: usize) -> Store {
    let view = f.view();
    Store { ni, v: (0..nj as isize).flat_map(|j| (0..ni as isize).map(move |i| [i, j])).map(|c| *view.at(c)).collect() }
}

fn cons_of(c: &ConsFieldsGeneric<2, 3, HostMemory, f64>, ni: usize, nj: usize) -> Cons {
    Cons {
        den: store(&c.den, ni, nj),
        mom: std::array::from_fn(|k| store(&c.mom[k], ni, nj)),
        nrg: store(c.nrg_field().expect("nrg"), ni, nj),
    }
}

/// one stage's operands and output
struct Frame {
    dt: f64,
    a0: f64,
    ac: f64,
    input: Cons,
    u_n: Cons,
    flux: [Cons; 2],
    output: Cons,
}

/// runs `steps` steps of the production stage and keeps every stage's frame
macro_rules! frames {
    ($metric:expr, $metric_ty:ty, $n:expr, $lo:expr, $hi:expr, $kind:expr, $cell_work:expr, $steps:expr, $state:expr) => {
        frames!(@bodies $metric, $metric_ty, $n, $lo, $hi, $cell_work, $steps, $state, BodyCollection::new().add(
            Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING).with_softening_kind($kind),
        ))
    };
    (@bodies $metric:expr, $metric_ty:ty, $n:expr, $lo:expr, $hi:expr, $cell_work:expr, $steps:expr, $state:expr, $bodies:expr) => {{
        type Sim = SimStateGeneric<Newtonian, 2, 3, $metric_ty, IdealGas<f64>, CpuSpace, HostMemory>;
        let (n, lo, hi): ([usize; 2], [f64; 2], [f64; 2]) = ($n, $lo, $hi);
        let state = $state;
        let mut sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $metric)
            .cells(n)
            .origin(lo)
            .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
            .boundaries(Boundaries::per_axis([[BoundaryType::Outflow; 2]; 2]))
            .cfl(CFL)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the plane with three velocity components")
            .set_initial(|x: [f64; 2]| {
                let (rho, v, p): (f64, [f64; 3], f64) = state(x);
                Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p))
            })
            .build()
            .with_bodies($bodies);
        let mut kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated);
        kernels.cell_momentum_gravity_work = $cell_work;
        KernelSet::c2p(&kernels, &sim);
        KernelSet::ghost_fill(&kernels, &sim);
        let mut frames = Vec::new();
        for _ in 0..$steps {
            let dt = KernelSet::cfl(&kernels, &sim);
            assert!(dt > 0.0, "the timestep is {dt}");
            sim.dt = dt;
            let stages = sim.timestepping.stages();
            if needs_step_snapshot(stages) {
                KernelSet::snapshot(&kernels, &sim);
            }
            for spec in stage_schedule(stages) {
                let input = cons_of(&sim.fields.cons, n[0], n[1]);
                let u_n = cons_of(&sim.workspace.u_n, n[0], n[1]);
                let mut flux = None;
                let store_ref: &FieldStore<2, 3, HostMemory, f64> = &sim;
                let outcome = fold_stage(
                    store_ref,
                    &kernels,
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
                    &mut |point| {
                        if point == HookPoint::AfterFlux {
                            flux = Some([cons_of(&store_ref.fields.flux[0], n[0] + 1, n[1]), cons_of(&store_ref.fields.flux[1], n[0], n[1] + 1)]);
                        }
                    },
                );
                assert_eq!(outcome, StageOutcome::Accepted, "this fixture has no retry path");
                frames.push(Frame { dt, a0: spec.a0, ac: spec.ac, input, u_n, flux: flux.expect("the stage's fluxes"), output: cons_of(&sim.fields.cons, n[0], n[1]) });
            }
            sim.time += dt;
            sim.iteration += 1;
        }
        frames
    }};
}

/// the cell's closed-form geometry, with each face's center as (distance from the point mass)
struct Cell {
    volume: f64,
    area: [[f64; 2]; 2],
    /// the distance from the origin at the centroid and at each face center
    r_c: f64,
    r_f: [[f64; 2]; 2],
    /// the unit radial direction's components along the two gridded momenta
    radial: [f64; 2],
    /// the momentum slot of each gridded axis
    slots: [usize; 2],
}

fn sphere_cell(n: [usize; 2], lo: [f64; 2], hi: [f64; 2], i: usize, j: usize) -> Cell {
    let (dr, dt) = ((hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64);
    let r = [lo[0] + i as f64 * dr, lo[0] + (i + 1) as f64 * dr];
    let t = [lo[1] + j as f64 * dt, lo[1] + (j + 1) as f64 * dt];
    let i_r2 = (r[1].powi(3) - r[0].powi(3)) / 3.0;
    let i_r = (r[1] * r[1] - r[0] * r[0]) / 2.0;
    let i_sin = t[0].cos() - t[1].cos();
    let r_c = (r[1].powi(4) - r[0].powi(4)) / 4.0 / i_r2;
    Cell {
        volume: 2.0 * PI * i_r2 * i_sin,
        area: [[2.0 * PI * r[0] * r[0] * i_sin, 2.0 * PI * r[1] * r[1] * i_sin], [2.0 * PI * i_r * t[0].sin(), 2.0 * PI * i_r * t[1].sin()]],
        r_c,
        r_f: [[r[0], r[1]], [r_c, r_c]],
        radial: [1.0, 0.0],
        slots: [0, 1],
    }
}

fn cylinder_cell(n: [usize; 2], lo: [f64; 2], hi: [f64; 2], i: usize, j: usize) -> Cell {
    let (dr, dz) = ((hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64);
    let r = [lo[0] + i as f64 * dr, lo[0] + (i + 1) as f64 * dr];
    let z = [lo[1] + j as f64 * dz, lo[1] + (j + 1) as f64 * dz];
    let i_r = (r[1] * r[1] - r[0] * r[0]) / 2.0;
    let r_centroid = (r[1].powi(3) - r[0].powi(3)) / 3.0 / i_r;
    let z_c = 0.5 * (z[0] + z[1]);
    let distance = |rr: f64, zz: f64| (rr * rr + zz * zz).sqrt();
    let r_c = distance(r_centroid, z_c);
    Cell {
        volume: 2.0 * PI * i_r * dz,
        area: [[2.0 * PI * r[0] * dz, 2.0 * PI * r[1] * dz], [2.0 * PI * i_r, 2.0 * PI * i_r]],
        r_c,
        r_f: [[distance(r[0], z_c), distance(r[1], z_c)], [distance(r_centroid, z[0]), distance(r_centroid, z[1])]],
        radial: [r_centroid / r_c, z_c / r_c],
        // the (R, z) plane holds (M_R, M_phi, M_z) in its three momentum slots
        slots: [0, 2],
    }
}

/// the softened potential in closed form
fn potential(kind: SofteningKind, r: f64) -> f64 {
    match kind {
        SofteningKind::Plummer => -MASS / (r * r + SOFTENING * SOFTENING).sqrt(),
        SofteningKind::Compact if r < SOFTENING => {
            let u2 = (r / SOFTENING).powi(2);
            MASS / SOFTENING * (1.25 * u2 - 0.375 * u2 * u2 - 1.875)
        }
        SofteningKind::Compact => -MASS / r,
    }
}

/// the radial acceleration -d Phi / dr in closed form
fn acceleration(kind: SofteningKind, r: f64) -> f64 {
    match kind {
        SofteningKind::Plummer => -MASS * r / (r * r + SOFTENING * SOFTENING).powf(1.5),
        SofteningKind::Compact if r < SOFTENING => {
            let u = r / SOFTENING;
            -MASS / (SOFTENING * SOFTENING) * (2.5 * u - 1.5 * u * u * u)
        }
        SofteningKind::Compact => -MASS / (r * r),
    }
}

fn face(frame: &Frame, comp_energy: bool, d: usize, side: usize, i: usize, j: usize) -> f64 {
    let f = &frame.flux[d];
    let (fi, fj) = if d == 0 { (i + side, j) } else { (i, j + side) };
    if comp_energy { f.nrg.at(fi, fj) } else { f.den.at(fi, fj) }
}

fn ring(lo: [f64; 2], hi: [f64; 2]) -> impl Fn([f64; 2]) -> (f64, [f64; 3], f64) {
    let center = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let width = [0.2 * (hi[0] - lo[0]), 0.2 * (hi[1] - lo[1])];
    move |x| {
        let bump = (-((x[0] - center[0]) / width[0]).powi(2) - ((x[1] - center[1]) / width[1]).powi(2)).exp();
        (0.01 + bump, [-0.2 * bump, 0.1 * bump, 0.6 * bump], 0.01 + 0.1 * bump)
    }
}

/// the flux form of E + rho Phi and the grid closure on every stage, for each constant
fn assert_flux_form(label: &str, frames: &[Frame], n: [usize; 2], kind: SofteningKind, cell: impl Fn(usize, usize) -> Cell) {
    assert!(!frames.is_empty(), "{label}: no stages ran");
    let mut moved = 0.0f64;
    for frame in frames {
        for constant in [0.0, 7.0, -1e3] {
            let (mut grid, mut scale, mut walls, mut worst) = (0.0, 0.0, 0.0, 0.0f64);
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let g = cell(i, j);
                    let phi_c = potential(kind, g.r_c) + constant;
                    let tot = |c: &Cons| c.nrg.at(i, j) + c.den.at(i, j) * phi_c;
                    let mut divergence = 0.0;
                    // the scale sums each term's magnitude: E and rho Phi cancel where a bound gas sits near
                    // zero total energy, 4.4e-5 against terms of 0.10 at one measured cell, whose residual
                    // of 6.6e-18 is 6.4e-17 of its terms
                    let mut magnitude: f64 = [&frame.input, &frame.u_n, &frame.output]
                        .iter()
                        .map(|c| c.nrg.at(i, j).abs() + c.den.at(i, j).abs() * phi_c.abs())
                        .sum();
                    for d in 0..2 {
                        for side in 0..2 {
                            let sign = if side == 1 { 1.0 } else { -1.0 };
                            let phi_f = potential(kind, g.r_f[d][side]) + constant;
                            let (f_e, f_rho) = (face(frame, true, d, side, i, j), face(frame, false, d, side, i, j));
                            let flux = g.area[d][side] * (f_e + phi_f * f_rho);
                            divergence += sign * flux / g.volume;
                            magnitude += frame.dt * g.area[d][side] * (f_e.abs() + (phi_f * f_rho).abs()) / g.volume;
                            let on_wall = (d == 0 && ((side == 0 && i == 0) || (side == 1 && i + 1 == n[0])))
                                || (d == 1 && ((side == 0 && j == 0) || (side == 1 && j + 1 == n[1])));
                            if on_wall {
                                walls += sign * flux;
                            }
                        }
                    }
                    let swept = tot(&frame.output) - frame.a0 * tot(&frame.u_n) - frame.ac * tot(&frame.input);
                    let residual = (swept + frame.ac * frame.dt * divergence).abs();
                    worst = worst.max(residual / magnitude);
                    grid += swept * g.volume;
                    scale += magnitude * g.volume;
                    moved = moved.max(frame.dt * divergence.abs());
                }
            }
            let closure = (grid + frame.ac * frame.dt * walls).abs() / scale;
            assert!(worst < 1e-13, "{label} constant {constant}: cell residual {worst:e}");
            assert!(closure < 1e-13, "{label} constant {constant}: grid closure {closure:e}");
        }
    }
    assert!(moved > 1e-6, "{label}: the state barely moved ({moved:e}); the identity holds vacuously");
}

/// a softened interior is present, so both branches of the compact potential are read
fn assert_covers_softened_interior(label: &str, n: [usize; 2], cell: impl Fn(usize, usize) -> Cell) {
    let radii: Vec<f64> = (0..n[1]).flat_map(|j| (0..n[0]).map(move |i| (i, j))).map(|(i, j)| cell(i, j).r_c).collect();
    assert!(radii.iter().any(|r| *r < SOFTENING), "{label}: no cell lies inside the softening");
    assert!(radii.iter().any(|r| *r > SOFTENING), "{label}: no cell lies outside the softening");
}

/// the face-work arm against the cell-momentum arm on the first stage's identical input
fn assert_arms_differ_by_the_work(label: &str, face_arm: &[Frame], cell_arm: &[Frame], n: [usize; 2], kind: SofteningKind, cell: impl Fn(usize, usize) -> Cell) {
    let (a, b) = (&face_arm[0], &cell_arm[0]);
    assert_eq!(a.input.nrg.v, b.input.nrg.v, "{label}: the arms read different inputs");
    assert_eq!(a.flux[0].den.v, b.flux[0].den.v, "{label}: the arms' mass fluxes differ");
    assert_eq!(a.output.den.v, b.output.den.v, "{label}: the density left the cell arm's bits");
    for k in 0..3 {
        assert_eq!(a.output.mom[k].v, b.output.mom[k].v, "{label}: momentum {k} left the cell arm's bits");
    }
    let mut largest = 0.0f64;
    for j in 0..n[1] {
        for i in 0..n[0] {
            let g = cell(i, j);
            let phi_c = potential(kind, g.r_c);
            let mut face_rate = 0.0;
            for d in 0..2 {
                for side in 0..2 {
                    let sign = if side == 1 { -1.0 } else { 1.0 };
                    face_rate += sign * g.area[d][side] / g.volume * face(a, false, d, side, i, j) * (potential(kind, g.r_f[d][side]) - phi_c);
                }
            }
            let along = g.radial[0] * a.input.mom[g.slots[0]].at(i, j) + g.radial[1] * a.input.mom[g.slots[1]].at(i, j);
            let cell_rate = along * acceleration(kind, g.r_c);
            let expected = a.ac * a.dt * (face_rate - cell_rate);
            let measured = a.output.nrg.at(i, j) - b.output.nrg.at(i, j);
            let tolerance = 1e-13 * a.output.nrg.at(i, j).abs() + 1e-12 * expected.abs();
            assert!((measured - expected).abs() <= tolerance, "{label} [{i}, {j}]: energy difference {measured:e} against {expected:e}");
            largest = largest.max(expected.abs());
        }
    }
    assert!(largest > 0.0, "{label}: the two work rates agree everywhere; the comparison is vacuous");
}

const STEPS: usize = 6;
const N: [usize; 2] = [48, 32];
const SPHERE: ([f64; 2], [f64; 2]) = ([0.1, PI / 4.0], [2.0, 3.0 * PI / 4.0]);
const PLANE: ([f64; 2], [f64; 2]) = ([0.1, -1.0], [2.0, 1.0]);

#[test]
fn the_spherical_plane_holds_the_flux_form_under_either_softening() {
    let (lo, hi) = SPHERE;
    let cell = |i, j| sphere_cell(N, lo, hi, i, j);
    assert_covers_softened_interior("sphere", N, cell);
    for kind in [SofteningKind::Compact, SofteningKind::Plummer] {
        let face_arm = frames!(Spherical, Spherical, N, lo, hi, kind, false, STEPS, ring(lo, hi));
        assert_flux_form("sphere", &face_arm, N, kind, cell);
        let cell_arm = frames!(Spherical, Spherical, N, lo, hi, kind, true, 1, ring(lo, hi));
        assert_arms_differ_by_the_work("sphere", &face_arm, &cell_arm, N, kind, cell);
    }
}

#[test]
fn the_cylindrical_plane_holds_the_flux_form_under_either_softening() {
    let (lo, hi) = PLANE;
    let cell = |i, j| cylinder_cell(N, lo, hi, i, j);
    assert_covers_softened_interior("cylinder", N, cell);
    for kind in [SofteningKind::Compact, SofteningKind::Plummer] {
        let face_arm = frames!(Cylindrical, Cylindrical, N, lo, hi, kind, false, STEPS, ring(lo, hi));
        assert_flux_form("cylinder", &face_arm, N, kind, cell);
        let cell_arm = frames!(Cylindrical, Cylindrical, N, lo, hi, kind, true, 1, ring(lo, hi));
        assert_arms_differ_by_the_work("cylinder", &face_arm, &cell_arm, N, kind, cell);
    }
}

/// the mean difference of the two work rates on a smooth radial flow, away from the walls
fn work_difference(kind: SofteningKind, n: [usize; 2], lo: [f64; 2], hi: [f64; 2]) -> f64 {
    let smooth = |x: [f64; 2]| (1.0 + 0.2 * (3.0 * x[0]).sin(), [0.1, 0.0, 0.3], 1.0);
    let frames = frames!(Spherical, Spherical, n, lo, hi, kind, false, 1, smooth);
    let frame = &frames[0];
    let (mut sum, mut volume) = (0.0, 0.0);
    for j in n[1] / 4..3 * n[1] / 4 {
        for i in n[0] / 4..3 * n[0] / 4 {
            let g = sphere_cell(n, lo, hi, i, j);
            let phi_c = potential(kind, g.r_c);
            let mut face_rate = 0.0;
            for d in 0..2 {
                for side in 0..2 {
                    let sign = if side == 1 { -1.0 } else { 1.0 };
                    face_rate += sign * g.area[d][side] / g.volume * face(frame, false, d, side, i, j) * (potential(kind, g.r_f[d][side]) - phi_c);
                }
            }
            let cell_rate = frame.input.mom[0].at(i, j) * acceleration(kind, g.r_c);
            sum += (face_rate - cell_rate).abs() * g.volume;
            volume += g.volume;
        }
    }
    sum / volume
}

/// the compact force has a curvature jump at the softening length, which a cell straddling it
/// sees at first order on a vanishing fraction of the cells: the L1 mean stays near second order,
/// 1.8 being the bound the interior and exterior measurements clear
#[test]
fn the_two_work_rates_agree_at_second_order_through_the_softening() {
    let angles = (PI / 4.0, 3.0 * PI / 4.0);
    // the measured band runs over the middle half of the radial range
    let bands = [("outside", 1.0, 2.0), ("inside", 0.1, 0.45), ("across", 0.2, 0.8)];
    for kind in [SofteningKind::Compact, SofteningKind::Plummer] {
        for (label, r_lo, r_hi) in bands {
            let (lo, hi) = ([r_lo, angles.0], [r_hi, angles.1]);
            let errors: Vec<f64> = [32, 64, 128].iter().map(|m| work_difference(kind, [*m, *m], lo, hi)).collect();
            let orders = [(errors[0] / errors[1]).log2(), (errors[1] / errors[2]).log2()];
            eprintln!("{kind:?} {label}: the work difference at 32, 64, 128: {errors:?}, orders {orders:?}");
            assert!(errors[2] > 0.0, "{kind:?} {label}: the difference vanished; the order is undefined");
            assert!(orders[1] > 1.8, "{kind:?} {label}: the order between 64 and 128 cells is {}", orders[1]);
        }
    }
}

/// the production force is minus the gradient of the production potential, by centered difference
/// at radii inside, at and outside the softening, in each kind
#[test]
fn the_production_force_is_the_gradient_of_the_production_potential() {
    let kinds = [SofteningKind::Compact, SofteningKind::Plummer];
    for kind in kinds {
        let k = kind.as_scalar();
        let phi = |x: [f64; 3]| body_potential(x, MASS, SOFTENING, k);
        for r in [0.01, 0.1, 0.3, 0.49, 0.499, 0.501, 0.51, 0.7, 1.0, 3.0] {
            let direction = [0.6, 0.0, 0.8];
            let at = |rr: f64| std::array::from_fn(|ii| direction[ii] * rr);
            let force = body_gravity(at(r), MASS, SOFTENING, k);
            let h = 1e-5 * r;
            let along = direction[0] * force[0] + direction[1] * force[1] + direction[2] * force[2];
            let gradient = (phi(at(r + h)) - phi(at(r - h))) / (2.0 * h);
            // a centered difference at h = 1e-5 r holds about 1e-9 of the force in roundoff and curvature
            assert!((along + gradient).abs() < 1e-8 * along.abs().max(1.0), "{kind:?} r = {r}: force {along:e}, -grad Phi {:e}", -gradient);
            assert!((along - acceleration(kind, r)).abs() < 1e-13 * along.abs().max(1.0), "{kind:?} r = {r}: force {along:e} against the closed form {:e}", acceleration(kind, r));
            assert!((phi(at(r)) - potential(kind, r)).abs() < 1e-14 * phi(at(r)).abs(), "{kind:?} r = {r}: the potential left its closed form");
        }
    }
    // the compact pair is continuous through the softening length
    let k = SofteningKind::Compact.as_scalar();
    let side = |r: f64| (body_potential([r, 0.0, 0.0], MASS, SOFTENING, k), body_gravity([r, 0.0, 0.0], MASS, SOFTENING, k)[0]);
    let (inner, outer) = (side(SOFTENING * (1.0 - 1e-12)), side(SOFTENING * (1.0 + 1e-12)));
    assert!((inner.0 - outer.0).abs() < 1e-11, "the potential jumps by {:e}", inner.0 - outer.0);
    assert!((inner.1 - outer.1).abs() < 1e-10, "the force jumps by {:e}", inner.1 - outer.1);
    // at zero softening both kinds are the bare point mass
    for kind in kinds {
        let k = kind.as_scalar();
        for r in [0.05, 1.0, 3.0] {
            let bare = -MASS / r;
            assert!((body_potential([r, 0.0, 0.0], MASS, 0.0, k) - bare).abs() < 1e-14 * bare.abs(), "{kind:?} r = {r}: zero softening potential");
            let force = body_gravity([r, 0.0, 0.0], MASS, 0.0, k)[0];
            assert!((force + MASS / (r * r)).abs() < 1e-13 * force.abs(), "{kind:?} r = {r}: zero softening force {force:e}");
        }
    }
}

/// the body pass alone, on a uniform gas carrying momentum: the momenta gain dt rho g in the two
/// meridional slots and none azimuthally, through the softened interior. the face work reads
/// the stage's mass flux, here zero, so the same pass adds no energy under the admission, and the
/// cell momentum's arm adds dt m . g
macro_rules! assert_body_pass {
    ($label:expr, $metric:expr, $metric_ty:ty, $lo:expr, $hi:expr, $kind:expr, $cell:expr) => {{
        type Sim = SimStateGeneric<Newtonian, 2, 3, $metric_ty, IdealGas<f64>, CpuSpace, HostMemory>;
        let (lo, hi): ([f64; 2], [f64; 2]) = ($lo, $hi);
        for cell_work in [false, true] {
            let mut sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $metric)
                .cells(N)
                .origin(lo)
                .spacing([(hi[0] - lo[0]) / N[0] as f64, (hi[1] - lo[1]) / N[1] as f64])
                .boundaries(Boundaries::per_axis([[BoundaryType::Outflow; 2]; 2]))
                .cfl(CFL)
                .timestepping(Timestepping::Rk2)
                .allocate()
                .expect("the plane with three velocity components")
                .set_initial(|_| Prim::adiabatic(Density(2.0), Tensor::new([0.3, -0.2, 0.5]), Pressure(1.0)))
                .build()
                .with_bodies(BodyCollection::new().add(
                    Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING).with_softening_kind($kind),
                ));
            let mut kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated);
            kernels.cell_momentum_gravity_work = cell_work;
            KernelSet::c2p(&kernels, &sim);
            sim.mark_stage_input_captured();
            KernelSet::snapshot_stage(&kernels, &sim);
            sim.dt = 0.0;
            let store_ref: &FieldStore<2, 3, HostMemory, f64> = &sim;
            let before = cons_of(&store_ref.fields.cons, N[0], N[1]);
            let dt = 1e-3;
            KernelSet::body_source(&kernels, store_ref, dt);
            let after = cons_of(&store_ref.fields.cons, N[0], N[1]);
            let mut inside = 0usize;
            for j in 0..N[1] {
                for i in 0..N[0] {
                    let g: Cell = $cell(N, lo, hi, i, j);
                    inside += usize::from(g.r_c < SOFTENING);
                    let accel = acceleration($kind, g.r_c);
                    for (slot, along) in g.slots.iter().zip(g.radial) {
                        let rate = (after.mom[*slot].at(i, j) - before.mom[*slot].at(i, j)) / dt;
                        let expected = before.den.at(i, j) * accel * along;
                        assert!((rate - expected).abs() < 1e-11 * accel.abs().max(1.0), "{} {:?} [{i}, {j}] slot {slot}: {rate:e} against {expected:e}", $label, $kind);
                    }
                    let azimuthal = 3 - g.slots[0] - g.slots[1];
                    assert_eq!(after.mom[azimuthal].at(i, j), before.mom[azimuthal].at(i, j), "{}: the azimuthal momentum moved", $label);
                    assert_eq!(after.den.at(i, j), before.den.at(i, j), "{}: the density moved", $label);
                    let gained = after.nrg.at(i, j) - before.nrg.at(i, j);
                    let m_along = g.radial[0] * before.mom[g.slots[0]].at(i, j) + g.radial[1] * before.mom[g.slots[1]].at(i, j);
                    let expected = if cell_work { dt * m_along * accel } else { 0.0 };
                    assert!((gained - expected).abs() < 1e-12 * before.nrg.at(i, j).abs(), "{} {:?} [{i}, {j}] cell work {cell_work}: energy gain {gained:e} against {expected:e}", $label, $kind);
                }
            }
            assert!(inside > 0, "{}: no cell lies inside the softening", $label);
        }
    }};
}

#[test]
fn the_body_pass_is_rho_g_through_the_softened_interior() {
    for kind in [SofteningKind::Compact, SofteningKind::Plummer] {
        assert_body_pass!("sphere", Spherical, Spherical, SPHERE.0, SPHERE.1, kind, sphere_cell);
        assert_body_pass!("cylinder", Cylindrical, Cylindrical, PLANE.0, PLANE.1, kind, cylinder_cell);
    }
}

/// every body the admission refuses runs the cell momentum's work bit for bit, and the admitted body
/// differs from it, so a refused body's run is the one it was before the face work existed
macro_rules! assert_refused_bodies {
    ($label:expr, $metric:expr, $metric_ty:ty, $lo:expr, $hi:expr) => {{
        let (lo, hi): ([f64; 2], [f64; 2]) = ($lo, $hi);
        let run = |work: bool, bodies: BodyCollection<f64, 2>| frames!(@bodies $metric, $metric_ty, N, lo, hi, work, 2, ring(lo, hi), bodies);
        let bare = Body::<f64, 2>::gravitational(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING);
        let (admitted, cell) = (run(false, BodyCollection::new().add(bare)), run(true, BodyCollection::new().add(bare)));
        assert_ne!(admitted[0].output.nrg.v, cell[0].output.nrg.v, "{}: the admitted body runs the cell work; the comparison is vacuous", $label);
        let mut moved = bare;
        moved.position = Tensor::new([0.0, 0.05]);
        let mut moving = bare;
        moving.velocity = Tensor::new([0.0, 1e-9]);
        let second = Body::<f64, 2>::gravitational(1, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING);
        let cases: [(&str, BodyCollection<f64, 2>); 4] = [
            ("off the origin", BodyCollection::new().add(moved)),
            ("in motion", BodyCollection::new().add(moving)),
            ("with a reaction", BodyCollection::new().add(bare.with_two_way_coupling(true))),
            ("beside a second body", BodyCollection::new().add(bare).add(second)),
        ];
        for (case, bodies) in cases {
            let (default, control) = (run(false, bodies.clone()), run(true, bodies));
            for (a, b) in default.iter().zip(&control) {
                assert_eq!(a.output.nrg.v, b.output.nrg.v, "{} {case}: the refused body left the cell work", $label);
            }
        }
    }};
}

#[test]
fn every_refused_body_runs_the_cell_work_bit_for_bit() {
    assert_refused_bodies!("sphere", Spherical, Spherical, SPHERE.0, SPHERE.1);
    assert_refused_bodies!("cylinder", Cylindrical, Cylindrical, PLANE.0, PLANE.1);
}

#[test]
fn the_admission_holds_on_the_planes_under_one_bare_point_mass_at_the_origin() {
    let bare = Body::<f64, 2>::gravitational(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING);
    let admits = |g: Geometry, dof: usize, bodies: &[Body<f64, 2>]| admits_face_gravity_work(g, dof, bodies);
    assert!(admits(Geometry::Spherical, 3, &[bare]));
    assert!(admits(Geometry::Cylindrical, 3, &[bare]));
    assert!(admits(Geometry::Spherical, 3, &[bare.with_softening_kind(SofteningKind::Compact)]));
    assert!(!admits(Geometry::Cartesian, 3, &[bare]), "a cartesian plane has no torque form");
    assert!(!admits(Geometry::Spherical, 2, &[bare]), "a plane with two velocity components keeps the cell work");
    assert!(!admits(Geometry::Spherical, 3, &[]), "no body");
    assert!(!admits(Geometry::Spherical, 3, &[bare, bare]), "two bodies");
    let mut moved = bare;
    moved.position = Tensor::new([0.0, 0.5]);
    assert!(!admits(Geometry::Spherical, 3, &[moved]), "a body away from the origin");
    let mut moving = bare;
    moving.velocity = Tensor::new([0.0, 0.5]);
    assert!(!admits(Geometry::Spherical, 3, &[moving]), "a moving body's potential is unsteady");
    assert!(!admits(Geometry::Spherical, 3, &[bare.with_two_way_coupling(true)]), "a body with a reaction");
    let sink = Body::<f64, 2>::black_hole(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING, 1.0, 0.0, 0.2);
    assert!(!admits(Geometry::Spherical, 3, &[sink]), "a sink drains the gas");
    let three_d = Body::<f64, 3>::gravitational(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING);
    assert!(!admits_face_gravity_work(Geometry::Spherical, 3, &[three_d]), "a three-dimensional grid");
}
