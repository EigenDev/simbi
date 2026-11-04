// =============================================================================
// azimuthal_viscous_transport.rs
//
// the azimuthal shear operator alone, stepped through the production kernel sets
// without hydrodynamics, on uniform and log-graded radial axes, against closed forms:
// - rigid rotation carries no stress: the operator leaves it unchanged to rounding
// - shear diffusion: the lamb-oseen vortex on the (R, z) plane and the spherical
//   bessel mode v_phi = j_1(k r) sin(theta) on the (r, theta) plane, at constant nu,
//   converge to their exact decay, adiabatic and isothermal
// - a domain sealed with zero-torque faces holds its angular momentum and total
//   energy to rounding while its kinetic energy falls, and every cell's internal
//   energy rises; the isothermal closure has no energy and the same torques
// - the locally isothermal closure, cs^2 varying per cell: the stress reads no sound speed,
//   so the same closed forms and budgets hold with the closure field set
// - continued domain faces: the change of angular momentum and total energy equals the
//   ledger of the torque and work the faces passed
// - the coefficient law: the same momentum law set kinematic or dynamic, scaling with density
//   as nu and as mu = rho nu
// - the timestep bound: stable at the maximum-principle bound, unstable past its 2x
//   stability limit
//
// run: cargo test -p symbi --test azimuthal_viscous_transport
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_kernels::azimuthal_viscous_rate;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cylindrical, Spherical};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_hydro::viscous::{AzimuthalShear, DynamicViscosityLaw, ViscousFace};
use symbi_sim::substrate_seam::{KernelSet, WithViscosity};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const OPEN: [BoundaryType; 2] = [BoundaryType::Outflow, BoundaryType::Outflow];
const WALL: [BoundaryType; 2] = [BoundaryType::Reflect, BoundaryType::Reflect];
const AXIS: [BoundaryType; 2] = [BoundaryType::Axis, BoundaryType::Axis];
const ZERO: [ViscousFace; 2] = [ViscousFace::ZeroTorque, ViscousFace::ZeroTorque];
const CONTINUED: [ViscousFace; 2] = [ViscousFace::ContinuedShear, ViscousFace::ContinuedShear];

fn law(nu: f64) -> DynamicViscosityLaw {
    DynamicViscosityLaw::kinematic(nu, 1.0, 0.0, 1.0, 0.0)
}

/// how far a run steps: a count of steps, or to a time at the fraction of the bound
struct Budget {
    steps: usize,
    until: f64,
}

impl From<usize> for Budget {
    fn from(steps: usize) -> Self {
        Self { steps, until: f64::INFINITY }
    }
}

fn until(time: f64) -> Budget {
    Budget { steps: usize::MAX, until: time }
}

/// one cell of a plane: bounding faces on each axis, conserved and primitive state
struct Cell {
    faces: [[f64; 2]; 2],
    den: f64,
    mom: f64,
    vphi: f64,
    nrg: Option<f64>,
}

impl Cell {
    fn center(&self) -> [f64; 2] {
        [0.5 * (self.faces[0][0] + self.faces[0][1]), 0.5 * (self.faces[1][0] + self.faces[1][1])]
    }
}

/// builds a plane in the closure the arm names, then runs the viscous operator alone `steps`
/// times at a fraction of the bound, recovering the primitives and filling the ghosts as the
/// post-step recipe does, and returns the opening and final cells, the time reached and the ledger
macro_rules! shear_run {
    (adiabatic, $metric:expr, $metric_ty:ty, $slot:expr, $n:expr, $lo:expr, $hi:expr, $maps:expr, $walls:expr, $model:expr, $steps:expr, $fraction:expr, $state:expr) => {{
        type Sim = SimStateGeneric<Newtonian, 2, 3, $metric_ty, IdealGas<f64>, CpuSpace, HostMemory>;
        let (n, lo, hi): ([usize; 2], [f64; 2], [f64; 2]) = ($n, $lo, $hi);
        let state = $state;
        let sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $metric)
            .cells(n)
            .origin(lo)
            .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
            .coord_maps($maps)
            .boundaries(Boundaries::per_axis($walls))
            .allocate()
            .expect("the plane")
            .set_initial(|x: [f64; 2]| {
                let (rho, vphi, p): (f64, f64, f64) = state(x);
                let mut v = [0.0; 3];
                v[$slot] = vphi;
                Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p))
            })
            .build();
        let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated).with_viscosity(1.0).with_azimuthal_shear(Some($model));
        step_shear!(sim, kernels, $slot, $steps, $fraction, $model)
    }};
    (isothermal, $metric:expr, $metric_ty:ty, $slot:expr, $n:expr, $lo:expr, $hi:expr, $maps:expr, $walls:expr, $model:expr, $steps:expr, $fraction:expr, $state:expr) => {{
        type Sim = SimStateGeneric<IsoNewtonian, 2, 3, $metric_ty, Isothermal<f64>, CpuSpace, HostMemory>;
        let (n, lo, hi): ([usize; 2], [f64; 2], [f64; 2]) = ($n, $lo, $hi);
        let state = $state;
        let sim: Sim = Sim::build(IsoNewtonian, Isothermal { cs: 1.0 }, $metric)
            .cells(n)
            .origin(lo)
            .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
            .coord_maps($maps)
            .boundaries(Boundaries::per_axis($walls))
            .allocate()
            .expect("the plane")
            .set_initial(|x: [f64; 2]| {
                let (rho, vphi, _p): (f64, f64, f64) = state(x);
                let mut v = [0.0; 3];
                v[$slot] = vphi;
                PrimG::<f64, 3, IsoModel>::isothermal(Density(rho), Tensor::new(v))
            })
            .build();
        let kernels = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(1.0, 0.4, &sim.geom.allocated).with_viscosity(1.0).with_azimuthal_shear(Some($model));
        step_shear!(sim, kernels, $slot, $steps, $fraction, $model)
    }};
    (locally_isothermal, $metric:expr, $metric_ty:ty, $slot:expr, $n:expr, $lo:expr, $hi:expr, $maps:expr, $walls:expr, $model:expr, $steps:expr, $fraction:expr, $state:expr) => {{
        type Sim = SimStateGeneric<IsoNewtonian, 2, 3, $metric_ty, Isothermal<f64>, CpuSpace, HostMemory>;
        let (n, lo, hi): ([usize; 2], [f64; 2], [f64; 2]) = ($n, $lo, $hi);
        let state = $state;
        let sim: Sim = Sim::build(IsoNewtonian, Isothermal { cs: 1.0 }, $metric)
            .cells(n)
            .origin(lo)
            .spacing([(hi[0] - lo[0]) / n[0] as f64, (hi[1] - lo[1]) / n[1] as f64])
            .coord_maps($maps)
            .boundaries(Boundaries::per_axis($walls))
            .allocate()
            .expect("the plane")
            .set_initial(|x: [f64; 2]| {
                let (rho, vphi, _p): (f64, f64, f64) = state(x);
                let mut v = [0.0; 3];
                v[$slot] = vphi;
                PrimG::<f64, 3, IsoModel>::isothermal(Density(rho), Tensor::new(v))
            })
            .build();
        let kernels = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(1.0, 0.4, &sim.geom.allocated).with_viscosity(1.0).with_azimuthal_shear(Some($model));
        for c in sim.geom.allocated.iter() {
            kernels.cs2.set(c, closure_cs2(lo[0] + (c[0] as f64 + 0.5) * (hi[0] - lo[0]) / n[0] as f64, lo[1] + (c[1] as f64 + 0.5) * (hi[1] - lo[1]) / n[1] as f64));
        }
        step_shear!(sim, kernels, $slot, $steps, $fraction, $model)
    }};
}

/// a prescribed closure varying across the plane, ghosts included
fn closure_cs2(x0: f64, x1: f64) -> f64 {
    1.0 + 0.3 * (3.0 * x0).sin() * (2.0 * x1).cos()
}

macro_rules! step_shear {
    ($sim:ident, $kernels:ident, $slot:expr, $steps:expr, $fraction:expr, $model:expr) => {{
        let slot: usize = $slot;
        let cells = |sim: &FieldStore<2, 3, HostMemory, f64>| -> Vec<Cell> {
            sim.geom
                .interior
                .iter()
                .map(|c| {
                    let bounds = |d: usize| -> [f64; 2] {
                        match &sim.geom.maps {
                            Some(m) => [m[d].face(c[d]), m[d].face(c[d] + 1)],
                            None => [sim.geom.x_lo[d] + c[d] as f64 * sim.geom.dx[d], sim.geom.x_lo[d] + (c[d] + 1) as f64 * sim.geom.dx[d]],
                        }
                    };
                    Cell {
                        faces: [bounds(0), bounds(1)],
                        den: *sim.fields.cons.den.view().at(c),
                        mom: *sim.fields.cons.mom[slot].view().at(c),
                        vphi: *sim.fields.prim.vel[slot].view().at(c),
                        nrg: sim.fields.cons.nrg_field().map(|e| *e.view().at(c)),
                    }
                })
                .collect()
        };
        KernelSet::c2p(&$kernels, &$sim);
        KernelSet::ghost_fill(&$kernels, &$sim);
        let opening = cells(&$sim);
        let mut time = 0.0;
        let scratch = &$kernels.cfl_scratch;
        let budget: Budget = $steps.into();
        for _ in 0..budget.steps {
            if time >= budget.until {
                break;
            }
            let rate = azimuthal_viscous_rate(&*$sim, &$model, scratch);
            let dt = ($fraction / rate).min(budget.until - time);
            KernelSet::viscous(&$kernels, &$sim, dt);
            KernelSet::c2p(&$kernels, &$sim);
            KernelSet::ghost_fill(&$kernels, &$sim);
            time += dt;
        }
        let ledger = *$kernels.viscous_boundary_ledger.lock().unwrap();
        (opening, cells(&$sim), time, ledger)
    }};
}

fn log_radius(n: usize, lo: f64, hi: f64, n1: usize, lo1: f64, hi1: f64) -> Option<[AxisMap; 2]> {
    Some([
        AxisMap::Log { start: lo, log_slope: (hi / lo).log10() / n as f64 },
        AxisMap::Uniform { start: lo1, dx: (hi1 - lo1) / n1 as f64 },
    ])
}

fn sphere_lever_volume(c: &Cell) -> (f64, f64) {
    let ([r0, r1], [t0, t1]) = (c.faces[0], c.faces[1]);
    let i_r2 = (r1.powi(3) - r0.powi(3)) / 3.0;
    let i_sin = t0.cos() - t1.cos();
    let r_c = (r1.powi(4) - r0.powi(4)) / 4.0 / i_r2;
    let t_c = ((t1.sin() - t1 * t1.cos()) - (t0.sin() - t0 * t0.cos())) / i_sin;
    (r_c * t_c.sin(), i_r2 * i_sin * 2.0 * PI)
}

fn rz_lever_volume(c: &Cell) -> (f64, f64) {
    let ([r0, r1], [z0, z1]) = (c.faces[0], c.faces[1]);
    let i_r = (r1 * r1 - r0 * r0) / 2.0;
    ((r1.powi(3) - r0.powi(3)) / 3.0 / i_r, i_r * (z1 - z0) * 2.0 * PI)
}

fn angular_momentum(cells: &[Cell], lever_volume: fn(&Cell) -> (f64, f64)) -> f64 {
    cells.iter().map(|c| { let (lever, volume) = lever_volume(c); c.mom * lever * volume }).sum()
}

fn total_energy(cells: &[Cell], lever_volume: fn(&Cell) -> (f64, f64)) -> f64 {
    cells.iter().map(|c| c.nrg.expect("an energy closure") * lever_volume(c).1).sum()
}

fn kinetic(cells: &[Cell], lever_volume: fn(&Cell) -> (f64, f64)) -> f64 {
    cells.iter().map(|c| 0.5 * c.mom * c.mom / c.den * lever_volume(c).1).sum()
}

// ------------------------------- rigid rotation -------------------------------

const ROUNDING: f64 = 1e-13;

macro_rules! rigid_case {
    ($test:ident, $closure:tt, $metric:expr, $metric_ty:ty, $slot:expr, $lo:expr, $hi:expr, $maps:expr, $walls:expr, $radius:expr) => {
        #[test]
        fn $test() {
            let model = AzimuthalShear { law: law(0.01), faces: [ZERO, ZERO] };
            let (lo, hi): ([f64; 2], [f64; 2]) = ($lo, $hi);
            let width = [(hi[0] - lo[0]) / 32.0, (hi[1] - lo[1]) / 16.0];
            let (a, b, t, _) = shear_run!($closure, $metric, $metric_ty, $slot, [32, 16], lo, hi, $maps, $walls, model, 20, 0.5, |x: [f64; 2]| {
                let lever: fn(&Cell) -> (f64, f64) = $radius;
                // the uniform angular velocity on the distance the operator divides by, the centroid's
                let cell = Cell { faces: [[x[0] - 0.5 * width[0], x[0] + 0.5 * width[0]], [x[1] - 0.5 * width[1], x[1] + 0.5 * width[1]]], den: 1.0, mom: 0.0, vphi: 0.0, nrg: None };
                (1.0, 0.5 * lever(&cell).0, 1.0)
            });
            assert!(t > 0.0, "no time passed");
            let scale = a.iter().map(|c| c.mom.abs()).fold(0.0, f64::max);
            assert!(scale > 0.0, "the gas does not rotate");
            let moved = a.iter().zip(&b).map(|(x, y)| (x.mom - y.mom).abs()).fold(0.0, f64::max);
            eprintln!("{} rigid rotation moved the momentum by {moved:e}", stringify!($test));
            assert!(moved < ROUNDING * scale, "rigid rotation felt a stress: {moved:e} of {scale:e}");
        }
    };
}

rigid_case!(rigid_rotation_feels_no_stress_spherical_adiabatic, adiabatic, Spherical, Spherical, 2, [1.0, 0.5], [2.0, 2.5], None, [OPEN, OPEN], sphere_lever_volume);
rigid_case!(rigid_rotation_feels_no_stress_spherical_isothermal, isothermal, Spherical, Spherical, 2, [1.0, 0.5], [2.0, 2.5], None, [OPEN, OPEN], sphere_lever_volume);
rigid_case!(rigid_rotation_feels_no_stress_rz_adiabatic, adiabatic, Cylindrical, Cylindrical, 1, [1.0, 0.0], [2.0, 1.0], None, [OPEN, OPEN], rz_lever_volume);
rigid_case!(rigid_rotation_feels_no_stress_rz_isothermal, isothermal, Cylindrical, Cylindrical, 1, [1.0, 0.0], [2.0, 1.0], None, [OPEN, OPEN], rz_lever_volume);
rigid_case!(rigid_rotation_feels_no_stress_spherical_locally_isothermal, locally_isothermal, Spherical, Spherical, 2, [1.0, 0.5], [2.0, 2.5], None, [OPEN, OPEN], sphere_lever_volume);
rigid_case!(rigid_rotation_feels_no_stress_rz_locally_isothermal, locally_isothermal, Cylindrical, Cylindrical, 1, [1.0, 0.0], [2.0, 1.0], None, [OPEN, OPEN], rz_lever_volume);

// ------------------------------- closed-form diffusion -------------------------------

const NU: f64 = 1.0e-2;

/// the lamb-oseen vortex with circulation scale 1 at age `t`
fn lamb_oseen(r: f64, t: f64) -> f64 {
    (1.0 - (-r * r / (4.0 * NU * t)).exp()) / r
}

const T0: f64 = 1.0;
/// the outflow ghost's wrong shear diffuses in over sqrt(nu t) = 0.07; the measured zones stand
/// 0.5 from the walls, where erfc(0.5 / (2 * 0.07)) is below 1e-6 of it
const AGE: f64 = 0.5;

fn rz_diffusion_error(cells: &[Cell], elapsed: f64) -> f64 {
    let inside: Vec<&Cell> = cells.iter().filter(|c| (1.0..2.5).contains(&c.center()[0])).collect();
    assert!(!inside.is_empty());
    inside.iter().map(|c| (c.vphi - lamb_oseen(rz_lever_volume(c).0, T0 + elapsed)).abs()).sum::<f64>() / inside.len() as f64
}

fn converging(label: &str, measured: &[f64], factor: f64) {
    eprintln!("{label} by resolution: {measured:?}");
    assert!(measured[0] > 0.0, "{label}: exactly zero");
    for pair in measured.windows(2) {
        assert!(pair[1] < factor * pair[0], "{label} does not converge by {factor}: {measured:?}");
    }
}

macro_rules! lamb_oseen_case {
    ($test:ident, $closure:tt, $graded:expr) => {
        #[test]
        fn $test() {
            let errors: Vec<f64> = [32usize, 64, 128]
                .into_iter()
                .map(|n| {
                    let (lo, hi) = (0.5, 3.0);
                    let maps = if $graded { log_radius(n, lo, hi, 4, 0.0, 0.5) } else { None };
                    let model = AzimuthalShear { law: law(NU), faces: [CONTINUED, ZERO] };
                    let (_, b, t, _) = shear_run!($closure, Cylindrical, Cylindrical, 1, [n, 4], [lo, 0.0], [hi, 0.5], maps, [OPEN, OPEN], model, until(AGE), 0.25, |[r, _z]: [f64; 2]| (1.0, lamb_oseen(r, T0), 1.0));
                    assert_eq!(t, AGE, "the run stopped short");
                    rz_diffusion_error(&b, t)
                })
                .collect();
            converging(stringify!($test), &errors, 0.45);
        }
    };
}

lamb_oseen_case!(lamb_oseen_decays_on_a_uniform_rz_plane_adiabatic, adiabatic, false);
lamb_oseen_case!(lamb_oseen_decays_on_a_uniform_rz_plane_isothermal, isothermal, false);
lamb_oseen_case!(lamb_oseen_decays_on_a_graded_rz_plane_adiabatic, adiabatic, true);
lamb_oseen_case!(lamb_oseen_decays_on_a_graded_rz_plane_isothermal, isothermal, true);
lamb_oseen_case!(lamb_oseen_decays_on_a_uniform_rz_plane_locally_isothermal, locally_isothermal, false);
lamb_oseen_case!(lamb_oseen_decays_on_a_graded_rz_plane_locally_isothermal, locally_isothermal, true);

/// the spherical bessel mode v_phi = j_1(k r) sin(theta), decaying at exp(-nu k^2 t): at constant
/// density and nu, the azimuthal stress's solenoidal eigenfunction on the sphere
const K_MODE: f64 = PI;

fn j1(x: f64) -> f64 {
    x.sin() / (x * x) - x.cos() / x
}

fn bessel_error(cells: &[Cell], elapsed: f64) -> f64 {
    let decay = (-NU * K_MODE * K_MODE * elapsed).exp();
    let inside: Vec<&Cell> = cells.iter().filter(|c| (1.5..2.5).contains(&c.center()[0])).collect();
    assert!(!inside.is_empty());
    inside
        .iter()
        .map(|c| {
            let ([r0, r1], [t0, t1]) = (c.faces[0], c.faces[1]);
            let r_c = (r1.powi(4) - r0.powi(4)) / 4.0 / ((r1.powi(3) - r0.powi(3)) / 3.0);
            let t_c = ((t1.sin() - t1 * t1.cos()) - (t0.sin() - t0 * t0.cos())) / (t0.cos() - t1.cos());
            (c.vphi - j1(K_MODE * r_c) * t_c.sin() * decay).abs()
        })
        .sum::<f64>()
        / inside.len() as f64
}

macro_rules! bessel_case {
    ($test:ident, $closure:tt, $graded:expr) => {
        #[test]
        fn $test() {
            let errors: Vec<f64> = [32usize, 64, 128]
                .into_iter()
                .map(|n| {
                    let (lo, hi) = (1.0, 3.0);
                    let maps = if $graded { log_radius(n, lo, hi, n, 0.0, PI) } else { None };
                    let model = AzimuthalShear { law: law(NU), faces: [CONTINUED, ZERO] };
                    let (_, b, t, _) = shear_run!($closure, Spherical, Spherical, 2, [n, n], [lo, 0.0], [hi, PI], maps, [OPEN, AXIS], model, until(AGE), 0.25, |[r, theta]: [f64; 2]| (1.0, j1(K_MODE * r) * theta.sin(), 1.0));
                    assert_eq!(t, AGE, "the run stopped short");
                    bessel_error(&b, t)
                })
                .collect();
            converging(stringify!($test), &errors, 0.45);
        }
    };
}

bessel_case!(bessel_mode_decays_on_a_uniform_spherical_plane_adiabatic, adiabatic, false);
bessel_case!(bessel_mode_decays_on_a_uniform_spherical_plane_isothermal, isothermal, false);
bessel_case!(bessel_mode_decays_on_a_graded_spherical_plane_adiabatic, adiabatic, true);
bessel_case!(bessel_mode_decays_on_a_graded_spherical_plane_isothermal, isothermal, true);
bessel_case!(bessel_mode_decays_on_a_uniform_spherical_plane_locally_isothermal, locally_isothermal, false);
bessel_case!(bessel_mode_decays_on_a_graded_spherical_plane_locally_isothermal, locally_isothermal, true);

// ------------------------------- the sealed budget -------------------------------

const SEALED_BOUND: f64 = 1e-12;

fn bump_swirl(r: f64, x: f64, x0: f64) -> f64 {
    0.5 * (-((r - 1.5) / 0.2).powi(2)).exp() * (-((x - x0) / 0.4).powi(2)).exp()
}

macro_rules! sealed_case {
    ($test:ident, $closure:tt, $metric:expr, $metric_ty:ty, $slot:expr, $lo:expr, $hi:expr, $walls:expr, $lv:expr, $x0:expr, $energy:expr) => {
        #[test]
        fn $test() {
            let model = AzimuthalShear { law: law(0.05), faces: [ZERO, ZERO] };
            for graded in [false, true] {
                let (lo, hi): ([f64; 2], [f64; 2]) = ($lo, $hi);
                let maps = if graded { log_radius(48, lo[0], hi[0], 32, lo[1], hi[1]) } else { None };
                let (a, b, _, _) = shear_run!($closure, $metric, $metric_ty, $slot, [48, 32], lo, hi, maps, $walls, model, 50, 0.5, |[r, x]: [f64; 2]| (1.0, bump_swirl(r, x, $x0), 1.0));
                let lv: fn(&Cell) -> (f64, f64) = $lv;
                let (l0, l1) = (angular_momentum(&a, lv), angular_momentum(&b, lv));
                assert!(l0.abs() > 0.0 && (kinetic(&a, lv) - kinetic(&b, lv)) > 1e-3 * kinetic(&a, lv), "the stress never acted");
                let change = (l1 - l0).abs() / l0.abs();
                eprintln!("{} graded {graded}: relative L change {change:e}", stringify!($test));
                assert!(change < SEALED_BOUND, "L moved by {change:e}");
                assert!(kinetic(&b, lv) < kinetic(&a, lv), "kinetic energy rose");
                if $energy {
                    let (e0, e1) = (total_energy(&a, lv), total_energy(&b, lv));
                    let change = (e1 - e0).abs() / e0.abs();
                    eprintln!("{} graded {graded}: relative E change {change:e}", stringify!($test));
                    assert!(change < SEALED_BOUND, "total energy moved by {change:e}");
                    for (x, y) in a.iter().zip(&b) {
                        let (ex, ey) = (x.nrg.unwrap() - 0.5 * x.mom * x.mom / x.den, y.nrg.unwrap() - 0.5 * y.mom * y.mom / y.den);
                        assert!(ey >= ex - SEALED_BOUND, "a cell's internal energy fell: {ex} to {ey}");
                    }
                }
            }
        }
    };
}

sealed_case!(a_sealed_spherical_shell_budgets_the_stress_adiabatic, adiabatic, Spherical, Spherical, 2, [1.0, 0.0], [2.0, PI], [WALL, AXIS], sphere_lever_volume, 1.2, true);
sealed_case!(a_sealed_spherical_shell_budgets_the_stress_isothermal, isothermal, Spherical, Spherical, 2, [1.0, 0.0], [2.0, PI], [WALL, AXIS], sphere_lever_volume, 1.2, false);
sealed_case!(a_sealed_rz_annulus_budgets_the_stress_adiabatic, adiabatic, Cylindrical, Cylindrical, 1, [1.0, 0.0], [2.0, 1.0], [WALL, WALL], rz_lever_volume, 0.5, true);
sealed_case!(a_sealed_rz_annulus_budgets_the_stress_isothermal, isothermal, Cylindrical, Cylindrical, 1, [1.0, 0.0], [2.0, 1.0], [WALL, WALL], rz_lever_volume, 0.5, false);
sealed_case!(a_sealed_spherical_shell_budgets_the_stress_locally_isothermal, locally_isothermal, Spherical, Spherical, 2, [1.0, 0.0], [2.0, PI], [WALL, AXIS], sphere_lever_volume, 1.2, false);
sealed_case!(a_sealed_rz_annulus_budgets_the_stress_locally_isothermal, locally_isothermal, Cylindrical, Cylindrical, 1, [1.0, 0.0], [2.0, 1.0], [WALL, WALL], rz_lever_volume, 0.5, false);

// ------------------------------- the boundary ledger -------------------------------

#[test]
fn a_continued_face_books_the_torque_and_work_it_passes() {
    let model = AzimuthalShear { law: law(0.05), faces: [[ViscousFace::ZeroTorque, ViscousFace::ContinuedShear], ZERO] };
    let (a, b, _, ledger) = shear_run!(adiabatic, Cylindrical, Cylindrical, 1, [48, 8], [1.0, 0.0], [2.0, 0.5], None, [OPEN, OPEN], model, 200, 0.5, |[r, _z]: [f64; 2]| (1.0, 0.3 + 0.2 * r, 1.0));
    let lv: fn(&Cell) -> (f64, f64) = rz_lever_volume;
    let (dl, de) = (angular_momentum(&b, lv) - angular_momentum(&a, lv), total_energy(&b, lv) - total_energy(&a, lv));
    eprintln!("L change {dl:e} against the ledger {:e}; E change {de:e} against {:e}", ledger.0, ledger.1);
    assert!(ledger.0.abs() > 1e-6, "the face passed no torque; the ledger is vacuous");
    assert!((dl - ledger.0).abs() < 1e-11 * ledger.0.abs().max(1.0), "L {dl:e} against {:e}", ledger.0);
    assert!((de - ledger.1).abs() < 1e-11 * ledger.1.abs().max(1.0), "E {de:e} against {:e}", ledger.1);
}

// ------------------------------- the coefficient law -------------------------------

/// the first step's momentum change at density `rho`, at a fixed time step
fn first_step_change(coefficient: DynamicViscosityLaw, rho: f64) -> f64 {
    let model = AzimuthalShear { law: coefficient, faces: [ZERO, ZERO] };
    type Sim = SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([32, 4])
        .origin([1.0, 0.0])
        .spacing([1.0 / 32.0, 0.25 / 4.0])
        .boundaries(Boundaries::per_axis([OPEN, OPEN]))
        .allocate()
        .unwrap()
        .set_initial(|[r, _z]: [f64; 2]| Prim::adiabatic(Density(rho), Tensor::new([0.0, r * r, 0.0]), Pressure(1.0)))
        .build();
    let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated).with_viscosity(1.0).with_azimuthal_shear(Some(model));
    KernelSet::c2p(&kernels, &sim);
    KernelSet::ghost_fill(&kernels, &sim);
    let c = [16isize, 2];
    let before = *sim.fields.cons.mom[1].view().at(c);
    KernelSet::viscous(&kernels, &sim, 1.0e-4);
    *sim.fields.cons.mom[1].view().at(c) - before
}

#[test]
fn a_kinematic_viscosity_and_a_dynamic_viscosity_scale_differently_with_density() {
    let kinematic = DynamicViscosityLaw::kinematic(0.02, 1.0, 0.0, 1.0, 0.0);
    let dynamic = DynamicViscosityLaw::dynamic(0.02, 1.0, 0.0, 1.0, 0.0);
    let (k1, k4) = (first_step_change(kinematic, 1.0), first_step_change(kinematic, 4.0));
    let (d1, d4) = (first_step_change(dynamic, 1.0), first_step_change(dynamic, 4.0));
    assert!(k1.abs() > 0.0, "the shear never acted");
    assert!((k4 / k1 - 4.0).abs() < 1e-10, "the kinematic torque at 4 rho: {} times", k4 / k1);
    assert!((d4 / d1 - 1.0).abs() < 1e-10, "the dynamic torque at 4 rho: {} times", d4 / d1);
    assert!((k1 / d1 - 1.0).abs() < 1e-10, "the two forms differ at the normalizing density");
}

// ------------------------------- the time-step bound -------------------------------

/// the kinetic energy ratio after `steps` at `fraction` of the bound, on a log-graded (r, theta) plane
fn stability(fraction: f64, steps: usize) -> f64 {
    let model = AzimuthalShear { law: law(0.02), faces: [ZERO, ZERO] };
    let maps = log_radius(48, 0.1, 4.0, 32, 0.0, PI);
    let (a, b, _, _) = shear_run!(adiabatic, Spherical, Spherical, 2, [48, 32], [0.1, 0.0], [4.0, PI], maps, [WALL, AXIS], model, steps, fraction, |[r, t]: [f64; 2]| {
        (1.0, 0.5 * (r * 8.0).sin() * t.sin() + 0.1 * ((r * 40.0).sin() * (t * 17.0).sin()), 1.0)
    });
    kinetic(&b, sphere_lever_volume) / kinetic(&a, sphere_lever_volume)
}

#[test]
fn the_shear_is_stable_at_its_maximum_principle_bound_and_not_past_twice_it() {
    let (half, bound, past) = (stability(0.5, 200), stability(1.0, 200), stability(2.2, 200));
    eprintln!("kinetic energy after 200 steps: 0.5 bound {half:e}, 1.0 bound {bound:e}, 2.2 bound {past:e}");
    assert!(half < 1.0 && bound < 1.0, "unstable at or below the bound: {half}, {bound}");
    assert!(!past.is_finite() || past > 1.0, "the bound is not sharp: stable at 2.2 times it ({past})");
}
