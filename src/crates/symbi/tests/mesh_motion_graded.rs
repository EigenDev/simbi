// =============================================================================
// mesh_motion_graded.rs
//
// homologous expansion on a graded (non-uniform) mesh, on both a curvilinear and a cartesian
// chart, for each axis map that grades: logarithmic and geometric.
//
// homologous motion multiplies every coordinate by one a(t). a graded axis therefore stays graded
// with its ratios untouched, because scaling the axis start and the axis length parameter while
// leaving the dimensionless shape parameter alone reproduces the whole face list scaled:
//   log        face(i) = start 10^(i s)               -> a start 10^(i s)            = a face(i)
//   geometric  face(i) = start + w (q^i - 1)/(q - 1)  -> a start + a w (q^i-1)/(q-1) = a face(i)
// the split between "length" (start, width, dx) and "shape" (s, q) is what makes this exact for
// every map rather than for one of them.
//
// the probe is a state at rest in comoving coordinates with a uniform comoving density. that is
// the exact solution of the expanding equations -- physically free expansion, v = H r, the
// physical density falling as a^-p for a cell volume going as r^p -- so the comoving state must
// not move at all, on any mesh. the exactness is what makes this a gate rather than a convergence
// study.
//
// the cancellation it rests on is mesh-independent. in spherical geometry the grid-velocity mass
// flux through a face is rho H r * 4 pi r^2, so its divergence over a cell is
//   [4 pi H rho r_hi^3 - 4 pi H rho r_lo^3] / [(4 pi/3)(r_hi^3 - r_lo^3)] = 3 H rho,
// which is exactly the dilution `mesh_hdil = 3 H` for any face positions; the cartesian case is
// the same telescope with area 1 and volume (x_hi - x_lo), giving 1 H rho. it therefore holds only
// while the grid velocity and the cell geometry agree on where each face is -- and the bug this
// gate was written for was precisely that `vface` reconstructed the face as `x_lo + i*dx` while
// the volumes and areas used the axis map. on a log mesh spanning a decade that mismatch corrupted
// a state that cannot move by 65 percent, growing outward with the cell width.
//
// the cartesian charts carry a second, independent instance of the same class of fault: the CFL
// length. a graded cartesian axis has no single width, so pricing the timestep off one is a
// stability violation wherever cells are narrower than that width.
//
// run: cargo test -p symbi --test mesh_motion_graded -- --nocapture
// =============================================================================

use symbi::prelude::KernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_geometry::{AxisMap, Cartesian, Metric, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 64;
const CFL: f64 = 0.4;
const RHO0: f64 = 1.0;
const PRE0: f64 = 1.0;
/// a decade in position, so the widest cell is many times the narrowest and a per-cell scaling
/// error cannot hide as a uniform offset.
const R_LO: f64 = 1.0;
const R_HI: f64 = 10.0;
const A_DOT: f64 = 1.0;
const T_END: f64 = 0.5;
/// cell-to-cell width ratio of the geometric mesh. over N cells it grades by `q^(N-1)`, which at
/// this value is comparable to the decade the log mesh spans.
const RATIO: f64 = 1.03;

type Sim<M> = SimState<Newtonian, 1, M, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

/// which axis map lays out the faces. `Uniform` is the ungraded control.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Grading {
    Uniform,
    Log,
    Geometric,
}

/// the maps that grade, for tests that assert a property of grading itself.
const GRADED: [Grading; 2] = [Grading::Log, Grading::Geometric];

impl Grading {
    /// the axis map spanning [R_LO, R_HI] over N cells, or `None` for the uniform control (which
    /// takes the plain `spacing` path and carries no map at all).
    fn maps(self) -> Option<[AxisMap; 1]> {
        match self {
            Grading::Uniform => None,
            Grading::Log => Some([AxisMap::Log {
                start: R_LO,
                log_slope: (R_HI / R_LO).log10() / N as f64,
            }]),
            // the first width is fixed by requiring the last face to land on R_HI:
            // sum_{i<N} w q^i = w (q^N - 1)/(q - 1) = R_HI - R_LO.
            Grading::Geometric => Some([AxisMap::Geometric {
                start: R_LO,
                width: (R_HI - R_LO) * (RATIO - 1.0) / (RATIO.powi(N as i32) - 1.0),
                ratio: RATIO,
            }]),
        }
    }
}

fn uniform_state(_x: [f64; 1]) -> Prim<f64, 1> {
    Prim {
        rho: RHO0,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: PRE0,
    }
}

/// the same 1d probe on whichever chart is passed; `coords` is the only thing that differs
/// between the curvilinear and cartesian instances of the test.
fn build<M>(coords: M, grading: Grading, expanding: bool) -> Sim<M>
where
    M: Metric<f64, 1> + Copy,
    Sim<M>: Sized,
{
    let dr = (R_HI - R_LO) / N as f64;
    let mut sim = Sim::<M>::build(Newtonian, IdealGas { gamma: GAMMA }, coords)
        .cells([N])
        .origin([R_LO])
        .spacing([dr])
        .coord_maps(grading.maps())
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(uniform_state)
        .build();
    if expanding {
        sim.motion = symbi_geometry::MotionState::homologous(1.0, A_DOT);
    }
    sim
}

/// the largest departure from a flat comoving profile, and the largest comoving speed. the
/// interior is trimmed by the halo: an outflow boundary copies its ghost, which is not the
/// expanding state and drives a real (physical) edge transient.
fn departures<M: Metric<f64, 1>>(sim: &Sim<M>) -> (f64, f64) {
    let rho = sim.fields.prim.rho.view();
    let vel = sim.fields.prim.vel[0].view();
    let all: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let trim = all.len() / 8;
    let cells = &all[trim..all.len() - trim];
    let mean: f64 = cells.iter().map(|c| *rho.at(*c)).sum::<f64>() / cells.len() as f64;
    (
        cells
            .iter()
            .map(|c| (rho.at(*c) - mean).abs() / mean)
            .fold(0.0_f64, f64::max),
        cells
            .iter()
            .map(|c| vel.at(*c).abs())
            .fold(0.0_f64, f64::max),
    )
}

/// the widest-to-narrowest cell width ratio of the mesh as built.
fn width_span<M: Metric<f64, 1>>(sim: &Sim<M>) -> f64 {
    let widths: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| sim.geom.cell_width(c, 0))
        .collect();
    widths.iter().cloned().fold(0.0_f64, f64::max)
        / widths.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// non-vacuity: a near-uniform mesh would make a per-cell scaling error look like a uniform one,
/// and every claim resting on it would hold for the wrong reason.
fn assert_actually_graded(span: f64, grading: Grading) {
    assert!(
        span > 5.0,
        "the {grading:?} mesh spans only {span:.2}x in cell width; it is not graded enough to \
         distinguish a per-cell scaling error from a uniform offset"
    );
}

fn run_spherical(sim: &mut Sim<Spherical>) {
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
    symbi::sim::evolve::evolve(sim, &kernels, T_END).expect("evolve");
}

fn run_cartesian(sim: &mut Sim<Cartesian>) {
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
    symbi::sim::evolve::evolve(sim, &kernels, T_END).expect("evolve");
}

#[test]
fn a_graded_mesh_expands_homologously() {
    for grading in GRADED {
        let span = width_span(&build(Spherical, grading, true));
        println!("\n{grading:?} mesh: cell widths span {span:.2}x");
        assert_actually_graded(span, grading);

        let mut sim = build(Spherical, grading, true);
        run_spherical(&mut sim);
        let a = sim.motion.a;
        let (flatness, speed) = departures(&sim);
        println!("  a = {a:.4}: comoving flatness {flatness:.3e}, comoving |v| {speed:.3e}");
        assert!(
            a > 1.4,
            "the scale factor only reached {a:.4}; the mesh barely expanded"
        );
        // a comoving-static uniform state is the exact solution and may not develop structure.
        assert!(
            flatness < 1.0e-11,
            "the comoving density developed {flatness:.3e} of structure on an expanding \
             {grading:?} mesh; a uniform state carries no gradient to drive it, so an axis width \
             is not scaling with the rest of the mesh"
        );
        assert!(
            speed < 1.0e-11,
            "the comoving state reached |v| = {speed:.3e} on a {grading:?} mesh; it must stay \
             exactly at rest"
        );
    }
}

#[test]
fn the_expansion_is_carried_by_the_scale_factor() {
    // spherical volumes go as r^3, so homologous expansion dilutes the physical density as
    // a^-3 however the radial axis is graded. what the state stores is the comoving density,
    // which must therefore not move at all.
    for grading in GRADED {
        let mut sim = build(Spherical, grading, true);
        run_spherical(&mut sim);
        let a = sim.motion.a;
        let rho = sim.fields.prim.rho.view();
        let mid = sim.geom.interior.iter().nth(N / 2).unwrap();
        let comoving = *rho.at(mid);
        println!(
            "  {grading:?}: a = {a:.4}, comoving rho = {comoving:.12e}, physical = {:.6e}",
            comoving / a.powi(3)
        );
        assert!(
            (comoving - RHO0).abs() / RHO0 < 1.0e-11,
            "the comoving density moved from {RHO0} to {comoving:.12e} on a {grading:?} mesh; \
             the expansion is being applied to the stored state instead of carried by the \
             scale factor"
        );
    }
}

#[test]
fn grading_costs_no_accuracy_under_expansion() {
    // separates "the graded path is self-consistent" from "it agrees with the uniform path
    // that was already trusted".
    let mut uniform = build(Spherical, Grading::Uniform, true);
    run_spherical(&mut uniform);
    let (uf, us) = departures(&uniform);
    for grading in GRADED {
        let mut graded = build(Spherical, grading, true);
        run_spherical(&mut graded);
        let (gf, gs) = departures(&graded);
        println!(
            "  {grading:?}: flatness {gf:.3e} vs uniform {uf:.3e};  |v| {gs:.3e} vs uniform {us:.3e}"
        );
        assert!(
            (graded.motion.a - uniform.motion.a).abs() < 1.0e-12,
            "the {grading:?} and uniform runs reached different scale factors ({} vs {})",
            graded.motion.a,
            uniform.motion.a
        );
        assert!(
            gf <= uf.max(1.0e-11) && gs <= us.max(1.0e-11),
            "the {grading:?} mesh held the state less exactly (flatness {gf:.3e}, |v| {gs:.3e}) \
             than the uniform one ({uf:.3e}, {us:.3e}); grading is costing accuracy it should not"
        );
    }
}

#[test]
fn a_graded_cartesian_mesh_expands_homologously() {
    // the cartesian charts reach the mesh through a different width path than the curvilinear
    // ones: their lame factors are all 1, so a single precomputed inverse width is enough for a
    // uniform axis and is wrong for a graded one -- both for the divergence and, separately, for
    // the CFL length, where using one width for a mesh that has many prices the step off cells
    // that are not the narrowest.
    for grading in GRADED {
        let span = width_span(&build(Cartesian, grading, true));
        println!("\ncartesian {grading:?} mesh: cell widths span {span:.2}x");
        assert_actually_graded(span, grading);

        let mut sim = build(Cartesian, grading, true);
        run_cartesian(&mut sim);
        let a = sim.motion.a;
        let (flatness, speed) = departures(&sim);
        let mid = sim.geom.interior.iter().nth(N / 2).unwrap();
        let comoving = *sim.fields.prim.rho.view().at(mid);
        println!(
            "  a = {a:.4}: flatness {flatness:.3e}, |v| {speed:.3e}, comoving rho \
             {comoving:.12e}, physical {:.6e}",
            comoving / a
        );
        assert!(
            a > 1.4,
            "the scale factor only reached {a:.4}; the mesh barely expanded"
        );
        assert!(
            flatness < 1.0e-11,
            "the comoving density developed {flatness:.3e} of structure on an expanding graded \
             cartesian {grading:?} mesh; a uniform state carries no gradient to drive it"
        );
        assert!(
            speed < 1.0e-11,
            "the comoving state reached |v| = {speed:.3e} on a graded cartesian {grading:?} \
             mesh; it must stay exactly at rest"
        );
        // a cartesian cell volume goes as x, so the physical density dilutes as a^-1 while the
        // stored comoving density is unmoved.
        assert!(
            (comoving - RHO0).abs() / RHO0 < 1.0e-11,
            "the comoving density moved from {RHO0} to {comoving:.12e} on a graded cartesian \
             {grading:?} mesh"
        );
    }
}

#[test]
fn a_graded_cartesian_mesh_is_stepped_on_its_own_narrowest_cell() {
    // the CFL length is the fault the state-at-rest probe above cannot see: a state that does
    // not move is stable at any timestep, so it reports nothing about whether the step was
    // priced correctly. a graded mesh whose step came from one uniform width would take the
    // same step as the uniform mesh; the correct step is set by the narrowest cell, which is
    // narrower than uniform by construction, so the two must differ by that ratio.
    for grading in GRADED {
        let graded = build(Cartesian, grading, false);
        let uniform = build(Cartesian, Grading::Uniform, false);
        let widths: Vec<f64> = graded
            .geom
            .interior
            .iter()
            .map(|c| graded.geom.cell_width(c, 0))
            .collect();
        let narrowest = widths.iter().cloned().fold(f64::INFINITY, f64::min);
        let uniform_width = (R_HI - R_LO) / N as f64;

        // the CFL reduces over the allocated domain, so the ghost bands have to hold real gas
        // before it is meaningful -- an unfilled ghost carries rho = 0 and its sound speed is
        // 0/0. this is the same prologue `evolve` runs before its first timestep.
        let dt_of = |sim: &mut Sim<Cartesian>| {
            let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
            kernels.c2p(sim);
            kernels.ghost_fill(sim);
            kernels.cfl(&sim.store)
        };
        let (mut graded, mut uniform) = (graded, uniform);
        let dt_graded = dt_of(&mut graded);
        let dt_uniform = dt_of(&mut uniform);
        assert!(
            dt_graded.is_finite() && dt_uniform.is_finite(),
            "the probe produced a non-finite timestep (graded {dt_graded}, uniform              {dt_uniform}); the comparison below would be vacuous"
        );
        // a uniform sound-speed state has one wave speed everywhere, so the step ratio is
        // exactly the width ratio.
        let expected = narrowest / uniform_width;
        let got = dt_graded / dt_uniform;
        println!(
            "  cartesian {grading:?}: dt {dt_graded:.6e} vs uniform {dt_uniform:.6e}; \
             ratio {got:.6} expected {expected:.6}"
        );
        assert!(
            (got - expected).abs() / expected < 1.0e-10,
            "the graded cartesian {grading:?} mesh stepped at {got:.6} of the uniform step but \
             its narrowest cell is {expected:.6} of the uniform width; the CFL is being priced \
             off a width that mesh does not have"
        );
    }
}

#[test]
#[ignore = "diagnostic: what the stored state does under expansion, uniform vs graded"]
fn diagnose_stored_state_convention() {
    for grading in [Grading::Uniform, Grading::Log, Grading::Geometric] {
        let mut sim = build(Spherical, grading, true);
        run_spherical(&mut sim);
        let a = sim.motion.a;
        let rho = sim.fields.prim.rho.view();
        let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
        let sample: Vec<String> = [4usize, N / 4, N / 2, 3 * N / 4, N - 5]
            .iter()
            .map(|&i| format!("{:.6}", rho.at(cells[i])))
            .collect();
        println!("{grading:>10?}  a = {a:.4}  rho at i=[4,N/4,N/2,3N/4,N-5]: {sample:?}");
        println!(
            "{:>10}  a^-3 = {:.6}, a^-2 = {:.6}",
            "",
            a.powi(-3),
            a.powi(-2)
        );
    }
}
