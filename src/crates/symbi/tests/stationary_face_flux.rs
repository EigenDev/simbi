// =============================================================================
// stationary_face_flux.rs
//
// the stationary twin of the adiabatic face flux takes the grid velocity as zero and reads
// neither the axis map nor the mesh rates; the general kernel evaluates
// `vface = mesh_adot * x_face + mesh_vtrans` at every face. on faces that stand still the two
// must agree bit for bit:
// - on a uniform and a graded (geometric, log) stationary mesh, 1d and 2d, every sweep
// - for HLLE and HLLC, under plm, the first-order (theta = 0) arm, and ppm on the uniform mesh
// - on a state holding shocks, smooth flow and a region at rest, where fluxes are exactly zero
// the selection applies the twin to a baked scheme on stationary faces alone: a homologous or a
// translating mesh, a well-balanced or HLLC+ scheme, and the isothermal family take the general
// kernel, and a stationary dispatch forced onto moving faces is refused.
//
// usage:
//   cargo test -p symbi --test stationary_face_flux
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_kernels::{FluxSpec, dispatch_flux_with_motion, face_flux_kernel_name, stationary_flux_applies};
use symbi::kernels::support::FaceDomain;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::coords::{Balance, FaceMotion};
use symbi_geometry::{AxisMap, Cartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::HostMemory;

const GAMMA: f64 = 1.4;
const N: usize = 24;

fn spec(solver: Solver, recon: symbi_discretize::Recon, theta: f64) -> FluxSpec {
    FluxSpec {
        theta,
        solver,
        recon,
        eos: symbi_discretize::EosArm::IdealGamma,
        flatten: (0.0, 0.0),
        balance: Balance::Plain,
        rusanov: false,
    }
}

/// a state with a shock across the middle of each axis, smooth flow on one side, and a region at
/// rest (uniform density and pressure, zero velocity) where every mass flux is exactly zero
fn prim_at<const D: usize>(x: [f64; D]) -> ([f64; D], f64, f64) {
    let at_rest = x.iter().all(|&xx| xx < 0.3);
    if at_rest {
        return ([0.0; D], 1.0, 1.0);
    }
    let left = x[0] < 0.6;
    let rho = if left { 1.0 + 0.3 * (7.0 * x[D - 1]).sin() } else { 0.25 };
    let pre = if left { 1.0 } else { 0.1 };
    let vel: [f64; D] = std::array::from_fn(|kk| 0.8 * ((3.0 + kk as f64) * x[kk] + 1.0).cos());
    (vel, rho, pre)
}

fn sim_1d(maps: Option<[AxisMap; 1]>, ghosts: usize) -> SimState<Newtonian, 1, Cartesian, IdealGas<f64>, symbi_xpu::CpuSpace, HostMemory> {
    SimState::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([0.1])
        .spacing([1.0 / N as f64])
        .coord_maps(maps)
        .ghosts(ghosts)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            let (vel, rho, pre) = prim_at(x);
            Prim::adiabatic(Density(rho), Tensor::new(vel), Pressure(pre))
        })
        .build()
}

fn sim_2d(maps: Option<[AxisMap; 2]>, ghosts: usize) -> SimState<Newtonian, 2, Cartesian, IdealGas<f64>, symbi_xpu::CpuSpace, HostMemory> {
    SimState::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([0.1, 0.1])
        .spacing([1.0 / N as f64, 1.0 / N as f64])
        .coord_maps(maps)
        .ghosts(ghosts)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            let (vel, rho, pre) = prim_at(x);
            Prim::adiabatic(Density(rho), Tensor::new(vel), Pressure(pre))
        })
        .build()
}

/// every flux field along `dir`, downloaded whole
fn fluxes<const D: usize>(store: &FieldStore<D, D, HostMemory, f64>, dir: usize) -> Vec<Vec<f64>> {
    let f = &store.fields.flux[dir];
    let mut fields = vec![&f.den];
    fields.extend(f.mom.iter());
    fields.extend(f.nrg.iter());
    fields.iter().map(|field| field.download().expect("download")).collect()
}

/// the general and the stationary flux along every sweep of `store` agree bit for bit; returns
/// the count of exactly-zero mass fluxes the comparison covered
fn assert_twin_matches<const D: usize>(store: &FieldStore<D, D, HostMemory, f64>, spec: FluxSpec, label: &str) -> usize {
    let kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, D>::new(GAMMA, 0.4, &store.geom.allocated);
    symbi::sim::evolve::KernelSet::c2p(&kernels, store);
    symbi::sim::evolve::KernelSet::ghost_fill(&kernels, store);
    let pre = store.fields.prim.pre_field().expect("pressure");
    let mut zeros = 0;
    for dir in 0..D {
        assert!(stationary_flux_applies(store, "adiabatic", dir, spec), "{label}: the stationary twin does not apply; the comparison is vacuous");
        // the two dispatches run distinct kernels: the general one declares the moving-face
        // scalars, the twin none of them
        let general_name = face_flux_kernel_name(store, "adiabatic", dir, spec, FaceMotion::Moving);
        let twin_name = face_flux_kernel_name(store, "adiabatic", dir, spec, FaceMotion::Stationary);
        let ir = |name: &str| symbi_aot::kernel_by_name::<f64>(name).unwrap_or_else(|| panic!("{label}: kernel '{name}' is not baked")).1;
        assert!(ir(&general_name).contains("mesh_adot"), "{label}: '{general_name}' evaluates no face velocity; the comparison is vacuous");
        assert!(
            !ir(&twin_name).contains("mesh_adot") && !ir(&twin_name).contains("map_kind"),
            "{label}: the twin '{twin_name}' declares a moving-face scalar"
        );
        dispatch_flux_with_motion(store, pre, "adiabatic", dir, GAMMA, spec, FaceMotion::Moving);
        let general = fluxes(store, dir);
        let f = &store.fields.flux[dir];
        let mut written = vec![&f.den];
        written.extend(f.mom.iter());
        written.extend(f.nrg.iter());
        for field in written {
            field.upload(&vec![f64::NAN; field.domain().volume()]).expect("upload");
        }
        dispatch_flux_with_motion(store, pre, "adiabatic", dir, GAMMA, spec, FaceMotion::Stationary);
        let stationary = fluxes(store, dir);
        let face = store.geom.interior.face_domain(dir);
        let allocated = store.fields.flux[dir].den.domain().clone();
        let cells: Vec<usize> = face.iter().map(|c| allocated.flat_index(c)).collect();
        for (kk, (g, s)) in general.iter().zip(&stationary).enumerate() {
            let differing: Vec<usize> = cells.iter().copied().filter(|&ii| g[ii].to_bits() != s[ii].to_bits()).collect();
            assert!(
                differing.is_empty(),
                "{label}: flux component {kk} along axis {dir} differs at {} of {} faces, first {:?}: general {:e} stationary {:e}",
                differing.len(),
                cells.len(),
                differing.first(),
                g[differing[0]],
                s[differing[0]],
            );
            assert!(cells.iter().all(|&ii| s[ii].is_finite()), "{label}: the stationary flux left a face unwritten");
        }
        zeros += cells.iter().filter(|&&ii| general[0][ii] == 0.0).count();
    }
    zeros
}

fn schemes() -> Vec<(&'static str, FluxSpec)> {
    use symbi_discretize::Recon::Plm;
    vec![
        ("hlle plm", spec(Solver::Hlle, Plm, 1.5)),
        ("hllc plm", spec(Solver::Hllc, Plm, 1.5)),
        ("hlle first order", spec(Solver::Hlle, Plm, 0.0)),
        ("hllc first order", spec(Solver::Hllc, Plm, 0.0)),
    ]
}

#[test]
fn the_stationary_twin_matches_the_general_flux_on_a_uniform_mesh() {
    let mut zeros = 0;
    for (label, spec) in schemes() {
        zeros += assert_twin_matches(&sim_1d(None, 2), spec, &format!("1d uniform {label}"));
        zeros += assert_twin_matches(&sim_2d(None, 2), spec, &format!("2d uniform {label}"));
    }
    for solver in [Solver::Hlle, Solver::Hllc] {
        let ppm = spec(solver, symbi_discretize::Recon::Ppm, 1.0);
        zeros += assert_twin_matches(&sim_1d(None, 3), ppm, &format!("1d uniform ppm {solver:?}"));
        zeros += assert_twin_matches(&sim_2d(None, 3), ppm, &format!("2d uniform ppm {solver:?}"));
    }
    assert!(zeros > 0, "no face carried an exactly-zero mass flux; the signed-zero cases went untested");
}

#[test]
fn the_stationary_twin_matches_the_general_flux_on_a_graded_mesh() {
    let geometric = AxisMap::Geometric { start: 0.1, width: 0.02, ratio: 1.08 };
    let log = AxisMap::Log { start: 0.1, log_slope: 1.0 / N as f64 };
    // the premise: both maps grade their widths, so the general kernel evaluates nonuniform faces
    for map in [geometric, log] {
        let (w0, w1) = (map.face(1) - map.face(0), map.face(N as isize) - map.face(N as isize - 1));
        assert!((w1 / w0 - 1.0).abs() > 0.5, "{map:?} is barely graded; the graded comparison is vacuous");
    }
    for (label, spec) in schemes() {
        assert_twin_matches(&sim_1d(Some([geometric]), 2), spec, &format!("1d geometric {label}"));
        assert_twin_matches(&sim_1d(Some([log]), 2), spec, &format!("1d log {label}"));
        assert_twin_matches(&sim_2d(Some([geometric, log]), 2), spec, &format!("2d geometric x log {label}"));
    }
}

#[test]
fn moving_faces_and_other_schemes_take_the_general_flux() {
    let hlle = spec(Solver::Hlle, symbi_discretize::Recon::Plm, 1.5);
    let mut sim = sim_2d(None, 2);
    assert!(stationary_flux_applies(&sim.store, "adiabatic", 0, hlle));
    sim.store.motion.homologous = true;
    sim.store.motion.a_dot = 0.25;
    assert!(!stationary_flux_applies(&sim.store, "adiabatic", 0, hlle), "a homologous mesh took the stationary flux");
    sim.store.motion.homologous = false;
    sim.store.motion.a_dot = 0.3;
    assert!(!stationary_flux_applies(&sim.store, "adiabatic", 0, hlle), "a translating mesh took the stationary flux along x");
    // translation moves axis 0 alone; the y faces stand still
    assert!(stationary_flux_applies(&sim.store, "adiabatic", 1, hlle), "the stationary y faces of a translating mesh took the general flux");
    sim.store.motion.a_dot = 0.0;
    let balanced = FluxSpec { balance: Balance::Hydrostatic, ..hlle };
    let plus = FluxSpec { solver: Solver::HllcPlus, ..hlle };
    assert!(!stationary_flux_applies(&sim.store, "adiabatic", 0, balanced), "a well-balanced flux has no stationary twin");
    assert!(!stationary_flux_applies(&sim.store, "adiabatic", 0, plus), "an HLLC+ flux has no stationary twin");
    assert!(!stationary_flux_applies(&sim.store, "iso", 0, hlle), "the isothermal flux has no stationary twin");
}

#[test]
#[should_panic(expected = "applies to a baked scheme on faces that stand still")]
fn a_stationary_flux_forced_onto_moving_faces_is_refused() {
    let mut sim = sim_2d(None, 2);
    sim.store.motion.homologous = true;
    sim.store.motion.a_dot = 0.25;
    let pre = sim.fields.prim.pre_field().expect("pressure");
    let hlle = spec(Solver::Hlle, symbi_discretize::Recon::Plm, 1.5);
    dispatch_flux_with_motion(&sim.store, pre, "adiabatic", 0, GAMMA, hlle, FaceMotion::Stationary);
}
