// =============================================================================
// mhd_1p5d_curvilinear.rs
//
// end-to-end validation of the 1.5D curvilinear Newtonian-MHD kernels (the spherical
// and cylindrical radial charts — `{n,r,i}mhd_godunov_stage_{sph,cyl}_1d`, the matching wave-speed
// maps, and the `rmhd_bcell_godunov_euler_{sph,cyl}_1d` out-of-plane predictor). the cartesian 1.5D
// MHD kernels do not cover the (1, spherical) / (1, cylindrical) charts. no
// constrained transport at D=1 (C(1,2)=0 edges), so:
//   - the normal radial field B_r is carried on the r-faces and never curled (seeded div-free,
//     B_r = B0/r^2 on spherical / B0/r on cylindrical, so the area-weighted radial divergence is
//     zero), and stays at its IC to machine precision,
//   - the out-of-plane toroidal B_phi rides the induction-flux divergence (the bcell predictor)
//     and must actually evolve as the radial pressure bump drives a radial flow,
//   - the gas (rho, v, p) stays physical (positive, finite) on the curvilinear shell.
// this gates the curvilinear-chart kernels' wiring + stability. it asserts structure and
// physicality, the only checkable properties absent a closed-form curvilinear MHD reference.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
use symbi_hydro::newtonian_mhd::{NewtonianMhd, nmhd_recover};
use symbi_hydro::quantity::{Density, EnergyDensity, Pressure};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 128;
const R_LO: f64 = 2.0; // radial shell away from the origin / axis
const DR: f64 = 0.05;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1; // radial-field amplitude (div-free normalization)
const BPHI0: f64 = 0.2; // out-of-plane toroidal seed
const T_FINAL: f64 = 0.2;

// the div-free normal radial field at radius r: B0/r^2 (spherical) or B0/r (cylindrical), so the
// area-weighted radial divergence (A_r ~ r^2 / r) of B_r telescopes to zero.
fn radial_bfield(spherical: bool, r: f64) -> f64 {
    if spherical { B0 / (r * r) } else { B0 / r }
}

fn recover_1d<M>(
    sim: &SimStateGeneric<NewtonianMhd, 1, 3, M, IdealGas<f64>, CpuSpace, HostMemory>,
    c: [isize; 1],
) -> (f64, f64)
where
    M: symbi_geometry::Metric<f64, 1> + symbi_geometry::Metric<f64, 3> + Copy + Send + Sync,
{
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let cons = MhdCons::<f64, 3>::new(
        Cons::adiabatic(
            Density(*sim.fields.cons.den.view().at(c)),
            Tensor::new([
                *sim.fields.cons.mom[0].view().at(c),
                *sim.fields.cons.mom[1].view().at(c),
                *sim.fields.cons.mom[2].view().at(c),
            ]),
            EnergyDensity(*cnrg.view().at(c)),
        ),
        Tensor::new([
            *mhd.bcell[0].view().at(c),
            *mhd.bcell[1].view().at(c),
            *mhd.bcell[2].view().at(c),
        ]),
    );
    let prim = nmhd_recover(&IdealGas { gamma: GAMMA }, &cons);
    (prim.rho(), prim.pre())
}

// drive one curvilinear 1.5D MHD shell (spherical or cylindrical) and assert stability, the normal
// field's persistence, and the out-of-plane field's evolution.
fn run_shell<M>(
    mut sim: SimStateGeneric<NewtonianMhd, 1, 3, M, IdealGas<f64>, CpuSpace, HostMemory>,
    spherical: bool,
    label: &str,
) where
    M: symbi_geometry::Metric<f64, 1> + symbi_geometry::Metric<f64, 3> + Copy + Send + Sync,
{
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 1>::new(
        GAMMA,
        CFL,
        /* theta */ 1.5,
        &sim.geom.allocated,
    );

    // sample the toroidal B_phi IC per cell (it must move off this by the end).
    let mhd0 = sim.fields.mhd.as_ref().unwrap();
    let bphi_ic: Vec<f64> = (0..N as isize)
        .map(|i| *mhd0.bcell[2].view().at([i]))
        .collect();

    let mut steps = 0u64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        // the normal radial face field is never curled at D=1 — it holds its div-free IC.
        let mhd = s.fields.mhd.as_ref().unwrap();
        for c in &s.geom.interior.extend(0, 0, 1) {
            let rf = s.geom.face_coord(c, 0)[0];
            let br = *mhd.bface[0].view().at(c);
            let want = radial_bfield(spherical, rf);
            assert!(
                (br - want).abs() < 1e-10,
                "{label}: normal B_r drifted at face {c:?} (r={rf:.4}): {br:e} vs {want:e} (iter {})",
                s.iteration,
            );
        }
        steps = s.iteration;
    })
    .unwrap_or_else(|e| panic!("{label}: 1.5D curvilinear MHD evolve failed: {e:?}"));

    assert!(
        steps >= 10,
        "{label}: only {steps} steps — gate barely exercised"
    );

    // physicality on the shell + the out-of-plane toroidal field actually evolved.
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let mut max_dphi = 0.0_f64;
    for i in 0..N as isize {
        let (rho, p) = recover_1d(&sim, [i]);
        assert!(rho.is_finite() && rho > 0.0, "{label}: cell {i} rho={rho}");
        assert!(p.is_finite() && p > 0.0, "{label}: cell {i} p={p}");
        let bphi = *mhd.bcell[2].view().at([i]);
        max_dphi = max_dphi.max((bphi - bphi_ic[i as usize]).abs());
    }
    assert!(
        max_dphi > 1e-6,
        "{label}: out-of-plane B_phi never evolved (max change {max_dphi:e}) — the curvilinear \
         1.5D induction path is not running",
    );

    eprintln!(
        "[{label}] DONE iter={} t={:.4e} max |dB_phi| = {:e}",
        sim.iteration, sim.time, max_dphi
    );
}

// build the common 1.5D shell IC: div-free radial B_r on the r-faces + cell-centered
// (B_r, 0, B_phi) + a smooth radial pressure bump driving a mild radial flow.
fn seed_and_run<M>(spherical: bool, geometry: M, label: &str)
where
    M: symbi_geometry::Metric<f64, 1> + symbi_geometry::Metric<f64, 3> + Copy + Send + Sync,
{
    let sim = SimStateGeneric::<NewtonianMhd, 1, 3, M, IdealGas<f64>, CpuSpace, HostMemory>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        geometry,
    )
    .cells([N])
    .origin([R_LO])
    .spacing([DR])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .unwrap_or_else(|e| panic!("{label}: construction failed: {e:?}"))
    .set_initial(|[r]| {
        let br = radial_bfield(spherical, r);
        let pre = 1.0 + 0.4 * (-((r - (R_LO + 0.5 * N as f64 * DR)) / (4.0 * DR)).powi(2)).exp();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(pre)),
            Tensor::new([br, 0.0, BPHI0]),
        )
    })
    .seed_faces(|axis, [r]| {
        if axis == 0 {
            radial_bfield(spherical, r)
        } else {
            0.0
        }
    })
    .build();
    run_shell(sim, spherical, label);
}

#[test]
fn nmhd_1p5d_spherical_shell_stable_and_evolves() {
    // the reduced-dim (D=1, ungridded theta) spherical geometric source evaluates its angular
    // Christoffels at the equatorial default theta = pi/2 (the ungridded polar slot fill); a
    // theta = 0 fill diverges cot(theta) and NaNs the state on step 1.
    seed_and_run(true, Spherical, "nmhd_1p5d_sph");
}

#[test]
fn nmhd_1p5d_cylindrical_shell_stable_and_evolves() {
    seed_and_run(false, Cylindrical, "nmhd_1p5d_cyl");
}
