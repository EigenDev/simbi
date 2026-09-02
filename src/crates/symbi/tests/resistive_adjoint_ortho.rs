// =============================================================================
// resistive_adjoint_ortho.rs
//
// the mimetic-adjoint oracle for the covariant orthogonal-chart resistive EMF, across every 2.5D
// chart: cyl r-z, cyl r-phi, spherical r-theta. one kernel (the dec codifferential via the chart's
// Lame scale factors) must be the exact adjoint of that chart's induction curl, certified by the
// geometry-agnostic dissipation identity
//
//     <B, curl(J B)>_F = <J B, J B>_E >= 0     (eta = 1),
//
// to machine precision, with the dec Hodge weights w_E = h2, w_{B0} = h1 h2, w_{B1} = h0 h2 (h2 the
// out-of-plane scale factor). random compact-support fields keep every stencil in the full-stencil
// interior so the identity is exact. cyl r-z carries no metric factor in its current; cyl r-phi and
// spherical exercise the (1/r) d_r(r .) metric — the proof that one covariant kernel serves all
// charts.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_substrate::regimes::mhd_substrate::{apply_resistive_emf, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

type Store = symbi_sim::state::FieldStore<2, 3, HostMemory, f64>;

const N: usize = 24;
const GAMMA: f64 = 5.0 / 3.0;
const PAD: isize = 3; // compact support keeps curl(J .) (two-cell reach) in the full-stencil interior

// the in-plane axis-0, in-plane axis-1, and out-of-plane Lame scale factors at coordinate (x0, x1).
// these must match Metric::scale_factors the kernel bakes: cyl -> CylindricalRPhi, sph -> Spherical.
#[derive(Clone, Copy)]
enum Chart {
    CylRz,
    CylRphi,
    Sph,
}
fn scale_factors(chart: Chart, x0: f64, x1: f64) -> [f64; 3] {
    match chart {
        Chart::CylRz => [1.0, 1.0, x0],         // (h_r, h_z, h_phi = r)
        Chart::CylRphi => [1.0, x0, 1.0],       // (h_r, h_phi = r, h_z = 1)
        Chart::Sph => [1.0, x0, x0 * x1.sin()], // (h_r, h_theta = r, h_phi = r sin theta)
    }
}

fn rnd(i: isize, j: isize, salt: u64) -> f64 {
    let mut x = (i as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}
fn in_window(c: [isize; 2]) -> bool {
    c[0] >= PAD && c[0] < N as isize - PAD && c[1] >= PAD && c[1] < N as isize - PAD
}
fn b0_seed(c: [isize; 2]) -> f64 {
    if in_window(c) {
        rnd(c[0], c[1], 1)
    } else {
        0.0
    }
}
fn b1_seed(c: [isize; 2]) -> f64 {
    if in_window(c) {
        rnd(c[0], c[1], 2)
    } else {
        0.0
    }
}

// seed the random poloidal face field, zero the corner EMF.
fn seed(fs: &Store) {
    let m = fs.fields.mhd.as_ref().unwrap();
    for c in m.bface[0].domain().iter() {
        m.bface[0].set(c, b0_seed(c));
    }
    for c in m.bface[1].domain().iter() {
        m.bface[1].set(c, b1_seed(c));
    }
    for c in m.efield[0].domain().iter() {
        m.efield[0].set(c, 0.0);
    }
}
fn reset_bface(fs: &Store) {
    let m = fs.fields.mhd.as_ref().unwrap();
    for c in m.bface[0].domain().iter() {
        m.bface[0].set(c, 0.0);
    }
    for c in m.bface[1].domain().iter() {
        m.bface[1].set(c, 0.0);
    }
}

// the coordinate positions of the staggered locations at field coord [i,j].
fn face0(fs: &Store, i: isize) -> f64 {
    fs.geom.x_lo[0] + i as f64 * fs.geom.dx[0]
}
fn cen0(fs: &Store, i: isize) -> f64 {
    fs.geom.x_lo[0] + (i as f64 + 0.5) * fs.geom.dx[0]
}
fn face1(fs: &Store, j: isize) -> f64 {
    fs.geom.x_lo[1] + j as f64 * fs.geom.dx[1]
}
fn cen1(fs: &Store, j: isize) -> f64 {
    fs.geom.x_lo[1] + (j as f64 + 0.5) * fs.geom.dx[1]
}

// <J B, J B>_E = sum_corner (J B)^2 * w_E, with w_E = h2 at the corner (x0_face, x1_face).
fn edge_norm(fs: &Store, chart: Chart) -> f64 {
    let m = fs.fields.mhd.as_ref().unwrap();
    m.efield[0]
        .domain()
        .iter()
        .map(|c| {
            let jb = *m.efield[0].at(c);
            let w_e = scale_factors(chart, face0(fs, c[0]), face1(fs, c[1]))[2];
            jb * jb * w_e
        })
        .sum()
}

// <B, curl(J B)>_F with curl(J B) = -bface (dt = 1, bface reset to 0). w_{B0} = h1 h2 at B0's
// location (x0_face, x1_center); w_{B1} = h0 h2 at B1's (x0_center, x1_face).
fn face_pairing(fs: &Store, chart: Chart) -> f64 {
    let m = fs.fields.mhd.as_ref().unwrap();
    let mut s = 0.0;
    for c in m.bface[0].domain().iter() {
        let h = scale_factors(chart, face0(fs, c[0]), cen1(fs, c[1]));
        s += b0_seed(c) * (-*m.bface[0].at(c)) * (h[1] * h[2]);
    }
    for c in m.bface[1].domain().iter() {
        let h = scale_factors(chart, cen0(fs, c[0]), face1(fs, c[1]));
        s += b1_seed(c) * (-*m.bface[1].at(c)) * (h[0] * h[2]);
    }
    s
}

// the shared certificate: <B, curl(J B)>_F == <J B, J B>_E > 0 to machine precision.
fn assert_adjoint(fs: &Store, chart: Chart, label: &str) {
    seed(fs);
    apply_resistive_emf::<2, 3, HostMemory, f64>(fs, 1.0);
    let norm_e = edge_norm(fs, chart);
    reset_bface(fs);
    ct_curl::<2, 3, HostMemory, f64>(fs, 1.0);
    let pair_f = face_pairing(fs, chart);
    assert!(
        norm_e > 1e-6,
        "{label}: degenerate oracle, <JB,JB>_E = {norm_e} ~ 0"
    );
    let rel = (pair_f - norm_e).abs() / norm_e;
    assert!(
        rel < 1e-10,
        "{label}: covariant resistive J is NOT the induction-curl adjoint: <B,curl(JB)>_F = {pair_f}, \
         <JB,JB>_E = {norm_e} (rel {rel:.3e})"
    );
}

fn still_gas() -> MhdPrim<f64, 3> {
    MhdPrim::new(
        Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
        Tensor::new([0.0, 0.0, 0.0]),
    )
}

#[test]
fn cyl_rz_covariant_adjoint() {
    let sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cylindrical,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
    .cells([N, N])
    .bounds([1.0, 0.0], [3.0, 1.0])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .allocate()
    .expect("cyl-rz sim")
    .set_initial(|_| still_gas())
    .seed_faces(|_, _| 0.0)
    .build();
    assert_adjoint(&sim, Chart::CylRz, "cyl r-z");
}

#[test]
fn cyl_rphi_covariant_adjoint() {
    let sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cylindrical,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
    .cells([N, N])
    .bounds([1.0, 0.0], [3.0, 1.0])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .cyl_plane(CylPlane::RPhi)
    .allocate()
    .expect("cyl-rphi sim")
    .set_initial(|_| still_gas())
    .seed_faces(|_, _| 0.0)
    .build();
    assert_adjoint(&sim, Chart::CylRphi, "cyl r-phi");
}

#[test]
fn spherical_covariant_adjoint() {
    // theta in [0.6, 2.4] keeps sin(theta) well away from the r=0 / pole singularities.
    let sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Spherical,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Spherical)
    .cells([N, N])
    .bounds([1.0, 0.6], [3.0, 2.4])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .allocate()
    .expect("spherical sim")
    .set_initial(|_| still_gas())
    .seed_faces(|_, _| 0.0)
    .build();
    assert_adjoint(&sim, Chart::Sph, "spherical r-theta");
}
