// =============================================================================
// resistive_adjoint_3d.rs
//
// the mimetic-adjoint oracle for 3D curvilinear resistive MHD (spherical, cylindrical). the same
// metric-free difference curl that serves 3D cartesian is the exact adjoint of the 3D curvilinear
// induction curl (whose metric lives in C + the physical face-area weights), certified by the
// geometry-agnostic dissipation identity
//
//     <B, curl(J B)>_F = <J B, J B>_E >= 0     (eta = 1),
//
// to machine precision, across all three edges/faces. dec weights: w_{B_k} = product of the other two
// scale factors at the k-face; w_{E_k} = h_k at the k-edge. random compact-support fields keep every
// stencil in the full-stencil interior so the identity is exact.
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

type Store = symbi_sim::state::FieldStore<3, 3, HostMemory, f64>;

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const PAD: isize = 3;

#[derive(Clone, Copy)]
enum Chart {
    Sph,
    Cyl,
}
// (h0, h1, h2) at coordinate (x0, x1, x2). must match Metric::scale_factors of the chart.
fn h(chart: Chart, x: [f64; 3]) -> [f64; 3] {
    match chart {
        Chart::Sph => [1.0, x[0], x[0] * x[1].sin()], // (h_r, h_theta=r, h_phi=r sin theta)
        Chart::Cyl => [1.0, x[0], 1.0],               // (h_r, h_phi=r, h_z=1)
    }
}

fn rnd(c: [isize; 3], salt: u64) -> f64 {
    let mut x = (c[0] as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (c[1] as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ (c[2] as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}
fn in_window(c: [isize; 3]) -> bool {
    (0..3).all(|a| c[a] >= PAD && c[a] < N as isize - PAD)
}
fn b_seed(c: [isize; 3], k: usize) -> f64 {
    if in_window(c) {
        rnd(c, k as u64 + 1)
    } else {
        0.0
    }
}

fn pos(fs: &Store, c: [isize; 3], face_axis: usize) -> [f64; 3] {
    // the staggered coordinate: `face_axis` sits on the face (integer), the others at cell centers.
    std::array::from_fn(|a| {
        let base = fs.geom.x_lo[a];
        let d = fs.geom.dx[a];
        if a == face_axis {
            base + c[a] as f64 * d
        } else {
            base + (c[a] as f64 + 0.5) * d
        }
    })
}
// the edge along axis k: axis k at the cell center, the two transverse axes on faces.
fn pos_edge(fs: &Store, c: [isize; 3], edge_axis: usize) -> [f64; 3] {
    std::array::from_fn(|a| {
        let base = fs.geom.x_lo[a];
        let d = fs.geom.dx[a];
        if a == edge_axis {
            base + (c[a] as f64 + 0.5) * d
        } else {
            base + c[a] as f64 * d
        }
    })
}

fn seed(fs: &Store) {
    let m = fs.fields.mhd.as_ref().unwrap();
    for k in 0..3 {
        for c in m.bface[k].domain().iter() {
            m.bface[k].set(c, b_seed(c, k));
        }
    }
    for k in 0..3 {
        for c in m.efield[k].domain().iter() {
            m.efield[k].set(c, 0.0);
        }
    }
}

fn assert_adjoint_3d(fs: &Store, chart: Chart, label: &str) {
    seed(fs);
    apply_resistive_emf::<3, 3, HostMemory, f64>(fs, 1.0);
    // <J B, J B>_E = sum_k sum_edge (efield[k])^2 * w_{E_k}, w_{E_k} = h_k at the k-edge.
    let m = fs.fields.mhd.as_ref().unwrap();
    let mut norm_e = 0.0;
    for k in 0..3 {
        for c in m.efield[k].domain().iter() {
            let jb = *m.efield[k].at(c);
            let w = h(chart, pos_edge(fs, c, k))[k];
            norm_e += jb * jb * w;
        }
    }
    // curl(J B) = -bface after reset + ct_curl(dt=1).
    for k in 0..3 {
        for c in m.bface[k].domain().iter() {
            m.bface[k].set(c, 0.0);
        }
    }
    ct_curl::<3, 3, HostMemory, f64>(fs, 1.0);
    // <B, curl(J B)>_F = sum_k sum_face B_k*(-bface[k])*w_{B_k}, w_{B_k} = prod of the other two h.
    let mut pair_f = 0.0;
    for k in 0..3 {
        for c in m.bface[k].domain().iter() {
            let hk = h(chart, pos(fs, c, k));
            let w: f64 = (0..3).filter(|&a| a != k).map(|a| hk[a]).product();
            pair_f += b_seed(c, k) * (-*m.bface[k].at(c)) * w;
        }
    }
    assert!(
        norm_e > 1e-6,
        "{label}: degenerate oracle, <JB,JB>_E = {norm_e} ~ 0"
    );
    let rel = (pair_f - norm_e).abs() / norm_e;
    assert!(
        rel < 1e-10,
        "{label}: 3D resistive J is NOT the induction-curl adjoint: <B,curl(JB)>_F = {pair_f}, \
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
fn spherical_3d_adjoint() {
    // r in [1,2], theta in [0.8, 2.3] (off the poles), phi in [0,1].
    let sim = SimState::<NewtonianMhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Spherical,
    )
    .cells([N, N, N])
    .bounds([1.0, 0.8, 0.0], [2.0, 2.3, 1.0])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .allocate()
    .expect("3d spherical sim")
    .set_initial(|_| still_gas())
    .seed_faces(|_, _| 0.0)
    .build();
    assert_adjoint_3d(&sim, Chart::Sph, "3D spherical");
}

#[test]
fn cylindrical_3d_adjoint() {
    // r in [1,2], phi in [0,1], z in [0,1].
    let sim = SimState::<NewtonianMhd, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([N, N, N])
    .bounds([1.0, 0.0, 0.0], [2.0, 1.0, 1.0])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .allocate()
    .expect("3d cylindrical sim")
    .set_initial(|_| still_gas())
    .seed_faces(|_, _| 0.0)
    .build();
    assert_adjoint_3d(&sim, Chart::Cyl, "3D cylindrical");
}
