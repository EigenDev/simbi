// =============================================================================
// resistive_rate_curvilinear.rs
//
// the RATE certificate for curvilinear resistive MHD: the adjoint oracle proves the operator is a
// stable (negative-definite) diffusion, but not that it diffuses at the CORRECT physical rate. this
// seeds the analytic free-decay EIGENMODES and confirms the discrete operator returns their
// eigenvalue via the Rayleigh quotient
//
//     lambda = - <B, L B>_F / (eta <B, B>_F)  ==  k^2     (L B = eta grad^2 B),
//
// with the physical DEC weights, over an interior sub-window (so boundary conditions never enter).
//   - CYLINDRICAL: the poloidal B_z(r) = J_0(k r) mode diffuses by the scalar cyl Laplacian
//     (1/r) d_r(r d_r .), eigenvalue k^2 (Bessel's equation). J_0 via Abramowitz & Stegun 9.4.
//   - SPHERICAL: the l=1 dipole free-decay mode from the vector potential A_phi = j_1(k r) sin(theta)
//     (spherical Bessel j_1 -- ELEMENTARY -- times the Legendre P_1^1 = sin theta), eigenvalue k^2
//     of the l=1 radial operator d_rr + (2/r) d_r - 2/r^2. this is the textbook magnetic-dipole decay.
// the discrete Rayleigh quotient matches k^2 to the O(dx^2) truncation error -- the metric is handled
// correctly (it lives in the induction curl + the weights).
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_substrate::regimes::mhd_substrate::{apply_resistive_emf, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

type Store = symbi_sim::state::FieldStore<2, 3, HostMemory, f64>;

const N: usize = 64;
const GAMMA: f64 = 5.0 / 3.0;
const K: f64 = std::f64::consts::PI; // mode wavenumber; k^2 is the target eigenvalue
const BAND: isize = 5; // exclude a boundary band from the Rayleigh sum (sub-window is bc-free)

// the cylindrical Bessel J_0 (Abramowitz & Stegun 9.4.1 / 9.4.3), ~1e-7 accurate.
fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.0 {
        let t2 = (x / 3.0).powi(2);
        1.0 - 2.2499997 * t2 + 1.2656208 * t2.powi(2) - 0.3163866 * t2.powi(3)
            + 0.0444479 * t2.powi(4)
            - 0.0039444 * t2.powi(5)
            + 0.0002100 * t2.powi(6)
    } else {
        let z = 3.0 / ax;
        let f0 = 0.79788456 - 0.00000077 * z - 0.00552740 * z * z - 0.00009512 * z.powi(3)
            + 0.00137237 * z.powi(4)
            - 0.00072805 * z.powi(5)
            + 0.00014476 * z.powi(6);
        let th = ax - 0.78539816 - 0.04166397 * z - 0.00003954 * z * z + 0.00262573 * z.powi(3)
            - 0.00054125 * z.powi(4)
            - 0.00029333 * z.powi(5)
            + 0.00013558 * z.powi(6);
        f0 * th.cos() / ax.sqrt()
    }
}
// elementary spherical Bessel functions.
fn sph_j0(x: f64) -> f64 {
    if x.abs() < 1e-8 { 1.0 } else { x.sin() / x }
}
fn sph_j1(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        0.0
    } else {
        x.sin() / (x * x) - x.cos() / x
    }
}

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

fn in_band(c: [isize; 2]) -> bool {
    c[0] >= BAND && c[0] < N as isize - BAND && c[1] >= BAND && c[1] < N as isize - BAND
}

// L B via one isolated resistive substep (efield=0 -> add eta J -> curl), returning dB = dt L B.
// eta = dt = 1, so dB = L B; then lambda = -<B,dB>/<B,B>. `seed` fills the faces; `w0`,`w1` are the
// physical face weights; the seed closures give the analytic mode value at a face coordinate.
fn rayleigh(
    fs: &Store,
    seed0: impl Fn(&Store, [isize; 2]) -> f64,
    seed1: impl Fn(&Store, [isize; 2]) -> f64,
    w0: impl Fn(&Store, [isize; 2]) -> f64,
    w1: impl Fn(&Store, [isize; 2]) -> f64,
) -> f64 {
    let m = fs.fields.mhd.as_ref().unwrap();
    for c in m.bface[0].domain().iter() {
        m.bface[0].set(c, seed0(fs, c));
    }
    for c in m.bface[1].domain().iter() {
        m.bface[1].set(c, seed1(fs, c));
    }
    for c in m.efield[0].domain().iter() {
        m.efield[0].set(c, 0.0);
    }
    apply_resistive_emf::<2, 3, HostMemory, f64>(fs, 1.0);
    ct_curl::<2, 3, HostMemory, f64>(fs, 1.0);
    // dB = bface_after - seed. lambda = -sum B*dB*w / sum B^2*w over the interior sub-window.
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for c in m.bface[0].domain().iter() {
        if !in_band(c) {
            continue;
        }
        let b = seed0(fs, c);
        let db = *m.bface[0].at(c) - b;
        let w = w0(fs, c);
        num += -b * db * w;
        den += b * b * w;
    }
    for c in m.bface[1].domain().iter() {
        if !in_band(c) {
            continue;
        }
        let b = seed1(fs, c);
        let db = *m.bface[1].at(c) - b;
        let w = w1(fs, c);
        num += -b * db * w;
        den += b * b * w;
    }
    num / den
}

fn still_gas() -> MhdPrim<f64, 3> {
    MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        },
        mag: Tensor::new([0.0, 0.0, 0.0]),
    }
}

#[test]
fn cylindrical_bessel_decay_rate() {
    // cyl r-z: B_z(r) = J_0(k r), B_r = 0. diffuses by (1/r) d_r(r d_r) -> eigenvalue k^2.
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
    // B_z on z-faces sits at (r_center, z_face); B_r = 0. weight w_{B_z} = r_center.
    let lambda = rayleigh(
        &sim,
        |_, _| 0.0,
        |fs, c| bessel_j0(K * cen0(fs, c[0])),
        |_, _| 1.0,
        |fs, c| cen0(fs, c[0]),
    );
    let target = K * K;
    let rel = (lambda - target).abs() / target;
    // the O(dx^2) truncation lands at ~0.08% here; 0.5% bites on any real rate error (a wrong metric
    // factor is O(1)) while tolerating the discretization + the finite-domain mode content.
    assert!(
        rel < 0.005,
        "cyl Bessel decay rate off: lambda = {lambda:.5}, k^2 = {target:.5} (rel {rel:.3e})"
    );
}

#[test]
fn spherical_dipole_decay_rate() {
    // sph r-theta: the l=1 dipole from A_phi = j_1(k r) sin(theta). B_r = 2 j_1(kr) cos(theta)/r,
    // B_theta = -[(r j_1)'/r] sin(theta) = -[k j_0(kr) - j_1(kr)/r] sin(theta). eigenvalue k^2 of the
    // l=1 radial operator. weights: w_{B_r} = r^2 sin(theta), w_{B_theta} = r sin(theta).
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
    .bounds([1.0, 0.5], [3.0, 2.6])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .allocate()
    .expect("spherical sim")
    .set_initial(|_| still_gas())
    .seed_faces(|_, _| 0.0)
    .build();
    let lambda = rayleigh(
        &sim,
        // B_r on r-faces at (r_face, theta_center).
        |fs, c| {
            let r = face0(fs, c[0]);
            let th = cen1(fs, c[1]);
            2.0 * sph_j1(K * r) * th.cos() / r
        },
        // B_theta on theta-faces at (r_center, theta_face).
        |fs, c| {
            let r = cen0(fs, c[0]);
            let th = face1(fs, c[1]);
            -(K * sph_j0(K * r) - sph_j1(K * r) / r) * th.sin()
        },
        |fs, c| {
            let r = face0(fs, c[0]);
            let th = cen1(fs, c[1]);
            r * r * th.sin()
        },
        |fs, c| {
            let r = cen0(fs, c[0]);
            let th = face1(fs, c[1]);
            r * th.sin()
        },
    );
    let target = K * K;
    let rel = (lambda - target).abs() / target;
    // ~0.9% here (dipole mode + O(dx^2)); 2% bites on a wrong metric factor (O(1)).
    assert!(
        rel < 0.02,
        "spherical dipole decay rate off: lambda = {lambda:.5}, k^2 = {target:.5} (rel {rel:.3e})"
    );
}
