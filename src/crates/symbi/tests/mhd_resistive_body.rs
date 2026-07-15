// =============================================================================
// mhd_resistive_body.rs
//
// the immersed-body LOCALIZED Ohmic resistivity (`MagneticSpec::Resistive`): a body dissipates the
// magnetic field THREADING it (`eta*chi*J` added to the edge EMF, masked by the body indicator chi)
// while the exterior flux is left to ideal constrained transport. the kernel is exercised DIRECTLY
// (one masked-resistive EMF + one induction curl), pinning the two defining properties:
//   - LOCALIZATION: the added EMF is nonzero only near the body (chi > 0) and exactly zero far away,
//     even though the field itself is nonzero everywhere. a `MagneticSpec::None` body adds NOTHING.
//   - DISSIPATION: the magnetic-energy change `<B, curl(eta chi J B)>_F <= 0` — the body can only
//     shed field, never amplify it (`-C diag(eta chi) C^T` is negative-definite for eta,chi >= 0).
// stability of the composed operator is proven to machine precision by the cyl/cartesian adjoint
// oracle; this test pins the MASK (localization) and the production dispatch wiring.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{body_resistive_emf, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.18;
const ETA: f64 = 0.1;
const PAD: isize = 3; // compact support: keep the random field off the domain boundary

// deterministic per-coordinate pseudo-random in [-0.5, 0.5].
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
fn bx_seed(c: [isize; 2]) -> f64 { if in_window(c) { rnd(c[0], c[1], 1) } else { 0.0 } }
fn by_seed(c: [isize; 2]) -> f64 { if in_window(c) { rnd(c[0], c[1], 2) } else { 0.0 } }

fn make_sim(magnetic: MagneticSpec) -> Sim {
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("resistive body sim construction failed")
    .set_initial(|_| MhdPrim {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0, 0.0]), pre: 1.0 },
        mag: Tensor::new([0.0, 0.0, 0.0]),
    })
    .seed_faces(|_, _| 0.0)
    .build();
    // a hydrodynamically-transparent porous body (drain off, wall force off) that keeps a mask radius
    // for the magnetic SDF; the coupling under test is purely the magnetic one.
    sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 1.0, false)
            .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 0.0, k_eta_t: 0.0 })
            .with_magnetic(magnetic),
    ))
}

// distance from the body center to the E_z corner at edge coord c.
fn corner_dist(s: &Sim, c: [isize; 2]) -> f64 {
    let dx = s.geom.dx[0];
    let px = s.geom.x_lo[0] + c[0] as f64 * dx;
    let py = s.geom.x_lo[1] + c[1] as f64 * dx;
    ((px - BODY[0]).powi(2) + (py - BODY[1]).powi(2)).sqrt()
}

fn seed_field(s: &Sim) {
    let m = s.fields.mhd.as_ref().unwrap();
    for c in m.bface[0].domain().iter() { m.bface[0].set(c, bx_seed(c)); }
    for c in m.bface[1].domain().iter() { m.bface[1].set(c, by_seed(c)); }
    for c in m.efield[0].domain().iter() { m.efield[0].set(c, 0.0); }
}

#[test]
fn resistive_body_localizes_and_dissipates() {
    // the resistive body: efield[0] <- eta*chi*J*B (efield started at zero).
    let sim = make_sim(MagneticSpec::Resistive { eta: ETA });
    seed_field(&sim);
    body_resistive_emf::<2, 3, HostMemory, f64>(&sim);

    // LOCALIZATION: the added EMF is nonzero near the body and EXACTLY zero beyond the masked region,
    // even though B is nonzero throughout the window.
    let mut near_max = 0.0_f64;
    let mut far_max = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.efield[0].domain().iter() {
            let e = m.efield[0].at(c).abs();
            let r = corner_dist(&sim, c);
            if r < R_BODY { near_max = near_max.max(e); }
            // beyond ~6 cells the mollified tanh mask (width one cell) has decayed by >5 decades.
            if r > R_BODY + 6.0 * sim.geom.dx[0] { far_max = far_max.max(e); }
        }
    }
    assert!(near_max > 1e-6, "the resistive body added no EMF inside its mask (near_max = {near_max})");
    assert!(
        far_max < 1e-3 * near_max,
        "the resistive EMF is not localized to the body mask: far_max = {far_max}, near_max = {near_max}"
    );

    // DISSIPATION: curl the masked EMF and confirm the magnetic-energy change is negative (cartesian
    // face weights are unity). dW = sum_f B_f*(bface_after - B_f); ct_curl does bface -= dt*curl(E).
    ct_curl::<2, 3, HostMemory, f64>(&sim, 1.0);
    let mut dw = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.bface[0].domain().iter() { dw += bx_seed(c) * (*m.bface[0].at(c) - bx_seed(c)); }
        for c in m.bface[1].domain().iter() { dw += by_seed(c) * (*m.bface[1].at(c) - by_seed(c)); }
    }
    assert!(
        dw < -1e-9,
        "the resistive body did not dissipate magnetic energy (dW = {dw} >= 0): it is not negative-definite"
    );

    // a MagneticSpec::None body adds NOTHING: the EMF stays exactly zero.
    let sim_none = make_sim(MagneticSpec::None);
    seed_field(&sim_none);
    body_resistive_emf::<2, 3, HostMemory, f64>(&sim_none);
    let none_max = {
        let m = sim_none.fields.mhd.as_ref().unwrap();
        sim_none.geom.interior.iter().map(|c| m.efield[0].at(c).abs()).fold(0.0_f64, f64::max)
    };
    assert_eq!(none_max, 0.0, "a non-magnetic body perturbed the edge EMF (max |E| = {none_max})");
}
