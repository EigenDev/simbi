// =============================================================================
// resistive_adjoint_cyl.rs
//
// the mimetic-adjoint oracle for the cylindrical r-z Ohmic resistive edge EMF. the resistive current
// J_phi must be the discrete adjoint of the induction curl C w.r.t. the physical (volume-weighted)
// inner products, so that the resistive operator -curl(eta J) is symmetric negative-definite and the
// magnetic energy can only decay. the certificate is the discrete dissipation identity
//
//     <B, curl(J B)>_F = <J B, J B>_E >= 0
//
// (eta = 1), which holds to machine precision iff J = C^T with the cyl weights w_r = w_E = r_face,
// w_z = r_cell. it stands on the adjoint identity alone, so it is geometry-agnostic: any wrong
// r-weight or a flipped sign (which would make the resistive term grow energy) breaks it. random compact-support
// fields keep every stencil evaluation in the full-stencil interior, so the identity is exact.
// =============================================================================

use std::f64::consts::PI;
use symbi_hydro::quantity::{Density, Pressure};

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_substrate::regimes::mhd_substrate::{apply_resistive_emf, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

// 2.5D cylindrical r-z MHD: D=2 spatial (r, z), DOF=3 (the out-of-plane B_phi/v_phi).
type Sim =
    SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 24;
const GAMMA: f64 = 5.0 / 3.0;
const R_MIN: f64 = 1.0; // stay off the r=0 axis (the metric is singular there)
const R_MAX: f64 = 3.0;
const Z_MAX: f64 = 1.0;
// compact support: random fields live in [pad, N-pad) on both axes so curl(J .) (a two-cell reach)
// only ever samples full-stencil interior cells -> the discrete adjoint identity is exact.
const PAD: isize = 3;

// a deterministic per-coordinate pseudo-random value in [-0.5, 0.5] (splitmix-style hash). keeps the
// test reproducible with no rng dependency; `salt` decorrelates the B_r / B_z components.
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

// the random poloidal face field on the interior window (zero elsewhere, incl. ghosts).
fn br_seed(c: [isize; 2]) -> f64 {
    if in_window(c) {
        rnd(c[0], c[1], 1)
    } else {
        0.0
    }
}
fn bz_seed(c: [isize; 2]) -> f64 {
    if in_window(c) {
        rnd(c[0], c[1], 2)
    } else {
        0.0
    }
}

fn make_sim() -> Sim {
    SimStateGeneric::<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([N, N])
    .bounds([R_MIN, 0.0], [R_MAX, Z_MAX])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(0.3)
    .allocate()
    .expect("cyl r-z adjoint sim construction failed")
    .set_initial(|_| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([0.0, 0.0, 0.0])))
    .seed_faces(|_, _| 0.0)
    .build()
}

#[test]
fn cyl_rz_resistive_current_is_the_curl_adjoint() {
    let sim = make_sim();
    let dr = sim.geom.dx[0];
    let r_min = sim.geom.x_lo[0];
    // the mimetic cyl weights: r-faces (B_r) and corners (E_phi) sit at r = r_min + i*dr; the r-cell
    // center (B_z, a z-face) at r = r_min + (i + 1/2)*dr. these are exactly the w that make J = C^T.
    let w_face = |i: isize| r_min + (i as f64) * dr;
    let w_cell = |i: isize| r_min + (i as f64 + 0.5) * dr;

    // seed the random poloidal face field B and zero the corner EMF.
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.bface[0].domain().iter() {
            m.bface[0].set(c, br_seed(c));
        }
        for c in m.bface[1].domain().iter() {
            m.bface[1].set(c, bz_seed(c));
        }
        for c in m.efield[0].domain().iter() {
            m.efield[0].set(c, 0.0);
        }
    }

    // J: efield[0] <- eta * J_phi(B), eta = 1. after this efield[0] holds J.B on the corners.
    apply_resistive_emf::<2, 3, HostMemory, f64>(&sim, 1.0);

    // the edge-norm <J B, J B>_E = sum_corner (J B)^2 * r_face.
    let mut norm_e = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.efield[0].domain().iter() {
            let jb = *m.efield[0].at(c);
            norm_e += jb * jb * w_face(c[0]);
        }
    }

    // curl(J B): reset the faces to zero, keep efield[0] = J B, apply the induction curl with dt = 1.
    // ct_curl writes bface <- bface - dt*curl(efield), so with bface = 0 the result is -curl(J B).
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.bface[0].domain().iter() {
            m.bface[0].set(c, 0.0);
        }
        for c in m.bface[1].domain().iter() {
            m.bface[1].set(c, 0.0);
        }
    }
    ct_curl::<2, 3, HostMemory, f64>(&sim, 1.0);

    // the face-pairing <B, curl(J B)>_F = sum_face B_r*curl_r*w_face + sum_cell B_z*curl_z*w_cell,
    // where curl(J B) = -bface (dt = 1, bface started at 0).
    let mut pair_f = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.bface[0].domain().iter() {
            let curl_r = -*m.bface[0].at(c);
            pair_f += br_seed(c) * curl_r * w_face(c[0]);
        }
        for c in m.bface[1].domain().iter() {
            let curl_z = -*m.bface[1].at(c);
            pair_f += bz_seed(c) * curl_z * w_cell(c[0]);
        }
    }

    // the identity: the face-pairing equals the (strictly positive) edge-norm to machine precision.
    // a wrong r-weight breaks the equality; a flipped sign makes pair_f negative (energy growth).
    assert!(
        norm_e > 1e-6,
        "degenerate oracle: <J B, J B>_E = {norm_e} is ~0, the test proves nothing"
    );
    let rel = (pair_f - norm_e).abs() / norm_e;
    assert!(
        rel < 1e-10,
        "cyl r-z resistive J is NOT the induction-curl adjoint: <B, curl(J B)>_F = {pair_f}, \
         <J B, J B>_E = {norm_e} (relative mismatch {rel:.3e}). J != C^T -> the resistive operator \
         is not symmetric-negative-definite and the magnetic energy would not provably decay."
    );
}

// ---- end-to-end: the production evolve path (CFL fold + post_godunov routing) ----

const B0: f64 = 1e-2; // tiny: magnetic pressure << p, so ideal dynamics are negligible over the run
const T_FINAL: f64 = 0.3;

// a div-free poloidal seed: B_z = B0 sin(k (r - r_min)), B_r = 0. div B = (1/r) d_r(r B_r) + d_z B_z
// = 0 (B_r = 0 and B_z has no z-dependence), so the CT machinery starts monopole-free. two radial
// wavelengths (k (R_MAX - R_MIN) = 4 pi) -> the mode vanishes at both r-walls and decays briskly.
fn make_decay_sim() -> Sim {
    let k = 4.0 * PI / (R_MAX - R_MIN);
    SimStateGeneric::<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([N, N])
    .bounds([R_MIN, 0.0], [R_MAX, Z_MAX])
    .boundaries(Boundaries::per_axis([
        [BoundaryType::Reflect, BoundaryType::Reflect],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]))
    .cfl(0.3)
    .allocate()
    .expect("cyl r-z decay sim construction failed")
    .set_initial(|[r, _z]| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([0.0, 0.0, B0 * (k * (r - R_MIN)).sin()])))
    .seed_faces(|axis, [r, _z]| if axis == 1 { B0 * (k * (r - R_MIN)).sin() } else { 0.0 })
    .build()
}

// max |B_z| (cell-centered) over the interior — the poloidal field amplitude. cyl r-z stores the
// cell B in coordinate order [B_r, B_phi, B_z], so B_z is component 2 (component 1 is out-of-plane).
fn bz_amplitude(s: &Sim) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    s.geom
        .interior
        .iter()
        .map(|c| m.bcell[2].view().at(c).abs())
        .fold(0.0_f64, f64::max)
}

fn evolve_decay(eta: f64) -> f64 {
    let mut sim = make_decay_sim();
    let a0 = bz_amplitude(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(eta);
    evolve_with_callback(&mut sim, &sub, T_FINAL, u64::MAX, |_| {})
        .expect("cyl decay evolve failed");
    bz_amplitude(&sim) / a0
}

#[test]
fn cyl_rz_resistivity_dominates_the_ideal_numerical_diffusion() {
    // the end-to-end production path: the resistive CFL fold keeps the diffusion stable and
    // post_godunov routes the cyl chart through the adjoint resistive EMF. proof that the resistive
    // term is active (not silently skipped or a no-op): eta > 0 must lose substantially more field
    // than the ideal scheme's own finite-resolution numerical diffusion floor.
    let ideal = evolve_decay(0.0);
    let resistive = evolve_decay(0.05);
    // "dominates" = the resistive term erases far more field than the scheme's own numerical
    // diffusion floor. compare the fractional losses (the cyl sinusoid mixes eigenmodes of the
    // (1/r) d_r(r d_r) operator, so its absolute decay rate departs from the clean cartesian
    // exp(-eta k^2 t); the loss ratio is the geometry-robust statement). the resistive loss
    // dwarfs the ideal one.
    let ideal_loss = 1.0 - ideal;
    let resistive_loss = 1.0 - resistive;
    assert!(
        resistive_loss > 4.0 * ideal_loss,
        "cyl r-z resistivity did not dominate the numerical diffusion: ideal ratio {ideal} \
         (loss {ideal_loss:.4}), resistive {resistive} (loss {resistive_loss:.4})"
    );
}
