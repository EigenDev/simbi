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
//
// the seed B_z = b0 sin(k (r - r_min)), B_r = 0 (k (R_MAX - R_MIN) = 4 pi) is divergence-free on
// the faces and sits between reflecting radial walls with a periodic z. a reflecting wall mirrors
// the tangential B_z evenly, so the resistive decay is the radial diffusion
//     dB_z/dt = eta (1/r) d_r (r d_r B_z),  d_r B_z = 0 at both walls.
// decay is scored by the r-weighted projection of the cell B_z onto the seed profile, which reads
// the seeded mode; a per-step transverse check holds every arm z-uniform, independently of it.
// the ideal arm's loss is the scheme's numerical diffusion, second order in the spacing (its
// projection loss falls 3.5x from N = 24 to 48 and 3.96x from 48 to 96); at N = 24 that floor is
// 0.155 against a resistive loss of 0.431 and resistivity does not dominate it, at N = 48 it is
// 0.044 against 0.390 (ratio 8.8). the resistive projection converges to the radial diffusion
// reference at second order (error 0.054, 0.012, 0.0027 at N = 24, 48, 96).

const B0: f64 = 1e-2; // magnetic pressure / gas pressure = 5e-5
const T_FINAL: f64 = 0.3;
const ETA: f64 = 0.05;
const K: f64 = 4.0 * PI / (R_MAX - R_MIN);
/// the coarse fixture and the resolved one, on square grids
const COARSE: usize = 24;
const RESOLVED: usize = 48;
/// the dominance the resolved fixture asserts: resistive loss over the ideal numerical floor
const DOMINANCE: f64 = 4.0;
/// a second-order quantity shrinks 4x per doubling; 3 accepts it and rejects first order (2x)
const SECOND_ORDER_GAIN: f64 = 3.0;
/// the fraction of the resolved resistive error that the reference's own discretization error
/// and the neglected magnetic backreaction may each reach: at 1% neither moves the convergence
/// gain by more than 1%
const REFERENCE_SHARE: f64 = 1e-2;
/// the transverse bound on either face field's z-departure, in units of b0
const TRANSVERSE: f64 = 1e-12;

fn shape(r: f64) -> f64 {
    (K * (r - R_MIN)).sin()
}

fn make_decay_sim(n: usize, b0: f64) -> Sim {
    SimStateGeneric::<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([n, n])
    .bounds([R_MIN, 0.0], [R_MAX, Z_MAX])
    .boundaries(Boundaries::per_axis([
        [BoundaryType::Reflect, BoundaryType::Reflect],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]))
    .cfl(0.3)
    .allocate()
    .expect("cyl r-z decay sim construction failed")
    .set_initial(move |[r, _z]| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([0.0, 0.0, b0 * shape(r)])))
    .seed_faces(move |axis, [r, _z]| if axis == 1 { b0 * shape(r) } else { 0.0 })
    .build()
}

/// the r-weighted projection of the cell B_z onto the seed profile. cyl r-z stores the cell B in
/// coordinate order [B_r, B_phi, B_z], so B_z is component 2.
fn projection(s: &Sim) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    let lo = s.geom.interior.spaces[0].lo;
    let dr = s.geom.dx[0];
    let (mut num, mut den) = (0.0, 0.0);
    for c in s.geom.interior.iter() {
        let r = R_MIN + ((c[0] - lo) as f64 + 0.5) * dr;
        num += *m.bcell[2].view().at(c) * shape(r) * r;
        den += shape(r) * shape(r) * r;
    }
    num / den
}

/// the largest z-departure (max_z - min_z per radial row) of either face field
fn transverse(s: &Sim) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    let i = &s.geom.interior.spaces;
    let mut worst = 0.0f64;
    for f in m.bface.b.iter() {
        for ir in i[0].lo..i[0].hi {
            let (lo, hi) = (i[1].lo..i[1].hi).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), iz| {
                let v = *f.view().at([ir, iz]);
                (lo.min(v), hi.max(v))
            });
            worst = worst.max(hi - lo);
        }
    }
    worst
}

/// one arm: the final projection relative to the seed's, and the largest per-step transverse
/// departure relative to b0
#[derive(Clone, Copy, Debug)]
struct Decay {
    retained: f64,
    transverse: f64,
}

fn evolve_decay(n: usize, eta: f64, b0: f64) -> Decay {
    let mut sim = make_decay_sim(n, b0);
    let p0 = projection(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
        .with_resistivity(eta);
    let mut worst = 0.0f64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| worst = worst.max(transverse(s)))
        .expect("cyl decay evolve failed");
    assert!((sim.time - T_FINAL).abs() < 1e-12, "the arm ended at t = {}", sim.time);
    Decay { retained: projection(&sim) / p0, transverse: worst / b0 }
}

/// the radial diffusion reference: crank-nicolson on `cells` uniform radial cells over `steps`
/// equal steps, the finite-volume flux r d_r B through each cell face and zero flux through the
/// walls, started from the seed profile at the cell centers and scored by the same r-weighted
/// projection, relative to its start
fn radial_reference(cells: usize, steps: usize) -> f64 {
    let dr = (R_MAX - R_MIN) / cells as f64;
    let dt = T_FINAL / steps as f64;
    let rc: Vec<f64> = (0..cells).map(|ii| R_MIN + (ii as f64 + 0.5) * dr).collect();
    // A b = eta/(r dr^2) [r_+ (b_{i+1} - b_i) - r_- (b_i - b_{i-1})], wall faces carrying no flux
    let face = |ii: usize| R_MIN + ii as f64 * dr;
    let coef = |ii: usize| -> (f64, f64) {
        let w = ETA / (rc[ii] * dr * dr);
        let lo = if ii == 0 { 0.0 } else { w * face(ii) };
        let hi = if ii == cells - 1 { 0.0 } else { w * face(ii + 1) };
        (lo, hi)
    };
    let project = |b: &[f64]| -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (r, v) in rc.iter().zip(b) {
            num += v * shape(*r) * r;
            den += shape(*r) * shape(*r) * r;
        }
        num / den
    };
    let mut b: Vec<f64> = rc.iter().map(|&r| shape(r)).collect();
    let p0 = project(&b);
    let (mut sub, mut diag, mut sup, mut rhs) = (vec![0.0; cells], vec![0.0; cells], vec![0.0; cells], vec![0.0; cells]);
    for _ in 0..steps {
        for ii in 0..cells {
            let (lo, hi) = coef(ii);
            let left = if ii == 0 { 0.0 } else { b[ii - 1] };
            let right = if ii == cells - 1 { 0.0 } else { b[ii + 1] };
            rhs[ii] = b[ii] + 0.5 * dt * (lo * (left - b[ii]) + hi * (right - b[ii]));
            sub[ii] = -0.5 * dt * lo;
            sup[ii] = -0.5 * dt * hi;
            diag[ii] = 1.0 + 0.5 * dt * (lo + hi);
        }
        // thomas elimination of the tridiagonal system
        for ii in 1..cells {
            let m = sub[ii] / diag[ii - 1];
            diag[ii] -= m * sup[ii - 1];
            rhs[ii] -= m * rhs[ii - 1];
        }
        b[cells - 1] = rhs[cells - 1] / diag[cells - 1];
        for ii in (0..cells - 1).rev() {
            b[ii] = (rhs[ii] - sup[ii] * b[ii + 1]) / diag[ii];
        }
    }
    project(&b) / p0
}

/// every arm the two decay tests read, run once
struct Arms {
    ideal: [Decay; 2],
    resistive: [Decay; 2],
    weak_field: Decay,
    reference: f64,
    reference_refined: f64,
}

fn arms() -> &'static Arms {
    static ARMS: std::sync::OnceLock<Arms> = std::sync::OnceLock::new();
    ARMS.get_or_init(|| {
        let arms = Arms {
            ideal: [evolve_decay(COARSE, 0.0, B0), evolve_decay(RESOLVED, 0.0, B0)],
            resistive: [evolve_decay(COARSE, ETA, B0), evolve_decay(RESOLVED, ETA, B0)],
            weak_field: evolve_decay(RESOLVED, ETA, 0.1 * B0),
            reference: radial_reference(2000, 1000),
            reference_refined: radial_reference(4000, 2000),
        };
        for (arm, d) in [
            ("ideal N=24", arms.ideal[0]),
            ("ideal N=48", arms.ideal[1]),
            ("resistive N=24", arms.resistive[0]),
            ("resistive N=48", arms.resistive[1]),
            ("resistive N=48, b0/10", arms.weak_field),
        ] {
            eprintln!("{arm}: retained {:.6}, transverse {:.3e} b0", d.retained, d.transverse);
        }
        eprintln!("reference {:.9}, refined {:.9}", arms.reference, arms.reference_refined);
        arms
    })
}

fn assert_transverse(arms: &Arms) {
    for (arm, d) in [
        ("ideal N=24", arms.ideal[0]),
        ("ideal N=48", arms.ideal[1]),
        ("resistive N=24", arms.resistive[0]),
        ("resistive N=48", arms.resistive[1]),
        ("resistive N=48 at b0/10", arms.weak_field),
    ] {
        assert!(
            d.transverse <= TRANSVERSE,
            "{arm}: a face field departed from z-uniformity by {:.3e} b0",
            d.transverse
        );
    }
}

#[test]
fn cyl_rz_resistivity_dominates_the_ideal_numerical_diffusion() {
    let a = arms();
    assert_transverse(a);
    let floor = [1.0 - a.ideal[0].retained, 1.0 - a.ideal[1].retained];
    // premise: the ideal loss is the resolved scheme's second-order numerical floor
    assert!(floor[1] > 0.0, "the ideal arm lost nothing at N = {RESOLVED}; the floor is unmeasured");
    assert!(
        floor[0] / floor[1] >= SECOND_ORDER_GAIN,
        "the ideal loss fell only {:.2}x from N = {COARSE} to {RESOLVED} ({:.4} -> {:.4}); it is not a \
         resolved numerical floor",
        floor[0] / floor[1],
        floor[0],
        floor[1]
    );
    let resistive_loss = 1.0 - a.resistive[1].retained;
    assert!(
        resistive_loss > DOMINANCE * floor[1],
        "cyl r-z resistivity did not dominate the numerical diffusion at N = {RESOLVED}: projection \
         loss {resistive_loss:.4} resistive against {:.4} ideal (ratio {:.2})",
        floor[1],
        resistive_loss / floor[1]
    );
}

#[test]
fn cyl_rz_resistive_decay_converges_to_the_radial_diffusion_reference() {
    let a = arms();
    assert_transverse(a);
    let error = [
        (a.resistive[0].retained - a.reference).abs(),
        (a.resistive[1].retained - a.reference).abs(),
    ];
    // the reference resolves the continuum far below the resolved fixture's error
    let refinement = (a.reference_refined - a.reference).abs();
    assert!(
        refinement <= REFERENCE_SHARE * error[1],
        "refining the reference moved it by {refinement:.3e}, against a resolved error of {:.3e}",
        error[1]
    );
    // the reference evolves magnetic diffusion alone; the flow the field drives changes the
    // resolved projection by far less than its error
    let backreaction = (a.weak_field.retained - a.resistive[1].retained).abs();
    assert!(
        backreaction <= REFERENCE_SHARE * error[1],
        "a tenfold weaker field changed the resolved projection by {backreaction:.3e}, against an \
         error of {:.3e}",
        error[1]
    );
    assert!(
        error[0] / error[1] >= SECOND_ORDER_GAIN,
        "the resistive projection error fell only {:.2}x from N = {COARSE} to {RESOLVED} ({:.4e} -> \
         {:.4e} against the reference {:.6})",
        error[0] / error[1],
        error[0],
        error[1],
        a.reference
    );
}
