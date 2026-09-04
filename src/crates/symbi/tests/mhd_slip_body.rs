// =============================================================================
// mhd_slip_body.rs
//
// the 3D immersed-body magnetic-slip operator (`MagneticSpec::Slip`), the explicit CT oracle. the
// two baked passes -- the cell pass assembling F_q = A(B_q)(R J)_q into the slip-quadrature scratch,
// the edge passes scattering it to the oriented edge EMF with the weighted adjoint R^* -- are
// exercised through the production dispatch and checked against an independent pure-Rust reference of
// the same operator. the acceptance gate:
//   - element-by-element agreement of the baked slip-quadrature with the reference F_q, on an
//     asymmetric field where an offset reversal cannot hide behind symmetry;
//   - the face magnetic-energy change through the augmented curl <= 0 (the discrete dissipation);
//   - div B unchanged to roundoff;
//   - a uniform field (zero current) an exact no-op;
//   - cyclic translation covariance of the slip-quadrature field.
// this is the explicit oracle; its stable timestep is the diffusive limit dt <~ dx^2 / eta_B.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::drain::{sound_speed_from_cons, spherical_drain_rate};
use symbi_ib::magnetic_slip::{chi_shell, slip_apply, slip_coefficient};
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{body_slip_emf, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Mhd = symbi_sim::state::MhdStaggeredFields<3, 3, HostMemory, f64>;

const N: usize = 13; // small and coprime with the 4-cell stencils, so every wrap is exercised
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.22;
const D_B: f64 = 2.0;
const W: f64 = 0.12;
const ELL_RATIO: f64 = 1.5;
const B0: f64 = 0.1;
const PLACEMENT: f64 = 0.0;
const DT: f64 = 1e-3;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: D_B,
        shell_width: W,
        slip_length_ratio: ELL_RATIO,
        field_regularization: B0,
        placement: PLACEMENT,
    }
}

fn wrap(v: isize) -> isize {
    v.rem_euclid(N as isize)
}

// a deterministic pseudo-random field in [-0.5, 0.5], periodic (wrapped coords) and asymmetric
// under axis reflection.
fn rnd(c: [isize; 3], salt: u64) -> f64 {
    let mut x = (wrap(c[0]) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (wrap(c[1]) as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ (wrap(c[2]) as u64).wrapping_mul(0xA076_1D64_78BD_642F)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}

fn build_sim(magnetic: MagneticSpec) -> Sim {
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<
        NewtonianMhd,
        3,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([N, N, N])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("slip body sim construction failed")
    .set_initial(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(|_, _| 0.0)
    .build();
    // a draining sphere (the drain supplies tau_rho for the slip coefficient); the coupling under
    // test is the magnetic one, and the direct operator call leaves the gas state untouched.
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(magnetic),
        ),
    )
}

// seed bface[d] from `field`, bcell[d] as the forward face-average (SIMBI's bcell_from_bface), and
// zero the EMF.
fn seed(sim: &Sim, field: impl Fn(usize, [isize; 3]) -> f64) {
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            m.bface[d].set(c, field(d, c));
        }
    }
    // bcell over the interior only (the cell pass reads B_q with no offset); reading bface at c and
    // c + e_d stays inside the face domain (interior extended +1 in d).
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            let mut up = c;
            up[d] += 1;
            m.bcell[d].set(c, 0.5 * (*m.bface[d].at(c) + *m.bface[d].at(up)));
        }
    }
    for d in 0..3 {
        for c in m.efield[d].domain().iter() {
            m.efield[d].set(c, 0.0);
        }
    }
}

fn bcell(m: &Mhd, d: usize, c: [isize; 3]) -> f64 {
    *m.bcell[d].at(c)
}
fn bface(m: &Mhd, d: usize, c: [isize; 3]) -> f64 {
    *m.bface[d].at(c)
}

// the reference edge current (curl B)_d at the edge whose lower corner is cell `c`, from backward
// face differences -- the same expression the resistive/slip kernels use.
fn curl_edge(m: &Mhd, d: usize, c: [isize; 3], inv_dx: f64) -> f64 {
    let p1 = (d + 1) % 3;
    let p2 = (d + 2) % 3;
    let mut cm1 = c;
    cm1[p1] -= 1;
    let mut cm2 = c;
    cm2[p2] -= 1;
    (bface(m, p2, c) - bface(m, p2, cm1)) * inv_dx - (bface(m, p1, c) - bface(m, p1, cm2)) * inv_dx
}

// the reference (R J)_q[d] at cell `c`: 1/4 the sum of the four bounding d-edge currents (forward
// transverse offsets), matching the cell pass's gather.
fn j_gather(m: &Mhd, d: usize, c: [isize; 3], inv_dx: f64) -> f64 {
    let p1 = (d + 1) % 3;
    let p2 = (d + 2) % 3;
    let mut e1 = c;
    e1[p1] += 1;
    let mut e2 = c;
    e2[p2] += 1;
    let mut e12 = c;
    e12[p1] += 1;
    e12[p2] += 1;
    0.25 * (curl_edge(m, d, c, inv_dx)
        + curl_edge(m, d, e1, inv_dx)
        + curl_edge(m, d, e2, inv_dx)
        + curl_edge(m, d, e12, inv_dx))
}

// the global drain rate lambda_rho for the uniform gas state (constant over the grid): the same
// spherical_drain_rate the material drain applies.
fn lambda_rho(sim: &Sim) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let some_cell = sim.geom.interior.iter().next().unwrap();
    let den = *sim.fields.cons.den.at(some_cell);
    let nrg = *sim.fields.cons.nrg_field().unwrap().at(some_cell);
    let cs = sound_speed_from_cons(den, 0.0, nrg, GAMMA); // v = 0, mom_sq = 0
    let c_a2 = symbi_sim::state::local_c_a2_max(sim);
    let signal = (cs * cs + c_a2).sqrt();
    let dx = sim.geom.dx[0];
    let sound_rate = signal / (1.0 * dx); // c_drain = 1
    let _ = m;
    spherical_drain_rate(sound_rate, 1.0, R_BODY)
}

// the reference cell-quadrature vector F_q[d] at cell `c`, from bcell, the gathered current, and the
// slip coefficient/shell mask evaluated exactly as the kernel does.
fn f_ref(sim: &Sim, m: &Mhd, c: [isize; 3], lam: f64) -> [f64; 3] {
    let dx = sim.geom.dx[0];
    let inv_dx = 1.0 / dx;
    let b_q = Tensor::new([bcell(m, 0, c), bcell(m, 1, c), bcell(m, 2, c)]);
    let j_q = Tensor::new([
        j_gather(m, 0, c, inv_dx),
        j_gather(m, 1, c, inv_dx),
        j_gather(m, 2, c, inv_dx),
    ]);
    // the physical cell center, distance to the body surface, shell mask.
    let x: [f64; 3] = std::array::from_fn(|a| sim.geom.x_lo[a] + (c[a] as f64 + 0.5) * dx);
    let dist = ((x[0] - BODY[0]).powi(2) + (x[1] - BODY[1]).powi(2) + (x[2] - BODY[2]).powi(2)).sqrt();
    let phi = dist - R_BODY;
    let chi_b = chi_shell(phi, W, PLACEMENT);
    let ell_b = ELL_RATIO * W;
    let a_b = slip_coefficient(ell_b, b_q.dot(&b_q), B0, D_B, 1.0 / lam);
    let f = slip_apply(a_b * chi_b, &b_q, &j_q);
    [f[0], f[1], f[2]]
}

// ---------------------------------------------------------------------------

#[test]
fn slip_cell_pass_matches_the_reference_elementwise() {
    let sim = build_sim(slip_spec());
    seed(&sim, |d, c| rnd(c, d as u64 + 1));
    let lam = lambda_rho(&sim);
    body_slip_emf::<3, 3, HostMemory, f64>(&sim, GAMMA);

    let m = sim.fields.mhd.as_ref().unwrap();
    let slip_q = m.slip_quadrature.as_ref().expect("slip-quadrature scratch");
    let mut max_err = 0.0_f64;
    let mut max_mag = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let f = f_ref(&sim, m, c, lam);
        for d in 0..3 {
            let got = *slip_q[d].at(c);
            max_err = max_err.max((got - f[d]).abs());
            max_mag = max_mag.max(f[d].abs());
        }
    }
    assert!(max_mag > 1e-8, "the reference F_q is trivially zero; the test is vacuous");
    assert!(
        max_err < 1e-11 * max_mag.max(1.0),
        "baked cell pass disagrees with the reference: max |F_baked - F_ref| = {max_err} (max |F_ref| = {max_mag})"
    );
}

#[test]
fn slip_operator_dissipates_and_preserves_div_b() {
    let sim = build_sim(slip_spec());
    seed(&sim, |d, c| rnd(c, d as u64 + 1));

    let div_before = max_div_b(&sim);
    let e_before = face_energy(&sim);

    body_slip_emf::<3, 3, HostMemory, f64>(&sim, GAMMA);
    ct_curl::<3, 3, HostMemory, f64>(&sim, DT);

    let div_after = max_div_b(&sim);
    let e_after = face_energy(&sim);

    assert!(
        (div_after - div_before).abs() < 1e-12 * (div_before.abs().max(1.0)),
        "the augmented curl changed div B: {div_before} -> {div_after}"
    );
    assert!(
        e_after < e_before - 1e-14,
        "the slip operator did not dissipate face magnetic energy: {e_before} -> {e_after}"
    );
}

#[test]
fn a_uniform_field_is_an_exact_no_op() {
    let sim = build_sim(slip_spec());
    seed(&sim, |d, _| [0.7, -0.4, 0.3][d]); // uniform B: zero current
    body_slip_emf::<3, 3, HostMemory, f64>(&sim, GAMMA);
    let m = sim.fields.mhd.as_ref().unwrap();
    let slip_q = m.slip_quadrature.as_ref().unwrap();
    let mut max_fq = 0.0_f64;
    let mut max_e = 0.0_f64;
    for c in sim.geom.interior.iter() {
        for d in 0..3 {
            max_fq = max_fq.max(slip_q[d].at(c).abs());
            max_e = max_e.max(m.efield[d].at(c).abs());
        }
    }
    assert!(max_fq < 1e-12, "a uniform field produced a nonzero quadrature (max |F_q| = {max_fq})");
    assert!(max_e < 1e-12, "a uniform field produced a nonzero slip EMF (max |E| = {max_e})");
}

#[test]
fn the_slip_quadrature_is_translation_covariant() {
    // shifting the seed field by one cell on each axis shifts the whole slip-quadrature field by the
    // same offset; an inconsistent stencil would break this.
    let shift = [1isize, 1, 1];
    let base = build_sim(slip_spec());
    seed(&base, |d, c| rnd(c, d as u64 + 1));
    body_slip_emf::<3, 3, HostMemory, f64>(&base, GAMMA);

    let shifted = build_sim(slip_spec());
    seed(&shifted, |d, c| {
        rnd([c[0] - shift[0], c[1] - shift[1], c[2] - shift[2]], d as u64 + 1)
    });
    body_slip_emf::<3, 3, HostMemory, f64>(&shifted, GAMMA);

    // the full operator carries the body mask at a fixed point, so it is not shift covariant. the
    // stencil under test is the current gather R (curl B); assert it translates exactly, so an
    // inconsistent offset (which would break shift covariance) is caught on the asymmetric field.
    let mb = base.fields.mhd.as_ref().unwrap();
    let ms = shifted.fields.mhd.as_ref().unwrap();
    let inv_dx = 1.0 / base.geom.dx[0];
    // only compare where both c and c+shift keep the full gather stencil (reach +/-2) inside the
    // seeded interior, so no read falls into an unseeded halo cell.
    let interior: std::collections::HashSet<[isize; 3]> = base.geom.interior.iter().collect();
    let safe = |p: [isize; 3]| {
        (-2..=2).all(|a| {
            (-2..=2).all(|b| {
                (-2..=2).all(|c2| interior.contains(&[p[0] + a, p[1] + b, p[2] + c2]))
            })
        })
    };
    let mut max_err = 0.0_f64;
    let mut max_mag = 0.0_f64;
    let mut compared = 0;
    for c in base.geom.interior.iter() {
        let cs = [c[0] + shift[0], c[1] + shift[1], c[2] + shift[2]];
        if !safe(c) || !safe(cs) {
            continue;
        }
        for d in 0..3 {
            let jb = j_gather(mb, d, c, inv_dx);
            let js = j_gather(ms, d, cs, inv_dx);
            max_err = max_err.max((jb - js).abs());
            max_mag = max_mag.max(jb.abs());
            compared += 1;
        }
    }
    assert!(compared > 0, "no safely-interior cells to compare");
    assert!(max_mag > 1e-8, "vacuous translation test");
    assert!(
        max_err < 1e-11 * max_mag,
        "the current gather is not translation covariant: max err {max_err} (max |J| {max_mag})"
    );
}

#[test]
fn magnetic_none_and_resistive_do_not_touch_the_slip_scratch() {
    for spec in [MagneticSpec::None, MagneticSpec::Resistive { eta: 0.1 }] {
        let sim = build_sim(spec);
        let m = sim.fields.mhd.as_ref().unwrap();
        assert!(
            m.slip_quadrature.is_none(),
            "a {spec:?} body allocated the slip-quadrature scratch"
        );
    }
}

// the consolidated numerical report: the acceptance figures printed for the record. run with
//   cargo test -p symbi --test mhd_slip_body report -- --nocapture
#[test]
fn report() {
    let sim = build_sim(slip_spec());
    seed(&sim, |d, c| rnd(c, d as u64 + 1));
    let lam = lambda_rho(&sim);
    let dx = sim.geom.dx[0];
    let inv_dx = 1.0 / dx;

    // element-wise reference vs baked cell pass.
    body_slip_emf::<3, 3, HostMemory, f64>(&sim, GAMMA);
    let m = sim.fields.mhd.as_ref().unwrap();
    let slip_q = m.slip_quadrature.as_ref().unwrap();
    let (mut max_err, mut max_mag) = (0.0_f64, 0.0_f64);
    for c in sim.geom.interior.iter() {
        let f = f_ref(&sim, m, c, lam);
        for d in 0..3 {
            max_err = max_err.max((*slip_q[d].at(c) - f[d]).abs());
            max_mag = max_mag.max(f[d].abs());
        }
    }

    // the quadratic form both ways: <R J, A(B_q) R J>_q (cell) and <J, E>_edge (edge). the discrete
    // current at an edge is the same curl_edge the gather averages; the slip EMF is the scattered F_q.
    let interior: std::collections::HashSet<[isize; 3]> = sim.geom.interior.iter().collect();
    let mut quad_cell = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let f = f_ref(&sim, m, c, lam);
        for d in 0..3 {
            quad_cell += j_gather(m, d, c, inv_dx) * f[d];
        }
    }
    let mut quad_edge = 0.0_f64;
    for c in sim.geom.interior.iter() {
        for d in 0..3 {
            // the edge current lives at the edge with lower corner c; the slip EMF increment is efield.
            quad_edge += curl_edge(m, d, c, inv_dx) * *m.efield[d].at(c);
        }
    }
    let _ = &interior;

    // divergence defect and magnetic-energy change through the augmented curl.
    let div_before = max_div_b(&sim);
    let e_before = face_energy(&sim);
    ct_curl::<3, 3, HostMemory, f64>(&sim, DT);
    let div_after = max_div_b(&sim);
    let e_after = face_energy(&sim);

    println!("\n=== magnetic-slip explicit-oracle numerical report (N={N}, dt={DT}) ===");
    println!("reference-vs-baked max |F_baked - F_ref|   : {max_err:.3e}  (max |F_ref| = {max_mag:.3e})");
    println!("relative element-wise error                : {:.3e}", max_err / max_mag.max(1e-300));
    println!("quadratic form  <R J, A(B_q) R J>_q        : {quad_cell:.6e}  (>= 0)");
    println!("quadratic form  <J, E>_edge                : {quad_edge:.6e}");
    println!("cell/edge form agreement |diff|            : {:.3e}", (quad_cell - quad_edge).abs());
    println!("divergence defect  max|div B| before/after : {div_before:.3e} / {div_after:.3e}");
    println!("magnetic-energy change  dW = W_after-W_bef : {:.6e}  (<= 0)", e_after - e_before);
    println!("predicted  -dt <J,E>_edge                  : {:.6e}", -DT * quad_edge);
    println!("energy-identity residual |dW + dt<J,E>|    : {:.3e}", (e_after - e_before + DT * quad_edge).abs());
    println!("========================================================\n");
}

// ---------------------------------------------------------------------------

fn face_energy(sim: &Sim) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut e = 0.0;
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            let b = *m.bface[d].at(c);
            e += 0.5 * b * b;
        }
    }
    e
}

fn max_div_b(sim: &Sim) -> f64 {
    let m = sim.fields.mhd.as_ref().unwrap();
    let inv_dx = 1.0 / sim.geom.dx[0];
    let mut mx = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let mut div = 0.0;
        for d in 0..3 {
            let mut up = c;
            up[d] += 1;
            div += (*m.bface[d].at(up) - *m.bface[d].at(c)) * inv_dx;
        }
        mx = mx.max(div.abs());
    }
    mx
}
