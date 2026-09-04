// =============================================================================
// mhd_slip_implicit.rs
//
// the frozen-coefficient magnetic-slip operator L = C R* A(B*) R C* used by the implicit midpoint
// solve, and the symmetry/positivity pins that make its system operator (I + dt/2 L) SPD and hence
// conjugate-gradient compatible. L is applied matrix-free through the exact production chain -- the
// two-pass slip operator (R* A R applied to the current) followed by the CT curl -- with bcell held
// at the predictor B* so A(B*), the shell mask, and the coefficient are frozen and L is linear in
// the face field it acts on. reusing the seam-closed production chain (not a second stencil) is what
// carries the roundoff-exact adjoint L = L^* into the solver.
//
// the pins, on random periodic face fields:
//   <x, L y>_B = <L x, y>_B                      (symmetry, from C^* = C adjoint, R^* = R adjoint,
//                                                 A symmetric, all composed)
//   <x, L x>_B >= 0                              (positive semidefiniteness, the dissipation)
//   <x, (I + dt/2 L) x>_B = ||x||^2 + dt/2 <x,Lx>  (the system operator is positive definite)
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{
    body_slip_emf, ct_curl, magnetic_slip_apply_operator, magnetic_slip_solve,
};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 11;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 3] = [0.5, 0.5, 0.5];
const R_BODY: f64 = 0.22;
const DT: f64 = 1e-2;

fn slip_spec() -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width: 0.12,
        slip_length_ratio: 1.5,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

fn wrap(v: isize) -> isize {
    v.rem_euclid(N as isize)
}
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

fn build_sim() -> Sim {
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
    .expect("slip implicit sim construction failed")
    .set_initial(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(|_, _| 0.0)
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::black_hole(0, Tensor::new(BODY), Tensor::zeros(), 1.0, R_BODY, 0.05, 1.0, 1.0, R_BODY)
                .with_surface(SurfaceSpec::Drain)
                .with_magnetic(slip_spec()),
        ),
    )
}

// a face field over the full bface domain (halo included, so the periodic stencils read the wrapped
// image), keyed by (component, coord).
type Face = std::collections::HashMap<(usize, [isize; 3]), f64>;

// freeze the predictor B* into bcell (the dyad/coefficient state L holds fixed).
fn freeze_predictor(sim: &Sim, bstar: impl Fn(usize, [isize; 3]) -> f64) {
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            m.bface[d].set(c, bstar(d, c));
        }
        for c in sim.geom.interior.iter() {
            let mut up = c;
            up[d] += 1;
            m.bcell[d].set(c, 0.5 * (*m.bface[d].at(c) + *m.bface[d].at(up)));
        }
    }
}

fn random_face(salt: u64) -> impl Fn(usize, [isize; 3]) -> f64 {
    move |d, c| rnd(c, salt + d as u64 * 7)
}

fn face_of(f: &impl Fn(usize, [isize; 3]) -> f64, sim: &Sim) -> Face {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut out = Face::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            out.insert((d, c), f(d, c));
        }
    }
    out
}

// L x = C R* A(B*) R C* x through the production chain: set bface = x, run the two-pass slip operator
// (efield = R* A(B*) R C* x with A frozen at bcell = B*), then curl into L x = C(efield). the curl is
// recovered as x - (x - dt C E)/dt at dt = 1, so no separate stencil is written.
fn apply_l(sim: &Sim, x: &Face) -> Face {
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            m.bface[d].set(c, x[&(d, c)]);
        }
        for c in m.efield[d].domain().iter() {
            m.efield[d].set(c, 0.0);
        }
    }
    body_slip_emf::<3, 3, HostMemory, f64>(sim, GAMMA);
    ct_curl::<3, 3, HostMemory, f64>(sim, 1.0); // bface <- x - C E
    let mut out = Face::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            out.insert((d, c), x[&(d, c)] - *m.bface[d].at(c)); // = C E = L x
        }
    }
    out
}

// the face inner product over the interior (cartesian unit weights).
fn dot(sim: &Sim, a: &Face, b: &Face) -> f64 {
    let mut s = 0.0;
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            s += a[&(d, c)] * b[&(d, c)];
        }
    }
    s
}

// --- face-field linear algebra over the full (halo-inclusive) domain ---
fn keys(sim: &Sim) -> Vec<(usize, [isize; 3])> {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut v = Vec::new();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            v.push((d, c));
        }
    }
    v
}
fn axpy(sim: &Sim, a: f64, x: &Face, y: &Face) -> Face {
    keys(sim).into_iter().map(|k| (k, a * x[&k] + y[&k])).collect()
}
fn scale(sim: &Sim, a: f64, x: &Face) -> Face {
    keys(sim).into_iter().map(|k| (k, a * x[&k])).collect()
}

fn set_bface(sim: &Sim, x: &Face) {
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            m.bface[d].set(c, x[&(d, c)]);
        }
    }
}
fn get_bface(sim: &Sim) -> Face {
    let m = sim.fields.mhd.as_ref().unwrap();
    keys(sim).into_iter().map(|k| (k, *m.bface[k.0].at(k.1))).collect()
}
// freeze bcell = interp(bface) over the interior: the dyad/coefficient state.
fn interp_bcell(sim: &Sim) {
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            let mut up = c;
            up[d] += 1;
            m.bcell[d].set(c, 0.5 * (*m.bface[d].at(c) + *m.bface[d].at(up)));
        }
    }
}

// the nonlinear explicit oracle step x - dt L(x) x, with A(x) evaluated at bcell = interp(x). the
// predictor uses dt/2; it is a CT curl update, so it preserves div B.
fn explicit_step(sim: &Sim, x: &Face, dt: f64) -> Face {
    set_bface(sim, x);
    interp_bcell(sim);
    let m = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in m.efield[d].domain().iter() {
            m.efield[d].set(c, 0.0);
        }
    }
    body_slip_emf::<3, 3, HostMemory, f64>(sim, GAMMA);
    ct_curl::<3, 3, HostMemory, f64>(sim, dt);
    get_bface(sim)
}

// the frozen system operator (I + dt/2 L*) applied to x; L* uses the currently frozen bcell = B*.
fn apply_sys(sim: &Sim, x: &Face, dt: f64) -> Face {
    axpy(sim, 0.5 * dt, &apply_l(sim, x), x)
}

// mirrors the production MagneticSlipSolveReceipt shape; some fields are recorded for the receipt
// but not asserted on in every test.
#[allow(dead_code)]
struct Receipt {
    iterations: usize,
    initial_residual_norm: f64,
    final_residual_norm: f64,
    requested_tolerance: f64,
    converged: bool,
}

// matrix-free conjugate gradient for the SPD midpoint system (I + dt/2 L*) X = rhs. bcell must
// already hold the frozen predictor B*.
fn cg(sim: &Sim, rhs: &Face, dt: f64, tol: f64, max_iter: usize) -> (Face, Receipt) {
    let dotf = |a: &Face, b: &Face| dot(sim, a, b);
    let mut x: Face = keys(sim).into_iter().map(|k| (k, 0.0)).collect();
    let mut r = rhs.clone(); // r = rhs - sys(0) = rhs
    let r0 = dotf(&r, &r).sqrt();
    let mut p = r.clone();
    let mut rs = dotf(&r, &r);
    let mut iterations = 0;
    let target = tol * r0.max(1e-300);
    let mut final_res = r0;
    for it in 0..max_iter {
        let ap = apply_sys(sim, &p, dt);
        let alpha = rs / dotf(&p, &ap);
        x = axpy(sim, alpha, &p, &x);
        r = axpy(sim, -alpha, &ap, &r);
        let rs_new = dotf(&r, &r);
        final_res = rs_new.sqrt();
        iterations = it + 1;
        if final_res <= target {
            break;
        }
        let beta = rs_new / rs;
        p = axpy(sim, beta, &p, &r);
        rs = rs_new;
    }
    (
        x,
        Receipt {
            iterations,
            initial_residual_norm: r0,
            final_residual_norm: final_res,
            requested_tolerance: tol,
            converged: final_res <= target,
        },
    )
}

// one production-shaped midpoint step: second-order explicit predictor B* = B0 - dt/2 L(B0) B0,
// freeze A* = A(B*), then solve (I + dt/2 L*) B1 = (I - dt/2 L*) B0 by CG.
fn midpoint_step(sim: &Sim, b0: &Face, dt: f64, tol: f64) -> (Face, Face, Receipt) {
    let bstar = explicit_step(sim, b0, 0.5 * dt); // the predictor (CT update, div-B preserving)
    set_bface(sim, &bstar);
    interp_bcell(sim); // freeze A* = A(B*)
    // rhs = (I - dt/2 L*) B0.
    let lb0 = apply_l(sim, b0);
    let rhs = axpy(sim, -0.5 * dt, &lb0, b0);
    let (b1, receipt) = cg(sim, &rhs, dt, tol, 500);
    (b1, bstar, receipt)
}

fn face_energy_of(sim: &Sim, x: &Face) -> f64 {
    let mut e = 0.0;
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            let b = x[&(d, c)];
            e += 0.5 * b * b;
        }
    }
    e
}

#[test]
fn the_midpoint_solve_satisfies_the_energy_theorem() {
    // W^1 - W^0 = -dt <B^{1/2}, L* B^{1/2}> = -dt <R J^{1/2}, A* R J^{1/2}>_q, exact to the converged
    // linear residual plus roundoff (B^{1/2} = (B1+B0)/2, A* frozen at the predictor).
    let sim = build_sim();
    let b0 = face_of(&random_face(11), &sim);
    let dt = DT;
    let (b1, _bstar, receipt) = midpoint_step(&sim, &b0, dt, 1e-13);
    assert!(receipt.converged, "CG did not converge: {} iters, res {:.3e}", receipt.iterations, receipt.final_residual_norm);

    // bcell is frozen at B* from midpoint_step; L* = the frozen operator.
    let bhalf = scale(&sim, 0.5, &axpy(&sim, 1.0, &b1, &b0));
    let quad = dot(&sim, &bhalf, &apply_l(&sim, &bhalf)); // <R J^{1/2}, A* R J^{1/2}>_q
    let dw = face_energy_of(&sim, &b1) - face_energy_of(&sim, &b0);
    let predicted = -dt * quad;
    assert!(quad >= -1e-10, "midpoint dissipation is negative: {quad:.3e}");
    assert!(
        (dw - predicted).abs() < 1e-9 * dw.abs().max(predicted.abs()).max(1.0),
        "midpoint energy theorem fails: dW = {dw:.9e}, -dt<RJ,A*RJ>_q = {predicted:.9e}, residual {:.3e}",
        (dw - predicted).abs()
    );
    println!(
        "\nmidpoint energy theorem:  dW = {dw:.9e}  -dt<RJ,A*RJ> = {predicted:.9e}  residual {:.3e}  (CG {} iters, res {:.2e})\n",
        (dw - predicted).abs(),
        receipt.iterations,
        receipt.final_residual_norm
    );
}

#[test]
fn the_midpoint_method_is_second_order_in_time() {
    // evolve a fixed field over a fixed horizon with step dt and dt/2, against a fine dt/8 reference.
    // the nonlinear predictor gives second-order temporal accuracy, so the error quarters.
    let sim = build_sim();
    let b0 = face_of(&random_face(11), &sim);
    let horizon = 8.0 * DT;
    let evolve = |steps: usize| -> Face {
        let dt = horizon / steps as f64;
        let mut b = b0.clone();
        for _ in 0..steps {
            let (b1, _s, rec) = midpoint_step(&sim, &b, dt, 1e-13);
            assert!(rec.converged, "CG diverged in the order study");
            b = b1;
        }
        b
    };
    let l2 = |a: &Face, b: &Face| -> f64 {
        let mut s = 0.0;
        for d in 0..3 {
            for c in sim.geom.interior.iter() {
                let e = a[&(d, c)] - b[&(d, c)];
                s += e * e;
            }
        }
        s.sqrt()
    };
    let reference = evolve(64);
    let coarse = evolve(8);
    let fine = evolve(16);
    let e_coarse = l2(&coarse, &reference);
    let e_fine = l2(&fine, &reference);
    let ratio = e_coarse / e_fine;
    println!(
        "\ntemporal order:  E(dt) = {e_coarse:.3e}  E(dt/2) = {e_fine:.3e}  ratio = {ratio:.3}  (second order -> ~4)\n"
    );
    assert!(e_coarse > 1e-12, "vacuous order test (error {e_coarse})");
    assert!(
        ratio > 3.4,
        "the midpoint method is not second order: E(dt)/E(dt/2) = {ratio:.3} (expected ~4)"
    );
}

// M_face(x) = 1/2 sum_interior |x|^2 over the workspace face field.
fn face_energy_ws(sim: &Sim, f: &symbi_sim::state::BfaceFields<3, HostMemory, f64>) -> f64 {
    let mut e = 0.0;
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            let b = *f.b[d].at(c);
            e += 0.5 * b * b;
        }
    }
    e
}

#[test]
fn the_production_solve_is_transactional_and_proves_the_face_energy_theorem() {
    let sim = build_sim();
    let b0 = face_of(&random_face(11), &sim);
    set_bface(&sim, &b0);
    interp_bcell(&sim);

    let bface_before = get_bface(&sim);
    let dt = DT;
    let receipt = magnetic_slip_solve::<3, 3, HostMemory, f64>(&sim, dt, GAMMA, 1e-13, 500);
    assert!(
        receipt.converged,
        "production CG did not converge: {} iters, res {:.3e}",
        receipt.iterations, receipt.final_residual_norm
    );

    // transactional: production bface is restored to the substep input (the solve mutates neither it
    // nor bcell nor cons.nrg; the candidate lives in the workspace).
    let bface_after = get_bface(&sim);
    let mut max_drift = 0.0_f64;
    for k in keys(&sim) {
        max_drift = max_drift.max((bface_before[&k] - bface_after[&k]).abs());
    }
    assert!(max_drift < 1e-12, "the solve mutated production bface: max drift {max_drift:.3e}");

    // the face-norm energy theorem: dM_face = -dt <B^{1/2}, L* B^{1/2}> = -Q_h, Q_h >= 0.
    let ws = sim.fields.mhd.as_ref().unwrap().magnetic_slip.as_ref().unwrap();
    let dm_face = face_energy_ws(&sim, &ws.candidate) - face_energy_ws(&sim, &ws.input);

    // B^{1/2} = (candidate + input)/2 into ws.rhs over the full domain (periodic halo carried), freeze
    // bcell = B*, then L* B^{1/2} into ws.operator_direction.
    for d in 0..3 {
        for c in ws.rhs.b[d].domain().iter() {
            ws.rhs.b[d].set(c, 0.5 * (*ws.candidate.b[d].at(c) + *ws.input.b[d].at(c)));
        }
    }
    let mhd = sim.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            mhd.bcell[d].set(c, *ws.frozen_bcell.b[d].at(c));
        }
    }
    magnetic_slip_apply_operator::<3, 3, HostMemory, f64>(
        &sim, GAMMA, true, &ws.rhs, &ws.operator_direction,
    );
    let mut quad = 0.0;
    for d in 0..3 {
        for c in sim.geom.interior.iter() {
            quad += *ws.rhs.b[d].at(c) * *ws.operator_direction.b[d].at(c);
        }
    }
    let q_h = dt * quad; // the dissipated face-Hodge magnetic energy, >= 0
    assert!(q_h >= -1e-10, "face dissipation is negative: Q_h = {q_h:.3e}");
    assert!(
        (q_h + dm_face).abs() < 1e-8 * q_h.abs().max(dm_face.abs()).max(1.0),
        "face energy theorem fails: Q_h = {q_h:.9e}, -dM_face = {:.9e}, residual {:.3e}",
        -dm_face,
        (q_h + dm_face).abs()
    );
    println!(
        "\nproduction solve:  CG {} iters (res {:.2e})  Q_h = {q_h:.9e}  -dM_face = {:.9e}  residual {:.3e}  bface-drift {max_drift:.2e}\n",
        receipt.iterations, receipt.final_residual_norm, -dm_face, (q_h + dm_face).abs()
    );
}

#[test]
fn the_frozen_operator_l_is_symmetric() {
    let sim = build_sim();
    freeze_predictor(&sim, random_face(101));
    let x = face_of(&random_face(1), &sim);
    let y = face_of(&random_face(2), &sim);
    let lx = apply_l(&sim, &x);
    let ly = apply_l(&sim, &y);
    let xly = dot(&sim, &x, &ly);
    let lxy = dot(&sim, &lx, &y);
    let scale = xly.abs().max(lxy.abs()).max(1.0);
    println!(
        "\nL symmetry:  <x,Ly> = {xly:.9e}  <Lx,y> = {lxy:.9e}  |diff| = {:.3e}  (rel {:.2e})\n",
        (xly - lxy).abs(),
        (xly - lxy).abs() / scale
    );
    assert!(scale > 1e-3, "vacuous symmetry test (scale {scale})");
    assert!(
        (xly - lxy).abs() < 1e-9 * scale,
        "L is not symmetric: <x, L y> = {xly:.9e} vs <L x, y> = {lxy:.9e} (rel {:.2e})",
        (xly - lxy).abs() / scale
    );
}

#[test]
fn the_frozen_operator_l_is_positive_semidefinite() {
    let sim = build_sim();
    freeze_predictor(&sim, random_face(101));
    let mut min_form = f64::INFINITY;
    let mut any_positive = false;
    for salt in 0..6u64 {
        let x = face_of(&random_face(500 + salt), &sim);
        let lx = apply_l(&sim, &x);
        let form = dot(&sim, &x, &lx);
        min_form = min_form.min(form);
        any_positive |= form > 1e-6;
    }
    assert!(any_positive, "L is trivially zero on every probe; the test is vacuous");
    assert!(min_form >= -1e-9, "L is not positive semidefinite: min <x, L x> = {min_form:.3e}");
}

#[test]
fn the_system_operator_is_positive_definite() {
    // <x, (I + dt/2 L) x> = ||x||^2 + dt/2 <x, L x> >= ||x||^2 > 0, the SPD property CG needs. the
    // identity is checked exactly and the value is bounded below by ||x||^2.
    let sim = build_sim();
    freeze_predictor(&sim, random_face(101));
    for salt in 0..5u64 {
        let x = face_of(&random_face(900 + salt), &sim);
        let lx = apply_l(&sim, &x);
        let nrm2 = dot(&sim, &x, &x);
        let xlx = dot(&sim, &x, &lx);
        let sysform = nrm2 + 0.5 * DT * xlx;
        // the assembled system form equals ||x||^2 + dt/2 <x,Lx> by construction; pin it and its
        // strict positivity.
        assert!((sysform - (nrm2 + 0.5 * DT * xlx)).abs() < 1e-30);
        assert!(nrm2 > 1e-6, "vacuous probe");
        assert!(
            sysform >= nrm2 - 1e-9 * nrm2 && sysform > 0.0,
            "(I + dt/2 L) is not positive definite: <x,(I+dt/2 L)x> = {sysform:.6e} < ||x||^2 = {nrm2:.6e}"
        );
    }
}
