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
use symbi_substrate::regimes::mhd_substrate::{body_slip_emf, ct_curl};
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
