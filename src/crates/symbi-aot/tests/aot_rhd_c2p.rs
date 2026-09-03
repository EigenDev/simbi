// =============================================================================
// aot_rhd_c2p.rs
//
// numerical validation of the build-time-generated RHD cons->prim kernel — the
// first iterative substrate kernel (a fixed-bound masked Newton-Raphson for the
// relativistic pressure, `operators::iterate`, lowered + emitted to compiled Rust
// via the DAG-preserving lowering). this is the proof the deep
// iterate produces correct numbers at run time.
//
// two independent checks on the same compiled kernel `rhd_c2p_1d`:
//   - round-trip: pick analytic primitives (rho, v, p), forward-map to the
//     conserved (D, S, tau) via the standard RHD relations, run c2p, and assert
//     it recovers the originals. this is the physical ground truth.
//   - reference: an independent Rust `rhd_to_primitive` (a do-while Newton
//     iteration run to convergence) on the same conserved
//     states; the compiled 20-step masked unroll must match it.
//
// generated signature (OUT_DIR/rhd_c2p_generated.rs):
//   rhd_c2p_1d(cons_den, cons_mom_0, cons_nrg : &[f64],         // inputs
//               prim_rho, prim_vel_0, prim_pre : &mut [f64],     // outputs
//               grid_size_0, dom_lo_0, buf_lo_0_0..buf_lo_5_0 : i32, gamma: f64)
// =============================================================================

use symbi_aot::NamedKernel;

// shim binding the emitted c2p by field name (NamedKernel) — order-independent,
// loud + named on manifest drift. every buffer here is 1D (lo = 0).
#[allow(non_snake_case, clippy::too_many_arguments)]
fn rhd_c2p_1d(
    cons_den: &[f64],
    cons_mom: &[f64],
    cons_nrg: &[f64],
    prim_rho: &mut [f64],
    prim_vel: &mut [f64],
    prim_pre: &mut [f64],
    c2p_status: &mut [f64],
    grid_size_0: i32,
    dom_lo_0: i32,
    _l0: i32,
    _l1: i32,
    _l2: i32,
    _l3: i32,
    _l4: i32,
    _l5: i32,
    gamma: f64,
) {
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("rhd_c2p_1d")
        .input("cons.den", cons_den)
        .input("cons.mom_0", cons_mom)
        .input("cons.nrg", cons_nrg)
        .output("prim.rho", prim_rho)
        .output("prim.vel_0", prim_vel)
        .output("prim.pre", prim_pre)
        .output("scratch", c2p_status)
        .grid(&grid)
        .dom_lo(&dom)
        .scalar("gamma", gamma)
        .run();
}

const GAMMA: f64 = 5.0 / 3.0;

// forward map (1D, ideal gas): primitives -> conserved (D, S, tau).
//   eps = p/((gamma-1)*rho);  h = 1 + eps + p/rho;  W = 1/sqrt(1 - v^2)
//   D = rho*W;  S = rho*h*W^2*v;  tau = rho*h*W^2 - p - D
fn prim_to_cons(rho: f64, v: f64, p: f64, gamma: f64) -> (f64, f64, f64) {
    let w = 1.0 / (1.0 - v * v).sqrt();
    let eps = p / ((gamma - 1.0) * rho);
    let h = 1.0 + eps + p / rho;
    let rhw2 = rho * h * w * w;
    let d = rho * w;
    (d, rhw2 * v, rhw2 - p - d)
}

// independent reference: the relativistic-pressure Newton run to convergence
// (the do-while of conversion.hpp / rhd.rs). returns (rho, v, p).
fn rhd_to_primitive_ref(d: f64, s: f64, tau: f64, gamma: f64) -> (f64, f64, f64) {
    let smag = s.abs(); // |S| in 1D
    let tol = d * 1.0e-12;
    let mut p = (smag - d - tau).abs(); // initial guess, matching rhd_c2p
    for _ in 0..200 {
        let et = tau + d + p;
        let v2 = (smag * smag) / (et * et);
        let w = 1.0 / (1.0 - v2).sqrt();
        let rho = d / w;
        let eps = (tau + (1.0 - w) * d + (1.0 - w * w) * p) / (d * w);
        let c2 = ((gamma - 1.0) * gamma * eps) / (1.0 + gamma * eps);
        let f = (gamma - 1.0) * rho * eps - p;
        let g = c2 * v2 - 1.0;
        let p_next = p - f / g;
        let done = (p_next - p).abs() < tol;
        p = p_next;
        if done {
            break;
        }
    }
    // recovery (3-velocity, signed): vel = S/et, W = 1/sqrt(1-v^2), rho = D/W.
    let inv_et = 1.0 / (tau + d + p);
    let vel = s * inv_et;
    let w = 1.0 / (1.0 - vel * vel).sqrt();
    (d / w, vel, p)
}

// (rho, v, p): static + mild-relativistic states (|v| <= 0.8, W <= 1.67) that
// converge well within the 20 baked Newton steps.
const CASES: &[(f64, f64, f64)] = &[
    (1.0, 0.0, 1.0),
    (1.0, 0.3, 0.5),
    (2.0, 0.5, 1.0),
    (0.5, -0.4, 0.2),
    (1.0, 0.6, 2.0),
    (3.0, 0.8, 5.0),
    (1.0, -0.7, 1.0),
];

// run the compiled kernel over the conserved states built from cases.
fn run_kernel(den: &[f64], mom: &[f64], nrg: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = den.len();
    let mut prim_rho = vec![0.0_f64; n];
    let mut prim_vel = vec![0.0_f64; n];
    let mut prim_pre = vec![0.0_f64; n];
    let mut c2p_status = vec![-1.0_f64; n];
    rhd_c2p_1d(
        den,
        mom,
        nrg,
        &mut prim_rho,
        &mut prim_vel,
        &mut prim_pre,
        &mut c2p_status,
        n as i32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        GAMMA,
    );
    // the status channel rides the recovery kernel: every case here is a
    // physical state, so the accept/reject fact is zero on each cell.
    for (i, s) in c2p_status.iter().enumerate() {
        assert_eq!(*s, 0.0, "c2p_status[{i}] on a physical state");
    }
    (prim_rho, prim_vel, prim_pre)
}

#[test]
fn rhd_c2p_round_trips_prim_to_cons() {
    let den: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).0)
        .collect();
    let mom: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).1)
        .collect();
    let nrg: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).2)
        .collect();

    let (rho, vel, pre) = run_kernel(&den, &mom, &nrg);

    for (i, &(r0, v0, p0)) in CASES.iter().enumerate() {
        let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(1.0);
        assert!(rel(rho[i], r0) < 1e-9, "case {i}: rho {} != {r0}", rho[i]);
        assert!(rel(vel[i], v0) < 1e-9, "case {i}: vel {} != {v0}", vel[i]);
        assert!(rel(pre[i], p0) < 1e-9, "case {i}: pre {} != {p0}", pre[i]);
    }
}

#[test]
fn rhd_face_flux_uniform_state_is_consistent() {
    // HLLE consistency: for a uniform field (PLM gives L == R), the HLLE flux
    // equals the analytic physical flux F(U) exactly — independent of the wave
    // speeds. so this runs the compiled relativistic flux kernel and checks its
    // U(prim)/F(U) against the closed-form RHD fluxes.
    let (rho, v, p) = (1.5_f64, 0.4_f64, 0.8_f64);
    let n = 8usize;
    let den = vec![rho; n];
    let v0 = vec![v; n];
    let pre = vec![p; n];
    let (mut fden, mut fmom, mut fnrg) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    // interior only: the stencil reads coord-2..coord+1, so iterate cells 2..6
    // (dom_lo=2, grid=4) — all reads land inside the size-8 buffers.
    NamedKernel::new("rhd_face_flux_1d_0")
        .input("prim.rho", &den)
        .input("prim.vel[0]", &v0)
        .input("prim.pre", &pre)
        .output("flux.den", &mut fden)
        .output("flux.mom_0", &mut fmom)
        .output("flux.nrg", &mut fnrg)
        .grid(&[4])
        .dom_lo(&[2])
        // gamma/theta + the cartesian geometry scalars the flux kernel gained with the
        // curvilinear/moving-mesh work (no motion -> mesh_adot/mesh_vtrans = 0; cartesian
        // flux is position-independent so x_lo/dx are inert, dx = 1).
        .scalar("gamma", GAMMA)
        .scalar("theta", 1.0)
        .scalar("mesh_adot_0", 0.0)
        .scalar("x_lo_0", 0.0)
        .scalar("dx_0", 1.0)
        .scalar("mesh_vtrans_0", 0.0)
        .run();

    // analytic F(U): W=1/sqrt(1-v^2), h=1+gamma/(gamma-1)*p/rho, rhW2=rho*h*W^2;
    // D=rho*W, S=rhW2*v; F_D=D*v, F_S=S*v+p, F_tau=S-D*v.
    let w = 1.0 / (1.0 - v * v).sqrt();
    let h = 1.0 + GAMMA / (GAMMA - 1.0) * p / rho;
    let rhw2 = rho * h * w * w;
    let (d, s) = (rho * w, rhw2 * v);
    let (fd, fs, ftau) = (d * v, s * v + p, s - d * v);
    for i in 2..6 {
        assert!(
            (fden[i] - fd).abs() < 1e-9,
            "cell {i}: F_D {} != {fd}",
            fden[i]
        );
        assert!(
            (fmom[i] - fs).abs() < 1e-9,
            "cell {i}: F_S {} != {fs}",
            fmom[i]
        );
        assert!(
            (fnrg[i] - ftau).abs() < 1e-9,
            "cell {i}: F_tau {} != {ftau}",
            fnrg[i]
        );
    }
}

#[test]
fn rhd_c2p_matches_reference_newton() {
    let den: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).0)
        .collect();
    let mom: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).1)
        .collect();
    let nrg: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).2)
        .collect();

    let (rho, vel, pre) = run_kernel(&den, &mom, &nrg);

    for i in 0..CASES.len() {
        let (rr, vr, pr) = rhd_to_primitive_ref(den[i], mom[i], nrg[i], GAMMA);
        assert!(
            (rho[i] - rr).abs() < 1e-10,
            "case {i}: rho {} != ref {rr}",
            rho[i]
        );
        assert!(
            (vel[i] - vr).abs() < 1e-10,
            "case {i}: vel {} != ref {vr}",
            vel[i]
        );
        assert!(
            (pre[i] - pr).abs() < 1e-10,
            "case {i}: pre {} != ref {pr}",
            pre[i]
        );
    }
}

#[test]
fn rhd_out_of_cone_rejects_with_the_pressure_sentinel() {
    // momentum well past the cone bound (q = tau/D + 1 - sqrt(1 + r^2) < 0
    // at r = 10): outside the physical cone at a moderate scale, so the
    // ceiling-clamped candidate stays finite in f64.
    // the kernel recovers a finite ceiling-clamped candidate, drives the
    // pressure to the density-scaled sentinel -|D|, and the status channel
    // carries the rejection — data and control ride separate channels. the
    // flat subluminal clause is belt here: the velocity ceiling keeps the
    // candidate norm below one, so the rejection arrives through the
    // sentinel pressure.
    let n = 1;
    let (mut rho, mut vel, mut pre) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut status = vec![-1.0; n];
    NamedKernel::new("rhd_c2p_1d")
        .input("cons.den", &[1.0])
        .input("cons.mom_0", &[10.0])
        .input("cons.nrg", &[0.1])
        .output("prim.rho", &mut rho)
        .output("prim.vel_0", &mut vel)
        .output("prim.pre", &mut pre)
        .output("scratch", &mut status)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("gamma", GAMMA)
        .run();
    assert_eq!(status[0], 64.0, "the status channel carries the rejection");
    assert_eq!(
        pre[0], -1.0,
        "the raw cone sentinel -|D| survives on the candidate"
    );
    assert!(rho[0].is_finite() && vel[0].is_finite());
}
