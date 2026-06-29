// =============================================================================
// rmhd_ct_curl_2d_sph_poloidal_divb.rs
//
// the dedicated div(B)=0 preservation test for the 2D AXISYMMETRIC spherical
// poloidal constrained-transport curl (rmhd_ct_curl_2d_sph_gv) — the test its own
// doc comment (ct_emf.rs) flagged as still missing: the toroidal-injection case is
// trivially div-free (zero in-plane field), so a genuine POLOIDAL (B_r, B_theta)
// field is needed to actually exercise the staggered cancellation.
//
// the 2D builder evolves a poloidal field by the single out-of-plane corner EMF
// E_phi (`ez`):
//   dir=0 (B_r,   r-face):    dB_r/dt  = -(1/(r_f sin th_c)) d_th(sin th E_phi)
//   dir=1 (B_th, theta-face): dB_th/dt = +(1/r_c) d_r(r E_phi)
// axisymmetric div(B) is the area-weighted flux balance (the phi-extent dphi is a
// common constant factor, dropped):
//   div(B)[i,j] = A_r(i+1) B_r(i+1,j) - A_r(i) B_r(i,j)
//               + A_th(j+1) B_th(i,j+1) - A_th(j) B_th(i,j)
// with the point-form face areas A_r = r_f^2 sin(th_c) dth, A_th = r_c sin(th_f) dr
// (the 3D spherical areas without dphi). these are exactly the weights for which
// div(curl)=0 telescopes against the kernel's 1/(r sin th) and 1/r prefactors.
//
// B is initialized div-free as B = curl(A_phi) (a single out-of-plane vector
// potential through the SAME kernel: b=0, dt=1), then evolved one step by
// curl(E_phi). div(B) must stay machine zero AND unchanged.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{rmhd_ct_curl_2d_sph_gv, Spacing};

const M: usize = 8; // buffer extent per axis (r, theta)
const R0: f64 = 1.0;
const DR: f64 = 0.1;
const T0: f64 = 0.5; // theta start (shell away from the poles 0, pi)
const DTH: f64 = 0.04;
const DT: f64 = 0.013;

fn idx2(i: usize, j: usize) -> usize {
    i + j * M // axis-0-fastest, matching the harness/interp/Field storage convention
}
fn at(b: &[f64], i: usize, j: usize) -> f64 {
    b[idx2(i, j)]
}

// face position along axis at the integer index (start + idx*dx) — matches the
// kernel's gv_axis_face_at(Uniform).
fn af(ax: usize, idx: usize) -> f64 {
    match ax {
        0 => R0 + idx as f64 * DR,
        _ => T0 + idx as f64 * DTH,
    }
}
// transverse cell midpoint (the kernel's prefactor center on the non-dir axis).
fn mid(ax: usize, idx: usize) -> f64 {
    0.5 * (af(ax, idx) + af(ax, idx + 1))
}

// point-form r-face area at face index fi, cell-center theta j: r_f^2 sin(th_c) dth.
fn area_r(fi: usize, j: usize) -> f64 {
    let r = af(0, fi);
    r * r * mid(1, j).sin() * DTH
}
// point-form theta-face area at face index fj, cell-center radius i: r_c sin(th_f) dr.
fn area_th(fj: usize, i: usize) -> f64 {
    mid(0, i) * af(1, fj).sin() * DR
}

// point-form area-weighted div(B) at cell (i,j).
fn div_b(br: &[f64], bth: &[f64], i: usize, j: usize) -> f64 {
    (area_r(i + 1, j) * at(br, i + 1, j) - area_r(i, j) * at(br, i, j))
        + (area_th(j + 1, i) * at(bth, i, j + 1) - area_th(j, i) * at(bth, i, j))
}

// run the 2D poloidal CT curl once per face axis (out-of-place): new_b[dir] =
// b_in[dir] + dt * curl(ez)[dir]. b_in[0] is B_r, b_in[1] is B_theta; ez is the
// single out-of-plane corner EMF E_phi shared by both.
fn run_curl(b_in: &[Vec<f64>; 2], ez: &[f64], dt: f64) -> [Vec<f64>; 2] {
    let f = M * M;
    let mut new_b: [Vec<f64>; 2] = [vec![0.0; f], vec![0.0; f]];
    for dir in 0..2usize {
        let (bvec, ezv) = (b_in[dir].clone(), ez.to_vec());
        let out = KernelRun::new(rmhd_ct_curl_2d_sph_gv(dir, &[Spacing::Uniform; 2]))
            .grid([M, M])
            .compute_window([0, 0], [M - 1, M - 1])
            .field_with("b", move |c| bvec[idx2(c[0], c[1])])
            .field_with("ez", move |c| ezv[idx2(c[0], c[1])])
            .scalars(&[("dt", dt), ("x_lo_0", R0), ("dx_0", DR), ("x_lo_1", T0), ("dx_1", DTH)])
            .run();
        new_b[dir] = out.values("b_new").to_vec();
    }
    new_b
}

#[test]
fn ct_curl_2d_sph_poloidal_preserves_div_b() {
    let f = M * M;
    // an arbitrary smooth out-of-plane vector potential A_phi and a separate EMF E_phi,
    // both nontrivial in (r, theta) so the poloidal B = curl(A) has BOTH components.
    let avec = |i: usize, j: usize| -> f64 {
        let (x, y) = (i as f64, j as f64);
        (0.3 * x).sin() * (0.2 * y).cos() + 0.15 * (0.1 * x + 0.25 * y).sin()
    };
    let evec = |i: usize, j: usize| -> f64 {
        let (x, y) = (i as f64, j as f64);
        (0.4 * x).cos() * (0.3 * y).sin() + 0.2 * (0.2 * x - 0.1 * y).cos()
    };

    let mut a = vec![0.0; f];
    let mut e = vec![0.0; f];
    for i in 0..M {
        for j in 0..M {
            a[idx2(i, j)] = avec(i, j);
            e[idx2(i, j)] = evec(i, j);
        }
    }

    // divergence-free init: B = curl(A_phi) through the same kernel (b=0, dt=1).
    let zero = [vec![0.0; f], vec![0.0; f]];
    let b = run_curl(&zero, &a, 1.0);

    // the init must already have a NONTRIVIAL poloidal field (else the test is vacuous,
    // exactly the trivial toroidal-injection case the doc comment warns about).
    let mut max_b = 0.0_f64;
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            max_b = max_b.max(at(&b[0], i, j).abs()).max(at(&b[1], i, j).abs());
        }
    }
    assert!(max_b > 1e-2, "poloidal init is ~zero ({max_b:e}) — test would be vacuously div-free");

    // sanity: the init is area-weighted divergence-free to machine precision (div(curl)=0).
    let mut max_init = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            max_init = max_init.max(div_b(&b[0], &b[1], i, j).abs());
        }
    }
    assert!(max_init < 1e-10, "init area-weighted div(B) not zero: max = {max_init:e}");

    // evolve one CT step by curl(E_phi).
    let b2 = run_curl(&b, &e, DT);

    // div(B) after the update must still be machine zero AND unchanged.
    let mut max_after = 0.0_f64;
    let mut max_change = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            let after = div_b(&b2[0], &b2[1], i, j);
            let before = div_b(&b[0], &b[1], i, j);
            max_after = max_after.max(after.abs());
            max_change = max_change.max((after - before).abs());
        }
    }
    assert!(max_after < 1e-10, "post-update poloidal div(B) not zero: max = {max_after:e}");
    assert!(max_change < 1e-11, "poloidal div(B) changed under CT: max delta = {max_change:e}");
    eprintln!(
        "2d spherical poloidal CT div(B): |B|={max_b:e} init={max_init:e} after={max_after:e} change={max_change:e}"
    );
}
