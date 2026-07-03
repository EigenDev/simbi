// =============================================================================
// rmhd_ct_curl_2d_sph_gr_divb.rs
//
// div(B)=0 preservation for the CURVED-SPACETIME 2D poloidal CT curl
// (rmhd_ct_curl_2d_sph_gr_gv, schwarzschild): the update evolves the densitized
// Btilde^i = sqrt(gamma) B^i with the coordinate curl of the densitized corner
// EMF Etilde_phi, dividing back by the face's own constant weight
//   w_r  = sqrt(gamma)(r_f, th_c) dth,   w_th = sqrt(gamma)(r_c, th_f) dr,
// with sqrt(gamma) = r^2 sin(theta)/sqrt(1 - 2M/r). the preserved discrete
// divergence is the w-weighted flux balance
//   div(B)[i,j] = w_r(i+1,j) B_r(i+1,j) - w_r(i,j) B_r(i,j)
//               + w_th(j+1,i) B_th(i,j+1) - w_th(j,i) B_th(i,j)
// (the corner-EMF telescoping is weight-independent, so this holds to roundoff
// for any smooth Etilde). B initializes div-free as the curl of an arbitrary
// A_phi through the SAME kernel (b=0, dt=1), then evolves one step by
// curl(Etilde_phi). div(B) must stay machine zero AND unchanged.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{rmhd_ct_curl_2d_sph_gr_gv, Spacetime, Spacing};

const M: usize = 8; // buffer extent per axis (r, theta)
const R0: f64 = 3.0; // outside the horizon (mass 1)
const DR: f64 = 0.4;
const T0: f64 = 0.5;
const DTH: f64 = 0.04;
const DT: f64 = 0.013;
const MASS: f64 = 1.0;

fn idx2(i: usize, j: usize) -> usize {
    i + j * M
}
fn at(b: &[f64], i: usize, j: usize) -> f64 {
    b[idx2(i, j)]
}
fn af(ax: usize, idx: usize) -> f64 {
    match ax {
        0 => R0 + idx as f64 * DR,
        _ => T0 + idx as f64 * DTH,
    }
}
fn mid(ax: usize, idx: usize) -> f64 {
    0.5 * (af(ax, idx) + af(ax, idx + 1))
}

// schwarzschild sqrt(gamma) = r^2 sin(theta) / sqrt(1 - 2M/r).
fn sqrtg(r: f64, th: f64) -> f64 {
    r * r * th.sin() / (1.0 - 2.0 * MASS / r).sqrt()
}

// the per-face constant weights the kernel divides by.
fn w_r(fi: usize, j: usize) -> f64 {
    sqrtg(af(0, fi), mid(1, j)) * DTH
}
fn w_th(fj: usize, i: usize) -> f64 {
    sqrtg(mid(0, i), af(1, fj)) * DR
}

fn div_b(br: &[f64], bth: &[f64], i: usize, j: usize) -> f64 {
    (w_r(i + 1, j) * at(br, i + 1, j) - w_r(i, j) * at(br, i, j))
        + (w_th(j + 1, i) * at(bth, i, j + 1) - w_th(j, i) * at(bth, i, j))
}

fn run_curl(b_in: &[Vec<f64>; 2], ez: &[f64], dt: f64) -> [Vec<f64>; 2] {
    let f = M * M;
    let mut new_b: [Vec<f64>; 2] = [vec![0.0; f], vec![0.0; f]];
    for dir in 0..2usize {
        let (bvec, ezv) = (b_in[dir].clone(), ez.to_vec());
        let out = KernelRun::new(rmhd_ct_curl_2d_sph_gr_gv(
            dir,
            Spacetime::Schwarzschild,
            &[Spacing::Uniform; 2],
        ))
        .grid([M, M])
        .compute_window([0, 0], [M - 1, M - 1])
        .field_with("b", move |c| bvec[idx2(c[0], c[1])])
        .field_with("ez", move |c| ezv[idx2(c[0], c[1])])
        .scalars(&[
            ("dt", dt),
            ("x_lo_0", R0),
            ("dx_0", DR),
            ("x_lo_1", T0),
            ("dx_1", DTH),
            ("schwarzschild_mass", MASS),
        ])
        .run();
        new_b[dir] = out.values("b_new").to_vec();
    }
    new_b
}

#[test]
fn gr_ct_curl_2d_sph_poloidal_preserves_weighted_div_b() {
    let f = M * M;
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

    let zero = [vec![0.0; f], vec![0.0; f]];
    let b = run_curl(&zero, &a, 1.0);

    let mut max_b = 0.0_f64;
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            max_b = max_b.max(at(&b[0], i, j).abs()).max(at(&b[1], i, j).abs());
        }
    }
    assert!(max_b > 1e-2, "poloidal init is ~zero ({max_b:e}) — vacuously div-free");

    let mut max_init = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            max_init = max_init.max(div_b(&b[0], &b[1], i, j).abs());
        }
    }
    assert!(max_init < 1e-10, "init w-weighted div(B) not zero: max = {max_init:e}");

    let b2 = run_curl(&b, &e, DT);

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
    assert!(max_after < 1e-10, "post-update w-weighted div(B) not zero: max = {max_after:e}");
    assert!(max_change < 1e-11, "w-weighted div(B) changed under CT: max delta = {max_change:e}");
    eprintln!(
        "gr 2d spherical poloidal CT div(B): |B|={max_b:e} init={max_init:e} after={max_after:e} change={max_change:e}"
    );
}
