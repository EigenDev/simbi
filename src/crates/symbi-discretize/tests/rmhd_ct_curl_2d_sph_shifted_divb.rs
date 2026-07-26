// =============================================================================
// rmhd_ct_curl_2d_sph_shifted_divb.rs
//
// div(B)=0 preservation for the 2D poloidal CT curl on the SHIFTED charts
// (kerr-schild + spinning kerr) — the phase-C diagnostic. same construction as
// rmhd_ct_curl_2d_sph_gr_divb.rs (schwarzschild) but with the shifted metrics'
// sqrt(gamma). if the curl preserves the w-weighted div(B) here, the kerr div
// drift observed in the full sim is NOT in the curl (consistent with the
// weight-independent symbolic telescoping proof) and lives in the integration.
//   kerr-schild: sqrt(gamma) = r^2 sin(theta) sqrt(1 + 2M/r)
//   kerr:        sqrt(gamma) = Sigma sin(theta) sqrt(1 + 2Mr/Sigma),
//                Sigma = r^2 + a^2 cos^2(theta)
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{Coords, Spacetime, Spacing, rmhd_ct_curl_2d_sph_gr_gv};

const M: usize = 8;
const R0: f64 = 3.0;
const DR: f64 = 0.4;
const T0: f64 = 0.5;
const DTH: f64 = 0.04;
const DT: f64 = 0.013;
const MASS: f64 = 1.0;
const SPIN: f64 = 0.9;

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

fn sqrtg(st: Spacetime, r: f64, th: f64) -> f64 {
    match st {
        Spacetime::KerrSchild => r * r * th.sin() * (1.0 + 2.0 * MASS / r).sqrt(),
        Spacetime::Kerr => {
            let sigma = r * r + SPIN * SPIN * th.cos() * th.cos();
            sigma * th.sin() * (1.0 + 2.0 * MASS * r / sigma).sqrt()
        }
        _ => unreachable!(),
    }
}

fn w_r(st: Spacetime, fi: usize, j: usize) -> f64 {
    sqrtg(st, af(0, fi), mid(1, j)) * DTH
}
fn w_th(st: Spacetime, fj: usize, i: usize) -> f64 {
    sqrtg(st, mid(0, i), af(1, fj)) * DR
}

fn div_b(st: Spacetime, br: &[f64], bth: &[f64], i: usize, j: usize) -> f64 {
    (w_r(st, i + 1, j) * at(br, i + 1, j) - w_r(st, i, j) * at(br, i, j))
        + (w_th(st, j + 1, i) * at(bth, i, j + 1) - w_th(st, j, i) * at(bth, i, j))
}

fn run_curl(st: Spacetime, b_in: &[Vec<f64>; 2], ez: &[f64], dt: f64) -> [Vec<f64>; 2] {
    let f = M * M;
    let mut new_b: [Vec<f64>; 2] = [vec![0.0; f], vec![0.0; f]];
    for dir in 0..2usize {
        let (bvec, ezv) = (b_in[dir].clone(), ez.to_vec());
        new_b[dir] = KernelRun::new(rmhd_ct_curl_2d_sph_gr_gv(
            dir,
            st,
            Coords::Spherical,
            &[Spacing::Uniform; 2],
            &[0, 1],
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
            ("kerr_spin", SPIN),
        ])
        .run()
        .values("b_new")
        .to_vec();
    }
    new_b
}

fn preserves_div(st: Spacetime) {
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
    let b = run_curl(st, &zero, &a, 1.0);

    let mut max_init = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            max_init = max_init.max(div_b(st, &b[0], &b[1], i, j).abs());
        }
    }
    assert!(
        max_init < 1e-10,
        "{st:?}: init w-div(B) not zero: {max_init:e}"
    );

    let b2 = run_curl(st, &b, &e, DT);
    let mut max_after = 0.0_f64;
    let mut max_change = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            let after = div_b(st, &b2[0], &b2[1], i, j);
            let before = div_b(st, &b[0], &b[1], i, j);
            max_after = max_after.max(after.abs());
            max_change = max_change.max((after - before).abs());
        }
    }
    eprintln!(
        "{st:?} 2d poloidal CT div(B): init={max_init:e} after={max_after:e} change={max_change:e}"
    );
    assert!(
        max_after < 1e-10,
        "{st:?}: post-update w-div(B) not zero: {max_after:e}"
    );
    assert!(
        max_change < 1e-11,
        "{st:?}: w-div(B) changed under CT: {max_change:e}"
    );
}

#[test]
fn kerr_schild_ct_curl_preserves_div_b() {
    preserves_div(Spacetime::KerrSchild);
}

#[test]
fn kerr_ct_curl_preserves_div_b() {
    preserves_div(Spacetime::Kerr);
}

// ---- the ACTUAL contact EMF kernel + curl, in isolation ----
// drives rmhd_edge_emf_gr_gv (the real corner EMF, with the shift's beta_r) then the curl on a
// div-free B, and checks div preservation. if this holds, the EMF+curl kernels are consistent and
// the sim div drift is in the buffer/field management (swirl layout); if it breaks, the EMF kernel
// itself feeds the curl an inconsistent (multi-valued) corner EMF on the shifted chart.
use symbi_discretize::gv::rmhd_edge_emf_gr_gv;

fn run_emf(st: Spacetime) -> Vec<f64> {
    KernelRun::new(rmhd_edge_emf_gr_gv(
        st,
        Coords::Spherical,
        &[Spacing::Uniform; 2],
        &[0, 1],
    ))
    .grid([M, M])
    .compute_window([1, 1], [M - 1, M - 1])
    .field_with("edge_vp1", |c| {
        0.1 * (0.2 * c[0] as f64).sin() - 0.05 * (0.15 * c[1] as f64).cos()
    })
    .field_with("edge_vp2", |c| {
        -0.08 * (0.13 * c[1] as f64).sin() + 0.04 * (0.1 * c[0] as f64).cos()
    })
    .field_with("edge_bp1", |c| {
        0.3 * (0.18 * c[0] as f64).cos() * (0.12 * c[1] as f64).sin()
    })
    .field_with("edge_bp2", |c| {
        0.25 * (0.11 * c[0] as f64).sin() + 0.1 * (0.2 * c[1] as f64).cos()
    })
    .field_with("edge_bflux_a", |c| {
        0.2 * (0.17 * c[0] as f64 + 0.1 * c[1] as f64).sin()
    })
    .field_with("edge_bflux_b", |c| {
        0.15 * (0.12 * c[0] as f64 - 0.14 * c[1] as f64).cos()
    })
    .field_with("edge_fden_p1", |c| 0.5 + 0.1 * (0.1 * c[0] as f64).sin())
    .field_with("edge_fden_p2", |c| 0.4 - 0.08 * (0.13 * c[1] as f64).cos())
    .scalars(&[
        ("x_lo_0", R0),
        ("dx_0", DR),
        ("x_lo_1", T0),
        ("dx_1", DTH),
        ("schwarzschild_mass", MASS),
        ("kerr_spin", SPIN),
    ])
    .run()
    .values("emf")
    .to_vec()
}

fn emf_then_curl_preserves_div(st: Spacetime) {
    let f = M * M;
    let avec = |i: usize, j: usize| (0.3 * i as f64).sin() * (0.2 * j as f64).cos();
    let mut a = vec![0.0; f];
    for i in 0..M {
        for j in 0..M {
            a[idx2(i, j)] = avec(i, j);
        }
    }
    let zero = [vec![0.0; f], vec![0.0; f]];
    let b = run_curl(st, &zero, &a, 1.0); // div-free B

    let ez = run_emf(st);
    let b2 = run_curl(st, &b, &ez, DT);

    let mut max_change = 0.0_f64;
    for i in 1..M - 3 {
        for j in 1..M - 3 {
            let d = (div_b(st, &b2[0], &b2[1], i, j) - div_b(st, &b[0], &b[1], i, j)).abs();
            max_change = max_change.max(d);
        }
    }
    eprintln!("{st:?} EMF+curl div(B) change = {max_change:e}");
    assert!(
        max_change < 1e-11,
        "{st:?}: real EMF+curl broke div(B): {max_change:e}"
    );
}

#[test]
fn kerr_schild_emf_then_curl_preserves_div() {
    emf_then_curl_preserves_div(Spacetime::KerrSchild);
}
#[test]
fn kerr_emf_then_curl_preserves_div() {
    emf_then_curl_preserves_div(Spacetime::Kerr);
}
