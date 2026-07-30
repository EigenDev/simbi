// =============================================================================
// rmhd_ct_curl_sph_divb.rs
//
// proves the CURVILINEAR (spherical) constrained-transport curl
// (rmhd_ct_curl_3d_dir under Coords::Spherical) preserves the AREA-WEIGHTED
// div(B) = 0 to machine precision — the curvilinear generalization of the
// cartesian rmhd_ct_curl3d_divb gate. this is the
// non-negotiable correctness check for curvilinear CT.
//
// the curl is the ONE coord-generic orthogonal-curl in the scale factors h_p
// (geometry.rs ct_curl_metric). adjacent faces SHARE each edge, and the edge
// line-element weight h_p is a function of the GLOBAL edge index (same corner ->
// same h, scale_factor independent of the edge's own axis), so the point-form
// area-weighted divergence telescopes to 0 exactly — same guarantee as
// cartesian, independent of the metric.
//
// staggering (cell-indexed, row-major (i*M+j)*M+k): B_dir on the LOW dir-face,
// E_p on the edge parallel to p. div(B) is the point-form spherical flux balance
//   div(B)[i,j,k] = A_r(i+1)Bx(i+1) - A_r(i)Bx(i)
//                 + A_th(j+1)By(j+1) - A_th(j)By(j)
//                 + A_ph(k+1)Bz(k+1) - A_ph(k)Bz(k)
// with point-form face areas at the dir-face center (= 1 / the kernel's
// 1/(h_p1c h_p2c) prefactor x dx_p1 dx_p2). B is initialized div-free as the
// discrete curl of a vector potential A (run through the SAME kernel: b=0,
// dt=1 -> B = curl A), then evolved one step by curl(E).
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{Coords, Spacing, rmhd_ct_curl_3d_dir_gv};

const M: usize = 8; // buffer extent per axis
const R0: f64 = 1.0;
const DR: f64 = 0.1;
const T0: f64 = 0.5; // theta start (shell away from the poles 0, pi)
const DTH: f64 = 0.04;
const P0: f64 = 0.2; // phi start
const DPH: f64 = 0.06;
const DT: f64 = 0.011;

fn idx3(i: usize, j: usize, k: usize) -> usize {
    i + (j + k * M) * M // axis-0-fastest, matching the harness/interp/Field storage convention
}
fn at(b: &[f64], i: usize, j: usize, k: usize) -> f64 {
    b[idx3(i, j, k)]
}

// face position along axis at the integer index (start + idx*dx) — matches the
// kernel's axis_face_at(Uniform).
fn af(ax: usize, idx: usize) -> f64 {
    match ax {
        0 => R0 + idx as f64 * DR,
        1 => T0 + idx as f64 * DTH,
        _ => P0 + idx as f64 * DPH,
    }
}
// transverse cell midpoint (the kernel's prefactor center on the non-dir axes).
fn mid(ax: usize, idx: usize) -> f64 {
    0.5 * (af(ax, idx) + af(ax, idx + 1))
}

// point-form face areas at the dir-face center: h_p1c h_p2c dx_p1 dx_p2. these
// are exactly 1 / (kernel inv_pref) * dx_p1 dx_p2, so A_dir * (dB_dir/dt)
// collapses to the raw edge circulation that telescopes.
fn area_r(fi: usize, j: usize, _k: usize) -> f64 {
    // dir=0: h_th h_ph dth dph = r * (r sin(th_c)) dth dph at (r=af_r, th_c=mid).
    let r = af(0, fi);
    r * r * mid(1, j).sin() * DTH * DPH
}
fn area_th(fj: usize, i: usize, _k: usize) -> f64 {
    // dir=1: h_ph h_r dph dr = (r_c sin(th)) * 1 dph dr at (r_c=mid, th=af_th).
    mid(0, i) * af(1, fj).sin() * DR * DPH
}
fn area_ph(_fk: usize, i: usize, _j: usize) -> f64 {
    // dir=2: h_r h_th dr dth = 1 * r_c dr dth at (r_c=mid). (h_th=r, theta-free.)
    mid(0, i) * DR * DTH
}

// point-form area-weighted div(B) at cell (i,j,k).
fn div_b(bx: &[f64], by: &[f64], bz: &[f64], i: usize, j: usize, k: usize) -> f64 {
    (area_r(i + 1, j, k) * at(bx, i + 1, j, k) - area_r(i, j, k) * at(bx, i, j, k))
        + (area_th(j + 1, i, k) * at(by, i, j + 1, k) - area_th(j, i, k) * at(by, i, j, k))
        + (area_ph(k + 1, i, j) * at(bz, i, j, k + 1) - area_ph(k, i, j) * at(bz, i, j, k))
}

// run the spherical CT curl once per face axis (out-of-place): new_b[dir] =
// b_in[dir] + dt * curl(e)[dir], on a shell of (M-1)^3 interior cells.
fn run_curl(b_in: &[Vec<f64>; 3], e: &[Vec<f64>; 3], dt: f64) -> [Vec<f64>; 3] {
    let f = M * M * M;
    let mut new_b: [Vec<f64>; 3] = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    for dir in 0..3usize {
        let p1 = (dir + 1) % 3;
        let p2 = (dir + 2) % 3;
        // b = bface[dir]; e_p1 = efield[p1]; e_p2 = efield[p2]; the spherical curl reads the
        // axis-start + cell-width geometry scalars (uniform spacing) in coordinate order.
        let (bvec, ep1, ep2) = (b_in[dir].clone(), e[p1].clone(), e[p2].clone());
        let out = KernelRun::new(rmhd_ct_curl_3d_dir_gv(
            Coords::Spherical,
            &[Spacing::Uniform; 3],
            dir,
        ))
        .grid([M, M, M])
        .compute_window([0, 0, 0], [M - 1, M - 1, M - 1])
        .field_with("b", move |c| bvec[idx3(c[0], c[1], c[2])])
        .field_with("e_p1", move |c| ep1[idx3(c[0], c[1], c[2])])
        .field_with("e_p2", move |c| ep2[idx3(c[0], c[1], c[2])])
        .scalars(&[
            ("dt", dt),
            ("x_lo_0", R0),
            ("dx_0", DR),
            ("x_lo_1", T0),
            ("dx_1", DTH),
            ("x_lo_2", P0),
            ("dx_2", DPH),
        ])
        .run();
        new_b[dir] = out.values("b_new").to_vec();
    }
    new_b
}

#[test]
fn ct_curl_sph_preserves_div_b() {
    let f = M * M * M;
    // an arbitrary smooth edge vector potential A and a separate edge EMF E.
    let avec = |c: usize, i: usize, j: usize, k: usize| -> f64 {
        let (x, y, z) = (i as f64, j as f64, k as f64);
        match c {
            0 => (0.3 * x).sin() * (0.2 * y + 0.1 * z).cos(),
            1 => (0.25 * y).cos() * (0.15 * z - 0.2 * x).sin(),
            _ => (0.2 * z).sin() * (0.3 * x + 0.1 * y).cos(),
        }
    };
    let evec = |c: usize, i: usize, j: usize, k: usize| -> f64 {
        let (x, y, z) = (i as f64, j as f64, k as f64);
        (0.4 * x + c as f64).sin() * (0.3 * y).cos() + 0.2 * (0.2 * z - c as f64).sin()
    };

    let mut a = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    let mut e = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    for i in 0..M {
        for j in 0..M {
            for k in 0..M {
                let idx = idx3(i, j, k);
                for c in 0..3 {
                    a[c][idx] = avec(c, i, j, k);
                    e[c][idx] = evec(c, i, j, k);
                }
            }
        }
    }

    // divergence-free init: B = curl(A) through the same kernel (b=0, dt=1).
    let zero = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    let b = run_curl(&zero, &a, 1.0);

    // sanity: the init is area-weighted divergence-free to machine precision.
    let mut max_init = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            for k in 0..M - 2 {
                max_init = max_init.max(div_b(&b[0], &b[1], &b[2], i, j, k).abs());
            }
        }
    }
    assert!(
        max_init < 1e-10,
        "init area-weighted div(B) not zero: max = {max_init:e}"
    );

    // evolve one CT step by curl(E).
    let b2 = run_curl(&b, &e, DT);

    // div(B) after the update must still be machine zero AND unchanged.
    let mut max_after = 0.0_f64;
    let mut max_change = 0.0_f64;
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            for k in 0..M - 2 {
                let after = div_b(&b2[0], &b2[1], &b2[2], i, j, k);
                let before = div_b(&b[0], &b[1], &b[2], i, j, k);
                max_after = max_after.max(after.abs());
                max_change = max_change.max((after - before).abs());
            }
        }
    }
    assert!(
        max_after < 1e-10,
        "post-update div(B) not zero: max = {max_after:e}"
    );
    assert!(
        max_change < 1e-11,
        "div(B) changed under CT: max delta = {max_change:e}"
    );
    eprintln!("spherical CT div(B): init={max_init:e} after={max_after:e} change={max_change:e}");
}

// the spherical scale factor h_c (Rust mirror of geometry.rs scale_factor).
fn sf(c: usize, r: f64, theta: f64) -> f64 {
    match c {
        1 => r,               // h_theta
        2 => r * theta.sin(), // h_phi
        _ => 1.0,             // h_r
    }
}

#[test]
fn ct_curl_sph_constant_edge_flux_is_curl_free() {
    // E_c = C / h_c(edge) => the edge line-flux h_c E_c = C is CONSTANT along every
    // axis, so every weighted forward difference vanishes => curl = 0 for all three
    // face components. a WRONG metric weight would not cancel. b=0, dt=1 => new_b
    // must be machine zero. this pins the per-dir h-weighting (div B alone cannot —
    // cartesian h=1 also telescopes).
    let f = M * M * M;
    let c0 = 1.7;
    let mut e = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    for i in 0..M {
        for j in 0..M {
            for k in 0..M {
                let (r, th) = (af(0, i), af(1, j));
                for c in 0..3 {
                    e[c][idx3(i, j, k)] = c0 / sf(c, r, th);
                }
            }
        }
    }
    let zero = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    let b = run_curl(&zero, &e, 1.0);
    let mut mx = 0.0_f64;
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            for k in 0..M - 1 {
                for dir in 0..3 {
                    mx = mx.max(at(&b[dir], i, j, k).abs());
                }
            }
        }
    }
    assert!(
        mx < 1e-12,
        "constant edge-flux gave nonzero curl: max = {mx:e}"
    );
    eprintln!("spherical constant-flux curl: max = {mx:e}");
}

#[test]
fn ct_curl_sph_b_r_matches_analytic() {
    // sharp magnitude test for B_r. E_theta = ALPHA*phi and E_phi = BETA*theta/sin(theta)
    // make h_theta E_theta (= r * ALPHA*phi) and h_phi E_phi (= r sin th * BETA th/sin th
    // = r BETA th) LINEAR in their difference variables (phi, theta resp.), so the
    // forward differences are EXACT. then the discrete
    //   curl_r = (1/(r sin th_c)) [ dE_theta/dphi - d(sin th E_phi)/dtheta ]
    //          = (1/(r sin th_c)) (ALPHA - BETA)
    // matches the closed-form substrate curl (= -(curl E)_r) at the face center to
    // machine precision.
    const ALPHA: f64 = 0.37;
    const BETA: f64 = 0.53;
    let f = M * M * M;
    let mut e = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    for i in 0..M {
        for j in 0..M {
            for k in 0..M {
                // E_theta on the theta-edge: its phi coord is the phi-face af(2,k).
                e[1][idx3(i, j, k)] = ALPHA * af(2, k);
                // E_phi on the phi-edge: its theta coord is the theta-face af(1,j).
                let th = af(1, j);
                e[2][idx3(i, j, k)] = BETA * th / th.sin();
            }
        }
    }
    let zero = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    let b = run_curl(&zero, &e, 1.0); // dt=1 => b = curl
    let mut mx = 0.0_f64;
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            for k in 0..M - 1 {
                let r = af(0, i); // r at the r-face
                let th_c = mid(1, j); // theta center (the prefactor)
                let expect = (ALPHA - BETA) / (r * th_c.sin());
                mx = mx.max((at(&b[0], i, j, k) - expect).abs());
            }
        }
    }
    assert!(
        mx < 1e-11,
        "spherical curl_r magnitude off analytic: max = {mx:e}"
    );
    eprintln!("spherical curl_r vs analytic: max = {mx:e}");
}
