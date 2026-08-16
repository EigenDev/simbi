// =============================================================================
// rmhd_ct_curl_divb.rs
//
// proves the substrate constrained-transport curl update (rmhd_ct_curl_2d)
// preserves the divergence-free constraint div(B) = 0 to machine precision — the
// defining property of CT. B is initialized divergence-free from a discrete vector
// potential Az (Bx = dAz/dy, By = -dAz/dx, which makes the discrete div telescope
// to exactly 0), then evolved one step by the curl of an arbitrary edge EMF Ez.
// because the discrete curl + discrete divergence share the mixed Ez differences,
// d(div B)/dt is identically 0, so div(B) stays at machine zero.
//
// staggering (cell-indexed storage): Bx on x-faces, By on y-faces, Ez on corners.
//   div(B)[i,j] = idx*(Bx[i+1,j]-Bx[i,j]) + idy*(By[i,j+1]-By[i,j])
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::rmhd_ct_curl_2d_dir_gv;

const M: usize = 10; // buffer extent per axis (axis-0-fastest: flat = i + j*M, the canonical Field convention)
const IDX: f64 = 10.0; // 1/dx
const IDY: f64 = 10.0; // 1/dy
const DT: f64 = 0.013;

fn at(buf: &[f64], i: usize, j: usize) -> f64 {
    buf[i + j * M]
}

// div(B) at cell (i,j) from the face-centered B (needs Bx[i+1,j], By[i,j+1]).
fn div_b(bx: &[f64], by: &[f64], i: usize, j: usize) -> f64 {
    IDX * (at(bx, i + 1, j) - at(bx, i, j)) + IDY * (at(by, i, j + 1) - at(by, i, j))
}

#[test]
fn ct_curl_preserves_div_b() {
    // arbitrary smooth vector potential Az and edge EMF Ez.
    let az = |i: usize, j: usize| (0.3 * i as f64).sin() * (0.25 * j as f64).cos();
    let ez_fn = |i: usize, j: usize| {
        (0.4 * i as f64).sin() * (0.3 * j as f64).sin() + 0.2 * ((i + j) as f64).cos()
    };

    // divergence-free init: Bx = dAz/dy, By = -dAz/dx (defined on i,j in [0, M-1)).
    let mut bx = vec![0.0_f64; M * M];
    let mut by = vec![0.0_f64; M * M];
    let mut ez = vec![0.0_f64; M * M];
    for i in 0..M {
        for j in 0..M {
            ez[i + j * M] = ez_fn(i, j);
        }
    }
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            bx[i + j * M] = (az(i, j + 1) - az(i, j)) * IDY;
            by[i + j * M] = -(az(i + 1, j) - az(i, j)) * IDX;
        }
    }

    // sanity: the init is divergence-free to machine precision.
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            assert!(
                div_b(&bx, &by, i, j).abs() < 1e-12,
                "init div(B) nonzero at {i},{j}"
            );
        }
    }

    // build + run the gv CT curl update over [0, M-1)^2 so Ez[i+1,j]/Ez[i,j+1] stay in bounds.
    // the combined 2d curl was split per in-plane direction: dir=0 updates B_x from d_y(Ez),
    // dir=1 updates B_y from d_x(Ez). run both per-dir kernels (the generic `b` field bound to
    // bx for dir=0, by for dir=1) to advance the full in-plane field one step. out-of-place
    // writes b_new so the before/after div comparison reads the originals.
    let bx_built = rmhd_ct_curl_2d_dir_gv(0);
    assert_eq!(
        bx_built.0.scalar_params,
        vec!["dt".to_string(), "idy".to_string()]
    );
    let by_built = rmhd_ct_curl_2d_dir_gv(1);
    assert_eq!(
        by_built.0.scalar_params,
        vec!["dt".to_string(), "idx".to_string()]
    );

    let (bxc, ezc0) = (bx.clone(), ez.clone());
    let bx_out = KernelRun::new(bx_built)
        .grid([M, M])
        .compute_window([0, 0], [M - 1, M - 1])
        .field_with("b", move |c| bxc[c[0] + c[1] * M])
        .field_with("ez", move |c| ezc0[c[0] + c[1] * M])
        .scalars(&[("dt", DT), ("idy", IDY)])
        .run();
    let bx_new = bx_out.values("b_new");

    let (byc, ezc1) = (by.clone(), ez.clone());
    let by_out = KernelRun::new(by_built)
        .grid([M, M])
        .compute_window([0, 0], [M - 1, M - 1])
        .field_with("b", move |c| byc[c[0] + c[1] * M])
        .field_with("ez", move |c| ezc1[c[0] + c[1] * M])
        .scalars(&[("dt", DT), ("idx", IDX)])
        .run();
    let by_new = by_out.values("b_new");

    // div(B) after the update must still be machine zero (and unchanged from before).
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            let after = div_b(bx_new, by_new, i, j);
            let before = div_b(&bx, &by, i, j);
            assert!(
                after.abs() < 1e-12,
                "post-update div(B) nonzero at {i},{j}: {after}"
            );
            assert!(
                (after - before).abs() < 1e-13,
                "div(B) changed at {i},{j}: {before} -> {after}"
            );
        }
    }
}
