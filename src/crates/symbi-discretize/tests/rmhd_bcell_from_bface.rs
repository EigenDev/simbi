// =============================================================================
// rmhd_bcell_from_bface.rs
//
// validates the CT face->cell B interpolation + magnetic-energy correction
// (rmhd_bcell_from_bface) against its straight-Rust reference: each cell B is the
// average of the two bracketing faces, and cons.nrg picks up the magnetic-energy
// change (total energy conserved as the cell B is re-derived from the CT-evolved
// face B). pointwise per cell with a +1 face offset; tested on a small 3D grid.
//   bcell_c = 0.5*(bface_c[coord] + bface_c[coord+e_c])
//   nrg += 0.5*(|bcell_new|^2 - |bcell_old|^2)
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::rmhd_bcell_from_bface_gv;

const M: usize = 6;

#[test]
fn rmhd_bcell_from_bface_matches_reference() {
    // arbitrary smooth face B, old cell B, and energy.
    let bfx_fn = |i: usize, j: usize, k: usize| 0.5 + 0.3 * (0.2 * i as f64 + 0.1 * j as f64).sin() + 0.05 * k as f64;
    let bfy_fn = |i: usize, j: usize, k: usize| 0.2 * (0.3 * j as f64).cos() - 0.1 * (i as f64 - k as f64);
    let bfz_fn = |i: usize, _j: usize, k: usize| 0.1 + 0.4 * (0.15 * k as f64 + 0.2 * i as f64).sin();
    let bcx_fn = |i: usize, j: usize, _k: usize| 0.3 + 0.01 * (i + j) as f64;
    let bcy_fn = |i: usize, j: usize, _k: usize| 0.1 * (i as f64 - j as f64);
    let bcz_fn = |_i: usize, _j: usize, k: usize| 0.2 + 0.02 * k as f64;
    let nrg_fn = |i: usize, j: usize, k: usize| 1.0 + 0.1 * (i + j + k) as f64;

    // per-cell pullback of each generator into the harness. the kernel uses positional,
    // component-indexed field keys (bf_0/bf_1/bf_2 = faces, bc_0/bc_1/bc_2 = old cells, nrg). the
    // writes (bc_0_new / bc_1_new / bc_2_new / nrg_new) are out-of-place so they read back the
    // post-update cell B / nrg. run on [0, M-1)^3 so the +1 face offsets stay in bounds.
    let out = KernelRun::new(rmhd_bcell_from_bface_gv(3))
        .grid([M, M, M])
        .compute_window([0, 0, 0], [M - 1, M - 1, M - 1])
        .field_with("bf_0", move |c| bfx_fn(c[0], c[1], c[2]))
        .field_with("bf_1", move |c| bfy_fn(c[0], c[1], c[2]))
        .field_with("bf_2", move |c| bfz_fn(c[0], c[1], c[2]))
        .field_with("bc_0", move |c| bcx_fn(c[0], c[1], c[2]))
        .field_with("bc_1", move |c| bcy_fn(c[0], c[1], c[2]))
        .field_with("bc_2", move |c| bcz_fn(c[0], c[1], c[2]))
        .field_with("nrg", move |c| nrg_fn(c[0], c[1], c[2]))
        .run();

    for i in 0..M - 1 {
        for j in 0..M - 1 {
            for k in 0..M - 1 {
                let bx = 0.5 * (bfx_fn(i, j, k) + bfx_fn(i + 1, j, k));
                let by = 0.5 * (bfy_fn(i, j, k) + bfy_fn(i, j + 1, k));
                let bz = 0.5 * (bfz_fn(i, j, k) + bfz_fn(i, j, k + 1));
                let old2 = bcx_fn(i, j, k).powi(2) + bcy_fn(i, j, k).powi(2) + bcz_fn(i, j, k).powi(2);
                let new2 = bx * bx + by * by + bz * bz;
                let wn = nrg_fn(i, j, k) + 0.5 * (new2 - old2);
                let c = [i, j, k];
                assert!((out.get(c, "bc_0_new") - bx).abs() < 1e-13, "bc_0 {i},{j},{k}: {} != {bx}", out.get(c, "bc_0_new"));
                assert!((out.get(c, "bc_1_new") - by).abs() < 1e-13, "bc_1 {i},{j},{k}: {} != {by}", out.get(c, "bc_1_new"));
                assert!((out.get(c, "bc_2_new") - bz).abs() < 1e-13, "bc_2 {i},{j},{k}: {} != {bz}", out.get(c, "bc_2_new"));
                assert!((out.get(c, "nrg_new") - wn).abs() < 1e-13, "nrg {i},{j},{k}: {} != {wn}", out.get(c, "nrg_new"));
            }
        }
    }
}
