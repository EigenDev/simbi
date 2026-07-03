// =============================================================================
// rmhd_bcell_godunov_metric.rs
//
// regression for the cylindrical r-z OUT-OF-PLANE B_phi induction divergence (docs/design/30).
// the cell-B flux predictor (rmhd_bcell_godunov_euler) evolves the out-of-plane component as a
// flux divergence. for the r-z plane (axes [0,2]) the out-of-plane is B_phi (comp 1), whose
// induction curl (curl E)_phi = d_z E_r - d_r E_z is METRIC-FREE — yet the gas area-weighted FV
// divergence carries h_phi=r in the cell volume, which would inject a SPURIOUS -F_r/r source.
//
// the unambiguous discrete witness: a UNIFORM B_phi with a UNIFORM radial induction flux
// F_r(B_phi) = C (and F_z = 0). the correct (plain) divergence d_r F_r + d_z F_z = 0, so B_phi
// must be UNCHANGED. the buggy area-weighted divergence gives (1/r) d_r(r C) = C/r, decaying
// B_phi by dt*C/r (r-dependent). this test fails loudly on the bug, passes on the fix.
//
// the IN-PLANE B_r (comp 0) / B_z (comp 2) and the r-phi disk (out-of-plane z, h_z=1, where the
// area-weighting IS the correct z-curl) are exercised by the rotor + GPU gates in the symbi crate.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{rmhd_bcell_godunov_euler_gv, Coords, Spacing};
use symbi_discretize::Spacetime;

const MR: usize = 8; // r cells
const MZ: usize = 8; // z cells
const R0: f64 = 1.0; // r_min (avoid the axis)
const DR: f64 = 0.1;
const Z0: f64 = 0.0;
const DZ: f64 = 0.1;
const B0: f64 = 0.7; // uniform out-of-plane B_phi
const CFLUX: f64 = 0.3; // uniform radial induction flux F_r(B_phi)
const DT: f64 = 0.5;

fn idx2(i: usize, j: usize) -> usize {
    i * MZ + j
}

#[test]
fn cyl_rz_out_of_plane_bphi_uses_metric_free_divergence() {
    let n = MR * MZ;
    // uniform B_phi (comp 1); B_r (comp 0) / B_z (comp 2) = 0.
    let bc1: Vec<f64> = vec![B0; n];
    let zero: Vec<f64> = vec![0.0; n];
    // uniform radial induction flux of B_phi: bf_{d=0}_{c=1} = C; everything else 0.
    let bf01: Vec<f64> = vec![CFLUX; n];

    let (bc1c, bf01c) = (bc1.clone(), bf01.clone());
    // the cyl r-z bcell predictor (axes [0,2], DOF=3): out-of-plane = B_phi (comp 1).
    let out = KernelRun::new(rmhd_bcell_godunov_euler_gv(Coords::Cylindrical, Spacetime::Minkowski, &[Spacing::Uniform; 2], 2, 3, &[0, 2]))
        .grid([MR, MZ])
        // run over cells whose +1 neighbor (the flux divergence stencil) stays in bounds.
        .compute_window([0, 0], [MR - 1, MZ - 1])
        .field_with("bc_0", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .field_with("bc_1", move |c| bc1c[idx2(c[0], c[1])])
        .field_with("bc_2", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .field_with("bf_0_0", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .field_with("bf_0_1", move |c| bf01c[idx2(c[0], c[1])])
        .field_with("bf_0_2", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .field_with("bf_1_0", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .field_with("bf_1_1", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .field_with("bf_1_2", { let z = zero.clone(); move |c| z[idx2(c[0], c[1])] })
        .scalars(&[("dt", DT), ("x_lo_0", R0), ("dx_0", DR), ("x_lo_1", Z0), ("dx_1", DZ)])
        .run();

    let bphi_new = out.values("bc_1_new").to_vec();

    // B_phi must be UNCHANGED (plain div of a uniform flux = 0). the bug would give
    // B0 - dt*C/r_c (decaying with 1/r), which at r_c in [1.05, 1.75] is a ~0.09..0.14 drop.
    let mut max_dev = 0.0_f64;
    for i in 0..MR {
        for j in 0..MZ {
            // skip the high-r / high-z boundary cells whose +1 stencil reads the outside default.
            if i + 1 >= MR || j + 1 >= MZ {
                continue;
            }
            let v = bphi_new[idx2(i, j)];
            assert!(v.is_finite(), "B_phi non-finite at ({i},{j}): {v}");
            max_dev = max_dev.max((v - B0).abs());
        }
    }
    assert!(
        max_dev < 1e-13,
        "cyl r-z out-of-plane B_phi changed under a UNIFORM radial flux (dev={max_dev:e}) — the \
         metric-free divergence regressed (the area-weighted gas div injects a spurious -C/r source)"
    );
}
