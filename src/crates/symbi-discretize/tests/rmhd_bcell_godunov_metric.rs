// =============================================================================
// rmhd_bcell_godunov_metric.rs
//
// metric-consistency regressions for the cell-B flux predictor (rmhd_bcell_godunov_euler), which
// evolves the OUT-OF-PLANE (non-CT) cell B components as a flux divergence — the in-plane
// components live on faces and are re-derived by bcell_from_bface, so the predictor leaves them
// alone (oop_predictor_spec.md). the out-of-plane divergence operator depends on the chart, the
// gridded plane, and the component storage:
//
// flat (PHYSICAL components), out-of-plane curl per plane:
//   - cyl r-z (axes [0,2]): (curl E)_phi = d_z E_r - d_r E_z is METRIC-FREE; the gas
//     area-weighted divergence (h_phi = r in the volume) would inject a spurious -F_r/r.
//   - sph r-theta (axes [0,1]): d_t B_phi = -(1/r)[d_r(r F^r) + d_theta F^theta] — face
//     weights (r, 1) on the r dr dtheta measure; the gas r^2 sin(theta) measure would inject
//     spurious -F^r/r - cot(theta) F^theta/r sources.
//
// curved (CONTRAVARIANT components), the out-of-plane component obeys the densitized conservation
// law d_t(sqrt(gamma) B^i) + d_j(alpha sqrt(gamma) G^j) = 0: the covariant area-weighted divergence
// times the cell lapse alpha (the face kernel writes G = F - (beta^n/alpha) U, deferring one alpha
// to the divergence — the same contract the gas godunov honors), with NO flat out-of-plane shortcut
// (B is contravariant). the lapse witness is measure-free: the det-g-flat covariant measure is
// M-independent, so update(M)/update(0) must equal alpha(centroid) exactly.
//
// the IN-PLANE components (CT-evolved; the predictor does not write them) and the r-phi disk (out-of-plane
// z, where the area-weighting IS the correct z-curl) are exercised by the rotor + GPU gates in the
// symbi crate.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::Spacetime;
use symbi_discretize::{Coords, Spacing, rmhd_bcell_godunov_euler_gv};

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
    let out = KernelRun::new(rmhd_bcell_godunov_euler_gv(
        Coords::Cylindrical,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 2],
        2,
        3,
        &[0, 2],
    ))
    .grid([MR, MZ])
    // run over cells whose +1 neighbor (the flux divergence stencil) stays in bounds.
    .compute_window([0, 0], [MR - 1, MZ - 1])
    .field_with("bc_0", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .field_with("bc_1", move |c| bc1c[idx2(c[0], c[1])])
    .field_with("bc_2", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .field_with("bf_0_0", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .field_with("bf_0_1", move |c| bf01c[idx2(c[0], c[1])])
    .field_with("bf_0_2", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .field_with("bf_1_0", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .field_with("bf_1_1", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .field_with("bf_1_2", {
        let z = zero.clone();
        move |c| z[idx2(c[0], c[1])]
    })
    .scalars(&[
        ("dt", DT),
        ("x_lo_0", R0),
        ("dx_0", DR),
        ("x_lo_1", Z0),
        ("dx_1", DZ),
    ])
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

/// the out-of-plane component set for a chart: the B-vector slots whose coordinate is NOT a grid
/// axis. the predictor writes ONLY these (the in-plane components are CT / interp(bface)).
fn oop_comps(axes: &[usize]) -> Vec<usize> {
    (0..3usize).filter(|c| !axes.contains(c)).collect()
}

/// run the euler cell-B predictor on an 8x8 grid with uniform B0 in every component and the
/// uniform induction flux `bflux(d, c)`; returns the `bc_c_new` grid per component, `Some` for the
/// OUT-OF-PLANE components the predictor writes and `None` for the in-plane (CT) components it
/// leaves untouched.
fn run_bcell_euler(
    coords: Coords,
    spacetime: Spacetime,
    axes: &'static [usize],
    scalars: &[(&str, f64)],
    bflux: impl Fn(usize, usize) -> f64,
    b0: f64,
) -> Vec<Option<Vec<f64>>> {
    let mut run = KernelRun::new(rmhd_bcell_godunov_euler_gv(
        coords,
        spacetime,
        &[Spacing::Uniform; 2],
        2,
        3,
        axes,
    ))
    .grid([MR, MZ])
    .compute_window([0, 0], [MR - 1, MZ - 1]);
    for c in 0..3usize {
        run = run.field_with(&format!("bc_{c}"), move |_| b0);
    }
    for d in 0..2usize {
        for c in 0..3usize {
            let v = bflux(d, c);
            run = run.field_with(&format!("bf_{d}_{c}"), move |_| v);
        }
    }
    let out = run.scalars(scalars).run();
    let oop = oop_comps(axes);
    (0..3)
        .map(|c| {
            oop.contains(&c)
                .then(|| out.values(&format!("bc_{c}_new")).to_vec())
        })
        .collect()
}

#[test]
fn sph_rtheta_out_of_plane_bphi_uses_curl_weighted_divergence() {
    // flat spherical (r,theta), out-of-plane B_phi (comp 2), physical storage:
    // d_t B_phi = -(1/r)[d_r(r F^r) + d_theta F^theta].
    let th0 = 0.4; // theta_min (off the pole so the buggy cot(theta) signal is finite)
    let dth = 0.1;
    let scalars: &[(&str, f64)] = &[
        ("dt", DT),
        ("x_lo_0", R0),
        ("dx_0", DR),
        ("x_lo_1", th0),
        ("dx_1", dth),
    ];

    // uniform RADIAL flux F^r = C: update = -C/r_c (arithmetic midpoint; the r-weight is
    // linear so the midpoint form is exact). the gas r^2 sin(theta) divergence gives ~2C/r_c.
    let rad = run_bcell_euler(
        Coords::Spherical,
        Spacetime::Minkowski,
        &[0, 1],
        scalars,
        |d, c| if d == 0 && c == 2 { CFLUX } else { 0.0 },
        B0,
    );
    // uniform THETA flux F^theta = C: d_theta F^theta = 0, so B_phi is UNCHANGED. the gas
    // sin(theta) face weights give a spurious -C cot(theta)/r drive.
    let ang = run_bcell_euler(
        Coords::Spherical,
        Spacetime::Minkowski,
        &[0, 1],
        scalars,
        |d, c| if d == 1 && c == 2 { CFLUX } else { 0.0 },
        B0,
    );

    // sph r-theta out-of-plane = B_phi (comp 2); the predictor writes only that slot.
    let rad2 = rad[2].as_ref().expect("sph r-theta oop = B_phi (comp 2)");
    let ang2 = ang[2].as_ref().expect("sph r-theta oop = B_phi (comp 2)");
    for i in 0..MR - 1 {
        let r_c = R0 + (i as f64 + 0.5) * DR;
        let expected = B0 - DT * CFLUX / r_c;
        for j in 0..MZ - 1 {
            let got = rad2[i + j * MR];
            assert!(
                (got - expected).abs() < 1e-13,
                "sph r-theta B_phi radial-flux update wrong at ({i},{j}): got {got}, want \
                 {expected} (B0 - dt*C/r_c) — the gas r^2 sin(theta) divergence injects -C/r"
            );
            let dev = (ang2[i + j * MR] - B0).abs();
            assert!(
                dev < 1e-13,
                "sph r-theta B_phi changed under a UNIFORM theta flux at ({i},{j}) \
                 (dev={dev:e}) — the sin(theta) face weights inject a spurious cot(theta)/r drive"
            );
        }
    }
}

/// per-cell covariant radial divergence + centroid lapse checks shared by the GR charts, applied to
/// the OUT-OF-PLANE component the predictor writes (the in-plane slots are `None`). with a uniform
/// radial flux C, that component's update must be the covariant divergence (NO flat out-of-plane
/// shortcut on curved charts — B is contravariant), matching `div_formula(i)` at M = 0 and scaling
/// by `alpha(i, j)` at M > 0.
fn assert_gr_bcell_updates(
    upd0: &[Option<Vec<f64>>],
    upd_m: &[Option<Vec<f64>>],
    div_formula: impl Fn(usize) -> f64,
    alpha: impl Fn(usize, usize) -> f64,
    chart: &str,
) {
    for i in 0..MR - 1 {
        let want0 = DT * CFLUX * div_formula(i);
        for j in 0..MZ - 1 {
            let k = i + j * MR;
            for c in 0..3 {
                let (Some(u0), Some(um)) = (&upd0[c], &upd_m[c]) else {
                    continue;
                };
                assert!(
                    (u0[k] - want0).abs() < 1e-12 * want0.abs().max(1.0),
                    "{chart} comp {c} M=0 update at ({i},{j}): got {}, want {want0} (the \
                     covariant divergence; a flat out-of-plane shortcut on a curved chart \
                     mis-evolves the contravariant component)",
                    u0[k]
                );
                let want_m = alpha(i, j) * want0;
                assert!(
                    (um[k] - want_m).abs() < 1e-12 * want_m.abs().max(1.0),
                    "{chart} comp {c} M>0 update at ({i},{j}): got {}, want {want_m} = \
                     alpha*update(M=0) — the divergence is missing the lapse weight",
                    um[k]
                );
            }
        }
    }
}

#[test]
fn schwarzschild_sph_bcell_predictor_applies_lapse_weight() {
    // schwarzschild (r,theta), contravariant B: the out-of-plane B^phi (comp 2) obeys the densitized
    // law with the covariant r^2 sin(theta) measure (M-independent), lapse-weighted by
    // alpha = sqrt(1 - 2M/r_cen) at the volume-weighted radial centroid.
    let mass = 0.2;
    let th0 = 0.4;
    let dth = 0.1;
    let base: Vec<(&str, f64)> = vec![
        ("dt", DT),
        ("x_lo_0", R0),
        ("dx_0", DR),
        ("x_lo_1", th0),
        ("dx_1", dth),
    ];
    let uniform_radial = |d: usize, _c: usize| if d == 0 { CFLUX } else { 0.0 };
    let run = |m: f64| {
        let mut s = base.clone();
        s.push(("schwarzschild_mass", m));
        let b = run_bcell_euler(
            Coords::Spherical,
            Spacetime::Schwarzschild,
            &[0, 1],
            &s,
            uniform_radial,
            B0,
        );
        // per-component UPDATE (B0 - bnew), the divergence signal itself.
        b.into_iter()
            .map(|g| g.map(|grid| grid.iter().map(|v| B0 - v).collect::<Vec<f64>>()))
            .collect::<Vec<Option<Vec<f64>>>>()
    };
    let upd0 = run(0.0);
    let upd_m = run(mass);

    // uniform radial flux C: div = C * (A_hi - A_lo)/V = 3C (rh^2 - rl^2)/(rh^3 - rl^3)
    // (the angular measure cancels between the r-face weight and the volume).
    let div = |i: usize| {
        let (rl, rh) = (R0 + i as f64 * DR, R0 + (i + 1) as f64 * DR);
        3.0 * (rh * rh - rl * rl) / (rh * rh * rh - rl * rl * rl)
    };
    // lapse at the volume-weighted radial centroid r_cen = (3/4)(rh^4 - rl^4)/(rh^3 - rl^3).
    let alpha = |i: usize, _j: usize| {
        let (rl, rh) = (R0 + i as f64 * DR, R0 + (i + 1) as f64 * DR);
        let r_cen = 0.75 * (rh.powi(4) - rl.powi(4)) / (rh.powi(3) - rl.powi(3));
        (1.0 - 2.0 * mass / r_cen).sqrt()
    };
    assert_gr_bcell_updates(&upd0, &upd_m, div, alpha, "schwarzschild sph");
}

#[test]
fn kerr_schild_cyl_rz_bcell_oop_uses_covariant_divergence_and_lapse() {
    // kerr-schild cylindrical (R,z), contravariant B: the out-of-plane B^phi must take the
    // SAME covariant R-measure divergence as the in-plane components (the flat metric-free
    // shortcut would give a ZERO update for a uniform radial flux — dropping the G^R/R
    // geometric term of d_t B^phi = -(1/sqrt(gamma))[d_R(R G^R) + d_z(R G^z)] entirely),
    // lapse-weighted by alpha = 1/sqrt(1 + 2M/r), r = sqrt(R_cen^2 + z_cen^2).
    let mass = 0.2;
    let base: Vec<(&str, f64)> = vec![
        ("dt", DT),
        ("x_lo_0", R0),
        ("dx_0", DR),
        ("x_lo_1", Z0),
        ("dx_1", DZ),
    ];
    let uniform_radial = |d: usize, _c: usize| if d == 0 { CFLUX } else { 0.0 };
    let run = |m: f64| {
        let mut s = base.clone();
        s.push(("schwarzschild_mass", m));
        let b = run_bcell_euler(
            Coords::Cylindrical,
            Spacetime::KerrSchild,
            &[0, 2],
            &s,
            uniform_radial,
            B0,
        );
        b.into_iter()
            .map(|g| g.map(|grid| grid.iter().map(|v| B0 - v).collect::<Vec<f64>>()))
            .collect::<Vec<Option<Vec<f64>>>>()
    };
    let upd0 = run(0.0);
    let upd_m = run(mass);

    // uniform radial flux C on the R-measure: div = C (Rh - Rl)/Ir2 = 2C/(Rh + Rl).
    let div = |i: usize| {
        let (rl, rh) = (R0 + i as f64 * DR, R0 + (i + 1) as f64 * DR);
        2.0 / (rh + rl)
    };
    let alpha = |i: usize, j: usize| {
        let (rl, rh) = (R0 + i as f64 * DR, R0 + (i + 1) as f64 * DR);
        let r_cen = (2.0 / 3.0) * (rh.powi(3) - rl.powi(3)) / (rh.powi(2) - rl.powi(2));
        let z_cen = Z0 + (j as f64 + 0.5) * DZ;
        let r_sph = (r_cen * r_cen + z_cen * z_cen).sqrt();
        1.0 / (1.0 + 2.0 * mass / r_sph).sqrt()
    };
    assert_gr_bcell_updates(&upd0, &upd_m, div, alpha, "kerr-schild cyl r-z");
}
