// =============================================================================
// fused_body_equals_two_pass.rs
//
// proves the immersed-body fold into `godunov_stage_gv_with_fused_built` (n_bodies > 0)
// is bit-for-bit identical to the standalone two-pass execution: plain `godunov_stage_gv`
// followed by the `body_source_gv` pass. this is the correctness gate that lets the body
// ride INSIDE the single fused update sweep (one launch) instead of a separate full-grid
// CONS read+write.
//
// the body is a POST-combine operator `(cons_g + ac_dt*S_grav) * exp(-drain*ac_dt)`, so the
// fused kernel wraps the godunov-combined nodes with it while the two-pass writes cons_g to a
// buffer and the body reads it back. an f64 store/load is exact, so the register-resident
// cons_g the fused body reads equals the memory value the standalone body reads — the two must
// agree to the bit. exercised at the euler stage (a0=0, ac=1) and an rk2 corrector (a0=ac=0.5,
// u_n != cons) to stress the ac*dt body weight and the a0*u_n snapshot combine.
//
// the flux divergence is zero (uniform flux), so the godunov combine reduces to a0*u_n + ac*cons;
// the body wrap still varies per cell through the cell centroid (gravity + drain depend on the
// distance to the body), so the per-cell centroid indexing is exercised. a `body_changed` guard
// keeps the oracle from passing vacuously on a no-op body.
//
// run: cargo test -p symbi-discretize --test fused_body_equals_two_pass
// =============================================================================

mod harness;

use harness::KernelRun;
use symbi_discretize::body_source_gv;
use symbi_discretize::coords::{Coords, Spacing, Spacetime};
use symbi_discretize::gv::{godunov_stage_gv, godunov_stage_gv_with_fused_built, GeoSource};

const GAMMA: f64 = 1.4;
const DT: f64 = 0.01;
const DX: f64 = 0.25;

// 4x4 grid, interior 2x2 window [1,1]+[2,2]; body params chosen so the interior cells sit inside
// the accretion mask (drain active) and feel gravity (both body operators exercised).
const GRID: [usize; 2] = [4, 4];
const WLO: [i32; 2] = [1, 1];
const WSIZE: [usize; 2] = [2, 2];

// the godunov flux fields for a 2D Newtonian (ncomp = 2 + energy) state.
fn flux_names() -> Vec<String> {
    let mut v = vec!["mass_flux_0".to_string(), "mass_flux_1".to_string()];
    for k in 0..2 {
        for i in 0..2 {
            v.push(format!("mom_flux_{k}_{i}"));
        }
    }
    v.push("nrg_flux_0".to_string());
    v.push("nrg_flux_1".to_string());
    v
}

// geometry + one-body scalars, shared across every leg (extra names are ignored per leg). a single
// point mass at (0.55, 0.55) with a wide accretion mask and a large sink rate so the interior cells
// drain at the sound-crossing cap; softened gravity keeps the force finite.
fn geom_and_body_scalars() -> Vec<(&'static str, f64)> {
    vec![
        ("gamma", GAMMA),
        ("mesh_hdil", 0.0),
        ("dx_0", DX),
        ("dx_1", DX),
        ("x_lo_0", 0.0),
        ("x_lo_1", 0.0),
        ("body_0_mass", 1.0),
        ("body_0_soft", 0.1),
        ("body_0_pos_0", 0.55),
        ("body_0_pos_1", 0.55),
        ("body_0_racc", 0.4),
        ("body_0_sink", 100.0),
    ]
}

fn assert_bits_eq(a: f64, b: f64, cell: [usize; 2], name: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{name} at {cell:?}: fused={a:?} two_pass={b:?} (delta={:?})",
        a - b,
    );
}

// axis-0-fastest flat index over GRID, matching the harness/`Field` layout.
fn flat(c: &[usize]) -> usize {
    c[0] + c[1] * GRID[0]
}

fn body_oracle(a0: f64, ac: f64, stage: &str) {
    // uniform stage state + snapshot; zero flux (div = 0) so the godunov combine is a0*u_n + ac*cons
    // and the only nontrivial transform is the body wrap.
    let (rho, mom, nrg) = (1.5_f64, [0.3_f64, -0.2], 5.0_f64);
    let (rho_n, mom_n, nrg_n) = (1.4_f64, [0.25_f64, -0.15], 4.8_f64);

    let mut fields: Vec<(&str, f64)> = vec![
        ("rho", rho),
        ("mom_0", mom[0]),
        ("mom_1", mom[1]),
        ("nrg", nrg),
        ("u_n_rho", rho_n),
        ("u_n_mom_0", mom_n[0]),
        ("u_n_mom_1", mom_n[1]),
        ("u_n_nrg", nrg_n),
    ];
    let fnames = flux_names();
    for n in &fnames {
        fields.push((n.as_str(), 0.0));
    }

    let mut stage_scalars: Vec<(&str, f64)> = vec![("dt", DT), ("a0", a0), ("ac", ac)];
    stage_scalars.extend(geom_and_body_scalars());

    // FUSED: godunov + body welded into one kernel (n_bodies = 1, no user sources).
    let out_fused = KernelRun::new(godunov_stage_gv_with_fused_built(
        Coords::Cartesian, Spacetime::Minkowski, &[Spacing::Uniform; 2], &[0, 1], 2, 2, true,
        GeoSource::Hydro { inertial: true }, &[], false, 1,
    ))
    .grid(GRID)
    .compute_window(WLO, WSIZE)
    .fields(&fields)
    .scalars(&stage_scalars)
    .run();

    // TWO-PASS step 1: the plain godunov stage (n_bodies = 0).
    let out_god = KernelRun::new(godunov_stage_gv(
        Coords::Cartesian, Spacetime::Minkowski, &[Spacing::Uniform; 2], &[0, 1], 2, 2, true,
        GeoSource::Hydro { inertial: true },
    ))
    .grid(GRID)
    .compute_window(WLO, WSIZE)
    .fields(&fields)
    .scalars(&stage_scalars)
    .run();

    // TWO-PASS step 2: the standalone body pass, reading the godunov output. its `dt` scalar is the
    // SSP stage weight ac*dt — the same product the fused kernel forms internally as ac_dt.
    let den_buf: Vec<f64> = out_god.values("rho").to_vec();
    let m0_buf: Vec<f64> = out_god.values("mom_0").to_vec();
    let m1_buf: Vec<f64> = out_god.values("mom_1").to_vec();
    let nrg_buf: Vec<f64> = out_god.values("nrg").to_vec();

    let mut body_scalars: Vec<(&str, f64)> = vec![("dt", ac * DT)];
    body_scalars.extend(geom_and_body_scalars());

    let out_body = KernelRun::new(body_source_gv(1, Coords::Cartesian, 2, 2, &[0, 1]))
        .grid(GRID)
        .compute_window(WLO, WSIZE)
        .field_with("den", move |c| den_buf[flat(c)])
        .field_with("mom_0", move |c| m0_buf[flat(c)])
        .field_with("mom_1", move |c| m1_buf[flat(c)])
        .field_with("nrg", move |c| nrg_buf[flat(c)])
        .scalars(&body_scalars)
        .run();

    let mut body_changed = false;
    for j in 1..3 {
        for i in 1..3 {
            let cell = [i, j];
            assert_bits_eq(out_fused.get(cell, "rho"), out_body.get(cell, "den_new"), cell, "den");
            assert_bits_eq(out_fused.get(cell, "mom_0"), out_body.get(cell, "mom_0_new"), cell, "mom_0");
            assert_bits_eq(out_fused.get(cell, "mom_1"), out_body.get(cell, "mom_1_new"), cell, "mom_1");
            assert_bits_eq(out_fused.get(cell, "nrg"), out_body.get(cell, "nrg_new"), cell, "nrg");
            if out_fused.get(cell, "rho") != out_god.get(cell, "rho") {
                body_changed = true;
            }
        }
    }
    assert!(body_changed, "{stage}: body wrap was a no-op (drain + gravity inert) — oracle is vacuous");
}

#[test]
fn body_fused_equals_two_pass_euler() {
    body_oracle(0.0, 1.0, "euler");
}

#[test]
fn body_fused_equals_two_pass_rk2_corrector() {
    body_oracle(0.5, 0.5, "rk2");
}
