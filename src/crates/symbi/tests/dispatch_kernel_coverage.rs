// =============================================================================
// dispatch_kernel_coverage.rs
//
// every CLI-reachable (regime x dimension x geometry) must have EVERY kernel family its substrate
// requests emitted in the AOT registry — not just the face-flux family that `solver_coverage_gate.rs`
// covers. per step the substrate builds, on a FLAT (Minkowski) metric:
//   - `{prefix}_godunov_stage{geom}_{D}d`   (the conserved update)
//   - `{prefix}_wave_speed_map{geom}_{D}d`  (the CFL bound)
//   - `rmhd_bcell_godunov_{euler,rk2}{geom}_{D}d`   (mhd out-of-plane B predictor, D < DOF)
//   - `rmhd_ct_curl_2d_{dir}{geom}`                 (mhd 2D in-plane metric CT curl)
// a `(dims, coords)` whose dispatch arm exists but whose kernel was never baked panics MID-RUN at
// `expect_kernel` (symbi-exec/layout.rs). this asserts each name the flat dispatch can request either
// resolves in the registry OR is a KNOWN_UNBAKED gap with a recorded reason — so a NEW gap (a chart
// that regresses out of the bake) fails here instead of at a user's first step.
//
// SPACING IS NOT IN THE NAME: it is a per-axis RUNTIME kernel scalar (`map_kind_{ax}`: 0 uniform,
// 1 log), so ONE kernel per (regime, geometry) serves uniform, log-radial, log-theta, ... — there is
// no `_logr` name axis to enumerate. the log paths are exercised by physics gates (gr_bondi,
// mhd_cyl_rz_logr, michel-logr), not by name here.
//
// PREFIX MAP (the name protocol is per-family, not one slug per regime): the godunov + c2p families
// use the regime prefix; the wave-speed family uses rhd->"rhd" but adiabatic->"iso" (Newtonian
// shares the isothermal cs map) and iso->"iso"; the mhd families use the regime prefix; the bcell
// predictor + ct curl are always "rmhd_" (faraday induction is regime-agnostic).
//
// SCOPE: flat spacetime only. non-minkowski combinations are guarded arm-or-Err by the C4 fail-loud
// guard (`test_dispatch_rejects_unbaked_gr`). the c2p family is geometry-light on a flat metric and
// is not enumerated here.
// =============================================================================

use symbi::regimes::substrate_kernels::{geom_suffix, kernel_exists, mhd_geom_suffix};
use symbi_geometry::Geometry;

// known-unbaked flat combos the dispatch can request that are DELIBERATELY not baked (they fail
// loud at dispatch rather than running silently). empty: every flat gap this gate surfaced — 1D
// curvilinear MHD (spherical + cylindrical), the 2D spherical nmhd/imhd wave-speed slug bug, and
// (formerly) the log-radial charts — is now baked / covered (spacing is runtime, not a name). a NEW
// gap fails the test until it is baked or listed here.
const KNOWN_UNBAKED: &[&str] = &[];

fn assert_family(missing: &mut Vec<String>, name: String) {
    if !kernel_exists(&name) && !KNOWN_UNBAKED.contains(&name.as_str()) {
        missing.push(name);
    }
}

// the hydro godunov + wave-speed families. on a flat metric every hydro dispatch arm has DOF == D
// (the azimuthal-momentum swirl lift, DOF > D, appears ONLY on the curved-spacetime arms), so the
// geometry suffix is coord-only (`geom_suffix(c, D, D)` = "" / "_sph" / "_cyl"). the godunov name
// uses the regime prefix; the CFL wave-speed name uses "rhd" for rhd and "iso" for both adiabatic
// (shares the cs map) and iso. rhd + adiabatic + iso reach cartesian / spherical / cylindrical 1-3D.
#[test]
fn every_hydro_stage_and_cfl_kernel_is_emitted() {
    let mut missing = Vec::new();
    let charts = [Geometry::Cartesian, Geometry::Spherical, Geometry::Cylindrical];
    for (godunov_prefix, cfl_prefix) in [("rhd", "rhd"), ("adiabatic", "iso"), ("iso", "iso")] {
        for coords in charts {
            for d in 1..=3usize {
                let geom = geom_suffix(coords, d, d);
                assert_family(&mut missing, format!("{godunov_prefix}_godunov_stage{geom}_{d}d"));
                assert_family(&mut missing, format!("{cfl_prefix}_wave_speed_map{geom}_{d}d"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} flat hydro godunov/wave-speed kernel(s) the dispatch can request are NOT emitted and \
         are not in KNOWN_UNBAKED (mid-run expect_kernel panic):\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

// the mhd gas godunov + wave-speed + out-of-plane bcell predictor + 2D in-plane CT curl. the
// geometry suffix keys on the grid-axis set (`mhd_geom_suffix`): 1D radial [0] -> "_cyl"/"_sph",
// 2D cyl r-z [0,2] -> "_cyl_rz" / r-phi [0,1] -> "_cyl_rphi", 3D [0,1,2] -> "_cyl". the bcell
// predictor runs where an out-of-plane B component exists (D < DOF, DOF = 3), i.e. every 1D / 2D
// chart; it + the CT curl are always the regime-agnostic rmhd_* kernel.
#[test]
fn every_mhd_stage_cfl_and_bcell_kernel_is_emitted() {
    // (coords, dims, grid-axis set) for every flat mhd dispatch arm (mhd_dispatch! in
    // symbi-py/src/lib.rs): cartesian / spherical 1-3D + cylindrical 1D, 2D r-z, 2D r-phi, 3D.
    let arms: &[(Geometry, usize, &[usize])] = &[
        (Geometry::Cartesian, 1, &[0]),
        (Geometry::Cartesian, 2, &[0, 1]),
        (Geometry::Cartesian, 3, &[0, 1, 2]),
        (Geometry::Spherical, 1, &[0]),
        (Geometry::Spherical, 2, &[0, 1]),
        (Geometry::Spherical, 3, &[0, 1, 2]),
        (Geometry::Cylindrical, 1, &[0]),
        (Geometry::Cylindrical, 2, &[0, 2]),
        (Geometry::Cylindrical, 2, &[0, 1]),
        (Geometry::Cylindrical, 3, &[0, 1, 2]),
    ];
    let mut missing = Vec::new();
    for prefix in ["rmhd", "nmhd", "imhd"] {
        for &(coords, d, axes) in arms {
            let geom = mhd_geom_suffix(coords, axes);
            assert_family(&mut missing, format!("{prefix}_godunov_stage{geom}_{d}d"));
            assert_family(&mut missing, format!("{prefix}_wave_speed_map{geom}_{d}d"));
            // the out-of-plane predictor exists only where D < DOF (DOF = 3): the 1D / 2D charts.
            if d < 3 {
                assert_family(&mut missing, format!("rmhd_bcell_godunov_euler{geom}_{d}d"));
                assert_family(&mut missing, format!("rmhd_bcell_godunov_rk2{geom}_{d}d"));
            }
            // the in-plane metric CT curl is a 2D-curvilinear-only regime-agnostic kernel (the 1/r
            // curl reads the face radii). cartesian's curl is metric-free; 1D/3D take a different
            // CT structure.
            if d == 2 && prefix == "rmhd" && coords != Geometry::Cartesian {
                for dir in 0..2 {
                    assert_family(&mut missing, format!("rmhd_ct_curl_2d_{dir}{geom}"));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} flat mhd godunov/wave-speed/bcell/ct-curl kernel(s) the dispatch can request are NOT \
         emitted and are not in KNOWN_UNBAKED (mid-run expect_kernel panic):\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
