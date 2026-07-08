// =============================================================================
// dispatch_kernel_coverage.rs
//
// every CLI-reachable (regime x dimension x geometry x spacing) must have EVERY kernel family its
// substrate requests emitted in the AOT registry — not just the face-flux family that
// `solver_coverage_gate.rs` covers. per step the substrate builds, on a FLAT (Minkowski) metric:
//   - `{prefix}_godunov_stage{geom}{spacing}_{D}d`   (the conserved update)
//   - `{prefix}_wave_speed_map{geom}{spacing}_{D}d`  (the CFL bound)
//   - `rmhd_bcell_godunov_{euler,rk2}{geom}_{D}d`    (mhd out-of-plane B predictor, D < DOF)
// a `(dims, coords, spacing)` whose dispatch arm exists but whose kernel was never baked panics
// MID-RUN at `expect_kernel` (symbi-exec/layout.rs). this asserts each name the flat dispatch can
// request either resolves in the registry OR is a KNOWN_UNBAKED gap with a recorded reason — so a
// NEW gap (a chart that regresses out of the bake) fails here instead of at a user's first step.
//
// PREFIX MAP (the name protocol is per-family, not one slug per regime — the M5 fragility this
// gate pins): the godunov + c2p families use the regime prefix; the wave-speed family uses rhd->
// "rhd" but adiabatic->"iso" (Newtonian shares the isothermal cs map, substrate_newton.rs) and
// iso->"iso"; the mhd families use the regime prefix; the bcell predictor is always "rmhd_"
// (faraday induction is regime-agnostic).
//
// SCOPE: flat spacetime only. non-minkowski combinations are guarded arm-or-Err by the C4
// fail-loud guard (`test_dispatch_rejects_unbaked_gr`), so a GR gap surfaces as a rejection, not a
// silent-flat run or a panic; GR-kernel coverage across spacing/chart is a separate gate. the c2p
// family is geometry-light on a flat metric (rhd/adiabatic carry `{geom}` only; iso + flat-mhd c2p
// are bare `_c2p_{D}d`), so its gap surface is small and it is not enumerated here. a log-spaced
// CARTESIAN axis (no radial coordinate) is not a realized target and is not enumerated.
// =============================================================================

use symbi::regimes::substrate_kernels::{geom_suffix, kernel_exists, mhd_geom_suffix};
use symbi_geometry::Geometry;

// the flat spacing axis: uniform (empty tag) + log-radial (`_logr`, the geometric-mean curvilinear
// variant, matching `params.rs::spacing_suffix`). enumerated only on the curvilinear charts.
const UNIFORM: &str = "";
const LOGR: &str = "_logr";

// known-unbaked flat combos the dispatch can request that are DELIBERATELY not baked (rejected
// C4-style at dispatch, not silently run). empty: the flat gaps this gate first surfaced — 1D
// curvilinear MHD (`{r,n,i}mhd_*_{sph,cyl}_1d` + bcell), cylindrical hydro on a log-radial grid
// (`*_cyl_logr_*`), and the 2D spherical nmhd/imhd wave-speed (`{n,i}mhd_wave_speed_map_sph_2d`,
// a slug asymmetry that baked the dead `_sph_swirl_2d` instead) — are now all BAKED, so the two
// tests below assert their presence directly. a NEW gap fails the test until it is baked or listed.
const KNOWN_UNBAKED: &[&str] = &[];

fn assert_family(missing: &mut Vec<String>, name: String) {
    if !kernel_exists(&name) && !KNOWN_UNBAKED.contains(&name.as_str()) {
        missing.push(name);
    }
}

// the flat curvilinear charts a log-radial grid is realizable on (a radial axis 0 that can be
// log-spaced): spherical and cylindrical. cartesian has no radial coordinate.
fn logr_spacings(coords: Geometry) -> &'static [&'static str] {
    match coords {
        Geometry::Cartesian => &[UNIFORM],
        _ => &[UNIFORM, LOGR],
    }
}

// the hydro godunov + wave-speed families. on a flat metric every hydro dispatch arm has DOF == D
// (the azimuthal-momentum swirl lift, DOF > D, appears ONLY on the curved-spacetime arms), so the
// geometry suffix is coord-only (`geom_suffix(c, D, D)` = "" / "_sph" / "_cyl"). the godunov name
// uses the regime prefix; the CFL wave-speed name uses "rhd" for rhd and "iso" for both adiabatic
// (shares the cs map) and iso. rhd + adiabatic + iso reach cartesian / spherical / cylindrical in
// 1/2/3D.
#[test]
fn every_hydro_stage_and_cfl_kernel_is_emitted() {
    let mut missing = Vec::new();
    let charts = [Geometry::Cartesian, Geometry::Spherical, Geometry::Cylindrical];
    for (godunov_prefix, cfl_prefix) in [("rhd", "rhd"), ("adiabatic", "iso"), ("iso", "iso")] {
        for coords in charts {
            for d in 1..=3usize {
                let geom = geom_suffix(coords, d, d);
                for &sp in logr_spacings(coords) {
                    assert_family(&mut missing, format!("{godunov_prefix}_godunov_stage{geom}{sp}_{d}d"));
                    assert_family(&mut missing, format!("{cfl_prefix}_wave_speed_map{geom}{sp}_{d}d"));
                }
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

// the mhd gas godunov + wave-speed + out-of-plane bcell predictor. the geometry suffix keys on the
// grid-axis set (`mhd_geom_suffix`): 1D radial [0] -> "_cyl"/"_sph", 2D cyl r-z [0,2] ->
// "_cyl_rz" / r-phi [0,1] -> "_cyl_rphi", 3D [0,1,2] -> "_cyl". flat mhd does NOT append the
// spacing tag (only the GR branch does), so a log-radial flat mhd run silently reuses the uniform
// kernel — a wrong-geometry latent a kernel-existence probe cannot see; only uniform is asserted.
// the bcell predictor runs where an out-of-plane B component exists (D < DOF, DOF = 3), i.e. every
// 1D / 2D chart; it is always the regime-agnostic rmhd_* kernel.
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
        }
    }
    assert!(
        missing.is_empty(),
        "{} flat mhd godunov/wave-speed/bcell kernel(s) the dispatch can request are NOT emitted \
         and are not in KNOWN_UNBAKED (mid-run expect_kernel panic):\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
