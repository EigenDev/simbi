// =============================================================================
// fusion_godunov_bcell.rs
//
// **fusion-algebra phase A diagnostic**: take the two real builders that the
// rmhd substrate currently launches back-to-back inside `god_stage` —
// `godunov_stage_gv` (cons update) and `rmhd_bcell_godunov_*_gv` (bcell
// update) — tag both with the same canonical 3D interior launch grade, and
// call `try_fuse`. assert the algebra ACCEPTS the pair.
//
// the goal here is binary: either the algebra accepts the fusion (in which
// case phase B can wire the fused kernel into build.rs + the substrate
// dispatcher and we land a real perf win), OR it rejects with `InterDep` /
// `WriteConflict`, which is the architectural diagnostic — a hidden data
// hazard that must be untangled before the substrate can issue one launch
// in place of two.
//
// run: cargo test -p symbi-discretize --test fusion_godunov_bcell
// =============================================================================

use symbi_algebra::{domain, Domain, Space};
use symbi_discretize::coords::{Coords, Spacing};
use symbi_discretize::gv::{godunov_stage_gv, rmhd_bcell_godunov_euler_gv, rmhd_bcell_godunov_rk2_gv, GeoSource};
use symbi_ir::gv::{try_fuse, FusionError, LaunchGrade};

// canonical cell-centered 3D interior launch shape. structural, not numeric:
// build-time fusion only needs both kernels to declare the same `LaunchGrade`
// fingerprint. runtime supplies the concrete extents at dispatch time.
fn interior_3d() -> Domain<3> {
    domain([
        Space { name: "i", lo: 0, hi: 1 },
        Space { name: "j", lo: 0, hi: 1 },
        Space { name: "k", lo: 0, hi: 1 },
    ])
}

#[test]
fn godunov_plus_bcell_euler_fuses() {
    // RMHD 3D Cartesian, has_energy=true, 3 momentum components — the shape the
    // substrate's `god_stage` issues for mhd evolution.
    let (mut k_god, w_god) = godunov_stage_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
        3,
        true,
        GeoSource::Rmhd,
    );
    let (mut k_bcell, w_bcell) = rmhd_bcell_godunov_euler_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; 3],
        3, 3, &[0, 1, 2],
    );

    // both built by `end_trace()` (currently untagged). retag to the canonical
    // 3D interior grade so the algebra is willing to consider the fusion.
    let grade = LaunchGrade::from_domain(&interior_3d());
    k_god.grade = grade.clone();
    k_bcell.grade = grade;

    let (fused, fused_w) = match try_fuse(k_god, w_god.clone(), k_bcell, w_bcell.clone()) {
        Ok(r) => r,
        Err(e) => panic!(
            "the algebra REJECTED godunov + bcell_godunov_euler: {e}\n\
             this is the architectural diagnostic — the two kernels can't be \
             fused without untangling the rejected hazard.",
        ),
    };

    // write set: union, disjoint. the cons writes (rho/mom_k/nrg → cons.*) +
    // the 3 bcell writes (bc_c → bc_c). 4 (with energy) + 3 + ncomp(=3) = 7.
    let write_paths: Vec<String> = fused_w.iter().map(|(_, p, _)| p.name()).collect();
    assert!(write_paths.iter().any(|p| p == "cons.den"), "fused must write cons.den, got {write_paths:?}");
    assert!(write_paths.iter().any(|p| p == "cons.nrg"), "fused must write cons.nrg, got {write_paths:?}");
    for k in 0..3 {
        let want = format!("cons.mom_{k}");
        assert!(write_paths.iter().any(|p| *p == want),
            "fused must write {want}, got {write_paths:?}");
    }
    for c in 0..3 {
        let want = format!("bc_{c}");
        assert!(write_paths.iter().any(|p| *p == want),
            "fused must write {want}, got {write_paths:?}");
    }

    // input manifest: union of both sides, deduped. godunov's `cons.*`/`u_n.*`/
    // flux fields + bcell's `bc_*` + `bf_d_c`. spot-check.
    let input_paths: Vec<String> = fused.field_inputs.iter().map(|(_, p)| p.name()).collect();
    assert!(input_paths.iter().any(|p| p == "cons.den"), "fused must read cons.den");
    for c in 0..3 {
        let want = format!("bc_{c}");
        assert!(input_paths.iter().any(|p| *p == want),
            "fused must read {want}, got {input_paths:?}");
    }
    for d in 0..3 {
        for c in 0..3 {
            let want = format!("bf_{d}_{c}");
            assert!(input_paths.iter().any(|p| *p == want),
                "fused must read {want}, got {input_paths:?}");
        }
    }

    // shared scalars (`dt`) collapse to one entry — `add_param` dedup by Symbol.
    let dt_count = fused.scalar_params.iter().filter(|s| s.as_str() == "dt").count();
    assert_eq!(dt_count, 1, "shared scalar `dt` must dedupe in the fused manifest");
}

#[test]
fn godunov_plus_bcell_rk2_fuses() {
    // identical to the euler case but using the rk2 bcell variant. validates the
    // algebra is variant-agnostic — both euler and rk2 stages compose with the
    // same cons update.
    let (mut k_god, w_god) = godunov_stage_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; 3],
        &[0, 1, 2],
        3,
        3,
        true,
        GeoSource::Rmhd,
    );
    let (mut k_bcell, w_bcell) = rmhd_bcell_godunov_rk2_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; 3],
        3, 3, &[0, 1, 2],
    );

    let grade = LaunchGrade::from_domain(&interior_3d());
    k_god.grade = grade.clone();
    k_bcell.grade = grade;

    let result = try_fuse(k_god, w_god, k_bcell, w_bcell);
    assert!(result.is_ok(),
        "the algebra REJECTED godunov + bcell_godunov_rk2: {:?}", result.err());
}

#[test]
fn untagged_pair_is_rejected() {
    // the symmetric guard: if either side is untagged (the default `end_trace()`
    // output), the algebra MUST refuse to fuse — opt-in is the only way to
    // promise the kernel will be dispatched over a known grade.
    let (k_god, w_god) = godunov_stage_gv(
        Coords::Cartesian, &[Spacing::Uniform; 3], &[0, 1, 2], 3, 3, true, GeoSource::Rmhd,
    );
    let (k_bcell, w_bcell) = rmhd_bcell_godunov_euler_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; 3],
        3, 3, &[0, 1, 2],
    );

    match try_fuse(k_god, w_god, k_bcell, w_bcell) {
        Err(FusionError::UntaggedKernel) => {}
        other => panic!("expected UntaggedKernel for default end_trace() output, got {other:?}"),
    }
}
