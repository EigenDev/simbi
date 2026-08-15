// =============================================================================
// dispatch_kernel_coverage.rs
//
// every CLI-reachable (regime x dimension x geometry) must have EVERY kernel family its substrate
// requests emitted in the AOT registry, spanning past the face-flux family alone. per step the
// substrate builds, on a FLAT (Minkowski) metric:
//   - `{prefix}_godunov_stage{geom}_{D}d`   (the conserved update)
//   - `{prefix}_wave_speed_map{geom}_{D}d`  (the CFL bound)
//   - `rmhd_bcell_godunov_{euler,rk2}{geom}_{D}d`   (mhd out-of-plane B predictor, D < DOF)
//   - `rmhd_ct_curl_2d_{dir}{geom}`                 (mhd 2D in-plane metric CT curl)
// a `(dims, coords)` whose dispatch arm exists but whose kernel was never baked panics MID-RUN at
// `expect_kernel` (symbi-exec/layout.rs). this asserts each name the flat dispatch can request either
// resolves in the registry OR is a KNOWN_UNBAKED gap with a recorded reason — so a NEW gap (a chart
// that regresses out of the bake) fails at this gate ahead of a user's first step.
//
// SPACING IS NOT IN THE NAME: it is a per-axis RUNTIME kernel scalar (`map_kind_{ax}`: 0 uniform,
// 1 log), so ONE kernel per (regime, geometry) serves uniform, log-radial, log-theta, ... — there is
// no `_logr` name axis to enumerate. the log paths are exercised by physics gates (gr_bondi,
// mhd_cyl_rz_logr, michel-logr).
//
// PREFIX MAP (the name protocol is per-family, each family picking its own prefix): the godunov + c2p families
// use the regime prefix; the wave-speed family uses rhd->"rhd" but adiabatic->"iso" (Newtonian
// shares the isothermal cs map) and iso->"iso"; the mhd families use the regime prefix; the bcell
// predictor + ct curl are always "rmhd_" (faraday induction is regime-agnostic).
//
// SCOPE: flat spacetime only. non-minkowski combinations are guarded arm-or-Err
// (`test_dispatch_rejects_unbaked_gr`). the c2p family is geometry-light on a flat metric, so it is
// enumerated by the curved-family gates instead.
// =============================================================================

use symbi::regimes::substrate_kernels::{geom_suffix, kernel_exists, mhd_geom_suffix};
use symbi_geometry::Geometry;

// known-unbaked flat combos the dispatch can request that are DELIBERATELY not baked (they fail
// loud at dispatch). the list is empty: every flat combination the dispatch can request — including
// 1D curvilinear MHD (spherical + cylindrical), 2D spherical nmhd/imhd wave speeds, and the
// log-radial charts (spacing is a runtime scalar, so they share the uniform kernel) — is baked. a
// NEW gap fails the test until it is baked or listed here.
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
    let charts = [
        Geometry::Cartesian,
        Geometry::Spherical,
        Geometry::Cylindrical,
    ];
    for (godunov_prefix, cfl_prefix) in [("rhd", "rhd"), ("adiabatic", "iso"), ("iso", "iso")] {
        for coords in charts {
            for d in 1..=3usize {
                let geom = geom_suffix(coords, d, d);
                assert_family(
                    &mut missing,
                    format!("{godunov_prefix}_godunov_stage{geom}_{d}d"),
                );
                assert_family(
                    &mut missing,
                    format!("{cfl_prefix}_wave_speed_map{geom}_{d}d"),
                );
            }
        }
    }
    // the taub-mathews eos axis: `.with_eos(TaubMathews)` on the rhd set selects
    // `_tm` twins for the c2p + wave-speed map, baked flat-cartesian only (the
    // dispatch refuses every other chart before a name is formed).
    for d in 1..=3usize {
        assert_family(&mut missing, format!("rhd_wave_speed_map_tm_{d}d"));
        assert_family(&mut missing, format!("rhd_c2p_tm_{d}d"));
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

// the passive-scalar (dye) family: dimension-only names — no chart suffix, the
// dye rides the cartesian mass flux — requested by the newtonian chi_update
// dispatch for every grid dimension. enumerated here so a future chart- or
// regime-suffixed chi kernel cannot silently narrow this gate's completeness.
#[test]
fn chi_family_is_baked_for_every_dimension() {
    for d in 1..=3u8 {
        for name in [
            format!("chi_godunov_{d}d"),
            format!("chi_c2p_{d}d"),
            format!("chi_snapshot_{d}d"),
        ] {
            assert!(
                symbi_aot::kernel_by_name::<f64>(&name).is_some(),
                "missing baked passive-scalar kernel '{name}'"
            );
        }
    }
}

// =============================================================================
// the CURVED-spacetime families.
//
// the gates above are FLAT-only, and they enumerate the four families a flat step
// requests. the curved-only families — the wu 2017 source-admissibility CFL
// and the admissible-boundary projection — are enumerated here: they appear on no flat
// arm, so nothing above reaches them.
//
// the failure mode is a chart-segment disagreement between the dispatch and the bake. MHD
// carries no momentum-DOF lift, which invites an EMPTY chart segment on the dispatch side,
// while the bake names the kernel with the grid-axis chart segment. the two then agree only
// on cartesian (segment empty either way) and diverge on every curvilinear chart, so the
// spherical GRMHD projection panics the first time a cell needs it, in 1D and 2D alike. a
// name divergence must fail HERE, at a gate that builds both sides, ahead of a physics test
// whose failure names a missing kernel instead of a bug.
//
// the arms mirror the curved match arms of `hydro_dispatch!` / `mhd_dispatch!` in
// symbi-py: those are the (dimension, chart, spacetime) combinations a config can select.
// =============================================================================

use symbi::regimes::substrate_kernels::{fofc_project_name, spacetime_slug};
use symbi_discretize::kernel_slug::{ChartKeying, fofc_project_chart};
use symbi_geometry::Spacetime;

/// one CLI-reachable curved dispatch arm: the grid shape and background a config selects.
/// `dof` is the momentum-component count — the spherical GR arms lift the azimuthal
/// component (DOF = 3 on a 2-axis grid), which is what makes the hydro chart segment
/// `_sph_swirl` rather than `_sph`.
struct CurvedArm {
    dims: usize,
    coords: Geometry,
    axes: &'static [usize],
    dof: usize,
    spacetime: Spacetime,
}

const fn arm(
    dims: usize,
    coords: Geometry,
    axes: &'static [usize],
    dof: usize,
    spacetime: Spacetime,
) -> CurvedArm {
    CurvedArm {
        dims,
        coords,
        axes,
        dof,
        spacetime,
    }
}

/// the curved GRHD arms of `hydro_dispatch!`.
fn curved_hydro_arms() -> Vec<CurvedArm> {
    use Geometry::{Cartesian, Cylindrical, Spherical};
    use Spacetime::{KerrKS, SchwarzschildKS};
    vec![
        arm(1, Spherical, &[0], 1, SchwarzschildKS),
        // the 2D spherical wedge carries the azimuthal momentum lift.
        arm(2, Spherical, &[0, 1], 3, SchwarzschildKS),
        arm(2, Spherical, &[0, 1], 3, KerrKS),
        arm(2, Cartesian, &[0, 1], 2, SchwarzschildKS),
        arm(3, Cartesian, &[0, 1, 2], 3, SchwarzschildKS),
        arm(2, Cartesian, &[0, 1], 2, KerrKS),
        arm(3, Cartesian, &[0, 1, 2], 3, KerrKS),
        arm(2, Cylindrical, &[0, 2], 3, SchwarzschildKS),
        arm(3, Cylindrical, &[0, 1, 2], 3, SchwarzschildKS),
        arm(3, Cylindrical, &[0, 1, 2], 3, KerrKS),
    ]
}

/// the curved GRMHD arms of `mhd_dispatch!`. MHD momentum is always a 3-vector, so the
/// chart segment keys on the grid-axis set rather than on the DOF lift.
fn curved_mhd_arms() -> Vec<CurvedArm> {
    use Geometry::{Cartesian, Spherical};
    use Spacetime::{KerrKS, SchwarzschildKS};
    vec![
        arm(1, Spherical, &[0], 3, SchwarzschildKS),
        arm(2, Spherical, &[0, 1], 3, KerrKS),
        arm(2, Cartesian, &[0, 1], 3, SchwarzschildKS),
        arm(3, Cartesian, &[0, 1, 2], 3, SchwarzschildKS),
        arm(2, Cartesian, &[0, 1], 3, KerrKS),
        arm(3, Cartesian, &[0, 1, 2], 3, KerrKS),
    ]
}

#[test]
fn every_curved_admissibility_kernel_is_emitted() {
    let mut missing = Vec::new();

    // GRHD: the chart segment comes from the SAME derivation the dispatch runs. calling
    // `geom_suffix` directly here would re-spell it, and a gate that re-spells the thing it
    // is checking passes whatever the dispatch does — including an empty segment on both
    // sides.
    for a in curved_hydro_arms() {
        let chart = fofc_project_chart(ChartKeying::MomentumDof, a.coords, a.axes, a.dof, a.dims);
        assert_family(
            &mut missing,
            fofc_project_name("rhd", chart, a.spacetime, a.dims),
        );
        assert_family(
            &mut missing,
            format!(
                "rhd_source_cfl{chart}{}_{}d",
                spacetime_slug(a.spacetime),
                a.dims
            ),
        );
    }

    // GRMHD: same derivation, keyed on the grid-axis set.
    for a in curved_mhd_arms() {
        let chart = fofc_project_chart(ChartKeying::GridAxes, a.coords, a.axes, a.dof, a.dims);
        assert_family(
            &mut missing,
            fofc_project_name("rmhd", chart, a.spacetime, a.dims),
        );
        assert_family(
            &mut missing,
            format!(
                "rmhd_source_cfl{chart}{}_{}d",
                spacetime_slug(a.spacetime),
                a.dims
            ),
        );
    }

    assert!(
        missing.is_empty(),
        "{} curved-spacetime kernel(s) the dispatch can request are NOT emitted and are not \
         in KNOWN_UNBAKED (mid-run expect_kernel panic):\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

// the GRMHD HLL face flux does not compute its own fan: it READS `wave_speed_l[dir]` and
// `wave_speed_r[dir]` from the two cells sharing each face (the davis estimate), and a separate
// per-cell pass materializes them. that producer/consumer coupling is invisible in the names, and
// its failure mode is silent rather than loud: with the producer absent the flux reads the fields'
// ZERO initialization, so both fan speeds collapse onto the shift and every axis whose shift
// component vanishes loses its dissipation entirely. the sweep on that axis becomes one-sided and
// odd-even decoupled — no crash, no missing-kernel panic, just a smooth stationary state growing a
// grid-scale checkerboard over tens of dynamical times.
//
// so assert BOTH halves: the producer exists for every curved family whose HLL flux exists, and the
// flux really is a consumer (otherwise the pairing is vacuous and would keep passing if the flux
// were later changed to compute its speeds inline).
#[test]
fn every_curved_hll_flux_has_the_wave_speeds_it_reads() {
    use symbi::regimes::substrate_kernels::kernel_bindings;
    use symbi_ir::FieldRef;

    let mut missing = Vec::new();
    let mut non_consumers = Vec::new();
    for a in curved_mhd_arms() {
        let chart = mhd_geom_suffix(a.coords, a.axes);
        let st = spacetime_slug(a.spacetime);
        let dims = a.dims;
        // the HLL arm carries no solver tag; the HLLD arm solves its own five-wave fan and the
        // rusanov fallback uses the state-independent light-cone bound, so neither is a consumer.
        let flux = format!("rmhd_face_flux{chart}{st}_{dims}d_0");
        if !kernel_exists(&flux) {
            continue;
        }
        let reads = kernel_bindings(&flux).iter().any(|(field, is_output)| {
            !is_output && matches!(field, FieldRef::WaveSpeedL(_) | FieldRef::WaveSpeedR(_))
        });
        if !reads {
            non_consumers.push(flux.clone());
        }
        assert_family(
            &mut missing,
            format!("rmhd_wave_speeds_cell{chart}{st}_{dims}d"),
        );
    }

    assert!(
        non_consumers.is_empty(),
        "the curved HLL flux no longer reads the materialized per-cell wave speeds, so this gate \
         asserts nothing: {}",
        non_consumers.join(", ")
    );
    assert!(
        missing.is_empty(),
        "{} curved GRMHD family/families bake an HLL face flux with NO per-cell wave-speed \
         producer. the flux would read zeros and run with no dissipation:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

#[test]
fn the_projection_name_is_built_once_for_both_sides() {
    // the bake and the dispatch must not be able to spell this kernel two ways. they call
    // ONE builder, so the chart segment and the spacetime slug appear in one order, in one
    // place. this pins the resulting shape so a change to it is a deliberate edit rather
    // than a silent divergence that only shows up on one chart.
    assert_eq!(
        fofc_project_name("rmhd", "_sph", Spacetime::SchwarzschildKS, 2),
        "rmhd_fofc_project_sph_ks_2d"
    );
    // cartesian's chart segment is empty, so an empty segment agrees with the bake there
    // while breaking every curvilinear chart.
    assert_eq!(
        fofc_project_name("rmhd", "", Spacetime::SchwarzschildKS, 3),
        fofc_project_name("rmhd", "", Spacetime::SchwarzschildKS, 3),
    );
    assert_ne!(
        fofc_project_name("rmhd", "", Spacetime::SchwarzschildKS, 2),
        fofc_project_name("rmhd", "_sph", Spacetime::SchwarzschildKS, 2),
        "the chart segment must change the name, else the dispatch cannot select a chart"
    );
}

// =============================================================================
// the SOLVER matrix. `Solver::valid_for(regime)` is what a config is checked against, so every pair
// it accepts is a pair a user can select — and selecting one whose face-flux kernel was never baked
// panics at `expect_kernel` on the first step, after the queue slot is spent.
//
// nothing else covers this. the families above enumerate (regime x dimension x geometry) at the
// DEFAULT solver; widening the matrix to admit a new (solver, regime) pair leaves them all green.
// =============================================================================

#[test]
fn every_solver_the_matrix_accepts_has_its_face_flux_baked() {
    use symbi_sim::substrate_seam::{RegimeKind, Solver};

    // the flux-name prefix each regime family uses, and the dimensions its dispatch is built for.
    // cartesian only: the HLLC family is unbaked on curvilinear charts by design, and the matrix is
    // not what gates that (the substrate's own dispatch arm is).
    let regimes = [
        (RegimeKind::Newtonian, "adiabatic", &[1u8, 2, 3][..]),
        (RegimeKind::IsoNewtonian, "iso", &[1, 2, 3][..]),
        (RegimeKind::Rhd, "rhd", &[1, 2, 3][..]),
        (RegimeKind::Rmhd, "rmhd", &[1, 3][..]),
        (RegimeKind::NewtonianMhd, "nmhd", &[1, 3][..]),
    ];
    // ENUMERATE FROM THE TYPE. a hand-written array is what let `HllcAcoustic` ship unswept:
    // this gate's own header claims "widening the matrix to admit a new (solver, regime) pair
    // leaves them all green", and that is exactly what happened.
    let solvers = Solver::ALL;

    let mut missing = Vec::new();
    let mut checked = 0usize;
    for (regime, prefix, dims) in regimes {
        for solver in solvers {
            if !solver.valid_for(regime) {
                continue;
            }
            for &ndim in dims {
                for dir in 0..ndim {
                    // the reconstruction axis: the ppm twin exists for every solver the
                    // NEWTONIAN matrix accepts (the runtime `.reconstruction(Ppm)` composes
                    // with any adiabatic solver); every other regime is plm-only and its
                    // dispatch refuses ppm before a name is formed.
                    let recons: &[symbi_discretize::Recon] = if regime == RegimeKind::Newtonian {
                        &[symbi_discretize::Recon::Plm, symbi_discretize::Recon::Ppm]
                    } else {
                        &[symbi_discretize::Recon::Plm]
                    };
                    // the eos axis: the taub-mathews `_tm` flux twin exists for every
                    // solver the RHD matrix accepts; every other regime is gamma-law
                    // only and its dispatch refuses the tm arm before a name is formed.
                    let eoses: &[symbi_discretize::EosArm] = if regime == RegimeKind::Rhd {
                        &[
                            symbi_discretize::EosArm::IdealGamma,
                            symbi_discretize::EosArm::TaubMathews,
                        ]
                    } else {
                        &[symbi_discretize::EosArm::IdealGamma]
                    };
                    // the BALANCE axis, and the CHART axis that rides with it. a well-balanced
                    // reconstruction reads cartesian positions, so it is baked per chart while
                    // every plain flux stays chart-agnostic -- and this gate spelled the name
                    // itself, with no chart segment at all, which is how twelve curvilinear
                    // kernels were baked under a name the dispatch could never ask for.
                    // baked for the newtonian arms that carry it: HLLE (the first-order redo)
                    // and the clamp-free low-mach arm.
                    let balances: &[symbi_discretize::coords::Balance] = if regime
                        == RegimeKind::Newtonian
                        && matches!(solver, Solver::Hlle | Solver::HllcLm)
                    {
                        &[
                            symbi_discretize::coords::Balance::Plain,
                            symbi_discretize::coords::Balance::Hydrostatic,
                        ]
                    } else {
                        &[symbi_discretize::coords::Balance::Plain]
                    };
                    for &recon in recons {
                        for &eos in eoses {
                            for &balance in balances {
                                // the redo arm is baked plm-only; ppm has no first-order twin.
                                if balance == symbi_discretize::coords::Balance::Hydrostatic
                                    && *solver == Solver::Hlle
                                    && recon != symbi_discretize::Recon::Plm
                                {
                                    continue;
                                }
                                let charts: &[symbi_geometry::Geometry] = match balance {
                                    symbi_discretize::coords::Balance::Plain => {
                                        &[symbi_geometry::Geometry::Cartesian]
                                    }
                                    symbi_discretize::coords::Balance::Hydrostatic => &[
                                        symbi_geometry::Geometry::Cartesian,
                                        symbi_geometry::Geometry::Cylindrical,
                                        symbi_geometry::Geometry::Spherical,
                                    ],
                                };
                                for &chart in charts {
                                    checked += 1;
                                    // through the SAME composer the bake and the dispatch use.
                                    let name = symbi_discretize::kernel_slug::FaceFluxName {
                                        prefix,
                                        solver: solver.kernel_suffix(),
                                        recon: recon.suffix(),
                                        balance: balance.suffix(),
                                        chart: symbi_discretize::kernel_slug::coord_suffix(chart),
                                        eos: eos.suffix(),
                                        ndim: ndim as usize,
                                        dir: dir as usize,
                                        ..Default::default()
                                    }
                                    .build();
                                    if !kernel_exists(&name)
                                        && !KNOWN_UNBAKED.contains(&name.as_str())
                                    {
                                        missing.push(format!(
                                            "{solver:?}/{recon:?}/{eos:?}/{balance:?}/{chart:?} \
                                             on {regime:?}: {name}"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // the premise: the enumeration must actually reach the interesting pairs. a `valid_for` that
    // rejected everything, or a name protocol that drifted, would leave `missing` empty and this
    // gate silently vacuous.
    assert!(
        // MEASURED, not guessed: the sweep over
        // (regime x solver x dim x dir x recon x eos x balance x chart) is exactly 168
        // combinations today -- it was 180 until 2026-08-15, when the clamped hllc_lm
        // variant was retired and its solver row left the matrix (the surviving hllc_lm
        // carries the balance x chart arms the retired name introduced). the floor sits AT
        // the measurement so any silent collapse of any axis fails here; move it only with
        // a deliberate matrix change, recorded like this one.
        checked >= 168,
        "only {checked} (solver, regime, dim, dir) combination(s) were checked; the matrix or the          name protocol has drifted and this gate is not covering anything"
    );
    assert!(
        missing.is_empty(),
        "the solver matrix accepts {} pair(s) whose face-flux kernel is not baked. selecting one          panics at dispatch on the first step:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
