// =============================================================================
// godunov_with_fused_source.rs
//
// proves `godunov_euler_gv_with_fused_source` produces a
// single kernel that combines the flux-divergence integrator + a spec-driven
// user momentum source (e.g., `uniform_acceleration_sources`). validates:
//
//   - structural — with `user_source = None` the fused builder reproduces
//     `godunov_euler_gv` exactly; same Writes shape;
//   - semantic — for a uniform state (zero flux divergence), running the
//     fused kernel with `uniform_acceleration` as the user source produces
//     `mom_k_new = mom_k + dt * rho * g_ext_k`, exactly the analytical
//     newtonian response to a uniform external force, so the spliced spec
//     contribution reads the register-resident state at the right point.
//
// performance motivation: a newtonian + external-acceleration
// problem runs two kernels per RK stage (godunov + body_source). the spec
// source rides inside godunov instead — one launch, one set of
// register-resident `cons` reads, one fused CSE pass over the divergence +
// source expressions. zero added kernel dispatch overhead.
//
// run: cargo test -p symbi-discretize --test godunov_with_fused_source
// =============================================================================

mod harness;

use harness::KernelRun;
use symbi_discretize::coords::{Coords, Spacetime, Spacing};
use symbi_discretize::gv::{
    GeoSource, godunov_stage_gv, godunov_stage_gv_with_fused_sources, source_apply_gv,
};

// the godunov-stage kernel reads the u_n snapshot + the (a0, ac) SSP coefficients. these tests
// exercise the source fusion + the analytical forward-euler update, so they run at the euler
// stage (a0=0, ac=1): the `a0*u_n` term drops out, so the bound u_n values are immaterial.
const EULER_AC: [(&str, f64); 2] = [("a0", 0.0), ("ac", 1.0)];

// u_n snapshot field bindings for a newtonian 3D (ncomp=3 + energy) state — at the euler stage
// these are multiplied by a0=0, but the kernel still binds them, so the harness must supply them.
fn u_n_fields(rho: f64, mom: [f64; 3], nrg: f64) -> Vec<(&'static str, f64)> {
    vec![
        ("u_n_rho", rho),
        ("u_n_mom_0", mom[0]),
        ("u_n_mom_1", mom[1]),
        ("u_n_mom_2", mom[2]),
        ("u_n_nrg", nrg),
    ]
}
use symbi_hydro::source_spec::uniform_acceleration_sources;

#[test]
fn user_source_none_matches_writes_of_plain_godunov() {
    // structural equivalence: passing `None` for `user_source` produces a kernel with
    // the same Writes shape as the plain godunov builder, so a caller passing `None`
    // binds exactly the same ABI. (the check is Writes-shape equivalence, the contract
    // the runtime binds against; NodeId identity differs between two `begin_trace`
    // sessions.)
    let coords = Coords::Cartesian;
    let spacing = vec![Spacing::Uniform; 3];
    let axes = vec![0, 1, 2];
    let (_k_plain, w_plain) = godunov_stage_gv(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        3,
        3,
        true,
        GeoSource::Hydro { inertial: false },
    );
    let (_k_fused, w_fused) = godunov_stage_gv_with_fused_sources(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        3,
        3,
        true,
        GeoSource::Hydro { inertial: false },
        &[],
        false,
    );
    let names_plain: Vec<&str> = w_plain.iter().map(|write| write.key.as_str()).collect();
    let names_fused: Vec<&str> = w_fused.iter().map(|write| write.key.as_str()).collect();
    assert_eq!(
        names_plain, names_fused,
        "writes name list must match plain godunov when user_source=None"
    );
    let dests_plain: Vec<String> = w_plain
        .iter()
        .map(|write| write.destination.name())
        .collect();
    let dests_fused: Vec<String> = w_fused
        .iter()
        .map(|write| write.destination.name())
        .collect();
    assert_eq!(
        dests_plain, dests_fused,
        "writes dest list must match plain godunov when user_source=None"
    );
}

#[test]
fn uniform_state_picks_up_only_the_user_source_contribution() {
    // **load-bearing semantic check**: build the fused-source godunov for
    // newtonian 3D cartesian + uniform-acceleration user source. configure
    // a uniform state — zero flux divergence — and verify the momentum
    // update equals the analytical user-source contribution:
    //
    //     mom_k_new = mom_k + dt * rho * g_ext_k       (for each k)
    //     rho_new   = rho                              (mass invariant under user mom source)
    //     nrg_new   = nrg                              (uniform_acceleration's energy spec is
    //                                                   a separate SourceSpec, omitted here)
    //
    // this proves the spliced graph is fused at the right state — `rho`
    // and `g_ext_k` resolve to the same Gv values the godunov is already
    // using, so the source contribution rides for free on the existing
    // register-resident reads.
    const D: usize = 3;
    let coords = Coords::Cartesian;
    let spacing = vec![Spacing::Uniform; D];
    let axes = vec![0, 1, 2];

    // the user source: external acceleration on the momentum slot alone.
    // `uniform_acceleration_sources(D, false)[0]` is the momentum SourceSpec.
    let specs = uniform_acceleration_sources(D, false);
    let user_source = &specs[0];
    assert_eq!(
        user_source.target_field, "mom",
        "expecting momentum-targeting overlay"
    );

    // build the fused-source kernel: newtonian 3D + user momentum overlay.
    let kernel = godunov_stage_gv_with_fused_sources(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        D as u8,
        D,
        true,
        GeoSource::Hydro { inertial: false },
        &[user_source],
        false,
    );

    // a uniform 3x3x3 state: rho/mom/nrg all constant, all flux fields too.
    // with constant fluxes, divergence is identically zero on the interior.
    let rho_v = 1.5_f64;
    let mom_v = [0.3_f64, -0.2, 0.4];
    let nrg_v = 5.0_f64;
    let g_ext = [-0.1_f64, -0.2, -9.81];
    let dt = 0.01_f64;

    // constant fluxes => zero divergence in the godunov integrator. names:
    // mass_flux_{i} / mom_flux_{k}_{i} / nrg_flux_{i} for axis i.
    let grid = [3usize; D];
    let lo = [1i32, 1, 1]; // strictly interior cell
    let size = [1usize; D];
    let interior_cell = [1usize, 1, 1];

    let mut fields: Vec<(&str, f64)> = vec![
        ("rho", rho_v),
        ("mom_0", mom_v[0]),
        ("mom_1", mom_v[1]),
        ("mom_2", mom_v[2]),
        ("nrg", nrg_v),
    ];
    let u_n = u_n_fields(rho_v, mom_v, nrg_v);
    fields.extend_from_slice(&u_n);
    // own the dynamically-constructed flux names so the &str borrow stays valid.
    let mass_flux_names: Vec<String> = (0..D).map(|i| format!("mass_flux_{i}")).collect();
    let mom_flux_names: Vec<String> = (0..D)
        .flat_map(|k| (0..D).map(move |i| format!("mom_flux_{k}_{i}")))
        .collect();
    let nrg_flux_names: Vec<String> = (0..D).map(|i| format!("nrg_flux_{i}")).collect();
    for n in &mass_flux_names {
        fields.push((n.as_str(), 0.0));
    }
    for n in &mom_flux_names {
        fields.push((n.as_str(), 0.0));
    }
    for n in &nrg_flux_names {
        fields.push((n.as_str(), 0.0));
    }

    let out = KernelRun::new(kernel)
        .grid(grid)
        .compute_window(lo, size)
        .fields(&fields)
        .scalars(&[
            ("dt", dt),
            EULER_AC[0],
            EULER_AC[1],
            ("mesh_hdil", 0.0),
            ("dx_0", 1.0),
            ("dx_1", 1.0),
            ("dx_2", 1.0),
            ("g_ext_0", g_ext[0]),
            ("g_ext_1", g_ext[1]),
            ("g_ext_2", g_ext[2]),
        ])
        .run();

    let tol = 1e-12;
    out.expect(
        interior_cell,
        &[
            // mass is unaffected by the spec momentum source.
            ("rho", rho_v),
            // momentum picks up exactly the analytical contribution.
            ("mom_0", mom_v[0] + dt * rho_v * g_ext[0]),
            ("mom_1", mom_v[1] + dt * rho_v * g_ext[1]),
            ("mom_2", mom_v[2] + dt * rho_v * g_ext[2]),
            // energy unchanged: the energy-side of uniform_acceleration is a
            // separate SourceSpec (`[1]` in the returned Vec), outside this
            // momentum-targeting fusion. a per-field overlay list
            // extends to it.
            ("nrg", nrg_v),
        ],
        tol,
    );
}

#[test]
fn fused_source_kernel_includes_spec_param_in_signature() {
    // **structural fingerprint** that the spec source actually fused into
    // the godunov kernel. the kernel's IR declares a Param named `g_ext_0`
    // (the uniform_acceleration scalar), a name that enters the graph
    // through the splice alone — which is what makes it a fingerprint.
    const D: usize = 3;
    let coords = Coords::Cartesian;
    let spacing = vec![Spacing::Uniform; D];
    let axes = vec![0, 1, 2];

    let specs = uniform_acceleration_sources(D, false);
    let user_source = &specs[0];

    let (kernel, _writes) = godunov_stage_gv_with_fused_sources(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        D as u8,
        D,
        true,
        GeoSource::Hydro { inertial: false },
        &[user_source],
        false,
    );

    // sanity: the kernel's underlying graph contains a Param leaf named
    // `g_ext_0` — proves uniform_acceleration's `g_ext_k` reached
    // the godunov trace, fused with the divergence + integrator.
    let g = &kernel.graph;
    let mut found_g_ext_0 = false;
    for (_id, node, _ty) in g.iter() {
        if let symbi_ir::graph::Op::Param(s) = &node.op {
            if s.as_str() == "g_ext_0" {
                found_g_ext_0 = true;
                break;
            }
        }
    }
    assert!(
        found_g_ext_0,
        "fused godunov kernel must declare `g_ext_0` Param (from spliced uniform_acceleration source)",
    );

    // control: `g_ext_0` is absent from the plain godunov graph, so the param above is
    // attributable to the splice:
    let (k_plain, _) = godunov_stage_gv(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        D as u8,
        D,
        true,
        GeoSource::Hydro { inertial: false },
    );
    let mut plain_has_g_ext = false;
    for (_id, node, _ty) in k_plain.graph.iter() {
        if let symbi_ir::graph::Op::Param(s) = &node.op {
            if s.as_str() == "g_ext_0" {
                plain_has_g_ext = true;
                break;
            }
        }
    }
    assert!(
        !plain_has_g_ext,
        "control: plain godunov MUST NOT declare g_ext_0 — proves the param came from the spliced spec",
    );
}

// =============================================================================
// multi-source overlay (mom + nrg simultaneously fused)
// =============================================================================

#[test]
fn multi_source_fuses_mom_and_nrg_overlays_in_one_kernel() {
    // the load-bearing claim: `uniform_acceleration_sources(D, true)`
    // returns two SourceSpecs — `[0]` targeting "mom" (S_mom_k = rho*g_ext_k) and
    // `[1]` targeting "nrg" (S_nrg = rho*(v dot g_ext)). passing both to
    // `godunov_stage_gv_with_fused_sources` produces a single kernel that
    // applies both contributions at the right per-field update sites.
    //
    // for a uniform state (zero flux divergence), the analytical update is:
    //     rho_new   = rho
    //     mom_k_new = mom_k + dt * rho * g_ext_k
    //     nrg_new   = nrg   + dt * rho * (v dot g_ext)
    //
    // proving this in one kernel proves the per-spec target_field dispatch
    // wires each spliced output to the right conservation-law update site.
    const D: usize = 3;
    let coords = Coords::Cartesian;
    let spacing = vec![Spacing::Uniform; D];
    let axes = vec![0, 1, 2];

    let specs = uniform_acceleration_sources(D, true);
    assert_eq!(
        specs.len(),
        2,
        "uniform_acceleration with energy must yield 2 specs"
    );
    assert_eq!(specs[0].target_field, "mom");
    assert_eq!(specs[1].target_field, "nrg");
    let refs: Vec<&symbi_hydro::source_spec::SourceSpec> = specs.iter().collect();

    let kernel = godunov_stage_gv_with_fused_sources(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        D as u8,
        D,
        true,
        GeoSource::Hydro { inertial: false },
        &refs,
        false,
    );

    let rho_v = 1.5_f64;
    let mom_v = [0.3_f64, -0.2, 0.4];
    let nrg_v = 5.0_f64;
    let g_ext = [-0.1_f64, -0.2, -9.81];
    let dt = 0.01_f64;

    // primitive velocity = mom / rho; v dot g_ext = sum_k v_k * g_ext_k.
    let vel_v: [f64; D] = [mom_v[0] / rho_v, mom_v[1] / rho_v, mom_v[2] / rho_v];
    let v_dot_g = vel_v[0] * g_ext[0] + vel_v[1] * g_ext[1] + vel_v[2] * g_ext[2];

    let grid = [3usize; D];
    let lo = [1i32, 1, 1];
    let size = [1usize; D];
    let interior_cell = [1usize, 1, 1];

    let mut fields: Vec<(&str, f64)> = vec![
        ("rho", rho_v),
        ("mom_0", mom_v[0]),
        ("mom_1", mom_v[1]),
        ("mom_2", mom_v[2]),
        ("nrg", nrg_v),
    ];
    let mass_flux_names: Vec<String> = (0..D).map(|i| format!("mass_flux_{i}")).collect();
    let mom_flux_names: Vec<String> = (0..D)
        .flat_map(|k| (0..D).map(move |i| format!("mom_flux_{k}_{i}")))
        .collect();
    let nrg_flux_names: Vec<String> = (0..D).map(|i| format!("nrg_flux_{i}")).collect();
    for n in &mass_flux_names {
        fields.push((n.as_str(), 0.0));
    }
    for n in &mom_flux_names {
        fields.push((n.as_str(), 0.0));
    }
    for n in &nrg_flux_names {
        fields.push((n.as_str(), 0.0));
    }
    let u_n = u_n_fields(rho_v, mom_v, nrg_v);
    fields.extend_from_slice(&u_n);

    let out = KernelRun::new(kernel)
        .grid(grid)
        .compute_window(lo, size)
        .fields(&fields)
        .scalars(&[
            ("dt", dt),
            EULER_AC[0],
            EULER_AC[1],
            ("mesh_hdil", 0.0),
            ("dx_0", 1.0),
            ("dx_1", 1.0),
            ("dx_2", 1.0),
            ("g_ext_0", g_ext[0]),
            ("g_ext_1", g_ext[1]),
            ("g_ext_2", g_ext[2]),
        ])
        .run();

    let tol = 1e-12;
    out.expect(
        interior_cell,
        &[
            ("rho", rho_v),
            ("mom_0", mom_v[0] + dt * rho_v * g_ext[0]),
            ("mom_1", mom_v[1] + dt * rho_v * g_ext[1]),
            ("mom_2", mom_v[2] + dt * rho_v * g_ext[2]),
            ("nrg", nrg_v + dt * rho_v * v_dot_g),
        ],
        tol,
    );
}

#[test]
fn mom_and_nrg_overlays_share_one_g_ext_scalar_leaf() {
    // **CSE / vocabulary-sharing contract**: both specs declare `g_ext_k` as
    // a scalar param. fused into the same kernel, they bind to one shared
    // `Param("g_ext_0")` leaf, so each name maps to a single NodeId. proves
    // the scalar-leaf cache in the multi-source builder actually dedups
    // across specs.
    use std::collections::HashSet;
    const D: usize = 3;
    let coords = Coords::Cartesian;
    let spacing = vec![Spacing::Uniform; D];
    let axes = vec![0, 1, 2];

    let specs = uniform_acceleration_sources(D, true);
    let refs: Vec<&symbi_hydro::source_spec::SourceSpec> = specs.iter().collect();
    let (kernel, _writes) = godunov_stage_gv_with_fused_sources(
        coords,
        Spacetime::Minkowski,
        &spacing,
        &axes,
        D as u8,
        D,
        true,
        GeoSource::Hydro { inertial: false },
        &refs,
        false,
    );

    // every Param("g_ext_k") in the graph occurs exactly once — one
    // distinct NodeId per name, so the runtime fills a single leaf per
    // name.
    let mut seen: HashSet<String> = HashSet::new();
    for (_id, node, _ty) in kernel.graph.iter() {
        if let symbi_ir::graph::Op::Param(sym) = &node.op {
            let name = sym.as_str();
            if name.starts_with("g_ext_") {
                assert!(
                    seen.insert(name.to_string()),
                    "duplicate Param leaf '{name}' — multi-source param cache must dedup",
                );
            }
        }
    }
    assert_eq!(
        seen.len(),
        D,
        "expected D={D} distinct g_ext_* params, got {} ({seen:?})",
        seen.len()
    );
}

#[test]
fn ssp_combine_applies_runtime_coefficients() {
    // **carrier-oracle for the runtime (a0, ac) combine** — the load-bearing claim of the
    // integrator collapse. a uniform state (constant fluxes, so the divergence vanishes and the
    // build carries the combine alone) isolates the SSP shu-osher convex combine
    // `cons = a0*u_n + ac*cons` from the stencil. with u_n != cons and the SSP-RK2 corrector
    // coefficients (1/2, 1/2), the kernel evaluated on the CPU interpreter produces the analytical
    // convex combination — proving a single compiled kernel realizes any explicit SSP stage from
    // its runtime scalars. 1D cartesian, ncomp=1, mass + one momentum law (energy omitted).
    let coords = Coords::Cartesian;
    let kernel = godunov_stage_gv(
        coords,
        Spacetime::Minkowski,
        &[Spacing::Uniform],
        &[0],
        1,
        1,
        false,
        GeoSource::Hydro { inertial: false },
    );

    // distinct snapshot (u_n) and current (cons) states; zero fluxes => div == 0.
    let den_n = 1.0_f64;
    let den_c = 2.0_f64;
    let mom_n = 2.0_f64;
    let mom_c = 4.0_f64;
    let (a0, ac) = (0.5_f64, 0.5_f64); // SSP-RK2 corrector

    let out = KernelRun::new(kernel)
        .grid([3usize])
        .compute_window([1i32], [1usize])
        .fields(&[
            ("rho", den_c),
            ("mom_0", mom_c),
            ("u_n_rho", den_n),
            ("u_n_mom_0", mom_n),
            ("mass_flux_0", 0.0),
            ("mom_flux_0_0", 0.0),
        ])
        .scalars(&[
            ("dt", 0.01),
            ("a0", a0),
            ("ac", ac),
            ("mesh_hdil", 0.0),
            ("dx_0", 1.0),
        ])
        .run();

    out.expect(
        [1usize],
        &[
            // div == 0 => fe == cons, so cons_new = a0*u_n + ac*cons.
            ("rho", a0 * den_n + ac * den_c), // 0.5*1 + 0.5*2 = 1.5
            ("mom_0", a0 * mom_n + ac * mom_c), // 0.5*2 + 0.5*4 = 3.0
        ],
        1e-12,
    );
}

#[test]
fn unsupported_target_field_panics_loudly() {
    // **discipline**: a target_field outside godunov's routing vocabulary
    // (a typo, or a name outside the substrate's wiring) is a programmer
    // bug — the dispatch panics with the offending value, making the bad
    // spec loud at build time.
    // (the godunov vocabulary is den/mom/nrg/bcell; this uses a bogus name.)
    const D: usize = 1;
    let coords = Coords::Cartesian;
    let spacing = vec![Spacing::Uniform; D];
    let axes = vec![0];

    // hand-built spec with an unknown target_field; the panic fires on the
    // dispatch match, ahead of both splicing and any call to build_source.
    fn empty_builder(_d: usize) -> symbi_hydro::source_spec::BuiltSource {
        // 1 output, bound to rho; the dispatch panic fires first, so this
        // graph stands unevaluated.
        use symbi_ir::graph::Graph;
        let mut g = Graph::new();
        let rho = g.add_scalar_param("rho", symbi_ir::ElementTy::F64);
        symbi_hydro::source_spec::BuiltSource {
            graph: g,
            outputs: vec![rho],
            params: vec!["rho".to_string()],
        }
    }
    let bad_spec = symbi_hydro::source_spec::SourceSpec {
        kind: symbi_hydro::source_spec::SourceKind::UserDefined,
        target_field: "bogus_target", // outside the godunov vocabulary (den/mom/nrg/bcell)
        build_source: empty_builder,
    };
    let refs: Vec<&symbi_hydro::source_spec::SourceSpec> = vec![&bad_spec];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = godunov_stage_gv_with_fused_sources(
            coords,
            Spacetime::Minkowski,
            &spacing,
            &axes,
            D as u8,
            D,
            true,
            GeoSource::Hydro { inertial: false },
            &refs,
            false,
        );
    }));
    assert!(result.is_err(), "unsupported target_field MUST panic");
}

#[test]
fn source_apply_pass_adds_dt_times_source() {
    // the standalone additive source pass `cons += dt * \sum S` — the general source executor.
    // uniform state (this pass carries the source alone), uniform_accel (mom + nrg). assert the
    // in-place conserved update is exactly the analytical source contribution times dt:
    //   mom_k += dt * rho * g_ext_k ;  nrg += dt * rho * (v dot g_ext) ;  rho unchanged.
    // the driver passes dt = ac*dt, so this is the additive half of the fused stage.
    const D: usize = 3;
    let specs = uniform_acceleration_sources(D, true); // [mom, nrg]
    let refs: Vec<&symbi_hydro::source_spec::SourceSpec> = specs.iter().collect();
    let kernel = source_apply_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; D],
        &[0, 1, 2],
        D as u8,
        D,
        true,
        &refs,
    );

    let rho = 1.5_f64;
    let mom = [0.3_f64, -0.2, 0.4];
    let nrg = 5.0_f64;
    let g = [-0.1_f64, -0.2, -9.81];
    let dt = 0.01_f64; // the driver would pass ac*dt; here a bare dt suffices for the unit check
    let vel = [mom[0] / rho, mom[1] / rho, mom[2] / rho];
    let v_dot_g = vel[0] * g[0] + vel[1] * g[1] + vel[2] * g[2];

    let out = KernelRun::new(kernel)
        .grid([1usize, 1, 1])
        .compute_window([0i32, 0, 0], [1usize, 1, 1])
        .fields(&[
            // source-eval state (u_stage); standalone here so it equals the current state.
            ("rho", rho),
            ("mom_0", mom[0]),
            ("mom_1", mom[1]),
            ("mom_2", mom[2]),
            // add-base (cons), equal to the state: this pass runs standalone.
            ("cons_den", rho),
            ("cons_mom_0", mom[0]),
            ("cons_mom_1", mom[1]),
            ("cons_mom_2", mom[2]),
            ("cons_nrg", nrg),
        ])
        .scalars(&[
            ("dt", dt),
            ("g_ext_0", g[0]),
            ("g_ext_1", g[1]),
            ("g_ext_2", g[2]),
        ])
        .run();

    out.expect(
        [0usize, 0, 0],
        &[
            ("rho", rho), // mass source is zero
            ("mom_0", mom[0] + dt * rho * g[0]),
            ("mom_1", mom[1] + dt * rho * g[1]),
            ("mom_2", mom[2] + dt * rho * g[2]),
            ("nrg", nrg + dt * rho * v_dot_g),
        ],
        1e-13,
    );
}

#[test]
fn fused_stage_equals_plain_plus_additive_pass() {
    // the fused godunov stage computes bit-for-bit the
    // same conserved update as (plain godunov stage) followed by (the standalone additive source
    // pass) — the fused builder adds the user source as the same post-combine term
    // `+ \sum ac*dt*contrib` the pass adds, evaluated at the same stage-input state. tested at the
    // SSP-RK2 corrector (a0=ac=0.5 — the FP-distribution-sensitive case) with u_n != u.
    // uniform state + uniform fluxes => div == 0 (the flux is identical in both paths anyway).
    const D: usize = 3;
    let specs = uniform_acceleration_sources(D, true);
    let refs: Vec<&symbi_hydro::source_spec::SourceSpec> = specs.iter().collect();
    let coords = Coords::Cartesian;
    let sp = [Spacing::Uniform; D];
    let axes = [0usize, 1, 2];
    let geo = GeoSource::Hydro { inertial: false };

    let fused = godunov_stage_gv_with_fused_sources(
        coords,
        Spacetime::Minkowski,
        &sp,
        &axes,
        D as u8,
        D,
        true,
        geo,
        &refs,
        false,
    );
    let plain = godunov_stage_gv(
        coords,
        Spacetime::Minkowski,
        &sp,
        &axes,
        D as u8,
        D,
        true,
        geo,
    );
    let pass = source_apply_gv(coords, &sp, &axes, D as u8, D, true, &refs);

    let (rho, mom, nrg) = (1.5_f64, [0.3_f64, -0.2, 0.4], 5.0_f64);
    let (rho_n, mom_n, nrg_n) = (1.1_f64, [0.1_f64, 0.05, -0.2], 4.0_f64);
    let g = [-0.1_f64, -0.2, -9.81];
    let dt = 0.013_f64;
    let (a0, ac) = (0.5_f64, 0.5_f64); // RK2 corrector

    let u_n = u_n_fields(rho_n, mom_n, nrg_n);
    let mut gfields: Vec<(&str, f64)> = vec![
        ("rho", rho),
        ("mom_0", mom[0]),
        ("mom_1", mom[1]),
        ("mom_2", mom[2]),
        ("nrg", nrg),
    ];
    gfields.extend_from_slice(&u_n);
    let mass_f: Vec<String> = (0..D).map(|i| format!("mass_flux_{i}")).collect();
    let mom_f: Vec<String> = (0..D)
        .flat_map(|k| (0..D).map(move |i| format!("mom_flux_{k}_{i}")))
        .collect();
    let nrg_f: Vec<String> = (0..D).map(|i| format!("nrg_flux_{i}")).collect();
    for n in &mass_f {
        gfields.push((n.as_str(), 0.0));
    }
    for n in &mom_f {
        gfields.push((n.as_str(), 0.0));
    }
    for n in &nrg_f {
        gfields.push((n.as_str(), 0.0));
    }
    let gscalars = [
        ("dt", dt),
        ("a0", a0),
        ("ac", ac),
        ("mesh_hdil", 0.0),
        ("dx_0", 1.0),
        ("dx_1", 1.0),
        ("dx_2", 1.0),
        ("g_ext_0", g[0]),
        ("g_ext_1", g[1]),
        ("g_ext_2", g[2]),
    ];

    let cell = [1usize, 1, 1];
    let fused_out = KernelRun::new(fused)
        .grid([3, 3, 3])
        .compute_window([1, 1, 1], [1, 1, 1])
        .fields(&gfields)
        .scalars(&gscalars)
        .run();
    let plain_out = KernelRun::new(plain)
        .grid([3, 3, 3])
        .compute_window([1, 1, 1], [1, 1, 1])
        .fields(&gfields)
        .scalars(&gscalars)
        .run();

    // chain: additive pass — source-state u_stage = the stage input u; add-base = plain's output.
    // the driver passes dt = ac*dt; ac_dt in the fused kernel is the bit-identical product.
    let acdt = ac * dt;
    let add_out = KernelRun::new(pass)
        .grid([1usize, 1, 1])
        .compute_window([0, 0, 0], [1, 1, 1])
        .fields(&[
            ("rho", rho),
            ("mom_0", mom[0]),
            ("mom_1", mom[1]),
            ("mom_2", mom[2]), // u_stage
            ("cons_den", plain_out.get(cell, "rho")),
            ("cons_mom_0", plain_out.get(cell, "mom_0")),
            ("cons_mom_1", plain_out.get(cell, "mom_1")),
            ("cons_mom_2", plain_out.get(cell, "mom_2")),
            ("cons_nrg", plain_out.get(cell, "nrg")),
        ])
        .scalars(&[
            ("dt", acdt),
            ("g_ext_0", g[0]),
            ("g_ext_1", g[1]),
            ("g_ext_2", g[2]),
        ])
        .run();

    for name in ["rho", "mom_0", "mom_1", "mom_2", "nrg"] {
        let f = fused_out.get(cell, name);
        let a = add_out.get([0usize, 0, 0], name);
        assert_eq!(f, a, "fused != plain+additive for {name}: {f} vs {a}");
    }
}
