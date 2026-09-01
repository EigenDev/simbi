// =============================================================================
// spec_into_gv_trace.rs
//
// proves the mechanism for fusing spec-driven sources into
// the active Gv trace. validates that `splice_built_source_into` correctly
// recreates a `BuiltSource` graph inside the discretize crate's tracing
// session, with param leaves replaced by Gv-trace-native NodeIds.
//
// splicing is what lets a spec-driven source ride inside the godunov stage, so
// flux divergence + source contribution + integrator run in one launch, over one
// register-resident state, under one round of CSE.
//
// asserted here:
//   - inside an active Gv trace, calling `splice_built_source_into` with a
//     `BuiltSource` (from `point_mass_gravity_sources` or similar) produces
//     Gv-trace NodeIds that are valid `cx.gv(node)` values;
//   - the spliced outputs evaluate to the same f64 values as the standalone
//     BuiltSource at the same parameter state, so the splice preserves
//     semantics on top of structural validity;
//   - the trace closes cleanly into a well-formed kernel graph.
//
// run: cargo test -p symbi-discretize --test spec_into_gv_trace
// =============================================================================

use std::collections::HashMap;

use symbi_hydro::regime_spec::law_params;
use symbi_hydro::source_spec::{
    gravity_params, point_mass_gravity_sources, source_params, splice_built_source_into,
};
use symbi_ir::graph::NodeId;
use symbi_ir::gv::{Gv, TraceCx, trace};

/// helper: evaluate a Gv-trace NodeId at f64 against a known parameter state,
/// using scalarize + the Cpu interpreter on the trace's graph as it stands.
/// runs inside the caller's open trace via its `cx` token.
fn eval_in_trace(cx: TraceCx, out: NodeId, values: &[(&str, f64)]) -> f64 {
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::passes::scalarize::scalarize;
    // snapshot the trace's graph; scalarize the chosen output. the
    // scalarized fn's params come from the graph's declared params, in
    // declaration order.
    let (lowered, param_order) = cx.with_trace(|t| {
        let lowered = scalarize(t.graph(), out, "spliced_output");
        let params: Vec<String> = lowered.params.iter().map(|p| p.name.clone()).collect();
        (lowered, params)
    });
    let inputs: Vec<f64> = param_order
        .iter()
        .map(|pname| {
            values
                .iter()
                .find(|(n, _)| *n == pname.as_str())
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("eval_in_trace: missing param '{pname}'"))
        })
        .collect();
    Cpu.eval_elemental(&lowered, &inputs)[0]
}

#[test]
fn splice_produces_valid_gv_node_ids() {
    // basic structural check: the splice mechanism returns NodeIds that
    // wrap as Gv values inside the active trace. a structural claim; the
    // semantic one is gated below.
    let (_kernel, ()) = trace(|cx| {
        // declare the leaves the spec needs as Gv-trace nodes. these stand in
        // for what the godunov kernel would already have on hand (rho, vel,
        // position, mass).
        let leaves = declare_gravity_leaves(cx);

        // build the spec source standalone, then splice into the active trace.
        let built = (point_mass_gravity_sources(3, false)[0].build_source)(3);
        let spliced: Vec<NodeId> =
            cx.with_trace(|t| splice_built_source_into(&built, t.graph(), &leaves));

        // wrap as Gv values — this is the contract the godunov fusion needs.
        let s_mom: Vec<Gv> = spliced.iter().map(|&n| cx.gv(n)).collect();
        assert_eq!(
            s_mom.len(),
            3,
            "3D gravity momentum source emits 3 components"
        );

        // every component resolves to a valid NodeId in the trace's graph.
        // (`Gv::node()` panics if the underlying node is invalid; surviving this
        // round-trip is the proof.)
        for g in &s_mom {
            let _ = g.node();
        }
    });
    // the trace closed cleanly — the spliced ops + the leaves form a coherent
    // graph: every reference resolves and every Param is bound.
}

#[test]
fn spliced_outputs_match_standalone_at_same_param_state() {
    // **the load-bearing claim**: splicing preserves semantics — the evaluated
    // values match, beyond structural validity. for a known parameter state, the
    // spliced source evaluated inside the trace equals the standalone BuiltSource
    // evaluated via scalarize+interp directly. proves the spliced graph computes
    // the same function as the original.
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::passes::scalarize::scalarize;

    // ----- standalone reference -----
    let built = (point_mass_gravity_sources(3, false)[0].build_source)(3);
    let standalone: Vec<f64> = (0..3)
        .map(|k| {
            let lowered = scalarize(built.graph(), built.outputs()[k], "ref");
            // route inputs by name, in the order scalarize declared.
            let inputs: Vec<f64> = lowered
                .params
                .iter()
                .map(|p| sample_param_value(&p.name))
                .collect();
            Cpu.eval_elemental(&lowered, &inputs)[0]
        })
        .collect();

    // ----- in-trace splice + per-output eval -----
    let (_kernel, ()) = trace(|cx| {
        let leaves = declare_gravity_leaves(cx);
        let built2 = (point_mass_gravity_sources(3, false)[0].build_source)(3);
        let spliced: Vec<NodeId> =
            cx.with_trace(|t| splice_built_source_into(&built2, t.graph(), &leaves));

        // sample values for each declared leaf.
        let sample_vals: Vec<(String, f64)> = leaves
            .keys()
            .map(|name| (name.clone(), sample_param_value(name)))
            .collect();
        let vals_ref: Vec<(&str, f64)> =
            sample_vals.iter().map(|(n, v)| (n.as_str(), *v)).collect();

        for k in 0..3 {
            let traced = eval_in_trace(cx, spliced[k], &vals_ref);
            let expected = standalone[k];
            assert!(
                (traced - expected).abs() < 1e-12,
                "component {k}: spliced-in-trace {traced} != standalone {expected}",
            );
        }
    });
}

#[test]
fn splice_panics_loudly_on_missing_param_substitute() {
    // the discipline at the splice site: a Param in BuiltSource that has no
    // substitute in `name_to_node` is a programmer bug — surface it loudly
    // with a clear panic; silently continuing would compute a wrong value undetected.
    let (_kernel, ()) = trace(|cx| {
        let mut sparse_leaves: HashMap<String, NodeId> = HashMap::new();
        // only declare `rho` — the rest of gravity's params are missing.
        let rho = cx.scalar("rho");
        sparse_leaves.insert(law_params::RHO.to_string(), rho.node());

        let built = (point_mass_gravity_sources(3, false)[0].build_source)(3);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cx.with_trace(|t| splice_built_source_into(&built, t.graph(), &sparse_leaves))
        }));
        assert!(
            result.is_err(),
            "splice must panic on missing param substitute"
        );
    });
}

// ----- helpers -----

fn declare_gravity_leaves(cx: TraceCx) -> HashMap<String, NodeId> {
    let mut m: HashMap<String, NodeId> = HashMap::new();
    m.insert(
        law_params::RHO.to_string(),
        cx.scalar(law_params::RHO).node(),
    );
    m.insert(law_params::vel(0), cx.scalar(&law_params::vel(0)).node());
    m.insert(law_params::vel(1), cx.scalar(&law_params::vel(1)).node());
    m.insert(law_params::vel(2), cx.scalar(&law_params::vel(2)).node());
    m.insert(source_params::x(0), cx.scalar(&source_params::x(0)).node());
    m.insert(source_params::x(1), cx.scalar(&source_params::x(1)).node());
    m.insert(source_params::x(2), cx.scalar(&source_params::x(2)).node());
    m.insert(
        gravity_params::xm(0),
        cx.scalar(&gravity_params::xm(0)).node(),
    );
    m.insert(
        gravity_params::xm(1),
        cx.scalar(&gravity_params::xm(1)).node(),
    );
    m.insert(
        gravity_params::xm(2),
        cx.scalar(&gravity_params::xm(2)).node(),
    );
    m.insert(
        gravity_params::GM.to_string(),
        cx.scalar(gravity_params::GM).node(),
    );
    m.insert(
        gravity_params::EPS.to_string(),
        cx.scalar(gravity_params::EPS).node(),
    );
    m
}

fn sample_param_value(name: &str) -> f64 {
    // a fixed deterministic sample for cross-validation, held clear of the
    // singular positions (|x - xm| > 0).
    match name {
        "rho" => 1.5,
        "vel_0" => 0.3,
        "vel_1" => -0.2,
        "vel_2" => 0.1,
        "x_0" => 1.0,
        "x_1" => 2.0,
        "x_2" => 3.0,
        "xm_0" => 0.0,
        "xm_1" => 0.0,
        "xm_2" => 0.0,
        "gm" => 1.0,
        "eps" => 0.0, // bare 1/r^3 reference (matches the analytic cross-check)
        other => panic!("sample_param_value: unknown param '{other}'"),
    }
}
