// =============================================================================
// census_map_traced.rs
//
// the TRACED census map against the same graph evaluated in f64 — the equivalence a device path
// stands on. the host walks cells and interprets the registered expressions; this kernel is what a
// device runs instead, and the two must agree cell for cell.
//
// the failure this exists to catch is not a crash. a traced map that binned differently, or read a
// leaf the host resolves to something else, would still produce a smooth, positive, plausible
// profile — and on a machine where only one of the two paths ever runs, nothing would compare
// them. so the comparison happens here, where both are cheap.
// =============================================================================

mod harness;

use harness::KernelRun;
use symbi_discretize::coords::{Coords, Spacing};
use symbi_discretize::gv::census_map::{CensusAxis, census_map_gv};
use symbi_hydro::CensusConfig;

const N: usize = 16;
const R_LO: f64 = 1.0;
const DR: f64 = 0.4; // spans r ~ 1.1 to ~7.2, so cells fall BOTH inside and past the outer edge

struct Axis(Vec<f64>);
impl CensusAxis for Axis {
    fn edges(&self) -> &[f64] {
        &self.0
    }
}

fn rho_at(i: usize) -> f64 {
    1.0 + 0.4 * ((i as f64) * 0.37).sin()
}

fn pre_at(i: usize) -> f64 {
    0.5 + 0.2 * ((i as f64) * 0.21).cos()
}

/// volume-weighted centroid of a spherical shell — the coordinate `cell_geometry_gv` binds to
/// `x_0`, and therefore the radius the binning sees.
fn centroid(i: usize) -> f64 {
    let (lo, hi) = (R_LO + i as f64 * DR, R_LO + (i + 1) as f64 * DR);
    0.75 * (hi.powi(4) - lo.powi(4)) / (hi.powi(3) - lo.powi(3))
}

/// the spherical shell volume the `dv` leaf resolves to.
fn dv(i: usize) -> f64 {
    let (lo, hi) = (R_LO + i as f64 * DR, R_LO + (i + 1) as f64 * DR);
    4.0 / 3.0 * std::f64::consts::PI * (hi.powi(3) - lo.powi(3))
}

/// a census binned on radius, accumulating mass and mass-weighted pressure: two accumulators so a
/// mis-ordered output would swap them, and an axis coordinate that is a leaf the map must bind.
fn radial_census() -> CensusConfig {
    CensusConfig::from_json(
        r#"{
            "name": "shells",
            "axes": [{"name": "r", "expr": 0, "edges": [1.0, 2.0, 3.0, 5.0]}],
            "values": [3, 5],
            "value_names": ["mass", "mass_pre"],
            "op": "add",
            "params": [],
            "nodes": [
                {"op": "VARIABLE_X1"},
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 1, "right": 2},
                {"op": "VARIABLE_PRESSURE"},
                {"op": "MULTIPLY", "left": 3, "right": 4}
            ]
        }"#,
    )
    .expect("census config parses")
}

#[test]
fn the_traced_map_reproduces_the_host_evaluation_cell_for_cell() {
    let cfg = radial_census();
    let edges = vec![1.0, 2.0, 3.0, 5.0];
    let bin_axes = vec![Axis(edges.clone())];
    let n_segments = edges.len() - 1;

    let built = symbi_hydro::expr_bridge::build_census_expressions(&cfg).expect("census lowers");
    let out = KernelRun::new(census_map_gv(
        Coords::Spherical,
        &[Spacing::Uniform],
        &[0],
        1,
        1,
        &built,
        &bin_axes,
        2,
        n_segments,
    ))
    .grid([N])
    .compute_window([0], [N])
    .field_with("rho", |c| rho_at(c[0]))
    .field_with("pre", |c| pre_at(c[0]))
    .field_with("vel_0", |_| 0.0)
    .scalars(&[
        ("t", 0.0),
        ("x_lo_0", R_LO),
        ("dx_0", DR),
        ("map_kind_0", 0.0),
    ])
    .run();

    let mut binned = 0usize;
    for i in 0..N {
        let r = centroid(i);
        let want_mass = rho_at(i) * dv(i);
        let want_mass_pre = want_mass * pre_at(i);

        // the host binning, computed independently here from the edges.
        let want_seg = if r < edges[0] || r > edges[n_segments] {
            n_segments as f64
        } else {
            let k = edges.partition_point(|&e| e <= r);
            (k.saturating_sub(1).min(n_segments - 1)) as f64
        };
        if want_seg != n_segments as f64 {
            binned += 1;
        }

        let got_mass = out.get([i], "census_value_0");
        let got_pre = out.get([i], "census_value_1");
        let got_seg = out.get([i], "census_segment");

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
        assert!(
            rel(got_mass, want_mass) < 1.0e-12,
            "cell {i}: traced mass {got_mass} != host {want_mass}. the `dv` leaf is the usual \
             culprit — a coordinate width instead of the chart's volume."
        );
        assert!(
            rel(got_pre, want_mass_pre) < 1.0e-12,
            "cell {i}: traced mass_pre {got_pre} != host {want_mass_pre}; the accumulator outputs \
             may be ordered differently than registered"
        );
        assert_eq!(
            got_seg, want_seg,
            "cell {i} (r = {r:.4}): traced bucket {got_seg} != host {want_seg}"
        );
    }

    // the premise: the sweep must actually exercise BOTH outcomes. a grid entirely inside the
    // edges never tests the drop path, and one entirely outside never tests the binning.
    assert!(
        binned > 0 && binned < N,
        "the grid does not straddle the declared edges ({binned} of {N} cells binned); the \
         comparison exercises only one branch of the marker"
    );
}
