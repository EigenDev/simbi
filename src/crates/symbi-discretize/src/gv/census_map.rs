// =============================================================================
// census_map.rs
//
// the per-cell half of a census, TRACED: one kernel that evaluates the registered accumulator
// expressions and the destination bucket of every cell, which a segmented reduction then folds.
// the host walks cells and interprets the same graph; this is the form that runs on a device.
//
// the binning is NOT reimplemented here. `segment_marker_generic` is carrier-generic, so the
// bucket search traced into this kernel is the identical expression the host evaluates in f64 —
// the one part of a census where two implementations would disagree invisibly, since both would
// still produce a smooth, plausible profile and nothing would be comparing them.
//
// the leaf vocabulary matches `resolve_census_param` exactly: `rho`, `pre`, `dv`, `t`, `vel_k`,
// `x_k`, `p{i}`. a name the census reads and this does not bind is a lowering error, not a
// silent zero.
//
// usage:
//   let (k, writes) = census_map_gv(coords, &spacing, &axes, ndim, dof, &built, &bin_axes, n_val);
// =============================================================================

use std::collections::HashMap;

use symbi_algebra::algebra::Numeric;
use symbi_ir::algebra::Scalar;
use symbi_ir::graph::NodeId;
use symbi_ir::{FieldRef, Gv, GvKernel, begin_trace, end_trace, with_trace};

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;

/// the traced writes of a census map: one field per accumulator plus the bucket assignment.
pub type CensusWrites = Vec<(String, symbi_ir::FieldBind, NodeId)>;

/// trace the census map. `built` is the census's single lowered graph, whose outputs are the bin
/// axis coordinates followed by the accumulator values — the order `CensusConfig::output_nodes`
/// fixes. `bin_axes` are the registered axes, in the same order.
///
/// `n_segments` is passed rather than derived so this file needs no view of the spec type; the
/// caller already holds it and a disagreement would be a caller bug either way.
#[allow(clippy::too_many_arguments)]
pub fn census_map_gv<A: CensusAxis>(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    dof: usize,
    built: &symbi_hydro::source_spec::BuiltSource,
    bin_axes: &[A],
    n_values: usize,
    n_segments: usize,
) -> (GvKernel, CensusWrites) {
    begin_trace();

    // the LIVE primitives, not the stage input: a census bins the state at the time it is
    // sampled, which is the tail of an accepted step, after the recovery.
    let mut env: HashMap<String, NodeId> = HashMap::new();
    env.insert("rho".into(), Gv::field("rho", FieldRef::PrimRho).node());
    env.insert("pre".into(), Gv::field("pre", FieldRef::PrimPre).node());
    env.insert("t".into(), Gv::scalar("t").node());
    for k in 0..dof {
        env.insert(
            format!("vel_{k}"),
            Gv::field(&format!("vel_{k}"), FieldRef::PrimVel(k as u8)).node(),
        );
    }

    // the cell measure and the centroid come from ONE geometry evaluation. `dv` is what makes an
    // accumulator extensive, and it is the chart's own volume — not a coordinate width — so a
    // spherical shell's mass is its actual mass.
    let geo = cell_geometry_gv(coords, spacing, axes, ndim as usize);
    env.insert("dv".into(), (Gv::ONE / geo.inv_volume).node());
    for (d, c) in geo.centroid.iter().enumerate() {
        env.insert(format!("x_{d}"), c.node());
    }

    // the config's tunables, bound lazily by the names the graph actually reads: a census that
    // declares no parameters emits no scalar slots.
    for pname in &built.params {
        env.entry(pname.clone())
            .or_insert_with(|| Gv::scalar(pname).node());
    }

    let out = with_trace(|t| {
        symbi_hydro::source_spec::splice_built_source_into(built, t.graph(), &env)
    });
    let n_axes = bin_axes.len();
    assert_eq!(
        out.len(),
        n_axes + n_values,
        "census map: the lowered graph emits {} output(s), expected {n_axes} axis coordinate(s) \
         + {n_values} accumulator(s)",
        out.len()
    );

    // the bucket, from the SAME expression the host evaluates.
    let coords_gv: Vec<Gv> = out[..n_axes].iter().map(|&n| Gv::of(n)).collect();
    let segment = segment_marker_traced(bin_axes, &coords_gv, n_segments);

    let mut writes: CensusWrites = Vec::with_capacity(n_values + 1);
    for v in 0..n_values {
        writes.push((
            format!("census_value_{v}"),
            format!("census_value_{v}").into(),
            out[n_axes + v],
        ));
    }
    writes.push((
        "census_segment".to_string(),
        "census_segment".into(),
        segment.node(),
    ));
    (end_trace(), writes)
}

/// what the traced binning needs of an axis: its edges. a trait rather than a concrete type so
/// this crate does not depend on the census spec's home crate.
pub trait CensusAxis {
    fn edges(&self) -> &[f64];
}

/// the bucket index, branch-free, as the traced twin of the host search.
///
/// kept HERE rather than imported so this crate stays free of a dependency on the spec type, but
/// it is the same algorithm and is gated against the host's independent partition-point search:
/// `bin = #{edges at or below x} - 1`, clamped into the last bin so a value exactly on the outer
/// edge is data rather than a drop. a NaN coordinate compares false against both bounds and is
/// dropped, never binned.
fn segment_marker_traced<A: CensusAxis>(bin_axes: &[A], coords: &[Gv], n_segments: usize) -> Gv {
    let mut flat = Gv::ZERO;
    let mut all_in_range = Gv::ONE.cmp_gt(Gv::ZERO);
    for (axis, &x) in bin_axes.iter().zip(coords) {
        let edges = axis.edges();
        let n_bins = edges.len() - 1;
        all_in_range = all_in_range
            & x.cmp_ge(Gv::from_f64(edges[0]))
            & x.cmp_le(Gv::from_f64(edges[n_bins]));
        let mut count = Gv::ZERO;
        for &edge in edges {
            count = count + Gv::select(x.cmp_ge(Gv::from_f64(edge)), Gv::ONE, Gv::ZERO);
        }
        let bin = (count - Gv::ONE)
            .min(Gv::from_f64((n_bins - 1) as f64))
            .max(Gv::ZERO);
        flat = flat * Gv::from_f64(n_bins as f64) + bin;
    }
    Gv::select(all_in_range, flat, Gv::from_f64(n_segments as f64))
}
