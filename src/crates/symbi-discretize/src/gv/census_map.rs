// =============================================================================
// census_map.rs
//
// the per-cell half of a census, traced: one kernel that evaluates the registered accumulator
// expressions and the destination bucket of every cell, which a segmented reduction then folds.
// the host walks cells and interprets the same graph; this is the form that runs on a device.
//
// the binning has a single implementation. `segment_marker_generic` is carrier-generic, so the
// bucket search traced into this kernel is the identical expression the host evaluates in f64 —
// the one part of a census where two implementations would disagree invisibly, since each would
// still produce a smooth, plausible profile and no check would catch the difference.
//
// the leaf vocabulary matches `resolve_census_param` exactly: `rho`, `pre`, `dv`, `t`, `vel_k`,
// `x_k`, `p{i}`. every name the census reads is bound here; an unbound name surfaces as a
// lowering error.
//
// usage:
//   let (k, writes) = census_map_gv(coords, &spacing, &axes, ndim, dof, &built, &bin_axes, n_val);
// =============================================================================

use std::collections::HashMap;

use symbi_algebra::algebra::Numeric;
use symbi_ir::algebra::Scalar;
use symbi_ir::graph::NodeId;
use symbi_ir::{
    FieldRef, Gv, GvKernel, KernelWrite, KernelWrites, begin_trace, end_trace, with_trace,
};

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;

/// trace the census map. `built` is the census's single lowered graph, whose outputs are the bin
/// axis coordinates followed by the accumulator values — the order `CensusConfig::output_nodes`
/// fixes. `bin_axes` are the registered axes, in the same order.
///
/// `n_segments` is passed in, which keeps this file clear of the spec type; the caller already
/// holds it and a disagreement would be a caller bug either way.
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
) -> (GvKernel, KernelWrites) {
    begin_trace();

    // the live primitives: a census bins the state at the time it is sampled, which is the tail
    // of an accepted step, after the recovery (the stage input holds the step's starting state).
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

    // the cell measure and the centroid come from one geometry evaluation. `dv` is what makes an
    // accumulator extensive, and it is the chart's own volume, so a spherical shell's mass is its
    // actual mass.
    let geo = cell_geometry_gv(coords, spacing, axes, ndim as usize);
    env.insert("dv".into(), (Gv::ONE / geo.inv_volume).node());
    for (d, c) in geo.centroid.iter().enumerate() {
        env.insert(format!("x_{d}"), c.node());
    }

    // the config's tunables, bound lazily by the names the graph actually reads: a census that
    // declares parameters is the one emitting scalar slots.
    for pname in built.params() {
        env.entry(pname.clone())
            .or_insert_with(|| Gv::scalar(pname).node());
    }

    let out =
        with_trace(|t| symbi_hydro::source_spec::splice_built_source_into(built, t.graph(), &env));
    let n_axes = bin_axes.len();
    assert_eq!(
        out.len(),
        n_axes + n_values,
        "census map: the lowered graph emits {} output(s), expected {n_axes} axis coordinate(s) \
         + {n_values} accumulator(s)",
        out.len()
    );

    // the bucket, from the same expression the host evaluates.
    let coords_gv: Vec<Gv> = out[..n_axes].iter().map(|&n| Gv::of(n)).collect();
    let segment = segment_marker_traced(bin_axes, &coords_gv, n_segments);

    let mut writes = KernelWrites::with_capacity(n_values + 1);
    for v in 0..n_values {
        writes.push(KernelWrite::new(
            format!("census_value_{v}"),
            format!("census_value_{v}"),
            out[n_axes + v],
        ));
    }
    writes.push(KernelWrite::new(
        "census_segment",
        "census_segment",
        segment.node(),
    ));
    (end_trace(), writes)
}

/// what the traced binning needs of an axis: its edges. a trait, so this crate stands clear of
/// the census spec's home crate.
pub trait CensusAxis {
    fn edges(&self) -> &[f64];
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinLocator {
    Linear { lo: f64, inv_step: f64 },
    Log { ln_lo: f64, inv_ln_step: f64 },
    OrderedEdges,
}

fn bin_locator(edges: &[f64]) -> BinLocator {
    let n = edges.len() - 1;
    let scale = edges.iter().copied().map(f64::abs).fold(1.0_f64, f64::max);
    let tol = 64.0 * f64::EPSILON * scale;

    let step = (edges[n] - edges[0]) / n as f64;
    if edges
        .iter()
        .enumerate()
        .all(|(k, &edge)| (edge - (edges[0] + k as f64 * step)).abs() <= tol)
    {
        return BinLocator::Linear {
            lo: edges[0],
            inv_step: step.recip(),
        };
    }

    if edges[0] > 0.0 {
        let ln_lo = edges[0].ln();
        let ln_step = (edges[n].ln() - ln_lo) / n as f64;
        let log_tol = 64.0 * f64::EPSILON * edges[n].ln().abs().max(1.0);
        if edges
            .iter()
            .enumerate()
            .all(|(k, &edge)| (edge.ln() - (ln_lo + k as f64 * ln_step)).abs() <= log_tol)
        {
            return BinLocator::Log {
                ln_lo,
                inv_ln_step: ln_step.recip(),
            };
        }
    }

    BinLocator::OrderedEdges
}

/// branch-free upper-bound search: the first edge strictly above `x`.
/// the select tree has logarithmic depth and preserves the host partition-point
/// semantics exactly for arbitrary declared edges.
fn upper_bound_traced(edges: &[f64], x: Gv, lo: usize, hi: usize) -> Gv {
    if lo == hi {
        return Gv::from_f64(lo as f64);
    }
    let mid = lo + (hi - lo) / 2;
    Gv::select(
        x.cmp_lt(Gv::from_f64(edges[mid])),
        upper_bound_traced(edges, x, lo, mid),
        upper_bound_traced(edges, x, mid + 1, hi),
    )
}

fn bin_traced(edges: &[f64], x: Gv) -> Gv {
    let n_bins = edges.len() - 1;
    let locator = bin_locator(edges);
    let arithmetic = match locator {
        BinLocator::Linear { lo, inv_step } => {
            ((x - Gv::from_f64(lo)) * Gv::from_f64(inv_step)).floor()
        }
        BinLocator::Log { ln_lo, inv_ln_step } => {
            ((x.ln() - Gv::from_f64(ln_lo)) * Gv::from_f64(inv_ln_step)).floor()
        }
        BinLocator::OrderedEdges => {
            return (upper_bound_traced(edges, x, 0, edges.len()) - Gv::ONE)
                .min(Gv::from_f64((n_bins - 1) as f64))
                .max(Gv::ZERO);
        }
    };
    // The arithmetic locators are the fast common case, but an exactly declared edge must enter
    // the bin on its right. `log(edge)` and multiplication by a reciprocal can round an integer
    // coordinate one ulp downward (for example 41 -> 40.99999...), so `floor` alone violates that
    // contract. Correct exact edges from the original coordinate, before clamping the outer edge
    // into the final bin. Equality is intentionally against the serialized edge itself: it does
    // not move a representable value immediately below the edge into the next bucket.
    let mut exact_edge = Gv::ZERO;
    let mut is_exact_edge = x.cmp_eq(Gv::from_f64(edges[0]));
    for (k, &edge) in edges.iter().enumerate().skip(1) {
        let equal = x.cmp_eq(Gv::from_f64(edge));
        exact_edge = Gv::select(equal, Gv::from_f64(k as f64), exact_edge);
        is_exact_edge = is_exact_edge | equal;
    }
    let raw = Gv::select(is_exact_edge, exact_edge, arithmetic);
    raw.min(Gv::from_f64((n_bins - 1) as f64)).max(Gv::ZERO)
}

/// the bucket index, branch-free, as the traced twin of the host search.
///
/// kept here, so this crate stays free of a dependency on the spec type; it is the same
/// algorithm, gated against the host's independent partition-point search:
/// `bin = #{edges at or below x} - 1`, clamped into the last bin so a value exactly on the outer
/// edge counts as data. a NaN coordinate compares false against both bounds and lands in the
/// drop segment.
fn segment_marker_traced<A: CensusAxis>(bin_axes: &[A], coords: &[Gv], n_segments: usize) -> Gv {
    let mut flat = Gv::ZERO;
    let mut all_in_range = Gv::ONE.cmp_gt(Gv::ZERO);
    for (axis, &x) in bin_axes.iter().zip(coords) {
        let edges = axis.edges();
        let n_bins = edges.len() - 1;
        all_in_range =
            all_in_range & x.cmp_ge(Gv::from_f64(edges[0])) & x.cmp_le(Gv::from_f64(edges[n_bins]));
        let bin = bin_traced(edges, x);
        flat = flat * Gv::from_f64(n_bins as f64) + bin;
    }
    Gv::select(all_in_range, flat, Gv::from_f64(n_segments as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locator_recognizes_linear_log_and_arbitrary_edges() {
        let linear: Vec<_> = (0..=64).map(|k| -2.0 + k as f64 * 0.125).collect();
        assert!(matches!(bin_locator(&linear), BinLocator::Linear { .. }));

        let lo = 1.0e-3_f64;
        let step = (1.0e3_f64 / lo).ln() / 64.0;
        let log: Vec<_> = (0..=64)
            .map(|k| (lo.ln() + k as f64 * step).exp())
            .collect();
        assert!(matches!(bin_locator(&log), BinLocator::Log { .. }));

        assert_eq!(bin_locator(&[1.0, 1.5, 3.0, 9.0]), BinLocator::OrderedEdges);
    }
}
