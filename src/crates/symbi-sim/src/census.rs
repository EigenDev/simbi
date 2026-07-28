// =============================================================================
// census.rs
//
// the binning algebra of a census: a pointwise map followed by a segmented reduce.
// a census declares
// - a list of bin axes, each a set of explicit edges over some coordinate;
// - a list of accumulators;
// - a reduce op,
// and this module turns per-cell axis coordinates into the destination bucket the
// segmented reduction scatters into.
//
// axes take an OUTER PRODUCT, so one radial axis gives shell profiles, a radial and
// an angular-momentum axis give the histogram per shell, and NO axes give a global
// reduction over the whole grid.
//
// the accumulated object must be a commutative monoid — associative and
// order-agnostic — or it cannot be reduced in parallel, blocked, or combined across
// restart segments. sums and extrema are; means, variances and percentiles are not,
// being functions of sums. so a census registers `m*v` and `m` and the reader
// divides.
//
// usage:
//  let axis = BinAxis::new("log_r", log_spaced_edges)?;
//  let spec = CensusSpec::new("shells", vec![axis], value_names, ReductionOp::Add)?;
//  let segment = spec.segment(&[log_r_at_cell]);
// =============================================================================

use symbi_ir::emit::ReductionOp;

/// the segment index marking a cell that is not part of the reduction at all — covered
/// by finer data, inside an immersed body's mask, or otherwise not physical gas.
/// distinct from a cell that falls outside the declared bin edges, which is a genuine
/// shortfall of the binning and is counted as such.
pub use symbi_ir::SEGMENT_EXCLUDED;

/// one bin axis: a coordinate to bin on, plus the edges that cut it.
///
/// edges are supplied explicitly rather than as a spacing rule, so linear spacing, log
/// spacing and hand-chosen edges all work without a spacing enum. `n` edges give
/// `n - 1` bins, and bin `k` covers `[edges[k], edges[k+1])`; the last bin is closed at
/// its upper edge so a value sitting exactly on the domain's outer boundary is not
/// silently dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct BinAxis {
    name: String,
    edges: Vec<f64>,
}

impl BinAxis {
    /// validate and take a set of edges. edges must be finite and strictly increasing:
    /// a repeated edge would create an empty bin that no value can ever land in, and a
    /// decreasing one would make the search silently return the wrong bin.
    pub fn new(name: impl Into<String>, edges: Vec<f64>) -> Result<Self, String> {
        let name = name.into();
        if edges.len() < 2 {
            return Err(format!(
                "bin axis '{name}': {} edge(s) define no bin; at least 2 are needed",
                edges.len()
            ));
        }
        for (k, e) in edges.iter().enumerate() {
            if !e.is_finite() {
                return Err(format!("bin axis '{name}': edge {k} is not finite ({e})"));
            }
        }
        for (k, w) in edges.windows(2).enumerate() {
            if !(w[1] > w[0]) {
                return Err(format!(
                    "bin axis '{name}': edges must strictly increase, but edge {k} = {} is not \
                     below edge {} = {}",
                    w[0],
                    k + 1,
                    w[1]
                ));
            }
        }
        Ok(BinAxis { name, edges })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// how many bins this axis cuts its coordinate into.
    pub fn n_bins(&self) -> usize {
        self.edges.len() - 1
    }

    /// which bin `x` falls in, or `None` when it lies outside the edges. NaN is outside
    /// every bin: a comparison against it is false in both directions, so placing it
    /// anywhere would be arbitrary.
    pub fn bin(&self, x: f64) -> Option<usize> {
        if !(x >= self.edges[0]) || x > self.edges[self.n_bins()] {
            return None;
        }
        // the index of the last edge at or below x. saturates at the final bin so a value
        // exactly on the outer edge lands in the last bin rather than one past it.
        let k = self.edges.partition_point(|&e| e <= x);
        Some(k.saturating_sub(1).min(self.n_bins() - 1))
    }
}

/// a registered census: what to bin on, what to accumulate, and how to combine.
#[derive(Clone, Debug, PartialEq)]
pub struct CensusSpec {
    name: String,
    axes: Vec<BinAxis>,
    /// one label per accumulator, in the order the value fields are supplied. the labels
    /// travel with the output so a reader can name a column without re-deriving the
    /// registration order.
    value_names: Vec<String>,
    op: ReductionOp,
}

impl CensusSpec {
    /// validate a registration. a census with no axes is a global reduction over the grid
    /// and is explicitly allowed — that is the case total mass and energy occupy.
    pub fn new(
        name: impl Into<String>,
        axes: Vec<BinAxis>,
        value_names: Vec<String>,
        op: ReductionOp,
    ) -> Result<Self, String> {
        let name = name.into();
        if value_names.is_empty() {
            return Err(format!("census '{name}': registers no values"));
        }
        // a product over a bin's cells overflows to zero or infinity at any realistic cell
        // count and is not a census statistic.
        if matches!(op, ReductionOp::Mul) {
            return Err(format!(
                "census '{name}': a product over a bin is not a meaningful reduction; use \
                 Add, Min or Max"
            ));
        }
        for (i, a) in axes.iter().enumerate() {
            for b in &axes[..i] {
                if a.name() == b.name() {
                    return Err(format!(
                        "census '{name}': two bin axes are both named '{}'",
                        a.name()
                    ));
                }
            }
        }
        Ok(CensusSpec {
            name,
            axes,
            value_names,
            op,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn axes(&self) -> &[BinAxis] {
        &self.axes
    }

    pub fn value_names(&self) -> &[String] {
        &self.value_names
    }

    pub fn n_values(&self) -> usize {
        self.value_names.len()
    }

    pub fn op(&self) -> ReductionOp {
        self.op
    }

    /// the total bucket count: the product of the per-axis bin counts. an empty product is
    /// one, which is exactly right — a census with no axes has a single bucket holding the
    /// global reduction.
    pub fn n_segments(&self) -> usize {
        self.axes.iter().map(|a| a.n_bins()).product()
    }

    /// the bucket a cell lands in, given its coordinate on each axis in registration
    /// order. `None` when any axis places the cell outside its edges — a cell must fall
    /// inside EVERY axis to be counted, since the bucket is a point in their outer product.
    ///
    /// the last axis varies fastest, so a reader reshaping the flat output to
    /// `[n_0, .., n_{K-1}]` in row-major order recovers the axes in registration order.
    pub fn segment(&self, coords: &[f64]) -> Option<usize> {
        assert_eq!(
            coords.len(),
            self.axes.len(),
            "census '{}': expects one coordinate per bin axis",
            self.name
        );
        let mut flat = 0usize;
        for (axis, &x) in self.axes.iter().zip(coords) {
            flat = flat * axis.n_bins() + axis.bin(x)?;
        }
        Some(flat)
    }

    /// the segment index to write for a cell, ready for the segmented reduction: the
    /// bucket if the cell bins, else an index past the last bucket so the reduction counts
    /// it as outside the binning.
    pub fn segment_marker(&self, coords: &[f64]) -> u32 {
        self.segment(coords)
            .map_or(self.n_segments() as u32, |s| s as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(edges: &[f64]) -> BinAxis {
        BinAxis::new("a", edges.to_vec()).expect("valid edges")
    }

    #[test]
    fn edges_must_strictly_increase() {
        // a repeated edge makes a bin no value can land in, and a decreasing one makes the
        // search return the wrong bin. both must be refused at registration, not produce a
        // census that quietly bins nothing.
        assert!(BinAxis::new("a", vec![0.0]).is_err(), "one edge is no bin");
        assert!(BinAxis::new("a", vec![]).is_err());
        let repeated = BinAxis::new("a", vec![0.0, 1.0, 1.0, 2.0]).unwrap_err();
        assert!(repeated.contains("strictly increase"), "{repeated}");
        let backwards = BinAxis::new("a", vec![0.0, 2.0, 1.0]).unwrap_err();
        assert!(backwards.contains("strictly increase"), "{backwards}");
        assert!(BinAxis::new("a", vec![0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn bins_are_half_open_with_a_closed_outer_edge() {
        let a = axis(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.n_bins(), 3);
        // a value on a bin's lower edge belongs to that bin.
        assert_eq!(a.bin(0.0), Some(0));
        assert_eq!(a.bin(1.0), Some(1));
        assert_eq!(a.bin(2.0), Some(2));
        assert_eq!(a.bin(0.5), Some(0));
        assert_eq!(a.bin(2.999), Some(2));
        // the outermost edge is closed, so a value sitting exactly on the domain boundary
        // is counted rather than silently dropped.
        assert_eq!(a.bin(3.0), Some(2));
        // outside, on either side.
        assert_eq!(a.bin(-1.0e-9), None);
        assert_eq!(a.bin(3.000001), None);
        // NaN compares false in both directions, so it belongs to no bin.
        assert_eq!(a.bin(f64::NAN), None);
    }

    #[test]
    fn no_axes_is_a_single_bucket_global_reduction() {
        // total mass and energy are a census with no bins. if the mechanism cannot express
        // that, it is not a generalization of the whole-field reduction.
        let spec = CensusSpec::new(
            "conservation",
            vec![],
            vec!["mass".into(), "energy".into()],
            ReductionOp::Add,
        )
        .expect("valid");
        assert_eq!(spec.n_segments(), 1);
        assert_eq!(spec.segment(&[]), Some(0));
        assert_eq!(spec.segment_marker(&[]), 0);
    }

    #[test]
    fn axes_take_an_outer_product_with_the_last_varying_fastest() {
        // two axes give the histogram per shell. the flat index must be row-major over the
        // axes in registration order, so a reader reshaping to [n_0, n_1] recovers them.
        let spec = CensusSpec::new(
            "phase",
            vec![
                BinAxis::new("r", vec![0.0, 1.0, 2.0]).unwrap(),
                BinAxis::new("xi", vec![0.0, 10.0, 20.0, 30.0]).unwrap(),
            ],
            vec!["m".into()],
            ReductionOp::Add,
        )
        .expect("valid");
        assert_eq!(spec.n_segments(), 6);
        assert_eq!(spec.segment(&[0.5, 5.0]), Some(0));
        assert_eq!(spec.segment(&[0.5, 15.0]), Some(1));
        assert_eq!(spec.segment(&[0.5, 25.0]), Some(2));
        assert_eq!(spec.segment(&[1.5, 5.0]), Some(3));
        assert_eq!(spec.segment(&[1.5, 25.0]), Some(5));
    }

    #[test]
    fn a_cell_outside_any_axis_falls_outside_the_binning() {
        // the bucket is a point in the outer product, so a cell inside the radial edges but
        // outside the angular ones has no bucket. binning it on the axes that DID match
        // would silently attribute it to the wrong shell.
        let spec = CensusSpec::new(
            "phase",
            vec![
                BinAxis::new("r", vec![0.0, 1.0, 2.0]).unwrap(),
                BinAxis::new("xi", vec![0.0, 10.0]).unwrap(),
            ],
            vec!["m".into()],
            ReductionOp::Add,
        )
        .expect("valid");
        assert_eq!(spec.segment(&[0.5, 5.0]), Some(0));
        assert_eq!(spec.segment(&[0.5, 50.0]), None);
        assert_eq!(spec.segment(&[9.0, 5.0]), None);
        // the marker handed to the reduction is past the last bucket, so the cell is
        // counted as outside the binning rather than folded into bucket zero.
        assert_eq!(spec.segment_marker(&[0.5, 50.0]), 2);
        assert!(spec.segment_marker(&[0.5, 50.0]) >= spec.n_segments() as u32);
        // and it is NOT the excluded marker: the cell was meant to be reduced.
        assert_ne!(spec.segment_marker(&[0.5, 50.0]), SEGMENT_EXCLUDED);
    }

    #[test]
    fn a_product_reduction_is_refused_at_registration() {
        let err = CensusSpec::new("bad", vec![], vec!["m".into()], ReductionOp::Mul).unwrap_err();
        assert!(err.contains("not a meaningful reduction"), "{err}");
    }

    #[test]
    fn a_census_must_register_at_least_one_value() {
        let err = CensusSpec::new("bad", vec![], vec![], ReductionOp::Add).unwrap_err();
        assert!(err.contains("registers no values"), "{err}");
    }

    #[test]
    fn duplicate_axis_names_are_refused() {
        // the axis name labels the edges in the output, so two axes sharing one is an
        // unreadable result rather than a harmless collision.
        let err = CensusSpec::new(
            "bad",
            vec![
                BinAxis::new("r", vec![0.0, 1.0]).unwrap(),
                BinAxis::new("r", vec![0.0, 1.0]).unwrap(),
            ],
            vec!["m".into()],
            ReductionOp::Add,
        )
        .unwrap_err();
        assert!(err.contains("both named 'r'"), "{err}");
    }

    #[test]
    fn log_spaced_edges_bin_a_decade_per_bin() {
        // shells are cut in log radius, so the edges are not uniform. explicit edges mean
        // that needs no spacing rule — the axis bins whatever is handed to it.
        let edges: Vec<f64> = (0..=4).map(|k| 10.0_f64.powi(k)).collect();
        let a = BinAxis::new("r", edges).unwrap();
        assert_eq!(a.n_bins(), 4);
        assert_eq!(a.bin(1.0), Some(0));
        assert_eq!(a.bin(9.99), Some(0));
        assert_eq!(a.bin(10.0), Some(1));
        assert_eq!(a.bin(1000.0), Some(3));
        assert_eq!(a.bin(10000.0), Some(3));
        assert_eq!(a.bin(0.5), None);
    }
}
