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

    /// build a spec from the serialized wire form the python front door emits. the expression
    /// dags themselves are lowered separately (`expr_bridge::build_census_expressions`); this
    /// takes the binning and the reduce, which are what the spec is responsible for.
    pub fn from_config(cfg: &symbi_hydro::CensusConfig) -> Result<Self, String> {
        let op = match cfg.op.as_str() {
            "add" => ReductionOp::Add,
            "min" => ReductionOp::Min,
            "max" => ReductionOp::Max,
            // a product is not a census statistic, and mean/variance/percentile are not
            // order-agnostic, so they cannot be reduced in parallel or combined across restart
            // segments. those are functions of sums the reader forms offline.
            other => {
                return Err(format!(
                    "census '{}': unknown reduce op '{other}' (expected add, min or max)",
                    cfg.name
                ));
            }
        };
        let axes = cfg
            .axes
            .iter()
            .map(|a| BinAxis::new(a.name.clone(), a.edges.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("census '{}': {e}", cfg.name))?;
        Self::new(cfg.name.clone(), axes, cfg.value_names.clone(), op)
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

/// a registered census with its expressions lowered and compiled — the runtime artifact.
///
/// the bin-axis coordinates and the accumulator values share ONE compiled graph, so a
/// subexpression both use (a radius, its logarithm) is evaluated once per cell. cost scales
/// with the size of that graph, not with the number of registered accumulators.
pub struct CensusEvaluator {
    spec: CensusSpec,
    eval: symbi_hydro::SourceEvaluator,
    /// the config's tunable parameter values, indexed by `PARAMETER` node index.
    params: Vec<f64>,
    n_nodes: usize,
}

/// the compiled-expression kernel key. a census lowers to a single graph, so one entry.
const CENSUS_FIELD: &str = "census";

// the compiled kernels are not printable, so the report is the registration's shape:
// what it bins on, how many buckets that makes, what it accumulates, and how big the
// per-cell graph is — the numbers worth seeing before a job is submitted.
impl std::fmt::Debug for CensusEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CensusEvaluator")
            .field("name", &self.spec.name())
            .field(
                "axes",
                &self
                    .spec
                    .axes()
                    .iter()
                    .map(|a| (a.name(), a.n_bins()))
                    .collect::<Vec<_>>(),
            )
            .field("segments", &self.spec.n_segments())
            .field("values", &self.spec.value_names())
            .field("op", &self.spec.op())
            .field("nodes", &self.n_nodes)
            .finish()
    }
}

impl CensusEvaluator {
    /// lower and compile a serialized census.
    pub fn new(cfg: &symbi_hydro::CensusConfig) -> Result<Self, String> {
        let spec = CensusSpec::from_config(cfg)?;
        let built = symbi_hydro::expr_bridge::build_census_expressions(cfg)?;
        let n_nodes = built.graph.len();
        let eval = symbi_hydro::SourceEvaluator::from_built(&[(CENSUS_FIELD.to_string(), built)]);
        Ok(CensusEvaluator {
            spec,
            eval,
            params: cfg.params.clone(),
            n_nodes,
        })
    }

    pub fn spec(&self) -> &CensusSpec {
        &self.spec
    }

    /// the compiled node count. a bin axis that recomputes a square root and a logarithm per
    /// cell is not free, and this is the number that says so before a job is submitted.
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }

    /// the declared per-cell parameter names, in the order the compiled kernel expects them.
    pub fn params_for(&self) -> &[String] {
        self.eval
            .params_for(CENSUS_FIELD)
            .expect("a census lowers to exactly one compiled field")
    }

    /// does any expression read the per-cell pressure? an isothermal regime carries no
    /// pressure field, so such a census must be refused rather than silently fed a zero.
    pub fn reads_pressure(&self) -> bool {
        self.params_for().iter().any(|p| p == "pre")
    }
}

/// resolve one census parameter name at a cell. the vocabulary is the source path's
/// (`x_k`, `t`, `rho`, `vel_k`, `pre`, `p{i}`) plus `dv`, the cell's lab-frame volume
/// measure — the weight that makes an extensive sum correct on a curvilinear grid.
#[allow(clippy::too_many_arguments)]
fn resolve_census_param<const D: usize>(
    name: &str,
    rho: f64,
    vel: &[f64],
    pre: f64,
    dv: f64,
    x: &[f64; D],
    t: f64,
    params: &[f64],
) -> f64 {
    match name {
        "rho" => return rho,
        "pre" => return pre,
        "dv" => return dv,
        "t" => return t,
        _ => {}
    }
    if let Some(k) = name.strip_prefix("vel_") {
        let k: usize = k.parse().expect("vel_ index");
        // an out-of-plane component the regime does not carry reads zero rather than
        // panicking: a 2.5D grid has a third velocity, a 2D one does not.
        return vel.get(k).copied().unwrap_or(0.0);
    }
    if let Some(k) = name.strip_prefix("x_") {
        let k: usize = k.parse().expect("x_ index");
        return x.get(k).copied().unwrap_or(0.0);
    }
    if let Some(i) = name.strip_prefix('p')
        && let Ok(i) = i.parse::<usize>()
    {
        return *params
            .get(i)
            .unwrap_or_else(|| panic!("census: param p{i} not provided"));
    }
    panic!("census: unresolved cell param '{name}' (rho | vel_k | pre | dv | x_k | t | p{{i}})")
}

/// the per-cell artifacts a census sample reduces over: one field per accumulator, plus the
/// destination bucket of every cell.
pub struct CensusFields<const D: usize, Mem: symbi_xpu::MemorySpace> {
    pub values: Vec<symbi_grid::Field<f64, D, Mem>>,
    pub segment: symbi_grid::Field<u32, D, Mem>,
}

impl<R, const D: usize, const DOF: usize, M, E, S, Mem>
    crate::state::SimStateGeneric<R, D, DOF, M, E, S, Mem, f64>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: symbi_xpu::MemorySpace,
{
    /// evaluate a census over this level's interior, producing the value fields and the
    /// bucket assignment a segmented reduction consumes.
    ///
    /// the bin coordinates and the values come from ONE compiled graph per cell, so a
    /// subexpression shared between an axis and a value is evaluated once. the `dv` leaf
    /// resolves to the block geometry's lab-frame cell volume — the same measure the
    /// finite-volume update uses — which is what keeps an extensive sum correct on a
    /// curvilinear grid.
    ///
    /// returns `None` when the fields are not host-accessible (a device-resident run); the
    /// device path renders its own kernel rather than reading cells from the host.
    pub fn census_fields(&self, ev: &CensusEvaluator) -> Option<CensusFields<D, Mem>> {
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        // an isothermal regime carries no pressure field. a census that reads `pre` there
        // would silently accumulate zeros, so it is refused instead.
        assert!(
            !(ev.reads_pressure() && self.fields.prim.pre.is_none()),
            "census '{}' reads the per-cell pressure, but this regime carries no pressure \
             field (isothermal)",
            ev.spec.name()
        );

        let bg = self.geom.block_geometry(self.physics.metric);
        // the lab-frame measure: on a homologously expanding mesh the conserved density
        // multiplies the PHYSICAL volume, so an extensive total stays constant as a(t) grows.
        let a = self.motion.a;
        let n_axes = ev.spec.axes().len();
        let n_values = ev.spec.n_values();

        let values: Vec<symbi_grid::Field<f64, D, Mem>> = (0..n_values)
            .map(|_| symbi_grid::Field::<f64, D, Mem>::zeros(&self.geom.allocated))
            .collect::<Result<_, _>>()
            .ok()?;
        let segment = symbi_grid::Field::<u32, D, Mem>::zeros(&self.geom.allocated).ok()?;
        // a cell the sweep never visits is not part of the reduction. ghost cells sit
        // outside the interior, so they must not read as bucket zero.
        for c in self.geom.allocated.iter() {
            segment.view_mut().set(c, SEGMENT_EXCLUDED);
        }

        let params = ev.params_for();
        // the census dag has a handful of leaves (position, time, the local primitives, dv,
        // the config's tunables); the fixed buffers keep the per-cell path allocation-free.
        const MAX_PARAMS: usize = 32;
        const MAX_OUT: usize = 64;
        assert!(
            params.len() <= MAX_PARAMS,
            "census '{}': more than {MAX_PARAMS} declared params",
            ev.spec.name()
        );
        assert!(
            n_axes + n_values <= MAX_OUT,
            "census '{}': more than {MAX_OUT} axis + value expressions",
            ev.spec.name()
        );

        let pre_field = self.fields.prim.pre.as_ref();
        let jit = ev.eval.jit_components(CENSUS_FIELD);
        for c in self.geom.interior.iter() {
            let rho = *self.fields.prim.rho.view().at(c);
            let vel: [f64; DOF] = std::array::from_fn(|k| *self.fields.prim.vel[k].view().at(c));
            let pre = pre_field.map_or(0.0, |f| *f.view().at(c));
            let dv = bg.labframe_volume(c, a);
            let x = self.geom.cell_coord(c);

            let mut inbuf = [0.0f64; MAX_PARAMS];
            for (i, p) in params.iter().enumerate() {
                inbuf[i] =
                    resolve_census_param::<D>(p, rho, &vel, pre, dv, &x, self.time, &ev.params);
            }
            let inputs = &inbuf[..params.len()];

            // the native path when every expression compiled, else the interpreter (only
            // when a node fell outside the jit subset).
            let mut out = [0.0f64; MAX_OUT];
            if let Some(jit) = jit {
                for (k, cf) in jit.iter().enumerate() {
                    cf.call(inputs, &mut out[k..k + 1]);
                }
            } else {
                let named: Vec<(&str, f64)> = params
                    .iter()
                    .zip(inputs)
                    .map(|(n, v)| (n.as_str(), *v))
                    .collect();
                let s = ev
                    .eval
                    .eval(CENSUS_FIELD, &named)
                    .expect("census: compiled field missing");
                out[..s.len()].copy_from_slice(&s);
            }

            // the outputs are the axis coordinates followed by the accumulator values,
            // matching `CensusConfig::output_nodes`.
            segment
                .view_mut()
                .set(c, ev.spec.segment_marker(&out[..n_axes]));
            for v in 0..n_values {
                values[v].view_mut().set(c, out[n_axes + v]);
            }
        }

        Some(CensusFields { values, segment })
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
