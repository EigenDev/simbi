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

use symbi_ir::algebra::Scalar;
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
    /// delegates to the CARRIER-GENERIC form, so the host and any traced device kernel run the
    /// same binning by construction rather than by two implementations happening to agree.
    pub fn segment_marker(&self, coords: &[f64]) -> u32 {
        segment_marker_generic::<f64>(&self.axes, coords, self.n_segments()) as u32
    }
}

/// the bucket a cell's axis coordinates fall in, as a CARRIER-GENERIC branch-free expression:
/// the flat segment index, or `n_segments` (one past the last bucket) when any axis coordinate
/// lies outside its declared edges, which is what the reduction counts as dropped.
///
/// ONE definition for every carrier. `S = f64` evaluates it on the host; `S = Gv` traces it into a
/// kernel. the binning is the part of a census a device path would otherwise have to reimplement,
/// and a host/device split there is invisible: both produce a smooth, plausible profile, and only
/// their disagreement — which nothing would be comparing — reveals it.
///
/// the search is a COUNT rather than a branchy partition point: `bin = #{edges at or below x} - 1`,
/// saturated at the last bin so a value sitting exactly on the outer edge lands in it rather than
/// one past. edges are known at registration, so the loop unrolls at trace time. a NaN coordinate
/// compares false against both bounds and is therefore dropped, never binned.
pub fn segment_marker_generic<S: Scalar>(axes: &[BinAxis], coords: &[S], n_segments: usize) -> S {
    debug_assert_eq!(axes.len(), coords.len(), "one coordinate per bin axis");
    let mut flat = S::ZERO;
    let mut all_in_range = S::ONE.cmp_gt(S::ZERO); // a true mask, carrier-generically
    for (axis, &x) in axes.iter().zip(coords) {
        let edges = axis.edges();
        let n_bins = axis.n_bins();
        let lo = S::from_f64(edges[0]);
        let hi = S::from_f64(edges[n_bins]);
        all_in_range = all_in_range & x.cmp_ge(lo) & x.cmp_le(hi);

        let mut count = S::ZERO;
        for &edge in edges {
            count = count + S::select(x.cmp_ge(S::from_f64(edge)), S::ONE, S::ZERO);
        }
        // count >= 1 on any in-range x, so `count - 1` is the index of the last edge at or
        // below it; the clamp keeps the outer edge in the final bin.
        let bin = (count - S::ONE).min(S::from_f64((n_bins - 1) as f64)).max(S::ZERO);
        flat = flat * S::from_f64(n_bins as f64) + bin;
    }
    S::select(all_in_range, flat, S::from_f64(n_segments as f64))
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

/// the per-sample time series a registered census produces, columnar so the checkpoint
/// writer borrows each series as one flat dataset.
///
/// the series covers THIS RUN SEGMENT ONLY and restarts empty on checkpoint load — earlier
/// segments live in the earlier checkpoint files, and segments concatenate offline. keeping
/// the accumulator out of restart state is what lets a chain of restarts be combined by a
/// reader without the run having to carry the whole history forward.
#[derive(Clone, Debug)]
pub struct CensusHistory {
    n_segments: usize,
    n_values: usize,
    /// simulation time of each sample: shape [len].
    time: Vec<f64>,
    /// the accumulators, segment-major within a sample: shape [len, n_segments, n_values].
    values: Vec<f64>,
    /// cells that fell outside the binning, per sample: shape [len].
    dropped: Vec<u64>,
}

impl CensusHistory {
    pub fn new(n_segments: usize, n_values: usize) -> Self {
        CensusHistory {
            n_segments,
            n_values,
            time: Vec::new(),
            values: Vec::new(),
            dropped: Vec::new(),
        }
    }

    /// append one sample. the accumulator count is fixed by the registration, so a
    /// mismatch is a wiring error rather than a data condition.
    pub fn push(&mut self, time: f64, values: &[f64], dropped: u64) {
        assert_eq!(
            values.len(),
            self.n_segments * self.n_values,
            "census sample has {} accumulators; the registration declares {} segments x {} values",
            values.len(),
            self.n_segments,
            self.n_values
        );
        self.time.push(time);
        self.values.extend_from_slice(values);
        self.dropped.push(dropped);
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }

    pub fn n_segments(&self) -> usize {
        self.n_segments
    }

    pub fn n_values(&self) -> usize {
        self.n_values
    }

    pub fn time(&self) -> &[f64] {
        &self.time
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn dropped(&self) -> &[u64] {
        &self.dropped
    }
}

/// a registered census plus the samples taken so far this run segment.
pub struct RegisteredCensus {
    pub evaluator: CensusEvaluator,
    pub history: CensusHistory,
}

impl RegisteredCensus {
    pub fn new(evaluator: CensusEvaluator) -> Self {
        let history = CensusHistory::new(
            evaluator.spec().n_segments(),
            evaluator.spec().n_values(),
        );
        RegisteredCensus { evaluator, history }
    }
}

impl std::fmt::Debug for RegisteredCensus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredCensus")
            .field("evaluator", &self.evaluator)
            .field("samples", &self.history.len())
            .finish()
    }
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
    /// `covered` is this level's cells that a FINER level resolves. they are excluded from the
    /// reduction: the finer level contributes the same physical volume at its own resolution, so
    /// counting both would add the refined region twice — a total that is wrong by exactly the
    /// refined volume and otherwise entirely plausible. a level with no finer neighbour passes
    /// `None` and every interior cell is a leaf.
    ///
    /// returns `None` when the fields are not host-accessible (a device-resident run); the
    /// device path renders its own kernel rather than reading cells from the host.
    pub fn census_fields(
        &self,
        ev: &CensusEvaluator,
        covered: Option<&symbi_algebra::Domain<D>>,
    ) -> Option<CensusFields<D, Mem>> {
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        // a census reads the PRIMITIVES, and seeding writes only the conserved state. on a
        // store whose primitives have never been recovered they are still zeros, so a
        // census sampled there would report a total mass of zero as though it were physics.
        assert!(
            self.store.has_recovered_primitives(),
            "census '{}' sampled before the conserved-to-primitive recovery has run; the \
             primitive fields are still empty, so every accumulator would read zero",
            ev.spec.name()
        );
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
            // a covered cell is not a leaf: it stays EXCLUDED, so the finer level owns that
            // volume outright and the reduction visits it exactly once.
            if covered.is_some_and(|region| region.contains(c)) {
                continue;
            }
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

#[cfg(test)]
mod generic_binning_tests {
    use super::*;

    fn spec(axes: Vec<BinAxis>) -> CensusSpec {
        CensusSpec::new("t", axes, vec!["v".to_string()], ReductionOp::Add).expect("spec")
    }

    /// the carrier-generic binning must agree EXACTLY with the independent partition-point
    /// implementation, on every coordinate a cell can present.
    ///
    /// compared against `segment` — the `edges.partition_point` search — and NOT against
    /// `segment_marker`, which now delegates to the generic form and would compare the function
    /// to itself. two genuinely different searches (a binary partition versus a linear count of
    /// edges at or below x) agreeing on the whole edge-case sweep is the evidence; the delegation
    /// is then what keeps host and device from drifting apart afterwards.
    #[test]
    fn the_generic_marker_matches_the_host_marker_everywhere() {
        let cases: Vec<Vec<BinAxis>> = vec![
            vec![BinAxis::new("a", vec![0.0, 1.0, 2.0, 3.0]).unwrap()],
            vec![BinAxis::new("a", vec![-2.0, -0.5, 0.5, 4.0, 9.0]).unwrap()],
            vec![
                BinAxis::new("a", vec![0.0, 1.0, 2.0]).unwrap(),
                BinAxis::new("b", vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
            ],
        ];
        for axes in cases {
            let sp = spec(axes.clone());
            let n_seg = sp.n_segments();
            // a sweep that lands ON every edge, just inside and just outside each, between
            // edges, far outside both ends, and NaN — the coordinates where a binning rule
            // differs from a neighbouring one.
            let mut probes: Vec<f64> = vec![f64::NAN, -1.0e300, 1.0e300];
            for axis in &axes {
                for &e in axis.edges() {
                    probes.extend([e, e - 1.0e-12, e + 1.0e-12, e - 0.5, e + 0.5]);
                }
            }
            for &x in &probes {
                for &y in &probes {
                    let coords: Vec<f64> = if axes.len() == 1 { vec![x] } else { vec![x, y] };
                    let want = sp.segment(&coords).map_or(n_seg, |b| b);
                    let got = segment_marker_generic::<f64>(&axes, &coords, n_seg);
                    assert_eq!(
                        got, want as f64,
                        "binning disagrees at {coords:?}: counting search {got} vs partition \
                         point {want}"
                    );
                    if axes.len() == 1 {
                        break;
                    }
                }
            }
        }
    }

    /// the outer edge is IN the last bin, not one past it: a value sitting exactly on the
    /// domain's outer boundary is real data, and dropping it would quietly under-count the
    /// outermost shell of every profile.
    #[test]
    fn a_value_on_the_outer_edge_lands_in_the_last_bin() {
        let axes = vec![BinAxis::new("a", vec![0.0, 1.0, 2.0]).unwrap()];
        let sp = spec(axes.clone());
        assert_eq!(segment_marker_generic::<f64>(&axes, &[2.0], sp.n_segments()), 1.0);
        // and just past it is dropped, not folded back in.
        assert_eq!(
            segment_marker_generic::<f64>(&axes, &[2.0 + 1.0e-9], sp.n_segments()),
            sp.n_segments() as f64
        );
    }
}
