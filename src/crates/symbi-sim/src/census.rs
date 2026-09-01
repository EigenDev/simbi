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
// axes take an outer product, so one radial axis gives shell profiles, a radial and
// an angular-momentum axis give the histogram per shell, and an empty axis list gives
// a global reduction over the whole grid.
//
// the accumulated object must be a commutative monoid — associative and
// order-agnostic — for the reduction to run in parallel, blocked, and combined across
// restart segments. sums and extrema qualify; means, variances and percentiles are
// functions of sums and stand outside it. so a census registers `m*v` and `m` and the
// reader divides.
//
// usage:
//  let axis = BinAxis::new("log_r", log_spaced_edges)?;
//  let spec = CensusSpec::new("shells", vec![axis], value_names, ReductionOp::Add)?;
//  let segment = spec.segment(&[log_r_at_cell]);
// =============================================================================

use symbi_ir::algebra::Scalar;
use symbi_ir::emit::ReductionOp;

/// how far past the last bucket the excluded marker sits — a cell outside the reduction's scope
/// entirely: covered by finer data, inside an immersed body's mask, a ghost, or otherwise
/// something other than physical gas. distinct from a cell that falls outside the declared bin
/// edges, which was in scope and is a genuine shortfall of the binning, counted as such.
///
/// an offset rather than a fixed sentinel because the segment travels on the scalar carrier, so
/// every marker must stay a small integer to be exact in f32 as well as f64.
pub use symbi_ir::SEGMENT_EXCLUDED_OFFSET;

/// one bin axis: a coordinate to bin on, plus the edges that cut it.
///
/// edges are supplied explicitly rather than as a spacing rule, so linear spacing, log
/// spacing and hand-chosen edges all work through one representation. `n` edges give
/// `n - 1` bins, and bin `k` covers `[edges[k], edges[k+1])`; the last bin is closed at
/// its upper edge, so a value sitting exactly on the domain's outer boundary lands in
/// that bin.
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

/// when a census samples on a refinement hierarchy.
///
/// levels are time-aligned at root-step boundaries: level `l` subcycles once per parent step,
/// so its clock runs ahead of its parent's within a step and meets it again at the end.
///
/// `RootStep` reduces every level's leaf cells into one sample at that meeting point, so a row is
/// a consistent snapshot of the whole composite domain. `PerLevelStep` instead lets each level
/// sample on its own subcycle, tagged with its own time and level. the second is the better
/// statistic wherever refinement tracks the flow: with cell width scaling as radius and a sound
/// speed going as `r^{-1/2}`, a level's timestep scales as `r^{3/2}` — the same scaling as the
/// eddy turnover time — so samples per correlation time come out level-independent, and every
/// radius is sampled equally well in units of its own decorrelation. root-step sampling
/// under-resolves exactly the innermost, fastest-decorrelating shells.
///
/// what per-level sampling costs is a time skew of at most one root step between levels, which is
/// below the coarse level's own turnover; each row carries its level's time so a consumer can
/// account for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cadence {
    #[default]
    RootStep,
    PerLevelStep,
}

impl Cadence {
    /// the wire tag, matching the python front door.
    pub fn tag(&self) -> &'static str {
        match self {
            Cadence::RootStep => "root_step",
            Cadence::PerLevelStep => "per_level_step",
        }
    }

    pub fn from_tag(tag: &str) -> Result<Self, String> {
        match tag {
            "root_step" => Ok(Cadence::RootStep),
            "per_level_step" => Ok(Cadence::PerLevelStep),
            other => Err(format!(
                "unknown census cadence '{other}' (expected root_step or per_level_step)"
            )),
        }
    }
}

/// a registered census: what to bin on, what to accumulate, and how to combine.
#[derive(Clone, Debug, PartialEq)]
pub struct CensusSpec {
    name: String,
    /// shortest simulation-time interval between samples; `None` samples every step.
    sample_interval: Option<f64>,
    /// fold every sample into a single running row rather than emitting a row apiece.
    accumulate: bool,
    /// whether a hierarchy samples at root-step boundaries or on each level's own subcycle.
    cadence: Cadence,
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
        // count, which puts it outside the set of usable census statistics.
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
            sample_interval: None,
            accumulate: false,
            cadence: Cadence::default(),
            axes,
            value_names,
            op,
        })
    }

    /// set the shortest simulation-time interval between samples. `None` samples every step.
    ///
    /// a non-positive interval is refused rather than clamped: zero would sample every step, which
    /// is what `None` already means, and a negative one is a sign the caller computed it rather
    /// than chose it.
    pub fn every(mut self, interval: Option<f64>) -> Result<Self, String> {
        if let Some(dt) = interval
            && !(dt > 0.0)
        {
            return Err(format!(
                "census '{}': sample interval {dt} is not positive; omit it to sample every step",
                self.name
            ));
        }
        self.sample_interval = interval;
        Ok(self)
    }

    /// whether a sample is due at `now`, given when the last one was taken.
    ///
    /// `last` is `None` before the first sample, which is always due — a census that recorded
    /// nothing until one interval had elapsed would silently omit the initial state, which is the
    /// one sample a reader can check against the problem's own setup.
    pub fn is_due(&self, now: f64, last: Option<f64>) -> bool {
        match (self.sample_interval, last) {
            (None, _) | (_, None) => true,
            (Some(dt), Some(prev)) => now - prev >= dt,
        }
    }

    /// the configured interval, for a caller reporting the registration.
    pub fn sample_interval(&self) -> Option<f64> {
        self.sample_interval
    }

    /// fold every sample into a single running row in place of a row apiece.
    ///
    /// the fold is the census's own reduce op, extended over time: an additive census carries the
    /// running total, from which the reader forms a time average by dividing by the sample count,
    /// and an extremal one carries the extremum over space and time together. that consistency is
    /// what makes the mode safe — one commutative monoid merges two cells, two refinement levels
    /// and two samples alike, so a single combining rule governs all three.
    ///
    /// the motivation is storage: a two-dimensional histogram runs to order a hundred kilobytes
    /// per sample, and a run that only ever wanted the segment's time average would otherwise
    /// write every one of them to disk in order to average them back down.
    pub fn accumulating(mut self, accumulate: bool) -> Self {
        self.accumulate = accumulate;
        self
    }

    pub fn accumulate(&self) -> bool {
        self.accumulate
    }

    /// set when a hierarchy samples this census.
    pub fn at_cadence(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }

    pub fn cadence(&self) -> Cadence {
        self.cadence
    }

    /// build a spec from the serialized wire form the python front door emits. the expression
    /// dags themselves are lowered separately (`expr_bridge::build_census_expressions`); this
    /// takes the binning and the reduce, which are what the spec is responsible for.
    pub fn from_config(cfg: &symbi_hydro::CensusConfig) -> Result<Self, String> {
        let op = match cfg.op.as_str() {
            "add" => ReductionOp::Add,
            "min" => ReductionOp::Min,
            "max" => ReductionOp::Max,
            // add, min and max are the commutative monoids that reduce in parallel and combine
            // across restart segments. a product overflows, and mean/variance/percentile depend
            // on order; those are functions of sums the reader forms offline.
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
        let cadence =
            Cadence::from_tag(&cfg.cadence).map_err(|e| format!("census '{}': {e}", cfg.name))?;
        Ok(
            Self::new(cfg.name.clone(), axes, cfg.value_names.clone(), op)?
                .every(cfg.sample_interval)?
                .accumulating(cfg.accumulate)
                .at_cadence(cadence),
        )
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
    /// order. `None` when any axis places the cell outside its edges — a counted cell falls
    /// inside every axis, since the bucket is a point in their outer product.
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

    /// the marker for a cell outside the reduction's scope — a ghost, a cell a finer level
    /// resolves, anything other than physical gas. it sits strictly above the outside-the-edges
    /// marker so the reduction can tell the two apart: a cell that fell outside the declared edges
    /// was in scope and is a shortfall worth reporting, while an excluded one sits outside scope
    /// by construction.
    pub fn excluded_marker(&self) -> f64 {
        (self.n_segments() + SEGMENT_EXCLUDED_OFFSET as usize) as f64
    }

    /// the segment index to write for a cell, ready for the segmented reduction: the
    /// bucket if the cell bins, else an index past the last bucket so the reduction counts
    /// it as outside the binning.
    /// delegates to the carrier-generic form, so the host and any traced device kernel run the
    /// same binning by construction rather than by two implementations happening to agree.
    pub fn segment_marker(&self, coords: &[f64]) -> u32 {
        segment_marker_generic::<f64>(&self.axes, coords, self.n_segments()) as u32
    }
}

/// the bucket a cell's axis coordinates fall in, as a carrier-generic branch-free expression:
/// the flat segment index, or `n_segments` (one past the last bucket) when any axis coordinate
/// lies outside its declared edges, which is what the reduction counts as dropped.
///
/// one definition for every carrier. `S = f64` evaluates it on the host; `S = Gv` traces it into a
/// kernel. the binning is the part of a census a device path would otherwise have to reimplement,
/// and a host/device split there hides itself: both produce a smooth, plausible profile, and their
/// disagreement — which nothing would be comparing — is the sole tell.
///
/// the search is a count rather than a branchy partition point: `bin = #{edges at or below x} - 1`,
/// saturated at the last bin so a value sitting exactly on the outer edge lands in it rather than
/// one past. edges are known at registration, so the loop unrolls at trace time. a NaN coordinate
/// compares false against both bounds and is therefore dropped.
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
        let bin = (count - S::ONE)
            .min(S::from_f64((n_bins - 1) as f64))
            .max(S::ZERO);
        flat = flat * S::from_f64(n_bins as f64) + bin;
    }
    S::select(all_in_range, flat, S::from_f64(n_segments as f64))
}

/// a registered census with its expressions lowered and compiled — the runtime artifact.
///
/// the bin-axis coordinates and the accumulator values share one compiled graph, so a
/// subexpression both use (a radius, its logarithm) is evaluated once per cell. cost scales
/// with the size of that graph, independent of how many accumulators are registered.
pub struct CensusEvaluator {
    spec: CensusSpec,
    eval: symbi_hydro::SourceEvaluator,
    /// the registration, retained so a compiled path can lower the same expressions the
    /// interpreter walks. the config is held in place of the lowered `BuiltSource`, which carries
    /// a `proc_macro2::Span` and so lacks `Sync`; a store holding one would be unshareable with
    /// the rayon closures every parallel pass takes over it. lowering from the config on demand
    /// keeps both paths sourced from one registration and the store `Sync`.
    cfg: symbi_hydro::CensusConfig,
    /// the config's tunable parameter values, indexed by `PARAMETER` node index.
    params: Vec<f64>,
    n_nodes: usize,
}

/// the compiled-expression kernel key. a census lowers to a single graph, so one entry.
const CENSUS_FIELD: &str = "census";

// the report is the registration's shape, the compiled kernels being opaque: what it
// bins on, how many buckets that makes, what it accumulates, and how big the per-cell
// graph is — the numbers worth seeing before a job is submitted.
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
        let n_nodes = built.graph().len();
        let eval = symbi_hydro::SourceEvaluator::from_built(&[(CENSUS_FIELD.to_string(), built)]);
        Ok(CensusEvaluator {
            spec,
            eval,
            cfg: cfg.clone(),
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
    /// a content hash of the registration: two evaluators share it iff they describe the same
    /// census, whatever they are named.
    ///
    /// the name is not a usable identity for a cached artifact. names are unique within a run —
    /// the registration refuses a duplicate — but a process may run several sims, and a parameter
    /// sweep naturally reuses one name for censuses whose graphs differ. keying a compiled kernel
    /// on the name hands the second run the first one's kernel, which is a shape mismatch at best
    /// and silently wrong numbers at worst.
    pub fn content_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        // the debug rendering rather than a serializer: `CensusConfig` derives `Debug`, the
        // rendering is total over its fields, and this is an in-process cache key — it never
        // crosses a process or a version boundary, so stability across builds is not required of
        // it, only that two distinct registrations render differently.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}", self.cfg).hash(&mut h);
        h.finish()
    }

    /// lower the registration again, for a caller tracing the same expressions into a kernel.
    /// the same `CensusConfig` the interpreter was built from, so the two cannot describe
    /// different censuses.
    pub fn lower(&self) -> Result<symbi_hydro::source_spec::BuiltSource, String> {
        symbi_hydro::expr_bridge::build_census_expressions(&self.cfg)
    }

    /// the registration's tunable parameter values, indexed by `PARAMETER` node index.
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// whether every bin-axis expression depends only on geometry and fixed parameters.
    ///
    /// Such an axis has the same bucket assignment until the mesh geometry changes, so the
    /// per-level segment field can be cached while the state-dependent accumulator values are
    /// refreshed. This is deliberately a dependency proof over the registered DAG: an axis that
    /// reads time or any primitive is never guessed static from its name or current values.
    pub fn axes_are_geometry_only(&self) -> bool {
        let mut dynamic = vec![false; self.cfg.nodes.len()];
        for (i, node) in self.cfg.nodes.iter().enumerate() {
            let leaf_is_dynamic = matches!(
                node.op.as_str(),
                "VARIABLE_T"
                    | "VARIABLE_RHO"
                    | "VARIABLE_VEL1"
                    | "VARIABLE_VEL2"
                    | "VARIABLE_VEL3"
                    | "VARIABLE_PRESSURE"
            );
            let dependency_is_dynamic = [
                node.left,
                node.right,
                node.condition,
                node.true_case,
                node.false_case,
            ]
            .into_iter()
            .flatten()
            .any(|j| dynamic.get(j).copied().unwrap_or(true));
            dynamic[i] = leaf_is_dynamic || dependency_is_dynamic;
        }
        self.cfg
            .axes
            .iter()
            .all(|axis| !dynamic.get(axis.expr).copied().unwrap_or(true))
    }

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
/// the series covers this run segment only and restarts empty on checkpoint load — earlier
/// segments live in the earlier checkpoint files, and segments concatenate offline. keeping
/// the accumulator out of restart state is what lets a chain of restarts be combined by a
/// reader without the run having to carry the whole history forward.
#[derive(Clone, Debug)]
pub struct CensusHistory {
    n_segments: usize,
    n_values: usize,
    /// how two samples combine when accumulating: the census's own reduce op, extended over time.
    op: ReductionOp,
    /// fold every sample into one row rather than storing a row apiece.
    accumulate: bool,
    /// simulation time of each stored row: shape [len]. for an accumulating row this is the time
    /// of the last sample folded into it; `t_start` carries the other end.
    time: Vec<f64>,
    /// which refinement level produced each row: shape [len]. always zero for a census sampled at
    /// root-step cadence, where every level's partial is combined into one row before it is
    /// recorded; a per-level census records each level's own subcycle separately, and without the
    /// tag a consumer could not tell a level-2 row taken four times per root step from a root row.
    level: Vec<u64>,
    /// the accumulators, segment-major within a row: shape [len, n_segments, n_values].
    values: Vec<f64>,
    /// cells that fell outside the binning, per row: shape [len]. accumulating, this is the
    /// running total over every sample folded in, since it is a count of the same kind.
    dropped: Vec<u64>,
    /// samples folded into each row: shape [len]. all ones unless accumulating, and the divisor
    /// that turns an accumulated additive row back into a time average.
    n_samples: Vec<u64>,
    /// simulation time of the first sample folded into each row: shape [len].
    t_start: Vec<f64>,
}

impl CensusHistory {
    pub fn new(n_segments: usize, n_values: usize) -> Self {
        Self::with_mode(n_segments, n_values, ReductionOp::Add, false)
    }

    pub fn with_mode(
        n_segments: usize,
        n_values: usize,
        op: ReductionOp,
        accumulate: bool,
    ) -> Self {
        CensusHistory {
            n_segments,
            n_values,
            op,
            accumulate,
            time: Vec::new(),
            level: Vec::new(),
            values: Vec::new(),
            dropped: Vec::new(),
            n_samples: Vec::new(),
            t_start: Vec::new(),
        }
    }

    /// record one sample taken on the root level.
    pub fn push(&mut self, time: f64, values: &[f64], dropped: u64) {
        self.push_at_level(time, 0, values, dropped);
    }

    /// record one sample from refinement level `level`. the accumulator count is fixed by the
    /// registration, so a mismatch is a wiring error rather than a data condition.
    ///
    /// accumulating, the sample is folded into that level's existing row with the census's reduce
    /// op rather than appended — the same operator that merges two refinement levels' partials, so
    /// a row is the reduction over its whole space-time segment and no separate combining rule
    /// exists to disagree with the spatial one. the fold is per level: levels subcycle at
    /// different rates and cover different volumes, so merging their rows would weight the segment
    /// by the subcycle ratio rather than by anything physical.
    pub fn push_at_level(&mut self, time: f64, level: u64, values: &[f64], dropped: u64) {
        assert_eq!(
            values.len(),
            self.n_segments * self.n_values,
            "census sample has {} accumulators; the registration declares {} segments x {} values",
            values.len(),
            self.n_segments,
            self.n_values
        );
        let stride = self.n_segments * self.n_values;
        if self.accumulate
            && let Some(row) = self.level.iter().position(|&l| l == level)
        {
            for (acc, add) in self.values[row * stride..(row + 1) * stride]
                .iter_mut()
                .zip(values)
            {
                *acc = combine(self.op, *acc, *add);
            }
            self.time[row] = time;
            self.dropped[row] += dropped;
            self.n_samples[row] += 1;
            return;
        }
        self.time.push(time);
        self.level.push(level);
        self.values.extend_from_slice(values);
        self.dropped.push(dropped);
        self.n_samples.push(1);
        self.t_start.push(time);
    }

    /// whether this history folds its samples into one row per level.
    pub fn accumulate(&self) -> bool {
        self.accumulate
    }

    /// samples folded into each row; all ones unless accumulating.
    pub fn n_samples(&self) -> &[u64] {
        &self.n_samples
    }

    /// simulation time of the first sample folded into each row.
    pub fn t_start(&self) -> &[f64] {
        &self.t_start
    }

    /// the refinement level each row was sampled on.
    pub fn level(&self) -> &[u64] {
        &self.level
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

/// fold two accumulator values with a reduce op.
///
/// the one definition of a census's combining rule outside the rendered reduction kernel: it
/// merges two refinement levels' partials and two samples of an accumulating history. requiring
/// one commutative monoid for both is what makes those combinations order-agnostic — levels are
/// visited in whatever order the hierarchy holds them, and samples arrive at whatever cadence the
/// timestepper produces.
pub fn combine(op: ReductionOp, a: f64, b: f64) -> f64 {
    match op {
        ReductionOp::Add => a + b,
        ReductionOp::Min => a.min(b),
        ReductionOp::Max => a.max(b),
        ReductionOp::Mul => a * b,
    }
}

/// a registered census plus the samples taken so far this run segment.
pub struct RegisteredCensus {
    pub evaluator: CensusEvaluator,
    pub history: CensusHistory,
    /// the time of the most recent sample per refinement level, or `None` before the first.
    ///
    /// per level because levels subcycle independently: a shared marker would let whichever level
    /// happened to sample first satisfy the interval for all of them, so the coarse levels would
    /// record only the fraction of samples the finest one left them.
    ///
    /// carried on the registration rather than derived from the history because the history
    /// restarts empty on a checkpoint load, and a restarted run should resume its cadence rather
    /// than sample immediately and then drift by one interval for the rest of the segment.
    pub last_sample: Vec<Option<f64>>,
}

impl RegisteredCensus {
    /// whether this registration is due to sample level `level` at `now`.
    pub fn is_due_at_level(&self, now: f64, level: usize) -> bool {
        let last = self.last_sample.get(level).copied().flatten();
        self.evaluator.spec().is_due(now, last)
    }

    /// whether this registration is due on the root level.
    pub fn is_due(&self, now: f64) -> bool {
        self.is_due_at_level(now, 0)
    }

    /// record that level `level` sampled at `now`.
    pub fn mark_sampled(&mut self, level: usize, now: f64) {
        if self.last_sample.len() <= level {
            self.last_sample.resize(level + 1, None);
        }
        self.last_sample[level] = Some(now);
    }

    pub fn new(evaluator: CensusEvaluator) -> Self {
        let spec = evaluator.spec();
        let history = CensusHistory::with_mode(
            spec.n_segments(),
            spec.n_values(),
            spec.op(),
            spec.accumulate(),
        );
        RegisteredCensus {
            evaluator,
            history,
            last_sample: Vec::new(),
        }
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
///
/// double precision independent of the simulation's own carrier. an extensive accumulator is a sum
/// over every cell of a bin, so on a few million cells a single-precision running sum outgrows the
/// terms still being added to it and absorbs the whole tail — a total that is smooth, positive and
/// wrong in its third digit. the artifacts are the census's own, so widening them here costs one
/// buffer per accumulator and settles the question for every backend.
pub struct CensusFields<const D: usize, Mem: symbi_xpu::MemorySpace> {
    pub values: Vec<symbi_grid::Field<f64, D, Mem>>,
    pub segment: symbi_grid::Field<f64, D, Mem>,
    /// geometry/coverage stamp of a reusable, geometry-only segment field. `None` means the next
    /// sample must produce the segment along with its values.
    pub segment_stamp: std::sync::Mutex<Option<u64>>,
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
    /// the bin coordinates and the values come from one compiled graph per cell, so a
    /// subexpression shared between an axis and a value is evaluated once. the `dv` leaf
    /// resolves to the block geometry's lab-frame cell volume — the same measure the
    /// finite-volume update uses — which is what keeps an extensive sum correct on a
    /// curvilinear grid.
    ///
    /// `covered` is this level's cells that a finer level resolves. they are excluded from the
    /// reduction: the finer level contributes the same physical volume at its own resolution, so
    /// counting both would add the refined region twice — a total that is wrong by exactly the
    /// refined volume and otherwise entirely plausible. a level with no finer neighbor passes
    /// `None` and every interior cell is a leaf.
    ///
    /// returns `None` when the fields are not host-accessible (a device-resident run); the
    /// device path renders its own kernel rather than reading cells from the host.
    pub fn census_fields(
        &self,
        ev: &CensusEvaluator,
        covered: Option<&symbi_algebra::Domain<D>>,
    ) -> Option<CensusFields<D, Mem>> {
        let fields = self.census_scratch(ev)?;
        self.census_fill_interpreted(ev, &fields)?;
        if let Some(region) = covered {
            self.census_exclude_covered(ev, &fields, region);
        }
        Some(fields)
    }

    /// the persistent scratch of registration `index`, allocated on first use and reused after.
    ///
    /// the artifacts of a census have a fixed shape for the life of a run — one full-grid field per
    /// accumulator plus the bucket — so reallocating them per sample moves that memory for no
    /// information. reuse is sound because a fill writes every interior cell and the ghosts, which
    /// are the only cells a fill does not reach, are excluded once at allocation and never move.
    ///
    /// the pool is built for every registration at once, since it is published through a single
    /// write; `index` is a position in `censuses`.
    ///
    /// the registration list is passed in rather than read from this store because on a refinement
    /// hierarchy the registrations live on the root alone, while the scratch they are evaluated
    /// into belongs to whichever level is being reduced.
    pub fn census_scratch_pooled(
        &self,
        censuses: &[RegisteredCensus],
        index: usize,
    ) -> Option<&CensusFields<D, Mem>> {
        if self.store.workspace.census_scratch.get().is_none() {
            let pool: Vec<CensusFields<D, Mem>> = censuses
                .iter()
                .map(|registered| self.census_scratch(&registered.evaluator))
                .collect::<Option<_>>()?;
            let _ = self.store.workspace.census_scratch.set(pool);
        }
        let pool = self.store.workspace.census_scratch.get()?;
        assert!(
            index < pool.len(),
            "census registration {index} was sampled, but the scratch pool holds {} entry(ies). \
             the pool is sized once from the registration list, so a census registered after the \
             first sample has no artifacts and would otherwise reduce another census's buffers",
            pool.len()
        );
        pool.get(index)
    }

    /// allocate the per-cell artifacts with every cell excluded.
    ///
    /// separated from the fill so a compiled map can write into the same scratch: the exclusion
    /// default is what makes ghosts, and any cell a fill skips, absent from the reduction rather
    /// than silently binned into bucket zero.
    pub fn census_scratch(&self, ev: &CensusEvaluator) -> Option<CensusFields<D, Mem>> {
        let n_values = ev.spec.n_values();
        let values: Vec<symbi_grid::Field<f64, D, Mem>> = (0..n_values)
            .map(|_| symbi_grid::Field::<f64, D, Mem>::zeros(&self.geom.allocated))
            .collect::<Result<_, _>>()
            .ok()?;
        let segment = symbi_grid::Field::<f64, D, Mem>::zeros(&self.geom.allocated).ok()?;
        if Mem::IS_HOST_ACCESSIBLE {
            let excluded = ev.spec.excluded_marker();
            for c in self.geom.allocated.iter() {
                segment.view_mut().set(c, excluded);
            }
        }
        Some(CensusFields {
            values,
            segment,
            segment_stamp: std::sync::Mutex::new(None),
        })
    }

    /// mark a level's covered cells excluded, after a fill that did not know about them.
    ///
    /// host only, and reachable only behind the per-cell interpreter, which refuses a
    /// device-resident store. the production path dispatches a constant fill over the covered
    /// region instead, because walking it here on a device-resident hierarchy would leave the
    /// covered cells carrying the bucket the map assigned them — counting the refined volume on
    /// both levels and inflating every extensive total by exactly that volume.
    pub fn census_exclude_covered(
        &self,
        ev: &CensusEvaluator,
        fields: &CensusFields<D, Mem>,
        covered: &symbi_algebra::Domain<D>,
    ) {
        assert!(
            Mem::IS_HOST_ACCESSIBLE,
            "census '{}': the covered region cannot be excluded by a host walk on a \
             device-resident store; dispatch a constant fill over it instead",
            ev.spec.name()
        );
        let excluded = ev.spec.excluded_marker();
        for c in covered.iter() {
            fields.segment.view_mut().set(c, excluded);
        }
    }

    /// evaluate the registered expressions into `fields` over every interior cell.
    ///
    /// a finer level's coverage is not honoured here: `census_exclude_covered` marks those cells
    /// afterwards. skipping them during the walk instead would leave whatever the scratch already
    /// held, which on a reused buffer is the previous sample's bucket rather than the exclusion.
    pub fn census_fill_interpreted(
        &self,
        ev: &CensusEvaluator,
        fields: &CensusFields<D, Mem>,
    ) -> Option<()> {
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        // a census reads the primitives, and seeding writes only the conserved state. on a
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
        // multiplies the physical volume, so an extensive total stays constant as a(t) grows.
        let a = self.motion.a;
        let n_axes = ev.spec.axes().len();
        let n_values = ev.spec.n_values();

        let (values, segment) = (&fields.values, &fields.segment);

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
                .set(c, ev.spec.segment_marker(&out[..n_axes]) as f64);
            for v in 0..n_values {
                values[v].view_mut().set(c, out[n_axes + v]);
            }
        }

        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(edges: &[f64]) -> BinAxis {
        BinAxis::new("a", edges.to_vec()).expect("valid edges")
    }

    #[test]
    fn geometry_only_axes_are_cacheable_but_state_and_time_axes_are_not() {
        let evaluator = |axis_node: &str| {
            let cfg = symbi_hydro::CensusConfig::from_json(&format!(
                r#"{{
                    "name":"cacheability", "op":"add",
                    "axes":[{{"name":"a", "expr":0, "edges":[0.0,1.0]}}],
                    "values":[2], "value_names":["mass"], "params":[],
                    "nodes":[
                        {{"op":"{axis_node}"}},
                        {{"op":"VARIABLE_RHO"}},
                        {{"op":"VARIABLE_DV"}}
                    ]
                }}"#
            ))
            .unwrap();
            CensusEvaluator::new(&cfg).unwrap()
        };

        assert!(evaluator("VARIABLE_X1").axes_are_geometry_only());
        assert!(evaluator("VARIABLE_DV").axes_are_geometry_only());
        assert!(!evaluator("VARIABLE_RHO").axes_are_geometry_only());
        assert!(!evaluator("VARIABLE_T").axes_are_geometry_only());
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
        // outside the angular ones has no bucket. binning it on the axes that did match
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
        // and it is not the excluded marker, which sits strictly further out: this cell was
        // meant to be reduced and simply fell outside the edges, which is a shortfall to report
        // rather than a cell that was never part of the reduction.
        assert!(
            spec.segment_marker(&[0.5, 50.0]) < spec.n_segments() as u32 + SEGMENT_EXCLUDED_OFFSET
        );
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

    /// the carrier-generic binning must agree exactly with the independent partition-point
    /// implementation, on every coordinate a cell can present.
    ///
    /// compared against `segment` — the `edges.partition_point` search — and not against
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
            // a sweep that lands on every edge, just inside and just outside each, between
            // edges, far outside both ends, and NaN — the coordinates where a binning rule
            // differs from a neighboring one.
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

    /// the outer edge is in the last bin, not one past it: a value sitting exactly on the
    /// domain's outer boundary is real data, and dropping it would quietly under-count the
    /// outermost shell of every profile.
    #[test]
    fn a_value_on_the_outer_edge_lands_in_the_last_bin() {
        let axes = vec![BinAxis::new("a", vec![0.0, 1.0, 2.0]).unwrap()];
        let sp = spec(axes.clone());
        assert_eq!(
            segment_marker_generic::<f64>(&axes, &[2.0], sp.n_segments()),
            1.0
        );
        // and just past it is dropped, not folded back in.
        assert_eq!(
            segment_marker_generic::<f64>(&axes, &[2.0 + 1.0e-9], sp.n_segments()),
            sp.n_segments() as f64
        );
    }
}
