// =============================================================================
// gv.rs
//
// the `Gv` ("graph value") carrier + the thread-local trace it records into.
// `Gv` is a `symbi_algebra::Scalar` whose operations RECORD into this crate's
// tensor graph, evaluating nothing; instantiating carrier-generic physics
// (written over `S: Scalar`) at `S = Gv` traces it into the stencil IR — the
// foundational "code -> graph" boundary. graph and carrier live in the SAME
// crate so the IR machine is one layer.
//
// arena pattern: a thread-local graph holds the active trace; `Gv` is a Copy
// handle. `begin_trace()` opens it, ops push nodes, `end_trace()` takes the
// graph out. build-time only, sidestepping the proc-macro cross-invocation footgun.
//
// the trace (`GvTrace`/`with_trace`/`coord_node`) is `pub` so the discretization
// builders in symbi-discretize can construct raw index/stencil IR (integer coord
// arithmetic the f64 `Gv` carrier deliberately does not route through itself).
// =============================================================================

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::graph::{ConstValue, ElementWiseOp, Graph, NodeId, TranscendentalOp};
use crate::{ElementTy, Symbol};
use symbi_abi::FieldBind;
use symbi_algebra::{Domain, FieldElement, Space};

/// the finished trace: the stencil graph plus the kernel ABI manifest the emit
/// pipeline consumes — field-buffer inputs as `(ir_key, runtime_path)`, scalar
/// params, and the spatial coord axes a stencil references — all in first-seen order.
#[derive(Clone, Debug)]
pub struct GvKernel {
    pub graph: Graph,
    /// field-buffer reads as `(ir_key, FieldBind)`: the IR-side load name paired with the
    /// born-typed runtime binding the producer minted. the binding is a
    /// `FieldBind` (typed `Ref` for the closed cell vocabulary, `Raw` for hand-built paths) —
    /// no consumer re-parses a runtime string.
    pub field_inputs: Vec<(String, FieldBind)>,
    pub scalar_params: Vec<String>,
    /// spatial axes whose `_coord_N` the trace referenced (for stencil `load_at`);
    /// empty for pointwise kernels. feeds `KernelEmitInputs::coord_components`.
    pub coord_components: Vec<u8>,
    /// the launch grade this kernel is to be issued over. fusion (`try_fuse`)
    /// requires both sides to share a grade; untagged kernels never
    /// fuse. see `LaunchGrade` and `try_fuse`.
    pub grade: LaunchGrade,
    /// optional shared-memory tile specification. when `Some`, the CUDA emit is
    /// expected to allocate a per-block `__shared__` buffer for each tiled
    /// field, cooperatively prefetch the (block + halo) region, sync, and
    /// redirect that field's stencil `LoadAt` reads to smem. when `None`, the
    /// emit produces the prior gmem-per-thread pattern.
    ///
    /// the type carries the spec; the CUDA emit and LoadAt rewriting paths are
    /// **not implemented**. the field
    /// exists so builders, fusion, and the runtime can be extended without
    /// further enum-shape changes.
    pub tile_spec: Option<TileSpec>,
    /// the declared SUPPORT of this kernel's outputs:
    /// a region outside which every output is exactly zero for any field input.
    /// declared by the builder (where the saturation constants live), carried
    /// into the serialized `Prepared` blob, consumed by dispatch (reduction /
    /// launch regions). `None` = Everywhere (always sound).
    pub output_support: Option<crate::support::Support>,
    /// builder-tagged support balls by node, carried from the trace for
    /// `with_derived_support` propagation. build-time metadata only — never
    /// serialized, dropped by fusion (fused kernels derive before fusing).
    pub node_supports: std::collections::HashMap<NodeId, crate::support_infer::SupportBall>,
}

/// shared-memory tile spec for stencil kernels. names the fields to prefetch
/// into smem and the halo width per axis. consumed by the CUDA backend at
/// emit time and by the runtime when computing per-block smem requirements.
///
/// invariants:
///   - `halo` has one entry per axis (`halo.len() == kernel ndim`) and at least
///     one entry is `> 0` (an all-zero halo means no smem benefit — declare
///     `tile_spec = None`). a PER-AXIS halo lets a direction-`dir` flux kernel,
///     which reconstructs along ONE axis only, prefetch a thin SLAB (halo on
///     `dir`, 0 transverse); a fat cube would load ~7.5x
///     more cells than the physics reads and likely null the perf win.
///   - every `tiled_field_keys[i]` MUST appear in `GvKernel::field_inputs` —
///     fields not in the manifest can't be tiled because the dispatch has no
///     way to bind a buffer to them
///   - the per-block thread layout (`block_dims`) plus halo determines the
///     smem footprint: `prod_d (block_dims[d] + 2*halo[d]) * sizeof(S) * n_fields`.
///     callers must ensure this fits in the device's smem-per-block budget
///     (48 KB on Turing without opt-in, 64 KB with `cudaFuncSetAttribute`)
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TileSpec {
    /// halo cells PER AXIS (length = kernel ndim). a reconstruction along axis
    /// `dir` with PLM radius 2 sets `halo[dir] = 2`, the transverse axes `0`.
    /// the cooperative load extends the block by `halo[d]` on each side of axis
    /// `d`; an axis with `halo[d] == 0` is not extended (yielding a slab).
    pub halo: Vec<u8>,
    /// fields to prefetch into smem, by IR key (matching `field_inputs[i].0`).
    /// stencil `LoadAt` reads for these keys are routed through smem; pointwise
    /// reads of OTHER fields stay on gmem.
    pub tiled_field_keys: Vec<String>,
}

impl TileSpec {
    /// the smem footprint per block in bytes, given the per-axis thread block
    /// dimensions and the element size. `block_dims.len()` must equal the
    /// kernel's rank (== `self.halo.len()`).
    pub fn smem_bytes_per_block(&self, block_dims: &[u32], elem_bytes: usize) -> usize {
        assert_eq!(
            block_dims.len(),
            self.halo.len(),
            "smem_bytes_per_block: block_dims rank {} != halo rank {}",
            block_dims.len(),
            self.halo.len(),
        );
        let tile_extent: usize = block_dims
            .iter()
            .zip(self.halo.iter())
            .map(|(&b, &h)| (b + 2 * h as u32) as usize)
            .product();
        tile_extent * elem_bytes * self.tiled_field_keys.len()
    }
}

/// the grade of the gv-kernel monoid: a rank-erased structural fingerprint of
/// the `symbi_algebra::Domain<R>` the kernel will be dispatched over. two
/// kernels fuse iff their grades are structurally equal — same axis names,
/// same `[lo, hi)` per axis, in the same order. construction goes through a
/// real `Domain<R>`; `LaunchGrade` is the type-erased view used by the IR.
///
/// the empty grade (`LaunchGrade::untagged()`) is the sentinel for pre-algebra
/// kernels. it never fuses with anything, including itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct LaunchGrade {
    /// per-axis half-open intervals, in rank order. delegates to the algebra
    /// crate's `Space` so equality / hashing reuse that type's semantics.
    pub spaces: Vec<Space>,
}

impl LaunchGrade {
    /// fingerprint of a real `Domain<R>`. const-generic R is erased at the
    /// boundary; the fingerprint stays comparable across kernel sources.
    pub fn from_domain<const R: usize>(d: &Domain<R>) -> Self {
        Self {
            spaces: d.spaces.iter().cloned().collect(),
        }
    }

    /// the sentinel that opts out of fusion. used by `end_trace()`.
    pub fn untagged() -> Self {
        Self::default()
    }

    /// true iff this grade was tagged with a real `Domain<R>` fingerprint.
    pub fn is_tagged(&self) -> bool {
        !self.spaces.is_empty()
    }
}

/// the writes manifest each builder pairs with its `GvKernel`: per output, a
/// `(write_key, born-typed runtime binding, root_node)` triple. fusion concatenates
/// these after the splice remaps `root_node` into the fused graph.
pub type Writes = Vec<(String, FieldBind, NodeId)>;

#[derive(Clone, Debug)]
pub enum FusionError {
    /// the two kernels declare different launch grades. fusing would dispatch
    /// the merged body over a domain that's wrong for at least one half.
    GradeMismatch { a: LaunchGrade, b: LaunchGrade },
    /// either kernel is untagged — fusion requires both sides to be tagged
    /// with a real `Domain<R>` fingerprint via `end_trace_for_domain`.
    UntaggedKernel,
    /// both kernels write the same runtime path. data races by construction.
    WriteConflict { runtime_path: String },
    /// one kernel writes a path the other reads. fusing them would change
    /// semantics: the read in the would-be-second-launch would see the
    /// would-be-first-launch's update inside the same kernel body. callers
    /// must serialize via two launches (or restructure to remove the dep).
    InterDep { written: String, read: String },
    /// the two kernels declare incompatible tile specs (different halo widths,
    /// or one is tiled and the other isn't). a single fused launch can't
    /// run two block layouts simultaneously. callers must either align the
    /// specs or keep the kernels in separate launches.
    TileSpecMismatch {
        a: Option<TileSpec>,
        b: Option<TileSpec>,
    },
    /// the splice pass returned an error. carries the inner reason.
    Splice(String),
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::GradeMismatch { a, b } => {
                write!(f, "fuse: launch-grade mismatch — a={a:?}, b={b:?}")
            }
            FusionError::UntaggedKernel => write!(
                f,
                "fuse: at least one kernel is untagged; tag both via `end_trace_for_domain`"
            ),
            FusionError::WriteConflict { runtime_path } => {
                write!(f, "fuse: both kernels write `{runtime_path}`")
            }
            FusionError::InterDep { written, read } => write!(
                f,
                "fuse: one kernel writes `{written}` and the other reads `{read}`"
            ),
            FusionError::TileSpecMismatch { a, b } => {
                write!(f, "fuse: tile-spec mismatch — a={a:?}, b={b:?}")
            }
            FusionError::Splice(e) => write!(f, "fuse: splice failed: {e}"),
        }
    }
}
impl std::error::Error for FusionError {}

/// the active trace: graph + the manifest accumulated as field reads / scalar params /
/// coord references are recorded. dedup sets keep the manifest first-seen-unique.
pub struct GvTrace {
    graph: Graph,
    field_inputs: Vec<(String, FieldBind)>,
    field_keys: HashSet<String>,
    scalar_params: Vec<String>,
    scalar_keys: HashSet<String>,
    coord_components: Vec<u8>,
    coord_nodes: HashMap<u8, NodeId>,
    /// builder-tagged support balls by node: "this node is exactly zero
    /// outside the ball" (the saturation lemma, asserted where the mask is
    /// built, validated by the compiled-kernel sampler). consumed by
    /// `with_derived_support`.
    node_supports: HashMap<NodeId, crate::support_infer::SupportBall>,
}

thread_local! {
    static GV_TRACE: RefCell<Option<GvTrace>> = const { RefCell::new(None) };
}

/// open a fresh trace. `Gv` ops between this and `end_trace` record into it.
pub fn begin_trace() {
    GV_TRACE.with(|t| {
        *t.borrow_mut() = Some(GvTrace {
            graph: Graph::new(),
            field_inputs: Vec::new(),
            field_keys: HashSet::new(),
            scalar_params: Vec::new(),
            scalar_keys: HashSet::new(),
            coord_components: Vec::new(),
            coord_nodes: HashMap::new(),
            node_supports: HashMap::new(),
        });
    });
}

/// take the finished trace (graph + manifest) out, closing it.
///
/// the resulting kernel is **untagged** — it will not participate in
/// `try_fuse`. callers that don't care about fusion use this; callers that
/// want the fusion algebra should prefer `end_trace_for_domain(d)`.
pub fn end_trace() -> GvKernel {
    end_trace_with(LaunchGrade::untagged())
}

/// take the finished trace and tag it with the launch grade of the given
/// `Domain<R>`. only tagged kernels participate in `try_fuse`.
pub fn end_trace_for_domain<const R: usize>(d: &Domain<R>) -> GvKernel {
    end_trace_with(LaunchGrade::from_domain(d))
}

/// the underlying take-and-tag. exposed for symmetry with `noop`; most callers
/// should go through `end_trace_for_domain` instead.
pub fn end_trace_with(grade: LaunchGrade) -> GvKernel {
    let t = GV_TRACE
        .with(|t| t.borrow_mut().take())
        .expect("end_trace() without begin_trace()");
    GvKernel {
        graph: t.graph,
        field_inputs: t.field_inputs,
        scalar_params: t.scalar_params,
        coord_components: t.coord_components,
        grade,
        tile_spec: None,
        output_support: None,
        node_supports: t.node_supports,
    }
}

/// tag a traced value with a support ball: the builder asserts the value is
/// exactly zero (f64) outside |x - center| > radius for EVERY field input —
/// the saturation lemma, stated where the mask that makes it true is built.
/// consumed by `GvKernel::with_derived_support`, which propagates tags to the
/// write roots; validated downstream by the compiled-kernel support sampler.
pub fn tag_support_ball(
    v: &Gv,
    center: Vec<crate::support::ParamExpr>,
    radius: crate::support::ParamExpr,
) {
    GV_TRACE.with(|t| {
        let mut b = t.borrow_mut();
        let tr = b
            .as_mut()
            .expect("tag_support_ball outside an active trace");
        tr.node_supports.insert(
            v.node(),
            crate::support_infer::SupportBall { center, radius },
        );
    });
}

/// run `f` in a FRESH, isolated trace and return the finished (untagged) kernel + `f`'s result.
/// any trace already active on this thread is SAVED before and RESTORED after — so this is safe
/// to call WHILE another trace is open (e.g., building a sub-source `BuiltSource` partway through
/// a godunov trace). without this, the inner `begin_trace`/`end_trace` would clobber the outer
/// trace and the next outer `Gv` op would panic "outside an active trace".
pub fn in_isolated_trace<R>(f: impl FnOnce() -> R) -> (GvKernel, R) {
    let saved = GV_TRACE.with(|t| t.borrow_mut().take());
    begin_trace();
    let r = f();
    let kernel = end_trace();
    GV_TRACE.with(|t| *t.borrow_mut() = saved);
    (kernel, r)
}

impl GvKernel {
    /// infer this kernel's shared-memory tile intent. an explicit
    /// `tile_spec` overrides; otherwise a STENCIL kernel (non-empty
    /// `coord_components` — it does shifted `load_at` reads, so it has a halo and
    /// reusable neighbor data) gets a `TileSpec`, and a POINTWISE kernel (empty)
    /// gets `None` (no stencil reuse => smem can't help). this is the
    /// inferred-with-override policy (decision 1).
    ///
    /// the CPU emit cache-tiles EVERY kernel regardless (a runtime knob); this
    /// drives only the GPU smem path.
    ///
    /// **policy:** OPT-IN. only kernels that explicitly declare
    /// a `tile_spec` via `with_tile_spec` are tiled — the builder KNOWS the
    /// stencil structure (which axis reconstructs, how wide) and declares the
    /// correct per-axis SLAB. the previous auto-inference returned a fat halo-2
    /// CUBE for every stencil kernel; that over-loads smem ~7.5x for a 1D-along-dir
    /// reconstruction and reads transverse ghost cells the physics never touches,
    /// so it is NOT a safe default. auto-inference of a correct per-axis slab from
    /// the graph's `LoadAt` offsets is deferred to the promotion step.
    pub fn infer_tile_spec(&self) -> Option<TileSpec> {
        self.tile_spec.clone()
    }

    /// the field-input keys this kernel reads at a SHIFTED coord — the
    /// `Op::LoadAt` field symbols, in first-seen order, restricted to keys that
    /// are actual `field_inputs` (a LoadAt always names a registered field, so
    /// the restriction is a safety net). this is the smem-tile
    /// candidate set: only stencil-read fields have reusable neighbor data worth
    /// prefetching; a field read pointwise stays on gmem.
    pub fn stencil_read_field_keys(&self) -> Vec<String> {
        let manifest: std::collections::HashSet<&str> =
            self.field_inputs.iter().map(|(k, _)| k.as_str()).collect();
        let mut seen = std::collections::HashSet::new();
        let mut keys = Vec::new();
        for (_, node, _) in self.graph.iter() {
            if let crate::graph::Op::LoadAt(sym, _) = &node.op {
                let k = sym.as_str();
                if manifest.contains(k) && seen.insert(k.to_string()) {
                    keys.push(k.to_string());
                }
            }
        }
        keys
    }

    /// the identity element for the fusion monoid at the given grade. an
    /// empty graph, no writes, no reads. `try_fuse(noop(g), k) == k` for any
    /// tagged `k` with `k.grade == g` (identity law).
    pub fn noop(grade: LaunchGrade) -> (GvKernel, Writes) {
        (
            GvKernel {
                graph: Graph::new(),
                field_inputs: Vec::new(),
                scalar_params: Vec::new(),
                coord_components: Vec::new(),
                grade,
                tile_spec: None,
                output_support: None,
                node_supports: std::collections::HashMap::new(),
            },
            Vec::new(),
        )
    }

    /// builder-style: attach a smem tile spec. validates that every tiled key
    /// is present in this kernel's `field_inputs` manifest — fields not in the
    /// manifest can't be tiled because the dispatch has no way to bind a
    /// buffer to them. panics on mismatch (caller is the GvKernel builder,
    /// which OWNS the manifest and a mismatch is a build-side bug).
    pub fn with_tile_spec(mut self, spec: TileSpec) -> Self {
        let manifest: std::collections::HashSet<&str> =
            self.field_inputs.iter().map(|(k, _)| k.as_str()).collect();
        for key in &spec.tiled_field_keys {
            assert!(
                manifest.contains(key.as_str()),
                "with_tile_spec: tiled field `{key}` is not in this kernel's field_inputs manifest",
            );
        }
        assert!(
            spec.halo.iter().any(|&h| h > 0),
            "with_tile_spec: at least one axis halo must be > 0 (use `None` for no tiling)",
        );
        self.tile_spec = Some(spec);
        self
    }

    /// declare the support of this kernel's outputs: exactly zero outside the
    /// region, for every field input value (validated by sampling the compiled
    /// kernel — the support-validation gates). a ball's center/radius are
    /// expressions over THIS kernel's scalar params; every referenced param
    /// must be in the manifest, or dispatch could never evaluate the geometry.
    pub fn with_output_support(mut self, support: crate::support::Support) -> Self {
        if let crate::support::Support::Ball { center, radius } = &support {
            let manifest: std::collections::HashSet<&str> =
                self.scalar_params.iter().map(|s| s.as_str()).collect();
            let check = |e: &crate::support::ParamExpr| {
                collect_support_params(e, &mut |name| {
                    assert!(
                        manifest.contains(name),
                        "with_output_support: param `{name}` is not in this kernel's scalar manifest",
                    );
                });
            };
            for c in center {
                check(c);
            }
            check(radius);
        }
        self.output_support = Some(support);
        self
    }

    /// derive the output support from the trace's tagged mask nodes by
    /// propagation over the graph (`support_infer`), then declare it. a broken
    /// mask chain or an untagged trace derives Everywhere — fail-safe wide,
    /// never a stale narrow ball. ball params are manifest-validated exactly
    /// as a hand declaration would be.
    pub fn with_derived_support(self, writes: &Writes) -> Self {
        let support = crate::support_infer::derive_output_support(
            &self.graph,
            &self.node_supports,
            &self.field_inputs,
            writes,
        );
        match support {
            crate::support::Support::Everywhere => self,
            s => self.with_output_support(s),
        }
    }
}

/// walk a support param expression calling `f` on every referenced param name.
fn collect_support_params(e: &crate::support::ParamExpr, f: &mut impl FnMut(&str)) {
    use crate::support::ParamExpr::*;
    match e {
        Param(name) => f(name),
        Const(_) => {}
        Add(a, b) | Mul(a, b) => {
            collect_support_params(a, f);
            collect_support_params(b, f);
        }
        Min(items) => items.iter().for_each(|i| collect_support_params(i, f)),
    }
}

/// the fusion primitive. attempts to merge two gv-kernels into one launched
/// over the same grade, returning the merged kernel + the concatenated writes
/// manifest. fails if any of the three structural preconditions are violated:
///
/// - grade equality: both `index_space` must match (and neither be
///   `Untagged`). fusing different domains would launch at least one half
///   over the wrong index set.
///
/// - disjoint writes: no runtime_path appears in both writes lists. shared
///   writes would race.
///
/// - no inter-dependency: for each runtime_path one kernel writes, the
///   other must not read it. (in-place reads from one's own writes do not
///   count — that's intra-kernel.) inter-dep violates the sequential
///    semantics: in the two-launch baseline the second launch sees the first
///    launch's updates globally; in a fused launch a stencil read in the
///    second body would see the first body's writes only at the SAME thread
///    index.
///
/// returns `Ok((fused_kernel, fused_writes))` on success. the fused writes
/// preserve the original ordering: all of a's writes (with NodeIds unchanged,
/// since the fused graph starts as a clone of a's graph), then all of b's
/// writes with NodeIds remapped through the splice.
pub fn try_fuse(
    a: GvKernel,
    a_writes: Writes,
    b: GvKernel,
    b_writes: Writes,
) -> Result<(GvKernel, Writes), FusionError> {
    use crate::passes::splice::splice_graph;

    // law: grade-equality + tagged-only.
    if !a.grade.is_tagged() || !b.grade.is_tagged() {
        return Err(FusionError::UntaggedKernel);
    }
    if a.grade != b.grade {
        return Err(FusionError::GradeMismatch {
            a: a.grade,
            b: b.grade,
        });
    }

    // law: tile-spec equality. a fused launch can only run ONE block layout +
    // smem prelude. tiled-and-untiled cannot share a launch, and two different
    // halos / tiled-field sets can't either. callers must align before fusing.
    if a.tile_spec != b.tile_spec {
        return Err(FusionError::TileSpecMismatch {
            a: a.tile_spec.clone(),
            b: b.tile_spec.clone(),
        });
    }

    // law: disjoint writes. compare by the canonical path name (both reads and writes are
    // born-typed FieldBind now), so the in-place/inter-dep detection is spelling-invariant.
    let a_write_paths: HashSet<String> = a_writes.iter().map(|(_, p, _)| p.name()).collect();
    let b_write_paths: HashSet<String> = b_writes.iter().map(|(_, p, _)| p.name()).collect();
    if let Some(p) = a_write_paths.intersection(&b_write_paths).next() {
        return Err(FusionError::WriteConflict {
            runtime_path: p.clone(),
        });
    }

    // no inter-dependency: a field one kernel writes and the other reads is a TRUE pipeline hazard
    // where the reader would need the writer's fresh OUTPUT. EXCEPTION: a field the writer holds
    // IN-PLACE (it reads AND writes it) is a shared PRE-STATE — both kernels
    // read the same old value as a leaf, and the fused dataflow writes the new value as a root
    // (the field is read pointwise, so the in-place write is hazard-free). this is exactly the
    // god+bcell co-stage composition: the curvilinear gas geo source reads cell-B (bc_k) for the
    // magnetic pressure while the cell-B predictor flux-evolves that SAME bc_k in place — both on
    // the timestep's pre-state. excluding the writer's in-place fields lets that fuse safely
    // (symbi only ever fuses same-stage kernels, never a pipeline, so the reader always wants the
    // pre-state). disjoint writes are still enforced above, so this never permits a double-write.
    // compare by the CANONICAL path name on both sides (reads via FieldBind::name, writes via
    // from_path().name() below) so the in-place/inter-dep detection is spelling-invariant —
    // a field read as `prim.vel[k]` and written as `prim.vel_k` is the SAME buffer.
    let a_input_paths: HashSet<String> = a.field_inputs.iter().map(|(_, p)| p.name()).collect();
    let b_input_paths: HashSet<String> = b.field_inputs.iter().map(|(_, p)| p.name()).collect();
    let a_inplace: HashSet<&String> = a_write_paths.intersection(&a_input_paths).collect();
    let b_inplace: HashSet<&String> = b_write_paths.intersection(&b_input_paths).collect();
    if let Some(p) = a_write_paths
        .intersection(&b_input_paths)
        .find(|p| !a_inplace.contains(p))
    {
        return Err(FusionError::InterDep {
            written: p.clone(),
            read: p.clone(),
        });
    }
    if let Some(p) = b_write_paths
        .intersection(&a_input_paths)
        .find(|p| !b_inplace.contains(p))
    {
        return Err(FusionError::InterDep {
            written: p.clone(),
            read: p.clone(),
        });
    }

    // structural merge: clone a's graph as the target. a's writes carry
    // their NodeIds verbatim (clone preserves node order). splice b in,
    // remapping b's write roots through the splice.
    let mut target = a.graph.clone();

    // build param_subst for b's params by re-adding each via target.add_param.
    // add_param dedupes by Symbol, so b's params that share a Symbol with one
    // of a's params (same field_key / scalar_name) map to a's NodeId; new
    // ones get fresh NodeIds appended to target.
    let mut subst: HashMap<Symbol, NodeId> = HashMap::new();
    for (sym, b_param_id) in b.graph.params().iter() {
        let ty = b.graph.ty(*b_param_id).clone();
        let target_id = target.add_param(sym.clone(), ty, None);
        subst.insert(sym.clone(), target_id);
    }

    // splice b's full subgraph reachable from each write root.
    let b_roots: Vec<NodeId> = b_writes.iter().map(|(_, _, n)| *n).collect();
    let remapped_b_roots = splice_graph(&mut target, &b.graph, &b_roots, &subst)
        .map_err(|e| FusionError::Splice(format!("{}", e)))?;

    // manifest merge: deterministic first-seen order, a first, b appended.
    let mut field_inputs = a.field_inputs.clone();
    let a_field_keys: HashSet<String> = a.field_inputs.iter().map(|(k, _)| k.clone()).collect();
    for (k, p) in b.field_inputs.iter() {
        if !a_field_keys.contains(k) {
            field_inputs.push((k.clone(), p.clone()));
        }
    }
    let mut scalar_params = a.scalar_params.clone();
    let a_scalar_set: HashSet<String> = a.scalar_params.iter().cloned().collect();
    for s in b.scalar_params.iter() {
        if !a_scalar_set.contains(s) {
            scalar_params.push(s.clone());
        }
    }
    let mut coord_components = a.coord_components.clone();
    for c in b.coord_components.iter() {
        if !coord_components.contains(c) {
            coord_components.push(*c);
        }
    }

    // the fused outputs' support is the UNION of both sides'. representable
    // only when both declare the same region; any mismatch (or an undeclared
    // side) widens to Everywhere — always sound, never wrong.
    let output_support = match (&a.output_support, &b.output_support) {
        (Some(sa), Some(sb)) if sa == sb => a.output_support.clone(),
        _ => None,
    };
    let fused = GvKernel {
        graph: target,
        field_inputs,
        scalar_params,
        coord_components,
        grade: a.grade,
        // tile_spec is preserved across fusion: the pre-check guaranteed
        // a.tile_spec == b.tile_spec, so either side's value serves.
        tile_spec: a.tile_spec,
        output_support,
        // node tags do not survive fusion: derivation happens at build time,
        // before fusing, and the fused support is the declared-union above.
        node_supports: HashMap::new(),
    };

    // writes: a's preserved verbatim (target started as a's graph clone),
    // b's remapped to the spliced NodeIds.
    let mut fused_writes: Writes = a_writes;
    for ((k, p, _), new_root) in b_writes.into_iter().zip(remapped_b_roots) {
        fused_writes.push((k, p, new_root));
    }

    Ok((fused, fused_writes))
}

/// intern (or reuse) the synthetic `_coord_N` I32 param for spatial axis `ax`, recording
/// the axis in the manifest. a cell coordinate
/// is an i32, so stencil shifts (`_coord + off`) and buffer indices are pure integer
/// arithmetic (no float routing through the f64 Gv carrier).
fn coord_node(t: &mut GvTrace, ax: u8) -> NodeId {
    if let Some(&id) = t.coord_nodes.get(&ax) {
        return id;
    }
    let id = t
        .graph
        .add_scalar_param(&format!("_coord_{ax}"), ElementTy::I32);
    t.coord_nodes.insert(ax, id);
    t.coord_components.push(ax);
    id
}

/// run a closure with mutable access to the active trace.
pub fn with_trace<R>(f: impl FnOnce(&mut GvTrace) -> R) -> R {
    GV_TRACE.with(|t| {
        let mut b = t.borrow_mut();
        f(b.as_mut()
            .expect("Gv op outside an active trace — call begin_trace() first"))
    })
}

// the builder-facing trace API: the discretization layer (symbi-discretize) constructs
// raw index/stencil IR through these, so the trace's fields and the graph stay private.
impl GvTrace {
    /// raw graph access for builders constructing IR nodes directly (integer index
    /// arithmetic, select, load_at) — the addressing the f64 `Gv` carrier doesn't route.
    pub fn graph(&mut self) -> &mut Graph {
        &mut self.graph
    }

    /// register a per-cell field input in the ABI manifest (deduped, first-seen order).
    /// the buffer is bound by `key` at load time; `runtime` is its dotted dispatch path.
    pub fn register_field(&mut self, key: &str, runtime: impl Into<FieldBind>) {
        let runtime = runtime.into();
        if self.field_keys.insert(key.to_string()) {
            self.field_inputs.push((key.to_string(), runtime));
        }
    }

    /// a deduped I32 scalar param (an integer index / lattice-map arg), returning its node —
    /// the integer analog of `Gv::scalar` (index math).
    pub fn scalar_int(&mut self, name: &str) -> NodeId {
        let id = self.graph.add_scalar_param(name, ElementTy::I32);
        if self.scalar_keys.insert(name.to_string()) {
            self.scalar_params.push(name.to_string());
        }
        id
    }

    /// the synthetic `_coord_N` I32 param for spatial axis `ax` (deduped, recorded in the
    /// coord manifest) — stencil shifts are pure integer arithmetic on this.
    pub fn coord(&mut self, ax: u8) -> NodeId {
        coord_node(self, ax)
    }
}

/// a graph value: either a traced node, or a literal not yet materialized.
#[derive(Clone, Copy, Debug)]
enum GvVal {
    Node(NodeId),
    Lit(f64),
}

/// the tracing scalar carrier. `Copy` (a NodeId or a literal); every operation
/// records a node into the thread-local trace graph.
///
/// a traced graph value has no physical order or equality — `Scalar` does not
/// require `PartialOrd`/`PartialEq`, which are deliberately NOT implemented.
/// physics decides with the traceable `cmp_lt` / `cmp_gt` / `select`, never with
/// native `<` / `==` (which would silently compare node indices).
/// the type system enforces this — native ordering does not compile:
///
/// ```compile_fail
/// use symbi_ir::Gv;
/// let a = Gv::param("a");
/// let b = Gv::param("b");
/// let _ = a < b; // no `PartialOrd` for `Gv` — must use `cmp_lt`
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Gv(GvVal);

impl Gv {
    /// a fresh scalar param node named `name`, unrecorded in the ABI manifest
    /// (a bare leaf for unit tests). production inputs use `field` / `scalar`.
    pub fn param(name: &str) -> Gv {
        Gv(GvVal::Node(with_trace(|t| {
            t.graph.add_scalar_param(name, ElementTy::F64)
        })))
    }

    /// a per-cell field read: `key` is the IR-side buffer-load name, `runtime` the
    /// dotted path the dispatch binds the buffer to (e.g., `"cons.den"`). recorded
    /// (deduped) in the kernel ABI manifest — this is the input binding for a
    /// carrier-generic physics fn instantiated at Gv.
    pub fn field(key: &str, runtime: impl Into<FieldBind>) -> Gv {
        let runtime = runtime.into();
        Gv(GvVal::Node(with_trace(|t| {
            let id = t.graph.add_scalar_param(key, ElementTy::F64);
            if t.field_keys.insert(key.to_string()) {
                t.field_inputs.push((key.to_string(), runtime));
            }
            id
        })))
    }

    /// a SHIFTED per-cell field read for stencils (PLM reconstruction): the field `key`
    /// loaded at `cell + offset` along `axis`, over an `ndim`-spatial grid. `offset == 0`
    /// is the direct cell read (`Gv::field`); nonzero builds the integer coord arithmetic
    /// (`_coord_axis + offset`) + a `LoadAt`, registering the field AND the coord axes in
    /// the manifest. codegen-only — a stencil is not a pointwise `Scalar` op, so this is a
    /// Gv method (the host runtime reads neighbors from the
    /// Field buffer; only the traced kernel needs the explicit `load_at`).
    pub fn field_shifted(
        key: &str,
        runtime: impl Into<FieldBind>,
        ndim: u8,
        axis: u8,
        offset: i32,
    ) -> Gv {
        let runtime = runtime.into();
        if offset == 0 {
            return Gv::field(key, runtime);
        }
        Gv(GvVal::Node(with_trace(|t| {
            // register the field (the LoadAt resolves the buffer by this key, deduped).
            if t.field_keys.insert(key.to_string()) {
                t.field_inputs.push((key.to_string(), runtime));
            }
            let comps: Vec<NodeId> = (0..ndim).map(|ax| coord_node(t, ax)).collect();
            let off = t.graph.add_const(ConstValue::I32(offset), None);
            let mut shifted = comps.clone();
            shifted[axis as usize] =
                t.graph
                    .element_wise(ElementWiseOp::Add, vec![comps[axis as usize], off], None);
            t.graph.load_at(Symbol::intern(key), shifted, None)
        })))
    }

    /// read a field at a MULTI-axis integer offset from the current cell — the
    /// halo-stencil primitive for operators that read diagonals (e.g. the
    /// viscous transverse gradient). `field_shifted` is the single-axis case.
    /// `offsets[ax]` is the per-axis shift; `offsets.len()` must be `ndim`.
    pub fn field_offset(key: &str, runtime: impl Into<FieldBind>, ndim: u8, offsets: &[i32]) -> Gv {
        assert_eq!(
            offsets.len(),
            ndim as usize,
            "field_offset: one offset per axis"
        );
        let runtime = runtime.into();
        if offsets.iter().all(|&o| o == 0) {
            return Gv::field(key, runtime);
        }
        Gv(GvVal::Node(with_trace(|t| {
            if t.field_keys.insert(key.to_string()) {
                t.field_inputs.push((key.to_string(), runtime));
            }
            let mut shifted: Vec<NodeId> = (0..ndim).map(|ax| coord_node(t, ax)).collect();
            for (ax, &o) in offsets.iter().enumerate() {
                if o != 0 {
                    let off = t.graph.add_const(ConstValue::I32(o), None);
                    shifted[ax] =
                        t.graph
                            .element_wise(ElementWiseOp::Add, vec![shifted[ax], off], None);
                }
            }
            t.graph.load_at(Symbol::intern(key), shifted, None)
        })))
    }

    /// a scalar kernel param (e.g., `gamma`), recorded (deduped) in the manifest signature.
    pub fn scalar(name: &str) -> Gv {
        Gv(GvVal::Node(with_trace(|t| {
            let id = t.graph.add_scalar_param(name, ElementTy::F64);
            if t.scalar_keys.insert(name.to_string()) {
                t.scalar_params.push(name.to_string());
            }
            id
        })))
    }

    /// the cell coordinate along spatial `axis` as a Gv value — the index->physical bridge for
    /// in-kernel GEOMETRY. it is the integer `_coord_N` (recorded in the coord manifest, like
    /// `field_shifted`); arithmetic against the f64 grid scalars (`x_lo_d`, `dx_d`) auto-promotes
    /// at lowering (the IR's usual arithmetic conversions), so positions, scale factors, and cell
    /// widths trace as pure Gv expressions. this is what lets the substrate geometry (curvilinear
    /// metric, cell areas/volumes) be a Gv trace built directly from the cell index.
    pub fn coord(axis: u8) -> Gv {
        Gv(GvVal::Node(with_trace(|t| coord_node(t, axis))))
    }

    /// the NodeId this value resolves to, materializing a literal to a `Const`
    /// node on demand (a literal touches the graph only here, on first use). pub so a gv
    /// builder in symbi-discretize can extract its write roots.
    pub fn node(self) -> NodeId {
        match self.0 {
            GvVal::Node(n) => n,
            GvVal::Lit(v) => with_trace(|t| t.graph.add_const(ConstValue::F64(v), None)),
        }
    }

    #[inline]
    pub fn of(n: NodeId) -> Gv {
        Gv(GvVal::Node(n))
    }

    #[inline]
    fn binop(self, rhs: Gv, op: ElementWiseOp) -> Gv {
        let a = self.node();
        let b = rhs.node();
        Gv::of(with_trace(|t| t.graph.element_wise(op, vec![a, b], None)))
    }

    #[inline]
    fn unop(self, op: ElementWiseOp) -> Gv {
        let x = self.node();
        Gv::of(with_trace(|t| t.graph.element_wise(op, vec![x], None)))
    }

    #[inline]
    fn transcendental_unop(self, op: TranscendentalOp) -> Gv {
        let x = self.node();
        Gv::of(with_trace(|t| t.graph.transcendental(op, vec![x], None)))
    }
}

// ---- std::ops: record element-wise nodes ----
impl Add for Gv {
    type Output = Gv;
    fn add(self, r: Gv) -> Gv {
        self.binop(r, ElementWiseOp::Add)
    }
}
impl Sub for Gv {
    type Output = Gv;
    fn sub(self, r: Gv) -> Gv {
        self.binop(r, ElementWiseOp::Sub)
    }
}
impl Mul for Gv {
    type Output = Gv;
    fn mul(self, r: Gv) -> Gv {
        self.binop(r, ElementWiseOp::Mul)
    }
}
impl Div for Gv {
    type Output = Gv;
    fn div(self, r: Gv) -> Gv {
        self.binop(r, ElementWiseOp::Div)
    }
}
impl Neg for Gv {
    type Output = Gv;
    fn neg(self) -> Gv {
        self.unop(ElementWiseOp::Neg)
    }
}
impl AddAssign for Gv {
    fn add_assign(&mut self, r: Gv) {
        *self = *self + r;
    }
}
impl SubAssign for Gv {
    fn sub_assign(&mut self, r: Gv) {
        *self = *self - r;
    }
}
impl MulAssign for Gv {
    fn mul_assign(&mut self, r: Gv) {
        *self = *self * r;
    }
}
impl DivAssign for Gv {
    fn div_assign(&mut self, r: Gv) {
        *self = *self / r;
    }
}

impl std::iter::Sum for Gv {
    fn sum<I: Iterator<Item = Gv>>(iter: I) -> Gv {
        // direct construction; ZERO comes from `<Gv as crate::algebra::Scalar>::ZERO`
        // but qualifying inline keeps this independent of import scope.
        iter.fold(Gv(GvVal::Lit(0.0)), |a, b| a + b)
    }
}

impl Default for Gv {
    // direct construction (matches `<Gv as crate::algebra::Scalar>::ZERO`); kept as
    // a direct expression to stay independent of the trait import scope.
    fn default() -> Gv {
        Gv(GvVal::Lit(0.0))
    }
}

impl std::fmt::Display for Gv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

// Gv is a Copy fixed-size handle, only ever live during a build-time trace — it is
// never stored in a `Field<Gv>` buffer, so the layout contract is sound and unexercised.
unsafe impl FieldElement for Gv {
    type Scalar = Gv;
}

// structural numeric impl — satisfies the in-crate `Tensor` / `Matrix` / `Indexed`
// method bounds (`Tensor::dot`, `Tensor::norm`, etc.). delegates to the trace-recording
// ops the production `Scalar` impl below uses. `symbi_algebra::Numeric` is NOT the
// production carrier-generic surface — it is the minimal subset that breaks the
// `symbi-algebra` <-> `symbi-ir` dep cycle for `Tensor`'s scalar-bounded methods.
impl symbi_algebra::algebra::Numeric for Gv {
    const ZERO: Self = Gv(GvVal::Lit(0.0));
    const ONE: Self = Gv(GvVal::Lit(1.0));
    #[inline]
    fn from_f64(v: f64) -> Self {
        Gv(GvVal::Lit(v))
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.unop(ElementWiseOp::Sqrt)
    }
    #[inline]
    fn abs(self) -> Self {
        self.unop(ElementWiseOp::Abs)
    }
    #[inline]
    fn min(self, o: Self) -> Self {
        self.binop(o, ElementWiseOp::Min)
    }
    #[inline]
    fn max(self, o: Self) -> Self {
        self.binop(o, ElementWiseOp::Max)
    }
}

// =============================================================================
// GvMask + impl `crate::algebra::Scalar for Gv` — the production carrier interface.
//
// Mask discipline: `GvMask` is a NEWTYPE around `Gv` that wraps a Bool-typed
// graph node. construction is `pub(crate)` so ONLY `Scalar::cmp_*` produces
// masks — the type system enforces "masks at the graph layer are always
// Bool-typed." `BitAnd` / `BitOr` / `Not` emit the corresponding graph Bool
// ops (`ElementWiseOp::BitAnd` / `BitOr` / `BitNot`).
//
// this is the single carrier surface workspace-wide.
// =============================================================================

/// type-safe Mask wrapper for Gv. wraps a Gv carrying a Bool-typed graph
/// node. `pub(crate)` constructor — only `Scalar::cmp_*` produces these.
#[derive(Copy, Clone, Debug)]
pub struct GvMask(pub(crate) Gv);

impl std::ops::BitAnd for GvMask {
    type Output = GvMask;
    #[inline]
    fn bitand(self, rhs: GvMask) -> GvMask {
        let a = self.0.node();
        let b = rhs.0.node();
        GvMask(Gv::of(with_trace(|t| {
            t.graph
                .element_wise(ElementWiseOp::BitAnd, vec![a, b], None)
        })))
    }
}

impl std::ops::BitOr for GvMask {
    type Output = GvMask;
    #[inline]
    fn bitor(self, rhs: GvMask) -> GvMask {
        let a = self.0.node();
        let b = rhs.0.node();
        GvMask(Gv::of(with_trace(|t| {
            t.graph.element_wise(ElementWiseOp::BitOr, vec![a, b], None)
        })))
    }
}

impl std::ops::Not for GvMask {
    type Output = GvMask;
    #[inline]
    fn not(self) -> GvMask {
        let a = self.0.node();
        GvMask(Gv::of(with_trace(|t| {
            t.graph.element_wise(ElementWiseOp::BitNot, vec![a], None)
        })))
    }
}

impl crate::algebra::Mask for GvMask {}

impl crate::algebra::Scalar for Gv {
    type Mask = GvMask;

    // ZERO / ONE inherited from `Numeric for Gv`.
    const INFINITY: Gv = Gv(GvVal::Lit(f64::INFINITY));
    const NEG_INFINITY: Gv = Gv(GvVal::Lit(f64::NEG_INFINITY));
    const NAN: Gv = Gv(GvVal::Lit(f64::NAN));

    // from_f64 inherited from `Numeric for Gv`.

    fn to_f64(self) -> f64 {
        match self.0 {
            GvVal::Lit(v) => v,
            // HOST-BOUNDARY ESCAPE — see `crate::algebra::Scalar::to_f64` doc.
            // a `Gv` on a traced node is a graph handle; extracting a concrete
            // value inside carrier-generic physics is an A1 violation.
            GvVal::Node(_) => panic!(
                "Gv::to_f64 on a traced node — carrier-generic physics must decide with \
                 cmp_*/select, not extract a concrete value"
            ),
        }
    }

    // ── comparisons return GvMask — the Mask discipline ──────────
    fn cmp_lt(self, o: Gv) -> GvMask {
        GvMask(self.binop(o, ElementWiseOp::Lt))
    }
    fn cmp_le(self, o: Gv) -> GvMask {
        GvMask(self.binop(o, ElementWiseOp::Le))
    }
    fn cmp_gt(self, o: Gv) -> GvMask {
        GvMask(self.binop(o, ElementWiseOp::Gt))
    }
    fn cmp_ge(self, o: Gv) -> GvMask {
        GvMask(self.binop(o, ElementWiseOp::Ge))
    }
    fn cmp_eq(self, o: Gv) -> GvMask {
        GvMask(self.binop(o, ElementWiseOp::Eq))
    }

    fn select(m: GvMask, yes: Gv, no: Gv) -> Gv {
        let c = m.0.node();
        let y = yes.node();
        let n = no.node();
        Gv::of(with_trace(|t| t.graph.select(c, y, n, None)))
    }

    // scope frame at S = Gv.
    //
    // semantics: snapshot the graph's node count BEFORE running `body`; all
    // NodeIds pushed during the closure (the lexical region of this scope)
    // become the Op::Scope's body list — including any nested Op::Scope
    // nodes the closure itself emits. the snapshot..end range captures only
    // FRESH nodes; hash-consing that resolves an expression to a pre-
    // existing NodeId contributes nothing (its NodeId is outside the
    // range).
    //
    // the resulting Op::Scope bypasses hash-cons (see Graph::push) so two
    // structurally identical scopes from distinct call sites stay distinct.
    //
    // empty body short-circuit: if the closure produces no new nodes (e.g.
    // the result is a pre-existing constant or param), there is no scope
    // to record — return the result directly. this preserves the identity
    // law for trivial closures.
    fn scope<F>(body: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        let mark = with_trace(|t| t.graph.len());
        let result_gv = body();
        let result_node = result_gv.node();
        let body_nodes: Vec<NodeId> = with_trace(|t| {
            let end = t.graph.len();
            (mark..end).map(|i| NodeId(i as u32)).collect()
        });
        if body_nodes.is_empty() {
            return result_gv;
        }
        Gv::of(with_trace(|t| {
            t.graph.scope_op(body_nodes, result_node, None)
        }))
    }

    // the DUAL of `iterate` for BRANCHES: a lazy conditional at S = Gv.
    //
    // semantics: snapshot the graph BEFORE each arm's closure; the NodeIds
    // pushed during the closure are that arm's body (the `mark..end` range —
    // same convention as `scope`). emit an Op::IfElse carrying the cond, both
    // arm bodies, and each arm's result. scalarize lowers the arm bodies
    // INSIDE their respective `if`/`else` brace so only the taken arm executes
    // — the carrier-portable form of an early-out conditional.
    //
    // shared upstream values (the cond, any pre-branch subexpression) are
    // created BEFORE the closures, so they fall OUTSIDE both ranges and stay in
    // the outer body (computed once). cross-arm / leaks-outside hash-cons
    // sharing is resolved by scalarize's eviction pass, exactly as for Scope.
    fn cond(m: GvMask, t: impl FnOnce() -> Gv, f: impl FnOnce() -> Gv) -> Gv {
        let cond_node = m.0.node();
        let t_mark = with_trace(|tr| tr.graph.len());
        let t_res = t().node();
        let then_body: Vec<NodeId> = with_trace(|tr| {
            let end = tr.graph.len();
            (t_mark..end).map(|i| NodeId(i as u32)).collect()
        });
        let f_mark = with_trace(|tr| tr.graph.len());
        let f_res = f().node();
        let else_body: Vec<NodeId> = with_trace(|tr| {
            let end = tr.graph.len();
            (f_mark..end).map(|i| NodeId(i as u32)).collect()
        });
        Gv::of(with_trace(|tr| {
            tr.graph.if_else(
                cond_node,
                then_body,
                vec![t_res],
                else_body,
                vec![f_res],
                None,
            )
        }))
    }

    // the N-output lazy branch: one Op::IfElse, N results, N Op::Proj outputs.
    // the SHARED arm computation is traced once (each closure runs once); each
    // returned Gv is a projection of the same branch. this is what lets a
    // multi-output fast-path (e.g., the (sl, sr) wave-speed Eq.57/58/quartic
    // selection) skip the whole quartic on the fast path. mirrors `cond` per
    // arm; the only addition is the N-element result vectors + the projections.
    fn cond_vec<const N: usize>(
        m: GvMask,
        t: impl FnOnce() -> [Gv; N],
        f: impl FnOnce() -> [Gv; N],
    ) -> [Gv; N] {
        let cond_node = m.0.node();
        let t_mark = with_trace(|tr| tr.graph.len());
        let t_res = t();
        let then_body: Vec<NodeId> = with_trace(|tr| {
            let end = tr.graph.len();
            (t_mark..end).map(|i| NodeId(i as u32)).collect()
        });
        let f_mark = with_trace(|tr| tr.graph.len());
        let f_res = f();
        let else_body: Vec<NodeId> = with_trace(|tr| {
            let end = tr.graph.len();
            (f_mark..end).map(|i| NodeId(i as u32)).collect()
        });
        let then_results: Vec<NodeId> = t_res.iter().map(|&g| g.node()).collect();
        let else_results: Vec<NodeId> = f_res.iter().map(|&g| g.node()).collect();
        let ifelse = with_trace(|tr| {
            tr.graph.if_else(
                cond_node,
                then_body,
                then_results,
                else_body,
                else_results,
                None,
            )
        });
        std::array::from_fn(|j| Gv::of(with_trace(|tr| tr.graph.proj(ifelse, j as u32, None))))
    }

    // sqrt / abs / min / max inherited from `Numeric for Gv`.
    fn recip(self) -> Gv {
        // 1 / self — use the const directly to avoid trait-method ambiguity.
        let one = Gv(GvVal::Lit(1.0));
        one / self
    }

    // ── transcendentals (mix of ElementWise + Transcendental graph ops) ───
    fn sin(self) -> Gv {
        self.unop(ElementWiseOp::Sin)
    }
    fn cos(self) -> Gv {
        self.unop(ElementWiseOp::Cos)
    }
    fn tan(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Tan)
    }
    fn asin(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Asin)
    }
    fn acos(self) -> Gv {
        self.unop(ElementWiseOp::Acos)
    }
    fn atan2(self, o: Gv) -> Gv {
        let (y, x) = (self.node(), o.node());
        Gv::of(with_trace(|t| {
            t.graph
                .transcendental(TranscendentalOp::Atan2, vec![y, x], None)
        }))
    }
    fn exp(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Exp)
    }
    fn ln(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Log)
    }
    fn log10(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Log10)
    }

    fn powi(self, n: i32) -> Gv {
        // lower to repeated multiplication (exponentiation by squaring):
        // `f64::powi` raises a NEGATIVE base exactly (e.g., (-2)^2 = 4), but CUDA
        // `powf(neg, 2.0)` = NaN — a carrier-equivalence break (f64 host != Gv kernel).
        // n is a small integer constant at trace time, so the multiply chain unrolls into
        // the DAG; it also avoids the transcendental `powf` call entirely.
        if n == 0 {
            return Gv(GvVal::Lit(1.0));
        }
        let mut base = self;
        let mut exp = n.unsigned_abs();
        let mut acc: Option<Gv> = None;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = Some(match acc {
                    None => base,
                    Some(a) => a * base,
                });
            }
            exp >>= 1;
            if exp > 0 {
                base = base * base;
            }
        }
        let pos = acc.expect("n != 0 implies acc is set");
        if n < 0 {
            Gv(GvVal::Lit(1.0)) / pos
        } else {
            pos
        }
    }
    fn powf(self, e: Gv) -> Gv {
        self.binop(e, ElementWiseOp::Pow)
    }

    fn floor(self) -> Gv {
        self.unop(ElementWiseOp::Floor)
    }
    fn ceil(self) -> Gv {
        self.unop(ElementWiseOp::Ceil)
    }

    // ── hyperbolics — graph-op lowerings ────────────────
    fn sinh(self) -> Gv {
        self.unop(ElementWiseOp::Sinh)
    }
    fn cosh(self) -> Gv {
        self.unop(ElementWiseOp::Cosh)
    }
    fn tanh(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Tanh)
    }
    fn asinh(self) -> Gv {
        self.unop(ElementWiseOp::Asinh)
    }
    fn acosh(self) -> Gv {
        self.unop(ElementWiseOp::Acosh)
    }
    fn atanh(self) -> Gv {
        self.transcendental_unop(TranscendentalOp::Atanh)
    }

    // ── HIGHER-ORDER: iterate + iterate_vec with the FREEZE LAW ───────────
    fn iterate(
        self,
        max_steps: usize,
        body: impl Fn(Self) -> Self,
        converged: impl Fn(Self, Self) -> GvMask,
    ) -> Gv {
        // carrier equivalence: traced kernel must return the SAME value the host
        // loop returns for ANY input. freeze on convergence — see the FREEZE LAW
        // in `crate::algebra::Scalar::iterate` doc. ALSO pass `conv` as the IR's
        // `break_when` so the loop EXITS once convergence fires, skipping the
        // remaining `max_steps` worth of dead body (the freeze nulled the writes
        // but the cone's arithmetic ran every iter — see RMHD c2p perf).
        let acc = with_trace(|t| t.graph.iter_acc(0, None));
        let cur = Gv::of(acc);
        let next = body(cur);
        let conv = converged(cur, next);
        let step = <Self as crate::algebra::Scalar>::select(conv, cur, next).node();
        let break_when = Some(conv.0.node());
        let init = self.node();
        Gv::of(with_trace(|t| {
            t.graph
                .iterate_inline_scalar(acc, init, step, max_steps, break_when, None)
        }))
    }

    fn iterate_vec<const N: usize>(
        init: [Self; N],
        max_steps: usize,
        body: impl Fn([Self; N]) -> [Self; N],
        converged: impl Fn([Self; N], [Self; N]) -> GvMask,
        result: usize,
    ) -> Gv {
        let accs: [NodeId; N] =
            std::array::from_fn(|j| with_trace(|t| t.graph.iter_acc(j as u32, None)));
        let acc_gv: [Gv; N] = accs.map(Gv::of);
        let next = body(acc_gv);
        let conv = converged(acc_gv, next);
        let steps: [Gv; N] = std::array::from_fn(|j| {
            <Self as crate::algebra::Scalar>::select(conv, acc_gv[j], next[j])
        });
        let break_when = Some(conv.0.node());
        let inits_n: Vec<NodeId> = init.iter().map(|g| g.node()).collect();
        let steps_n: Vec<NodeId> = steps.iter().map(|g| g.node()).collect();
        Gv::of(with_trace(|t| {
            t.graph.iterate_inline(
                accs.to_vec(),
                inits_n,
                steps_n,
                max_steps,
                result as u32,
                break_when,
                None,
            )
        }))
    }
}

// =============================================================================
// fusion algebra: 9 laws on `try_fuse` over the grade `LaunchGrade`.
//
// the algebra is a graded commutative monoid:
//   - grade = LaunchGrade = fingerprint of a `symbi_algebra::Domain<R>`
//   - identity per grade = GvKernel::noop(grade)
//   - operation = try_fuse, partial (fails on mismatched grade, shared writes,
//     or inter-dependency)
//
// the tests are SYNTHETIC (no physics): small traces with two independent
// fields, writes declared manually, asserting the algebraic properties hold
// structurally. these properties are what let consumer code (godunov +
// bcell_godunov, snapshot + bcell_snapshot, c2p + wave_speed_map) fuse via
// `try_fuse` soundly.
// =============================================================================

#[cfg(test)]
mod fusion_laws {
    use super::*;
    use symbi_algebra::{Space, domain};

    // shared interior fixture: a 4-cell axis. tests that need two distinct
    // grades use `edge_grade()` for an axis-shifted half-extent domain.
    fn interior_grade() -> LaunchGrade {
        LaunchGrade::from_domain(&domain([Space {
            name: "i",
            lo: 0,
            hi: 4,
        }]))
    }

    // distinct grade: same axis name, different extent. structural equality
    // separates these from `interior_grade()`.
    fn edge_grade() -> LaunchGrade {
        LaunchGrade::from_domain(&domain([Space {
            name: "i",
            lo: 0,
            hi: 5,
        }]))
    }

    // build a single-output kernel that reads `in_key`@`in_path`, doubles it,
    // declares its root as `out_key`@`out_path`. used as a uniform fixture.
    fn doubler(
        in_key: &str,
        in_path: &str,
        out_key: &str,
        out_path: &str,
        grade: LaunchGrade,
    ) -> (GvKernel, Writes) {
        begin_trace();
        let x = Gv::field(in_key, in_path);
        let two = Gv(GvVal::Lit(2.0));
        let y = x * two;
        let root = y.node();
        let kern = end_trace_with(grade);
        let writes = vec![(out_key.to_string(), out_path.into(), root)];
        (kern, writes)
    }

    // structural-equality helper: two manifests are "the same" iff their
    // field-input sets, scalar-param sets, and write-path sets coincide. node
    // counts may differ across orderings (CSE inside splice can collapse
    // duplicates), so the comparison is over semantic sets.
    fn manifest_sets(
        k: &GvKernel,
        w: &Writes,
    ) -> (HashSet<(String, String)>, HashSet<String>, HashSet<String>) {
        let inputs: HashSet<_> = k
            .field_inputs
            .iter()
            .map(|(k, b)| (k.clone(), b.name()))
            .collect();
        let scalars: HashSet<_> = k.scalar_params.iter().cloned().collect();
        let writes: HashSet<_> = w.iter().map(|(_, p, _)| p.name()).collect();
        (inputs, scalars, writes)
    }

    // identity. fusing with `noop(g)` is a no-op on the manifest sets.
    #[test]
    fn law_identity_left_and_right() {
        let g = interior_grade();
        let (k, w) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let baseline = manifest_sets(&k, &w);

        // left identity: fuse(noop(g), k) == k
        let (noop_k, noop_w) = GvKernel::noop(g.clone());
        let (left, left_w) =
            try_fuse(noop_k, noop_w, k.clone(), w.clone()).expect("fuse(noop, k) must succeed");
        assert_eq!(
            baseline,
            manifest_sets(&left, &left_w),
            "left-identity violated: fuse(noop(g), k) changed the manifest"
        );

        // right identity: fuse(k, noop(g)) == k
        let (noop_k2, noop_w2) = GvKernel::noop(g);
        let (right, right_w) =
            try_fuse(k.clone(), w.clone(), noop_k2, noop_w2).expect("fuse(k, noop) must succeed");
        assert_eq!(
            baseline,
            manifest_sets(&right, &right_w),
            "right-identity violated: fuse(k, noop(g)) changed the manifest"
        );
    }

    // associativity. fuse(fuse(a,b),c) and fuse(a,fuse(b,c)) produce
    // the same manifest sets. nodes may differ because of CSE order, but the
    // observable ABI is the same.
    #[test]
    fn law_associativity() {
        let g = interior_grade();
        let (a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g.clone());
        let (c, cw) = doubler("c_in", "p.c_in", "c_out", "p.c_out", g);

        let (ab, abw) = try_fuse(a.clone(), aw.clone(), b.clone(), bw.clone()).expect("fuse(a, b)");
        let (ab_c, ab_cw) = try_fuse(ab, abw, c.clone(), cw.clone()).expect("fuse(fuse(a, b), c)");

        let (bc, bcw) = try_fuse(b, bw, c, cw).expect("fuse(b, c)");
        let (a_bc, a_bcw) = try_fuse(a, aw, bc, bcw).expect("fuse(a, fuse(b, c))");

        assert_eq!(
            manifest_sets(&ab_c, &ab_cw),
            manifest_sets(&a_bc, &a_bcw),
            "associativity violated",
        );
    }

    // commutativity-mod-disjoint. when writes are disjoint and there is
    // no inter-dep, fuse(a,b) and fuse(b,a) have the same manifest sets.
    // first-seen ordering of the input list differs, but the SET is invariant.
    #[test]
    fn law_commutativity_mod_disjoint() {
        let g = interior_grade();
        let (a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);

        let (ab, abw) = try_fuse(a.clone(), aw.clone(), b.clone(), bw.clone()).expect("fuse(a, b)");
        let (ba, baw) = try_fuse(b, bw, a, aw).expect("fuse(b, a)");

        assert_eq!(
            manifest_sets(&ab, &abw),
            manifest_sets(&ba, &baw),
            "commutativity (mod-disjoint) violated"
        );
    }

    // equivalence. fusing a + b preserves a's write roots verbatim
    // (the target graph started as a clone of a's graph), and produces valid
    // node ids in the fused graph for b's writes (so b's roots survived the
    // splice and are addressable). this is the structural surrogate for "the
    // fused kernel computes the same outputs as the sequence".
    #[test]
    fn law_equivalence_structural() {
        let g = interior_grade();
        let (a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        let a_roots: Vec<NodeId> = aw.iter().map(|(_, _, n)| *n).collect();

        let (fused, fused_w) = try_fuse(a, aw, b, bw).expect("fuse(a, b)");

        // a's writes preserved verbatim at the head of the writes manifest.
        for (i, &a_root) in a_roots.iter().enumerate() {
            assert_eq!(
                fused_w[i].2, a_root,
                "a-write root #{i} was renumbered; equivalence broken"
            );
        }

        // every fused write root is a valid node in the fused graph.
        let n_nodes = fused.graph.len();
        for (k, p, root) in fused_w.iter() {
            assert!(
                (root.0 as usize) < n_nodes,
                "fused write {k}@{} has dangling NodeId {:?}",
                p.name(),
                root
            );
        }
    }

    // grade rejection. two kernels of different grades cannot fuse.
    #[test]
    fn law_grade_rejection() {
        let (a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", interior_grade());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", edge_grade());

        match try_fuse(a, aw, b, bw) {
            Err(FusionError::GradeMismatch { .. }) => {}
            other => panic!("expected GradeMismatch, got {other:?}"),
        }
    }

    // untagged is not fusable, even with itself. tagging is opt-in
    // — the algebra refuses to assume a default.
    #[test]
    fn law_untagged_rejection() {
        let (a, aw) = doubler(
            "a_in",
            "p.a_in",
            "a_out",
            "p.a_out",
            LaunchGrade::untagged(),
        );
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", interior_grade());

        match try_fuse(a.clone(), aw.clone(), b.clone(), bw.clone()) {
            Err(FusionError::UntaggedKernel) => {}
            other => panic!("expected UntaggedKernel, got {other:?}"),
        }
        match try_fuse(b, bw, a, aw) {
            Err(FusionError::UntaggedKernel) => {}
            other => panic!("expected UntaggedKernel (symmetric), got {other:?}"),
        }
    }

    // write-conflict rejection. two kernels writing the same runtime
    // path race in a fused launch. reject syntactically.
    #[test]
    fn law_write_conflict_rejection() {
        let g = interior_grade();
        let (a, aw) = doubler("a_in", "p.a_in", "shared_out", "p.shared", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "shared_out", "p.shared", g);

        match try_fuse(a, aw, b, bw) {
            Err(FusionError::WriteConflict { runtime_path }) => {
                assert_eq!(runtime_path, "p.shared");
            }
            other => panic!("expected WriteConflict, got {other:?}"),
        }
    }

    // inter-dep rejection. a writes X, b reads X — fusing would let
    // b's stencil reads observe a's just-written values at non-local
    // neighbor indices without a grid-wide barrier. reject.
    #[test]
    fn law_inter_dep_rejection_forward() {
        let g = interior_grade();
        // a: reads p.src, writes p.shared
        let (a, aw) = doubler("a_in", "p.src", "shared_out", "p.shared", g.clone());
        // b: reads p.shared (what a just wrote), writes p.b_out
        let (b, bw) = doubler("b_in", "p.shared", "b_out", "p.b_out", g);

        match try_fuse(a, aw, b, bw) {
            Err(FusionError::InterDep { written, .. }) => {
                assert_eq!(written, "p.shared");
            }
            other => panic!("expected InterDep, got {other:?}"),
        }
    }

    // inter-dep rejection is symmetric. b writes Y, a reads Y — same
    // hazard with arguments reversed.
    #[test]
    fn law_inter_dep_rejection_reverse() {
        let g = interior_grade();
        // a: reads p.shared, writes p.a_out
        let (a, aw) = doubler("a_in", "p.shared", "a_out", "p.a_out", g.clone());
        // b: writes p.shared (what a is reading from)
        let (b, bw) = doubler("b_in", "p.src", "shared_out", "p.shared", g);

        match try_fuse(a, aw, b, bw) {
            Err(FusionError::InterDep { written, .. }) => {
                assert_eq!(written, "p.shared");
            }
            other => panic!("expected InterDep (reverse), got {other:?}"),
        }
    }

    // grade derives from Domain<R>. two grades built from the same
    // Domain (same R, same Space[s]) compare equal even when the Domain
    // values were constructed by separate calls — the algebra is STRUCTURAL,
    // not identity-based. (DomainId differs across constructions.)
    #[test]
    fn law_grade_is_structural_over_domain() {
        let d1 = domain([Space {
            name: "i",
            lo: 0,
            hi: 4,
        }]);
        let d2 = domain([Space {
            name: "i",
            lo: 0,
            hi: 4,
        }]);
        assert_ne!(
            d1.id, d2.id,
            "DomainId is identity-based — independent constructions differ"
        );
        assert_eq!(
            LaunchGrade::from_domain(&d1),
            LaunchGrade::from_domain(&d2),
            "LaunchGrade must be structural, not identity-based",
        );

        // grades over different ranks are distinct.
        let d3 = domain([
            Space {
                name: "i",
                lo: 0,
                hi: 4,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 4,
            },
        ]);
        assert_ne!(
            LaunchGrade::from_domain(&d1),
            LaunchGrade::from_domain(&d3),
            "grades over different ranks must differ",
        );
    }

    // ----- tile-spec laws (new) -----

    /// fusion preserves the tile_spec when BOTH sides match. directly sets
    /// `tile_spec` on each kernel (bypassing the `with_tile_spec` builder's
    /// manifest check, which is tested separately) so a single shared spec sits
    /// on both halves regardless of their disjoint inputs.
    #[test]
    fn law_tile_spec_preserved_when_both_match() {
        let g = interior_grade();
        let (mut a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (mut b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        let spec = TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["shared".to_string()],
        };
        a.tile_spec = Some(spec.clone());
        b.tile_spec = Some(spec.clone());
        let (fused, _w) = try_fuse(a, aw, b, bw).expect("matched tile specs must fuse");
        assert_eq!(
            fused.tile_spec,
            Some(spec),
            "fused kernel must carry the shared spec"
        );
    }

    /// fuse(tiled, untiled) must fail with TileSpecMismatch — one kernel
    /// can't share a launch with another that needs a different smem layout.
    #[test]
    fn law_tile_spec_mismatch_rejected() {
        let g = interior_grade();
        let (mut a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        a.tile_spec = Some(TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["a_in".to_string()],
        });
        match try_fuse(a, aw, b, bw) {
            Err(FusionError::TileSpecMismatch { a, b }) => {
                assert!(
                    a.is_some() && b.is_none(),
                    "expected a=tiled, b=untiled in the mismatch report"
                );
            }
            other => panic!("expected TileSpecMismatch, got {other:?}"),
        }
    }

    /// two tiled kernels with DIFFERENT halo widths also can't fuse — a single
    /// smem prelude can only sit one halo at a time.
    #[test]
    fn law_tile_spec_different_halo_rejected() {
        let g = interior_grade();
        let (mut a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (mut b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        a.tile_spec = Some(TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["a_in".to_string()],
        });
        b.tile_spec = Some(TileSpec {
            halo: vec![3],
            tiled_field_keys: vec!["b_in".to_string()],
        });
        match try_fuse(a, aw, b, bw) {
            Err(FusionError::TileSpecMismatch { .. }) => {}
            other => panic!("expected TileSpecMismatch, got {other:?}"),
        }
    }

    /// `with_tile_spec` must reject keys absent from the kernel's manifest —
    /// otherwise the dispatch has no buffer to bind to the tiled slot.
    #[test]
    #[should_panic(expected = "not in this kernel's field_inputs manifest")]
    fn with_tile_spec_rejects_unknown_field() {
        let g = interior_grade();
        let (a, _aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g);
        let _ = a.with_tile_spec(TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["not_in_manifest".to_string()],
        });
    }

    /// an all-zero halo is meaningless (no smem benefit). builder rejects it.
    #[test]
    #[should_panic(expected = "at least one axis halo must be > 0")]
    fn with_tile_spec_rejects_zero_halo() {
        let g = interior_grade();
        let (a, _aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g);
        let _ = a.with_tile_spec(TileSpec {
            halo: vec![0, 0, 0],
            tiled_field_keys: vec!["a_in".to_string()],
        });
    }

    /// the smem-footprint helper is the load-bearing piece for the
    /// dispatch-side smem allocation. small canonical cases:
    ///   - cube halo=[2,2,2], block (16, 8, 4), 1 field, f64 (8 bytes):
    ///       tile = (16+4) * (8+4) * (4+4) = 20*12*8 = 1920 cells
    ///       bytes = 1920 * 8 * 1 = 15,360
    ///   - SLAB halo=[2,0,0] (reconstruct along axis 0 only), block (32, 8, 1),
    ///     8 fields, f64: tile = (32+4) * 8 * 1 = 288 cells
    ///       bytes = 288 * 8 * 8 = 18,432 — vs the 51,840 a fat cube would cost.
    #[test]
    fn tile_spec_smem_footprint_math() {
        let spec_cube = TileSpec {
            halo: vec![2, 2, 2],
            tiled_field_keys: vec!["x".to_string()],
        };
        assert_eq!(spec_cube.smem_bytes_per_block(&[16, 8, 4], 8), 1920 * 8);

        let keys: Vec<String> = (0..8).map(|k| format!("f{k}")).collect();
        let spec_slab = TileSpec {
            halo: vec![2, 0, 0],
            tiled_field_keys: keys,
        };
        assert_eq!(spec_slab.smem_bytes_per_block(&[32, 8, 1], 8), 288 * 8 * 8);
    }

    /// existing fusion laws don't regress when both kernels declare `None`
    /// for `tile_spec` (the untiled path). identity + grade-mismatch +
    /// untagged still flow through `try_fuse` unchanged.
    #[test]
    fn untiled_kernels_still_fuse_as_before() {
        let g = interior_grade();
        let (a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        let (fused, _w) = try_fuse(a, aw, b, bw).expect("legacy untiled fusion must still work");
        assert!(fused.tile_spec.is_none(), "untiled+untiled -> untiled");
    }
}

// =============================================================================
// Scalar::scope at S = Gv — emits a real frame.
//
// at f64 the default `Scalar::scope` impl is identity (the closure runs
// inline). at Gv the override above snapshots the trace, runs the closure,
// captures every NEW NodeId as the scope's body, and pushes an Op::Scope
// node carrying body + result. consequences:
//
//   - the scope-form graph contains ONE MORE node than the inline-form
//     graph (the Op::Scope itself);
//   - all arithmetic nodes are identical in both forms (the scope is a
//     PURE wrapper — it pushes nothing extra into the body region);
//   - nested Gv::scope produces nested Op::Scope nodes; structural shape
//     under graph.iter() is preserved (nested scopes appear as inner
//     Op::Scope nodes pushed BEFORE the outer one).
//
// these tests pin that contract.
// =============================================================================

#[cfg(test)]
mod scope_op_contract {
    use super::*;
    use crate::graph::Op;
    use symbi_algebra::Numeric;

    /// at Gv, `scope(|| body)` produces inline-form + exactly ONE extra
    /// Op::Scope node at the tail. body subgraph is otherwise identical.
    #[test]
    fn gv_scope_emits_one_op_scope_node() {
        // scope-form: (x + y) * (x + y) wrapped in Gv::scope.
        begin_trace();
        let x = Gv::scalar("x");
        let y = Gv::scalar("y");
        let scope_root = <Gv as crate::algebra::Scalar>::scope(|| {
            let s = x + y;
            s * s
        });
        let scope_graph = end_trace_with(LaunchGrade::untagged()).graph;
        let scope_root_node = scope_root.node();

        // inline-form: same math, no wrapper.
        begin_trace();
        let x = Gv::scalar("x");
        let y = Gv::scalar("y");
        let s = x + y;
        let inline_root = s * s;
        let inline_graph = end_trace_with(LaunchGrade::untagged()).graph;
        let inline_root_node = inline_root.node();

        // scope-form has +1 node (the Op::Scope itself); body subgraph
        // matches inline-form node-for-node.
        assert_eq!(
            scope_graph.len(),
            inline_graph.len() + 1,
            "scope-form must add exactly one Op::Scope node over inline-form"
        );
        // root of scope-form is the Op::Scope (last pushed); root of
        // inline-form is the trailing Mul.
        assert_eq!(
            scope_root_node.0 as usize,
            scope_graph.len() - 1,
            "scope root must be the Op::Scope tail node"
        );
        assert_eq!(
            inline_root_node.0 as usize,
            inline_graph.len() - 1,
            "inline root must be the trailing Mul node"
        );
        // the first N nodes of scope-form match inline-form (the body
        // subgraph is pushed identically — only the trailing Op::Scope
        // differs).
        for ((id, n_a, t_a), (_, n_b, t_b)) in scope_graph
            .iter()
            .take(inline_graph.len())
            .zip(inline_graph.iter())
        {
            assert_eq!(&n_a.op, &n_b.op, "node {id:?} op mismatch in body region");
            assert_eq!(t_a, t_b, "node {id:?} ty mismatch in body region");
        }
        // tail node MUST be Op::Scope whose result is the trailing Mul.
        match &scope_graph.node(scope_root_node).op {
            Op::Scope { body, result } => {
                assert_eq!(
                    *result, inline_root_node,
                    "Op::Scope.result must equal the inline trailing Mul NodeId"
                );
                // body lists every node added during the closure (the +
                // and the *), in insertion order. for this fixture that's
                // exactly the last two nodes before the Op::Scope.
                let expected_body: Vec<NodeId> = (inline_graph.len() - 2..inline_graph.len())
                    .map(|i| NodeId(i as u32))
                    .collect();
                assert_eq!(
                    body, &expected_body,
                    "Op::Scope.body must list the closure's new NodeIds in order"
                );
            }
            other => panic!("expected Op::Scope tail, got {other:?}"),
        }
    }

    /// nesting Gv::scope: inner Op::Scope appears BEFORE the outer one.
    /// outer body must contain the inner Op::Scope NodeId.
    #[test]
    fn gv_scope_nests_with_inner_op_scope() {
        begin_trace();
        let x = Gv::scalar("x");
        let outer_root = <Gv as crate::algebra::Scalar>::scope(|| {
            let a = <Gv as crate::algebra::Scalar>::scope(|| x + Gv::from_f64(1.0));
            let b = <Gv as crate::algebra::Scalar>::scope(|| x * Gv::from_f64(2.0));
            a + b
        });
        let graph = end_trace_with(LaunchGrade::untagged()).graph;
        let outer_root_node = outer_root.node();

        // outer Op::Scope is the tail.
        match &graph.node(outer_root_node).op {
            Op::Scope { body, result: _ } => {
                // outer body contains TWO inner Op::Scope NodeIds (a, b)
                // and one Add (a + b). assert at least one body entry IS
                // an Op::Scope.
                let mut inner_scope_count = 0;
                for &bid in body {
                    if matches!(graph.node(bid).op, Op::Scope { .. }) {
                        inner_scope_count += 1;
                    }
                }
                assert_eq!(
                    inner_scope_count, 2,
                    "outer Op::Scope.body must contain BOTH inner Op::Scope NodeIds"
                );
            }
            other => panic!("expected outer Op::Scope, got {other:?}"),
        }
    }

    /// empty-body short-circuit: a closure whose result is a pre-existing
    /// NodeId (no new nodes pushed) must NOT emit an Op::Scope.
    #[test]
    fn gv_scope_empty_body_short_circuits() {
        begin_trace();
        let x = Gv::scalar("x");
        let before = with_trace(|t| t.graph.len());
        // closure result is `x` itself — no new nodes.
        let r = <Gv as crate::algebra::Scalar>::scope(|| x);
        let after = with_trace(|t| t.graph.len());
        let g = end_trace_with(LaunchGrade::untagged()).graph;
        assert_eq!(before, after, "empty-body scope must not push any node");
        assert_eq!(
            r.node(),
            x.node(),
            "empty-body scope must return the input directly"
        );
        // and the final graph has no Op::Scope.
        for (_, n, _) in g.iter() {
            assert!(
                !matches!(n.op, Op::Scope { .. }),
                "empty-body scope must not emit Op::Scope"
            );
        }
    }
}

#[cfg(test)]
mod powi_carrier_equiv {
    use super::*;
    use crate::backends::interp::{Backend, Cpu};
    use crate::graph::{ElementWiseOp, Op};
    use crate::passes::scalarize::scalarize;

    // `Gv::powi` must NOT lower to Pow/powf: `f64::powi` raises a NEGATIVE base exactly
    // (e.g., (-2)^2 = 4), but `powf(neg, 2.0)` = NaN on CUDA — a carrier-equivalence break
    // (the f64 host path != the traced kernel). this pins the structural fix (multiply
    // chain, no Pow node) AND the numeric agreement with `f64::powi` on negative bases.
    #[test]
    fn powi_lowers_to_multiplies_and_matches_f64_on_negative_base() {
        for n in [0_i32, 1, 2, 3, 4, 5, -2, -3] {
            begin_trace();
            let x = Gv::scalar("x");
            let r = <Gv as crate::algebra::Scalar>::powi(x, n);
            let root = r.node();
            let graph = end_trace_with(LaunchGrade::untagged()).graph;

            // structural: NO Pow op anywhere (the bug was lowering to powf).
            let has_pow = (0..graph.len()).any(|i| {
                matches!(
                    graph.node(NodeId(i as u32)).op,
                    Op::ElementWise(ElementWiseOp::Pow, _)
                )
            });
            assert!(!has_pow, "powi({n}) must lower to multiplies, not Pow/powf");

            // numeric: evaluate the traced graph and require bit-agreement with f64::powi
            // for NEGATIVE (and a few positive) bases — the exact case powf gets wrong.
            let f = scalarize(&graph, root, "out");
            for &base in &[-2.0_f64, -1.5, -0.5, 0.5, 3.0] {
                let got = Cpu.eval_elemental(&f, &[base])[0];
                let want = base.powi(n);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "powi({base}, {n}): traced kernel = {got}, f64::powi = {want}"
                );
            }
        }
    }

    // the defaulted carrier helpers `safe_sqrt` (= sqrt(max(.,0))) and `clamp` must trace at
    // S=Gv AND match the f64 path bit-for-bit. safe_sqrt of a NEGATIVE radicand is 0 (the
    // clamp-before-sqrt that keeps an unguarded sqrt(neg) from tracing a NaN into the kernel).
    #[test]
    fn safe_sqrt_and_clamp_trace_and_match_f64() {
        use crate::algebra::Scalar;
        // safe_sqrt
        begin_trace();
        let x = Gv::scalar("x");
        let root = Scalar::safe_sqrt(x).node();
        let g = end_trace_with(LaunchGrade::untagged()).graph;
        let f = scalarize(&g, root, "out");
        for &v in &[-4.0_f64, -1e-9, 0.0, 0.25, 9.0] {
            let got = Cpu.eval_elemental(&f, &[v])[0];
            let want = v.max(0.0).sqrt();
            assert!(
                (got - want).abs() <= 1e-12,
                "safe_sqrt({v}): {got} != {want}"
            );
        }
        // clamp into [-1, 1]
        begin_trace();
        let y = Gv::scalar("y");
        let lo = Gv(GvVal::Lit(-1.0));
        let hi = Gv(GvVal::Lit(1.0));
        let croot = Scalar::clamp(y, lo, hi).node();
        let cg = end_trace_with(LaunchGrade::untagged()).graph;
        let cf = scalarize(&cg, croot, "out");
        for &v in &[-3.0_f64, -1.0, -0.2, 0.5, 1.0, 5.0] {
            let got = Cpu.eval_elemental(&cf, &[v])[0];
            let want = v.max(-1.0).min(1.0);
            assert!(
                (got - want).abs() <= 1e-12,
                "clamp({v}, -1, 1): {got} != {want}"
            );
        }
    }
}
