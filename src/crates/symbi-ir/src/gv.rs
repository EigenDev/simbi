// =============================================================================
// gv.rs
//
// the `Gv` ("graph value") carrier + the scoped trace it records into.
// `Gv<'t>` is a `symbi_algebra::Scalar` whose operations record a node into
// this crate's tensor graph and return a handle to it; instantiating
// carrier-generic physics (written over `S: Scalar`) at `S = Gv` traces it into
// the stencil IR — the foundational "code -> graph" boundary. graph and carrier
// live in one crate so the IR machine is a single layer.
//
// tracing is a scoped interpretation: `trace(|cx| ...)` opens a fresh graph,
// runs the closure with a `TraceCx<'t>` token, and returns the finished
// `GvKernel`. the closure is generative over the brand lifetime `'t`, so a
// `Gv<'t>` provably cannot escape its trace or combine with a value from a
// sibling trace — the compiler rejects both. a per-trace generation stamp on
// every node handle backs the brand at runtime: a value from an enclosing
// trace used inside a nested one panics at the recording site instead of
// silently splicing foreign node ids. build-time only.
//
// the trace (`GvTrace`, reached through `TraceCx::with_trace`) is `pub` so the
// discretization builders in symbi-discretize can construct raw index/stencil
// IR (integer coord arithmetic, which lives in the graph's I32 domain alongside
// the f64 carrier).
// =============================================================================

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::graph::{ConstValue, ElementWiseOp, Graph, NodeId};
use crate::{ElementTy, InputKey, OutputKey, ScalarParam, Symbol};
use symbi_abi::FieldBind;
use symbi_algebra::{Domain, FieldElement, Space};

/// the finished trace: the stencil graph plus the kernel ABI manifest the emit
/// pipeline consumes — field-buffer inputs as `(ir_key, runtime_path)`, scalar
/// params, and the spatial coord axes a stencil references — all in first-seen order.
#[derive(Clone, Debug)]
pub struct GvKernel {
    graph: Graph,
    /// field-buffer reads as `(ir_key, FieldBind)`: the IR-side load name paired with the
    /// born-typed runtime binding the producer minted. the binding is a
    /// `FieldBind` (typed `Ref` for the closed cell vocabulary, `Scratch` for generated paths) —
    /// structured data every consumer reads directly.
    field_inputs: Vec<(InputKey, FieldBind)>,
    scalar_params: Vec<ScalarParam>,
    /// spatial axes whose `_coord_N` the trace referenced (for stencil `load_at`);
    /// empty for pointwise kernels. feeds `KernelEmitInputs::coord_components`.
    coord_components: Vec<u8>,
    /// the launch grade this kernel is to be issued over. fusion (`try_fuse`)
    /// requires both sides to share a grade, and a tagged grade is what admits a
    /// kernel to fusion at all. see `LaunchGrade` and `try_fuse`.
    grade: LaunchGrade,
    /// floating-point rewrite theory under which this graph was constructed.
    numerical_policy: NumericalPolicy,
    /// optional shared-memory tile specification. when `Some`, the CUDA emit is
    /// expected to allocate a per-block `__shared__` buffer for each tiled
    /// field, cooperatively prefetch the (block + halo) region, sync, and
    /// redirect that field's stencil `LoadAt` reads to smem. when `None`, the
    /// emit produces the prior gmem-per-thread pattern.
    ///
    /// the type carries the spec; the CUDA emit and LoadAt rewriting paths
    /// remain to be built. the field exists so builders, fusion, and the
    /// runtime extend behind a stable enum shape.
    tile_spec: Option<TileSpec>,
    /// the declared support of this kernel's outputs:
    /// a region outside which every output is exactly zero for any field input.
    /// declared by the builder (where the saturation constants live), carried
    /// into the serialized `Prepared` blob, consumed by dispatch (reduction /
    /// launch regions). `None` = Everywhere (always sound).
    output_support: Option<crate::support::Support>,
    /// builder-tagged support balls by node, carried from the trace for
    /// `with_derived_support` propagation. build-time metadata that stays in
    /// memory; fusion drops it (fused kernels derive support before fusing).
    node_supports: std::collections::HashMap<NodeId, crate::support_infer::SupportBall>,
}

/// The analytic theory available to IR rewrites. A policy grants permission;
/// it never forces a backend to perform a relaxed rewrite.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum NumericalPolicy {
    #[default]
    StrictIeee,
    FiniteOnly,
    FastMath,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RewriteClass {
    Structural,
    BitExact,
    FiniteOnly,
    FastMath,
}

impl NumericalPolicy {
    pub const fn permits(self, rewrite: RewriteClass) -> bool {
        match (self, rewrite) {
            (_, RewriteClass::Structural | RewriteClass::BitExact) => true,
            (Self::FiniteOnly | Self::FastMath, RewriteClass::FiniteOnly) => true,
            (Self::FastMath, RewriteClass::FastMath) => true,
            _ => false,
        }
    }
}

/// shared-memory tile spec for stencil kernels. names the fields to prefetch
/// into smem and the halo width per axis. consumed by the CUDA backend at
/// emit time and by the runtime when computing per-block smem requirements.
///
/// invariants:
///   - `halo` has one entry per axis (`halo.len() == kernel ndim`) and at least
///     one entry is `> 0` (smem reuse requires an extended axis; the untiled
///     case is `tile_spec = None`). a per-axis halo lets a direction-`dir` flux
///     kernel, which reconstructs along a single axis, prefetch a thin slab
///     (halo on `dir`, 0 transverse); a fat cube loads ~7.5x
///     more cells than the physics reads and spends the perf win.
///   - every `tiled_field_keys[i]` appears in `GvKernel::field_inputs` — the
///     dispatch binds a buffer through the manifest entry, so tiling is defined
///     exactly for manifest fields
///   - the per-block thread layout (`block_dims`) plus halo determines the
///     smem footprint: `prod_d (block_dims[d] + 2*halo[d]) * sizeof(S) * n_fields`.
///     callers must ensure this fits in the device's smem-per-block budget
///     (48 KB on Turing by default, 64 KB with `cudaFuncSetAttribute`)
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TileSpec {
    /// halo cells per axis (length = kernel ndim). a reconstruction along axis
    /// `dir` with PLM radius 2 sets `halo[dir] = 2`, the transverse axes `0`.
    /// the cooperative load extends the block by `halo[d]` on each side of axis
    /// `d`; an axis with `halo[d] == 0` keeps the block extent (yielding a slab).
    pub halo: Vec<u8>,
    /// fields to prefetch into smem, by IR key (matching `field_inputs[i].0`).
    /// stencil `LoadAt` reads for these keys are routed through smem; pointwise
    /// reads of the remaining fields stay on gmem.
    pub tiled_field_keys: Vec<InputKey>,
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
/// kernels. fusion admits tagged grades only, so an untagged kernel stands
/// alone — even against another untagged kernel.
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

    /// the sentinel that opts out of fusion. used by the plain [`trace`] entry.
    pub fn untagged() -> Self {
        Self::default()
    }

    /// true iff this grade was tagged with a real `Domain<R>` fingerprint.
    pub fn is_tagged(&self) -> bool {
        !self.spaces.is_empty()
    }
}

/// one effectful kernel output: the IR-side output name, its runtime
/// destination, and the graph value stored there.
///
/// named fields make the three namespaces explicit. in particular, `key` is
/// an IR/ABI symbol while `destination` identifies storage; neither can be
/// mistaken for the graph-owned `value` as they could in a positional tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelWrite {
    pub key: OutputKey,
    pub destination: FieldBind,
    pub value: NodeId,
}

impl KernelWrite {
    pub fn new(
        key: impl Into<OutputKey>,
        destination: impl Into<FieldBind>,
        value: NodeId,
    ) -> Self {
        Self {
            key: key.into(),
            destination: destination.into(),
            value,
        }
    }
}

/// the named write-effect manifest used by kernel composition.
pub type KernelWrites = Vec<KernelWrite>;

#[derive(Clone, Debug)]
pub enum FusionError {
    /// the two kernels declare different launch grades. fusing would dispatch
    /// the merged body over a domain that's wrong for at least one half.
    GradeMismatch { a: LaunchGrade, b: LaunchGrade },
    /// either kernel is untagged — fusion requires both sides to be tagged
    /// with a real `Domain<R>` fingerprint via `trace_for_domain`.
    UntaggedKernel,
    NumericalPolicyMismatch {
        a: NumericalPolicy,
        b: NumericalPolicy,
    },
    /// both kernels write the same runtime path. data races by construction.
    WriteConflict { runtime_path: String },
    /// one kernel writes a path the other reads. fusing them would change
    /// semantics: the read in the would-be-second-launch would see the
    /// would-be-first-launch's update inside the same kernel body. callers
    /// must serialize via two launches (or restructure to remove the dep).
    InterDep { written: String, read: String },
    /// the two kernels declare incompatible tile specs (different halo widths,
    /// or one tiled and the other untiled). a single fused launch runs one
    /// block layout, so callers either align the specs or keep the kernels in
    /// separate launches.
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
                "fuse: at least one kernel is untagged; tag both via `trace_for_domain`"
            ),
            FusionError::NumericalPolicyMismatch { a, b } => {
                write!(f, "fuse: numerical-policy mismatch — a={a:?}, b={b:?}")
            }
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

/// the invariant brand tying a symbolic value to the trace that owns it. the
/// lifetime appears in both covariant and contravariant position, so two
/// distinct trace scopes can never unify their brands.
type Brand<'t> = std::marker::PhantomData<fn(&'t ()) -> &'t ()>;

/// the active trace: graph + the manifest accumulated as field reads / scalar params /
/// coord references are recorded. dedup sets keep the manifest first-seen-unique.
pub struct GvTrace {
    /// process-unique generation stamp minted at trace open. every node handle
    /// carries its trace's stamp, and recording verifies the stamps agree —
    /// the runtime backstop for the `'t` brand across nested traces.
    id: u64,
    graph: Graph,
    field_inputs: Vec<(InputKey, FieldBind)>,
    field_keys: HashSet<InputKey>,
    scalar_params: Vec<ScalarParam>,
    scalar_keys: HashSet<ScalarParam>,
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

fn fresh_trace() -> GvTrace {
    static NEXT_TRACE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    GvTrace {
        id: NEXT_TRACE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        graph: Graph::new(),
        field_inputs: Vec::new(),
        field_keys: HashSet::new(),
        scalar_params: Vec::new(),
        scalar_keys: HashSet::new(),
        coord_components: Vec::new(),
        coord_nodes: HashMap::new(),
        node_supports: HashMap::new(),
    }
}

/// restores the trace that surrounded a scoped [`trace`] call.
///
/// `Drop` is load-bearing: if the closure panics, unwinding discards the
/// incomplete inner graph and reinstates the outer graph before the panic can
/// be caught. the ambient slot therefore behaves transactionally.
struct TraceScope {
    previous: Option<GvTrace>,
}

impl Drop for TraceScope {
    fn drop(&mut self) {
        GV_TRACE.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

/// run `f` in a fresh isolated trace, returning the finished untagged kernel
/// and the closure's ordinary result.
///
/// the closure receives a [`TraceCx`] token branded with a lifetime unique to
/// this call; every symbolic value it mints carries that brand, so the value
/// is usable only inside the closure and only with values of the same trace.
/// an enclosing trace, if present, is restored on both normal return and
/// panic, so deliberate nesting is explicit and cannot overwrite the outer
/// graph.
///
/// the resulting kernel is **untagged**, so it stands outside `try_fuse`.
/// callers that want the fusion algebra use [`trace_for_domain`].
///
/// the brand makes escape a compile error:
///
/// ```compile_fail
/// use symbi_ir::{trace, TraceCx};
/// let (_kernel, leaked) = trace(|cx: TraceCx| cx.lit(1.0));
/// ```
///
/// as is smuggling through a captured slot:
///
/// ```compile_fail
/// use symbi_ir::{trace, Gv, TraceCx};
/// let mut slot = None;
/// trace(|cx: TraceCx| slot = Some(cx.lit(1.0)));
/// ```
///
/// and cross-trace arithmetic:
///
/// ```compile_fail
/// use symbi_ir::{trace, TraceCx};
/// trace(|outer: TraceCx| {
///     let a = outer.lit(1.0);
///     trace(move |inner: TraceCx| {
///         let b = inner.lit(2.0);
///         let _ = a + b; // distinct brands
///     });
/// });
/// ```
pub fn trace<R>(f: impl for<'t> FnOnce(TraceCx<'t>) -> R) -> (GvKernel, R) {
    trace_with(LaunchGrade::untagged(), f)
}

/// [`trace`], with the finished kernel tagged with the launch grade of the
/// given `Domain<R>`. only tagged kernels participate in `try_fuse`.
pub fn trace_for_domain<const D: usize, R>(
    d: &Domain<D>,
    f: impl for<'t> FnOnce(TraceCx<'t>) -> R,
) -> (GvKernel, R) {
    trace_with(LaunchGrade::from_domain(d), f)
}

/// the underlying open-run-close. exposed for callers that already hold a
/// [`LaunchGrade`]; most go through [`trace`] or [`trace_for_domain`].
pub fn trace_with<R>(grade: LaunchGrade, f: impl for<'t> FnOnce(TraceCx<'t>) -> R) -> (GvKernel, R) {
    let previous = GV_TRACE.with(|slot| slot.borrow_mut().take());
    let scope = TraceScope { previous };
    GV_TRACE.with(|slot| *slot.borrow_mut() = Some(fresh_trace()));

    let result = f(TraceCx {
        _brand: std::marker::PhantomData,
    });

    let t = GV_TRACE
        .with(|slot| slot.borrow_mut().take())
        .expect("the trace closure ran with the trace slot filled");
    drop(scope);
    let kernel = GvKernel {
        graph: t.graph,
        field_inputs: t.field_inputs,
        scalar_params: t.scalar_params,
        coord_components: t.coord_components,
        grade,
        numerical_policy: NumericalPolicy::StrictIeee,
        tile_spec: None,
        output_support: None,
        node_supports: t.node_supports,
    };
    (kernel, result)
}

/// the capability token for the trace scope opened by [`trace`]. Copy and
/// zero-sized; its only state is the brand lifetime `'t`, which every minted
/// `Gv<'t>` inherits. construction is private to [`trace_with`], so holding a
/// `TraceCx` proves a trace is active.
#[derive(Clone, Copy)]
pub struct TraceCx<'t> {
    _brand: Brand<'t>,
}

impl<'t> TraceCx<'t> {
    /// a literal value awaiting materialization; touches the graph only when
    /// first used in an operation. equivalent to `Gv::from_f64`, spelled on
    /// the token for symmetry with the node-minting constructors.
    pub fn lit(self, v: f64) -> Gv<'t> {
        Gv(GvVal::Lit(v), std::marker::PhantomData)
    }

    /// brand a raw graph node built through [`Self::with_trace`]. the node is
    /// stamped with the active trace, so it participates in carrier arithmetic
    /// exactly like a minted value.
    pub fn gv(self, node: NodeId) -> Gv<'t> {
        Gv::of(node)
    }

    /// a fresh scalar param node named `name`, held out of the ABI manifest
    /// (a bare leaf for unit tests). production inputs use `field` / `scalar`.
    pub fn param(self, name: &str) -> Gv<'t> {
        Gv::of(with_trace(|t| {
            t.graph.add_scalar_param(name, ElementTy::F64)
        }))
    }

    /// a per-cell field read: `key` is the IR-side buffer-load name, `runtime` the
    /// dotted path the dispatch binds the buffer to (e.g., `"cons.den"`). recorded
    /// (deduped) in the kernel ABI manifest — this is the input binding for a
    /// carrier-generic physics fn instantiated at Gv.
    pub fn field(self, key: &str, runtime: impl Into<FieldBind>) -> Gv<'t> {
        let runtime = runtime.into();
        Gv::of(with_trace(|t| {
            assert!(
                !t.scalar_keys.iter().any(|scalar| scalar.as_str() == key),
                "graph ABI name `{key}` cannot be both a field input and scalar parameter",
            );
            let id = t.graph.add_scalar_param(key, ElementTy::F64);
            let key = InputKey::new(key);
            if t.field_keys.insert(key.clone()) {
                t.field_inputs.push((key, runtime));
            }
            id
        }))
    }

    /// a shifted per-cell field read for stencils (PLM reconstruction): the field `key`
    /// loaded at `cell + offset` along `axis`, over an `ndim`-spatial grid. `offset == 0`
    /// is the direct cell read (`Self::field`); nonzero builds the integer coord arithmetic
    /// (`_coord_axis + offset`) + a `LoadAt`, registering the field and the coord axes in
    /// the manifest. codegen-only: a stencil reaches past the pointwise `Scalar` surface,
    /// so it lives on the trace token (the host runtime reads neighbors straight from the
    /// Field buffer; the traced kernel is what needs the explicit `load_at`).
    pub fn field_shifted(
        self,
        key: &str,
        runtime: impl Into<FieldBind>,
        ndim: u8,
        axis: u8,
        offset: i32,
    ) -> Gv<'t> {
        let runtime = runtime.into();
        if offset == 0 {
            return self.field(key, runtime);
        }
        Gv::of(with_trace(|t| {
            assert!(
                !t.scalar_keys.iter().any(|scalar| scalar.as_str() == key),
                "graph ABI name `{key}` cannot be both a field input and scalar parameter",
            );
            // register the field (the LoadAt resolves the buffer by this key, deduped).
            let input_key = InputKey::new(key);
            if t.field_keys.insert(input_key.clone()) {
                t.field_inputs.push((input_key, runtime));
            }
            let comps: Vec<NodeId> = (0..ndim).map(|ax| coord_node(t, ax)).collect();
            let off = t.graph.add_const(ConstValue::I32(offset), None);
            let mut shifted = comps.clone();
            shifted[axis as usize] =
                t.graph
                    .element_wise(ElementWiseOp::Add, vec![comps[axis as usize], off], None);
            t.graph.load_at(Symbol::intern(key), shifted, None)
        }))
    }

    /// read a field at a multi-axis integer offset from the current cell — the
    /// halo-stencil primitive for operators that read diagonals (e.g. the
    /// viscous transverse gradient). `field_shifted` is the single-axis case.
    /// `offsets[ax]` is the per-axis shift; `offsets.len()` must be `ndim`.
    pub fn field_offset(
        self,
        key: &str,
        runtime: impl Into<FieldBind>,
        ndim: u8,
        offsets: &[i32],
    ) -> Gv<'t> {
        assert_eq!(
            offsets.len(),
            ndim as usize,
            "field_offset: one offset per axis"
        );
        let runtime = runtime.into();
        if offsets.iter().all(|&o| o == 0) {
            return self.field(key, runtime);
        }
        Gv::of(with_trace(|t| {
            assert!(
                !t.scalar_keys.iter().any(|scalar| scalar.as_str() == key),
                "graph ABI name `{key}` cannot be both a field input and scalar parameter",
            );
            let input_key = InputKey::new(key);
            if t.field_keys.insert(input_key.clone()) {
                t.field_inputs.push((input_key, runtime));
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
        }))
    }

    /// a scalar kernel param (e.g., `gamma`), recorded (deduped) in the manifest signature.
    pub fn scalar(self, name: &str) -> Gv<'t> {
        Gv::of(with_trace(|t| {
            assert!(
                !t.field_keys.iter().any(|field| field.as_str() == name),
                "graph ABI name `{name}` cannot be both a scalar parameter and field input",
            );
            let id = t.graph.add_scalar_param(name, ElementTy::F64);
            let name = ScalarParam::new(name);
            if t.scalar_keys.insert(name.clone()) {
                t.scalar_params.push(name);
            }
            id
        }))
    }

    /// the cell coordinate along spatial `axis` as a Gv value — the index->physical bridge for
    /// in-kernel geometry. it is the integer `_coord_N` (recorded in the coord manifest, like
    /// `field_shifted`); arithmetic against the f64 grid scalars (`x_lo_d`, `dx_d`) auto-promotes
    /// at lowering (the IR's usual arithmetic conversions), so positions, scale factors, and cell
    /// widths trace as pure Gv expressions. this is what lets the substrate geometry (curvilinear
    /// metric, cell areas/volumes) be a Gv trace built directly from the cell index.
    pub fn coord(self, axis: u8) -> Gv<'t> {
        Gv::of(with_trace(|t| coord_node(t, axis)))
    }

    /// run a closure with mutable access to the active trace — the raw-IR door
    /// for discretization builders (integer index arithmetic, select, load_at).
    pub fn with_trace<R>(self, f: impl FnOnce(&mut GvTrace) -> R) -> R {
        with_trace(f)
    }

    /// tag a traced value with a support ball: the builder asserts the value is
    /// exactly zero (f64) outside |x - center| > radius for every field input —
    /// the saturation lemma, stated where the mask that makes it true is built.
    /// consumed by `GvKernel::with_derived_support`, which propagates tags to the
    /// write roots; validated downstream by the compiled-kernel support sampler.
    pub fn tag_support_ball(
        self,
        v: &Gv<'t>,
        center: Vec<crate::support::ParamExpr>,
        radius: crate::support::ParamExpr,
    ) {
        let node = v.node();
        with_trace(|t| {
            t.node_supports.insert(
                node,
                crate::support_infer::SupportBall { center, radius },
            );
        });
    }
}

impl GvKernel {
    /// Explicit compiler view of the graph representation. Physics builders
    /// should use the carrier API; lowering, validation, and backends use this.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn field_inputs(&self) -> &[(InputKey, FieldBind)] {
        &self.field_inputs
    }
    pub fn scalar_params(&self) -> &[ScalarParam] {
        &self.scalar_params
    }
    pub fn coord_components(&self) -> &[u8] {
        &self.coord_components
    }
    pub fn grade(&self) -> &LaunchGrade {
        &self.grade
    }
    pub fn numerical_policy(&self) -> NumericalPolicy {
        self.numerical_policy
    }
    pub fn tile_spec(&self) -> Option<&TileSpec> {
        self.tile_spec.as_ref()
    }
    pub fn output_support(&self) -> Option<&crate::support::Support> {
        self.output_support.as_ref()
    }
    #[cfg(test)]
    pub(crate) fn node_supports(&self) -> &HashMap<NodeId, crate::support_infer::SupportBall> {
        &self.node_supports
    }

    pub fn with_numerical_policy(mut self, policy: NumericalPolicy) -> Self {
        self.numerical_policy = policy;
        self
    }
    /// infer this kernel's shared-memory tile intent. an explicit
    /// `tile_spec` overrides; otherwise a stencil kernel (non-empty
    /// `coord_components` — it does shifted `load_at` reads, so it has a halo and
    /// reusable neighbor data) gets a `TileSpec`, and a pointwise kernel (empty)
    /// gets `None` (its reads are single-use, so gmem serves).
    ///
    /// the CPU emit cache-tiles every kernel regardless (a runtime knob); this
    /// drives the GPU smem path alone.
    ///
    /// **policy:** opt-in. tiling applies to kernels that declare a `tile_spec`
    /// via `with_tile_spec` — the builder holds the stencil structure (which
    /// axis reconstructs, how wide) and declares the matching per-axis slab. a
    /// blanket halo-2 cube over-loads smem ~7.5x for a reconstruction along one
    /// direction and reads transverse ghost cells outside the physics stencil,
    /// which makes the explicit declaration the safe default. deriving the
    /// per-axis slab from the graph's `LoadAt` offsets is future work.
    pub fn infer_tile_spec(&self) -> Option<TileSpec> {
        self.tile_spec.clone()
    }

    /// the field-input keys this kernel reads at a shifted coord — the
    /// `Op::LoadAt` field symbols, in first-seen order, restricted to keys that
    /// are actual `field_inputs` (a LoadAt always names a registered field, so
    /// the restriction is a safety net). this is the smem-tile
    /// candidate set: stencil-read fields carry reusable neighbor data worth
    /// prefetching; a field read pointwise stays on gmem.
    pub fn stencil_read_field_keys(&self) -> Vec<InputKey> {
        let manifest: std::collections::HashSet<&str> =
            self.field_inputs.iter().map(|(k, _)| k.as_str()).collect();
        let mut seen = std::collections::HashSet::new();
        let mut keys = Vec::new();
        for (_, node, _) in self.graph.iter() {
            if let crate::graph::Op::LoadAt(sym, _) = &node.op {
                let k = sym.as_str();
                if manifest.contains(k) && seen.insert(InputKey::new(k)) {
                    keys.push(InputKey::new(k));
                }
            }
        }
        keys
    }

    /// the identity element for the fusion monoid at the given grade. an
    /// empty graph with empty writes and reads. `try_fuse(noop(g), k) == k` for any
    /// tagged `k` with `k.grade == g` (identity law).
    pub fn noop(grade: LaunchGrade) -> (GvKernel, KernelWrites) {
        (
            GvKernel {
                graph: Graph::new(),
                field_inputs: Vec::new(),
                scalar_params: Vec::new(),
                coord_components: Vec::new(),
                grade,
                numerical_policy: NumericalPolicy::StrictIeee,
                tile_spec: None,
                output_support: None,
                node_supports: std::collections::HashMap::new(),
            },
            Vec::new(),
        )
    }

    /// builder-style: attach a smem tile spec. validates that every tiled key
    /// is present in this kernel's `field_inputs` manifest — the dispatch binds
    /// a buffer to a tiled slot through its manifest entry. panics on mismatch
    /// (the caller is the GvKernel builder, which owns the manifest, so a
    /// mismatch is a build-side bug).
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
    /// expressions over this kernel's scalar params; every referenced param is
    /// in the manifest, which is what lets dispatch evaluate the geometry.
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
    /// mask chain or an untagged trace derives Everywhere — fail-safe wide, so
    /// the declared region always contains the true support. ball params are
    /// manifest-validated exactly as a hand declaration would be.
    pub fn with_derived_support(self, writes: &[KernelWrite]) -> Self {
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
/// - grade equality: both `index_space` match, and both are tagged. fusing
///   different domains would launch at least one half over the wrong index
///   set.
///
/// - disjoint writes: each runtime_path appears in at most one writes list.
///   shared writes would race.
///
/// - independence: each runtime_path one kernel writes is absent from the
///   other's reads. (a kernel reading its own writes is intra-kernel and
///   stays fine.) inter-dep violates the sequential
///    semantics: in the two-launch baseline the second launch sees the first
///    launch's updates globally; in a fused launch a stencil read in the
///    second body would see the first body's writes at one thread index
///    alone.
///
/// returns `Ok((fused_kernel, fused_writes))` on success. the fused writes
/// preserve the original ordering: all of a's writes (with NodeIds unchanged,
/// since the fused graph starts as a clone of a's graph), then all of b's
/// writes with NodeIds remapped through the splice.
pub fn try_fuse(
    a: GvKernel,
    a_writes: KernelWrites,
    b: GvKernel,
    b_writes: KernelWrites,
) -> Result<(GvKernel, KernelWrites), FusionError> {
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
    if a.numerical_policy != b.numerical_policy {
        return Err(FusionError::NumericalPolicyMismatch {
            a: a.numerical_policy,
            b: b.numerical_policy,
        });
    }

    // law: tile-spec equality. a fused launch carries one block layout + smem
    // prelude, so both halves declare the same spec: identical halos and
    // identical tiled-field sets, tiled or untiled together. callers align
    // before fusing.
    if a.tile_spec != b.tile_spec {
        return Err(FusionError::TileSpecMismatch {
            a: a.tile_spec.clone(),
            b: b.tile_spec.clone(),
        });
    }

    // law: disjoint writes. compare semantic field identities; render a name only for diagnostics.
    let a_write_paths: HashSet<FieldBind> = a_writes
        .iter()
        .map(|write| write.destination.clone())
        .collect();
    let b_write_paths: HashSet<FieldBind> = b_writes
        .iter()
        .map(|write| write.destination.clone())
        .collect();
    if let Some(p) = a_write_paths.intersection(&b_write_paths).next() {
        return Err(FusionError::WriteConflict {
            runtime_path: p.name(),
        });
    }

    // independence: a field one kernel writes and the other reads is a pipeline hazard, where the
    // reader wants the writer's fresh output. the exception is a field the writer holds in place
    // (it both reads and writes it): that field is a shared pre-state — both kernels read the same
    // old value as a leaf, and the fused dataflow writes the new value as a root (the read is
    // pointwise, so the in-place write is hazard-free). the curvilinear gas geometric source reads
    // cell-B (bc_k) for the magnetic pressure while the cell-B predictor flux-evolves that same
    // bc_k in place, both off the timestep's pre-state; excluding the writer's in-place fields
    // lets that pair fuse safely. fusion joins same-stage kernels, so the reader always wants the
    // pre-state. the disjoint-writes law above still holds, keeping every fused output
    // single-writer. both sides compare the born-typed `FieldBind`, so aliases classified to the
    // same `FieldRef` cannot evade dependency detection.
    let a_input_paths: HashSet<FieldBind> = a.field_inputs.iter().map(|(_, p)| p.clone()).collect();
    let b_input_paths: HashSet<FieldBind> = b.field_inputs.iter().map(|(_, p)| p.clone()).collect();
    let a_inplace: HashSet<&FieldBind> = a_write_paths.intersection(&a_input_paths).collect();
    let b_inplace: HashSet<&FieldBind> = b_write_paths.intersection(&b_input_paths).collect();
    if let Some(p) = a_write_paths
        .intersection(&b_input_paths)
        .find(|p| !a_inplace.contains(p))
    {
        return Err(FusionError::InterDep {
            written: p.name(),
            read: p.name(),
        });
    }
    if let Some(p) = b_write_paths
        .intersection(&a_input_paths)
        .find(|p| !b_inplace.contains(p))
    {
        return Err(FusionError::InterDep {
            written: p.name(),
            read: p.name(),
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
    let b_roots: Vec<NodeId> = b_writes.iter().map(|write| write.value).collect();
    let remapped_b_roots = splice_graph(&mut target, &b.graph, &b_roots, &subst)
        .map_err(|e| FusionError::Splice(format!("{}", e)))?;

    // manifest merge: deterministic first-seen order, a first, b appended.
    let mut field_inputs = a.field_inputs.clone();
    let a_field_keys: HashSet<InputKey> = a.field_inputs.iter().map(|(k, _)| k.clone()).collect();
    for (k, p) in b.field_inputs.iter() {
        if !a_field_keys.contains(k) {
            field_inputs.push((k.clone(), p.clone()));
        }
    }
    let mut scalar_params = a.scalar_params.clone();
    let a_scalar_set: HashSet<ScalarParam> = a.scalar_params.iter().cloned().collect();
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

    // the fused outputs' support is the union of both sides'. representable
    // when both declare the same region; any mismatch (or an undeclared side)
    // widens to Everywhere, which is sound for any pair of declarations.
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
        numerical_policy: a.numerical_policy,
        // tile_spec is preserved across fusion: the pre-check guaranteed
        // a.tile_spec == b.tile_spec, so either side's value serves.
        tile_spec: a.tile_spec,
        output_support,
        // node tags stay at build time: derivation runs before fusing, and the
        // fused kernel carries the declared union above.
        node_supports: HashMap::new(),
    };

    // writes: a's preserved verbatim (target started as a's graph clone),
    // b's remapped to the spliced NodeIds.
    let mut fused_writes: KernelWrites = a_writes;
    for (write, new_root) in b_writes.into_iter().zip(remapped_b_roots) {
        fused_writes.push(KernelWrite {
            value: new_root,
            ..write
        });
    }

    Ok((fused, fused_writes))
}

/// intern (or reuse) the synthetic `_coord_N` I32 param for spatial axis `ax`, recording
/// the axis in the manifest. a cell coordinate
/// is an i32, so stencil shifts (`_coord + off`) and buffer indices are pure integer
/// arithmetic, computed in the graph's I32 domain beside the f64 Gv carrier.
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

/// run a closure with mutable access to the active trace. crate-internal; the
/// contributor-facing door is `TraceCx::with_trace`, whose brand proves a
/// trace is open.
pub(crate) fn with_trace<R>(f: impl FnOnce(&mut GvTrace) -> R) -> R {
    GV_TRACE.with(|t| {
        let mut b = t.borrow_mut();
        f(b.as_mut()
            .expect("Gv op outside an active trace — enter one via trace(|cx| ...)"))
    })
}

// the builder-facing trace API: the discretization layer (symbi-discretize) constructs
// raw index/stencil IR through these, so the trace's fields and the graph stay private.
impl GvTrace {
    /// raw graph access for builders constructing IR nodes directly (integer index
    /// arithmetic, select, load_at): addressing lives in the graph's I32 domain,
    /// value arithmetic in the f64 `Gv` carrier.
    pub fn graph(&mut self) -> &mut Graph {
        &mut self.graph
    }

    /// register a per-cell field input in the ABI manifest (deduped, first-seen order).
    /// the buffer is bound by `key` at load time; `runtime` is its dotted dispatch path.
    pub fn register_field(&mut self, key: &str, runtime: impl Into<FieldBind>) {
        let runtime = runtime.into();
        let key = InputKey::new(key);
        assert!(
            !self
                .scalar_keys
                .iter()
                .any(|scalar| scalar.as_str() == key.as_str()),
            "graph ABI name `{key}` cannot be both a field input and scalar parameter",
        );
        if self.field_keys.insert(key.clone()) {
            self.field_inputs.push((key, runtime));
        }
    }

    /// a deduped I32 scalar param (an integer index / lattice-map arg), returning its node —
    /// the integer analog of `Gv::scalar` (index math).
    pub fn scalar_int(&mut self, name: &str) -> NodeId {
        assert!(
            !self.field_keys.iter().any(|field| field.as_str() == name),
            "graph ABI name `{name}` cannot be both a scalar parameter and field input",
        );
        let id = self.graph.add_scalar_param(name, ElementTy::I32);
        let name = ScalarParam::new(name);
        if self.scalar_keys.insert(name.clone()) {
            self.scalar_params.push(name);
        }
        id
    }

    /// the synthetic `_coord_N` I32 param for spatial axis `ax` (deduped, recorded in the
    /// coord manifest) — stencil shifts are pure integer arithmetic on this.
    pub fn coord(&mut self, ax: u8) -> NodeId {
        coord_node(self, ax)
    }
}

/// a graph value: either a traced node stamped with its trace generation, or
/// a literal awaiting materialization.
#[derive(Clone, Copy, Debug)]
enum GvVal {
    Node(NodeId, u64),
    Lit(f64),
}

/// the tracing scalar carrier. `Copy` (a NodeId or a literal); every operation
/// records a node into the active trace graph. the `'t` brand ties the value
/// to the [`trace`] scope whose `TraceCx` minted it: escaping the scope or
/// combining values across scopes fails to compile.
///
/// a traced graph value carries no physical order or equality: `Scalar` leaves
/// `PartialOrd`/`PartialEq` out of its bounds, and `Gv` leaves them unimplemented.
/// physics decides with the traceable `cmp_lt` / `cmp_gt` / `select`, which record a
/// comparison node; native `<` / `==` would compare node indices, so the type system
/// rejects them at compile time:
///
/// ```compile_fail
/// use symbi_ir::{trace, TraceCx};
/// trace(|cx: TraceCx| {
///     let a = cx.param("a");
///     let b = cx.param("b");
///     let _ = a < b; // no `PartialOrd` for `Gv` — must use `cmp_lt`
/// });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Gv<'t>(GvVal, Brand<'t>);

impl<'t> Gv<'t> {
    /// the NodeId this value resolves to, materializing a literal to a `Const`
    /// node on demand (a literal touches the graph only here, on first use). pub so a gv
    /// builder in symbi-discretize can extract its write roots.
    ///
    /// a node handle carries the generation stamp of the trace that minted it;
    /// resolving it inside a different (nested) trace is a cross-graph use and
    /// panics here, at the recording site.
    pub fn node(self) -> NodeId {
        match self.0 {
            GvVal::Node(n, stamp) => {
                let active = with_trace(|t| t.id);
                assert!(
                    stamp == active,
                    "Gv value belongs to a different trace — a value minted in an enclosing \
                     trace cannot be used inside a nested one"
                );
                n
            }
            GvVal::Lit(v) => with_trace(|t| t.graph.add_const(ConstValue::F64(v), None)),
        }
    }

    /// wrap a node of the active trace, stamping it with the trace generation.
    /// crate-internal; builders go through `TraceCx::gv`.
    #[inline]
    pub(crate) fn of(n: NodeId) -> Gv<'t> {
        let stamp = with_trace(|t| t.id);
        Gv(GvVal::Node(n, stamp), std::marker::PhantomData)
    }

    #[inline]
    const fn lit(v: f64) -> Gv<'t> {
        Gv(GvVal::Lit(v), std::marker::PhantomData)
    }

    #[inline]
    fn binop(self, rhs: Gv<'t>, op: ElementWiseOp) -> Gv<'t> {
        let a = self.node();
        let b = rhs.node();
        Gv::of(with_trace(|t| t.graph.element_wise(op, vec![a, b], None)))
    }

    #[inline]
    fn unop(self, op: ElementWiseOp) -> Gv<'t> {
        let x = self.node();
        Gv::of(with_trace(|t| t.graph.element_wise(op, vec![x], None)))
    }
}

// ---- std::ops: record element-wise nodes ----
impl<'t> Add for Gv<'t> {
    type Output = Gv<'t>;
    fn add(self, r: Gv<'t>) -> Gv<'t> {
        self.binop(r, ElementWiseOp::Add)
    }
}
impl<'t> Sub for Gv<'t> {
    type Output = Gv<'t>;
    fn sub(self, r: Gv<'t>) -> Gv<'t> {
        self.binop(r, ElementWiseOp::Sub)
    }
}
impl<'t> Mul for Gv<'t> {
    type Output = Gv<'t>;
    fn mul(self, r: Gv<'t>) -> Gv<'t> {
        self.binop(r, ElementWiseOp::Mul)
    }
}
impl<'t> Div for Gv<'t> {
    type Output = Gv<'t>;
    fn div(self, r: Gv<'t>) -> Gv<'t> {
        self.binop(r, ElementWiseOp::Div)
    }
}
impl<'t> Neg for Gv<'t> {
    type Output = Gv<'t>;
    fn neg(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Neg)
    }
}
impl<'t> AddAssign for Gv<'t> {
    fn add_assign(&mut self, r: Gv<'t>) {
        *self = *self + r;
    }
}
impl<'t> SubAssign for Gv<'t> {
    fn sub_assign(&mut self, r: Gv<'t>) {
        *self = *self - r;
    }
}
impl<'t> MulAssign for Gv<'t> {
    fn mul_assign(&mut self, r: Gv<'t>) {
        *self = *self * r;
    }
}
impl<'t> DivAssign for Gv<'t> {
    fn div_assign(&mut self, r: Gv<'t>) {
        *self = *self / r;
    }
}

impl<'t> std::iter::Sum for Gv<'t> {
    fn sum<I: Iterator<Item = Gv<'t>>>(iter: I) -> Gv<'t> {
        // direct construction; zero comes from `<Gv as crate::algebra::Scalar>::ZERO`
        // but qualifying inline keeps this independent of import scope.
        iter.fold(Gv::lit(0.0), |a, b| a + b)
    }
}

impl<'t> Default for Gv<'t> {
    // direct construction (matches `<Gv as crate::algebra::Scalar>::ZERO`); kept as
    // a direct expression to stay independent of the trait import scope.
    fn default() -> Gv<'t> {
        Gv::lit(0.0)
    }
}

impl<'t> std::fmt::Display for Gv<'t> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

// Gv is a Copy fixed-size handle whose lifetime is the build-time trace; it lives in the
// trace alone, so the `Field<Gv>` buffer-layout contract holds vacuously.
unsafe impl<'t> FieldElement for Gv<'t> {
    type Scalar = Gv<'t>;
}

// structural numeric impl — satisfies the in-crate `Tensor` / `Matrix` / `Indexed`
// method bounds (`Tensor::dot`, `Tensor::norm`, etc.). delegates to the trace-recording
// ops the production `Scalar` impl below uses. `symbi_algebra::Numeric` is the minimal
// subset that breaks the `symbi-algebra` <-> `symbi-ir` dep cycle for `Tensor`'s
// scalar-bounded methods; the production carrier-generic surface is `Scalar`.
impl<'t> symbi_algebra::algebra::Numeric for Gv<'t> {
    const ZERO: Self = Gv::lit(0.0);
    const ONE: Self = Gv::lit(1.0);
    #[inline]
    fn from_f64(v: f64) -> Self {
        Gv::lit(v)
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
// Mask discipline: `GvMask` is a newtype around `Gv` that wraps a Bool-typed
// graph node. construction is `pub(crate)`, so `Scalar::cmp_*` is the sole
// producer of masks — the type system enforces "masks at the graph layer are
// always Bool-typed." `BitAnd` / `BitOr` / `Not` emit the corresponding graph Bool
// ops (`ElementWiseOp::BitAnd` / `BitOr` / `BitNot`).
//
// this is the single carrier surface workspace-wide.
// =============================================================================

/// type-safe Mask wrapper for Gv. wraps a Gv carrying a Bool-typed graph
/// node. `pub(crate)` constructor — `Scalar::cmp_*` is the sole producer.
#[derive(Copy, Clone, Debug)]
pub struct GvMask<'t>(pub(crate) Gv<'t>);

impl<'t> std::ops::BitAnd for GvMask<'t> {
    type Output = GvMask<'t>;
    #[inline]
    fn bitand(self, rhs: GvMask<'t>) -> GvMask<'t> {
        let a = self.0.node();
        let b = rhs.0.node();
        GvMask(Gv::of(with_trace(|t| {
            t.graph
                .element_wise(ElementWiseOp::BitAnd, vec![a, b], None)
        })))
    }
}

impl<'t> std::ops::BitOr for GvMask<'t> {
    type Output = GvMask<'t>;
    #[inline]
    fn bitor(self, rhs: GvMask<'t>) -> GvMask<'t> {
        let a = self.0.node();
        let b = rhs.0.node();
        GvMask(Gv::of(with_trace(|t| {
            t.graph.element_wise(ElementWiseOp::BitOr, vec![a, b], None)
        })))
    }
}

impl<'t> std::ops::Not for GvMask<'t> {
    type Output = GvMask<'t>;
    #[inline]
    fn not(self) -> GvMask<'t> {
        let a = self.0.node();
        GvMask(Gv::of(with_trace(|t| {
            t.graph.element_wise(ElementWiseOp::BitNot, vec![a], None)
        })))
    }
}

impl<'t> crate::algebra::Mask for GvMask<'t> {}

impl<'t> crate::algebra::Scalar for Gv<'t> {
    type Mask = GvMask<'t>;

    // zero / one inherited from `Numeric for Gv`.
    const INFINITY: Gv<'t> = Gv::lit(f64::INFINITY);
    const NEG_INFINITY: Gv<'t> = Gv::lit(f64::NEG_INFINITY);
    const NAN: Gv<'t> = Gv::lit(f64::NAN);
    const HALF: Gv<'t> = Gv::lit(0.5);
    const TWO: Gv<'t> = Gv::lit(2.0);
    const THREE: Gv<'t> = Gv::lit(3.0);
    const FOUR: Gv<'t> = Gv::lit(4.0);

    // from_f64 inherited from `Numeric for Gv`.

    fn to_f64(self) -> f64 {
        match self.0 {
            GvVal::Lit(v) => v,
            // host-boundary escape — see `crate::algebra::Scalar::to_f64` doc.
            // a `Gv` on a traced node is a graph handle; extracting a concrete
            // value inside carrier-generic physics breaks the homomorphism.
            GvVal::Node(..) => panic!(
                "Gv::to_f64 on a traced node — carrier-generic physics must decide with \
                 cmp_*/select, not extract a concrete value"
            ),
        }
    }

    // ── comparisons return GvMask — the Mask discipline ──────────
    fn cmp_lt(self, o: Gv<'t>) -> GvMask<'t> {
        GvMask(self.binop(o, ElementWiseOp::Lt))
    }
    fn cmp_le(self, o: Gv<'t>) -> GvMask<'t> {
        GvMask(self.binop(o, ElementWiseOp::Le))
    }
    fn cmp_gt(self, o: Gv<'t>) -> GvMask<'t> {
        GvMask(self.binop(o, ElementWiseOp::Gt))
    }
    fn cmp_ge(self, o: Gv<'t>) -> GvMask<'t> {
        GvMask(self.binop(o, ElementWiseOp::Ge))
    }
    fn cmp_eq(self, o: Gv<'t>) -> GvMask<'t> {
        GvMask(self.binop(o, ElementWiseOp::Eq))
    }

    fn select(m: GvMask<'t>, yes: Gv<'t>, no: Gv<'t>) -> Gv<'t> {
        let c = m.0.node();
        let y = yes.node();
        let n = no.node();
        Gv::of(with_trace(|t| t.graph.select(c, y, n, None)))
    }

    // scope frame at S = Gv.
    //
    // semantics: snapshot the graph's node count before running `body`; all
    // NodeIds pushed during the closure (the lexical region of this scope)
    // become the Op::Scope's body list — including any nested Op::Scope
    // nodes the closure itself emits. the snapshot..end range captures the
    // fresh nodes; hash-consing that resolves an expression to a pre-
    // existing NodeId contributes nothing (its NodeId lies outside the
    // range).
    //
    // the resulting Op::Scope bypasses hash-cons (see Graph::push) so two
    // structurally identical scopes from distinct call sites stay distinct.
    //
    // empty body short-circuit: when the closure produces no new nodes (e.g.
    // the result is a pre-existing constant or param), the body is empty and
    // the result returns directly, preserving the identity law for trivial
    // closures.
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

    // the dual of `iterate` for branches: a lazy conditional at S = Gv.
    //
    // semantics: snapshot the graph before each arm's closure; the NodeIds
    // pushed during the closure are that arm's body (the `mark..end` range —
    // same convention as `scope`). emit an Op::IfElse carrying the cond, both
    // arm bodies, and each arm's result. scalarize lowers the arm bodies
    // inside their respective `if`/`else` brace so the taken arm alone executes
    // — the carrier-portable form of an early-out conditional.
    //
    // shared upstream values (the cond, any pre-branch subexpression) are
    // created before the closures, so they fall outside both ranges and stay in
    // the outer body (computed once). cross-arm / leaks-outside hash-cons
    // sharing is resolved by scalarize's eviction pass, exactly as for Scope.
    fn cond(m: GvMask<'t>, t: impl FnOnce() -> Gv<'t>, f: impl FnOnce() -> Gv<'t>) -> Gv<'t> {
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
    // the shared arm computation is traced once (each closure runs once); each
    // returned Gv is a projection of the same branch. this is what lets a
    // multi-output fast-path (e.g., the (sl, sr) wave-speed Eq.57/58/quartic
    // selection) skip the whole quartic on the fast path. mirrors `cond` per
    // arm, adding the N-element result vectors + the projections.
    fn cond_vec<const N: usize>(
        m: GvMask<'t>,
        t: impl FnOnce() -> [Gv<'t>; N],
        f: impl FnOnce() -> [Gv<'t>; N],
    ) -> [Gv<'t>; N] {
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
    fn recip(self) -> Gv<'t> {
        // 1 / self — use the const directly to avoid trait-method ambiguity.
        let one = Gv::lit(1.0);
        one / self
    }

    // ── transcendentals (one graph tag: ElementWise) ───
    fn sin(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Sin)
    }
    fn cos(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Cos)
    }
    fn tan(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Tan)
    }
    fn asin(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Asin)
    }
    fn acos(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Acos)
    }
    fn atan(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Atan)
    }
    fn atan2(self, o: Gv<'t>) -> Gv<'t> {
        let (y, x) = (self.node(), o.node());
        Gv::of(with_trace(|t| {
            t.graph.element_wise(ElementWiseOp::Atan2, vec![y, x], None)
        }))
    }
    fn exp(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Exp)
    }
    fn ln(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Log)
    }
    fn log10(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Log10)
    }

    fn powi(self, n: i32) -> Gv<'t> {
        // lower to repeated multiplication (exponentiation by squaring):
        // `f64::powi` raises a negative base exactly (e.g., (-2)^2 = 4) while CUDA
        // `powf(neg, 2.0)` is NaN, so a `powf` lowering would break carrier equivalence
        // (the f64 host and the Gv kernel would disagree). n is a small integer constant
        // at trace time, so the multiply chain unrolls into the DAG and the kernel keeps
        // to plain multiplies.
        if n == 0 {
            return Gv::lit(1.0);
        }
        let mut base = self;
        let mut exp = n.unsigned_abs();
        let mut acc: Option<Gv<'t>> = None;
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
            Gv::lit(1.0) / pos
        } else {
            pos
        }
    }
    fn powf(self, e: Gv<'t>) -> Gv<'t> {
        self.binop(e, ElementWiseOp::Pow)
    }

    fn floor(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Floor)
    }
    fn ceil(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Ceil)
    }

    // ── hyperbolics — graph-op lowerings ────────────────
    fn sinh(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Sinh)
    }
    fn cosh(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Cosh)
    }
    fn tanh(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Tanh)
    }
    fn asinh(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Asinh)
    }
    fn acosh(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Acosh)
    }
    fn atanh(self) -> Gv<'t> {
        self.unop(ElementWiseOp::Atanh)
    }

    // ── higher-order: iterate + iterate_vec with the freeze law ───────────
    fn iterate(
        self,
        max_steps: usize,
        body: impl Fn(Self) -> Self,
        converged: impl Fn(Self, Self) -> GvMask<'t>,
    ) -> Gv<'t> {
        // carrier equivalence: the traced kernel returns the value the host loop
        // returns, for every input. freeze on convergence — see the freeze law
        // in `crate::algebra::Scalar::iterate` doc. `conv` also rides as the IR's
        // `break_when`, so the loop exits once convergence fires and skips the
        // remaining `max_steps` worth of dead body (the freeze nulls the writes
        // while the cone's arithmetic would otherwise run every iteration — the
        // RMHD c2p cost).
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
        converged: impl Fn([Self; N], [Self; N]) -> GvMask<'t>,
        result: usize,
    ) -> Gv<'t> {
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
// the tests are synthetic (purely structural): small traces with two
// independent fields, writes declared manually, asserting the algebraic
// properties hold structurally. these properties are what let consumer code (godunov +
// bcell_godunov, snapshot + bcell_snapshot, c2p + wave_speed_map) fuse via
// `try_fuse` soundly.
// =============================================================================

#[cfg(test)]
mod trace_scope_laws {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn scoped_trace_restores_outer_trace_after_panic() {
        let (kernel, root) = trace(|cx: TraceCx| {
            let outer = cx.param("outer");

            let panic = catch_unwind(AssertUnwindSafe(|| {
                let _ = trace(|inner: TraceCx| {
                    let v = inner.param("inner");
                    let _ = v + inner.lit(1.0);
                    panic!("deliberate inner-trace failure");
                });
            }));
            assert!(panic.is_err());

            // an operation after the caught panic must append to the restored
            // outer graph, not to the discarded inner graph or an empty slot.
            (outer + cx.lit(2.0)).node()
        });
        assert!(kernel.graph.param(&Symbol::intern("outer")).is_some());
        assert!(kernel.graph.param(&Symbol::intern("inner")).is_none());
        assert!((root.0 as usize) < kernel.graph.len());
    }

    #[test]
    fn nested_trace_is_isolated_from_outer_graph() {
        let (outer_kernel, inner_kernel) = trace(|cx: TraceCx| {
            let _outer = cx.param("outer");
            let (inner_kernel, ()) = trace(|inner: TraceCx| {
                let _ = inner.param("inner");
            });
            inner_kernel
        });
        assert!(outer_kernel.graph.param(&Symbol::intern("outer")).is_some());
        assert!(outer_kernel.graph.param(&Symbol::intern("inner")).is_none());
        assert!(inner_kernel.graph.param(&Symbol::intern("inner")).is_some());
        assert!(inner_kernel.graph.param(&Symbol::intern("outer")).is_none());
    }

    #[test]
    fn cross_trace_value_use_panics_at_recording() {
        let _ = trace(|cx: TraceCx| {
            let outer = cx.param("outer");
            let panicked = catch_unwind(AssertUnwindSafe(|| {
                let _ = trace(move |_inner: TraceCx| {
                    // `outer + outer` shares one brand, so it compiles; the
                    // generation stamp catches the foreign graph at runtime.
                    let _ = outer + outer;
                });
            }));
            assert!(
                panicked.is_err(),
                "an enclosing-trace value recorded inside a nested trace must panic"
            );
        });
    }
}

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

    #[test]
    fn numerical_policy_is_a_permission_lattice() {
        assert!(NumericalPolicy::StrictIeee.permits(RewriteClass::BitExact));
        assert!(!NumericalPolicy::StrictIeee.permits(RewriteClass::FiniteOnly));
        assert!(NumericalPolicy::FiniteOnly.permits(RewriteClass::FiniteOnly));
        assert!(!NumericalPolicy::FiniteOnly.permits(RewriteClass::FastMath));
        assert!(NumericalPolicy::FastMath.permits(RewriteClass::FastMath));
    }

    #[test]
    fn fusion_rejects_mixed_numerical_theories() {
        let grade = interior_grade();
        let (a, aw) = GvKernel::noop(grade.clone());
        let (b, bw) = GvKernel::noop(grade);
        let b = b.with_numerical_policy(NumericalPolicy::FiniteOnly);
        assert!(matches!(
            try_fuse(a, aw, b, bw),
            Err(FusionError::NumericalPolicyMismatch { .. })
        ));
    }

    // build a single-output kernel that reads `in_key`@`in_path`, doubles it,
    // declares its root as `out_key`@`out_path`. used as a uniform fixture.
    fn doubler(
        in_key: &str,
        in_path: &str,
        out_key: &str,
        out_path: &str,
        grade: LaunchGrade,
    ) -> (GvKernel, KernelWrites) {
        let (kern, root) = trace_with(grade, |cx: TraceCx| {
            let x = cx.field(in_key, in_path);
            let y = x * cx.lit(2.0);
            y.node()
        });
        let writes = vec![KernelWrite::new(out_key, out_path, root)];
        (kern, writes)
    }

    // structural-equality helper: two manifests are "the same" iff their
    // field-input sets, scalar-param sets, and write-path sets coincide. node
    // counts may differ across orderings (CSE inside splice can collapse
    // duplicates), so the comparison is over semantic sets.
    fn manifest_sets(
        k: &GvKernel,
        w: &KernelWrites,
    ) -> (
        HashSet<(InputKey, String)>,
        HashSet<ScalarParam>,
        HashSet<String>,
    ) {
        let inputs: HashSet<_> = k
            .field_inputs
            .iter()
            .map(|(k, b)| (k.clone(), b.name()))
            .collect();
        let scalars: HashSet<_> = k.scalar_params.iter().cloned().collect();
        let writes: HashSet<_> = w.iter().map(|write| write.destination.name()).collect();
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
    // first-seen ordering of the input list differs while the set is invariant.
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
        let a_roots: Vec<NodeId> = aw.iter().map(|write| write.value).collect();

        let (fused, fused_w) = try_fuse(a, aw, b, bw).expect("fuse(a, b)");

        // a's writes preserved verbatim at the head of the writes manifest.
        for (i, &a_root) in a_roots.iter().enumerate() {
            assert_eq!(
                fused_w[i].value, a_root,
                "a-write root #{i} was renumbered; equivalence broken"
            );
        }

        // every fused write root is a valid node in the fused graph.
        let n_nodes = fused.graph.len();
        for write in &fused_w {
            assert!(
                (write.value.0 as usize) < n_nodes,
                "fused write {}@{} has dangling NodeId {:?}",
                write.key,
                write.destination.name(),
                write.value
            );
        }
    }

    // grade rejection. fusion admits equal grades; a mismatched pair is rejected.
    #[test]
    fn law_grade_rejection() {
        let (a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", interior_grade());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", edge_grade());

        match try_fuse(a, aw, b, bw) {
            Err(FusionError::GradeMismatch { .. }) => {}
            other => panic!("expected GradeMismatch, got {other:?}"),
        }
    }

    // an untagged kernel stands outside fusion, even against another untagged
    // kernel. tagging is opt-in — the algebra requires an explicit grade.
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
    // neighbor indices, which would demand a grid-wide barrier. reject.
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
    // values were constructed by separate calls — the algebra compares by
    // structure, while DomainId is identity-based and differs across
    // constructions.
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

    // ----- tile-spec laws -----

    /// fusion preserves the tile_spec when both sides match. directly sets
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
            tiled_field_keys: vec!["shared".into()],
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

    /// fuse(tiled, untiled) fails with TileSpecMismatch — a launch carries one
    /// smem layout, so both halves declare the same one.
    #[test]
    fn law_tile_spec_mismatch_rejected() {
        let g = interior_grade();
        let (mut a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        a.tile_spec = Some(TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["a_in".into()],
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

    /// two tiled kernels with different halo widths also fail — a single smem
    /// prelude carries one halo.
    #[test]
    fn law_tile_spec_different_halo_rejected() {
        let g = interior_grade();
        let (mut a, aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g.clone());
        let (mut b, bw) = doubler("b_in", "p.b_in", "b_out", "p.b_out", g);
        a.tile_spec = Some(TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["a_in".into()],
        });
        b.tile_spec = Some(TileSpec {
            halo: vec![3],
            tiled_field_keys: vec!["b_in".into()],
        });
        match try_fuse(a, aw, b, bw) {
            Err(FusionError::TileSpecMismatch { .. }) => {}
            other => panic!("expected TileSpecMismatch, got {other:?}"),
        }
    }

    /// `with_tile_spec` rejects keys absent from the kernel's manifest — the
    /// dispatch binds the tiled slot's buffer through its manifest entry.
    #[test]
    #[should_panic(expected = "not in this kernel's field_inputs manifest")]
    fn with_tile_spec_rejects_unknown_field() {
        let g = interior_grade();
        let (a, _aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g);
        let _ = a.with_tile_spec(TileSpec {
            halo: vec![2],
            tiled_field_keys: vec!["not_in_manifest".into()],
        });
    }

    /// an all-zero halo leaves the block unextended, so smem would repeat gmem.
    /// the builder rejects it.
    #[test]
    #[should_panic(expected = "at least one axis halo must be > 0")]
    fn with_tile_spec_rejects_zero_halo() {
        let g = interior_grade();
        let (a, _aw) = doubler("a_in", "p.a_in", "a_out", "p.a_out", g);
        let _ = a.with_tile_spec(TileSpec {
            halo: vec![0, 0, 0],
            tiled_field_keys: vec!["a_in".into()],
        });
    }

    /// the smem-footprint helper is the load-bearing piece for the
    /// dispatch-side smem allocation. small canonical cases:
    ///   - cube halo=[2,2,2], block (16, 8, 4), 1 field, f64 (8 bytes):
    ///       tile = (16+4) * (8+4) * (4+4) = 20*12*8 = 1920 cells
    ///       bytes = 1920 * 8 * 1 = 15,360
    ///   - slab halo=[2,0,0] (reconstruct along axis 0 alone), block (32, 8, 1),
    ///     8 fields, f64: tile = (32+4) * 8 * 1 = 288 cells
    ///       bytes = 288 * 8 * 8 = 18,432 — vs the 51,840 a fat cube would cost.
    #[test]
    fn tile_spec_smem_footprint_math() {
        let spec_cube = TileSpec {
            halo: vec![2, 2, 2],
            tiled_field_keys: vec!["x".into()],
        };
        assert_eq!(spec_cube.smem_bytes_per_block(&[16, 8, 4], 8), 1920 * 8);

        let keys: Vec<InputKey> = (0..8).map(|k| InputKey::new(format!("f{k}"))).collect();
        let spec_slab = TileSpec {
            halo: vec![2, 0, 0],
            tiled_field_keys: keys,
        };
        assert_eq!(spec_slab.smem_bytes_per_block(&[32, 8, 1], 8), 288 * 8 * 8);
    }

    /// the fusion laws hold when both kernels declare `None` for `tile_spec`
    /// (the untiled path): identity, grade-mismatch, and untagged all flow
    /// through `try_fuse` unchanged.
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
// captures every fresh NodeId as the scope's body, and pushes an Op::Scope
// node carrying body + result. consequences:
//
//   - the scope-form graph contains one more node than the inline-form
//     graph (the Op::Scope itself);
//   - the arithmetic nodes are identical in both forms (the scope is a
//     pure wrapper: the body region holds exactly the closure's arithmetic);
//   - nested Gv::scope produces nested Op::Scope nodes; structural shape
//     under graph.iter() is preserved (nested scopes appear as inner
//     Op::Scope nodes pushed ahead of the outer one).
//
// these tests pin that contract.
// =============================================================================

#[cfg(test)]
mod scope_op_contract {
    use super::*;
    use crate::graph::Op;
    use symbi_algebra::Numeric;

    /// at Gv, `scope(|| body)` produces inline-form plus exactly one extra
    /// Op::Scope node at the tail; the body subgraph matches inline-form.
    #[test]
    fn gv_scope_emits_one_op_scope_node() {
        // scope-form: (x + y) * (x + y) wrapped in Gv::scope.
        let (scope_kernel, scope_root_node) = trace(|cx: TraceCx| {
            let x = cx.scalar("x");
            let y = cx.scalar("y");
            let scope_root = <Gv as crate::algebra::Scalar>::scope(|| {
                let s = x + y;
                s * s
            });
            scope_root.node()
        });
        let scope_graph = scope_kernel.graph;

        // inline-form: same math, no wrapper.
        let (inline_kernel, inline_root_node) = trace(|cx: TraceCx| {
            let x = cx.scalar("x");
            let y = cx.scalar("y");
            let s = x + y;
            (s * s).node()
        });
        let inline_graph = inline_kernel.graph;

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
        // tail node is Op::Scope whose result is the trailing Mul.
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

    /// nesting Gv::scope: the inner Op::Scope appears ahead of the outer one,
    /// and the outer body contains the inner Op::Scope NodeId.
    #[test]
    fn gv_scope_nests_with_inner_op_scope() {
        let (kernel, outer_root_node) = trace(|cx: TraceCx| {
            let x = cx.scalar("x");
            let outer_root = <Gv as crate::algebra::Scalar>::scope(|| {
                let a = <Gv as crate::algebra::Scalar>::scope(|| x + Gv::from_f64(1.0));
                let b = <Gv as crate::algebra::Scalar>::scope(|| x * Gv::from_f64(2.0));
                a + b
            });
            outer_root.node()
        });
        let graph = kernel.graph;

        // outer Op::Scope is the tail.
        match &graph.node(outer_root_node).op {
            Op::Scope { body, result: _ } => {
                // outer body contains two inner Op::Scope NodeIds (a, b)
                // and one Add (a + b). count the body entries that are
                // Op::Scope.
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
    /// NodeId (pushing no fresh nodes) returns that NodeId directly, leaving
    /// the graph free of an Op::Scope.
    #[test]
    fn gv_scope_empty_body_short_circuits() {
        let (kernel, (before, after, r_node, x_node)) = trace(|cx: TraceCx| {
            let x = cx.scalar("x");
            let before = cx.with_trace(|t| t.graph.len());
            // closure result is `x` itself, so the closure pushes nothing.
            let r = <Gv as crate::algebra::Scalar>::scope(|| x);
            let after = cx.with_trace(|t| t.graph.len());
            (before, after, r.node(), x.node())
        });
        let g = kernel.graph;
        assert_eq!(before, after, "empty-body scope must not push any node");
        assert_eq!(
            r_node, x_node,
            "empty-body scope must return the input directly"
        );
        // the final graph is free of Op::Scope.
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

    // `Gv::powi` lowers to a multiply chain: `f64::powi` raises a negative base exactly
    // (e.g., (-2)^2 = 4) while `powf(neg, 2.0)` is NaN on CUDA, so a Pow/powf lowering
    // would break carrier equivalence between the f64 host path and the traced kernel.
    // this pins the structure (a multiply chain, free of Pow nodes) and the numeric
    // agreement with `f64::powi` on negative bases.
    #[test]
    fn powi_lowers_to_multiplies_and_matches_f64_on_negative_base() {
        for n in [0_i32, 1, 2, 3, 4, 5, -2, -3] {
            let (kernel, root) = trace(|cx: TraceCx| {
                let x = cx.scalar("x");
                <Gv as crate::algebra::Scalar>::powi(x, n).node()
            });
            let graph = kernel.graph;

            // structural: the multiply-chain lowering leaves the graph free of Pow ops.
            let has_pow = (0..graph.len()).any(|i| {
                matches!(
                    graph.node(NodeId(i as u32)).op,
                    Op::ElementWise(ElementWiseOp::Pow, _)
                )
            });
            assert!(!has_pow, "powi({n}) must lower to multiplies, not Pow/powf");

            // numeric: evaluate the traced graph and require bit-agreement with f64::powi
            // for negative (and a few positive) bases — the case where powf returns NaN.
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

    // the defaulted carrier helpers `safe_sqrt` (= sqrt(max(.,0))) and `clamp` trace at S=Gv
    // and match the f64 path bit-for-bit. safe_sqrt of a negative radicand is 0: the clamp
    // ahead of the sqrt holds the radicand at 0, so the trace carries a finite value where an
    // unguarded sqrt(neg) would carry a NaN.
    #[test]
    fn safe_sqrt_and_clamp_trace_and_match_f64() {
        use crate::algebra::Scalar;
        // safe_sqrt
        let (kernel, root) = trace(|cx: TraceCx| {
            let x = cx.scalar("x");
            Scalar::safe_sqrt(x).node()
        });
        let g = kernel.graph;
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
        let (ckernel, croot) = trace(|cx: TraceCx| {
            let y = cx.scalar("y");
            Scalar::clamp(y, cx.lit(-1.0), cx.lit(1.0)).node()
        });
        let cg = ckernel.graph;
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
