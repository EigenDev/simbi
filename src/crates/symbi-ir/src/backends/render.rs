// =============================================================================
// emit_render.rs
//
// the Renderer trait + the SHARED kernel-emission driver (docs/design/12). a
// scalarized stencil kernel is lowered ONCE — scalarize, buffer assignment, the
// FieldLoadAt -> buffer-index rewrite, the base-cell-read gate, the skeleton
// sequencing — and a per-backend `KernelRenderer` supplies only the language
// SPELLING (types, the signature/qualifier, the cell loop vs thread index, the
// flat-index cast, statement/expression syntax). adding a backend (HIP, SYCL,
// Metal) is one `KernelRenderer` impl; a feature is one edit here, not N across
// parallel emitters.
//
// `emit_kernel_cpu` and `emit_kernel_from_lowering` are thin wrappers over
// `emit_kernel_render` with `RustRenderer` / `CRenderer` (the CPU emitter tests +
// the CUDA emitter tests + the PTX gate are the regression anchors). the base-read
// gate elides dead `double key = buf[cell];` reads on the CUDA path.
// =============================================================================

use std::collections::{BTreeMap, HashMap};

use crate::backends::kernel::KernelEmitInputs;
use crate::emit::{FieldBinding, KernelDescriptor};
use crate::passes::scalarize::{KernelScalarized, ScalarExpr, ScalarStmt, scalarize_kernel};
use crate::{ElementTy, Graph, NodeId};
use symbi_abi::FieldBind;
use symbi_abi::ScalarBind;

pub(crate) const COORD_VARS: [&str; 3] = ["ii", "jj", "kk"];

/// per-backend SPELLING for the shared kernel driver. every method renders a
/// fragment in the target language; the driver owns the structure + sequencing.
pub trait KernelRenderer {
    /// the lines emitted before the signature: Rust `#[allow(unused_parens)]`;
    /// CUDA the target header (empty for CUDA) + the device-function preamble.
    fn preamble(&self, device_preamble: &[String]) -> String {
        let _ = device_preamble;
        String::new()
    }
    /// a buffer parameter: `buf{idx}: &mut [f64]` / `double* buf{idx}`.
    fn buffer_param(&self, idx: u32, is_output: bool) -> String;
    /// the grid-extent param for an axis (CPU `grid_size_a: i32`, CUDA
    /// `unsigned int grid_size_a`).
    fn grid_size_param(&self, axis: usize) -> String;
    /// a signed integer index param (`dom_lo`, `buf_lo` — can be negative): CPU
    /// `name: i32`, CUDA `int name`.
    fn int_param(&self, name: &str) -> String;
    /// a buffer-extent param (a size — non-negative): CPU `name: i32`, CUDA
    /// `unsigned int name`. defaults to `int_param` (CPU does not distinguish).
    fn extent_param(&self, name: &str) -> String {
        self.int_param(name)
    }
    /// a user scalar param typed by its element (`name: f64` / `double name`).
    fn scalar_param(&self, name: &str, element: ElementTy) -> String;
    /// the signature line: `pub fn name(` / `extern "C" __global__ void name(`.
    /// the driver appends the joined params and `params_close`.
    fn open_signature(&self, name: &str) -> String;
    /// the text closing the parameter list + opening the body. Rust allows a
    /// trailing comma (`,\n) {\n`); C does NOT (`\n) {\n`).
    fn params_close(&self) -> &'static str;
    /// everything between the signature and the `_coord_N` decls, in the backend's
    /// own order: the iteration model (CPU `for _ia in 0..grid_size_a {` + coord
    /// `let ii: i32 = ..`; CUDA thread index + bounds check + coord) AND the
    /// per-buffer stride locals (multi-dim) — the two backends order these
    /// differently, so each owns the full block.
    fn cell_prelude(&self, ndim: usize, n_buffers: u32) -> Vec<String>;
    /// a `_coord_N` declaration referencing the axis coord var; `element` is the
    /// coord param's declared type (CPU is always I32; CUDA casts a float coord).
    fn coord_decl(&self, axis: u8, element: ElementTy) -> String;
    /// which surface language the renderer emits. drives the per-cell index-base
    /// preamble emitted by `render_source` (see `emit::emit_cell_index_base`).
    /// every other syntactic dispatch goes through the existing trait methods.
    fn index_lang(&self) -> crate::emit::IndexLang;
    /// when `true`, `render` rewrites eligible float bool/if bodies into the
    /// branch-free `cmp_*` / `S::select` spelling BEFORE emission (see
    /// `passes::mask_form`) — a straight-line body LLVM's SLP vectorizer can
    /// fuse. the serialized `Prepared` artifact is untouched (the rewrite runs
    /// on the render-time copy). default = `false`; the Rust CPU renderer
    /// exposes it behind the SYMBI_MASK_FORM a/b knob — `select` computes both
    /// arms, which measured SLOWER than the branch-predicted bool/if form on
    /// the flux body (docs/design/47 resolution).
    fn mask_form(&self) -> bool {
        false
    }
    /// told once per `render` whether the mask-form rewrite was APPLIED to this
    /// kernel (eligible bodies only — control-flow / untyped-comparison bodies
    /// keep the bool/if spelling). renderers that spell the mask type
    /// differently from native `bool` (the Rust `S::Mask` associated type)
    /// track this to type bool locals per kernel. default: ignore.
    fn note_mask_form(&self, _applied: bool) {}
    /// when `true`, `render_source` skips emitting the per-buffer `buf_lo_<b>_<a>`
    /// and `buf_extent_<b>_<a>` scalar kernel args — the renderer bundles that
    /// layout into its `buffer_param` (e.g., a `View` struct that carries the
    /// pointer + lo + pre-multiplied strides). default = `false` (the scattered
    /// ABI). CUDA returns `true` (the View struct bundles the layout).
    fn skip_scattered_buffer_layout_args(&self) -> bool {
        false
    }
    /// the flat buffer index from rendered component strings. ALWAYS delegates
    /// to `emit::emit_flat_index` with `index_lang()` — implementations are
    /// one-liners. defined as a trait method (not a free function) so future
    /// backends with a different ABI can override without touching the driver.
    fn flat_index(&self, ndim: u8, buf: u32, comps: &[String]) -> String;
    /// render a FieldLoadAt component (an index expression). CPU enforces integer
    /// arithmetic (panics on a data-dependent gather); CUDA renders generally.
    fn render_index_component(&self, e: &ScalarExpr, coord_vars: &[&str]) -> String;
    /// a base cell read: `let key: f64 = buf{buf}[{flat}];` / `double key = ...`.
    fn base_read(&self, key: &str, buf: u32, flat: &str) -> String;
    /// the rvalue spelling for a stencil load — `field{buf}.data[{flat}]` (Rust,
    /// the view-struct path) or `buf{buf}[{flat}]` (CUDA, the scattered-args
    /// path). default = scattered, CPU overrides.
    fn load_at_expr(&self, buf: u32, flat: &str) -> String {
        format!("buf{buf}[{flat}]")
    }
    /// render a body statement (the scalarized let/assign/for).
    fn render_stmt(&self, stmt: &ScalarStmt) -> String;
    /// render an output RHS expression.
    fn render_output(&self, expr: &ScalarExpr) -> String;
    /// a per-output store: `buf{buf}[{flat}] = {expr};`.
    fn store(&self, buf: u32, flat: &str, expr: &str) -> String;
    /// the kernel closer: CPU closes `ndim` loop braces then the fn; CUDA closes
    /// just the fn.
    fn close(&self, ndim: usize) -> String;

    // ----- Gate 3 shared-memory tiling (docs/design/22) -----
    // the smem hooks. defaults make tiling a NO-OP: `smem_prelude` emits nothing,
    // and the load/base hooks return `None` so the driver falls back to the gmem
    // path. only the C-family (CUDA) renderer overrides them. the CPU renderer
    // therefore ignores `tile_spec` entirely (it cache-tiles via loop structure),
    // which keeps the interp/CPU oracle the policy-agnostic flat reference.

    /// the block-level prelude: allocate one `__shared__` slab per tiled field and
    /// cooperatively prefetch the (block + per-axis halo) region from gmem, ending
    /// in `__syncthreads()`. `tiled` is `(field_key, buffer_index)` per tiled field.
    /// emitted BEFORE the bounds-check return so EVERY thread (incl. padding) reaches
    /// the barrier. default: nothing (no smem).
    fn smem_prelude(&self, _ndim: usize, _halo: &[u8], _tiled: &[(String, u32)]) -> Vec<String> {
        Vec::new()
    }
    /// the rvalue for a TILED stencil load: `tile_<key>[<local tile offset>]`.
    /// `comps` are the rendered absolute index components; the renderer derives the
    /// per-axis local offset `threadIdx + halo + (comp - coord_var)`. `None` =
    /// backend has no smem; the driver uses the gmem path.
    fn tiled_load_expr(
        &self,
        _key: &str,
        _halo: &[u8],
        _ndim: u8,
        _comps: &[String],
    ) -> Option<String> {
        None
    }
    /// the base (cell-center) read of a TILED field, as a full decl line
    /// `<ty> <key> = tile_<key>[<center>];` (delta 0). `None` = gmem base read.
    fn tiled_base_read(&self, _key: &str, _halo: &[u8], _ndim: u8) -> Option<String> {
        None
    }
}

/// the per-render tile context: the tiled-field set + per-axis halo + the
/// (key, buffer) pairs in field-input order. built once in `render` from
/// `Prepared::tile_spec` and threaded into the FieldLoadAt rewrite + `render_source`.
struct TileCtx {
    halo: Vec<u8>,
    fields: Vec<(String, u32)>,
    keys: std::collections::HashSet<String>,
}

/// the backend-NEUTRAL prepared form of a kernel: the scalarized body, the buffer
/// bindings, and the dispatch side-tables — everything `render` needs EXCEPT the
/// renderer-dependent FieldLoadAt rewrite (which bakes a backend's flat-index
/// spelling, so it must run per-backend, AFTER serialization). this is the durable
/// artifact of docs/design/15 §3: `build.rs` serializes it into the binary; the
/// runtime deserializes and `render`s it for any accelerator.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Prepared {
    pub kernel_name: String,
    pub ndim: u8,
    /// the shared scalarized body + outputs (FieldLoadAt NOT yet rewritten).
    pub scalarized: KernelScalarized,
    /// buffer assignment (inputs first, then output-only), in buffer order.
    pub bindings: Vec<FieldBinding>,
    /// (IR-side key, born-typed runtime binding) per field input, in buffer order. the
    /// key gates the base cell read + resolves FieldLoadAt; the FieldBind picks the buf.
    pub field_inputs: Vec<(String, FieldBind)>,
    /// born-typed runtime binding per field write, zipped against `scalarized.outputs`.
    /// classified once at `prepare` (the transient `KernelEmitInputs` still carries raw write
    /// strings; the producer-mint of typed writes is the Stage-2 step).
    pub field_writes: Vec<FieldBind>,
    /// scalar kernel args (dt, gamma, …), born typed: a `Ref` over the closed
    /// dispatch vocabulary or a `Spec` open spec/user knob. minted in `prepare`
    /// from the producer's string names; `name()` recovers the exact original
    /// spelling for codegen, so the emitted kernel source is unchanged.
    pub scalar_params: Vec<ScalarBind>,
    /// kernel-coord component axes referenced by the body.
    pub coord_components: Vec<u8>,
    /// device-function definitions to emit ahead of the kernel (GPU preamble).
    pub device_preamble: Vec<String>,
    /// element type per scalar/coord param (resolved from the graph's Param nodes;
    /// not derivable from the other fields, so it travels with the artifact).
    /// BTreeMap, NOT HashMap: this struct serializes into the build-time `.ir.json`
    /// artifact, and a HashMap serializes in random per-process order — making the
    /// generated IR non-reproducible build-to-build (spurious diffs / cache churn).
    /// only ever read via `.get(name)`, so the ordering is otherwise irrelevant.
    pub param_elem: BTreeMap<String, ElementTy>,
    /// the shared-memory tile spec (Gate 3, docs/design/22), threaded from the
    /// `GvKernel`. when `Some`, the C-family renderer emits a cooperative smem
    /// prefetch prelude + redirects the tiled fields' stencil reads to `__shared__`;
    /// other renderers (CPU) ignore it. serialized into the build-time artifact so
    /// the runtime render path (`render_from_ir`) sees the same intent.
    #[serde(default)]
    pub tile_spec: Option<crate::gv::TileSpec>,
    /// when true, every buffer is guaranteed to
    /// share buffer 0's allocated layout (same `lo`/`strides`), so the per-cell
    /// flat index is computed ONCE from buffer 0 and aliased to every other
    /// buffer — the `View` collapse of N strided index computations into one.
    /// only valid for kernels whose buffers are all co-located cell-centered
    /// fields over one grid (c2p, hydro flux, the cfl wave-speed maps); NOT for
    /// cross-grid (amr prolong/restrict) or staggered (mhd efield/bface) kernels.
    /// derived in `prepare` from the kernel name until real per-field layout
    /// identity lands. the carrier oracle is the correctness gate.
    #[serde(default)]
    pub coalesce_layout: bool,
}

/// emit a scalarized stencil kernel for `R`'s backend — `prepare` then `render`.
/// SHARED across backends: scalarize, buffer assignment, the FieldLoadAt rewrite,
/// the base-read gate, the skeleton sequencing. only the spelling comes from `r`.
pub fn emit_kernel_render<R: KernelRenderer>(
    graph: &Graph,
    inputs: &KernelEmitInputs,
    r: &R,
) -> KernelDescriptor {
    render(prepare(graph, inputs), r)
}

/// lower a kernel graph to its backend-NEUTRAL `Prepared` form: scalarize, assign
/// buffers, resolve param element types. NO renderer is involved — the output is
/// the serializable artifact, identical for every backend.
pub fn prepare(graph: &Graph, inputs: &KernelEmitInputs) -> Prepared {
    assert!(
        (1..=3).contains(&inputs.ndim),
        "prepare: ndim must be 1, 2, or 3 (got {})",
        inputs.ndim,
    );

    // ---- 1. scalarize all output RHSes through one shared body ----
    let output_nodes: Vec<NodeId> = inputs.field_writes.iter().map(|(_, _, id)| *id).collect();
    let mut scalarized = scalarize_kernel(graph, &output_nodes);
    // lazy scheduling of expensive select arms (passes::lazy_select): a select
    // whose arm-exclusive cost crosses the threshold becomes a real branch with
    // its exclusive lets sunk in — the taken arm's value is unchanged, so the
    // rewrite is bit-exact on every carrier and every backend inherits it.
    crate::passes::lazy_select::apply(&mut scalarized);

    // ---- 2. assign buffer indices in a CANONICAL order, independent of the builder's
    //          field-touch order: pure inputs first (field_inputs order), then ALL
    //          outputs in field_writes (writes) order. an in-place field (read AND
    //          written) belongs to the OUTPUT group at its writes position — it does NOT
    //          inherit its read position. so a geometric source that reads cons.mom early
    //          no longer shuffles cons.mom ahead of cons.den in the signature; the
    //          KernelSet always provides outputs in the natural [den, mom.., nrg] order.
    //          (Cartesian is a no-op: there touch order already equals writes order.)
    // reads AND writes are born-typed FieldBind now (docs/design/38 L2 Stage 2). the buffer map
    // keys on FieldBind, so the in-place detection (a read whose buffer is also written) is
    // spelling-invariant by construction.
    let write_fields: std::collections::HashSet<FieldBind> = inputs
        .field_writes
        .iter()
        .map(|(_, w, _)| w.clone())
        .collect();
    let mut bindings: Vec<FieldBinding> = Vec::new();
    let mut buf_idx_by_runtime: HashMap<FieldBind, u32> = HashMap::new();
    for (_, runtime_path) in inputs.field_inputs {
        if write_fields.contains(runtime_path) || buf_idx_by_runtime.contains_key(runtime_path) {
            continue; // in-place fields are placed with the outputs; dedup repeats.
        }
        let buf_idx = bindings.len() as u32;
        bindings.push(FieldBinding {
            field: runtime_path.clone(),
            buffer_index: buf_idx,
            is_output: false,
        });
        buf_idx_by_runtime.insert(runtime_path.clone(), buf_idx);
    }
    for (_, write_runtime, _) in inputs.field_writes {
        let wf = write_runtime.clone();
        if buf_idx_by_runtime.contains_key(&wf) {
            continue; // dedup a field written more than once.
        }
        let buf_idx = bindings.len() as u32;
        bindings.push(FieldBinding {
            field: wf.clone(),
            buffer_index: buf_idx,
            is_output: true,
        });
        buf_idx_by_runtime.insert(wf, buf_idx);
    }

    // ---- 3. scalar-param element types (from the Param nodes) ----
    let param_elem: BTreeMap<String, ElementTy> = graph
        .iter()
        .filter_map(|(_, node, ty)| match &node.op {
            crate::graph::Op::Param(sym) => Some((sym.as_str().to_string(), ty.element)),
            _ => None,
        })
        .collect();

    Prepared {
        kernel_name: inputs.kernel_name.to_string(),
        ndim: inputs.ndim,
        scalarized,
        bindings,
        field_inputs: inputs.field_inputs.to_vec(),
        field_writes: inputs
            .field_writes
            .iter()
            .map(|(_, rt, _)| rt.clone())
            .collect(),
        scalar_params: inputs
            .scalar_params
            .iter()
            .map(|s| ScalarBind::from_name(s))
            .collect(),
        coord_components: inputs.coord_components.to_vec(),
        device_preamble: inputs.device_preamble.to_vec(),
        param_elem,
        tile_spec: inputs.tile_spec.cloned(),
        coalesce_layout: inputs.coalesce_layout,
    }
}

/// the runtime buffer manifest for a serialized kernel (docs/design/18 D3): each
/// `(runtime_path, is_output)` in CANONICAL buffer order (pure inputs first, then
/// outputs; in-place fields fold into the output group). a metadata-driven dispatch
/// resolves each `runtime_path` against the sim's fields and binds in this order, so
/// no caller hand-reconstructs a kernel's `field_inputs` layout — the axis-role /
/// ncomp / curvilinear ordering quirks all read straight off the artifact.
pub fn kernel_bindings_from_ir(ir: &str) -> Vec<(FieldBind, bool)> {
    let prepared: Prepared = serde_json::from_str(ir).expect("deserialize Prepared from kernel IR");
    let mut by_idx = prepared.bindings;
    by_idx.sort_by_key(|b| b.buffer_index);
    by_idx.into_iter().map(|b| (b.field, b.is_output)).collect()
}

/// the TYPE-SORTED scalar manifest: each scalar-param name paired with its int/float sort, in
/// declared order (`true` = int). a kernel's scalar params are a disjoint union — INT lanes
/// (the `ints` ABI tail) ⊔ FLOAT lanes (the `scalars` tail) — and a metadata-driven dispatch
/// reads the sort to route each lane by name to the right tail (so a mixed kernel like
/// ghost-fill, with int `map_type`/`arg` + float `vel_sign`, resolves fully by name, never
/// positionally). the sort comes from the graph's param element types (`param_elem`).
pub fn kernel_scalar_params_typed_from_ir(ir: &str) -> Vec<(ScalarBind, bool)> {
    let prepared: Prepared = serde_json::from_str(ir).expect("deserialize Prepared from kernel IR");
    prepared
        .scalar_params
        .iter()
        .map(|bind| {
            let is_int = prepared
                .param_elem
                .get(&bind.name())
                .map(|e| e.is_integer())
                .unwrap_or(false);
            (bind.clone(), is_int)
        })
        .collect()
}

/// spell a `Prepared` kernel in `R`'s backend: rewrite FieldLoadAt to the backend's
/// flat-index form (the one renderer-dependent step, so it lives HERE not in
/// `prepare`), then emit the source. `render(deserialize(serialize(prepare(g))), r)`
/// equals `render(prepare(g), r)` — the round-trip is the correctness proof.
pub fn render<R: KernelRenderer>(mut prepared: Prepared, r: &R) -> KernelDescriptor {
    // branch-free spelling for backends that want it. runs BEFORE the
    // FieldLoadAt rewrite so index expressions are still identifiable (the
    // pass must not enter them). ineligible bodies (statement control flow,
    // untyped comparisons) keep the bool/if spelling.
    if r.mask_form() {
        let applied = crate::passes::mask_form::apply(&mut prepared.scalarized);
        r.note_mask_form(applied);
    }

    // buf-index lookups, rederived from the bindings (consistent by construction):
    // field-bind -> buffer, and IR key -> buffer (via the field-input path). the join
    // is keyed by `FieldBind`, NOT by the raw runtime string: a producer mints two
    // spellings for the SAME buffer (`prim.vel_k` c2p-write vs `prim.vel[k]`
    // reconstruction-read), and both parse to the same `FieldRef` — so a string key
    // would silently miss across the dual spellings. parsing each side to `FieldBind`
    // unifies them (and a hand-built `Raw` path matches itself verbatim).
    let buf_idx_by_field: HashMap<FieldBind, u32> = prepared
        .bindings
        .iter()
        .map(|b| (b.field.clone(), b.buffer_index))
        .collect();
    let key_to_buf: HashMap<String, u32> = prepared
        .field_inputs
        .iter()
        .map(|(key, runtime)| (key.clone(), buf_idx_by_field[runtime]))
        .collect();

    // Gate 3: build the tile context from the spec, restricted to keys that are
    // real field inputs. `fields` follows field_inputs order so the smem slab
    // layout is deterministic. an empty/None spec => no tiling (flat path).
    let tile: Option<TileCtx> = prepared.tile_spec.as_ref().map(|spec| {
        let keys: std::collections::HashSet<String> =
            spec.tiled_field_keys.iter().cloned().collect();
        let fields: Vec<(String, u32)> = prepared
            .field_inputs
            .iter()
            .filter(|(k, _)| keys.contains(k))
            .map(|(k, _)| (k.clone(), key_to_buf[k]))
            .collect();
        TileCtx {
            halo: spec.halo.clone(),
            fields,
            keys,
        }
    });

    // resolve every FieldLoadAt to a `buf<idx>[<flat>]` Var (backend-specific) — or
    // a `tile_<key>[..]` smem read for a tiled field on a smem-capable backend.
    rewrite_field_load_at_stmts(
        &mut prepared.scalarized.body,
        prepared.ndim,
        &key_to_buf,
        tile.as_ref(),
        r,
    );
    for out in prepared.scalarized.outputs.iter_mut() {
        rewrite_field_load_at(out, prepared.ndim, &key_to_buf, tile.as_ref(), r);
    }

    let source = render_source(&prepared, &buf_idx_by_field, tile.as_ref(), r);
    // the int/float ABI flag per scalar param (declared order), from the graph's
    // param element types — a runtime launcher packs each lane accordingly.
    let scalar_is_int: Vec<bool> = prepared
        .scalar_params
        .iter()
        .map(|bind| {
            prepared
                .param_elem
                .get(&bind.name())
                .map(|e| e.is_integer())
                .unwrap_or(false)
        })
        .collect();
    KernelDescriptor {
        source,
        kernel_name: prepared.kernel_name,
        field_bindings: prepared.bindings,
        // the descriptor's `param_names` stay string-typed (the producer-facing
        // signature order); `name()` recovers each bind's exact original spelling.
        param_names: prepared.scalar_params.iter().map(|b| b.name()).collect(),
        scalar_is_int,
        tile_spec: prepared.tile_spec,
    }
}

fn render_source<R: KernelRenderer>(
    p: &Prepared,
    buf_idx_by_field: &HashMap<FieldBind, u32>,
    tile: Option<&TileCtx>,
    r: &R,
) -> String {
    let scalarized = &p.scalarized;
    let ndim = p.ndim as usize;
    let n_buffers = p.bindings.len() as u32;
    let mut out = String::new();

    // ---- signature ----
    let mut params: Vec<String> = Vec::new();
    for b in &p.bindings {
        params.push(r.buffer_param(b.buffer_index, b.is_output));
    }
    for aa in 0..ndim {
        params.push(r.grid_size_param(aa));
    }
    for aa in 0..ndim {
        params.push(r.int_param(&format!("dom_lo_{aa}")));
    }
    // emit scattered per-buffer layout args ONLY if the backend's `buffer_param`
    // didn't already absorb them (View-struct backends like CUDA do).
    if !r.skip_scattered_buffer_layout_args() {
        if ndim >= 2 {
            for bb in 0..n_buffers {
                for aa in 0..ndim {
                    params.push(r.extent_param(&format!("buf_extent_{bb}_{aa}")));
                }
            }
        }
        for bb in 0..n_buffers {
            for aa in 0..ndim {
                params.push(r.int_param(&format!("buf_lo_{bb}_{aa}")));
            }
        }
    }
    for pn in &p.scalar_params {
        let pn = pn.name();
        let elem = p.param_elem.get(&pn).copied().unwrap_or(ElementTy::F64);
        params.push(r.scalar_param(&pn, elem));
    }
    out.push_str(&r.preamble(&p.device_preamble));
    out.push_str(&r.open_signature(&p.kernel_name));
    out.push_str(&params.join(",\n"));
    out.push_str(r.params_close());

    // ---- Gate 3 smem prelude (cooperative prefetch + __syncthreads) ----
    // emitted BEFORE the cell prelude so EVERY thread — including the padding
    // threads the bounds check will drop — participates in the block-wide load
    // and reaches the barrier (a thread returning early before __syncthreads
    // deadlocks the block). it uses only block/thread builtins + params, never
    // the per-thread cell index, so it is correct ahead of the bounds check.
    if let Some(t) = tile {
        for line in r.smem_prelude(ndim, &t.halo, &t.fields) {
            out.push_str(&line);
            out.push('\n');
        }
    }

    // ---- cell prelude (iteration + strides, backend-ordered) + coord decls ----
    for line in r.cell_prelude(ndim, n_buffers) {
        out.push_str(&line);
        out.push('\n');
    }
    for &axis in &p.coord_components {
        assert!(
            (axis as usize) < ndim,
            "coord_components axis {axis} but ndim {ndim}"
        );
        let elem = p
            .param_elem
            .get(&format!("_coord_{axis}"))
            .copied()
            .unwrap_or(ElementTy::F64);
        out.push_str(&r.coord_decl(axis, elem));
        out.push('\n');
    }

    // ---- per-buffer CELL-BASE index hoisting ----
    // emits `__idx_cell_buf{N}` per buffer = flat offset at cell coord. every
    // subsequent `flat_index()` for that buffer renders as `__idx_cell_buf{N} +
    // literal_stencil_delta`, which the compiler folds to a single immediate-
    // displaced load. the formula lives in `emit::emit_cell_index_base`, ONCE for
    // all backends — `r.index_lang()` picks the syntax (Rust vs CUDA vs ...).
    let lang = r.index_lang();
    for b in 0..n_buffers {
        out.push_str(&crate::emit::emit_cell_index_base(
            lang,
            p.ndim,
            b,
            p.coalesce_layout,
        ));
        out.push('\n');
    }

    // ---- base per-input loads at the cell coord (gated on usage) ----
    let base: Vec<String> = COORD_VARS[..ndim].iter().map(|s| s.to_string()).collect();
    for (ir_key, runtime_path) in &p.field_inputs {
        if !scalarized_uses_var(scalarized, ir_key) {
            continue;
        }
        let buf = buf_idx_by_field[runtime_path];
        // a tiled field's cell-center read comes from smem too (its slab holds
        // this thread's own cell at the tile center); otherwise gmem.
        let tiled_base = tile
            .filter(|t| t.keys.contains(ir_key))
            .and_then(|t| r.tiled_base_read(ir_key, &t.halo, p.ndim));
        match tiled_base {
            Some(line) => out.push_str(&line),
            None => {
                let flat = r.flat_index(p.ndim, buf, &base);
                out.push_str(&r.base_read(ir_key, buf, &flat));
            }
        }
        out.push('\n');
    }

    // ---- scalarized body ----
    for stmt in &scalarized.body {
        out.push_str(&r.render_stmt(stmt));
        out.push('\n');
    }

    // ---- diagnostic: a param the scalarizer kept but no caller declared (a
    //      wiring bug). emit a comment rather than silently drop. emits nothing
    //      for well-formed kernels, so output is unchanged in the normal case.
    let coord_names: Vec<String> = p
        .coord_components
        .iter()
        .map(|a| format!("_coord_{a}"))
        .collect();
    let scalar_names: Vec<String> = p.scalar_params.iter().map(|s| s.name()).collect();
    let declared: std::collections::HashSet<&str> = p
        .field_inputs
        .iter()
        .map(|(k, _)| k.as_str())
        .chain(scalar_names.iter().map(|s| s.as_str()))
        .chain(coord_names.iter().map(|s| s.as_str()))
        .collect();
    for param in &scalarized.params {
        if !declared.contains(param.name.as_str()) {
            out.push_str(&format!(
                "    // WARNING: scalarizer introduced undeclared param '{}'; \
                this is a wiring bug — caller must pass it via field_inputs or scalar_params\n",
                param.name
            ));
        }
    }

    // ---- per-output write ----
    for (write_runtime, output_expr) in p.field_writes.iter().zip(scalarized.outputs.iter()) {
        let buf = buf_idx_by_field[write_runtime];
        let flat = r.flat_index(p.ndim, buf, &base);
        let expr = r.render_output(output_expr);
        out.push_str(&r.store(buf, &flat, &expr));
        out.push('\n');
    }

    out.push_str(&r.close(ndim));
    out
}

// ----- FieldLoadAt rewrite (renderer-driven index spelling) -----

fn rewrite_field_load_at_stmts<R: KernelRenderer>(
    stmts: &mut [ScalarStmt],
    ndim: u8,
    key_to_buf: &HashMap<String, u32>,
    tile: Option<&TileCtx>,
    r: &R,
) {
    // derived from `ScalarStmt::child_expr_mut` + `child_stmt_bodies_mut`.
    // adding a new statement variant requires no change here — only updates to
    // those two accessors in `scalarize.rs`.
    for s in stmts.iter_mut() {
        if let Some(e) = s.child_expr_mut() {
            rewrite_field_load_at(e, ndim, key_to_buf, tile, r);
        }
        for body in s.child_stmt_bodies_mut() {
            rewrite_field_load_at_stmts(body, ndim, key_to_buf, tile, r);
        }
    }
}

fn rewrite_field_load_at<R: KernelRenderer>(
    e: &mut ScalarExpr,
    ndim: u8,
    key_to_buf: &HashMap<String, u32>,
    tile: Option<&TileCtx>,
    r: &R,
) {
    use ScalarExpr::*;
    match e {
        FieldLoadAt {
            field_key,
            components,
        } => {
            for c in components.iter_mut() {
                rewrite_field_load_at(c, ndim, key_to_buf, tile, r);
            }
            let buf_idx = *key_to_buf.get(field_key).unwrap_or_else(|| {
                panic!("FieldLoadAt: field_key '{field_key}' has no buffer slot")
            });
            assert_eq!(
                components.len(),
                ndim as usize,
                "FieldLoadAt '{field_key}' has {} components but ndim is {ndim}",
                components.len(),
            );
            let coord_vars = &COORD_VARS[..ndim as usize];
            let comp_strs: Vec<String> = components
                .iter()
                .map(|c| r.render_index_component(c, coord_vars))
                .collect();
            // a tiled field on a smem-capable backend reads from `__shared__`;
            // otherwise (untiled field, or CPU) the gmem flat-index path.
            let smem = tile
                .filter(|t| t.keys.contains(field_key))
                .and_then(|t| r.tiled_load_expr(field_key, &t.halo, ndim, &comp_strs));
            *e = match smem {
                Some(s) => Var(s),
                None => {
                    let flat = r.flat_index(ndim, buf_idx, &comp_strs);
                    Var(r.load_at_expr(buf_idx, &flat))
                }
            };
        }
        // every non-FieldLoadAt node just recurses its SSOT children (docs/design/38 P3b).
        _ => {
            for c in e.children_mut() {
                rewrite_field_load_at(c, ndim, key_to_buf, tile, r);
            }
        }
    }
}

// ----- the base-read gate (shared: applies to every backend) -----

pub(crate) fn scalarized_uses_var(
    s: &crate::passes::scalarize::KernelScalarized,
    name: &str,
) -> bool {
    s.body.iter().any(|st| stmt_uses_var(st, name))
        || s.outputs.iter().any(|e| expr_uses_var(e, name))
}

fn stmt_uses_var(s: &ScalarStmt, name: &str) -> bool {
    // derived from `ScalarStmt::child_expr` + `child_stmt_bodies`.
    s.child_expr().is_some_and(|e| expr_uses_var(e, name))
        || s.child_stmt_bodies()
            .iter()
            .flat_map(|b| b.iter())
            .any(|st| stmt_uses_var(st, name))
}

fn expr_uses_var(e: &ScalarExpr, name: &str) -> bool {
    matches!(e, ScalarExpr::Var(n) if n == name)
        || e.children().iter().any(|c| expr_uses_var(c, name))
}

#[cfg(test)]
mod tests {
    use crate::ConstValue;

    // the floats that broke naive serde_json (NaN/Inf -> null): the bit-pattern
    // ConstValue serde must round-trip them exactly.
    #[test]
    fn non_finite_consts_survive_serde() {
        for x in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.1_f64,
            -2.5_f64,
        ] {
            let cv = ConstValue::F64(x);
            let json = serde_json::to_string(&cv).expect("serialize ConstValue");
            let back: ConstValue = serde_json::from_str(&json).expect("deserialize ConstValue");
            // bit-identical: PartialEq on ConstValue compares bit patterns, so NaN
            // (which is != itself in IEEE) still round-trips equal here.
            assert_eq!(cv, back, "ConstValue::F64({x}) did not round-trip ({json})");
        }
    }
}
