// =============================================================================
// emit_kernel.rs
//
// chalkboard-pipeline kernel emitter (R.6.d.2). given a scalarized
// tensor IR graph plus the kernel-mode side-tables (field-read keys
// with their dotted runtime paths, field writes, scalar user params),
// produces a complete __global__ kernel over the per-cell view ABI
// (the `__symbi_View<T>` buffer struct).
//
// the per-cell math comes from `tensor::scalarize_kernel` — one
// shared body of let-bindings across all outputs. the kernel surface
// (signature, thread indexing, bounds check, per-buffer loads/stores)
// is target-parameterized via `TargetConfig`; CUDA, HIP, and Metal
// share the same shape via `header()` / `global_qualifier()`
// in `symbi_ir::emit`. SYCL/FPGA would slot in as sibling emitters
// (the IR is target-agnostic; only the source text shifts).
//
// the macro layer (R.6.d.3) wraps each ndim ∈ {1,2,3} in its own
// invocation to populate per-ndim KernelDescriptors.
//
// usage:
//   let desc = emit_kernel_from_lowering(&graph, &KernelEmitInputs {
//       kernel_name: "iso_c2p_1d",
//       ndim: 1,
//       target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
//       field_inputs: &[(("cons_den".into(), "cons.den".into()))],
//       scalar_params: &[],
//       field_writes: &[(("prim_rho".into(), "prim.rho".into(), rho_node))],
//   });
// =============================================================================

use crate::emit::{self, KernelDescriptor, Precision, ReductionOp, Target, TargetConfig};
use crate::{ElementTy, Graph, NodeId};
use symbi_abi::FieldBind;
use crate::passes::scalarize::{ScalarExpr, ScalarStmt};
use crate::backends::cuda::{emit_expr, emit_stmt};
use crate::backends::render::{emit_kernel_render, render, KernelRenderer, Prepared, COORD_VARS};

/// inputs to `emit_kernel_from_lowering`. the order of `field_inputs`
/// fixes the buffer indices for inputs; write-only fields are appended
/// after. dispatch on the macro side must match this order when packing
/// `__buf_extents` / `__buf_los` / `__field_ptrs`.
#[derive(Debug, Clone)]
pub struct KernelEmitInputs<'a> {
    pub kernel_name:   &'a str,
    pub ndim:          u8,
    pub target:        TargetConfig,
    /// whether all of this kernel's buffers share ONE allocated layout, so the
    /// cell index can be computed once and shared across reads. the PRODUCER sets
    /// this (it knows the kernel's buffer topology); the IR stays domain-agnostic
    /// and no longer infers it from the kernel name. true for single-layout
    /// cell-centered kernels (c2p, wave-speed maps, pure-hydro face flux); false
    /// for staggered mhd face-flux (edge efield) and amr prolong/restrict (two grids).
    pub coalesce_layout: bool,
    /// (IR-side synthesized key, born-typed runtime binding). the IR key
    /// matches a Param node in the graph; the FieldBind is what ends up in
    /// `FieldBinding::field` for the dispatch side (no re-parse).
    pub field_inputs:  &'a [(String, FieldBind)],
    /// IR-side param names that stay as scalar __global__ args (user
    /// scalars: dt, gamma, etc.) — not loaded from buffers.
    pub scalar_params: &'a [String],
    /// (write key, runtime path, RHS NodeId). each entry produces one
    /// buffer store; if a write's runtime path matches an input, the
    /// buffer is shared and marked is_output.
    pub field_writes:  &'a [(String, FieldBind, NodeId)],
    /// kernel-coord component axes referenced by the body. each entry
    /// gets a `double _coord_N = (double)<thread-axis>;` line emitted
    /// after the thread-index prelude. body Param references
    /// `_coord_0`/`_coord_1`/`_coord_2` resolve there.
    pub coord_components: &'a [u8],
    /// F1.B.11: device-function definitions to include in the kernel
    /// source, ahead of the `__global__` block. emitted in order — the
    /// caller (kernel macro) is responsible for topological order
    /// (callees before callers) and de-duplication. each entry is a
    /// complete `__device__ inline RET name(...) { ... }` string.
    pub device_preamble: &'a [String],
    /// the kernel's shared-memory tile intent for STENCIL
    /// kernels (halo + stencil-read field keys). `None` = no smem tiling.
    /// the CUDA emitter (Gate 3) cooperatively prefetches the (block + halo)
    /// region for these fields into `__shared__`; the CPU emitter ignores it
    /// (it cache-tiles every kernel unconditionally). inferred for stencil
    /// kernels via `infer_tile_spec`; threaded here from the `GvKernel`.
    pub tile_spec: Option<&'a crate::gv::TileSpec>,
}

/// the C-FAMILY backend spelling for the shared kernel driver (`emit_render`):
/// CUDA AND HIP. produces an `extern "C" __global__` kernel over raw
/// `<precision>*` buffers (the per-cell view ABI shape); the header + global qualifier
/// vary by `target.target` (`emit::header` / `global_qualifier`), so HIP is a
/// pure token-map with zero physics edits. Metal (MSL: buffer-index ABI, no
/// `double`) needs its OWN renderer, not this one.
pub struct CRenderer {
    pub target: TargetConfig,
}

impl CRenderer {
    fn ty(&self) -> &'static str {
        self.target.precision.c_type()
    }
}

impl KernelRenderer for CRenderer {
    fn preamble(&self, device_preamble: &[String]) -> String {
        // every CUDA kernel takes its buffers as `__symbi_View<T>`
        // structs — one struct per buffer carrying
        // ptr + lo + pre-multiplied strides. emit the struct typedef ONCE in the
        // preamble so the kernel signature can spell `__symbi_View field0`.
        // the host side packs a matching POD into the kernel arg buffer (see
        // `substrate_gpu::DeviceView`).
        let mut s = String::from(emit::header(self.target.target));
        s.push_str(&format!(
            "struct __symbi_View {{ {ty}* __restrict__ data; int lo[4]; int strides[4]; int extent[4]; }};\n",
            ty = self.ty(),
        ));
        for dev_fn in device_preamble {
            s.push_str(dev_fn);
            if !dev_fn.ends_with('\n') {
                s.push('\n');
            }
        }
        s
    }
    fn buffer_param(&self, idx: u32, _is_output: bool) -> String {
        // every buffer is a View struct (data + lo + strides). drops
        // the matching `buf_lo_*` / `buf_extent_*` scalar args (see
        // `skip_scattered_buffer_layout_args`).
        format!("    __symbi_View field{idx}")
    }
    fn grid_size_param(&self, axis: usize) -> String {
        format!("    unsigned int grid_size_{axis}")
    }
    fn int_param(&self, name: &str) -> String {
        format!("    int {name}")
    }
    fn extent_param(&self, name: &str) -> String {
        format!("    unsigned int {name}")
    }
    fn scalar_param(&self, name: &str, element: ElementTy) -> String {
        let pty = if matches!(element, ElementTy::I32 | ElementTy::U32) { "int" } else { self.ty() };
        format!("    {pty} {name}")
    }
    fn open_signature(&self, name: &str) -> String {
        format!("{} void {name}(\n", emit::global_qualifier(self.target.target))
    }
    fn params_close(&self) -> &'static str {
        "\n) {\n" // C forbids a trailing comma in the parameter list
    }
    fn cell_prelude(&self, ndim: usize, _n_buffers: u32) -> Vec<String> {
        // CUDA order: thread index, bounds check, absolute coord. strides used
        // to require `_ny_<N>` / `_nz_<N>` lets — under the View ABI they're
        // read directly off `field<N>.strides[..]`, so no per-buffer prelude.
        let dims = ["x", "y", "z"];
        let mut v = Vec::new();
        for aa in 0..ndim {
            let d = dims[aa];
            v.push(format!(
                "    unsigned int _i{aa} = blockIdx.{d} * blockDim.{d} + threadIdx.{d};"
            ));
        }
        let bounds: Vec<String> = (0..ndim).map(|aa| format!("_i{aa} >= grid_size_{aa}")).collect();
        v.push(format!("    if ({}) return;", bounds.join(" || ")));
        for aa in 0..ndim {
            v.push(format!("    int {} = (int)_i{aa} + dom_lo_{aa};", COORD_VARS[aa]));
        }
        v
    }
    fn coord_decl(&self, axis: u8, element: ElementTy) -> String {
        // an integer coord (the substrate) stays integer; a float coord (the
        // macro path) casts to the precision type.
        let cv = COORD_VARS[axis as usize];
        if matches!(element, ElementTy::I32 | ElementTy::U32) {
            format!("    int _coord_{axis} = {cv};")
        } else {
            let ty = self.ty();
            format!("    {ty} _coord_{axis} = ({ty}){cv};")
        }
    }
    fn index_lang(&self) -> emit::IndexLang { emit::IndexLang::Cuda }
    fn skip_scattered_buffer_layout_args(&self) -> bool { true }
    fn flat_index(&self, ndim: u8, buf: u32, comps: &[String]) -> String {
        // shared formula — see `emit::emit_flat_index`.
        let refs: Vec<&str> = comps.iter().map(|s| s.as_str()).collect();
        emit::emit_flat_index(emit::IndexLang::Cuda, ndim, buf, &refs)
    }
    fn render_index_component(&self, e: &ScalarExpr, _coord_vars: &[&str]) -> String {
        // CUDA renders the index generally (data-dependent gathers are allowed);
        // `_coord_N` resolves to the int decl emitted by `coord_decl`.
        let mut s = String::new();
        emit_expr(&mut s, e);
        s
    }
    fn base_read(&self, key: &str, buf: u32, flat: &str) -> String {
        format!("    {} {key} = field{buf}.data[{flat}];", self.ty())
    }
    fn load_at_expr(&self, buf: u32, flat: &str) -> String {
        // every stencil load goes through the View struct's `data` field —
        // mirrors the CPU path's `field{buf}.data[..]`. one method, every emitter.
        format!("field{buf}.data[{flat}]")
    }

    fn smem_prelude(&self, ndim: usize, halo: &[u8], tiled: &[(String, u32)]) -> Vec<String> {
        smem_prelude_cuda(self.ty(), ndim, halo, tiled)
    }
    fn tiled_load_expr(&self, key: &str, halo: &[u8], ndim: u8, comps: &[String]) -> Option<String> {
        // local tile offset per axis: threadIdx + halo + (absolute_comp - cell_coord).
        // the (comp - coord_var) folds to the integer stencil delta at compile time.
        let coord_vars = &COORD_VARS[..ndim as usize];
        let locals: Vec<String> = (0..ndim as usize)
            .map(|a| {
                format!(
                    "((int)threadIdx.{dim} + {h} + (({comp}) - {cv}))",
                    dim = CUDA_TDIM[a], h = halo[a], comp = comps[a], cv = coord_vars[a],
                )
            })
            .collect();
        Some(format!("tile_{key}[{}]", smem_flat_index(&locals)))
    }
    fn tiled_base_read(&self, key: &str, halo: &[u8], ndim: u8) -> Option<String> {
        // the cell-center slot (delta 0): threadIdx + halo on each axis.
        let locals: Vec<String> = (0..ndim as usize)
            .map(|a| format!("((int)threadIdx.{dim} + {h})", dim = CUDA_TDIM[a], h = halo[a]))
            .collect();
        Some(format!("    {} {key} = tile_{key}[{}];", self.ty(), smem_flat_index(&locals)))
    }
    fn render_stmt(&self, stmt: &ScalarStmt) -> String {
        let mut s = String::from("    ");
        emit_stmt(&mut s, stmt);
        s
    }
    fn render_output(&self, expr: &ScalarExpr) -> String {
        let mut s = String::new();
        emit_expr(&mut s, expr);
        s
    }
    fn store(&self, buf: u32, flat: &str, expr: &str) -> String {
        format!("    field{buf}.data[{flat}] = {expr};")
    }
    fn close(&self, _ndim: usize) -> String {
        // a GPU kernel is one thread per cell — no loop to close, just the fn.
        "}\n".to_string()
    }
}

/// emit a scalarized stencil kernel as an `extern "C" __global__` CUDA kernel —
/// the shared driver with the C-family (`CRenderer`) spelling.
pub fn emit_kernel_from_lowering(graph: &Graph, inputs: &KernelEmitInputs) -> KernelDescriptor {
    emit_kernel_render(graph, inputs, &CRenderer { target: inputs.target.clone() })
}

// ----- smem tiling: CUDA spelling -----

/// the CUDA thread/block builtin axis suffixes, indexed by spatial axis.
const CUDA_TDIM: [&str; 3] = ["x", "y", "z"];

/// combine per-axis local tile offsets into one flat smem index:
/// `l0 + l1*__tw0 + l2*(__tw0*__tw1)`. the slab is axis-0-fastest, matching the
/// `compute_strides` gmem convention so the cooperative load coalesces.
fn smem_flat_index(locals: &[String]) -> String {
    let mut terms = vec![locals[0].clone()];
    for a in 1..locals.len() {
        let stride = (0..a).map(|d| format!("__tw{d}")).collect::<Vec<_>>().join(" * ");
        terms.push(format!("{} * ({stride})", locals[a]));
    }
    terms.join(" + ")
}

/// the block-level smem prelude: one `__shared__` slab per tiled field + a
/// cooperative (block + per-axis halo) prefetch from gmem, ending in
/// `__syncthreads()`. each gmem read is CLAMPED to the field's allocated bounds
/// `[lo, lo+extent-1]` (a thin ternary, NVRTC-safe — no `min`/`max`/<math.h>), so
/// a boundary/padding tile cell re-reads a ghost edge instead of going OOB. the
/// tiled fields are assumed CELL-CENTERED with shared `lo`/`extent` (true for the
/// rmhd flux prim + wave-speed inputs); the clamp uses the first field's geometry.
fn smem_prelude_cuda(ty: &str, ndim: usize, halo: &[u8], tiled: &[(String, u32)]) -> Vec<String> {
    assert_eq!(halo.len(), ndim, "smem_prelude: halo rank {} != ndim {ndim}", halo.len());
    assert!(!tiled.is_empty(), "smem_prelude: no tiled fields");
    let buf0 = tiled[0].1; // shared cell-centered geometry for the clamp
    let mut v: Vec<String> = Vec::new();
    v.push("    extern __shared__ unsigned char __smem_raw[];".to_string());
    // per-axis tile widths (block + 2*halo on that axis) and total cell count.
    for a in 0..ndim {
        v.push(format!(
            "    const int __tw{a} = (int)blockDim.{dim} + {two_h};",
            dim = CUDA_TDIM[a], two_h = 2 * halo[a] as i32,
        ));
    }
    let tcells_prod = (0..ndim).map(|a| format!("__tw{a}")).collect::<Vec<_>>().join(" * ");
    v.push(format!("    const int __tcells = {tcells_prod};"));
    // one slab per tiled field, packed by byte offset into the single allocation.
    for (slot, (key, _)) in tiled.iter().enumerate() {
        v.push(format!(
            "    {ty}* tile_{key} = reinterpret_cast<{ty}*>(__smem_raw) + {slot} * __tcells;",
        ));
    }
    // linear thread id + block thread count, for the strided cooperative loop.
    let nthr = (0..ndim).map(|a| format!("(int)blockDim.{}", CUDA_TDIM[a])).collect::<Vec<_>>().join(" * ");
    v.push(format!("    const int __nthr = {nthr};"));
    let tid = (0..ndim)
        .map(|a| {
            if a == 0 {
                "(int)threadIdx.x".to_string()
            } else {
                let bd = (0..a).map(|d| format!("(int)blockDim.{}", CUDA_TDIM[d])).collect::<Vec<_>>().join(" * ");
                format!("(int)threadIdx.{} * {bd}", CUDA_TDIM[a])
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");
    v.push(format!("    const int __tid_lin = {tid};"));
    v.push("    for (int __t = __tid_lin; __t < __tcells; __t += __nthr) {".to_string());
    // decompose the flat tile index into per-axis tile coords.
    v.push("        int __t0 = __t % __tw0;".to_string());
    if ndim >= 2 {
        v.push("        int __t1 = (__t / __tw0) % __tw1;".to_string());
    }
    if ndim >= 3 {
        v.push("        int __t2 = __t / (__tw0 * __tw1);".to_string());
    }
    // global coord for each tile cell, then clamp to the field's allocated range.
    for a in 0..ndim {
        v.push(format!(
            "        int __g{a} = ((int)(blockIdx.{dim} * blockDim.{dim}) + dom_lo_{a}) + __t{a} - {h};",
            dim = CUDA_TDIM[a], h = halo[a],
        ));
    }
    for a in 0..ndim {
        v.push(format!(
            "        int __c{a} = (__g{a} < field{buf0}.lo[{a}]) ? field{buf0}.lo[{a}] \
             : ((__g{a} > field{buf0}.lo[{a}] + field{buf0}.extent[{a}] - 1) \
             ? field{buf0}.lo[{a}] + field{buf0}.extent[{a}] - 1 : __g{a});",
        ));
    }
    // store each tiled field's clamped gmem value into its slab slot.
    for (key, buf) in tiled {
        let flat = (0..ndim)
            .map(|a| format!("(__c{a} - field{buf}.lo[{a}]) * field{buf}.strides[{a}]"))
            .collect::<Vec<_>>()
            .join(" + ");
        v.push(format!("        tile_{key}[__t] = field{buf}.data[{flat}];"));
    }
    v.push("    }".to_string());
    v.push("    __syncthreads();".to_string());
    v
}

/// serialize a `Prepared` to the IR blob `build.rs` embeds per kernel (the inverse
/// of `prepared_from_ir`). keeps serde_json contained to symbi-ir — build.rs and
/// the runtime call these helpers, never the wire format directly.
pub fn prepared_to_ir(prepared: &Prepared) -> String {
    serde_json::to_string(prepared).expect("prepared_to_ir: Prepared is not serializable")
}

/// deserialize a `Prepared` IR blob — the backend-NEUTRAL artifact `build.rs`
/// embeds per kernel. hides serde_json from consumers so they
/// don't take the dep; pair with `render(_, &SomeRenderer)` (the choice of backend
/// is the renderer, never this function).
pub fn prepared_from_ir(ir: &str) -> Prepared {
    serde_json::from_str(ir).expect("prepared_from_ir: malformed Prepared IR blob")
}

/// the RUNTIME render path: deserialize a `Prepared` IR blob and
/// render it to `target` source at `precision`. `target` is a PARAMETER, not baked
/// into the name — adding HIP/Metal is a new match arm here, never a new `*_cuda`
/// function. one blob renders every backend AND both
/// precisions (precision is a render-algebra parameter); the source then
/// feeds the backend's runtime compiler (NVRTC/hiprtc/Metal). the
/// accelerator renders source at runtime rather than shipping pre-rendered text.
pub fn render_from_ir(ir: &str, target: Target, precision: Precision) -> KernelDescriptor {
    let prepared = prepared_from_ir(ir);
    let tcfg = TargetConfig { target, precision };
    match target {
        // CUDA and HIP share the C-family renderer: it already varies header +
        // global-qualifier by `Target` (emit::header / global_qualifier), so HIP
        // drops in as a token-map with zero physics edits.
        Target::Cuda | Target::Hip => render(prepared, &CRenderer { target: tcfg }),
        // Metal (MSL) is f32-only and needs its own renderer (the binding-index ABI
        // + no-`double` capability gate); it lands with that backend.
        Target::Metal => unimplemented!(
            "Metal renderer not yet implemented (docs/design/15 §4); render from IR \
             once MetalRenderer exists"
        ),
    }
}

// the NVRTC-safe identity + combine for a grid reduction at `precision`. min/max
// use the INLINE TERNARY (not fmin/fmax) — fmin/fmax come from <math.h> which
// NVRTC does not include (same class of bug the flux INFINITY fix caught), and the
// ternary matches the CPU carrier's min/max semantics. the identities are plain
// finite literals (no INFINITY macro).
fn reduction_identity_combine(op: ReductionOp, precision: Precision) -> (&'static str, fn(&str, &str) -> String) {
    let f32 = matches!(precision, Precision::F32);
    match op {
        ReductionOp::Add => (if f32 { "0.0f" } else { "0.0" }, |a, b| format!("({a} + {b})")),
        ReductionOp::Mul => (if f32 { "1.0f" } else { "1.0" }, |a, b| format!("({a} * {b})")),
        // sentinels safely beyond any physical value; finite (NVRTC has no INFINITY).
        // min/max MUST propagate NaN so a poisoned cell surfaces at the host dt
        // guard ([[feedback_no_silent_floors]]); the bare ternary `a < b ? a : b`
        // silently drops a NaN operand (NaN compares false). `x != x` is the
        // NVRTC-safe NaN test (no isnan/<math.h>), matching the host fold in
        // `substrate_gpu::host_identity_combine`.
        ReductionOp::Min => (
            if f32 { "1.0e38f" } else { "1.0e308" },
            |a, b| format!("(({a} != {a}) ? {a} : (({b} != {b}) ? {b} : ({a} < {b} ? {a} : {b})))"),
        ),
        ReductionOp::Max => (
            if f32 { "-1.0e38f" } else { "-1.0e308" },
            |a, b| format!("(({a} != {a}) ? {a} : (({b} != {b}) ? {b} : ({a} > {b} ? {a} : {b})))"),
        ),
    }
}

/// the block size for grid reductions — threads per block, also the `sdata` length.
pub const REDUCTION_BLOCK_SIZE: u32 = 256;

/// render a GRID reduction (the Reduce morphism): reduce ONE input
/// field by `op` over the dispatch window, emitting one partial per block (the host
/// folds the partials — only the partials cross, never per-cell). C-family
/// (CUDA/HIP) + precision-generic + NVRTC-renderable; the per-thread value is the
/// field load at the cell (the trivial morphism — a fused per-cell value expression
/// is a later extension). the CPU algebra of the same reduce is the host fold in
/// `substrate_gpu::field_max_reduce` (a host loop).
///
/// ABI (extends the per-cell view ABI with a linear thread map + partials):
///   buf0, total_cells, grid_size_{0..}, dom_lo_{0..}, [ndim>=2] buf_extent_0_{0..},
///   buf_lo_0_{0..}, partials
pub fn render_field_reduction(
    kernel_name: &str,
    ndim: usize,
    precision: Precision,
    op: ReductionOp,
) -> KernelDescriptor {
    assert!((1..=3).contains(&ndim), "render_field_reduction: ndim must be 1..=3 (got {ndim})");
    let ty = precision.c_type();
    let (identity, combine) = reduction_identity_combine(op, precision);
    let mut out = String::new();
    out.push_str(emit::header(Target::Cuda));
    // single View struct definition — same shape as the kernel emitters use.
    out.push_str(&format!(
        "struct __symbi_View {{ {ty}* __restrict__ data; int lo[4]; int strides[4]; int extent[4]; }};\n",
    ));

    // ---- signature ----
    // reduce kernel also takes its input buffer as a `__symbi_View`
    // struct (data + lo + strides bundled), matching the main kernel ABI. the
    // host packs an identical layout.
    let mut params: Vec<String> = vec![
        "    __symbi_View field0".to_string(),
        "    unsigned int total_cells".to_string(),
    ];
    for aa in 0..ndim {
        params.push(format!("    unsigned int grid_size_{aa}"));
    }
    for aa in 0..ndim {
        params.push(format!("    int dom_lo_{aa}"));
    }
    params.push(format!("    {ty}* partials"));
    out.push_str(&format!("{} void {kernel_name}(\n", emit::global_qualifier(Target::Cuda)));
    out.push_str(&params.join(",\n"));
    out.push_str("\n) {\n");

    // ---- shared mem + thread index + per-thread map ----
    out.push_str(&format!("    __shared__ {ty} sdata[{REDUCTION_BLOCK_SIZE}];\n"));
    out.push_str("    unsigned int tid = threadIdx.x;\n");
    out.push_str("    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str(&format!("    {ty} val = {identity};\n"));
    out.push_str("    if (gid < total_cells) {\n");
    // linear gid -> absolute multi-d coord (signed; coord is absolute, view ABI).
    match ndim {
        1 => out.push_str("        int ii = (int)gid + dom_lo_0;\n"),
        2 => {
            out.push_str("        int ii = (int)(gid / grid_size_1) + dom_lo_0;\n");
            out.push_str("        int jj = (int)(gid % grid_size_1) + dom_lo_1;\n");
        }
        _ => {
            out.push_str("        int ii = (int)(gid / (grid_size_1 * grid_size_2)) + dom_lo_0;\n");
            out.push_str("        int jj = (int)((gid / grid_size_2) % grid_size_1) + dom_lo_1;\n");
            out.push_str("        int kk = (int)(gid % grid_size_2) + dom_lo_2;\n");
        }
    }
    // single source of truth: the per-cell base AND the flat-index expression
    // come from `emit::emit_cell_index_base` / `emit::emit_flat_index`. for a
    // reduction every access is at the cell coord so the delta folds to zero —
    // but it still goes through the SAME formula every other emitter uses.
    // both formulas now reference `field0.lo[..]` and `field0.strides[..]` —
    // the View struct passed in by the host.
    out.push_str("    ");
    out.push_str(&emit::emit_cell_index_base(emit::IndexLang::Cuda, ndim as u8, 0, false));
    out.push('\n');
    let comps: &[&str] = &["ii", "jj", "kk"][..ndim];
    let flat = emit::emit_flat_index(emit::IndexLang::Cuda, ndim as u8, 0, comps);
    out.push_str(&format!("        val = field0.data[{flat}];\n"));
    out.push_str("    }\n");

    // ---- block tree reduction -> one partial per block ----
    out.push_str("    sdata[tid] = val;\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n");
    out.push_str(&format!("        if (tid < s) {{ sdata[tid] = {}; }}\n", combine("sdata[tid]", "sdata[tid + s]")));
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    if (tid == 0) partials[blockIdx.x] = sdata[0];\n");
    out.push_str("}\n");

    KernelDescriptor {
        source: out,
        kernel_name: kernel_name.to_string(),
        field_bindings: vec![crate::emit::FieldBinding {
            // the reduction scratch buffer is hand-built, not closed cell-centered
            // vocab — held verbatim as Raw.
            field: symbi_abi::FieldBind::Raw("buf0".into()),
            buffer_index: 0,
            is_output: false,
        }],
        param_names: vec![],
        scalar_is_int: vec![],
        tile_spec: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{Precision, Target};
    use crate::{ElementTy, ElementWiseOp, Graph, Symbol, TensorTy};

    fn cuda_cfg() -> TargetConfig {
        TargetConfig { target: Target::Cuda, precision: Precision::F64 }
    }

    fn scalar_param(g: &mut Graph, name: &str) -> NodeId {
        g.add_param(Symbol::intern(name), TensorTy::scalar(ElementTy::F64), None)
    }

    #[test]
    fn passthrough_1d_emits_global_and_two_buffers() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_from_lowering(&g, &KernelEmitInputs {
            kernel_name:   "pass_1d",
            coalesce_layout: false,            ndim:          1,
            target:        cuda_cfg(),
            field_inputs:  &[("cons_den".into(), "cons.den".into())],
            scalar_params: &[],
            field_writes:  &[("prim_den".into(), "prim.den".into(), cons_den)],
            coord_components: &[], device_preamble: &[], tile_spec: None,
        });
        assert_eq!(desc.kernel_name, "pass_1d");
        assert_eq!(desc.field_bindings.len(), 2);
        assert_eq!(desc.field_bindings[0].field.name(), "cons.den");
        assert!(!desc.field_bindings[0].is_output);
        assert_eq!(desc.field_bindings[1].field.name(), "prim.den");
        assert!(desc.field_bindings[1].is_output);
        // signature shape
        assert!(desc.source.contains("extern \"C\" __global__ void pass_1d("),
            "src:\n{}", desc.source);
        assert!(desc.source.contains("__symbi_View field0"));
        assert!(desc.source.contains("__symbi_View field1"));
        assert!(desc.source.contains("unsigned int grid_size_0"));
        assert!(desc.source.contains("int dom_lo_0"));
        // body: load + store
        assert!(desc.source.contains("double cons_den = field0.data["));
        assert!(desc.source.contains("field1.data[") && desc.source.contains("] = cons_den;"));
        // bounds check
        assert!(desc.source.contains("if (_i0 >= grid_size_0) return;"));
    }

    #[test]
    fn passthrough_2d_includes_stride_extents() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_from_lowering(&g, &KernelEmitInputs {
            kernel_name:   "pass_2d",
            coalesce_layout: false,            ndim:          2,
            target:        cuda_cfg(),
            field_inputs:  &[("cons_den".into(), "cons.den".into())],
            scalar_params: &[],
            field_writes:  &[("prim_den".into(), "prim.den".into(), cons_den)],
            coord_components: &[], device_preamble: &[], tile_spec: None,
        });
        assert!(desc.source.contains("__symbi_View field0"));
        assert!(desc.source.contains("__symbi_View field1"));
        // strides come from `field{N}.strides[..]` — no per-buffer
        // `_ny_<N>` lets needed; the View struct carries pre-multiplied strides.
        assert!(desc.source.contains("field0.strides[0]"));
        assert!(desc.source.contains("int jj = (int)_i1 + dom_lo_1;"));
    }

    #[test]
    fn arithmetic_kernel_lowers_through_scalarize_kernel() {
        // prim_pre = cons_den * 2.0 + cons_nrg
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let cons_nrg = scalar_param(&mut g, "cons_nrg");
        let two = g.add_const(crate::ConstValue::F64(2.0), None);
        let scaled = g.element_wise(ElementWiseOp::Mul, vec![cons_den, two], None);
        let summed = g.element_wise(ElementWiseOp::Add, vec![scaled, cons_nrg], None);
        let desc = emit_kernel_from_lowering(&g, &KernelEmitInputs {
            kernel_name:   "compute_pre_1d",
            coalesce_layout: false,            ndim:          1,
            target:        cuda_cfg(),
            field_inputs:  &[
                ("cons_den".into(), "cons.den".into()),
                ("cons_nrg".into(), "cons.nrg".into()),
            ],
            scalar_params: &[],
            field_writes:  &[("prim_pre".into(), "prim.pre".into(), summed)],
            coord_components: &[], device_preamble: &[], tile_spec: None,
        });
        assert_eq!(desc.field_bindings.len(), 3);
        // load both inputs.
        assert!(desc.source.contains("double cons_den = field0.data["));
        assert!(desc.source.contains("double cons_nrg = field1.data["));
        // store the summed expression; the exact textual form is
        // ((cons_den * 2.0) + cons_nrg).
        assert!(desc.source.contains("field2.data[") && desc.source.contains("] = ((cons_den * 2.0) + cons_nrg);"),
            "src:\n{}", desc.source);
    }

    #[test]
    fn multi_output_writes_emit_one_store_each_in_order() {
        let mut g = Graph::new();
        let cons_mom_0 = scalar_param(&mut g, "cons_mom_0");
        let cons_mom_1 = scalar_param(&mut g, "cons_mom_1");
        let desc = emit_kernel_from_lowering(&g, &KernelEmitInputs {
            kernel_name:   "split_1d",
            coalesce_layout: false,            ndim:          1,
            target:        cuda_cfg(),
            field_inputs:  &[
                ("cons_mom_0".into(), "cons.mom[0]".into()),
                ("cons_mom_1".into(), "cons.mom[1]".into()),
            ],
            scalar_params: &[],
            field_writes:  &[
                ("prim_vel_0".into(), "prim.vel[0]".into(), cons_mom_0),
                ("prim_vel_1".into(), "prim.vel[1]".into(), cons_mom_1),
            ],
            coord_components: &[], device_preamble: &[], tile_spec: None,
        });
        assert_eq!(desc.field_bindings.len(), 4);
        // writes appear in source order.
        let field2_store = desc.source.find("field2.data[").expect("field2 store");
        let field3_store = desc.source.find("field3.data[").expect("field3 store");
        assert!(field2_store < field3_store);
        // each write's RHS is the matching input's IR-key local.
        assert!(desc.source.contains("field2.data[") && desc.source.contains("] = cons_mom_0;"));
        assert!(desc.source.contains("field3.data[") && desc.source.contains("] = cons_mom_1;"));
    }

    #[test]
    fn input_also_written_shares_buffer_and_marks_output() {
        // an in-place update: cons_den is both read and written.
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let one = g.add_const(crate::ConstValue::F64(1.0), None);
        let updated = g.element_wise(ElementWiseOp::Add, vec![cons_den, one], None);
        let desc = emit_kernel_from_lowering(&g, &KernelEmitInputs {
            kernel_name:   "inplace_1d",
            coalesce_layout: false,            ndim:          1,
            target:        cuda_cfg(),
            field_inputs:  &[("cons_den".into(), "cons.den".into())],
            scalar_params: &[],
            field_writes:  &[("cons_den".into(), "cons.den".into(), updated)],
            coord_components: &[], device_preamble: &[], tile_spec: None,
        });
        // one buffer, marked as output.
        assert_eq!(desc.field_bindings.len(), 1);
        assert_eq!(desc.field_bindings[0].field.name(), "cons.den");
        assert!(desc.field_bindings[0].is_output);
        // load + store both reference buf0.
        assert!(desc.source.contains("double cons_den = field0.data["));
        assert!(desc.source.contains("field0.data[") && desc.source.contains("] = (cons_den + 1.0);"));
    }

    #[test]
    fn scalar_param_appears_in_signature_and_passes_through() {
        // out[coord] = a[coord] * dt;  with dt a scalar __global__ arg.
        let mut g = Graph::new();
        let a = scalar_param(&mut g, "a");
        let dt = scalar_param(&mut g, "dt");
        let prod = g.element_wise(ElementWiseOp::Mul, vec![a, dt], None);
        let desc = emit_kernel_from_lowering(&g, &KernelEmitInputs {
            kernel_name:   "scale_1d",
            coalesce_layout: false,            ndim:          1,
            target:        cuda_cfg(),
            field_inputs:  &[("a".into(), "a".into())],
            scalar_params: &["dt".to_string()],
            field_writes:  &[("out".into(), "out".into(), prod)],
            coord_components: &[], device_preamble: &[], tile_spec: None,
        });
        assert_eq!(desc.param_names, vec!["dt".to_string()]);
        // signature includes `double dt`.
        assert!(desc.source.contains("double dt"));
        // body multiplies the loaded `a` by `dt`.
        assert!(desc.source.contains("field1.data[") && desc.source.contains("] = (a * dt);"));
    }

    #[test]
    fn field_reduction_max_3d_is_nvrtc_safe_block_reduce() {
        let desc = render_field_reduction("rmhd_field_max_3d", 3, Precision::F64, ReductionOp::Max);
        let s = &desc.source;
        // a __global__ block-reduce over buf0 with a partials output.
        assert!(s.contains("extern \"C\" __global__ void rmhd_field_max_3d("), "src:\n{s}");
        assert!(s.contains("__symbi_View field0"));
        assert!(s.contains("double* partials"));
        assert!(s.contains("__shared__ double sdata[256];"));
        assert!(s.contains("partials[blockIdx.x] = sdata[0];"));
        // max via INLINE TERNARY, not fmax (NVRTC has no <math.h>); finite identity,
        // no INFINITY macro.
        assert!(s.contains("sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s]"), "src:\n{s}");
        assert!(s.contains("double val = -1.0e308;"));
        assert!(!s.contains("fmax"), "must not use fmax (NVRTC-unsafe): {s}");
        assert!(!s.contains("INFINITY"), "must not use INFINITY (NVRTC-unsafe): {s}");
        // NaN-propagation guard: a poisoned cell must survive the block reduce so it
        // reaches the host dt guard ([[feedback_no_silent_floors]]); the bare ternary
        // drops NaN. `x != x` is the NVRTC-safe NaN test.
        assert!(s.contains("(sdata[tid] != sdata[tid])"), "max must guard NaN via x!=x: {s}");
        // value loaded at the cell via the view index; one buffer binding (input).
        assert!(s.contains("val = field0.data["));
        assert_eq!(desc.field_bindings.len(), 1);
        assert!(!desc.field_bindings[0].is_output);
    }

    #[test]
    fn field_reduction_ops_and_precisions() {
        // add/mul/min/max all render their NVRTC-safe combine + identity, at f32 too.
        let add = render_field_reduction("add1", 1, Precision::F64, ReductionOp::Add);
        assert!(add.source.contains("double val = 0.0;") && add.source.contains("(sdata[tid] + sdata[tid + s])"));
        let mul = render_field_reduction("mul1", 1, Precision::F64, ReductionOp::Mul);
        assert!(mul.source.contains("double val = 1.0;") && mul.source.contains("(sdata[tid] * sdata[tid + s])"));
        let min = render_field_reduction("min1", 1, Precision::F64, ReductionOp::Min);
        assert!(min.source.contains("double val = 1.0e308;") && min.source.contains("< sdata[tid + s] ?"));
        let maxf = render_field_reduction("maxf1", 1, Precision::F32, ReductionOp::Max);
        assert!(maxf.source.contains("float val = -1.0e38f;"), "src:\n{}", maxf.source);
        assert!(maxf.source.contains("__symbi_View field0") && maxf.source.contains("float* partials"));
        // 1D: no buf_extent params, single buf_lo.
        assert!(!maxf.source.contains("buf_extent"));
        assert!(maxf.source.contains("__symbi_View field0"));
    }
}
