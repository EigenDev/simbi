// =============================================================================
// emit_kernel.rs
//
// chalkboard-pipeline kernel emitter. given a scalarized
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
// in `symbi_ir::emit`. sycl/fpga would slot in as sibling emitters
// (the IR is target-agnostic; only the source text shifts).
//
// the macro layer wraps each ndim in {1,2,3} in its own invocation to
// populate per-ndim KernelDescriptors.
//
// usage:
//   let desc = emit_kernel_from_lowering(&graph, &KernelEmitInputs {
//       kernel_name: "iso_c2p_1d",
//       ndim: 1,
//       target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
//       field_inputs: &[(("cons_den".into(), "cons.den".into()))],
//       scalar_params: &[],
//       field_writes: &[KernelWrite::new("prim_rho", "prim.rho", rho_node)],
//   });
// =============================================================================

#[cfg(test)]
use crate::NodeId;
use crate::backends::cuda::{emit_expr, emit_stmt};
use crate::backends::render::{COORD_VARS, KernelRenderer, Prepared, emit_kernel_render, render};
use crate::emit::{self, KernelDescriptor, Precision, ReductionOp, Target, TargetConfig};
use crate::passes::scalarize::{ScalarExpr, ScalarStmt};
use crate::{ElementTy, Graph};
use symbi_abi::FieldBind;

/// inputs to `emit_kernel_from_lowering`. the order of `field_inputs`
/// fixes the buffer indices for inputs; write-only fields are appended
/// after. dispatch on the macro side must match this order when packing
/// `__buf_extents` / `__buf_los` / `__field_ptrs`.
#[derive(Debug, Clone)]
pub struct KernelEmitInputs<'a> {
    pub kernel_name: &'a str,
    pub ndim: u8,
    pub target: TargetConfig,
    /// whether all of this kernel's buffers share one allocated layout, so the
    /// cell index can be computed once and shared across reads. the producer sets
    /// this (it knows the kernel's buffer topology); the IR stays domain-agnostic.
    /// true for single-layout
    /// cell-centered kernels (c2p, wave-speed maps, pure-hydro face flux); false
    /// for staggered mhd face-flux (edge efield) and amr prolong/restrict (two grids).
    pub coalesce_layout: bool,
    /// (IR-side synthesized key, born-typed runtime binding). the IR key
    /// matches a Param node in the graph; the FieldBind is what ends up in
    /// `FieldBinding::field`, which the dispatch side consumes verbatim.
    pub field_inputs: &'a [(String, FieldBind)],
    /// IR-side param names that stay as scalar __global__ args (user
    /// scalars: dt, gamma, etc.), passed by value.
    pub scalar_params: &'a [String],
    /// (write key, runtime path, RHS NodeId). each entry produces one
    /// buffer store; if a write's runtime path matches an input, the
    /// buffer is shared and marked is_output.
    pub field_writes: &'a [crate::gv::KernelWrite],
    /// kernel-coord component axes referenced by the body. each entry
    /// gets a `double _coord_N = (double)<thread-axis>;` line emitted
    /// after the thread-index prelude. body Param references
    /// `_coord_0`/`_coord_1`/`_coord_2` resolve there.
    pub coord_components: &'a [u8],
    /// device-function definitions to include in the kernel
    /// source, ahead of the `__global__` block. emitted in order — the
    /// caller (kernel macro) is responsible for topological order
    /// (callees before callers) and de-duplication. each entry is a
    /// complete `__device__ inline RET name(...) { ... }` string.
    pub device_preamble: &'a [String],
    /// the kernel's shared-memory tile intent for stencil
    /// kernels (halo + stencil-read field keys). `None` = gmem loads only.
    /// the CUDA emitter cooperatively prefetches the (block + halo)
    /// region for these fields into `__shared__`; the CPU emitter ignores it
    /// (it cache-tiles every kernel unconditionally). inferred for stencil
    /// kernels via `infer_tile_spec`; threaded here from the `GvKernel`.
    pub tile_spec: Option<&'a crate::gv::TileSpec>,
}

/// the C-family backend spelling for the shared kernel driver (`emit_render`):
/// CUDA and HIP. produces an `extern "C" __global__` kernel over raw
/// `<precision>*` buffers (the per-cell view ABI shape); the header + global qualifier
/// vary by `target.target` (`emit::header` / `global_qualifier`), so HIP is a
/// pure token-map with zero physics edits. Metal (msl: buffer-index ABI, f32-only)
/// takes its own renderer.
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
        // ptr + lo + pre-multiplied strides. emit the struct typedef once in the
        // preamble so the kernel signature can spell `__symbi_View field0`.
        // the host side packs a matching pod into the kernel arg buffer (see
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
        // every buffer is a View struct (data + lo + strides); the layout rides
        // inside it, in place of scattered `buf_lo_*` / `buf_extent_*` scalar args
        // (see `skip_scattered_buffer_layout_args`).
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
        let pty = if matches!(element, ElementTy::I32 | ElementTy::U32) {
            "int"
        } else {
            self.ty()
        };
        format!("    {pty} {name}")
    }
    fn open_signature(&self, name: &str) -> String {
        format!(
            "{} void {name}(\n",
            emit::global_qualifier(self.target.target)
        )
    }
    fn params_close(&self) -> &'static str {
        "\n) {\n" // C forbids a trailing comma in the parameter list
    }
    fn cell_prelude(&self, ndim: usize, _n_buffers: u32) -> Vec<String> {
        // CUDA order: thread index, bounds check, absolute coord. under the View
        // ABI strides are read straight off `field<N>.strides[..]`, so the prelude
        // is per-cell only.
        let dims = ["x", "y", "z"];
        let mut v = Vec::new();
        for aa in 0..ndim {
            let d = dims[aa];
            v.push(format!(
                "    unsigned int _i{aa} = blockIdx.{d} * blockDim.{d} + threadIdx.{d};"
            ));
        }
        let bounds: Vec<String> = (0..ndim)
            .map(|aa| format!("_i{aa} >= grid_size_{aa}"))
            .collect();
        v.push(format!("    if ({}) return;", bounds.join(" || ")));
        for aa in 0..ndim {
            v.push(format!(
                "    int {} = (int)_i{aa} + dom_lo_{aa};",
                COORD_VARS[aa]
            ));
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
    fn index_lang(&self) -> emit::IndexLang {
        emit::IndexLang::Cuda
    }
    fn skip_scattered_buffer_layout_args(&self) -> bool {
        true
    }
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
    fn tiled_load_expr(
        &self,
        key: &str,
        halo: &[u8],
        ndim: u8,
        comps: &[String],
    ) -> Option<String> {
        // local tile offset per axis: threadIdx + halo + (absolute_comp - cell_coord).
        // the (comp - coord_var) folds to the integer stencil delta at compile time.
        let coord_vars = &COORD_VARS[..ndim as usize];
        let locals: Vec<String> = (0..ndim as usize)
            .map(|a| {
                format!(
                    "((int)threadIdx.{dim} + {h} + (({comp}) - {cv}))",
                    dim = CUDA_TDIM[a],
                    h = halo[a],
                    comp = comps[a],
                    cv = coord_vars[a],
                )
            })
            .collect();
        Some(format!("tile_{key}[{}]", smem_flat_index(&locals)))
    }
    fn tiled_base_read(&self, key: &str, halo: &[u8], ndim: u8) -> Option<String> {
        // the cell-center slot (delta 0): threadIdx + halo on each axis.
        let locals: Vec<String> = (0..ndim as usize)
            .map(|a| {
                format!(
                    "((int)threadIdx.{dim} + {h})",
                    dim = CUDA_TDIM[a],
                    h = halo[a]
                )
            })
            .collect();
        Some(format!(
            "    {} {key} = tile_{key}[{}];",
            self.ty(),
            smem_flat_index(&locals)
        ))
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
        // a GPU kernel is one thread per cell, so closing the fn body is the
        // whole of it.
        "}\n".to_string()
    }
}

/// emit a scalarized stencil kernel as an `extern "C" __global__` CUDA kernel —
/// the shared driver with the C-family (`CRenderer`) spelling.
pub fn emit_kernel_from_lowering(graph: &Graph, inputs: &KernelEmitInputs) -> KernelDescriptor {
    emit_kernel_render(
        graph,
        inputs,
        &CRenderer {
            target: inputs.target.clone(),
        },
    )
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
        let stride = (0..a)
            .map(|d| format!("__tw{d}"))
            .collect::<Vec<_>>()
            .join(" * ");
        terms.push(format!("{} * ({stride})", locals[a]));
    }
    terms.join(" + ")
}

/// the block-level smem prelude: one `__shared__` slab per tiled field + a
/// cooperative (block + per-axis halo) prefetch from gmem, ending in
/// `__syncthreads()`. each gmem read is clamped to the field's allocated bounds
/// `[lo, lo+extent-1]` (a thin ternary; `min`/`max` live in <math.h>, which NVRTC
/// leaves out), so a boundary/padding tile cell re-reads a ghost
/// edge, staying in bounds. the tiled fields are cell-centered with shared
/// `lo`/`extent` (true for the
/// rmhd flux prim + wave-speed inputs); the clamp uses the first field's geometry.
fn smem_prelude_cuda(ty: &str, ndim: usize, halo: &[u8], tiled: &[(String, u32)]) -> Vec<String> {
    assert_eq!(
        halo.len(),
        ndim,
        "smem_prelude: halo rank {} != ndim {ndim}",
        halo.len()
    );
    assert!(!tiled.is_empty(), "smem_prelude: no tiled fields");
    let buf0 = tiled[0].1; // shared cell-centered geometry for the clamp
    let mut v: Vec<String> = Vec::new();
    v.push("    extern __shared__ unsigned char __smem_raw[];".to_string());
    // per-axis tile widths (block + 2*halo on that axis) and total cell count.
    for a in 0..ndim {
        v.push(format!(
            "    const int __tw{a} = (int)blockDim.{dim} + {two_h};",
            dim = CUDA_TDIM[a],
            two_h = 2 * halo[a] as i32,
        ));
    }
    let tcells_prod = (0..ndim)
        .map(|a| format!("__tw{a}"))
        .collect::<Vec<_>>()
        .join(" * ");
    v.push(format!("    const int __tcells = {tcells_prod};"));
    // one slab per tiled field, packed by byte offset into the single allocation.
    for (slot, (key, _)) in tiled.iter().enumerate() {
        v.push(format!(
            "    {ty}* tile_{key} = reinterpret_cast<{ty}*>(__smem_raw) + {slot} * __tcells;",
        ));
    }
    // linear thread id + block thread count, for the strided cooperative loop.
    let nthr = (0..ndim)
        .map(|a| format!("(int)blockDim.{}", CUDA_TDIM[a]))
        .collect::<Vec<_>>()
        .join(" * ");
    v.push(format!("    const int __nthr = {nthr};"));
    let tid = (0..ndim)
        .map(|a| {
            if a == 0 {
                "(int)threadIdx.x".to_string()
            } else {
                let bd = (0..a)
                    .map(|d| format!("(int)blockDim.{}", CUDA_TDIM[d]))
                    .collect::<Vec<_>>()
                    .join(" * ");
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
        v.push(format!(
            "        tile_{key}[__t] = field{buf}.data[{flat}];"
        ));
    }
    v.push("    }".to_string());
    v.push("    __syncthreads();".to_string());
    v
}

/// serialize a `Prepared` to the IR blob `build.rs` embeds per kernel (the inverse
/// of `prepared_from_ir`). keeps serde_json contained to symbi-ir: build.rs and
/// the runtime go through these helpers, so the wire format stays internal.
pub fn prepared_to_ir(prepared: &Prepared) -> String {
    super::wire::serialize(prepared).expect("prepared_to_ir: Prepared is not serializable")
}

/// deserialize the flat, versioned compiler-generated IR representation.
pub(crate) fn deserialize_prepared(ir: &str) -> Result<Prepared, serde_json::Error> {
    super::wire::deserialize(ir)
}

/// deserialize a `Prepared` IR blob — the backend-neutral artifact `build.rs`
/// embeds per kernel. hides serde_json from consumers so the dep stays here;
/// pair with `render(_, &SomeRenderer)`, where the renderer alone picks the
/// backend.
pub fn prepared_from_ir(ir: &str) -> Prepared {
    deserialize_prepared(ir).expect("prepared_from_ir: malformed Prepared IR blob")
}

/// the runtime render path: deserialize a `Prepared` IR blob and
/// render it to `target` source at `precision`. `target` is a parameter threaded
/// through the render call, so adding HIP/Metal is a new match arm here. one blob
/// renders every backend and both
/// precisions (precision is a render-algebra parameter); the source then
/// feeds the backend's runtime compiler (NVRTC/hiprtc/Metal). the
/// accelerator renders source at runtime.
pub fn render_from_ir(ir: &str, target: Target, precision: Precision) -> KernelDescriptor {
    let prepared = prepared_from_ir(ir);
    let tcfg = TargetConfig { target, precision };
    match target {
        // CUDA and HIP share the C-family renderer: it varies header +
        // global-qualifier by `Target` (emit::header / global_qualifier), so HIP
        // drops in as a token-map with zero physics edits.
        Target::Cuda | Target::Hip => render(prepared, &CRenderer { target: tcfg }),
        // Metal (msl) is f32-only and takes its own renderer (the binding-index ABI
        // + the f32 capability gate); it lands with that backend.
        Target::Metal => unimplemented!(
            "Metal renderer not implemented; render from IR once \
             MetalRenderer exists"
        ),
    }
}

// the NVRTC-safe identity + combine for a grid reduction at `precision`. min/max
// use the inline ternary: fmin/fmax live in <math.h>, which NVRTC leaves out (the
// same gap an infinity literal in a flux kernel falls into), and the ternary matches
// the CPU carrier's min/max semantics. the identities are plain finite literals,
// keeping the infinity macro out of the emitted source.
fn reduction_identity_combine(
    op: ReductionOp,
    precision: Precision,
) -> (&'static str, fn(&str, &str) -> String) {
    let f32 = matches!(precision, Precision::F32);
    match op {
        ReductionOp::Add => (if f32 { "0.0f" } else { "0.0" }, |a, b| {
            format!("({a} + {b})")
        }),
        ReductionOp::Mul => (if f32 { "1.0f" } else { "1.0" }, |a, b| {
            format!("({a} * {b})")
        }),
        // sentinels safely beyond any physical value, and finite, since NVRTC leaves
        // the infinity macro out. min/max propagate NaN so a poisoned cell surfaces at
        // the host dt guard ([[feedback_no_silent_floors]]); the bare ternary
        // `a < b ? a : b` silently drops a NaN operand (NaN compares false). `x != x`
        // is the NVRTC-safe NaN test (isnan lives in <math.h>), matching the host fold
        // in `substrate_gpu::host_identity_combine`.
        ReductionOp::Min => (if f32 { "1.0e38f" } else { "1.0e308" }, |a, b| {
            format!("(({a} != {a}) ? {a} : (({b} != {b}) ? {b} : ({a} < {b} ? {a} : {b})))")
        }),
        ReductionOp::Max => (if f32 { "-1.0e38f" } else { "-1.0e308" }, |a, b| {
            format!("(({a} != {a}) ? {a} : (({b} != {b}) ? {b} : ({a} > {b} ? {a} : {b})))")
        }),
    }
}

/// the block size for grid reductions — threads per block, also the `sdata` length.
pub const REDUCTION_BLOCK_SIZE: u32 = 256;

/// shared-memory budget a privatized segmented reduction may claim, in bytes. below the
/// 64 kB an MI250X compute unit carries so several blocks stay resident per unit —
/// claiming the whole allocation would serialize the grid at one block per unit and cost
/// more than the accumulator contention it saves.
pub const SEGMENTED_LDS_BUDGET_BYTES: usize = 32 * 1024;

/// blocks a segmented reduction launches, regardless of grid size. the kernel walks the
/// domain with a grid-stride loop, so this bounds the accumulator contention and makes the
/// launch shape independent of the resolution.
pub const SEGMENTED_MAX_BLOCKS: u32 = 1024;

/// the segment index marking a cell held out of the reduction entirely — a cell
/// covered by finer data, inside an immersed body's mask, or otherwise outside the physical
/// gas. an index past the last segment carries the other meaning: the cell was to be reduced
/// and fell outside the declared bin edges. that second case alone is a shortfall of the
/// binning and that second case alone is counted; conflating the two would report a body's
/// footprint as under-coverage.
/// the marker for a cell excluded from a reduction, as an offset past the last bucket. a census
/// writes `n_segments + EXCLUDED_OFFSET`; the reduction treats anything strictly above
/// `n_segments` as excluded and `n_segments` itself as a cell that fell outside the bin edges.
///
/// an offset keeps every marker a small integer, which the carrier requires: the
/// segment travels on the scalar carrier (a generated kernel has one scalar type for all of its
/// buffers), and a `u32::MAX` marker rounds to 2^32 in f32 and stops comparing equal, which
/// would silently reclassify every excluded cell.
pub const SEGMENT_EXCLUDED_OFFSET: u32 = 1;

/// does a segmented reduction of this shape privatize its accumulators into shared memory,
/// or accumulate straight into the global output? one home for the policy so the launcher
/// and the setup-time report agree.
pub fn segmented_privatizes(n_segments: usize, n_values: usize, precision: Precision) -> bool {
    let width = match precision {
        Precision::F64 => 8,
        Precision::F32 => 4,
    };
    n_segments.saturating_mul(n_values).saturating_mul(width) <= SEGMENTED_LDS_BUDGET_BYTES
}

/// the NVRTC-safe read-modify-write of one accumulator slot. `Add` has a native
/// `atomicAdd`; floating-point `Min`/`Max` compare-and-swap on the bit pattern until the
/// slot holds the combined value. every form is a device-wide atomic, so the combine order
/// is unspecified: `Min`/`Max` are order-agnostic and land on the same answer regardless,
/// while `Add` reassociates, so a summing segmented reduction is reproducible only up to
/// that reassociation.
fn segmented_atomic_helpers(op: ReductionOp, precision: Precision) -> String {
    let f32 = matches!(precision, Precision::F32);
    let ty = precision.c_type();
    if matches!(op, ReductionOp::Add) {
        return format!(
            "__device__ inline void __symbi_seg_accum({ty}* addr, {ty} v) {{ atomicAdd(addr, v); }}\n"
        );
    }
    let (bits_ty, to_bits, from_bits) = if f32 {
        ("unsigned int", "__float_as_uint", "__uint_as_float")
    } else {
        (
            "unsigned long long",
            "__double_as_longlong",
            "__longlong_as_double",
        )
    };
    let (_, combine) = reduction_identity_combine(op, precision);
    let combined = combine("cur", "v");
    // the cas retries until the slot's bits are the ones this thread computed from the value
    // it read, so every racing update is folded in. the combine is the same NaN-propagating
    // ternary the block reduction and the host fold use.
    format!(
        "__device__ inline void __symbi_seg_accum({ty}* addr, {ty} v) {{\n\
         \x20   {bits_ty}* p = ({bits_ty}*)addr;\n\
         \x20   {bits_ty} old = *p, assumed;\n\
         \x20   do {{\n\
         \x20       assumed = old;\n\
         \x20       {ty} cur = {from_bits}(assumed);\n\
         \x20       {ty} nxt = {combined};\n\
         \x20       old = atomicCAS(p, assumed, {to_bits}(nxt));\n\
         \x20   }} while (assumed != old);\n\
         }}\n"
    )
}

/// render a segmented reduction: the scatter-add half of a binned reduction (a census).
/// each cell carries a destination bucket in `segment`, and every one of the `n_values`
/// value fields is combined by `op` into that bucket's accumulator, giving
/// `n_segments * n_values` outputs from one pass over the domain. the data-dependent
/// destination puts this outside traced pointwise code, so it sits beside the
/// generated kernels as its own morphism, exactly as the whole-field reduction does.
///
/// two accumulation strategies, chosen by `segmented_privatizes`:
///   - privatized: a block-local shared accumulator absorbs the block's cells, then each
///     slot is folded into the global output once per block. contention scales with the
///     block count, which is far below the cell count.
///   - direct: an accumulator larger than the shared budget keeps cells combining straight
///     into the global output.
/// both are device-wide atomic at the final combine, so `Add` reassociates run to run.
///
/// a cell whose segment index is at or beyond `n_segments` lies outside the binning. it is
/// dropped and counted, because a silently under-covering binning is indistinguishable from
/// a physics result.
///
/// ABI: value views `field0..field{n_values-1}`, segment view `field{n_values}`,
/// total_cells, grid_size_{0..}, dom_lo_{0..}, out, dropped
pub fn render_field_segmented_reduction(
    kernel_name: &str,
    ndim: usize,
    precision: Precision,
    op: ReductionOp,
    n_values: usize,
    n_segments: usize,
) -> KernelDescriptor {
    assert!(
        (1..=3).contains(&ndim),
        "render_field_segmented_reduction: ndim must be 1..=3 (got {ndim})"
    );
    assert!(
        n_values >= 1,
        "render_field_segmented_reduction: needs at least one value field"
    );
    assert!(
        n_segments >= 1,
        "render_field_segmented_reduction: needs at least one segment"
    );
    // a product over bins overflows to zero or infinity on any realistic cell count and
    // carries no meaning as a census statistic, so the emitter refuses it loudly.
    assert!(
        !matches!(op, ReductionOp::Mul),
        "render_field_segmented_reduction: Mul is not a meaningful segmented reduction"
    );

    let ty = precision.c_type();
    let (identity, _) = reduction_identity_combine(op, precision);
    let privatize = segmented_privatizes(n_segments, n_values, precision);
    let n_slots = n_segments * n_values;
    let seg_buf = n_values;

    let mut out = String::new();
    out.push_str(emit::header(Target::Cuda));
    out.push_str(&format!(
        "struct __symbi_View {{ {ty}* __restrict__ data; int lo[4]; int strides[4]; int extent[4]; }};\n",
    ));
    // the segment view carries the same host-packed layout as a value view; the element
    // type is the one difference, and the index formulas read lo/strides alone.
    out.push_str("");
    out.push_str(&segmented_atomic_helpers(op, precision));

    // ---- signature ----
    let mut params: Vec<String> = (0..n_values)
        .map(|v| format!("    __symbi_View field{v}"))
        .collect();
    params.push(format!("    __symbi_View field{seg_buf}"));
    params.push("    unsigned int total_cells".to_string());
    for aa in 0..ndim {
        params.push(format!("    unsigned int grid_size_{aa}"));
    }
    for aa in 0..ndim {
        params.push(format!("    int dom_lo_{aa}"));
    }
    params.push(format!("    {ty}* out"));
    params.push("    unsigned long long* dropped".to_string());
    out.push_str(&format!(
        "{} void {kernel_name}(\n",
        emit::global_qualifier(Target::Cuda)
    ));
    out.push_str(&params.join(",\n"));
    out.push_str("\n) {\n");

    // ---- accumulator setup ----
    if privatize {
        out.push_str(&format!("    __shared__ {ty} acc[{n_slots}];\n"));
        out.push_str(&format!(
            "    for (unsigned int i = threadIdx.x; i < {n_slots}; i += blockDim.x) acc[i] = {identity};\n"
        ));
        out.push_str("    __syncthreads();\n");
    }
    out.push_str("    unsigned long long n_dropped = 0;\n");

    // ---- grid-stride walk ----
    // the block count is fixed by the launcher, so a thread visits a strided run of cells
    // and the launch shape stays independent of the resolution.
    out.push_str(
        "    for (unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total_cells; gid += blockDim.x * gridDim.x) {\n",
    );
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
    let comps: &[&str] = &["ii", "jj", "kk"][..ndim];
    // every buffer's cell index goes through the same formula the pointwise emitters use.
    for buf in 0..=seg_buf {
        out.push_str("    ");
        out.push_str(&emit::emit_cell_index_base(
            emit::IndexLang::Cuda,
            ndim as u8,
            buf as u32,
            false,
        ));
        out.push('\n');
    }
    let seg_flat = emit::emit_flat_index(emit::IndexLang::Cuda, ndim as u8, seg_buf as u32, comps);
    // the segment rides the scalar carrier, like every other kernel buffer: a generated kernel is
    // `fn k<S: Scalar>`, one type for all of its buffers, so a bucket index arrives on that same
    // carrier. every marker is therefore a small non-negative integer, exact in f32 and
    // f64 alike — bucket in `[0, n)`, `n` for a cell that fell outside the bin edges, and anything
    // above `n` for a cell excluded from the reduction entirely. every sentinel is carrier-width
    // independent; a `u32::MAX` marker would silently break once the carrier is f32.
    out.push_str(&format!(
        "        unsigned int seg = (unsigned int) field{seg_buf}.data[{seg_flat}];\n"
    ));
    // a cell excluded (covered by finer data, inside a body mask, a ghost) is skipped silently; a
    // cell that was to be reduced and fell outside the declared edges is a shortfall and counted.
    out.push_str(&format!("        if (seg > {n_segments}u) continue;\n"));
    out.push_str(&format!(
        "        if (seg == {n_segments}u) {{ n_dropped += 1ull; continue; }}\n"
    ));
    for v in 0..n_values {
        let flat = emit::emit_flat_index(emit::IndexLang::Cuda, ndim as u8, v as u32, comps);
        let slot = format!("seg * {n_values}u + {v}u");
        if privatize {
            out.push_str(&format!(
                "        __symbi_seg_accum(&acc[{slot}], field{v}.data[{flat}]);\n"
            ));
        } else {
            out.push_str(&format!(
                "        __symbi_seg_accum(&out[{slot}], field{v}.data[{flat}]);\n"
            ));
        }
    }
    out.push_str("    }\n");

    // ---- fold the block's private accumulator into the global output ----
    if privatize {
        out.push_str("    __syncthreads();\n");
        out.push_str(&format!(
            "    for (unsigned int i = threadIdx.x; i < {n_slots}; i += blockDim.x) __symbi_seg_accum(&out[i], acc[i]);\n"
        ));
    }
    out.push_str("    if (n_dropped) atomicAdd(dropped, n_dropped);\n");
    out.push_str("}\n");

    KernelDescriptor {
        source: out,
        kernel_name: kernel_name.to_string(),
        field_bindings: (0..=seg_buf)
            .map(|b| crate::emit::FieldBinding {
                // the census scratch buffers are hand-built, outside the closed cell-centered
                // vocab — held verbatim as Raw, as the whole-field reduction's input is.
                field: symbi_abi::FieldBind::Raw(format!("buf{b}").into()),
                buffer_index: b as u32,
                is_output: false,
            })
            .collect(),
        param_names: vec![],
        scalar_is_int: vec![],
        tile_spec: None,
    }
}

/// render a grid reduction (the Reduce morphism): reduce one input
/// field by `op` over the dispatch window, emitting one partial per block (the host
/// folds the partials; the partials alone cross the bus). C-family
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
    assert!(
        (1..=3).contains(&ndim),
        "render_field_reduction: ndim must be 1..=3 (got {ndim})"
    );
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
    out.push_str(&format!(
        "{} void {kernel_name}(\n",
        emit::global_qualifier(Target::Cuda)
    ));
    out.push_str(&params.join(",\n"));
    out.push_str("\n) {\n");

    // ---- shared mem + thread index + per-thread map ----
    out.push_str(&format!(
        "    __shared__ {ty} sdata[{REDUCTION_BLOCK_SIZE}];\n"
    ));
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
    // single source of truth: the per-cell base and the flat-index expression
    // come from `emit::emit_cell_index_base` / `emit::emit_flat_index`. for a
    // reduction every access is at the cell coord so the delta folds to zero,
    // and it still goes through the same formula every other emitter uses.
    // both formulas reference `field0.lo[..]` and `field0.strides[..]` —
    // the View struct passed in by the host.
    out.push_str("    ");
    out.push_str(&emit::emit_cell_index_base(
        emit::IndexLang::Cuda,
        ndim as u8,
        0,
        false,
    ));
    out.push('\n');
    let comps: &[&str] = &["ii", "jj", "kk"][..ndim];
    let flat = emit::emit_flat_index(emit::IndexLang::Cuda, ndim as u8, 0, comps);
    out.push_str(&format!("        val = field0.data[{flat}];\n"));
    out.push_str("    }\n");

    // ---- block tree reduction -> one partial per block ----
    out.push_str("    sdata[tid] = val;\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n");
    out.push_str(&format!(
        "        if (tid < s) {{ sdata[tid] = {}; }}\n",
        combine("sdata[tid]", "sdata[tid + s]")
    ));
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    if (tid == 0) partials[blockIdx.x] = sdata[0];\n");
    out.push_str("}\n");

    KernelDescriptor {
        source: out,
        kernel_name: kernel_name.to_string(),
        field_bindings: vec![crate::emit::FieldBinding {
            // the reduction scratch buffer is hand-built, outside the closed cell-centered
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
        TargetConfig {
            target: Target::Cuda,
            precision: Precision::F64,
        }
    }

    fn scalar_param(g: &mut Graph, name: &str) -> NodeId {
        g.add_param(Symbol::intern(name), TensorTy::scalar(ElementTy::F64), None)
    }

    #[test]
    fn prepared_ir_accepts_a_scalar_tree_past_jsons_default_depth() {
        use crate::passes::scalarize::{KernelScalarized, ScalarExpr};
        use crate::{BinaryKind, ConstValue};
        use std::collections::BTreeMap;

        let mut expression = ScalarExpr::Const(ConstValue::F64(1.0));
        for _ in 0..4096 {
            expression = ScalarExpr::BinOp(
                BinaryKind::Add,
                Box::new(expression),
                Box::new(ScalarExpr::Const(ConstValue::F64(1.0))),
            );
        }
        let prepared = Prepared {
            kernel_name: "deep_scalar_tree".into(),
            ndim: 1,
            scalarized: KernelScalarized {
                params: vec![],
                body: vec![],
                outputs: vec![expression],
            },
            bindings: vec![],
            field_inputs: vec![],
            field_writes: vec![],
            scalar_params: vec![],
            coord_components: vec![],
            device_preamble: vec![],
            param_elem: BTreeMap::new(),
            tile_spec: None,
            coalesce_layout: false,
            output_support: None,
        };

        let ir = prepared_to_ir(&prepared);
        assert!(ir.contains("\"version\":1"));
        let decoded = prepared_from_ir(&ir);
        assert_eq!(decoded.kernel_name, "deep_scalar_tree");
        assert_eq!(decoded.scalarized.outputs, prepared.scalarized.outputs);
    }

    #[test]
    fn prepared_ir_rejects_an_invalid_flat_expression_index() {
        let mut g = Graph::new();
        let value = scalar_param(&mut g, "value");
        let prepared = super::super::render::prepare(
            &g,
            &KernelEmitInputs {
                kernel_name: "invalid_flat_index",
                coalesce_layout: false,
                ndim: 1,
                target: cuda_cfg(),
                field_inputs: &[],
                scalar_params: &["value".into()],
                field_writes: &[crate::gv::KernelWrite::new("out", "prim.rho", value)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        let mut wire: serde_json::Value = serde_json::from_str(&prepared_to_ir(&prepared)).unwrap();
        wire["scalarized"]["outputs"][0] = serde_json::json!(u32::MAX);
        let error = deserialize_prepared(&serde_json::to_string(&wire).unwrap()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid or forward scalar expression index")
        );
    }

    #[test]
    fn passthrough_1d_emits_global_and_two_buffers() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_from_lowering(
            &g,
            &KernelEmitInputs {
                kernel_name: "pass_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cuda_cfg(),
                field_inputs: &[("cons_den".into(), "cons.den".into())],
                scalar_params: &[],
                field_writes: &[crate::gv::KernelWrite::new(
                    "prim_den", "prim.den", cons_den,
                )],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert_eq!(desc.kernel_name, "pass_1d");
        assert_eq!(desc.field_bindings.len(), 2);
        assert_eq!(desc.field_bindings[0].field.name(), "cons.den");
        assert!(!desc.field_bindings[0].is_output);
        assert_eq!(desc.field_bindings[1].field.name(), "prim.den");
        assert!(desc.field_bindings[1].is_output);
        // signature shape
        let signature = format!(
            "{} void pass_1d(",
            crate::emit::global_qualifier(Target::Cuda)
        );
        assert!(desc.source.contains(&signature), "src:\n{}", desc.source);
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
        let desc = emit_kernel_from_lowering(
            &g,
            &KernelEmitInputs {
                kernel_name: "pass_2d",
                coalesce_layout: false,
                ndim: 2,
                target: cuda_cfg(),
                field_inputs: &[("cons_den".into(), "cons.den".into())],
                scalar_params: &[],
                field_writes: &[crate::gv::KernelWrite::new(
                    "prim_den", "prim.den", cons_den,
                )],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert!(desc.source.contains("__symbi_View field0"));
        assert!(desc.source.contains("__symbi_View field1"));
        // the View struct carries pre-multiplied strides, so the body reads
        // `field{N}.strides[..]` directly.
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
        let desc = emit_kernel_from_lowering(
            &g,
            &KernelEmitInputs {
                kernel_name: "compute_pre_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cuda_cfg(),
                field_inputs: &[
                    ("cons_den".into(), "cons.den".into()),
                    ("cons_nrg".into(), "cons.nrg".into()),
                ],
                scalar_params: &[],
                field_writes: &[crate::gv::KernelWrite::new("prim_pre", "prim.pre", summed)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert_eq!(desc.field_bindings.len(), 3);
        // load both inputs.
        assert!(desc.source.contains("double cons_den = field0.data["));
        assert!(desc.source.contains("double cons_nrg = field1.data["));
        // store the summed expression; the exact textual form is
        // ((cons_den * 2.0) + cons_nrg).
        assert!(
            desc.source.contains("field2.data[")
                && desc.source.contains("] = ((cons_den * 2.0) + cons_nrg);"),
            "src:\n{}",
            desc.source
        );
    }

    #[test]
    fn multi_output_writes_emit_one_store_each_in_order() {
        let mut g = Graph::new();
        let cons_mom_0 = scalar_param(&mut g, "cons_mom_0");
        let cons_mom_1 = scalar_param(&mut g, "cons_mom_1");
        let desc = emit_kernel_from_lowering(
            &g,
            &KernelEmitInputs {
                kernel_name: "split_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cuda_cfg(),
                field_inputs: &[
                    ("cons_mom_0".into(), "cons.mom[0]".into()),
                    ("cons_mom_1".into(), "cons.mom[1]".into()),
                ],
                scalar_params: &[],
                field_writes: &[
                    crate::gv::KernelWrite::new("prim_vel_0", "prim.vel[0]", cons_mom_0),
                    crate::gv::KernelWrite::new("prim_vel_1", "prim.vel[1]", cons_mom_1),
                ],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
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
        let desc = emit_kernel_from_lowering(
            &g,
            &KernelEmitInputs {
                kernel_name: "inplace_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cuda_cfg(),
                field_inputs: &[("cons_den".into(), "cons.den".into())],
                scalar_params: &[],
                field_writes: &[crate::gv::KernelWrite::new("cons_den", "cons.den", updated)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        // one buffer, marked as output.
        assert_eq!(desc.field_bindings.len(), 1);
        assert_eq!(desc.field_bindings[0].field.name(), "cons.den");
        assert!(desc.field_bindings[0].is_output);
        // load + store both reference buf0.
        assert!(desc.source.contains("double cons_den = field0.data["));
        assert!(
            desc.source.contains("field0.data[") && desc.source.contains("] = (cons_den + 1.0);")
        );
    }

    #[test]
    fn scalar_param_appears_in_signature_and_passes_through() {
        // out[coord] = a[coord] * dt;  with dt a scalar __global__ arg.
        let mut g = Graph::new();
        let a = scalar_param(&mut g, "a");
        let dt = scalar_param(&mut g, "dt");
        let prod = g.element_wise(ElementWiseOp::Mul, vec![a, dt], None);
        let desc = emit_kernel_from_lowering(
            &g,
            &KernelEmitInputs {
                kernel_name: "scale_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cuda_cfg(),
                field_inputs: &[("a".into(), "a".into())],
                scalar_params: &["dt".to_string()],
                field_writes: &[crate::gv::KernelWrite::new("out", "out", prod)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
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
        let signature = format!(
            "{} void rmhd_field_max_3d(",
            crate::emit::global_qualifier(Target::Cuda)
        );
        assert!(s.contains(&signature), "src:\n{s}");
        assert!(s.contains("__symbi_View field0"));
        assert!(s.contains("double* partials"));
        assert!(s.contains("__shared__ double sdata[256];"));
        assert!(s.contains("partials[blockIdx.x] = sdata[0];"));
        // max via the inline ternary (fmax lives in <math.h>, which NVRTC leaves
        // out); the identity is a finite literal.
        assert!(
            s.contains("sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s]"),
            "src:\n{s}"
        );
        assert!(s.contains("double val = -1.0e308;"));
        assert!(!s.contains("fmax"), "must not use fmax (NVRTC-unsafe): {s}");
        assert!(
            !s.contains("INFINITY"),
            "must not use INFINITY (NVRTC-unsafe): {s}"
        );
        // NaN-propagation guard: a poisoned cell survives the block reduce and reaches
        // the host dt guard ([[feedback_no_silent_floors]]); the bare ternary
        // drops NaN. `x != x` is the NVRTC-safe NaN test.
        assert!(
            s.contains("(sdata[tid] != sdata[tid])"),
            "max must guard NaN via x!=x: {s}"
        );
        // value loaded at the cell via the view index; one buffer binding (input).
        assert!(s.contains("val = field0.data["));
        assert_eq!(desc.field_bindings.len(), 1);
        assert!(!desc.field_bindings[0].is_output);
    }

    #[test]
    fn segmented_reduction_privatizes_and_folds_once_per_block() {
        let desc = render_field_segmented_reduction(
            "census_add_2d",
            2,
            Precision::F64,
            ReductionOp::Add,
            3,
            8,
        );
        let s = &desc.source;
        let signature = format!(
            "{} void census_add_2d(",
            crate::emit::global_qualifier(Target::Cuda)
        );
        assert!(s.contains(&signature), "src:\n{s}");
        // three value views, then the segment view; one binding each.
        assert!(s.contains("__symbi_View field0"));
        assert!(s.contains("__symbi_View field2"));
        // the segment rides the scalar view like every other buffer: a generated kernel has one
        // scalar type for all of them, so the bucket index arrives as a small exact integer on
        // that same carrier.
        assert!(s.contains("__symbi_View field3"));
        assert_eq!(desc.field_bindings.len(), 4);
        assert!(desc.field_bindings.iter().all(|b| !b.is_output));

        // 8 segments * 3 values fits the shared budget, so the block absorbs its cells
        // privately and touches the global output once per slot.
        assert!(s.contains("__shared__ double acc[24];"), "src:\n{s}");
        assert!(
            s.contains("__symbi_seg_accum(&acc[seg * 3u + 0u]"),
            "src:\n{s}"
        );
        assert!(
            s.contains("__symbi_seg_accum(&out[i], acc[i]);"),
            "src:\n{s}"
        );
        // a fixed block count walking the domain with a grid stride.
        assert!(s.contains("gid += blockDim.x * gridDim.x"), "src:\n{s}");
        // add accumulates with the native `atomicAdd`; the compare-and-swap loop
        // belongs to min/max.
        assert!(s.contains("atomicAdd(addr, v)"), "src:\n{s}");
        assert!(!s.contains("atomicCAS"), "src:\n{s}");
    }

    #[test]
    fn segmented_reduction_falls_back_to_direct_global_accumulation() {
        // an accumulator past the shared budget stays in global memory, so cells combine
        // straight into the global output. the choice follows `segmented_privatizes`.
        let n_segments = SEGMENTED_LDS_BUDGET_BYTES / 8 + 1;
        assert!(!segmented_privatizes(n_segments, 1, Precision::F64));
        let desc = render_field_segmented_reduction(
            "census_wide_1d",
            1,
            Precision::F64,
            ReductionOp::Add,
            1,
            n_segments,
        );
        let s = &desc.source;
        assert!(!s.contains("__shared__"), "src:\n{s}");
        assert!(
            s.contains("__symbi_seg_accum(&out[seg * 1u + 0u]"),
            "src:\n{s}"
        );
    }

    #[test]
    fn segmented_reduction_min_uses_a_nan_propagating_cas() {
        // a floating-point min lands through compare-and-swap on the bit pattern, retried
        // until the slot holds the combined value. the combine is the same `x != x`
        // NaN-propagating ternary the block reduction uses, so a poisoned cell survives
        // to the host.
        let desc = render_field_segmented_reduction(
            "census_min_1d",
            1,
            Precision::F64,
            ReductionOp::Min,
            1,
            4,
        );
        let s = &desc.source;
        assert!(s.contains("atomicCAS(p, assumed"), "src:\n{s}");
        assert!(s.contains("__double_as_longlong"), "src:\n{s}");
        assert!(s.contains("(cur != cur)"), "src:\n{s}");
        assert!(!s.contains("INFINITY"), "src:\n{s}");
        assert!(s.contains("acc[i] = 1.0e308;"), "src:\n{s}");
    }

    #[test]
    fn segmented_reduction_counts_cells_outside_the_binning() {
        // a cell whose bin index lies past the last segment is outside the binning. it is
        // dropped and counted — an under-covering binning that reported nothing would be
        // indistinguishable from a physics result.
        let desc = render_field_segmented_reduction(
            "census_drop_1d",
            1,
            Precision::F64,
            ReductionOp::Add,
            1,
            5,
        );
        let s = &desc.source;
        assert!(s.contains("unsigned long long* dropped"), "src:\n{s}");
        assert!(
            s.contains("if (seg == 5u) { n_dropped += 1ull; continue; }"),
            "src:\n{s}"
        );
        assert!(s.contains("atomicAdd(dropped, n_dropped);"), "src:\n{s}");
    }

    #[test]
    #[should_panic(expected = "Mul is not a meaningful segmented reduction")]
    fn segmented_reduction_refuses_a_product() {
        // a product over a bin's cells overflows to zero or infinity at any realistic cell
        // count, leaving no meaningful census statistic.
        render_field_segmented_reduction(
            "census_mul_1d",
            1,
            Precision::F64,
            ReductionOp::Mul,
            1,
            4,
        );
    }

    #[test]
    fn field_reduction_ops_and_precisions() {
        // add/mul/min/max all render their NVRTC-safe combine + identity, at f32 too.
        let add = render_field_reduction("add1", 1, Precision::F64, ReductionOp::Add);
        assert!(
            add.source.contains("double val = 0.0;")
                && add.source.contains("(sdata[tid] + sdata[tid + s])")
        );
        let mul = render_field_reduction("mul1", 1, Precision::F64, ReductionOp::Mul);
        assert!(
            mul.source.contains("double val = 1.0;")
                && mul.source.contains("(sdata[tid] * sdata[tid + s])")
        );
        let min = render_field_reduction("min1", 1, Precision::F64, ReductionOp::Min);
        assert!(
            min.source.contains("double val = 1.0e308;")
                && min.source.contains("< sdata[tid + s] ?")
        );
        let maxf = render_field_reduction("maxf1", 1, Precision::F32, ReductionOp::Max);
        assert!(
            maxf.source.contains("float val = -1.0e38f;"),
            "src:\n{}",
            maxf.source
        );
        assert!(
            maxf.source.contains("__symbi_View field0") && maxf.source.contains("float* partials")
        );
        // 1D: the View struct carries extent and lo, so the signature is one view arg.
        assert!(!maxf.source.contains("buf_extent"));
        assert!(maxf.source.contains("__symbi_View field0"));
    }
}
