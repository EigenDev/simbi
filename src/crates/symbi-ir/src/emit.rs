// =============================================================================
// emit.rs
//
// the target/precision descriptors + the shared device-source helpers
// (`header`, `global_qualifier`, the flat-index stride formula) consumed by the
// LIVE tensor IR emitters (`tensor/emit_kernel.rs`, `tensor::emit_cuda` /
// `tensor::emit_cpu`).
//
// the live emitters walk the tensor IR directly. only the descriptors + the
// ABI-shared helpers remain in this module.
// =============================================================================

// ---- target and precision ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Cuda,
    Hip,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    pub fn c_type(&self) -> &'static str {
        match self {
            Precision::F32 => "float",
            Precision::F64 => "double",
        }
    }
    /// the Rust scalar type name for this precision (`f64` / `f32`). the CPU
    /// emitter spells float buffers, reads, params, and constant suffixes with it.
    pub fn rust_type(&self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TargetConfig {
    pub target: Target,
    pub precision: Precision,
}

// ---- kernel descriptor ----

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldBinding {
    /// the typed field this buffer binds (`Ref` for closed cell-centered vocab,
    /// `Raw` for hand-built staggered/ct/geom/refinement paths). born typed in the
    /// serialized manifest — no bare string crosses the trace -> dispatch ABI.
    pub field: symbi_abi::FieldBind,
    pub buffer_index: u32,
    pub is_output: bool,
}

#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    pub source: String,
    pub kernel_name: String,
    pub field_bindings: Vec<FieldBinding>,
    pub param_names: Vec<String>,
    /// per scalar param (in `param_names` order): is it an integer (i32/u32) vs a
    /// float? part of the kernel ABI — a runtime launcher must pack int params as
    /// `int` and float params as the precision type, in declared order (the CPU
    /// descriptor wrapper routes the same way).
    pub scalar_is_int: Vec<bool>,
    /// the shared-memory tile spec this kernel was rendered with (Gate 3). `Some`
    /// only for the C-family (CUDA) smem path; the dispatch reads it to size the
    /// per-block dynamic `__shared__` allocation (`smem_bytes_per_block`). `None`
    /// for flat kernels and every non-CUDA backend (the CPU descriptor carries it
    /// as `None` — CPU cache-tiles via loop structure, not smem).
    pub tile_spec: Option<crate::gv::TileSpec>,
}

// ---- shared device-source helpers (consumed by tensor/emit_kernel.rs) ----

pub(crate) fn header(target: Target) -> &'static str {
    match target {
        Target::Cuda => "",
        Target::Hip => "#include <hip/hip_runtime.h>\n",
        Target::Metal => "#include <metal_stdlib>\nusing namespace metal;\n",
    }
}

pub(crate) fn global_qualifier(target: Target) -> &'static str {
    match target {
        Target::Cuda | Target::Hip => "extern \"C\" __global__",
        Target::Metal => "kernel",
    }
}

// =============================================================================
// SINGLE SOURCE OF TRUTH for index arithmetic. every kernel emitter — CPU, CUDA,
// the reduction kernel — derives its `__idx_cell_buf{N}` declaration and its
// per-load flat index from these two functions. the formula lives ONCE; any new
// backend gets a new arm of the inner `match` instead of a third copy.
//
// the per-buffer ABI here is FIXED:
//   * `ii`, `jj`, `kk` are the absolute cell coords (i32 / int)
//   * `buf_lo_{b}_{a}` is buffer b's per-axis origin (i32)
//   * `ny_{b}` / `nz_{b}` (CPU) or `(int)_ny_{b}` / `(int)_nz_{b}` (CUDA) are
//     the per-buffer per-axis extents (the stride product is computed inline)
//
// this is the DRY consolidation of the per-buffer index arithmetic.
// =============================================================================

/// which source language the emitter renders to. drives the small syntactic
/// differences (let vs int decl, `as usize` cast, stride name spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexLang {
    Rust,
    Cuda,
}

/// emit ONE `__idx_cell_buf{b}` declaration: the flat offset of cell `(ii, jj, kk)`
/// for buffer `b`. the per-cell hoist that every subsequent load delta-adds to.
///
/// the formula references the buffer's `lo` and `strides` ARRAYS (pre-multiplied
/// at View construction). every backend uses the same formula — `IndexLang`
/// only picks the let/int decl syntax. adding a new backend is one new arm.
/// SYMBI_VEC_LOOP=1: drop the `*strides[last]` on the CONTIGUOUS axis (==1 for
/// row-major, guarded by a debug_assert in the vec loop) so the cell index is
/// affine in the inner coord with coefficient 1 — LLVM's loop-vectorizer then
/// reads consecutive lanes as one vector load. CPU (Rust) only; GPU is SIMT.
fn unit_stride_last() -> bool {
    // debug-emit-knobs gates the SYMBI_VEC_LOOP a/b knob; feature off (default) = canonical env-unset shape (false).
    #[cfg(feature = "debug-emit-knobs")]
    {
        std::env::var("SYMBI_VEC_LOOP").map(|v| v == "1").unwrap_or(false)
    }
    #[cfg(not(feature = "debug-emit-knobs"))]
    {
        false
    }
}

pub fn emit_cell_index_base(lang: IndexLang, ndim: u8, buf: u32, coalesce: bool) -> String {
    let (decl, ty) = match lang {
        IndexLang::Rust => ("let ", ": i32"),
        IndexLang::Cuda => ("int ",  ""),
    };
    let f = buf;
    // view collapse: when the kernel guarantees all buffers share buffer 0's
    // layout, every buffer's cell index is identical — alias it instead of
    // recomputing the full strided offset. buffer 0 still computes the real
    // index; the alias is a no-op binding the compiler folds away. this is the
    // measured 1.45x -> 1.08x lever on index-bound kernels.
    if coalesce && buf != 0 {
        return format!("    {decl}__idx_cell_buf{f}{ty} = __idx_cell_buf0;");
    }
    if lang == IndexLang::Rust && unit_stride_last() {
        let coords = ["ii", "jj", "kk"];
        let nd = ndim as usize;
        let terms: Vec<String> = (0..nd)
            .map(|a| {
                if a == nd - 1 {
                    format!("({} - field{f}.lo[{a}])", coords[a])
                } else {
                    format!("({} - field{f}.lo[{a}]) * field{f}.strides[{a}]", coords[a])
                }
            })
            .collect();
        return format!("    {decl}__idx_cell_buf{f}{ty} = {};", terms.join(" + "));
    }
    match ndim {
        1 => format!(
            "    {decl}__idx_cell_buf{f}{ty} = (ii - field{f}.lo[0]) * field{f}.strides[0];",
        ),
        2 => format!(
            "    {decl}__idx_cell_buf{f}{ty} = (ii - field{f}.lo[0]) * field{f}.strides[0] \
             + (jj - field{f}.lo[1]) * field{f}.strides[1];",
        ),
        3 => format!(
            "    {decl}__idx_cell_buf{f}{ty} = (ii - field{f}.lo[0]) * field{f}.strides[0] \
             + (jj - field{f}.lo[1]) * field{f}.strides[1] \
             + (kk - field{f}.lo[2]) * field{f}.strides[2];",
        ),
        _ => panic!("emit_cell_index_base: unsupported ndim {ndim}"),
    }
}

/// emit the per-access flat index expression: `__idx_cell_buf{b} + delta` form.
/// for cell-base reads the delta folds to zero; for `(_coord_a + literal)` stencil
/// reads it folds to `literal * stride`. either way the compiler reduces it to a
/// single immediate-displaced load against the precomputed base.
///
/// every backend uses the same formula — `IndexLang` only picks the cast/
/// terminator syntax (Rust ends `as usize`; CUDA stays an `int`).
pub fn emit_flat_index(lang: IndexLang, ndim: u8, buf: u32, comps: &[&str]) -> String {
    let terminator = match lang {
        IndexLang::Rust => " as usize",
        IndexLang::Cuda => "",
    };
    let f = buf;
    if lang == IndexLang::Rust && unit_stride_last() {
        let coords = ["ii", "jj", "kk"];
        let nd = ndim as usize;
        let terms: Vec<String> = (0..nd)
            .map(|a| {
                if a == nd - 1 {
                    format!("(({}) - {})", comps[a], coords[a])
                } else {
                    format!("(({}) - {}) * field{f}.strides[{a}]", comps[a], coords[a])
                }
            })
            .collect();
        return format!("(__idx_cell_buf{f} + {}){terminator}", terms.join(" + "));
    }
    match ndim {
        1 => format!(
            "(__idx_cell_buf{f} + (({c0}) - ii) * field{f}.strides[0]){terminator}",
            c0 = comps[0],
        ),
        2 => format!(
            "(__idx_cell_buf{f} + (({c0}) - ii) * field{f}.strides[0] \
             + (({c1}) - jj) * field{f}.strides[1]){terminator}",
            c0 = comps[0], c1 = comps[1],
        ),
        3 => format!(
            "(__idx_cell_buf{f} + (({c0}) - ii) * field{f}.strides[0] \
             + (({c1}) - jj) * field{f}.strides[1] \
             + (({c2}) - kk) * field{f}.strides[2]){terminator}",
            c0 = comps[0], c1 = comps[1], c2 = comps[2],
        ),
        _ => panic!("emit_flat_index: unsupported ndim {ndim}"),
    }
}


// ---- reduction op (the device reduction descriptor's combine semantics) ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Add,
    Mul,
    Min,
    Max,
}

