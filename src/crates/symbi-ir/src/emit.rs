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
    /// the shared-memory tile spec this kernel was rendered with. `Some`
    /// only for the C-family (CUDA) smem path; the dispatch reads it to size the
    /// per-block dynamic `__shared__` allocation (`smem_bytes_per_block`). `None`
    /// for flat kernels and every non-CUDA backend (the CPU descriptor carries it
    /// as `None` — CPU cache-tiles via loop structure).
    pub tile_spec: Option<crate::gv::TileSpec>,
}

// ---- shared device-source helpers (consumed by tensor/emit_kernel.rs) ----

// nvrtc and hiprtc both pre-define the device api -- builtin vector types, the
// `__global__`/`__device__` qualifiers, the thread/block index builtins, and the math
// intrinsics -- and compile from an in-memory buffer with no filesystem access. neither
// accepts an include of the toolkit's runtime header: under hiprtc `#include
// <hip/hip_runtime.h>` fails to resolve and aborts the compile. metal is compiled from
// source by a filesystem-backed compiler and does need its standard library named.
pub(crate) fn header(target: Target) -> &'static str {
    match target {
        Target::Cuda | Target::Hip => "",
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
// backend gets a new arm of the inner `match`, reusing the one formula.
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

/// drop the `* strides[CONTIGUOUS_AXIS]` factor from the cell index. this is an IDENTITY:
/// `strides_from_extent` sets that stride to exactly 1 for every buffer, so the emitted
/// address is unchanged. what changes is what the compiler can PROVE — with the multiply present the
/// index advances by an opaque runtime `i32` per iteration of the innermost loop, so LLVM cannot show
/// the access is unit-stride and will not vectorize the load. with it gone the index is affine in the
/// inner coord with coefficient 1.
///
/// the axis is DERIVED from the layout (`CONTIGUOUS_AXIS`), never assumed: the physical-x-fastest
/// convention makes axis 0 the contiguous one (column-major, unlike C row-major's last axis). Rust (CPU) only — the CUDA arm is SIMT and
/// maps `threadIdx.x` onto the same axis, so it needs no index rewrite.
const fn unit_stride_contiguous(lang: IndexLang) -> bool {
    matches!(lang, IndexLang::Rust)
}

/// emit ONE `__idx_cell_buf{b}` declaration: the flat offset of cell `(ii, jj, kk)`
/// for buffer `b`. the per-cell hoist that every subsequent load delta-adds to.
///
/// the formula references the buffer's `lo` and `strides` ARRAYS (pre-multiplied
/// at View construction). every backend uses the same formula — `IndexLang`
/// only picks the let/int decl syntax. adding a new backend is one new arm.
pub fn emit_cell_index_base(lang: IndexLang, ndim: u8, buf: u32, coalesce: bool) -> String {
    let (decl, ty) = match lang {
        IndexLang::Rust => ("let ", ": i32"),
        IndexLang::Cuda => ("int ", ""),
    };
    let f = buf;
    // view collapse: when the kernel guarantees all buffers share buffer 0's
    // layout, every buffer's cell index is identical — alias it, skipping the
    // full strided offset recompute. buffer 0 still computes the real
    // index; the alias is a no-op binding the compiler folds away. this is the
    // measured 1.45x -> 1.08x lever on index-bound kernels.
    if coalesce && buf != 0 {
        return format!("    {decl}__idx_cell_buf{f}{ty} = __idx_cell_buf0;");
    }
    if unit_stride_contiguous(lang) {
        let coords = ["ii", "jj", "kk"];
        let nd = ndim as usize;
        // the unit-stride term belongs to CONTIGUOUS_AXIS — the axis `strides_from_extent` gives a
        // stride of 1. dropping the multiply on any OTHER axis silently mis-indexes the buffer.
        let terms: Vec<String> = (0..nd)
            .map(|a| {
                if a == symbi_algebra::CONTIGUOUS_AXIS {
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
    if unit_stride_contiguous(lang) {
        let coords = ["ii", "jj", "kk"];
        let nd = ndim as usize;
        // the stencil delta drops its multiply on CONTIGUOUS_AXIS for the same reason the cell base
        // does: that stride is 1 by construction. any other axis keeps its stride factor.
        let terms: Vec<String> = (0..nd)
            .map(|a| {
                if a == symbi_algebra::CONTIGUOUS_AXIS {
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
            c0 = comps[0],
            c1 = comps[1],
        ),
        3 => format!(
            "(__idx_cell_buf{f} + (({c0}) - ii) * field{f}.strides[0] \
             + (({c1}) - jj) * field{f}.strides[1] \
             + (({c2}) - kk) * field{f}.strides[2]){terminator}",
            c0 = comps[0],
            c1 = comps[1],
            c2 = comps[2],
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
