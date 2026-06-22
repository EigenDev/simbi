// =============================================================================
// emit_kernel_cpu.rs
//
// the CPU-native (Rust) sibling of emit_kernel.rs's `emit_kernel_from_lowering`
// (which emits a CUDA `__global__`). this is the build-time AOT path for the CPU
// backend (docs/design/10 §4): a scalarized stencil kernel -> a compilable Rust
// `pub fn` that iterates the dispatch window over `&[f64]` / `&mut [f64]`
// buffers, with the SAME flat-index ABI as the CUDA emitter (coord is absolute;
// access is `buf[(coord - buf_lo) . strides]`).
//
// it reuses the proven `emit_cpu` ScalarExpr/ScalarStmt renderer for the f64
// body and writes, and the same buffer-binding + FieldLoadAt-resolution
// structure as the CUDA emitter — only the syntax (Rust slices, a cell loop)
// differs. precision is f64 (the substrate is double-precision; an f32 CPU path
// is a future concern).
//
// INDICES ARE INTEGERS. coord index params are `I32` in the IR, so coord and
// index arithmetic is pure `i32` — coord vars, strides, and buffer-lo offsets are
// all `i32`; integer stencil shifts render as integer literals (`+ 1`, not
// `+ 1.0`); a CSE'd shared shift is an `i32` body let, so a multi-law kernel's
// indices stay integer. the ONLY conversion is the `as usize` at the slice-index
// site, which Rust's `Index<usize>` requires. (32-bit indices match the existing
// CUDA `(int)` flat-index ABI.) data-dependent gather indexing (a float field
// value used as an index) is NOT yet supported here and panics loudly rather than
// silently routing integers through float.
//
// signature shape (1D; 2D/3D add stride extents and nested loops):
//   pub fn <name>(
//       buf0: &mut [f64], buf1: &[f64], ...,   // outputs &mut, inputs-only &
//       grid_size_0: i32, dom_lo_0: i32,
//       buf_lo_0_0: i32, ..., <scalar params>: f64,
//   ) {
//       for _i0 in 0..grid_size_0 {
//           let ii: i32 = _i0 + dom_lo_0;
//           let <key>: f64 = buf<N>[(ii - buf_lo_<N>_0) as usize];
//           <body>
//           buf<N>[<flat>] = <expr>;
//       }
//   }
//
// aliasing: a buffer read AND written (in-place conserved update) is one
// `&mut [f64]` param; reads go through it into f64 locals BEFORE the store, so
// there is no held borrow.
// =============================================================================

/// the default CPU cache-tile edge. the emitter parallelizes over `N^ndim` cache
/// blocks with serial nested loops inside each block, keeping a block's stencil
/// neighborhood + multi-field working set resident. measured ~1.4-2.1x full-step
/// over the flat emit, GROWING with grid size, and grid-size-independent
/// throughput (docs/design/26). 8 and 16 both measured ~optimal; 8 is the
/// conservative default (fits closer to L1/L2).
const CPU_TILE: usize = 8;

/// cache-tile edge length for the CPU emit. DEFAULT = `CPU_TILE` (tiled).
/// `SYMBI_TILE_CPU=0` forces the FLAT parallel-over-all-cells emit (debug / A/B);
/// `SYMBI_TILE_CPU=N` overrides the tile edge. read at emit time and tracked by
/// `symbi-aot/build.rs` (`rerun-if-env-changed`), so toggling regenerates kernels.
fn cpu_tile_size() -> usize {
    // debug-emit-knobs gates the SYMBI_TILE_CPU a/b knob; feature off (default) = canonical env-unset shape (CPU_TILE, tiled).
    #[cfg(feature = "debug-emit-knobs")]
    {
        match std::env::var("SYMBI_TILE_CPU").ok().and_then(|s| s.parse::<usize>().ok()) {
            Some(0) | Some(1) => 0,        // explicit flat override
            Some(t) => t,                  // explicit tile edge
            None => CPU_TILE,              // default: tiled
        }
    }
    #[cfg(not(feature = "debug-emit-knobs"))]
    {
        CPU_TILE
    }
}

/// SYMBI_UNCHECKED_LOADS=1: emit field loads/stores through `get_unchecked` instead
/// of bounds-checked `data[idx]`. the per-cell index is computed from the cell coord
/// + strides + stencil offset, always within the buffer's allocated domain (the
/// kernel's correctness contract, validated by the carrier oracle), so the bounds
/// check is dead weight — and an opaque-index check defeats vectorization. read at
/// emit time (build.rs tracks the env); default OFF (safe checked indexing).
/// part 1 of the loop-lowering: dropping the checks is the ~8x scalar win measured
/// in the simd spike. UNSAFE if an index is ever wrong — gate on the test suite.
fn unchecked_loads() -> bool {
    // vec mode needs the bounds-check branches gone to vectorize, so it implies unchecked.
    std::env::var("SYMBI_UNCHECKED_LOADS").map(|v| v == "1").unwrap_or(false) || vec_loop()
}

/// SYMBI_VEC_LOOP=1 (ndim>=2): emit the ROW-PARALLEL loop — parallelize over the
/// outer (non-contiguous) axes, with a COUNTABLE inner loop walking the contiguous
/// last axis. combined with the unit-stride index (emit::emit_*_index) this is the
/// shape LLVM loop-vectorizes (the simd-spike form). 1D / coalesce-incompatible
/// kernels fall back to flat/tiled. read at emit time; build.rs tracks the env.
fn vec_loop() -> bool {
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

use crate::backends::cpu::{emit_expr, emit_stmt, rust_type_name};
use crate::backends::kernel::KernelEmitInputs;
use crate::backends::render::{emit_kernel_render, KernelRenderer, COORD_VARS};
use crate::graph::ConstValue;
use crate::passes::scalarize::{BinaryKind, ScalarExpr, ScalarStmt, UnaryKind};
use crate::{ElementTy, Graph};
use crate::emit::KernelDescriptor;

/// the Rust (CPU) backend: the per-language spelling for the shared kernel driver
/// (`emit_render`). produces a compilable generic `pub fn k<S: Scalar>` over `&[S]`
/// slices with pure-integer indices (one `as usize` at the slice boundary). the
/// float scalar is the type parameter `S` (docs/design/15 §4: `Sim<f64>`/`Sim<f32>`
/// pick it by the buffer type they pass); constants render `S::lit(..)`, math
/// resolves to the `Scalar` trait. one kernel, every precision — no monomorphized
/// duplication, no dispatch.
///
/// **B11 — rayon-parallel outer cell loop**. the renderer tracks which buffer
/// indices are MUTABLE (outputs / in-place fields) as they're emitted via
/// `buffer_param`. `cell_prelude` then emits a `rayon::IntoParallelIterator`
/// fan-out over the outermost grid axis, with each mutable buffer's pointer
/// wrapped in an unsafe-Send newtype so workers can re-borrow disjoint cell
/// ranges. inputs (`&[S]`) capture by reference. the `with_min_len(64)` floor
/// keeps small grids on the calling thread. unchanged kernels (snapshot,
/// ghost_fill) get parallelism for free.
pub struct RustRenderer {
    /// the buffer indices (in declaration order) that the emitter saw as
    /// mutable — `&mut [S]` slices that the closure body writes to. populated
    /// during `buffer_param` calls; consumed by `cell_prelude` to emit the
    /// raw-ptr Send wrappers + closure-internal re-borrows.
    mut_buf_indices: std::cell::RefCell<Vec<u32>>,
    /// SERIAL variant: emit a plain nested `for` loop over the exec window with
    /// NO `into_par_iter` / closure / mut-ptr rebind — the kernel runs entirely
    /// on the caller's thread. the EXECUTOR owns the parallelism: it fans a
    /// disjoint cover (a BlockGrid / guillotine cover) out over rayon and calls
    /// this serial kernel per block, so the cover dispatches in ONE fork-join
    /// instead of one-per-block. soundness rests on the cover being a PARTITION
    /// (disjoint output writes) — the proven law.
    serial: bool,
}

impl RustRenderer {
    pub fn new() -> Self {
        Self { mut_buf_indices: std::cell::RefCell::new(Vec::new()), serial: false }
    }
    pub fn serial() -> Self {
        Self { mut_buf_indices: std::cell::RefCell::new(Vec::new()), serial: true }
    }
}
impl Default for RustRenderer {
    fn default() -> Self { Self::new() }
}

impl KernelRenderer for RustRenderer {
    fn preamble(&self, _device_preamble: &[String]) -> String {
        // the ScalarExpr renderer parenthesizes every BinOp; allow the redundant
        // outer pair rather than emit precedence-fragile minimal-paren code.
        // (device_preamble is a GPU device-function concept — unused on CPU.)
        "#[allow(unused_parens)]\n".to_string()
    }
    fn buffer_param(&self, idx: u32, is_output: bool) -> String {
        // Phase 1B-4: per-buffer arg is ONE view-struct reference. all of `lo`,
        // `extent`, and pre-multiplied `strides` ride along inside the struct.
        // the OLD scattered `buf_lo_X_a` / `buf_extent_X_a` scalar args are gone
        // (see `skip_scattered_buffer_layout_args` below).
        if is_output {
            self.mut_buf_indices.borrow_mut().push(idx);
            format!("    field{idx}: &mut CpuFieldMut<S>")
        } else {
            format!("    field{idx}: &CpuField<S>")
        }
    }
    fn grid_size_param(&self, axis: usize) -> String {
        format!("    grid_size_{axis}: i32")
    }
    fn int_param(&self, name: &str) -> String {
        format!("    {name}: i32")
    }
    fn scalar_param(&self, name: &str, element: ElementTy) -> String {
        format!("    {name}: {}", rust_type_name(element, true))
    }
    fn open_signature(&self, name: &str) -> String {
        // (no ledger reset here — `buffer_param` is called by the driver
        // BEFORE this point, so a clear here would wipe what we just collected.
        // each emission constructs a fresh `RustRenderer::new()` so the state
        // is naturally per-emission.)
        // **B11 — `+ Send + Sync`** so rayon par_iter can capture &[S] inputs +
        // S scalar params, and so the unsafe raw-ptr Send wrapper for mut
        // buffers compiles (the bound on T inside `unsafe impl Send`).
        // `__raw` is the positional ABI core — codegen-internal. its arity changes
        // whenever a builder adds/removes an input, so HAND-WRITTEN positional callers
        // drift silently. host + test code must call the name-keyed `NamedKernel`
        // (symbi-aot) instead, which binds by manifest field name. `#[doc(hidden)]`
        // keeps the slice-form wrapper + the registry working while removing `__raw`
        // as a discoverable/recommended API.
        // `unused_unsafe`: with SYMBI_UNCHECKED_LOADS the per-access `unsafe`
        // blocks nest (a stencil load inlined into a store's unsafe rhs) — the
        // inner ones are redundant but correct; this is machine-generated code.
        format!("#[doc(hidden)]\n#[allow(non_snake_case, unused_variables, unused_unsafe)]\npub fn {name}__raw<S: Scalar + OrderedNumeric + Send + Sync>(\n")
    }
    fn params_close(&self) -> &'static str {
        ",\n) {\n" // Rust allows the trailing comma
    }
    fn cell_prelude(&self, ndim: usize, _n_buffers: u32) -> Vec<String> {
        // Phase 1B-4: kernel signature now takes the view structs DIRECTLY:
        // `field0: &CpuField<S>, …, field_n: &mut CpuFieldMut<S>`. no per-cell
        // reconstruction needed — input fields capture by ref into the closure,
        // outputs use the standard mut-ptr-rebind dance.
        //
        // body emission (cell-base, flat index, base reads, stores) all spell
        // `field{N}.lo[..]`/`.strides[..]`/`.data` directly — SINGLE source of
        // truth (`CpuField` / `CpuFieldMut` in `symbi-aot`).
        let mut v = Vec::new();

        // SERIAL variant: plain nested `for` over the exec window, on the caller's
        // thread. no `into_par_iter`, no closure, no mut-ptr rebind — `field{N}`
        // (incl. the `&mut CpuFieldMut` outputs) is used directly. the cover
        // executor parallelizes over many such blocks, paying ONE fork-join total.
        if self.serial {
            for aa in 0..ndim {
                v.push(format!("    for _i{aa} in 0..(grid_size_{aa} as i32) {{"));
                v.push(format!("    let {}: i32 = _i{aa} + dom_lo_{aa};", COORD_VARS[aa]));
            }
            return v;
        }

        let mut_buf = self.mut_buf_indices.borrow();

        // for mut buffers: hoist `lo` + `strides` arrays (Copy `[i32; 4]`) so
        // the par_iter closure captures them BY VALUE; capture data ptr in
        // `__MutBufPtr`. inside the closure we rebuild the CpuFieldMut.
        for &bi in mut_buf.iter() {
            v.push(format!("    let __field{bi}_lo: [i32; 4] = field{bi}.lo;"));
            v.push(format!("    let __field{bi}_strides: [i32; 4] = field{bi}.strides;"));
        }

        // --- B11 rayon-parallel outer loop scaffold ---
        v.push("    #[allow(non_camel_case_types)]".to_string());
        v.push("    struct __MutBufPtr<T>(*mut T, usize);".to_string());
        v.push("    unsafe impl<T: Send> Send for __MutBufPtr<T> {}".to_string());
        v.push("    unsafe impl<T: Send> Sync for __MutBufPtr<T> {}".to_string());
        v.push("    impl<T> __MutBufPtr<T> {".to_string());
        v.push("        fn ptr(&self) -> *mut T { self.0 }".to_string());
        v.push("        fn len(&self) -> usize { self.1 }".to_string());
        v.push("    }".to_string());
        for &bi in mut_buf.iter() {
            v.push(format!(
                "    let __mb{bi} = __MutBufPtr(field{bi}.data.as_mut_ptr(), field{bi}.data.len());"
            ));
        }

        // the tiled loop emits a non-indexed `into_par_iter().for_each()` over the
        // tile count — IntoParallelIterator + ParallelIterator suffice, and pulling
        // in IndexedParallelIterator there would be a dead import (208 warnings
        // across the registry). the FLAT loop, however, calls `.with_min_len(16)`,
        // which lives on IndexedParallelIterator — so that trait MUST be in scope
        // there or the kernel fails to compile.
        if cpu_tile_size() == 0 {
            v.push("    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};".to_string());
        } else {
            v.push("    use rayon::iter::{IntoParallelIterator, ParallelIterator};".to_string());
        }

        // helper: rebind each mut buffer's raw ptr to a fresh CpuFieldMut inside
        // the closure (shadows the borrowed param; never held across threads).
        let push_rebind = |v: &mut Vec<String>| {
            for &bi in mut_buf.iter() {
                v.push(format!(
                    "        let buf{bi}: &mut [S] = unsafe {{ std::slice::from_raw_parts_mut(__mb{bi}.ptr(), __mb{bi}.len()) }};"
                ));
                v.push(format!(
                    "        let field{bi} = CpuFieldMut {{ data: buf{bi}, lo: __field{bi}_lo, strides: __field{bi}_strides }};"
                ));
            }
        };

        if vec_loop() && ndim >= 2 {
            // VECTORIZABLE ROW MODE (SYMBI_VEC_LOOP): parallelize over the outer
            // (non-contiguous) axes; a single COUNTABLE inner loop walks the
            // contiguous last axis. with the unit-stride index (emit::) this is the
            // shape LLVM loop-vectorizes (the simd-spike B/C form). row-major
            // guarantees strides[last]==1 — asserted once per row.
            let last = ndim - 1;
            let outer_expr = (0..last)
                .map(|aa| format!("(grid_size_{aa} as usize)"))
                .collect::<Vec<_>>()
                .join(" * ");
            v.push(format!("    let _orows: usize = {outer_expr};"));
            v.push("    (0.._orows).into_par_iter().for_each(|_orow| {".to_string());
            push_rebind(&mut v);
            // unflatten _orow -> outer coords (axis `last-1` fastest among them).
            if last == 1 {
                v.push("        let _i0: i32 = _orow as i32;".to_string());
            } else {
                v.push("        let mut _orem: usize = _orow;".to_string());
                for aa in (1..last).rev() {
                    v.push(format!("        let _i{aa}: i32 = (_orem % (grid_size_{aa} as usize)) as i32;"));
                    v.push(format!("        _orem /= grid_size_{aa} as usize;"));
                }
                v.push("        let _i0: i32 = _orem as i32;".to_string());
            }
            for aa in 0..last {
                v.push(format!("        let {}: i32 = _i{aa} + dom_lo_{aa};", COORD_VARS[aa]));
            }
            // unit-stride precondition (row-major); debug-only, carrier oracle gates correctness.
            v.push(format!(
                "        debug_assert!(field0.strides[{last}] == 1, \"SYMBI_VEC_LOOP needs a contiguous last axis\");"
            ));
            // countable inner loop over the contiguous axis (the vectorized dim).
            v.push(format!("        for _ic in 0..(grid_size_{last} as usize) {{"));
            v.push(format!("        let _i{last}: i32 = _ic as i32;"));
            v.push(format!("        let {}: i32 = _i{last} + dom_lo_{last};", COORD_VARS[last]));
        } else if cpu_tile_size() == 0 {
            // FLAT (default): parallelize over the flattened interior (every
            // cell), NOT just the outer axis — exposes all cells so the grid
            // scales like the 1D path (mirrors the GPU global thread index).
            let total_expr = (0..ndim)
                .map(|aa| format!("(grid_size_{aa} as usize)"))
                .collect::<Vec<_>>()
                .join(" * ");
            v.push(format!("    let _ptotal: usize = {total_expr};"));
            v.push("    (0.._ptotal).into_par_iter().with_min_len(16).for_each(|_flat| {".to_string());
            push_rebind(&mut v);
            // unflatten row-major (axis 0 outermost, axis ndim-1 contiguous).
            if ndim == 1 {
                v.push("        let _i0: i32 = _flat as i32;".to_string());
            } else {
                v.push("        let mut _rem: usize = _flat;".to_string());
                for aa in (1..ndim).rev() {
                    v.push(format!("        let _i{aa}: i32 = (_rem % (grid_size_{aa} as usize)) as i32;"));
                    v.push(format!("        _rem /= grid_size_{aa} as usize;"));
                }
                v.push("        let _i0: i32 = _rem as i32;".to_string());
            }
            for aa in 0..ndim {
                v.push(format!("    let {}: i32 = _i{aa} + dom_lo_{aa};", COORD_VARS[aa]));
            }
        } else {
            // TILED (SYMBI_TILE_CPU=N): parallelize over N^ndim cache blocks;
            // serial nested loops over each block's cells keep its stencil
            // neighborhood in cache. recovers the per-cell cache-miss penalty
            // once the grid working set exceeds cache (docs/design/26). per-tile
            // rebind (amortized over the block's cells).
            let tile = cpu_tile_size();
            v.push(format!("    let _ts: usize = {tile};"));
            for aa in 0..ndim {
                v.push(format!("    let _nt_{aa}: usize = ((grid_size_{aa} as usize) + _ts - 1) / _ts;"));
            }
            let ntiles_expr = (0..ndim).map(|aa| format!("_nt_{aa}")).collect::<Vec<_>>().join(" * ");
            v.push(format!("    let _ntiles: usize = {ntiles_expr};"));
            v.push("    (0.._ntiles).into_par_iter().for_each(|_tile| {".to_string());
            push_rebind(&mut v);
            // unflatten the tile index -> per-axis tile coord (ndim-1 fastest).
            if ndim == 1 {
                v.push("        let _tc0: usize = _tile;".to_string());
            } else {
                v.push("        let mut _trem: usize = _tile;".to_string());
                for aa in (1..ndim).rev() {
                    v.push(format!("        let _tc{aa}: usize = _trem % _nt_{aa};"));
                    v.push(format!("        _trem /= _nt_{aa};"));
                }
                v.push("        let _tc0: usize = _trem;".to_string());
            }
            // nested cell loops within the tile; declare each axis's coord right
            // after its index. `break` on the boundary handles partial tiles
            // (`_c{aa}` is monotonic in `_d{aa}`). close() emits ndim matching `}`.
            for aa in 0..ndim {
                v.push(format!("        for _d{aa} in 0.._ts {{"));
                v.push(format!("        let _c{aa}: usize = _tc{aa} * _ts + _d{aa};"));
                v.push(format!("        if _c{aa} >= grid_size_{aa} as usize {{ break; }}"));
                v.push(format!("        let _i{aa}: i32 = _c{aa} as i32;"));
                v.push(format!("        let {}: i32 = _i{aa} + dom_lo_{aa};", COORD_VARS[aa]));
            }
        }
        v
    }
    fn coord_decl(&self, axis: u8, _element: ElementTy) -> String {
        // the cell index is always i32; physical-space reals come from promoting
        // `index * dx` (the graph's usual arithmetic conversions), not a float coord.
        format!("    let _coord_{axis}: i32 = {};", COORD_VARS[axis as usize])
    }
    fn index_lang(&self) -> crate::emit::IndexLang { crate::emit::IndexLang::Rust }
    fn skip_scattered_buffer_layout_args(&self) -> bool { true }
    fn flat_index(&self, ndim: u8, buf: u32, comps: &[String]) -> String {
        rust_flat_index(ndim, buf, comps)
    }
    fn render_index_component(&self, e: &ScalarExpr, coord_vars: &[&str]) -> String {
        render_index_expr(e, coord_vars)
    }
    fn base_read(&self, key: &str, buf: u32, flat: &str) -> String {
        // single-source-of-truth: read through the field's data slice. the
        // `flat` expression comes from `emit::emit_flat_index(Rust, ...)` which
        // already references `field{buf}.strides[..]` / the precomputed base.
        if unchecked_loads() {
            format!("    let {key}: S = unsafe {{ *field{buf}.data.get_unchecked({flat}) }};")
        } else {
            format!("    let {key}: S = field{buf}.data[{flat}];")
        }
    }
    fn load_at_expr(&self, buf: u32, flat: &str) -> String {
        // every stencil load goes through the same view-struct: `field{N}.data`.
        // mirrors the C++ `view_t::operator()` — one method, every emitter.
        if unchecked_loads() {
            format!("(unsafe {{ *field{buf}.data.get_unchecked({flat}) }})")
        } else {
            format!("field{buf}.data[{flat}]")
        }
    }
    fn render_stmt(&self, stmt: &ScalarStmt) -> String {
        let mut s = String::from("    ");
        emit_stmt(&mut s, stmt, true);
        s
    }
    fn render_output(&self, expr: &ScalarExpr) -> String {
        let mut s = String::new();
        emit_expr(&mut s, expr, true);
        s
    }
    fn store(&self, buf: u32, flat: &str, expr: &str) -> String {
        // single-source-of-truth: write through the field's data slice. inside
        // the rayon closure, `field{buf}` is a freshly-constructed CpuFieldMut
        // (see `cell_prelude`) bound to the rebound &mut [S] slice.
        if unchecked_loads() {
            format!("    unsafe {{ *field{buf}.data.get_unchecked_mut({flat}) = {expr}; }}")
        } else {
            format!("    field{buf}.data[{flat}] = {expr};")
        }
    }
    fn close(&self, ndim: usize) -> String {
        if self.serial {
            // serial: close the `ndim` nested cell-fors, then the fn body.
            let mut s = String::new();
            for _ in 0..ndim {
                s.push_str("    }\n");
            }
            s.push_str("}\n");
            return s;
        }
        if vec_loop() && ndim >= 2 {
            // vec row mode: close the single contiguous inner `for`, then the
            // for_each closure and the fn body.
            "    }\n    });\n}\n".to_string()
        } else if cpu_tile_size() == 0 {
            // flat: the loop has NO inner serial fors — just close the for_each
            // closure and the fn body.
            "    });\n}\n".to_string()
        } else {
            // tiled: close the `ndim` nested cell-fors, then the for_each closure
            // and the fn body.
            let mut s = String::new();
            for _ in 0..ndim {
                s.push_str("    }\n");
            }
            s.push_str("    });\n}\n");
            s
        }
    }
}

/// emit a scalarized stencil kernel as a compilable Rust `pub fn` — the shared
/// driver with the `RustRenderer` spelling, plus a DESCRIPTOR-ABI wrapper. the
/// driver emits the flat `{name}__raw(buf0.., grid.., dom.., buf_extent.., buf_lo..,
/// scalars..)` body; `descriptor_wrapper` then emits the public
/// `{name}(inputs: &[CpuField], outputs: &mut [CpuFieldMut], grid, dom_lo, scalars)`
/// that expands the per-buffer/per-axis args from the shared-domain descriptors —
/// so callers never hand-marshal the ~3*nbuf*ndim integer args (which is unusable
/// at 3D). the `CpuField`/`CpuFieldMut` carry each buffer's `{data, lo, extent}`.
pub fn emit_kernel_cpu(graph: &Graph, inputs: &KernelEmitInputs) -> KernelDescriptor {
    emit_kernel_cpu_with(graph, inputs, &RustRenderer::new())
}

/// SERIAL variant of `emit_kernel_cpu`: the `__raw` body is a plain nested loop on
/// the caller's thread (no `into_par_iter`). build.rs generates a `{name}_serial`
/// alongside the parallel kernel (gated by `SYMBI_GEN_SERIAL`); the cover executor
/// fans a disjoint block cover out over rayon and calls this once per block — one
/// fork-join total. soundness = the cover is a partition (disjoint writes).
pub fn emit_kernel_cpu_serial(graph: &Graph, inputs: &KernelEmitInputs) -> KernelDescriptor {
    emit_kernel_cpu_with(graph, inputs, &RustRenderer::serial())
}

fn emit_kernel_cpu_with(
    graph: &Graph,
    inputs: &KernelEmitInputs,
    renderer: &RustRenderer,
) -> KernelDescriptor {
    let mut desc = emit_kernel_render(graph, inputs, renderer);
    // which scalar params are integer (i32/u32) — they come from the wrapper's
    // `ints: &[i32]` lane; float params from `scalars: &[f64]`.
    let scalar_is_int: Vec<bool> = inputs
        .scalar_params
        .iter()
        .map(|name| {
            graph.iter().any(|(_, node, ty)| {
                matches!(&node.op, crate::graph::Op::Param(s) if s.as_str() == name.as_str())
                    && matches!(ty.element, ElementTy::I32 | ElementTy::U32)
            })
        })
        .collect();
    desc.source.push_str(&descriptor_wrapper(
        inputs.kernel_name,
        &desc.field_bindings,
        inputs.ndim as usize,
        inputs.scalar_params,
        &scalar_is_int,
    ));
    desc
}

/// generate the descriptor-ABI wrapper `pub fn {name}(inputs, outputs, grid, dom_lo,
/// scalars)` that unpacks the `CpuField`/`CpuFieldMut` descriptors and calls
/// `{name}__raw` with the flat args in the exact `emit_render` order: buffers
/// (by buffer_index) / grid[aa] / dom_lo[aa] / [ndim>=2] buf_extent[bb][aa] /
/// buf_lo[bb][aa] / scalars. buffers split into inputs (is_output=false -> &[f64])
/// and outputs (is_output=true -> &mut [f64], incl. in-place fields). outputs are
/// `split_first_mut`'d into disjoint &mut, with extent/lo hoisted to locals before
/// the call so the &mut data reborrow doesn't alias the field reads.
fn descriptor_wrapper(
    name: &str,
    bindings: &[crate::emit::FieldBinding],
    _ndim: usize,
    scalars: &[String],
    scalar_is_int: &[bool],
) -> String {
    // Phase 1B-4: `__raw` takes the view-struct refs DIRECTLY. the wrapper just
    // splits outputs into disjoint `&mut` and passes refs through — no more
    // per-axis `lo` / `extent` unpacking, no more 7-args-per-buffer fanout.
    let mut binds: Vec<&crate::emit::FieldBinding> = bindings.iter().collect();
    binds.sort_by_key(|b| b.buffer_index);
    let mut buf_kind: Vec<(bool, usize)> = Vec::with_capacity(binds.len());
    let (mut n_in, mut n_out) = (0usize, 0usize);
    for b in &binds {
        if b.is_output {
            buf_kind.push((true, n_out));
            n_out += 1;
        } else {
            buf_kind.push((false, n_in));
            n_in += 1;
        }
    }
    let mut s = String::new();
    s.push_str(&format!(
        "#[allow(unused_variables)]\n\
         pub fn {name}<S: Scalar + OrderedNumeric + Send + Sync>(inputs: &[CpuField<S>], outputs: &mut [CpuFieldMut<S>], \
         grid: &[u32], dom_lo: &[i32], ints: &[i32], scalars: &[S]) {{\n"
    ));
    // split outputs into disjoint `&mut CpuFieldMut<S>` refs. `__o{k}` is the
    // SAME shape `__raw` wants — no unpacking.
    if n_out > 0 {
        s.push_str("    let mut __rest = &mut *outputs;\n");
        for k in 0..n_out {
            s.push_str(&format!(
                "    let (__o{k}, __next) = __rest.split_first_mut().expect(\"output count\");\n    __rest = __next;\n"
            ));
        }
    }
    s.push_str(&format!("    {name}__raw(\n"));
    // buffers (buffer_index order): one ref per buffer — input is `&inputs[i]`,
    // output is the already-mut `__o{k}` reference.
    for &(is_out, didx) in &buf_kind {
        if is_out {
            s.push_str(&format!("        __o{didx},\n"));
        } else {
            s.push_str(&format!("        &inputs[{didx}],\n"));
        }
    }
    for aa in 0.._ndim {
        s.push_str(&format!("        grid[{aa}] as i32,\n"));
    }
    for aa in 0.._ndim {
        s.push_str(&format!("        dom_lo[{aa}],\n"));
    }
    // route scalar params by element: int (i32/u32) -> ints lane, else scalars.
    let (mut int_idx, mut float_idx) = (0usize, 0usize);
    for i in 0..scalars.len() {
        if scalar_is_int.get(i).copied().unwrap_or(false) {
            s.push_str(&format!("        ints[{int_idx}],\n"));
            int_idx += 1;
        } else {
            s.push_str(&format!("        scalars[{float_idx}],\n"));
            float_idx += 1;
        }
    }
    s.push_str("    );\n}\n");
    s
}


// the Rust flat index: `(comp - buf_lo) . strides) as usize`. all i64 arithmetic
// (comps, buf_lo, ny/nz are all i64); the single `as usize` is the slice-index
// boundary. `comps` are integer expression strings (see `render_index_expr`).
/// flat buffer index = `__idx_cell_buf{N} + literal_stencil_delta`. delegates to
/// the canonical `emit::emit_flat_index` (shared with the CUDA path).
fn rust_flat_index(ndim: u8, buf: u32, comps: &[String]) -> String {
    let refs: Vec<&str> = comps.iter().map(|s| s.as_str()).collect();
    crate::emit::emit_flat_index(crate::emit::IndexLang::Rust, ndim, buf, &refs)
}

/// emit `let field{bb} = CpuField { data: buf{bb}, lo: [..], extent: [..],
// render a coord index expression in INTEGER space: coord vars (`_coord_N` ->
// `ii`/`jj`/`kk`), integer-valued constants as integer literals, and `+`/`-`/`*`.
// anything else (a float field value used as a gather index, division, a method
// call) panics — the CPU emitter does not silently route integers through float.
fn render_index_expr(e: &ScalarExpr, coord_vars: &[&str]) -> String {
    use ScalarExpr::*;
    match e {
        Var(name) => match name.strip_prefix("_coord_") {
            Some(axis) => {
                let a: usize = axis.parse().unwrap_or_else(|_| panic!("bad coord var '{name}'"));
                coord_vars
                    .get(a)
                    .unwrap_or_else(|| panic!("coord axis {a} out of range for index"))
                    .to_string()
            }
            None => name.clone(),
        },
        Const(ConstValue::F64(v)) => {
            assert!(v.fract() == 0.0, "non-integer constant {v} in an index expression");
            format!("{}", *v as i64)
        }
        Const(ConstValue::F32(v)) => {
            assert!(v.fract() == 0.0, "non-integer constant {v} in an index expression");
            format!("{}", *v as i64)
        }
        Const(ConstValue::I32(v)) => format!("{}", *v as i64),
        Const(ConstValue::U32(v)) => format!("{}", *v as i64),
        // integer arithmetic, and integer comparisons (the condition of a
        // data-INDEPENDENT index branch — e.g. a lattice-map `map_type == 1`).
        // Div is excluded: division in an index is float, not integer.
        BinOp(kind, a, b) if !matches!(kind, BinaryKind::Div) => {
            format!(
                "({} {} {})",
                render_index_expr(a, coord_vars),
                kind.rust_operator(),
                render_index_expr(b, coord_vars),
            )
        }
        // a data-independent integer branch: the lattice-map source coord picks a
        // periodic / reflect / outflow rule by a runtime map_type. cond, then and
        // else are all integer (or integer comparisons), so the result is integer.
        Select { cond, then, else_ } => format!(
            "(if {} {{ {} }} else {{ {} }})",
            render_index_expr(cond, coord_vars),
            render_index_expr(then, coord_vars),
            render_index_expr(else_, coord_vars),
        ),
        UnaryOp(UnaryKind::Neg, a) => format!("(-{})", render_index_expr(a, coord_vars)),
        _ => panic!(
            "emit_kernel_cpu: unsupported index expression — only integer coord arithmetic \
             (coord vars, integer offsets, +/-/*) is supported. data-dependent gather \
             indexing (a float field value used as an index) is not yet implemented for the \
             CPU backend",
        ),
    }
}

// does the scalarized kernel (body + outputs) reference a local named `name`?
// used to skip a dead base cell read (see the base-load loop). runs AFTER the
// FieldLoadAt rewrite, so a field's computed-coord reads are `buf<N>[..]` Vars,
// not the base key — the base read is kept only when the key is genuinely used.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{Precision, Target, TargetConfig};
    use crate::ElementTy;
    use crate::{ConstValue, ElementWiseOp, NodeId, Symbol, TensorTy};

    fn cfg() -> TargetConfig {
        TargetConfig { target: Target::Cuda, precision: Precision::F64 }
    }
    fn scalar_param(g: &mut Graph, name: &str) -> NodeId {
        g.add_param(Symbol::intern(name), TensorTy::scalar(ElementTy::F64), None)
    }

    #[test]
    fn passthrough_1d_emits_rust_fn_with_integer_indices() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "pass_1d",
            ndim:             1,
            target:           cfg(),
            field_inputs:     &[("cons_den".into(), "cons.den".into())],
            scalar_params:    &[],
            field_writes:     &[("prim_den".into(), "prim.den".into(), cons_den)],
            coord_components: &[],
            device_preamble:  &[],
            tile_spec: None,
        });
        assert_eq!(desc.field_bindings.len(), 2);
        assert!(!desc.field_bindings[0].is_output);
        assert!(desc.field_bindings[1].is_output);
        // the unused_parens preamble + the raw kernel and its descriptor wrapper.
        assert!(desc.source.contains("#[allow(unused_parens)]"), "src:\n{}", desc.source);
        assert!(desc.source.contains("pub fn pass_1d__raw<S: Scalar + OrderedNumeric + Send + Sync>("), "src:\n{}", desc.source);
        assert!(desc.source.contains("pub fn pass_1d<S: Scalar + OrderedNumeric + Send + Sync>(inputs: &[CpuField<S>]"), "src:\n{}", desc.source);
        assert!(desc.source.contains("field0: &CpuField<S>"));
        assert!(desc.source.contains("field1: &mut CpuFieldMut<S>"));
        // integer params, integer coord, integer index with one slice-boundary as usize.
        assert!(desc.source.contains("grid_size_0: i32"));
        assert!(desc.source.contains("dom_lo_0: i32"));
        // the DEFAULT CPU emit is cache-TILED: parallelize over N^ndim tile
        // blocks, serial nested loops over each block's cells. 1D: one block
        // dimension, `_i0` from the tile coord + intra-tile offset.
        assert!(desc.source.contains("(0.._ntiles).into_par_iter()"));
        assert!(desc.source.contains("for _d0 in 0.._ts {"));
        assert!(desc.source.contains("let _i0: i32 = _c0 as i32;"));
        assert!(desc.source.contains("let ii: i32 = _i0 + dom_lo_0;"));
        assert!(desc.source.contains("let cons_den: S = field0.data[(__idx_cell_buf0 + ((ii) - ii) * field0.strides[0]) as usize];"));
        assert!(desc.source.contains("field1.data[(__idx_cell_buf1 + ((ii) - ii) * field1.strides[0]) as usize] = cons_den;"));
        // no float-routed indices.
        assert!(!desc.source.contains("as f64 + dom_lo"), "index must not route through f64");
        assert!(!desc.source.contains("as i64"), "indices are i32; no i64 in the kernel");
    }

    #[test]
    fn arithmetic_1d_renders_the_generic_body_expression() {
        // prim_pre = cons_den * 2.0 + cons_nrg  (body math is generic over S: Scalar).
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let cons_nrg = scalar_param(&mut g, "cons_nrg");
        let two = g.add_const(ConstValue::F64(2.0), None);
        let scaled = g.element_wise(ElementWiseOp::Mul, vec![cons_den, two], None);
        let summed = g.element_wise(ElementWiseOp::Add, vec![scaled, cons_nrg], None);
        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "compute_pre_1d",
            ndim:             1,
            target:           cfg(),
            field_inputs:     &[
                ("cons_den".into(), "cons.den".into()),
                ("cons_nrg".into(), "cons.nrg".into()),
            ],
            scalar_params:    &[],
            field_writes:     &[("prim_pre".into(), "prim.pre".into(), summed)],
            coord_components: &[],
            device_preamble:  &[],
            tile_spec: None,
        });
        assert_eq!(desc.field_bindings.len(), 3);
        assert!(desc.source.contains("let cons_den: S = field0.data[(__idx_cell_buf0 + ((ii) - ii) * field0.strides[0]) as usize];"));
        // the f64-built ConstValue(2.0) renders scalar-parametric via Scalar::from_f64.
        assert!(
            desc.source.contains("] = ((cons_den * S::from_f64(2.0)) + cons_nrg);"),
            "src:\n{}", desc.source,
        );
    }

    #[test]
    fn cpu_kernels_are_generic_over_the_scalar() {
        // the CPU kernel is ONE generic `fn k<S: Scalar>` (docs/design/15 §4) — not
        // a monomorphized f64/f32 pair. Sim<f64>/Sim<f32> pick the precision by the
        // buffer type they pass (S inferred); no dispatch. every float spelling is S:
        // buffers &[S], reads `let x: S`, the f64-built ConstValue(2.0) -> S::lit(2.0),
        // the descriptor CpuField<S> + scalars: &[S]. zero f64 leaks; indices stay i32.
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let cons_nrg = scalar_param(&mut g, "cons_nrg");
        let two = g.add_const(ConstValue::F64(2.0), None);
        let scaled = g.element_wise(ElementWiseOp::Mul, vec![cons_den, two], None);
        let summed = g.element_wise(ElementWiseOp::Add, vec![scaled, cons_nrg], None);
        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "compute_pre_1d",
            ndim:             1,
            target:           cfg(),
            field_inputs:     &[
                ("cons_den".into(), "cons.den".into()),
                ("cons_nrg".into(), "cons.nrg".into()),
            ],
            scalar_params:    &[],
            field_writes:     &[("prim_pre".into(), "prim.pre".into(), summed)],
            coord_components: &[],
            device_preamble:  &[],
            tile_spec: None,
        });
        assert!(desc.source.contains("pub fn compute_pre_1d__raw<S: Scalar + OrderedNumeric + Send + Sync>("), "src:\n{}", desc.source);
        assert!(desc.source.contains("field0: &CpuField<S>"), "src:\n{}", desc.source);
        assert!(desc.source.contains("let cons_den: S = field0.data[(__idx_cell_buf0 + ((ii) - ii) * field0.strides[0]) as usize];"));
        assert!(desc.source.contains("] = ((cons_den * S::from_f64(2.0)) + cons_nrg);"), "src:\n{}", desc.source);
        assert!(desc.source.contains("pub fn compute_pre_1d<S: Scalar + OrderedNumeric + Send + Sync>(inputs: &[CpuField<S>]"), "src:\n{}", desc.source);
        assert!(desc.source.contains("outputs: &mut [CpuFieldMut<S>]"));
        // float spelling is fully S; integer index arithmetic stays i32 (never S). no
        // concrete float TYPE leaks — `: f64` annotations / `[f64]` slices. (the bare
        // `f64` inside `Scalar::from_f64` is the constructor name, not a type.)
        assert!(!desc.source.contains(": f64"), "no f64 type annotation:\n{}", desc.source);
        assert!(!desc.source.contains("[f64]"), "no f64 slice:\n{}", desc.source);
        assert!(!desc.source.contains("f32"), "no f32 should appear in a generic kernel:\n{}", desc.source);
        assert!(desc.source.contains("field0.lo[0]"), "indices stay i32:\n{}", desc.source);
    }

    #[test]
    fn in_place_buffer_is_a_single_mut_slice() {
        let mut g = Graph::new();
        let rho = scalar_param(&mut g, "rho");
        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "inplace_1d",
            ndim:             1,
            target:           cfg(),
            field_inputs:     &[("rho".into(), "cons.den".into())],
            scalar_params:    &[],
            field_writes:     &[("rho".into(), "cons.den".into(), rho)],
            coord_components: &[],
            device_preamble:  &[],
            tile_spec: None,
        });
        assert_eq!(desc.field_bindings.len(), 1, "in-place must dedup to one buffer");
        assert!(desc.field_bindings[0].is_output);
        assert!(desc.source.contains("buf0: &mut [S]"), "src:\n{}", desc.source);
    }

    #[test]
    fn diff_stencil_renders_shifted_load_with_integer_index() {
        // the godunov-critical path: a flux difference F[coord+1] - F[coord].
        // the shifted FieldLoadAt must render as a PURE INTEGER index `ii + 1`,
        // not a float `_coord_0 + 1.0` cast back to int.
        use crate::morphism::MorphismKind;
        let mut g = Graph::new();
        let flux = scalar_param(&mut g, "flux"); // base local, the `lo` term
        let c0 = g.add_param(Symbol::intern("_coord_0"), TensorTy::scalar(ElementTy::F64), None);
        let one = g.add_const(ConstValue::F64(1.0), None);
        let c0_hi = g.element_wise(ElementWiseOp::Add, vec![c0, one], None);
        let hi = g.load_at(Symbol::intern("flux"), vec![c0_hi], None);
        let diff = g.morphism(MorphismKind::Diff { axis: 0 }, vec![flux, hi], None); // hi - flux

        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "diff_1d",
            ndim:             1,
            target:           cfg(),
            field_inputs:     &[("flux".into(), "flux".into())],
            scalar_params:    &[],
            field_writes:     &[("out".into(), "out".into(), diff)],
            coord_components: &[0],
            device_preamble:  &[],
            tile_spec: None,
        });
        assert!(desc.source.contains("let flux: S = field0.data[(__idx_cell_buf0 + ((ii) - ii) * field0.strides[0]) as usize];"),
            "src:\n{}", desc.source);
        // the shifted load is a PURE INTEGER index; no f64 anywhere in it.
        // hoisted-base form: base + literal delta.
        assert!(
            desc.source.contains("field0.data[(__idx_cell_buf0 + (((ii + 1)) - ii) * field0.strides[0]) as usize]"),
            "shifted load not a pure integer index:\n{}", desc.source,
        );
        assert!(!desc.source.contains("as i64"), "indices are i32; no i64 cast:\n{}", desc.source);
        assert!(desc.source.contains("- flux);"));
    }

    #[test]
    fn i32_scalar_param_emits_an_integer_signature() {
        // a lattice-map source-coord arg (a shift / pivot / clamp) is an integer
        // param; the emitter must type it i32 from its declared element, not f64,
        // so index arithmetic on it stays integer.
        let mut g = Graph::new();
        let rho = scalar_param(&mut g, "rho");
        g.add_param(Symbol::intern("shift"), TensorTy::scalar(ElementTy::I32), None);
        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "shifted_1d",
            ndim:             1,
            target:           cfg(),
            field_inputs:     &[("rho".into(), "prim.rho".into())],
            scalar_params:    &["shift".into()],
            field_writes:     &[("out".into(), "out".into(), rho)],
            coord_components: &[],
            device_preamble:  &[],
            tile_spec: None,
        });
        assert!(desc.source.contains("shift: i32"), "shift must be i32:\n{}", desc.source);
        assert!(!desc.source.contains("shift: f64"));
    }

    #[test]
    fn render_index_handles_select_and_comparison() {
        // the lattice-map pullback source coord:
        //   if (map_type == 1) { ii + shift } else { pivot2 - ii }
        // must render as a pure-integer if/else index — no float, no cast.
        use ScalarExpr::*;
        let e = Select {
            cond: Box::new(BinOp(
                BinaryKind::Eq,
                Box::new(Var("map_type".into())),
                Box::new(Const(ConstValue::I32(1))),
            )),
            then: Box::new(BinOp(
                BinaryKind::Add,
                Box::new(Var("_coord_0".into())),
                Box::new(Var("shift".into())),
            )),
            else_: Box::new(BinOp(
                BinaryKind::Sub,
                Box::new(Var("pivot2".into())),
                Box::new(Var("_coord_0".into())),
            )),
        };
        let s = render_index_expr(&e, &["ii", "jj", "kk"]);
        assert_eq!(s, "(if (map_type == 1) { (ii + shift) } else { (pivot2 - ii) })");
    }

    #[test]
    fn two_d_declares_integer_stride_locals_and_flat_parallel() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_cpu(&g, &KernelEmitInputs {
            kernel_name:      "pass_2d",
            ndim:             2,
            target:           cfg(),
            field_inputs:     &[("cons_den".into(), "cons.den".into())],
            scalar_params:    &[],
            field_writes:     &[("prim_den".into(), "prim.den".into(), cons_den)],
            coord_components: &[],
            device_preamble:  &[],
            tile_spec: None,
        });
        // Phase 1B-4: field0 is the input — it's a `&CpuField<S>` kernel arg.
        // body indexes via `field0.lo[..]` / `field0.strides[..]` / `field0.data`.
        assert!(desc.source.contains("field0: &CpuField<S>"));
        assert!(desc.source.contains("field0.strides[0]"));
        // 2D default emit is cache-TILED: parallelize over 2D tile blocks with
        // nested intra-tile loops (one per axis). axis 1 comes from its own
        // `for _d1` loop + tile coord, not a flat unflatten.
        assert!(desc.source.contains("(0.._ntiles).into_par_iter()"));
        assert!(desc.source.contains("for _d1 in 0.._ts {"));
        assert!(desc.source.contains("let _i1: i32 = _c1 as i32;"));
        assert!(desc.source.contains("let jj: i32 = _i1 + dom_lo_1;"));
    }
}
