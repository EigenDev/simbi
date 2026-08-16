// =============================================================================
// emit_kernel_cpu.rs
//
// the CPU-native (Rust) sibling of emit_kernel.rs's `emit_kernel_from_lowering`
// (which emits a CUDA `__global__`). this is the build-time AOT path for the CPU
// backend: a scalarized stencil kernel -> a compilable Rust
// `pub fn` that iterates the dispatch window over `&[f64]` / `&mut [f64]`
// buffers, with the same flat-index ABI as the CUDA emitter (coord is absolute;
// access is `buf[(coord - buf_lo) . strides]`).
//
// it reuses the proven `emit_cpu` ScalarExpr/ScalarStmt renderer for the f64
// body and writes, and the same buffer-binding + FieldLoadAt-resolution
// structure as the CUDA emitter; the syntax (Rust slices, a cell loop) is the
// whole of the difference. precision is f64: the substrate is double-precision.
//
// indices are integers. coord index params are `I32` in the IR, so coord and
// index arithmetic is pure `i32` — coord vars, strides, and buffer-lo offsets are
// all `i32`; integer stencil shifts render as integer literals (`+ 1`); a
// CSE'd shared shift is an `i32` body let, so a multi-law kernel's
// indices stay integer. the sole conversion is the `as usize` at the slice-index
// site, which Rust's `Index<usize>` requires. (32-bit indices match the existing
// CUDA `(int)` flat-index ABI.) data-dependent gather indexing (a float field
// value used as an index) panics loudly here, so integer arithmetic stays in
// integer space.
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
// aliasing: a buffer both read and written (in-place conserved update) is one
// `&mut [f64]` param; reads go through it into f64 locals ahead of the store, so
// every borrow is released by the time the write happens.
// =============================================================================

/// the default CPU cache-tile edge. the emitter parallelizes over `N^ndim` cache
/// blocks with serial nested loops inside each block, keeping a block's stencil
/// neighborhood + multi-field working set resident. measured ~1.4-2.1x full-step
/// over the flat emit, growing with grid size, and grid-size-independent
/// throughput. 8 and 16 both measured ~optimal; 8 is the
/// conservative default (fits closer to L1/L2).
const CPU_TILE: usize = 8;

/// cache-tile edge length for the CPU emit. default = `CPU_TILE` (tiled).
/// `SYMBI_TILE_CPU=0` forces the flat parallel-over-all-cells emit (debug / A/B);
/// `SYMBI_TILE_CPU=N` overrides the tile edge. read at emit time and tracked by
/// `symbi-aot/build.rs` (`rerun-if-env-changed`), so toggling regenerates kernels.
fn cpu_tile_size() -> usize {
    // debug-emit-knobs gates the SYMBI_TILE_CPU a/b knob; feature off (default) = canonical env-unset shape (CPU_TILE, tiled).
    #[cfg(feature = "debug-emit-knobs")]
    {
        match std::env::var("SYMBI_TILE_CPU")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(0) | Some(1) => 0, // explicit flat override
            Some(t) => t,           // explicit tile edge
            None => CPU_TILE,       // default: tiled
        }
    }
    #[cfg(not(feature = "debug-emit-knobs"))]
    {
        CPU_TILE
    }
}

/// SYMBI_UNCHECKED_LOADS=1: emit field loads/stores through `get_unchecked`; the
/// default spelling is the bounds-checked `data[idx]`. the per-cell index is computed
/// from the cell coord + strides + stencil offset, always within the buffer's allocated
/// domain (the kernel's correctness contract, validated by the carrier oracle), so the
/// bounds check is dead weight — and an opaque-index check defeats vectorization. read at
/// emit time (build.rs tracks the env); default off (safe checked indexing).
/// dropping the checks is worth ~8x on the scalar loop, and puts the whole memory-safety
/// burden on the index contract.
fn unchecked_loads() -> bool {
    // vec mode needs the bounds-check branches gone to vectorize, so it implies unchecked.
    std::env::var("SYMBI_UNCHECKED_LOADS")
        .map(|v| v == "1")
        .unwrap_or(false)
        || vec_loop()
}

/// SYMBI_VEC_LOOP=1 (ndim>=2): emit the row-parallel loop — parallelize over the
/// outer (non-contiguous) axes, with a countable inner loop walking the contiguous
/// last axis. combined with the unit-stride index (emit::emit_*_index) this is the
/// shape LLVM loop-vectorizes (the simd-spike form). 1D / coalesce-incompatible
/// kernels fall back to flat/tiled. read at emit time; build.rs tracks the env.
fn vec_loop() -> bool {
    // debug-emit-knobs gates the SYMBI_VEC_LOOP a/b knob; feature off (default) = canonical env-unset shape (false).
    #[cfg(feature = "debug-emit-knobs")]
    {
        std::env::var("SYMBI_VEC_LOOP")
            .map(|v| v == "1")
            .unwrap_or(false)
    }
    #[cfg(not(feature = "debug-emit-knobs"))]
    {
        false
    }
}

use crate::backends::cpu::{emit_expr, emit_stmt, rust_type_name};
use crate::backends::kernel::KernelEmitInputs;
use crate::backends::render::{COORD_VARS, KernelRenderer};
use crate::emit::KernelDescriptor;
use crate::graph::ConstValue;
use crate::passes::scalarize::{BinaryKind, ScalarExpr, ScalarStmt, UnaryKind};
use crate::{ElementTy, Graph};

/// the Rust (CPU) backend: the per-language spelling for the shared kernel driver
/// (`emit_render`). produces a compilable generic `pub fn k<S: Scalar>` over `&[S]`
/// slices with pure-integer indices (one `as usize` at the slice boundary). the
/// float scalar is the type parameter `S` (`Sim<f64>`/`Sim<f32>`
/// pick it by the buffer type they pass); constants render `S::lit(..)`, math
/// resolves to the `Scalar` trait. one kernel serves every precision, with the
/// scalar type resolved statically at the call site.
///
/// rayon-parallel outer cell loop. the renderer tracks which buffer
/// indices are mutable (outputs / in-place fields) as they're emitted via
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
    /// serial variant: emit a plain nested `for` loop over the exec window; the
    /// kernel runs entirely on the caller's thread. the executor owns the
    /// parallelism: it fans a disjoint cover (a BlockGrid / guillotine cover) out
    /// over rayon and calls this serial kernel per block, so the cover dispatches
    /// in one fork-join across all blocks. soundness rests on the cover being a
    /// partition (disjoint output writes) — the proven law.
    serial: bool,
    /// whether the mask-form rewrite was applied to this kernel (set via
    /// `note_mask_form`). in a mask-formed body every bool local holds a
    /// `cmp_*` result, so its type is `<S as Scalar>::Mask`;
    /// fallback (control-flow) kernels keep native bools.
    mask_applied: std::cell::Cell<bool>,
}

impl RustRenderer {
    pub fn new() -> Self {
        Self {
            mut_buf_indices: std::cell::RefCell::new(Vec::new()),
            serial: false,
            mask_applied: std::cell::Cell::new(false),
        }
    }
    pub fn serial() -> Self {
        Self {
            mut_buf_indices: std::cell::RefCell::new(Vec::new()),
            serial: true,
            mask_applied: std::cell::Cell::new(false),
        }
    }
}
impl Default for RustRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelRenderer for RustRenderer {
    fn preamble(&self, _device_preamble: &[String]) -> String {
        // the ScalarExpr renderer parenthesizes every BinOp; allow the redundant
        // outer pair; minimal-paren code would be precedence-fragile.
        // (device_preamble is a GPU device-function concept; the CPU path ignores it.)
        "#[allow(unused_parens)]\n".to_string()
    }
    fn buffer_param(&self, idx: u32, is_output: bool) -> String {
        // per-buffer arg is one view-struct reference. `lo`, `extent`, and the
        // pre-multiplied `strides` all ride inside the struct, so a buffer's whole
        // layout travels as a single argument
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
        // the mut-buffer ledger carries through untouched: the driver calls
        // `buffer_param` ahead of this point, and each emission constructs a fresh
        // `RustRenderer::new()`, so the state is per-emission by construction.
        // `+ Send + Sync` so rayon par_iter can capture &[S] inputs +
        // S scalar params, and so the unsafe raw-ptr Send wrapper for mut
        // buffers compiles (the bound on T inside `unsafe impl Send`).
        // `__raw` is the positional ABI core — codegen-internal. its arity changes
        // whenever a builder adds or removes an input, so hand-written positional
        // callers drift silently. host + test code calls the name-keyed `NamedKernel`
        // (symbi-aot), which binds by manifest field name. `#[doc(hidden)]`
        // keeps the slice-form wrapper + the registry working while holding `__raw`
        // out of the discoverable API surface.
        // `unused_unsafe`: with SYMBI_UNCHECKED_LOADS the per-access `unsafe`
        // blocks nest (a stencil load inlined into a store's unsafe rhs) — the
        // inner ones are redundant but correct; this is machine-generated code.
        format!(
            "#[doc(hidden)]\n#[allow(non_snake_case, unused_variables, unused_unsafe)]\npub fn {name}__raw<S: Scalar + OrderedNumeric + Send + Sync>(\n"
        )
    }
    fn params_close(&self) -> &'static str {
        ",\n) {\n" // Rust allows the trailing comma
    }
    fn cell_prelude(&self, ndim: usize, _n_buffers: u32) -> Vec<String> {
        // the kernel signature takes the view structs directly:
        // `field0: &CpuField<S>, ..., field_n: &mut CpuFieldMut<S>`. the structs
        // arrive whole, so input fields capture by ref into the closure and
        // outputs use the standard mut-ptr-rebind dance.
        //
        // body emission (cell-base, flat index, base reads, stores) all spell
        // `field{N}.lo[..]`/`.strides[..]`/`.data` directly — a single source of
        // truth (`CpuField` / `CpuFieldMut` in `symbi-aot`).
        let mut v = Vec::new();

        // serial variant: plain nested `for` over the exec window, on the caller's
        // thread. `field{N}` (incl. the `&mut CpuFieldMut` outputs) is used directly.
        // the cover executor parallelizes over many such blocks, paying one
        // fork-join total.
        if self.serial {
            // the nest order is derived from the layout: `nest_order` puts
            // `CONTIGUOUS_AXIS` innermost, so the cell index advances by one element per iteration of
            // the hot loop. nesting the contiguous axis outermost is equally correct and strides the
            // hot loop by `extent[0]*extent[1]`, a difference correctness tests are blind to.
            for aa in symbi_algebra::nest_order(ndim) {
                v.push(format!("    for _i{aa} in 0..(grid_size_{aa} as i32) {{"));
                v.push(format!(
                    "    let {}: i32 = _i{aa} + dom_lo_{aa};",
                    COORD_VARS[aa]
                ));
            }
            return v;
        }

        let mut_buf = self.mut_buf_indices.borrow();

        // for mut buffers: hoist `lo` + `strides` arrays (Copy `[i32; 4]`) so
        // the par_iter closure captures them by value; capture data ptr in
        // `__MutBufPtr`. inside the closure the CpuFieldMut is rebuilt.
        for &bi in mut_buf.iter() {
            v.push(format!("    let __field{bi}_lo: [i32; 4] = field{bi}.lo;"));
            v.push(format!(
                "    let __field{bi}_strides: [i32; 4] = field{bi}.strides;"
            ));
        }

        // --- rayon-parallel outer loop scaffold ---
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
        // across the registry). the flat loop calls `.with_min_len(16)`, which lives
        // on IndexedParallelIterator, so that branch imports the trait as well.
        if cpu_tile_size() == 0 {
            v.push("    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};".to_string());
        } else {
            v.push("    use rayon::iter::{IntoParallelIterator, ParallelIterator};".to_string());
        }

        // helper: rebind each mut buffer's raw ptr to a fresh CpuFieldMut inside
        // the closure (shadows the borrowed param; the reborrow lives entirely
        // inside one worker's closure body).
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
            // vectorizable row mode (SYMBI_VEC_LOOP): parallelize over the transverse axes; a
            // single countable inner loop walks `CONTIGUOUS_AXIS`. paired with the unit-stride index
            // (`emit::emit_cell_index_base`) the inner trip advances the cell offset by exactly one
            // element, which is the shape LLVM's loop-vectorizer turns into whole-vector loads.
            //
            // the inner axis is derived from the layout. it is axis 0 under `strides_from_extent`,
            // the contiguous axis under this column-major layout (C row-major would put it last):
            // pointing this loop at the last axis leaves the kernel correct and
            // strided, and the debug_assert below is what reports it.
            let inner = symbi_algebra::CONTIGUOUS_AXIS;
            // outer axes, ascending, so the one adjacent to `inner` in memory varies fastest —
            // the same nesting `nest_order` gives, minus the innermost.
            let outer: Vec<usize> = (0..ndim).filter(|&a| a != inner).collect();
            let outer_expr = outer
                .iter()
                .map(|aa| format!("(grid_size_{aa} as usize)"))
                .collect::<Vec<_>>()
                .join(" * ");
            v.push(format!("    let _orows: usize = {outer_expr};"));
            v.push("    (0.._orows).into_par_iter().for_each(|_orow| {".to_string());
            push_rebind(&mut v);
            // unflatten _orow -> outer coords, canonical order (lowest outer axis fastest).
            if outer.len() == 1 {
                v.push(format!("        let _i{}: i32 = _orow as i32;", outer[0]));
            } else {
                v.push("        let mut _orem: usize = _orow;".to_string());
                for &aa in &outer {
                    v.push(format!(
                        "        let _i{aa}: i32 = (_orem % (grid_size_{aa} as usize)) as i32;"
                    ));
                    v.push(format!("        _orem /= grid_size_{aa} as usize;"));
                }
            }
            for &aa in &outer {
                v.push(format!(
                    "        let {}: i32 = _i{aa} + dom_lo_{aa};",
                    COORD_VARS[aa]
                ));
            }
            // unit-stride precondition; debug-only, the carrier oracle gates correctness.
            v.push(format!(
                "        debug_assert!(field0.strides[{inner}] == 1, \"the vectorized inner loop must walk the contiguous axis\");"
            ));
            // countable inner loop over the contiguous axis (the vectorized dim).
            v.push(format!(
                "        for _ic in 0..(grid_size_{inner} as usize) {{"
            ));
            v.push(format!("        let _i{inner}: i32 = _ic as i32;"));
            v.push(format!(
                "        let {}: i32 = _i{inner} + dom_lo_{inner};",
                COORD_VARS[inner]
            ));
        } else if cpu_tile_size() == 0 {
            // flat (default): parallelize over the flattened interior (every
            // cell) — exposes all cells so the grid
            // scales like the 1D path (mirrors the GPU global thread index).
            let total_expr = (0..ndim)
                .map(|aa| format!("(grid_size_{aa} as usize)"))
                .collect::<Vec<_>>()
                .join(" * ");
            v.push(format!("    let _ptotal: usize = {total_expr};"));
            v.push(
                "    (0.._ptotal).into_par_iter().with_min_len(16).for_each(|_flat| {".to_string(),
            );
            push_rebind(&mut v);
            // the emitted flat -> coord map mirrors `symbi_algebra::unflatten`: CONTIGUOUS_AXIS peels
            // first because it varies fastest, and the outermost axis takes the remainder. peeling
            // the outermost axis first walks the flat index along the slowest memory axis — correct,
            // cell-disjoint, and strided by extent[0]*extent[1] on every cell.
            let peel: Vec<usize> = symbi_algebra::nest_order(ndim).rev().collect();
            if ndim == 1 {
                v.push(format!("        let _i{}: i32 = _flat as i32;", peel[0]));
            } else {
                v.push("        let mut _rem: usize = _flat;".to_string());
                for &aa in &peel[..ndim - 1] {
                    v.push(format!(
                        "        let _i{aa}: i32 = (_rem % (grid_size_{aa} as usize)) as i32;"
                    ));
                    v.push(format!("        _rem /= grid_size_{aa} as usize;"));
                }
                v.push(format!(
                    "        let _i{}: i32 = _rem as i32;",
                    peel[ndim - 1]
                ));
            }
            for aa in 0..ndim {
                v.push(format!(
                    "    let {}: i32 = _i{aa} + dom_lo_{aa};",
                    COORD_VARS[aa]
                ));
            }
        } else {
            // tiled (SYMBI_TILE_CPU=N): parallelize over cache blocks; serial
            // nested loops over each block's cells keep its stencil
            // neighborhood in cache. recovers the per-cell cache-miss penalty
            // once the grid working set exceeds cache. per-tile
            // rebind (amortized over the block's cells).
            //
            // the contiguous axis runs its full extent (ndim >= 2): the vectorized
            // (mask-form/slp) bodies need long unit-stride inner trips —
            // edge-length trips invert their win (the
            // same law as the cover executor's row-elongated blocks). 1d tiles
            // its only axis, which is what exposes the parallelism there.
            let tile = cpu_tile_size();
            let contig = symbi_algebra::CONTIGUOUS_AXIS;
            let tiled_axes: Vec<usize> =
                (0..ndim).filter(|&aa| ndim == 1 || aa != contig).collect();
            v.push(format!("    let _ts: usize = {tile};"));
            for &aa in &tiled_axes {
                v.push(format!(
                    "    let _nt_{aa}: usize = ((grid_size_{aa} as usize) + _ts - 1) / _ts;"
                ));
            }
            let ntiles_expr = tiled_axes
                .iter()
                .map(|aa| format!("_nt_{aa}"))
                .collect::<Vec<_>>()
                .join(" * ");
            v.push(format!("    let _ntiles: usize = {ntiles_expr};"));
            v.push("    (0.._ntiles).into_par_iter().for_each(|_tile| {".to_string());
            push_rebind(&mut v);
            // the tile index -> tile coord map, peeled in the same canonical order as the cell map so
            // consecutive tile indices are adjacent in memory: a rayon worker that takes a contiguous
            // slice of tile indices then sweeps adjacent rows in memory.
            let peel: Vec<usize> = symbi_algebra::nest_order(ndim)
                .rev()
                .filter(|aa| tiled_axes.contains(aa))
                .collect();
            if peel.len() == 1 {
                v.push(format!("        let _tc{}: usize = _tile;", peel[0]));
            } else {
                v.push("        let mut _trem: usize = _tile;".to_string());
                for &aa in &peel[..peel.len() - 1] {
                    v.push(format!("        let _tc{aa}: usize = _trem % _nt_{aa};"));
                    v.push(format!("        _trem /= _nt_{aa};"));
                }
                v.push(format!(
                    "        let _tc{}: usize = _trem;",
                    peel[peel.len() - 1]
                ));
            }
            // nested cell loops within the tile; declare each axis's coord right
            // after its index. `break` on the boundary handles partial tiles
            // (`_c{aa}` is monotonic in `_d{aa}`). close() emits ndim matching `}`.
            // the nest order is derived from the layout, as in the serial branch; each `break`
            // exits its own axis's loop, so partial tiles stay correct under any nesting order.
            // the untiled contiguous axis is a plain full-extent loop.
            for aa in symbi_algebra::nest_order(ndim) {
                if !tiled_axes.contains(&aa) {
                    v.push(format!(
                        "        for _i{aa} in 0..(grid_size_{aa} as i32) {{"
                    ));
                    v.push(format!(
                        "        let {}: i32 = _i{aa} + dom_lo_{aa};",
                        COORD_VARS[aa]
                    ));
                    continue;
                }
                v.push(format!("        for _d{aa} in 0.._ts {{"));
                v.push(format!(
                    "        let _c{aa}: usize = _tc{aa} * _ts + _d{aa};"
                ));
                v.push(format!(
                    "        if _c{aa} >= grid_size_{aa} as usize {{ break; }}"
                ));
                v.push(format!("        let _i{aa}: i32 = _c{aa} as i32;"));
                v.push(format!(
                    "        let {}: i32 = _i{aa} + dom_lo_{aa};",
                    COORD_VARS[aa]
                ));
            }
        }
        v
    }
    fn coord_decl(&self, axis: u8, _element: ElementTy) -> String {
        // the cell index is always i32; physical-space reals come from promoting
        // `index * dx` (the graph's usual arithmetic conversions).
        format!(
            "    let _coord_{axis}: i32 = {};",
            COORD_VARS[axis as usize]
        )
    }
    fn index_lang(&self) -> crate::emit::IndexLang {
        crate::emit::IndexLang::Rust
    }
    // branch-free cmp/select spelling (passes::mask_form). default on, made
    // safe by the pass's arm-cost gate: `select` computes both arms of every
    // conditional, so a kernel whose select arms divide or call out (the hllc
    // star-state fan: measured 31 -> 79 ns/zone mask-formed) keeps the bool/if
    // spelling, while cheap-arm bodies (the clamped-hlle flux: 18.5 -> 13.3
    // ns/zone combined with unswitching) vectorize.
    fn mask_form(&self) -> bool {
        // debug-emit-knobs gates the SYMBI_MASK_FORM=0 opt-out a/b knob;
        // feature off (default) = mask form on, cost-gated.
        #[cfg(feature = "debug-emit-knobs")]
        {
            std::env::var("SYMBI_MASK_FORM")
                .map(|v| v != "0")
                .unwrap_or(true)
        }
        #[cfg(not(feature = "debug-emit-knobs"))]
        {
            true
        }
    }
    fn note_mask_form(&self, applied: bool) {
        self.mask_applied.set(applied);
    }
    fn skip_scattered_buffer_layout_args(&self) -> bool {
        true
    }
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
        // one indexing method, every emitter.
        if unchecked_loads() {
            format!("(unsafe {{ *field{buf}.data.get_unchecked({flat}) }})")
        } else {
            format!("field{buf}.data[{flat}]")
        }
    }
    fn render_stmt(&self, stmt: &ScalarStmt) -> String {
        let mut s = String::from("    ");
        emit_stmt(&mut s, stmt, true);
        // in a mask-formed body every bool local holds a `cmp_*` result — an
        // `S::Mask`. the shared emit_stmt spells the native
        // type; retype here so the CUDA/elemental spellings stay untouched.
        if self.mask_applied.get() {
            s = s.replace(": bool = ", ": <S as Scalar>::Mask = ");
        }
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
            // flat: the parallel for_each is the whole nest, so closing it and
            // the fn body suffices.
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
/// driver with the `RustRenderer` spelling, plus a descriptor-ABI wrapper. the
/// driver emits the flat `{name}__raw(buf0.., grid.., dom.., buf_extent.., buf_lo..,
/// scalars..)` body; `descriptor_wrapper` then emits the public
/// `{name}(inputs: &[CpuField], outputs: &mut [CpuFieldMut], grid, dom_lo, scalars)`
/// that expands the per-buffer/per-axis args from the shared-domain descriptors,
/// so a caller passes descriptors and the wrapper marshals the ~3*nbuf*ndim integer
/// args (a fanout that is unusable by hand at 3D). the `CpuField`/`CpuFieldMut`
/// carry each buffer's `{data, lo, extent}`.
pub fn emit_kernel_cpu(graph: &Graph, inputs: &KernelEmitInputs) -> KernelDescriptor {
    emit_kernel_cpu_with(graph, inputs, &RustRenderer::new())
}

/// serial variant of `emit_kernel_cpu`: the `__raw` body is a plain nested loop on
/// the caller's thread. build.rs generates a `{name}_serial`
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
    let prepared = crate::backends::render::prepare(graph, inputs);

    // loop unswitching (passes::unswitch): a select gated on a param-only
    // condition (the limiter pick `theta < 0`) takes the same arm in every
    // cell. render two specialized loop nests plus a dispatcher that branches
    // once per kernel call — each specialization is rendered independently, so
    // mask_form's arm-cost gate applies per branch (the division-heavy
    // van-Leer body keeps bool/if; the cheap-arm minmod body vectorizes).
    // specialization is bit-identical: select(true, t, f) == t by definition.
    let scalar_names: std::collections::HashSet<String> =
        prepared.scalar_params.iter().map(|b| b.name()).collect();
    let mut desc =
        if let Some(cand) = crate::passes::unswitch::find(&prepared.scalarized, &scalar_names) {
            let fresh = || {
                if renderer.serial {
                    RustRenderer::serial()
                } else {
                    RustRenderer::new()
                }
            };
            let mut p_t = prepared.clone();
            p_t.kernel_name = format!("{}__uswt", prepared.kernel_name);
            crate::passes::unswitch::specialize(&mut p_t.scalarized, &cand.cond_let, true);
            let mut p_f = prepared.clone();
            p_f.kernel_name = format!("{}__uswf", prepared.kernel_name);
            crate::passes::unswitch::specialize(&mut p_f.scalarized, &cand.cond_let, false);
            let dispatcher = unswitch_dispatcher(&prepared, &cand);
            let d_t = crate::backends::render::render(p_t, &fresh());
            let d_f = crate::backends::render::render(p_f, &fresh());
            KernelDescriptor {
                source: format!("{}\n{}\n{}", d_t.source, d_f.source, dispatcher),
                kernel_name: prepared.kernel_name,
                field_bindings: d_t.field_bindings,
                param_names: d_t.param_names,
                scalar_is_int: d_t.scalar_is_int,
                tile_spec: d_t.tile_spec,
            }
        } else {
            crate::backends::render::render(prepared, renderer)
        };
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

/// generate the unswitch dispatcher `pub fn {name}__raw(...)`: the same positional
/// signature as a rendered `__raw` kernel (buffers in binding order, grid, dom_lo,
/// scalars in declared order — mirroring `render_source`), forwarding every arg to
/// the `__uswt`/`__uswf` specialization picked by the param-only condition. one
/// branch per kernel call; the loop nests inside the specializations hold the
/// resolved arm inline.
fn unswitch_dispatcher(
    prepared: &crate::backends::render::Prepared,
    cand: &crate::passes::unswitch::Candidate,
) -> String {
    let name = &prepared.kernel_name;
    let ndim = prepared.ndim as usize;
    let mut binds: Vec<&crate::emit::FieldBinding> = prepared.bindings.iter().collect();
    binds.sort_by_key(|b| b.buffer_index);
    let mut params: Vec<String> = Vec::new();
    let mut args: Vec<String> = Vec::new();
    for b in &binds {
        let i = b.buffer_index;
        if b.is_output {
            params.push(format!("    field{i}: &mut CpuFieldMut<S>"));
        } else {
            params.push(format!("    field{i}: &CpuField<S>"));
        }
        args.push(format!("field{i}"));
    }
    for aa in 0..ndim {
        params.push(format!("    grid_size_{aa}: i32"));
        args.push(format!("grid_size_{aa}"));
    }
    for aa in 0..ndim {
        params.push(format!("    dom_lo_{aa}: i32"));
        args.push(format!("dom_lo_{aa}"));
    }
    for bind in &prepared.scalar_params {
        let pn = bind.name();
        let elem = prepared
            .param_elem
            .get(&pn)
            .copied()
            .unwrap_or(crate::ElementTy::F64);
        params.push(format!("    {pn}: {}", rust_type_name(elem, true)));
        args.push(pn);
    }
    let mut cond = String::new();
    emit_expr(&mut cond, &cand.cond_expr, true);
    let arg_list = args.join(", ");
    format!(
        "#[doc(hidden)]\n#[allow(non_snake_case, unused_variables, unused_parens)]\n\
         pub fn {name}__raw<S: Scalar + OrderedNumeric + Send + Sync>(\n{}\n) {{\n    \
         if {cond} {{\n        {name}__uswt__raw({arg_list});\n    }} else {{\n        \
         {name}__uswf__raw({arg_list});\n    }}\n}}\n",
        params.join(",\n"),
    )
}

/// generate the descriptor-ABI wrapper `pub fn {name}(inputs, outputs, grid, dom_lo,
/// scalars)` that unpacks the `CpuField`/`CpuFieldMut` descriptors and calls
/// `{name}__raw` with the flat args in the exact `emit_render` order: buffers
/// (by buffer_index) / grid[aa] / dom_lo[aa] / [ndim>=2] buf_extent[bb][aa] /
/// buf_lo[bb][aa] / scalars. buffers split into inputs (is_output=false -> &[f64])
/// and outputs (is_output=true -> &mut [f64], incl. in-place fields). outputs are
/// `split_first_mut`'d into disjoint &mut, with extent/lo hoisted to locals before
/// the call so the &mut data reborrow stays disjoint from the field reads.
fn descriptor_wrapper(
    name: &str,
    bindings: &[crate::emit::FieldBinding],
    _ndim: usize,
    scalars: &[String],
    scalar_is_int: &[bool],
) -> String {
    // `__raw` takes the view-struct refs directly, so the wrapper's whole job is
    // splitting outputs into disjoint `&mut` and passing the refs through; each
    // buffer's per-axis `lo` / `extent` travels inside its struct.
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
    // split outputs into disjoint `&mut CpuFieldMut<S>` refs. `__o{k}` is already
    // the shape `__raw` takes.
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
// render a coord index expression in integer space: coord vars (`_coord_N` ->
// `ii`/`jj`/`kk`), integer-valued constants as integer literals, and `+`/`-`/`*`.
// anything else (a float field value used as a gather index, division, a method
// call) panics loudly, keeping index arithmetic in integer space.
fn render_index_expr(e: &ScalarExpr, coord_vars: &[&str]) -> String {
    use ScalarExpr::*;
    match e {
        Var(name) => match name.strip_prefix("_coord_") {
            Some(axis) => {
                let a: usize = axis
                    .parse()
                    .unwrap_or_else(|_| panic!("bad coord var '{name}'"));
                coord_vars
                    .get(a)
                    .unwrap_or_else(|| panic!("coord axis {a} out of range for index"))
                    .to_string()
            }
            None => name.clone(),
        },
        Const(ConstValue::F64(v)) => {
            assert!(
                v.fract() == 0.0,
                "non-integer constant {v} in an index expression"
            );
            format!("{}", *v as i64)
        }
        Const(ConstValue::F32(v)) => {
            assert!(
                v.fract() == 0.0,
                "non-integer constant {v} in an index expression"
            );
            format!("{}", *v as i64)
        }
        Const(ConstValue::I32(v)) => format!("{}", *v as i64),
        Const(ConstValue::U32(v)) => format!("{}", *v as i64),
        // integer arithmetic, and integer comparisons (the condition of a
        // data-independent index branch — e.g., a lattice-map `map_type == 1`).
        // division in an index is float, so `Div` falls through to the panic arm.
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
// used to skip a dead base cell read (see the base-load loop). runs after the
// FieldLoadAt rewrite, so a field's computed-coord reads are `buf<N>[..]` Vars;
// the base read is kept only when the key is genuinely used.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementTy;
    use crate::emit::{Precision, Target, TargetConfig};
    use crate::{ConstValue, ElementWiseOp, NodeId, Symbol, TensorTy};

    fn cfg() -> TargetConfig {
        TargetConfig {
            target: Target::Cuda,
            precision: Precision::F64,
        }
    }
    fn scalar_param(g: &mut Graph, name: &str) -> NodeId {
        g.add_param(Symbol::intern(name), TensorTy::scalar(ElementTy::F64), None)
    }

    #[test]
    fn passthrough_1d_emits_rust_fn_with_integer_indices() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "pass_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[("cons_den".into(), "cons.den".into())],
                scalar_params: &[],
                field_writes: &[("prim_den".into(), "prim.den".into(), cons_den)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert_eq!(desc.field_bindings.len(), 2);
        assert!(!desc.field_bindings[0].is_output);
        assert!(desc.field_bindings[1].is_output);
        // the unused_parens preamble + the raw kernel and its descriptor wrapper.
        assert!(
            desc.source.contains("#[allow(unused_parens)]"),
            "src:\n{}",
            desc.source
        );
        assert!(
            desc.source
                .contains("pub fn pass_1d__raw<S: Scalar + OrderedNumeric + Send + Sync>("),
            "src:\n{}",
            desc.source
        );
        assert!(
            desc.source.contains(
                "pub fn pass_1d<S: Scalar + OrderedNumeric + Send + Sync>(inputs: &[CpuField<S>]"
            ),
            "src:\n{}",
            desc.source
        );
        assert!(desc.source.contains("field0: &CpuField<S>"));
        assert!(desc.source.contains("field1: &mut CpuFieldMut<S>"));
        // integer params, integer coord, integer index with one slice-boundary as usize.
        assert!(desc.source.contains("grid_size_0: i32"));
        assert!(desc.source.contains("dom_lo_0: i32"));
        // the default CPU emit is cache-tiled: parallelize over N^ndim tile
        // blocks, serial nested loops over each block's cells. 1D: one block
        // dimension, `_i0` from the tile coord + intra-tile offset.
        assert!(desc.source.contains("(0.._ntiles).into_par_iter()"));
        assert!(desc.source.contains("for _d0 in 0.._ts {"));
        assert!(desc.source.contains("let _i0: i32 = _c0 as i32;"));
        assert!(desc.source.contains("let ii: i32 = _i0 + dom_lo_0;"));
        assert!(
            desc.source.contains(
                "let cons_den: S = field0.data[(__idx_cell_buf0 + ((ii) - ii)) as usize];"
            )
        );
        assert!(
            desc.source
                .contains("field1.data[(__idx_cell_buf1 + ((ii) - ii)) as usize] = cons_den;")
        );
        // indices stay in integer space.
        assert!(
            !desc.source.contains("as f64 + dom_lo"),
            "index must not route through f64"
        );
        assert!(
            !desc.source.contains("as i64"),
            "indices are i32; no i64 in the kernel"
        );
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
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "compute_pre_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[
                    ("cons_den".into(), "cons.den".into()),
                    ("cons_nrg".into(), "cons.nrg".into()),
                ],
                scalar_params: &[],
                field_writes: &[("prim_pre".into(), "prim.pre".into(), summed)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert_eq!(desc.field_bindings.len(), 3);
        assert!(
            desc.source.contains(
                "let cons_den: S = field0.data[(__idx_cell_buf0 + ((ii) - ii)) as usize];"
            )
        );
        // the f64-built ConstValue(2.0) renders scalar-parametric via Scalar::from_f64.
        assert!(
            desc.source
                .contains("] = ((cons_den * S::from_f64(2.0)) + cons_nrg);"),
            "src:\n{}",
            desc.source,
        );
    }

    #[test]
    fn cpu_kernels_are_generic_over_the_scalar() {
        // the CPU kernel is one generic `fn k<S: Scalar>` compiled once over the
        // scalar type. Sim<f64>/Sim<f32> pick the precision by the
        // buffer type they pass (S inferred), resolved statically. every float
        // spelling is S: buffers &[S], reads `let x: S`, the f64-built
        // ConstValue(2.0) -> S::lit(2.0), the descriptor CpuField<S> + scalars: &[S].
        // every float type in the emitted source is S; indices stay i32.
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let cons_nrg = scalar_param(&mut g, "cons_nrg");
        let two = g.add_const(ConstValue::F64(2.0), None);
        let scaled = g.element_wise(ElementWiseOp::Mul, vec![cons_den, two], None);
        let summed = g.element_wise(ElementWiseOp::Add, vec![scaled, cons_nrg], None);
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "compute_pre_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[
                    ("cons_den".into(), "cons.den".into()),
                    ("cons_nrg".into(), "cons.nrg".into()),
                ],
                scalar_params: &[],
                field_writes: &[("prim_pre".into(), "prim.pre".into(), summed)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert!(
            desc.source
                .contains("pub fn compute_pre_1d__raw<S: Scalar + OrderedNumeric + Send + Sync>("),
            "src:\n{}",
            desc.source
        );
        assert!(
            desc.source.contains("field0: &CpuField<S>"),
            "src:\n{}",
            desc.source
        );
        assert!(
            desc.source.contains(
                "let cons_den: S = field0.data[(__idx_cell_buf0 + ((ii) - ii)) as usize];"
            )
        );
        assert!(
            desc.source
                .contains("] = ((cons_den * S::from_f64(2.0)) + cons_nrg);"),
            "src:\n{}",
            desc.source
        );
        assert!(desc.source.contains("pub fn compute_pre_1d<S: Scalar + OrderedNumeric + Send + Sync>(inputs: &[CpuField<S>]"), "src:\n{}", desc.source);
        assert!(desc.source.contains("outputs: &mut [CpuFieldMut<S>]"));
        // float spelling is fully S; integer index arithmetic stays i32. every float
        // type in the source is S — `: f64` annotations and `[f64]` slices are absent.
        // (the bare `f64` inside `Scalar::from_f64` is the constructor name.)
        assert!(
            !desc.source.contains(": f64"),
            "no f64 type annotation:\n{}",
            desc.source
        );
        assert!(
            !desc.source.contains("[f64]"),
            "no f64 slice:\n{}",
            desc.source
        );
        assert!(
            !desc.source.contains("f32"),
            "no f32 should appear in a generic kernel:\n{}",
            desc.source
        );
        assert!(
            desc.source.contains("field0.lo[0]"),
            "indices stay i32:\n{}",
            desc.source
        );
    }

    #[test]
    fn in_place_buffer_is_a_single_mut_slice() {
        let mut g = Graph::new();
        let rho = scalar_param(&mut g, "rho");
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "inplace_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[("rho".into(), "cons.den".into())],
                scalar_params: &[],
                field_writes: &[("rho".into(), "cons.den".into(), rho)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert_eq!(
            desc.field_bindings.len(),
            1,
            "in-place must dedup to one buffer"
        );
        assert!(desc.field_bindings[0].is_output);
        assert!(
            desc.source.contains("buf0: &mut [S]"),
            "src:\n{}",
            desc.source
        );
    }

    #[test]
    fn i32_scalar_param_emits_an_integer_signature() {
        // a lattice-map source-coord arg (a shift / pivot / clamp) is an integer
        // param; the emitter must type it i32 from its declared element,
        // so index arithmetic on it stays integer.
        let mut g = Graph::new();
        let rho = scalar_param(&mut g, "rho");
        g.add_param(
            Symbol::intern("shift"),
            TensorTy::scalar(ElementTy::I32),
            None,
        );
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "shifted_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[("rho".into(), "prim.rho".into())],
                scalar_params: &["shift".into()],
                field_writes: &[("out".into(), "out".into(), rho)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert!(
            desc.source.contains("shift: i32"),
            "shift must be i32:\n{}",
            desc.source
        );
        assert!(!desc.source.contains("shift: f64"));
    }

    #[test]
    fn render_index_handles_select_and_comparison() {
        // the lattice-map pullback source coord:
        //   if (map_type == 1) { ii + shift } else { pivot2 - ii }
        // renders as a pure-integer if/else index, free of floats and casts.
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
        assert_eq!(
            s,
            "(if (map_type == 1) { (ii + shift) } else { (pivot2 - ii) })"
        );
    }

    #[test]
    fn param_invariant_select_unswitches_into_two_specializations() {
        // four float selects gated on `theta < 0` (a param-only condition):
        // the emitter renders a true and a false specialization plus a
        // dispatcher branching once per call. each specialized body holds the
        // resolved arm inline.
        let mut g = Graph::new();
        let x = scalar_param(&mut g, "x");
        let theta = scalar_param(&mut g, "theta");
        let zero = g.add_const(ConstValue::F64(0.0), None);
        let cond = g.element_wise(ElementWiseOp::Lt, vec![theta, zero], None);
        let two = g.add_const(ConstValue::F64(2.0), None);
        let mut acc = x;
        for _ in 0..4 {
            let scaled = g.element_wise(ElementWiseOp::Mul, vec![acc, two], None);
            acc = g.select(cond, scaled, acc, None);
        }
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "limiter_pick_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[("x".into(), "prim.rho".into())],
                scalar_params: &["theta".into()],
                field_writes: &[("out".into(), "out".into(), acc)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert!(
            desc.source
                .contains("pub fn limiter_pick_1d__uswt__raw<S: Scalar"),
            "src:\n{}",
            desc.source
        );
        assert!(
            desc.source
                .contains("pub fn limiter_pick_1d__uswf__raw<S: Scalar"),
            "src:\n{}",
            desc.source
        );
        // the dispatcher branches on the rendered param condition and forwards.
        assert!(
            desc.source.contains("if (theta < S::from_f64(0.0)) {"),
            "src:\n{}",
            desc.source
        );
        assert!(
            desc.source.contains(
                "limiter_pick_1d__uswt__raw(field0, field1, grid_size_0, dom_lo_0, theta);"
            ),
            "src:\n{}",
            desc.source
        );
        // the descriptor wrapper still targets the dispatcher by the original name.
        assert!(
            desc.source.contains("limiter_pick_1d__raw(\n"),
            "src:\n{}",
            desc.source
        );
    }

    #[test]
    fn cell_varying_select_does_not_unswitch() {
        // the condition reads the field: it varies per cell, so the single loop
        // nest keeps the select.
        let mut g = Graph::new();
        let x = scalar_param(&mut g, "x");
        let zero = g.add_const(ConstValue::F64(0.0), None);
        let cond = g.element_wise(ElementWiseOp::Lt, vec![x, zero], None);
        let two = g.add_const(ConstValue::F64(2.0), None);
        let mut acc = x;
        for _ in 0..4 {
            let scaled = g.element_wise(ElementWiseOp::Mul, vec![acc, two], None);
            acc = g.select(cond, scaled, acc, None);
        }
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "cell_pick_1d",
                coalesce_layout: false,
                ndim: 1,
                target: cfg(),
                field_inputs: &[("x".into(), "prim.rho".into())],
                scalar_params: &[],
                field_writes: &[("out".into(), "out".into(), acc)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        assert!(!desc.source.contains("__uswt"), "src:\n{}", desc.source);
        assert!(!desc.source.contains("__uswf"), "src:\n{}", desc.source);
    }

    #[test]
    fn two_d_declares_integer_stride_locals_and_flat_parallel() {
        let mut g = Graph::new();
        let cons_den = scalar_param(&mut g, "cons_den");
        let desc = emit_kernel_cpu(
            &g,
            &KernelEmitInputs {
                kernel_name: "pass_2d",
                coalesce_layout: false,
                ndim: 2,
                target: cfg(),
                field_inputs: &[("cons_den".into(), "cons.den".into())],
                scalar_params: &[],
                field_writes: &[("prim_den".into(), "prim.den".into(), cons_den)],
                coord_components: &[],
                device_preamble: &[],
                tile_spec: None,
            },
        );
        // field0 is the input — it's a `&CpuField<S>` kernel arg.
        // body indexes via `field0.lo[..]` / `field0.strides[..]` / `field0.data`.
        assert!(desc.source.contains("field0: &CpuField<S>"));
        // axis 0 is CONTIGUOUS_AXIS: its stride is 1 by construction, so the emitter drops the
        // multiply. axis 1 keeps its stride factor.
        assert!(desc.source.contains("field0.strides[1]"));
        // 2D default emit is cache-tiled on the transverse axis only: axis 1
        // gets a `for _d1` tile loop; the contiguous axis 0 runs the full
        // extent (vectorized bodies need long unit-stride inner trips).
        assert!(desc.source.contains("(0.._ntiles).into_par_iter()"));
        assert!(desc.source.contains("for _d1 in 0.._ts {"));
        assert!(desc.source.contains("let _i1: i32 = _c1 as i32;"));
        assert!(desc.source.contains("let jj: i32 = _i1 + dom_lo_1;"));
        assert!(
            desc.source.contains("for _i0 in 0..(grid_size_0 as i32) {"),
            "contiguous axis must be a full-extent loop:\n{}",
            desc.source
        );
        assert!(
            !desc.source.contains("for _d0"),
            "contiguous axis must not be tiled in 2d:\n{}",
            desc.source
        );
    }
}
