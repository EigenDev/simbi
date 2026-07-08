// =============================================================================
// engine.rs
//
// the GPU mapping of the structured kernel ABI (docs/design/15 §5). the
// substrate KernelSet builds ONE backend-neutral `KernelInvocation` per kernel
// (ordered buffer handles + packed params); `symbi-aot::run_cpu` maps it to the
// generated CPU fn, and `run_gpu` here maps the SAME invocation to a GPU launch:
//
//   neutral IR blob --render_from_ir--> CUDA source --NVRTC(jit_kernel)--> module
//   buffers (reordered into kernel binding order) + params --> cuLaunchKernel
//
// `dispatch` picks the path on `Mem::IS_DEVICE_ACCESSIBLE` — the SAME invocation,
// one branch. unified memory (the default device memory) is host- AND
// device-addressable, so the buffer's pointer is usable on-device as-is; the host
// never dereferences it (doc 15 §6). an explicit-memory `BufHandle::Device` variant
// (host cannot deref) is a later addition — not needed for the unified milestone.
// =============================================================================

use symbi_aot::{CpuField, CpuFieldMut, KernelInvocation, OrderedNumeric, Scalar};
#[cfg(feature = "gpu")]
use symbi_aot::{compute_strides, copy_extent, copy_lo};
use symbi_ir::emit::ReductionOp;
use symbi_xpu::MemorySpace;

/// host-side POD that matches the CUDA `__symbi_View` struct emitted by
/// `crate::backends::kernel::CRenderer::preamble` — 8-byte ptr + 16 bytes lo +
/// 16 bytes strides + 16 bytes extent = 56 bytes, naturally 8-byte aligned. one
/// of these is passed by value per buffer to every GPU kernel. cfg-gated to
/// the `cuda` feature: cpu-only builds never construct one.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceView {
    data: *const std::ffi::c_void,
    lo: [i32; 4],
    strides: [i32; 4],
    extent: [i32; 4],
}

// static ABI assertions: the CUDA `__symbi_View` struct emitted by the kernel
// preamble assumes EXACTLY this layout (8-byte ptr at offset 0, 16 bytes of
// `lo` at offset 8, 16 bytes of `strides` at offset 24, 16 bytes of `extent` at
// offset 40 — total 56 bytes). a drift here (an added field, a reorder, an
// alignment change from a new `#[derive]`) would silently mis-bind every kernel
// arg. catch it at compile time rather than as garbled physics. `offset_of!` is
// stable since 1.77.
#[cfg(feature = "gpu")]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(
        size_of::<DeviceView>() == 56,
        "DeviceView size drifted from 56 bytes"
    );
    assert!(
        offset_of!(DeviceView, data) == 0,
        "DeviceView.data offset drifted"
    );
    assert!(
        offset_of!(DeviceView, lo) == 8,
        "DeviceView.lo offset drifted"
    );
    assert!(
        offset_of!(DeviceView, strides) == 24,
        "DeviceView.strides offset drifted"
    );
    assert!(
        offset_of!(DeviceView, extent) == 40,
        "DeviceView.extent offset drifted"
    );
};

// =============================================================================
// GpuBackend — the device-backend seam (docs/design/35 R2).
//
// every GPU leak that names CUDA explicitly — the render-target token, the kernel
// dispatcher, the launch-arg ABI for a field buffer — is funneled through this ONE
// trait. `run_gpu` / `field_reduce_device` are generic over `B: GpuBackend`, so a
// second backend (HIP/Metal/WebGPU) is a one-place add: a unit struct + impl that
// names its Target token, its `KernelDispatcher`, and how it packs a field view.
// no `&dyn` — the backend is a zero-size type param, fully monomorphized;
// `DefaultGpuBackend` binds the compiled-in choice at the dispatch boundary.
// =============================================================================
#[cfg(feature = "gpu")]
pub trait GpuBackend: 'static {
    // the device runtime (alloc/launch/sync) this backend drives.
    type Runtime: symbi_xpu::runtime::GpuRuntime;
    // the render-target token: `render_from_ir` emits this backend's source from the IR.
    const TARGET: symbi_ir::emit::Target;
    // the process-global JIT dispatcher (render-cache + module-cache + runtime handle).
    fn dispatcher() -> &'static symbi_xpu::runtime::KernelDispatcher<Self::Runtime>;
    // pack ONE field buffer into the launch ABI in this backend's binding convention.
    // cuda: a 56-byte `DeviceView` (ptr+lo+strides+extent) pushed by value; the arena
    // copies the bytes, so the on-stack `view` is sound for the launch.
    fn push_field(args: &mut symbi_xpu::KernelArgs, ptr: *const u8, lo: &[i32], extent: &[u32]);
}

// the cuda backend: `Target::Cuda` + the `cuda_runtime::DISPATCHER` + the `DeviceView` ABI.
#[cfg(feature = "cuda")]
pub struct CudaBackend;

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    type Runtime = symbi_xpu::runtime::cuda_runtime::CudaRuntime;
    const TARGET: symbi_ir::emit::Target = symbi_ir::emit::Target::Cuda;

    #[inline]
    fn dispatcher() -> &'static symbi_xpu::runtime::KernelDispatcher<Self::Runtime> {
        symbi_xpu::runtime::cuda_runtime::current_dispatcher()
    }

    #[inline]
    fn push_field(args: &mut symbi_xpu::KernelArgs, ptr: *const u8, lo: &[i32], extent: &[u32]) {
        let view = DeviceView {
            data: ptr as *const std::ffi::c_void,
            lo: copy_lo(lo),
            strides: compute_strides(extent),
            extent: copy_extent(extent),
        };
        args.push(&view);
    }
}

// the hip backend (docs/design/38): same `Target::Hip` (renders the identical cuda-c++ source),
// the hip per-device dispatcher, and the SAME `DeviceView` launch ABI as cuda.
#[cfg(feature = "hip")]
pub struct HipBackend;

#[cfg(feature = "hip")]
impl GpuBackend for HipBackend {
    type Runtime = symbi_xpu::runtime::hip_runtime::HipRuntime;
    const TARGET: symbi_ir::emit::Target = symbi_ir::emit::Target::Hip;

    #[inline]
    fn dispatcher() -> &'static symbi_xpu::runtime::KernelDispatcher<Self::Runtime> {
        symbi_xpu::runtime::hip_runtime::current_dispatcher()
    }

    #[inline]
    fn push_field(args: &mut symbi_xpu::KernelArgs, ptr: *const u8, lo: &[i32], extent: &[u32]) {
        let view = DeviceView {
            data: ptr as *const std::ffi::c_void,
            lo: copy_lo(lo),
            strides: compute_strides(extent),
            extent: copy_extent(extent),
        };
        args.push(&view);
    }
}

// the compiled-in default backend. each backend feature binds the alias to its own struct; the
// dispatch boundary calls `run_gpu::<DefaultGpuBackend, _>`. cuda wins if both are set.
#[cfg(feature = "cuda")]
pub type DefaultGpuBackend = CudaBackend;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub type DefaultGpuBackend = HipBackend;

/// the regime-AGNOSTIC CFL reduction: max over `domain` of a per-cell wave-speed
/// scratch field. thin wrapper over the general `field_reduce` — every regime
/// computes its OWN `wave_speed_map` (the regime-specific physics) into `scratch`,
/// then calls this; the reduce is shared (was copy-pasted in three substrate sets).
pub fn field_max_reduce<Sc: Scalar + OrderedNumeric, Mem: MemorySpace, const D: usize>(
    field: &symbi_grid::Field<Sc, D, Mem>,
    domain: &symbi_algebra::Domain<D>,
) -> f64 {
    field_reduce(field, domain, ReductionOp::Max)
}

/// reduce `field` over `domain` by `op` (Add/Mul/Min/Max) — the substrate Reduce
/// morphism (docs/design/15 §2). on HOST memory it's a plain fold; on DEVICE memory
/// it runs a GPU block-reduction so only the per-block partials cross device->host,
/// NOT a per-cell host scan over unified memory ([[feedback_no_compute_roundtrips]]).
/// the two algebras agree (max/min are exact; add/mul differ from the host's
/// sequential fold only by reassociated rounding).
pub fn field_reduce<Sc: Scalar + OrderedNumeric, Mem: MemorySpace, const D: usize>(
    field: &symbi_grid::Field<Sc, D, Mem>,
    domain: &symbi_algebra::Domain<D>,
    op: ReductionOp,
) -> f64 {
    if Mem::IS_DEVICE_ACCESSIBLE {
        #[cfg(feature = "gpu")]
        {
            return field_reduce_device::<DefaultGpuBackend, _, _, D>(field, domain, op);
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = op;
            unreachable!("device-accessible memory requires a gpu feature (cuda or hip)");
        }
    }
    // host fold (the CPU algebra of the Reduce morphism, doc 15 §2). LARGE
    // domains fold in parallel over outer-axis slabs — a serial fold here was
    // a measured per-root-step stall at production sizes (the cfl reduce runs
    // per level, the body-feedback sums per component). min/max are exact in
    // any order; add/mul reassociate within roundoff, same as the device
    // block-reduction this mirrors. small domains keep the sequential fold
    // (no rayon setup, bit-stable for the small exactness gates).
    let (identity, combine) = host_identity_combine(op);
    const PAR_THRESHOLD: usize = 1 << 16;
    let (outer_lo, outer_hi) = (domain.spaces[0].lo, domain.spaces[0].hi);
    if domain.volume() >= PAR_THRESHOLD && (outer_hi - outer_lo) > 1 {
        use rayon::prelude::*;
        return (outer_lo..outer_hi)
            .into_par_iter()
            .map(|ii| {
                let mut slab = domain.clone();
                slab.spaces[0].lo = ii;
                slab.spaces[0].hi = ii + 1;
                let mut acc = identity;
                for c in slab.iter() {
                    acc = combine(acc, field.view().at(c).to_f64());
                }
                acc
            })
            .reduce(|| identity, combine);
    }
    let mut acc = identity;
    for c in domain.iter() {
        acc = combine(acc, field.view().at(c).to_f64());
    }
    acc
}

/// the host (f64) identity + combine for a reduction op — the CPU algebra mirroring
/// the device's `reduction_identity_combine`.
fn host_identity_combine(op: ReductionOp) -> (f64, fn(f64, f64) -> f64) {
    // min/max MUST propagate NaN: `f64::min`/`f64::max` silently return the
    // non-NaN operand, which would drop a single poisoned wave-speed cell and let
    // garbage advance past `check_dt_or_panic` ([[feedback_no_silent_floors]]).
    // add/mul propagate NaN natively (a + NaN == NaN). `x != x` is the branchless
    // NaN test; it mirrors the CUDA in-block combine in `reduction_identity_combine`.
    match op {
        ReductionOp::Add => (0.0, |a, b| a + b),
        ReductionOp::Mul => (1.0, |a, b| a * b),
        ReductionOp::Min => (f64::INFINITY, |a, b| {
            if a != a || b != b { f64::NAN } else { a.min(b) }
        }),
        ReductionOp::Max => (f64::NEG_INFINITY, |a, b| {
            if a != a || b != b { f64::NAN } else { a.max(b) }
        }),
    }
}

/// **B12 — cached reduction partials buffer.** the CFL host-fold needs one
/// `Sc` per block (typically ~1K-4K bytes for a 512² grid). a fresh
/// `cuMemAllocManaged` for it on every step was the only per-step driver
/// alloc cost left in the pipeline. one slot per precision (`f64` / `f32`),
/// grow-only: when a larger grid needs more bytes the slot grows; otherwise
/// the existing block is reused step after step. lives for the process
/// lifetime — drop happens at exit.
///
/// the closure pattern holds the slot's mutex for the entire kernel launch +
/// host fold, so concurrent reductions from multiple threads serialize on
/// the cache rather than racing on the buffer. uncontended in the single-
/// simulation case (the common one); cheap when contested.
#[cfg(feature = "gpu")]
fn with_cached_partials<Sc: Scalar + OrderedNumeric, R>(
    bytes_needed: usize,
    f: impl FnOnce(*mut Sc) -> R,
) -> R {
    use std::sync::{Mutex, OnceLock};
    use symbi_xpu::DeviceMemory;
    use symbi_xpu::MemoryBlock;

    // one slot per precision. `is_f64` discriminates at compile time via
    // size_of::<Sc>() so the lookup is monomorphized.
    static F64_PARTIALS: OnceLock<Mutex<Option<MemoryBlock<DeviceMemory>>>> = OnceLock::new();
    static F32_PARTIALS: OnceLock<Mutex<Option<MemoryBlock<DeviceMemory>>>> = OnceLock::new();
    let slot = if std::mem::size_of::<Sc>() == std::mem::size_of::<f64>() {
        F64_PARTIALS.get_or_init(|| Mutex::new(None))
    } else {
        F32_PARTIALS.get_or_init(|| Mutex::new(None))
    };
    let mut guard = slot.lock().unwrap();
    let need_grow = match guard.as_ref() {
        Some(blk) => blk.bytes() < bytes_needed,
        None => true,
    };
    if need_grow {
        // round up to next power of 2 to amortize future growths.
        let alloc_bytes = bytes_needed.next_power_of_two().max(256);
        *guard = Some(
            MemoryBlock::<DeviceMemory>::new(alloc_bytes)
                .expect("unified alloc for reduction partials"),
        );
    }
    let ptr = guard.as_mut().unwrap().as_mut_ptr::<Sc>();
    f(ptr)
    // `guard` drops here — releases the mutex AFTER the closure has consumed
    // the pointer (kernel launched + ctx_sync'd + host-folded).
}

/// GPU reduction over `domain` of `field` via the substrate Reduce morphism: render
/// the block-reduce at the scalar's precision, NVRTC-compile (cached), launch over
/// the window, and fold the per-block partials on the host (only num_blocks scalars
/// cross). the field's allocated domain gives the view_t buffer layout; the `domain`
/// arg is the reduced window (interior).
#[cfg(feature = "gpu")]
fn field_reduce_device<
    B: GpuBackend,
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
    const D: usize,
>(
    field: &symbi_grid::Field<Sc, D, Mem>,
    domain: &symbi_algebra::Domain<D>,
    op: ReductionOp,
) -> f64 {
    use symbi_ir::emit::Precision;
    use symbi_ir::{REDUCTION_BLOCK_SIZE, render_field_reduction};
    use symbi_xpu::ctx_sync;
    use symbi_xpu::runtime::GpuRuntime;
    use symbi_xpu::LaunchConfig;

    let is_f64 = std::mem::size_of::<Sc>() == std::mem::size_of::<f64>();
    let precision = if is_f64 {
        Precision::F64
    } else {
        Precision::F32
    };
    let op_tag = match op {
        ReductionOp::Add => "add",
        ReductionOp::Mul => "mul",
        ReductionOp::Min => "min",
        ReductionOp::Max => "max",
    };
    let name = format!("symbi_field_reduce_{op_tag}_{D}d");
    // the reduction kernel SOURCE is still cuda-specific (`render_field_reduction` has no
    // target token); a second backend adds its own reduction renderer here. dispatcher +
    // launch ABI ARE backend-generic via `B` below.
    let desc = render_field_reduction(&name, D, precision, op);

    // the reduced window (interior) + the field's allocated buffer layout (view_t).
    let alloc = field.domain();
    let total_cells = domain.volume() as u32;
    let grid: Vec<u32> = (0..D).map(|a| domain.spaces[a].size() as u32).collect();
    let dom_lo: Vec<i32> = (0..D).map(|a| domain.spaces[a].lo as i32).collect();
    let buf_extent: Vec<u32> = (0..D).map(|a| alloc.spaces[a].size() as u32).collect();
    let buf_lo: Vec<i32> = (0..D).map(|a| alloc.spaces[a].lo as i32).collect();

    let num_blocks = total_cells.div_ceil(REDUCTION_BLOCK_SIZE).max(1);
    // **B12 — partials buffer is CACHED ACROSS STEPS, not allocated per call.**
    // a fresh `cuMemAllocManaged` on every reduction (every CFL step) was
    // pure smell: ~30 µs of driver-allocator time per step × 25K steps =
    // ~750 ms wasted in the alloc path of a typical Kepler run. the partials
    // shape only depends on `num_blocks` (which grows monotonically with grid
    // size) and `sizeof(Sc)` — both static across a sim. cache once per
    // precision, grow-only when num_blocks ever needs more.
    //
    // closure holds the cache mutex across the launch + sync + host fold so
    // the next reducer can't race on the buffer.
    let bytes_needed = (num_blocks as usize) * std::mem::size_of::<Sc>();
    with_cached_partials::<Sc, f64>(bytes_needed, |partials_ptr| {
        let module_key = format!("{name}#{}", if is_f64 { "f64" } else { "f32" });
        let kernel = B::dispatcher().jit_kernel_keyed(&desc.source, &module_key, &name);

        // pack args in the reduction ABI: View struct, total_cells,
        // grid, dom_lo, partials. `KernelArgs` copies each value into stable storage,
        // so the field view packed by `B::push_field` is sound for the launch.
        let partials_arg = partials_ptr as *const u8;
        let config = LaunchConfig::for_1d(total_cells, REDUCTION_BLOCK_SIZE);
        symbi_xpu::with_pooled_args(|args| {
            B::push_field(args, field.as_ptr() as *const u8, &buf_lo, &buf_extent);
            args.push(&total_cells);
            for g in &grid {
                args.push(g);
            }
            for d in &dom_lo {
                args.push(d);
            }
            args.push(&partials_arg);
            unsafe {
                B::dispatcher()
                    .runtime()
                    .launch(&kernel, config, args.as_mut_slice())
                    .unwrap_or_else(|e| panic!("GPU reduction launch '{name}' failed: {e:?}"));
            }
        });
        ctx_sync();

        // fold the per-block partials on the host with the SAME op — num_blocks scalars,
        // not per-cell. (the device combined each block; this combines the blocks.)
        let (identity, combine) = host_identity_combine(op);
        let mut acc = identity;
        for i in 0..num_blocks as usize {
            acc = combine(acc, unsafe { (*partials_ptr.add(i)).to_f64() });
        }
        acc
    })
}

/// dispatch a structured invocation to the GPU when `Mem` is device-accessible,
/// else to the generated CPU kernel `cpu`. the IR blob + kernel name drive the GPU
/// render+JIT; `cpu` is the AOT-compiled fn for the host path. one invocation, both
/// backends.
pub fn dispatch<Sc, Mem, F>(inv: KernelInvocation<Sc>, ir: &str, kernel_name: &str, cpu: F)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
    F: FnOnce(&[CpuField<'_, Sc>], &mut [CpuFieldMut<'_, Sc>], &[u32], &[i32], &[i32], &[Sc]),
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        #[cfg(feature = "gpu")]
        {
            let _ = &cpu;
            run_gpu::<DefaultGpuBackend, _>(inv, ir, kernel_name);
        }
        #[cfg(not(feature = "gpu"))]
        {
            // device-accessible memory (DeviceMemory) exists ONLY under a gpu feature, so
            // this branch is unreachable in a host-only build.
            let _ = (ir, kernel_name, cpu);
            unreachable!("device-accessible memory requires a gpu feature (cuda or hip)");
        }
    } else {
        let _ = (ir, kernel_name);
        inv.run_cpu(cpu);
    }
}

/// render the neutral IR blob, NVRTC-compile it (cached per kernel name), and launch
/// it over the invocation's buffers. unified-memory pointers are device-usable, so
/// the buffer handles map straight to kernel pointer args — reordered from the
/// invocation's (host-then-output) layout into the kernel's buffer-index binding
/// order via the rendered `field_bindings`, the exact inverse of `run_cpu`'s split.
// the rendered CUDA descriptor (source + bindings + scalar_is_int) per (kernel,
// precision). rendering = deserialize the IR blob + walk the scalarized body; the
// NVRTC MODULE is already cached by the dispatcher, so without this the render is
// pure waste on every launch after the first. keyed by precision since one kernel
// renders to both f32 and f64.
//
// the cache value bundles the descriptor with a precomputed `module_key` (the string
// the dispatcher uses to look up the JITed CUDA module) — formatting it on every
// launch was one String alloc per dispatch.
#[cfg(feature = "gpu")]
struct CachedDesc {
    desc: symbi_ir::emit::KernelDescriptor,
    module_key: String,
}

#[cfg(feature = "gpu")]
static RENDER_CACHE: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<(String, bool), std::sync::Arc<CachedDesc>>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// the per-block dynamic-smem budget tiled launches are sized against. Turing (sm_75,
/// the RTX 2070 dev part) allows 48 KB of dynamic `__shared__` without the
/// `cudaFuncSetAttribute` opt-in; staying under it keeps the launch portable.
#[cfg(feature = "gpu")]
const TILED_SMEM_LIMIT: usize = 48 * 1024;

/// pick a BLOCK shape for a smem-tiled launch: balanced (cube-ish), clamped to the
/// grid, total <= 256 threads, and — critically — whose `(block + 2*halo)` slab
/// times `cell_bytes` fits `TILED_SMEM_LIMIT`. tries decreasing cube edges and
/// shrinks the largest dim to meet the thread cap. `SYMBI_TILE_BLOCK="8,8,4"`
/// overrides for the tile-size sweep (docs/design/22 §G). this is used in place of the
/// warp-first `block_for` for tiled kernels, whose [32,8,1] shape blows the slab
/// when the halo sits on a thin (block=1) axis.
#[cfg(feature = "gpu")]
fn tiled_block(ndim: usize, grid: &[u32], halo: &[u8], cell_bytes: usize) -> [u32; 3] {
    if let Some(b) = env_tile_block(ndim) {
        return b;
    }
    for &edge in &[16u32, 8, 4, 2, 1] {
        let mut b = [1u32; 3];
        for a in 0..ndim {
            b[a] = edge.min(grid[a].max(1));
        }
        // cap total threads to 256 by halving the current largest dim.
        while b[0] * b[1] * b[2] > 256 {
            let amax = (0..ndim).max_by_key(|&a| b[a]).unwrap();
            if b[amax] <= 1 {
                break;
            }
            b[amax] = (b[amax] / 2).max(1);
        }
        let slab: usize = (0..ndim)
            .map(|a| (b[a] + 2 * halo[a] as u32) as usize)
            .product();
        if b[0] * b[1] * b[2] <= 256 && slab * cell_bytes <= TILED_SMEM_LIMIT {
            return b;
        }
    }
    [1, 1, 1]
}

/// the explicit per-process tiled-block override `SYMBI_TILE_BLOCK="bx,by,bz"`
/// (the leading `ndim` entries are used). `None` if unset/unparseable.
#[cfg(feature = "gpu")]
fn env_tile_block(ndim: usize) -> Option<[u32; 3]> {
    let raw = std::env::var("SYMBI_TILE_BLOCK").ok()?;
    let mut b = [1u32; 3];
    for (a, tok) in raw.split(',').enumerate().take(ndim) {
        b[a] = tok.trim().parse().ok()?;
    }
    Some(b)
}

/// the HOST-READ BARRIER of the B12 contract: kernel launches are
/// asynchronous (same-stream semantics order kernel-to-kernel; the per-launch
/// sync was removed for pipelining), so host code that reads — or writes —
/// device-accessible memory the queued kernels touch MUST drain the device
/// queue first. no-op on a host backend. call sites: the evolve callbacks,
/// the amr hierarchy's api boundaries and host scans, and any test comparing
/// device buffers right after a dispatch.
pub fn device_sync<Mem: MemorySpace>() {
    #[cfg(feature = "gpu")]
    if Mem::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
}

/// total GPU kernel launches since process start — a perf-diagnostic counter (read via
/// `gpu_launch_count`) to separate launch-bound from kernel-slow on the device path.
pub static GPU_LAUNCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// snapshot the cumulative GPU launch count.
pub fn gpu_launch_count() -> u64 {
    GPU_LAUNCH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "gpu")]
fn run_gpu<B: GpuBackend, Sc: Scalar + OrderedNumeric>(
    inv: KernelInvocation<Sc>,
    ir: &str,
    kernel_name: &str,
) {
    GPU_LAUNCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    use symbi_aot::BufHandle;
    use symbi_ir::emit::Precision;
    use symbi_ir::render_from_ir;
    // **B12** — ctx_sync no longer called per-launch; same-stream CUDA semantics
    // serialize kernel-to-kernel ordering. ctx_sync stays in field_reduce_device
    // (line 151) where it actually crosses host↔device for the cfl host-fold.
    use symbi_xpu::LaunchConfig;
    use symbi_xpu::runtime::GpuRuntime;

    // precision is the scalar's width: f64 -> 8 bytes, f32 -> 4. render the kernel at
    // that precision so the device reads the buffers (which ARE `Sc`) correctly.
    let is_f64 = std::mem::size_of::<Sc>() == std::mem::size_of::<f64>();
    let precision = if is_f64 {
        Precision::F64
    } else {
        Precision::F32
    };
    // render once per (kernel, precision); subsequent launches reuse the descriptor.
    // fast path: read-locked HashMap lookup. after the first launch of each kernel
    // this is uncontended and ~free; eliminates the per-launch Mutex acquire that
    // dominated the dispatch glue at ~50 launches/step.
    let cached = {
        let key = (kernel_name.to_string(), is_f64);
        if let Some(d) = RENDER_CACHE.read().unwrap().get(&key) {
            std::sync::Arc::clone(d)
        } else {
            let mut w = RENDER_CACHE.write().unwrap();
            w.entry(key)
                .or_insert_with(|| {
                    let mut d = render_from_ir(ir, B::TARGET, precision);
                    // sort field_bindings by buffer_index ONCE at cache time so the
                    // per-launch walk is just a linear iter, no heap-Vec, no sort.
                    d.field_bindings.sort_by_key(|b| b.buffer_index);
                    let module_key =
                        format!("{}#{}", d.kernel_name, if is_f64 { "f64" } else { "f32" },);
                    std::sync::Arc::new(CachedDesc {
                        desc: d,
                        module_key,
                    })
                })
                .clone()
        }
    };
    let desc = &cached.desc;
    debug_assert_eq!(
        desc.kernel_name, kernel_name,
        "IR blob kernel name mismatch"
    );

    let KernelInvocation {
        buffers,
        grid,
        dom_lo,
        ints,
        scalars,
    } = inv;
    let ndim = grid.len();

    // build two stack-resident lookup tables from inv.buffers: one for Host slots
    // (kernel inputs), one for HostMut slots (kernel outputs). per-bucket order
    // matches inv.buffers' Host-first-then-HostMut layout (the same order `run_cpu`
    // would re-split). MAX_BUFS_PER_KIND=48 covers the curvilinear FUSED god+bcell
    // (~36 inputs: geo-source prims + u_n + per-dir flux + bc/bcn/bf); overflow asserts
    // loudly rather than corrupting memory. (mirrors the CPU dispatch_named MAX_FIELDS=48.)
    const MAX_BUFS_PER_KIND: usize = 48;
    const EMPTY_I32: &[i32] = &[];
    const EMPTY_U32: &[u32] = &[];
    let mut host_lookup: [(*const u8, &[i32], &[u32]); MAX_BUFS_PER_KIND] =
        [(std::ptr::null(), EMPTY_I32, EMPTY_U32); MAX_BUFS_PER_KIND];
    let mut hostmut_lookup: [(*const u8, &[i32], &[u32]); MAX_BUFS_PER_KIND] =
        [(std::ptr::null(), EMPTY_I32, EMPTY_U32); MAX_BUFS_PER_KIND];
    let mut host_n = 0usize;
    let mut hostmut_n = 0usize;
    for b in &buffers {
        match &b.handle {
            BufHandle::Host(s) => {
                assert!(
                    host_n < MAX_BUFS_PER_KIND,
                    "run_gpu('{kernel_name}'): kernel has > {MAX_BUFS_PER_KIND} input buffers; raise MAX_BUFS_PER_KIND"
                );
                host_lookup[host_n] = (s.as_ptr() as *const u8, b.lo, b.extent);
                host_n += 1;
            }
            BufHandle::HostMut(s) => {
                assert!(
                    hostmut_n < MAX_BUFS_PER_KIND,
                    "run_gpu('{kernel_name}'): kernel has > {MAX_BUFS_PER_KIND} output buffers; raise MAX_BUFS_PER_KIND"
                );
                hostmut_lookup[hostmut_n] = (s.as_ptr() as *const u8, b.lo, b.extent);
                hostmut_n += 1;
            }
        }
    }

    // module cache keyed by precision too, so an f32 and f64 build of the same
    // kernel name don't shadow each other in one process. module_key is precomputed
    // at cache time (see `CachedDesc`) so the dispatch path here is a slice borrow,
    // not a `format!()`.
    let kernel =
        B::dispatcher().jit_kernel_keyed(&desc.source, &cached.module_key, &desc.kernel_name);

    // block shape is EXTENT-AWARE (`block_for`): a warp on the contiguous axis-0 (coalesced)
    // + transverse dims clamped to the actual `grid` extents, so a quasi-1D/2D run (a 3D
    // kernel over a thin transverse axis) doesn't idle most of each block. an explicit
    // `SYMBI_BLOCK_{1D,2D,3D}` env var overrides it.
    // Gate 3: a tiled kernel needs a block shape that BOUNDS the per-block smem
    // slab `prod_a (block_a + 2*halo_a) * sizeof(S) * n_fields` under the device
    // limit. the warp-first `block_for` shape ([32,8,1]) makes a pathological slab
    // when the halo is on a thin (block=1) axis — e.g., dir-2 flux: 32*8*(1+4) cells
    // -> ~100 KB for 10 fields, past Turing's 48 KB. so a tiled launch picks a
    // BALANCED block (cube-ish, fits smem) instead. block dims are read at runtime
    // by the kernel (blockDim.*), so only the byte count crosses here.
    let elem_bytes = std::mem::size_of::<Sc>();
    let b = match &desc.tile_spec {
        Some(ts) => tiled_block(ndim, grid, &ts.halo, ts.tiled_field_keys.len() * elem_bytes),
        None => symbi_xpu::block_for(ndim, grid),
    };
    let mut config = match ndim {
        1 => LaunchConfig::for_1d(grid[0], b[0]),
        2 => LaunchConfig::for_2d(grid[0], grid[1], b[0], b[1]),
        3 => LaunchConfig::for_3d(grid[0], grid[1], grid[2], b[0], b[1], b[2]),
        n => panic!("run_gpu: unsupported ndim {n}"),
    };
    if let Some(ts) = &desc.tile_spec {
        config.shared_mem_bytes = ts.smem_bytes_per_block(&b[..ndim], elem_bytes) as u32;
    }

    // pack the launch args in the kernel's emitted-signature order: View structs
    // (one per buffer, constructed inline and copied into the arena via push), then
    // grid, dom_lo, then scalar params interleaved int/float by `scalar_is_int`.
    // arena is the thread-local pool, so after warmup zero allocations here. the
    // field_bindings walk consumes one host or hostmut slot per binding in
    // bucket-order (matching inv.buffers' Host-first-then-HostMut layout) and
    // constructs the DeviceView ON-STACK before pushing — there is no `views` Vec.
    symbi_xpu::with_pooled_args(|args| {
        let (mut hi, mut mi) = (0usize, 0usize);
        for binding in &desc.field_bindings {
            let (ptr, lo, extent) = if binding.is_output {
                let t = hostmut_lookup[mi];
                mi += 1;
                t
            } else {
                let t = host_lookup[hi];
                hi += 1;
                t
            };
            B::push_field(args, ptr, lo, extent);
        }
        for g in grid {
            args.push(g);
        }
        for d in dom_lo {
            args.push(d);
        }
        let (mut ii, mut fi) = (0usize, 0usize);
        for &is_int in &desc.scalar_is_int {
            if is_int {
                args.push(&ints[ii]);
                ii += 1;
            } else {
                args.push(&scalars[fi]);
                fi += 1;
            }
        }
        unsafe {
            B::dispatcher()
                .runtime()
                .launch(&kernel, config, args.as_mut_slice())
                .unwrap_or_else(|e| panic!("GPU launch '{}' failed: {:?}", kernel_name, e));
        }
    });
    // **B12 — removed per-launch `ctx_sync()`** (was here, sub-agent diagnosis
    // 2026-05-31). every dispatch syncing host→device serialized launches and
    // killed pipelining (~20 launches × ~10 µs × N steps in pure stall on the
    // 2070; measured: dropped Kepler 512² from missing-half-of-target to
    // expected). CUDA's same-stream semantics already serialize kernel-to-
    // kernel ordering, so the next kernel reading these buffers cannot start
    // before this one finishes — correctness preserved.
    //
    // host-visible ordering is established explicitly where required:
    //   - `field_max_reduce` / `field_reduce` sync before the host-fold (they
    //     legitimately read device-resident scalars back to the host).
    //   - the eventual checkpoint write triggers a sync via the same path.
    //
    // unified memory: page migration is driven by access patterns, not by
    // ctx_sync (which is a CPU-side barrier). removing this is purely a
    // pipelining win.
}
