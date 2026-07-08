// =============================================================================
// policy.rs
//
// the CPU/GPU EXECUTOR seam for cell-centered structured invocations: `dispatch_fields`
// (shared-layout), `dispatch_fields_cover` (disjoint-cover fork-join), `dispatch_fields_each`
// (per-buffer layouts), the `ExecPolicy` strategy + its `policy_for` / `run_policy` /
// `auto_block_size` selection, and the env-gated dispatch micro-profiler. all parallelism
// lives here — physics dispatch (the substrate, in `symbi`) only chooses a policy and
// binds buffers.
// =============================================================================

use symbi_algebra::{BlockGrid, Domain, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use symbi_grid::Field;
use symbi_xpu::MemorySpace;

use symbi_aot::{Buf, BufHandle, KernelInvocation};

use crate::engine::dispatch;

use crate::layout::{alloc_layout, exec_layout, expect_kernel};

/// build + dispatch a structured invocation over cell-centered buffers: `inputs`
/// (read-only, `Host`) then `outputs` (`HostMut`, including in-place read+write), all
/// on the allocated layout, executed over `exec`. the kernel `name` is resolved to
/// its structured CPU fn + neutral IR blob via the generated registry; the CPU path
/// runs the AOT fn, a device-accessible `Mem` renders + JITs the IR (the GPU seam).
///
/// `inputs` MUST be in the generated kernel's input-binding order and `outputs` in
/// its output-binding order (`run_cpu` re-splits by handle, preserving order within
/// each group, so the call site need not interleave them the way the binding list
/// happens to).
///
/// SAFETY: the caller guarantees every `outputs` field is a distinct allocation from
/// the others and from `inputs` — the multiple `&mut` slices then alias nothing.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fields<Sc: Scalar + OrderedNumeric, Mem: MemorySpace, const D: usize>(
    name:      &str,
    allocated: &Domain<D>,
    exec:      &Domain<D>,
    inputs:    &[&Field<Sc, D, Mem>],
    outputs:   &[&Field<Sc, D, Mem>],
    ints:      &[i32],
    scalars:   &[Sc],
) {
    let (grid, dlo) = exec_layout(exec);
    // cell-centered: one SHARED allocated layout, replicated per field for the disjoint constructor.
    let shared = alloc_layout(allocated);
    let layouts: smallvec::SmallVec<[([i32; D], [u32; D], usize); 16]> =
        std::iter::repeat(shared).take(inputs.len() + outputs.len()).collect();
    let buffers = disjoint_host_buffers(name, inputs, outputs, &layouts);
    let inv = KernelInvocation { buffers, grid: &grid, dom_lo: &dlo, ints, scalars };
    let (cpu, ir) = expect_kernel::<Sc>(name);
    dispatch::<Sc, Mem, _>(inv, ir, name, cpu);
}

/// build a kernel's WHOLE-buffer host binding set from its input + output fields over a shared
/// `(lo, extent)` layout — the ONE place the `from_raw_parts` whole-buffer construction lives,
/// with the disjoint-write contract checked ONCE, RELEASE-ACTIVE. inputs bind `&[T]`, outputs
/// `&mut [T]`; if two bindings shared a backing allocation the resulting `&` + `&mut` (or two
/// `&mut`s) would alias — UB under Stacked/Tree Borrows, and a SILENT garbled-physics bug on
/// release (the class a debug-only guard misses). a manifest with a duplicate path or a caller
/// binding the same field twice fails loudly here. the per-block (`dispatch_fields_cover`) and
/// per-face (`dispatch_fields_each`) executors slice per-region and keep their own builds; this
/// is the whole-buffer case shared by `dispatch_fields` + the runtime-source dispatch.
///
/// SAFETY: the caller guarantees every `inputs`/`outputs` field outlives `'a` (the dispatch
/// scope) and that the kernel reads inputs immutably + writes only its own outputs; the
/// distinctness check above makes the no-aliasing precondition release-enforced, not convention.
pub fn disjoint_host_buffers<'a, Sc, const D: usize, Mem>(
    name: &str,
    inputs: &[&Field<Sc, D, Mem>],
    outputs: &[&Field<Sc, D, Mem>],
    layouts: &'a [([i32; D], [u32; D], usize)],
) -> Vec<Buf<'a, Sc>>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    // `layouts` carries one `(lo, extent, vol)` per field, in `inputs ++ outputs` order — a SHARED
    // cell layout (replicated) for `dispatch_fields`, or each field's own `Field::domain()` layout
    // for `dispatch_fields_each` (staggered / mixed-domain binds). ONE constructor, ONE distinctness
    // check (release-active) — the "DisjointBufferSet" SSOT (docs/design/38).
    debug_assert_eq!(
        layouts.len(), inputs.len() + outputs.len(),
        "disjoint_host_buffers('{name}'): one layout per field required",
    );
    let mut seen_ptrs = std::collections::HashSet::with_capacity(inputs.len() + outputs.len());
    assert!(
        inputs.iter().chain(outputs.iter()).all(|f| seen_ptrs.insert(f.as_ptr() as usize)),
        "disjoint_host_buffers('{name}'): two bindings resolve to the same allocation — input/output aliasing would be UB. either the kernel's IR manifest has a duplicate path, or the caller bound the same field twice."
    );
    let mut buffers: Vec<Buf<'a, Sc>> = Vec::with_capacity(inputs.len() + outputs.len());
    for (f, (lo, ext, vol)) in inputs.iter().zip(layouts.iter()) {
        buffers.push(Buf {
            handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(f.as_ptr(), *vol) }),
            lo,
            extent: ext,
        });
    }
    for (f, (lo, ext, vol)) in outputs.iter().zip(layouts.iter().skip(inputs.len())) {
        buffers.push(Buf {
            handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(f.as_mut_ptr(), *vol) }),
            lo,
            extent: ext,
        });
    }
    buffers
}

/// THE EXECUTOR INVERSION (cpu). dispatch `name` over a DISJOINT COVER of the exec
/// window in ONE rayon fork-join: the parallelism moves OUT of the kernel (which
/// runs serially per block) and INTO the executor, which fans the cover out. this
/// is what makes a block decomposition pay off — N blocks cost one launch, not N.
///
/// SOUNDNESS rests entirely on `cover` being a PARTITION of the exec window (the
/// proven `BlockGrid` / `guillotine_difference` law): the blocks write DISJOINT
/// output cells, so the per-task raw-`&mut` reconstruction (the same pattern the
/// parallel kernel uses internally, lifted to block granularity) never aliases a
/// live write. inputs are read-only and shared. the axiom is load-bearing here.
///
/// returns `false` (caller falls back) when the `{name}_serial` twin was not
/// generated (`SYMBI_GEN_SERIAL` off) or the memory space is device-resident —
/// the executor is a host scheduler; the gpu owns its own parallelism.
#[must_use]
pub fn dispatch_fields_cover<Sc: Scalar + OrderedNumeric, Mem: MemorySpace, const D: usize>(
    name:    &str,
    exec:    &Domain<D>,
    block:   [usize; D],
    inputs:  &[&Field<Sc, D, Mem>],
    outputs: &[&Field<Sc, D, Mem>],
    ints:    &[i32],
    scalars: &[Sc],
) -> bool {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    if Mem::IS_DEVICE_ACCESSIBLE {
        // host scheduler only; the gpu parallelizes a single launch over all cells
        // itself. defense-in-depth — the target-aware policy in `dispatch_named`
        // already passes `None` (whole window) on a device target, so this is
        // unreachable from there; it guards any other caller.
        return false;
    }
    let serial_name = format!("{name}_serial");
    let serial = match symbi_aot::kernel_by_name::<Sc>(&serial_name) {
        Some((cpu, _ir)) => cpu, // KernelFn is a Copy fn-pointer (Send + Sync).
        None => return false,    // twin not generated -> caller falls back.
    };
    // SAFETY KEYSTONE (hard, release-active). this executor slices each whole buffer
    // into per-block `&mut` via `from_raw_parts_mut`, trusting that every OUTPUT is a
    // DISTINCT allocation from every other binding: inter-BLOCK disjointness is the
    // proven partition law, but inter-FIELD disjointness is the caller's contract. a
    // duplicate output (or an output bound as an input) would alias live `&mut`s
    // across every block -> UB that silently garbles physics on release. DUPLICATE
    // INPUTS are sound (shared `&[T]` reads alias nothing) and intentional — the
    // prolong binds the same coarse buffer as src_old/src_new. the whole-window paths
    // assert this in debug; the cover path is the production CPU parallelism, so it
    // asserts ALWAYS, not just in test builds.
    assert!(
        {
            let mut out_ptrs = std::collections::HashSet::new();
            let in_ptrs: std::collections::HashSet<usize> =
                inputs.iter().map(|f| f.as_ptr() as usize).collect();
            outputs.iter().all(|f| {
                let p = f.as_ptr() as usize;
                out_ptrs.insert(p) && !in_ptrs.contains(&p)
            })
        },
        "dispatch_fields_cover('{name}'): an output aliases another output or an input \
         — the per-block &mut reconstruction would alias and be UB. either the kernel's \
         IR manifest has a duplicate output path, or the caller bound an output twice."
    );
    // per-field allocation layouts — coarse INPUTS and fine OUTPUTS may live in
    // DIFFERENT allocated domains (the prolong case), so each buffer carries its
    // own (lo, extent, vol). the block only restricts the exec WINDOW, not the
    // buffers. computed once; stack-resident up to 16 buffers.
    let layouts: smallvec::SmallVec<[([i32; D], [u32; D], usize); 16]> = inputs
        .iter()
        .chain(outputs.iter())
        .map(|f| alloc_layout(f.domain()))
        .collect();
    let n_in = inputs.len();

    // raw buffer pointers, shared across the rayon tasks. inputs are read-only;
    // outputs are written to DISJOINT cells per block (the partition law), so the
    // newtype's Send/Sync is sound for this access pattern.
    struct CoverPtrs<T>(Vec<*const T>, Vec<*mut T>);
    unsafe impl<T> Send for CoverPtrs<T> {}
    unsafe impl<T> Sync for CoverPtrs<T> {}
    impl<T> CoverPtrs<T> {
        // accessed through methods so the closure captures the whole (Send+Sync)
        // wrapper, not the inner raw-ptr Vecs (Rust 2021 disjoint capture would
        // otherwise grab `*const T`/`*mut T`, which are not Sync).
        fn ins(&self) -> &[*const T] { &self.0 }
        fn outs(&self) -> &[*mut T] { &self.1 }
    }
    let ptrs = CoverPtrs(
        inputs.iter().map(|f| f.as_ptr()).collect::<Vec<_>>(),
        outputs.iter().map(|f| f.as_mut_ptr()).collect::<Vec<_>>(),
    );

    // LAZY cover: iterate block INDICES and derive each window by arithmetic
    // (`BlockGrid::window`) — no `Vec<Domain>`, no per-block `DomainId` atomic.
    // an 8-edge tile makes 32k blocks of a 256^3 grid; materializing them per
    // dispatch was the large-grid crumble. one fork-join over all blocks.
    let bg = BlockGrid::new(exec.clone(), block);
    (0..bg.len()).into_par_iter().for_each(|bi| {
        let (lo, size) = bg.window(bi);
        let grid: [u32; D] = std::array::from_fn(|a| size[a] as u32);
        let dlo: [i32; D] = std::array::from_fn(|a| lo[a] as i32);
        let mut buffers: Vec<Buf<Sc>> = Vec::with_capacity(layouts.len());
        for (i, &p) in ptrs.ins().iter().enumerate() {
            let (l, ext, vol) = &layouts[i];
            // SAFETY: `p` spans the whole allocated field (`vol`); the block only
            // restricts the OUTPUT window — stencil reads stay in-bounds.
            buffers.push(Buf {
                handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(p, *vol) }),
                lo: l,
                extent: ext,
            });
        }
        for (j, &p) in ptrs.outs().iter().enumerate() {
            let (l, ext, vol) = &layouts[n_in + j];
            // SAFETY: blocks partition the exec window (proven) -> this block's
            // written cells are disjoint from every other task's.
            buffers.push(Buf {
                handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(p, *vol) }),
                lo: l,
                extent: ext,
            });
        }
        let inv = KernelInvocation { buffers, grid: &grid, dom_lo: &dlo, ints, scalars };
        inv.run_cpu(serial);
    });
    true
}

// env-gated micro-profiler (SYMBI_DISPATCH_PROF=1) for the per-call overhead of
// `dispatch_fields_each` — the amr-transfer / register hot path (flux/c2p take a
// different dispatch, so this is purely the AMR bookkeeping). splits the registry
// name lookup from the kernel execution and counts calls, to attribute the
// prolong cost to (1) rayon launch + work, (2) the 285-arm name match, (3) the
// per-field dispatch count. main-thread only (the rayon fan-out lives INSIDE the
// kernel), so the counters are uncontended.
static DISP_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static DISP_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISP_LOOKUP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISP_EXEC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn dispatch_profiling() -> bool {
    *DISP_ON.get_or_init(|| std::env::var("SYMBI_DISPATCH_PROF").is_ok())
}
/// (calls, total ns in registry lookup, total ns in kernel execution).
pub fn report_dispatch_profile() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DISP_COUNT.load(Relaxed), DISP_LOOKUP_NS.load(Relaxed), DISP_EXEC_NS.load(Relaxed))
}
/// clear the dispatch-profile accumulators (call after warmup).
pub fn reset_dispatch_profile() {
    use std::sync::atomic::Ordering::Relaxed;
    DISP_COUNT.store(0, Relaxed);
    DISP_LOOKUP_NS.store(0, Relaxed);
    DISP_EXEC_NS.store(0, Relaxed);
}

/// `dispatch_fields` for buffers that do NOT share one allocated layout: each
/// field's lo/extent come from ITS OWN domain. the amr transfer kernels need
/// this — the restrict/prolong source lives on one refinement level and the
/// destination on another, and the absolute-index load resolves against each
/// buffer's own lo. duplicate INPUT fields are allowed (the prolong binds the
/// same coarse buffer as src_old and src_new when no time interpolation is
/// wanted — shared `&[T]` reads alias soundly); outputs must stay distinct
/// from everything.
pub fn dispatch_fields_each<Sc: Scalar + OrderedNumeric, Mem: MemorySpace, const D: usize>(
    name:    &str,
    exec:    &Domain<D>,
    inputs:  &[&Field<Sc, D, Mem>],
    outputs: &[&Field<Sc, D, Mem>],
    ints:    &[i32],
    scalars: &[Sc],
) {
    let (grid, dlo) = exec_layout(exec);

    // UNIVERSAL EXECUTOR (the prolong / amr-transfer path too). cache-tile big
    // windows into a disjoint cover dispatched in ONE fork-join — at large grids
    // the prolong slabs are big, and the scattered coarse-stencil reads benefit
    // from staying L1-resident exactly like the interior kernels. host only (the
    // target-aware policy keeps one launch on device); falls through to the single
    // dispatch when small / when the serial twin isn't built. bit-identical: each
    // output cell is computed once over a partition of the window.
    let policy = policy_for(exec, Mem::IS_DEVICE_ACCESSIBLE);
    if run_policy(policy, |block| {
        dispatch_fields_cover::<Sc, Mem, D>(name, exec, block, inputs, outputs, ints, scalars)
    }) {
        return;
    }

    // PER-BUFFER layouts: each field's own `Field::domain()` (staggered/mixed-domain binds), then
    // the ONE disjoint constructor (release-active distinctness). materialized first so the Buf refs
    // stay valid; stack-resident (bounded by buffer count) so the hot AMR transfer / register
    // dispatches don't heap-allocate per call.
    let layouts: smallvec::SmallVec<[([i32; D], [u32; D], usize); 16]> = inputs
        .iter()
        .chain(outputs.iter())
        .map(|f| alloc_layout(f.domain()))
        .collect();
    let buffers = disjoint_host_buffers(name, inputs, outputs, &layouts);
    let inv = KernelInvocation { buffers, grid: &grid, dom_lo: &dlo, ints, scalars };
    if dispatch_profiling() {
        use std::sync::atomic::Ordering::Relaxed;
        let t0 = std::time::Instant::now();
        let (cpu, ir) = expect_kernel::<Sc>(name);
        let t1 = std::time::Instant::now();
        dispatch::<Sc, Mem, _>(inv, ir, name, cpu);
        let t2 = std::time::Instant::now();
        DISP_COUNT.fetch_add(1, Relaxed);
        DISP_LOOKUP_NS.fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
        DISP_EXEC_NS.fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
    } else {
        let (cpu, ir) = expect_kernel::<Sc>(name);
        dispatch::<Sc, Mem, _>(inv, ir, name, cpu);
    }
}

/// the ONE CPU-parallelism strategy for an interior dispatch, decided at the ONE
/// scheduling seam (`policy_for`) and consumed at the ONE site (`run_policy`).
/// `Whole` runs the kernel's own internal rayon over the whole exec window (the
/// small-domain / device / no-serial-twin fallback); `Cover` fans a SERIAL kernel
/// out over a disjoint `BlockGrid` cover in ONE fork-join (the big-domain win — the
/// parallelism moves OUT of the kernel and INTO the executor). bit-identical either
/// way: each output cell is computed once over a partition of the same window.
pub enum ExecPolicy<const D: usize> {
    Whole,
    Cover([usize; D]),
}

/// SELECT the CPU-parallelism strategy for `exec`. PRODUCTION behavior depends ONLY
/// on (domain size, target) — never on a build-time flag. DEVICE targets always run
/// `Whole`: a single domain is already one GPU launch over all cells (optimal); a
/// cover would be N launches (~10us each — a regression). HOST targets auto-tile:
/// small domains stay `Whole` (sit in cache; the kernel's own rayon suffices), big
/// domains take a cache-sized `Cover` (the 3D throughput lever).
///
/// `SYMBI_BLOCK` is a DEBUG-ONLY A/B override of the host heuristic (`"b"` /
/// `"bx,by,bz"` fixes the cover edge; `"off"` / `"0"` forces `Whole`); it does not
/// gate the production path. the `_serial` twins the `Cover` path needs are always
/// generated; `run_policy` falls back to `Whole` if a twin is missing.
pub fn policy_for<const D: usize>(exec: &Domain<D>, is_device: bool) -> ExecPolicy<D> {
    if is_device {
        return ExecPolicy::Whole;
    }
    enum Mode {
        Auto,
        Off,
        Fixed(Vec<usize>),
    }
    // debug-only A/B override of the host heuristic. unset in production.
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    let mode = MODE.get_or_init(|| match std::env::var("SYMBI_BLOCK") {
        Err(_) => Mode::Auto,
        Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Mode::Auto,
            "0" | "off" | "none" | "whole" => Mode::Off,
            other => {
                let vals: Vec<usize> = other
                    .split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .filter(|&v| v > 0)
                    .collect();
                if vals.is_empty() { Mode::Auto } else { Mode::Fixed(vals) }
            }
        },
    });
    match mode {
        Mode::Off => ExecPolicy::Whole,
        Mode::Fixed(vals) => ExecPolicy::Cover(std::array::from_fn(|a| vals[a.min(vals.len() - 1)])),
        Mode::Auto => match auto_block_size(exec.shape(), rayon::current_num_threads()) {
            Some(block) => ExecPolicy::Cover(block),
            None => ExecPolicy::Whole,
        },
    }
}

/// CONSUME an `ExecPolicy` — the ONE site that turns a chosen strategy into work.
/// `Cover(block)` runs `cover(block)`, which fans the SERIAL kernel over the disjoint
/// cover and returns whether it actually executed (it returns `false` when the
/// `_serial` twin isn't compiled in). returns `true` iff the cover handled the
/// dispatch; on `false` (or `Whole`) the caller runs its own whole-window fallback.
/// keeping the policy->cover decision here means neither call site re-derives the
/// "device? small? twin-missing?" logic — they only supply how to run the cover.
pub fn run_policy<const D: usize>(
    policy: ExecPolicy<D>,
    cover:  impl FnOnce([usize; D]) -> bool,
) -> bool {
    match policy {
        ExecPolicy::Whole => false,
        ExecPolicy::Cover(block) => cover(block),
    }
}

/// AUTO block policy — CACHE TILING. keep the WHOLE domain (`None`) only when it is
/// small enough to already sit in cache; otherwise tile it into small fixed-edge
/// blocks whose per-cell stencil working set stays in L1/L2 instead of streaming
/// the whole grid from RAM.
///
/// this is the 3D throughput lever. a stencil kernel re-reads each cell's neighbours
/// across adjacent output cells; if the block is large the reuse distance exceeds
/// cache and every read hits RAM (the memory wall — why 3D MZCS *falls* as the grid
/// grows). a small transverse tile keeps the reuse in cache. sizing
/// blocks for LOAD BALANCE (~4*threads big blocks) is exactly wrong here.
///
/// measured on M4 (sedov3d 128^3, ~8 fields f64): edge 8 (8^3 ~ 32 KB ~ L1) is the
/// sweet spot — 22 (old auto) / 20 (off) -> 33 MZCS, near AthenaK's 35. edges 16/32
/// fall off as the tile leaves cache; edge 4 loses to per-block overhead. `CACHE_EDGE`
/// is the one tunable; `SYMBI_BLOCK=<n>` overrides it per run.
fn auto_block_size<const D: usize>(shape: [usize; D], _threads: usize) -> Option<[usize; D]> {
    // the cache tile edge (cells per axis). 8 keeps ~8 f64 fields of an 8^3 block
    // (~32 KB) resident; clamps to the axis extent for thin domains.
    const CACHE_EDGE: usize = 8;
    // a domain that already fits cache gains nothing from tiling (and pays the
    // per-block overhead) — keep it whole. ~ one cache tile's worth across axes.
    const WHOLE_BELOW_CELLS: usize = 32 * 1024;
    let n: usize = shape.iter().product();
    if n < WHOLE_BELOW_CELLS {
        return None;
    }
    Some(std::array::from_fn(|a| CACHE_EDGE.min(shape[a]).max(1)))
}
