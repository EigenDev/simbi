// =============================================================================
// decomp.rs
//
// same-level domain decomposition: halo exchange between neighboring subdomains on a
// grid, behind a transport seam.
//
// the exchange LOGIC (which cells move where) is fixed and proven by the in-process
// oracle `symbi/tests/decomp_equivalence.rs`. the TRANSPORT (how the bytes move) varies
// behind `HaloTransport`: a local memory copy today, a gpu peer copy and an mpi
// pack/send/unpack later. swapping the transport never touches the decomposition.
//
// the exchange is a two-pass scheme: the caller processes axes in order, and a cut
// face's transverse extent is the interior for cut axes not yet exchanged, the full
// allocated extent otherwise. that carries corner ghosts to the diagonal neighbor
// without any explicit diagonal traffic (docs/design/36).
//
// everything is built on `Domain` (slab / boundary / iter): the strips are domains and
// the copy is a cell-for-cell walk of two equal-shape domains, so the index arithmetic
// lives in one tested place.
// =============================================================================

use crate::state::{FieldStore, PartitionGeometry, Timestepping};
use crate::substrate_seam::KernelSet;
use std::ops::ControlFlow;
use symbi_algebra::{Domain, Side};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;
#[cfg(feature = "gpu")]
use symbi_xpu::{can_access_peer, ctx_sync, memcpy_peer, DeviceMemory, MAX_GPUS};
#[cfg(feature = "gpu")]
use symbi_xpu::runtime::{current_dispatcher, GpuRuntime};
#[cfg(feature = "gpu")]
use symbi_xpu::{with_device, KernelArgs, LaunchConfig, MemoryBlock};

/// move `src` over `src_region` into `dst` over `dst_region`, cell-for-cell. the two
/// regions have identical shape (same per-axis extents); the transport pairs them by
/// iteration order. `src_dev`/`dst_dev` are the LOGICAL devices the two fields live on --
/// the one piece of device identity a cross-device exchange genuinely needs (the rest of the
/// code uses the ambient current-device model, docs/design/37). single-device transports
/// (`LocalCopy`, `DeviceCopy`, `StagedCopy`) ignore them; `PeerCopy` uses them to drive
/// `cuMemcpyPeer` between the two devices.
pub trait HaloTransport {
    fn copy_region<const D: usize, M: MemorySpace>(
        &self,
        src: &Field<f64, D, M>,
        src_region: &Domain<D>,
        dst: &Field<f64, D, M>,
        dst_region: &Domain<D>,
        src_dev: i32,
        dst_dev: i32,
    );
}

/// the proven in-process transport: a direct view-to-view copy.
pub struct LocalCopy;

impl HaloTransport for LocalCopy {
    fn copy_region<const D: usize, M: MemorySpace>(
        &self,
        src: &Field<f64, D, M>,
        src_region: &Domain<D>,
        dst: &Field<f64, D, M>,
        dst_region: &Domain<D>,
        _src_dev: i32,
        _dst_dev: i32,
    ) {
        let sv = src.view();
        let mut dv = dst.view_mut();
        for (sc, dc) in src_region.iter().zip(dst_region.iter()) {
            *dv.at_mut(dc) = *sv.at(sc);
        }
    }
}

// a D-independent gather/scatter copy kernel: thread i moves one strip cell, reading and
// writing precomputed flat offsets into each field's own buffer. one kernel covers every
// dimension and every stride because the geometry is baked into the index arrays. this is
// the same pack/move/unpack primitive a peer-copy or mpi transport reuses -- only the move
// (here a same-buffer-space device copy) changes.
#[cfg(feature = "gpu")]
const HALO_COPY_KERNEL: &str = r#"
extern "C" __global__ void halo_copy(
    const double* src,
    double* dst,
    const unsigned int* sidx,
    const unsigned int* didx,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[didx[i]] = src[sidx[i]];
}
"#;

// reusable unified index buffers for `DeviceCopy`, grown to the largest strip seen. the
// strip geometry is fixed across steps, so this reuses the same device allocations instead
// of a `cuMemAllocManaged` per `copy_region` (the dominant per-exchange cost). thread-local
// so it needs no Send/Sync on the raw device pointers; the parallel test threads each pool
// their own. the per-cell index WRITE still happens each call (cheap host memory writes).
#[cfg(feature = "gpu")]
struct HaloIdxBufs {
    sidx: MemoryBlock<DeviceMemory>,
    didx: MemoryBlock<DeviceMemory>,
    cap: usize,
}

#[cfg(feature = "gpu")]
thread_local! {
    static HALO_IDX_BUFS: std::cell::RefCell<Option<HaloIdxBufs>> =
        const { std::cell::RefCell::new(None) };
}

/// device-side transport: the strip copy runs as a gpu kernel, with no host roundtrip of
/// the field data. the per-cell flat-offset arrays are precomputed on the host into pooled
/// unified buffers; the data itself never leaves the device. on a host backend it falls
/// back to the proven cell-for-cell copy.
#[cfg(feature = "gpu")]
pub struct DeviceCopy;

#[cfg(feature = "gpu")]
impl HaloTransport for DeviceCopy {
    fn copy_region<const D: usize, M: MemorySpace>(
        &self,
        src: &Field<f64, D, M>,
        src_region: &Domain<D>,
        dst: &Field<f64, D, M>,
        dst_region: &Domain<D>,
        _src_dev: i32,
        _dst_dev: i32,
    ) {
        // the kernel dereferences device pointers; a host backend has none.
        if !M::IS_DEVICE_ACCESSIBLE {
            LocalCopy.copy_region(src, src_region, dst, dst_region, _src_dev, _dst_dev);
            return;
        }

        let n = src_region.volume();
        debug_assert_eq!(n, dst_region.volume(), "src/dst strip shapes differ");
        if n == 0 {
            return;
        }

        let sdom = src.domain();
        let ddom = dst.domain();
        let src_ptr = src.as_ptr() as u64;
        let dst_ptr = dst.as_mut_ptr() as u64;
        let n_u32 = n as u32;

        HALO_IDX_BUFS.with(|cell| {
            let mut slot = cell.borrow_mut();
            // (re)allocate only when missing or too small; the strip geometry is fixed.
            if slot.as_ref().map_or(true, |b| b.cap < n) {
                *slot = Some(HaloIdxBufs {
                    sidx: MemoryBlock::<DeviceMemory>::for_elements::<u32>(n).expect("sidx alloc"),
                    didx: MemoryBlock::<DeviceMemory>::for_elements::<u32>(n).expect("didx alloc"),
                    cap: n,
                });
            }
            let bufs = slot.as_mut().unwrap();

            // precompute the flat offset of each strip cell into its field buffer. these match
            // the View::at offsets because both index the field's allocated domain.
            let sp = bufs.sidx.as_mut_ptr::<u32>();
            let dp = bufs.didx.as_mut_ptr::<u32>();
            for (i, (sc, dc)) in src_region.iter().zip(dst_region.iter()).enumerate() {
                unsafe {
                    *sp.add(i) = sdom.flat_index(sc) as u32;
                    *dp.add(i) = ddom.flat_index(dc) as u32;
                }
            }
            let sidx_ptr = bufs.sidx.as_ptr::<u32>() as u64;
            let didx_ptr = bufs.didx.as_ptr::<u32>() as u64;

            let kernel =
                current_dispatcher().jit_kernel_keyed(HALO_COPY_KERNEL, "decomp/halo_copy", "halo_copy");
            let mut args = KernelArgs::new();
            args.push(&src_ptr);
            args.push(&dst_ptr);
            args.push(&sidx_ptr);
            args.push(&didx_ptr);
            args.push(&n_u32);

            let config = LaunchConfig::for_1d(n_u32, 64);
            unsafe {
                current_dispatcher()
                    .runtime()
                    .launch(&kernel, config, args.as_mut_slice())
                    .expect("halo_copy launch failed");
            }
            // drain before the pooled buffers are reused next call: the kernel reads them and
            // the launch is async.
            ctx_sync();
        });
    }
}

/// the ng-deep ghost strip on `side` of `axis`, with the transverse extent clipped to
/// the interior on any cut axis not yet exchanged (`!processed[b] && counts[b] > 1`) --
/// the two-pass corner rule. on a physical axis, or a cut axis already exchanged, the
/// full allocated extent is kept (its ghosts are valid).
fn ghost_strip<const D: usize>(
    geom: &PartitionGeometry<D>,
    axis: usize,
    side: Side,
    processed: &[bool; D],
    counts: &[usize; D],
) -> Domain<D> {
    let ng = geom.ng as isize;
    let mut strip = geom.allocated.boundary(axis, side, ng);
    for b in 0..D {
        if b != axis && !processed[b] && counts[b] > 1 {
            strip = strip.slab(b, (geom.interior.spaces[b].lo, geom.interior.spaces[b].hi));
        }
    }
    strip
}

/// every per-cell DATA component a checkpoint needs: the conserved set (den, momentum, energy)
/// plus the primitives (rho, velocity, pressure). used by `gather_interiors` to reassemble a
/// decomposed run into one global store for output -- the gathered global then writes through
/// the existing single-grid checkpoint path unchanged (docs/design/37 M4, v1 hydro). mhd
/// staggered B is not gathered here (multi-gpu mhd is a later increment).
fn data_fields<const D: usize, const DOF: usize, M: MemorySpace>(
    store: &FieldStore<D, DOF, M>,
) -> Vec<&Field<f64, D, M>> {
    let c = &store.fields.cons;
    let p = &store.fields.prim;
    std::iter::once(&c.den)
        .chain(c.mom.iter())
        .chain(c.nrg.as_ref())
        .chain(std::iter::once(&p.rho))
        .chain(p.vel.iter())
        .chain(p.pre.as_ref())
        .collect()
}

/// reassemble a decomposed run into one full-size `global` store: copy each tile's INTERIOR
/// (cons + prim) into the matching sub-box of `global` (docs/design/37 M4). the inverse of the
/// tile build's IC scatter; the gathered `global` is then written by the existing single-grid
/// checkpoint writer, so decomposed output is byte-identical in format to a single-device run.
///
/// `tiles` are in flat tile order (matching `flatten`); `counts` is the per-axis tile grid.
/// each tile holds `global_interior / counts` cells per axis. the per-field copy reuses the
/// proven `LocalCopy` cell-walk (host-side over managed memory; the caller drains the devices
/// first), so the gather shares the exchange's tested index arithmetic.
pub fn gather_interiors<const D: usize, const DOF: usize, M: MemorySpace>(
    global: &FieldStore<D, DOF, M>,
    tiles: &[&FieldStore<D, DOF, M>],
    counts: [usize; D],
) {
    let gint = &global.geom.interior;
    // cells per tile per axis = global interior extent / tile count.
    let m: [usize; D] = std::array::from_fn(|ax| gint.spaces[ax].size() / counts[ax]);
    let glo: [isize; D] = std::array::from_fn(|ax| gint.spaces[ax].lo);

    for (flat, tile) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        let src_region = tile.geom.interior.clone();
        // the m-cell sub-box of the global interior this tile owns (global coords).
        let mut dst_region = global.geom.interior.clone();
        for ax in 0..D {
            let lo = glo[ax] + (tc[ax] * m[ax]) as isize;
            dst_region = dst_region.slab(ax, (lo, lo + m[ax] as isize));
        }
        for (gf, tf) in data_fields(global).into_iter().zip(data_fields(tile)) {
            LocalCopy.copy_region(tf, &src_region, gf, &dst_region, 0, 0);
        }
    }
}

/// the CELL-CENTERED components the flux stage reconstructs from: rho, each velocity, pressure,
/// and -- for MHD -- the cell-centered magnetic field `bcell` (the reconstruction reads it like
/// any other primitive). hydro/iso stores have no `mhd`, so the bcell tail is empty there and
/// the exchange is unchanged. the STAGGERED `bface` is exchanged separately (`bface_strips`),
/// since its faces live on a different domain than the cells.
fn prim_fields<const D: usize, const DOF: usize, M: MemorySpace>(
    store: &FieldStore<D, DOF, M>,
) -> Vec<&Field<f64, D, M>> {
    let mut fields: Vec<&Field<f64, D, M>> = std::iter::once(&store.fields.prim.rho)
        .chain(store.fields.prim.vel.iter())
        .chain(store.fields.prim.pre.as_ref())
        .collect();
    if let Some(mhd) = store.fields.mhd.as_ref() {
        fields.extend(mhd.bcell.b.iter());
    }
    fields
}

/// the ghost strip on a STAGGERED face field's own allocated domain (for MHD `bface`). mirrors
/// `ghost_strip` but takes the field's `alloc` domain instead of the cell `geom.allocated`, and
/// extends the interior by one on the field's NORMAL axis `d` (a face field has one extra face
/// past the last cell on its normal axis). used only for `d != axis`, where `axis` is transverse
/// to the face and indexes like cells.
fn face_ghost_strip<const D: usize>(
    alloc: &Domain<D>,
    geom: &PartitionGeometry<D>,
    axis: usize,
    side: Side,
    d: usize,
    processed: &[bool; D],
    counts: &[usize; D],
) -> Domain<D> {
    let ng = geom.ng as isize;
    let mut strip = alloc.boundary(axis, side, ng);
    for b in 0..D {
        if b != axis && !processed[b] && counts[b] > 1 {
            // clip to interior on un-processed cut axes (two-pass corner rule). on the face's
            // own normal axis there is one extra face past the last interior cell.
            let hi_ext = if b == d { 1 } else { 0 };
            strip = strip.slab(b, (geom.interior.spaces[b].lo, geom.interior.spaces[b].hi + hi_ext));
        }
    }
    strip
}

/// exchange same-level halos across the shared face between `lo` (its hi face on `axis`)
/// and `hi` (its lo face). `processed` marks the axes already exchanged this pass (for
/// the two-pass transverse rule); `counts` is the per-axis tile count. fills the prim + bcell
/// ghost cells and -- for MHD -- the staggered `bface` transverse halos (cons ghosts are never
/// read by the flux stage).
///
/// each ghost strip and its source strip share global cell positions and shape; the
/// source strip is the ghost strip shifted along `axis` to the neighbor's matching
/// interior cells, so the transport copies them cell-for-cell.
pub fn exchange_faces<const D: usize, const DOF: usize, M: MemorySpace, T: HaloTransport>(
    lo: &FieldStore<D, DOF, M>,
    hi: &FieldStore<D, DOF, M>,
    axis: usize,
    processed: &[bool; D],
    counts: &[usize; D],
    lo_dev: i32,
    hi_dev: i32,
    transport: &T,
) {
    let ng = lo.geom.ng as isize;
    let i_hi_lo = lo.geom.interior.spaces[axis].hi; // one past lo's last interior cell
    let i_lo_hi = hi.geom.interior.spaces[axis].lo; // hi's first interior cell

    // lo's hi-ghost strip; hi's source is the same strip shifted to its first ng interior.
    let lo_ghost = ghost_strip(&lo.geom, axis, Side::Hi, processed, counts);
    let hi_src = lo_ghost.slab(axis, (i_lo_hi, i_lo_hi + ng));
    // hi's lo-ghost strip; lo's source is its last ng interior.
    let hi_ghost = ghost_strip(&hi.geom, axis, Side::Lo, processed, counts);
    let lo_src = hi_ghost.slab(axis, (i_hi_lo - ng, i_hi_lo));

    for (fl, fr) in prim_fields(lo).into_iter().zip(prim_fields(hi)) {
        // hi interior -> lo ghost: source is the hi tile (hi_dev), dest is the lo tile (lo_dev).
        transport.copy_region(fr, &hi_src, fl, &lo_ghost, hi_dev, lo_dev);
        // lo interior -> hi ghost: source is the lo tile, dest is the hi tile.
        transport.copy_region(fl, &lo_src, fr, &hi_ghost, lo_dev, hi_dev);
    }

    // MHD: exchange the STAGGERED face-B transverse halos. only `bface[d]` with `d != axis`
    // carries a halo on `axis` (the face is transverse to the cut, so `axis` indexes it like
    // cells); fill that halo from the neighbor exactly like the cell ghosts. the NORMAL face
    // `bface[axis]` (the shared interface face) is NOT copied: it is seeded identically and both
    // tiles apply the same CT curl -- the consistent edge emfs come from the exchanged transverse
    // halos + prim + bcell -- so it stays bit-identical. the div-B oracle validates this.
    if let (Some(lo_mhd), Some(hi_mhd)) = (lo.fields.mhd.as_ref(), hi.fields.mhd.as_ref()) {
        for d in 0..D {
            // the NORMAL face (d == axis) is the shared interface, owned by both tiles. it must
            // NOT be copied: it is the result of each tile's CT curl, and the discrete div(B)=0
            // depends on it being the curl output (overwriting it injects a monopole -- confirmed
            // empirically). only the TRANSVERSE faces carry a halo across this cut.
            if d == axis {
                continue;
            }
            let fl = &lo_mhd.bface[d];
            let fr = &hi_mhd.bface[d];
            let lo_alloc = fl.domain();
            let hi_alloc = fr.domain();
            let lo_ghost_f = face_ghost_strip(&lo_alloc, &lo.geom, axis, Side::Hi, d, processed, counts);
            let hi_src_f = lo_ghost_f.slab(axis, (i_lo_hi, i_lo_hi + ng));
            let hi_ghost_f = face_ghost_strip(&hi_alloc, &hi.geom, axis, Side::Lo, d, processed, counts);
            let lo_src_f = hi_ghost_f.slab(axis, (i_hi_lo - ng, i_hi_lo));
            transport.copy_region(fr, &hi_src_f, fl, &lo_ghost_f, hi_dev, lo_dev);
            transport.copy_region(fl, &lo_src_f, fr, &hi_ghost_f, lo_dev, hi_dev);
        }
    }
}

// prime factors of `n` in DESCENDING order (largest first), so the decomposition places the
// big cuts before the small ones onto the longest axes.
fn prime_factors_desc(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut d = 2;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors.reverse(); // ascending -> descending
    factors
}

/// choose a per-axis tile count whose product is `n_parts`, minimizing halo surface by cutting
/// the LONGEST axes first, subject to each axis count evenly dividing that axis's cells. errors
/// if no such factorization exists (e.g. a prime `n_parts` that divides no axis). `n_parts == 1`
/// is the monolithic `[1; D]`.
///
/// greedy and deterministic: prime-factor `n_parts` largest-first, and place each factor on the
/// axis with the most cells-per-current-tile that stays evenly divisible. good enough for the
/// regular grids simbi runs; a user override (Config) can bypass it when a specific shape is
/// wanted.
pub fn decompose_grid<const D: usize>(
    n_cells: [usize; D],
    n_parts: usize,
) -> Result<[usize; D], String> {
    let mut counts = [1usize; D];
    if n_parts <= 1 {
        return Ok(counts);
    }
    for f in prime_factors_desc(n_parts) {
        // among axes that remain evenly divisible after multiplying by `f`, take the longest
        // current tile edge (n_cells/counts) -- cut the biggest piece to balance surface area.
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for ax in 0..D {
            if n_cells[ax] % (counts[ax] * f) == 0 {
                let len = n_cells[ax] / counts[ax];
                if len > best_len {
                    best_len = len;
                    best = Some(ax);
                }
            }
        }
        match best {
            Some(ax) => counts[ax] *= f,
            None => {
                return Err(format!(
                    "cannot split grid {n_cells:?} across {n_parts} gpus: prime factor {f} divides \
                     no remaining axis evenly (each per-axis tile count must divide that axis's \
                     cells). pick a gpu count whose factors divide the resolution, or set an \
                     explicit decomposition."
                ))
            }
        }
    }
    Ok(counts)
}

// row-major (axis-0 slowest) flat index over a box of size `dims`, and its inverse. the
// tile grid uses this ONE convention for both construction and neighbor lookup (a `Domain`
// is avoided here on purpose: its iter order and flat_index order differ, which is a
// footgun for tile bookkeeping; `Domain` does its job at the cell level in `exchange_faces`).
pub fn flatten<const D: usize>(idx: [usize; D], dims: [usize; D]) -> usize {
    let mut f = 0;
    for a in 0..D {
        f = f * dims[a] + idx[a];
    }
    f
}

pub fn unflatten<const D: usize>(mut flat: usize, dims: [usize; D]) -> [usize; D] {
    let mut idx = [0usize; D];
    for a in (0..D).rev() {
        idx[a] = flat % dims[a];
        flat /= dims[a];
    }
    idx
}

/// two-pass same-level halo exchange over a grid of `counts` tiles. `tiles` holds each
/// subdomain's field store indexed by `flatten(tile_coord, counts)`. axes are processed in
/// order so a later axis reads the ghosts an earlier axis filled, carrying corner values to
/// the diagonal neighbor without explicit diagonal traffic. `transport` moves each strip
/// (local host copy, gpu kernel, or a future peer/mpi impl). each adjacent pair is
/// independent (reads interior, writes ghosts; an interior tile's lo/hi ghosts are distinct
/// cells), so pair order within an axis is free.
/// `devices[flatten(tile_coord, counts)]` is the logical device each tile lives on, parallel
/// to `tiles`. device identity lives here, in the decomposition's tile->device map (the way a
/// real spmd driver keeps it), not on the field -- the exchange is the one cross-device step.
pub fn exchange_grid<const D: usize, const DOF: usize, M: MemorySpace, T: HaloTransport>(
    tiles: &[&FieldStore<D, DOF, M>],
    counts: [usize; D],
    devices: &[i32],
    transport: &T,
) {
    let total: usize = counts.iter().product();
    debug_assert_eq!(tiles.len(), total, "tiles slice length does not match counts");
    debug_assert_eq!(devices.len(), total, "devices slice length does not match counts");
    let mut processed = [false; D];
    for axis in 0..D {
        for flat in 0..total {
            let tc = unflatten(flat, counts);
            if tc[axis] + 1 >= counts[axis] {
                continue; // no neighbor on the hi side of this axis
            }
            let mut tc_hi = tc;
            tc_hi[axis] += 1;
            let lo_flat = flatten(tc, counts);
            let hi_flat = flatten(tc_hi, counts);
            let lo = tiles[lo_flat];
            let hi = tiles[hi_flat];
            exchange_faces(lo, hi, axis, &processed, &counts, devices[lo_flat], devices[hi_flat], transport);
        }
        processed[axis] = true;
    }
}

// drain every UNIQUE tile device's context so async writes are visible to a consumer running
// in another context (the cross-device read barrier). no-op on a host backend. mirrors the
// oracle's `sync_devices` (docs/design/37 M2).
#[cfg(feature = "gpu")]
fn drain_devices<M: MemorySpace>(devices: &[i32]) {
    if !M::IS_DEVICE_ACCESSIBLE {
        return;
    }
    let mut seen: Vec<i32> = Vec::new();
    for &d in devices {
        if !seen.contains(&d) {
            seen.push(d);
            symbi_xpu::with_device(d, ctx_sync);
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn drain_devices<M: MemorySpace>(_devices: &[i32]) {}

/// drive a decomposed simulation: `stores.len()` tiles evolved in LOCKSTEP at a shared dt,
/// with a same-level halo exchange after each ssp stage. this IS the proven oracle loop
/// (`symbi/tests/decomp_equivalence.rs::run`), lifted into production so the multi-gpu python
/// entry and the oracle share ONE tested path (docs/design/37 M4).
///
/// - `stores[i]` / `kernels[i]`: tile i's field store + kernel set, in flat tile order
///   (matching `flatten(tile_coord, counts)`).
/// - `counts`: the per-axis tile grid; `devices[i]`: tile i's logical device (device identity
///   lives in the decomposition, not the field).
/// - `transport`: moves halos (`LocalCopy` host, `DeviceCopy`/`StagedCopy` single-gpu,
///   `PeerCopy` cross-gpu).
/// - `on_checkpoint(iteration, time)`: fires every `interval` steps (and once at the end);
///   returning `Break` stops the run. the device queue is drained before each call so the
///   callback can read coherent tile state for output.
///
/// the global dt is the min over all tiles' cfl -- identical to a monolithic run, since the
/// union of the tile interiors is the monolithic interior. rk2 exchanges BETWEEN stages so the
/// corrector reconstructs from each neighbor's stage-updated interior.
#[allow(clippy::too_many_arguments)]
pub fn evolve_decomposed<const D: usize, const DOF: usize, M, K, T, F>(
    stores: &[&FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
    counts: [usize; D],
    devices: &[i32],
    ts: Timestepping,
    start_time: f64,
    t_final: f64,
    interval: u64,
    transport: &T,
    mut on_checkpoint: F,
) where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    F: FnMut(u64, f64) -> ControlFlow<()>,
{
    let stages = ts.stages();
    let multistage = stages.len() > 1;
    let n = stores.len();
    debug_assert_eq!(n, kernels.len(), "stores/kernels length mismatch");
    debug_assert_eq!(n, devices.len(), "stores/devices length mismatch");

    // prime prim + ghosts (the stage entry contract), then seed the cut halos.
    for i in 0..n {
        symbi_xpu::with_device(devices[i], || {
            kernels[i].c2p(stores[i]);
            kernels[i].ghost_fill(stores[i]);
        });
    }
    drain_devices::<M>(devices);
    exchange_grid(stores, counts, devices, transport);
    // re-fill PHYSICAL boundary ghosts AFTER the exchange. at a corner where a domain-boundary
    // (outflow/reflect) meets a tile cut, the boundary ghost is derived from cells that include
    // the cut halo -- only valid post-exchange. with ghost_fill BEFORE the exchange, that corner
    // reads a stale (unexchanged) cut cell; for hydro it is harmless (uniform corners), but for
    // mhd the edge-EMF there is spurious and poisons the RK2 corrector. no-op interior cost.
    for i in 0..n {
        symbi_xpu::with_device(devices[i], || kernels[i].ghost_fill(stores[i]));
    }

    let mut t = start_time;
    let mut iter: u64 = 0;
    let mut last_cb: u64 = 0;
    while t < t_final {
        // global dt = min over tiles' cfl, clamped so the last step lands exactly on t_final.
        let mut dt = t_final - t;
        for i in 0..n {
            dt = dt.min(symbi_xpu::with_device(devices[i], || kernels[i].cfl(stores[i])));
        }
        // snapshot u_n once before the stages for multi-stage schemes (the corrector reads it).
        if multistage {
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || kernels[i].snapshot(stores[i]));
            }
        }
        for (sidx, &(a0, ac)) in stages.iter().enumerate() {
            let stage = (sidx + 1) as u8; // 1-based: post_godunov saves the emf at stage 1, averages at 2.
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || {
                    // the full per-stage pipeline (evolve.rs STAGE_PIPELINE). wave_speeds / efield
                    // / post_godunov are the MHD constrained-transport hooks; they are no-op
                    // defaults for hydro + iso, so this is byte-identical to the prior sequence
                    // there, and drives the CT curl (edge emf -> bface -> bcell) for mhd. source /
                    // body phases are gated off (multi-gpu v1 has neither).
                    kernels[i].wave_speeds(stores[i]);
                    for dd in 0..D {
                        kernels[i].flux(stores[i], dd);
                    }
                    kernels[i].efield(stores[i]);
                    kernels[i].godunov_stage(stores[i], dt, a0, ac);
                    kernels[i].post_godunov(stores[i], dt, stage);
                    kernels[i].c2p(stores[i]);
                    kernels[i].ghost_fill(stores[i]);
                });
            }
            // refresh the cut halos from each neighbor's stage-updated interior.
            drain_devices::<M>(devices);
            exchange_grid(stores, counts, devices, transport);
            // re-fill physical boundary ghosts post-exchange (cut-corner consistency, see prime).
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || kernels[i].ghost_fill(stores[i]));
            }
        }
        t += dt;
        iter += 1;
        if iter - last_cb >= interval {
            last_cb = iter;
            drain_devices::<M>(devices);
            if on_checkpoint(iter, t).is_break() {
                return;
            }
        }
    }
    drain_devices::<M>(devices);
    let _ = on_checkpoint(iter, t);
}

// the gather/scatter kernels for `StagedCopy`: pack a strided strip into a contiguous
// buffer, and scatter a contiguous buffer back into a strided strip. the contiguous buffer
// is the interchange a peer-copy (nvlink) or mpi transfer moves across the link.
#[cfg(feature = "gpu")]
const HALO_GATHER_KERNEL: &str = r#"
extern "C" __global__ void halo_gather(
    const double* src, double* buf, const unsigned int* idx, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = src[idx[i]];
}
"#;

#[cfg(feature = "gpu")]
const HALO_SCATTER_KERNEL: &str = r#"
extern "C" __global__ void halo_scatter(
    double* dst, const double* buf, const unsigned int* idx, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[idx[i]] = buf[i];
}
"#;

// pooled staging buffers: the gather/scatter index arrays plus the contiguous f64 strip.
#[cfg(feature = "gpu")]
struct HaloStageBufs {
    sidx: MemoryBlock<DeviceMemory>,
    didx: MemoryBlock<DeviceMemory>,
    buf: MemoryBlock<DeviceMemory>,
    cap: usize,
}

#[cfg(feature = "gpu")]
thread_local! {
    static HALO_STAGE_BUFS: std::cell::RefCell<Option<HaloStageBufs>> =
        const { std::cell::RefCell::new(None) };
}

/// staged device transport: gather the strided strip into a CONTIGUOUS device buffer, then
/// scatter it into the destination. on a single device the buffer is the staging area
/// (gather then scatter, no move). ACROSS devices the contiguous buffer is exactly what a
/// `cuMemcpyPeer` over nvlink (intra-node) or an mpi send/recv (multi-node) moves between
/// the two ranks -- so this validates the pack/unpack halves of the multi-gpu transport on
/// a single card; only the move in the middle is added on real hardware. host fallback for
/// a cpu backend.
#[cfg(feature = "gpu")]
pub struct StagedCopy;

#[cfg(feature = "gpu")]
impl HaloTransport for StagedCopy {
    fn copy_region<const D: usize, M: MemorySpace>(
        &self,
        src: &Field<f64, D, M>,
        src_region: &Domain<D>,
        dst: &Field<f64, D, M>,
        dst_region: &Domain<D>,
        _src_dev: i32,
        _dst_dev: i32,
    ) {
        if !M::IS_DEVICE_ACCESSIBLE {
            LocalCopy.copy_region(src, src_region, dst, dst_region, _src_dev, _dst_dev);
            return;
        }
        let n = src_region.volume();
        debug_assert_eq!(n, dst_region.volume(), "src/dst strip shapes differ");
        if n == 0 {
            return;
        }
        let sdom = src.domain();
        let ddom = dst.domain();
        let src_ptr = src.as_ptr() as u64;
        let dst_ptr = dst.as_mut_ptr() as u64;
        let n_u32 = n as u32;
        let config = LaunchConfig::for_1d(n_u32, 64);

        HALO_STAGE_BUFS.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().map_or(true, |b| b.cap < n) {
                *slot = Some(HaloStageBufs {
                    sidx: MemoryBlock::<DeviceMemory>::for_elements::<u32>(n).expect("sidx alloc"),
                    didx: MemoryBlock::<DeviceMemory>::for_elements::<u32>(n).expect("didx alloc"),
                    buf: MemoryBlock::<DeviceMemory>::for_elements::<f64>(n).expect("stage alloc"),
                    cap: n,
                });
            }
            let bufs = slot.as_mut().unwrap();
            let sp = bufs.sidx.as_mut_ptr::<u32>();
            let dp = bufs.didx.as_mut_ptr::<u32>();
            for (i, (sc, dc)) in src_region.iter().zip(dst_region.iter()).enumerate() {
                unsafe {
                    *sp.add(i) = sdom.flat_index(sc) as u32;
                    *dp.add(i) = ddom.flat_index(dc) as u32;
                }
            }
            let sidx_ptr = bufs.sidx.as_ptr::<u32>() as u64;
            let didx_ptr = bufs.didx.as_ptr::<u32>() as u64;
            let buf_ptr = bufs.buf.as_mut_ptr::<f64>() as u64;

            // gather: buf[i] = src[sidx[i]] -- pack the strided strip into the contiguous buffer.
            let gather =
                current_dispatcher().jit_kernel_keyed(HALO_GATHER_KERNEL, "decomp/halo_gather", "halo_gather");
            let mut g = KernelArgs::new();
            g.push(&src_ptr);
            g.push(&buf_ptr);
            g.push(&sidx_ptr);
            g.push(&n_u32);
            unsafe {
                current_dispatcher()
                    .runtime()
                    .launch(&gather, config, g.as_mut_slice())
                    .expect("halo_gather launch failed");
            }

            // MOVE: single device -- the buffer IS the staging, nothing to move. across devices
            // this is where cuMemcpyPeer (nvlink) or mpi send/recv copies `buf` to the neighbor's
            // buffer before the scatter runs there.

            // scatter: dst[didx[i]] = buf[i].
            let scatter = current_dispatcher().jit_kernel_keyed(
                HALO_SCATTER_KERNEL,
                "decomp/halo_scatter",
                "halo_scatter",
            );
            let mut s = KernelArgs::new();
            s.push(&dst_ptr);
            s.push(&buf_ptr);
            s.push(&didx_ptr);
            s.push(&n_u32);
            unsafe {
                current_dispatcher()
                    .runtime()
                    .launch(&scatter, config, s.as_mut_slice())
                    .expect("halo_scatter launch failed");
            }
            // drain before the pooled buffers are reused next call.
            ctx_sync();
        });
    }
}

// pooled staging buffers for `PeerCopy`, one set PER logical device: the index array (a strip
// is gathered/scattered on its own device) and the contiguous f64 buffer the peer move
// transfers. allocated in the owning device's context so each buffer is resident on its gpu.
#[cfg(feature = "gpu")]
struct PeerBufs {
    idx: MemoryBlock<DeviceMemory>,
    buf: MemoryBlock<DeviceMemory>,
    cap: usize,
}

#[cfg(feature = "gpu")]
thread_local! {
    static PEER_BUFS: std::cell::RefCell<[Option<PeerBufs>; MAX_GPUS]> =
        const { std::cell::RefCell::new([const { None }; MAX_GPUS]) };
}

// ensure device `dev` has staging buffers of at least `n` elements, allocated in its context.
#[cfg(feature = "gpu")]
fn ensure_peer_bufs(pool: &mut [Option<PeerBufs>; MAX_GPUS], dev: i32, n: usize) {
    let slot = &mut pool[dev as usize];
    if slot.as_ref().map_or(true, |b| b.cap < n) {
        let (idx, buf) = with_device(dev, || {
            (
                MemoryBlock::<DeviceMemory>::for_elements::<u32>(n).expect("peer idx alloc"),
                MemoryBlock::<DeviceMemory>::for_elements::<f64>(n).expect("peer buf alloc"),
            )
        });
        *slot = Some(PeerBufs { idx, buf, cap: n });
    }
}

// cached `can_access_peer(src, dst)` results: -1 unknown, 0 no, 1 yes. the driver query is
// cheap but called per face per field per step, so memoize it (device topology is fixed for a
// run). thread_local to match the pooled buffers (no Send/Sync on the cache).
#[cfg(feature = "gpu")]
thread_local! {
    static PEER_CAP: std::cell::RefCell<[[i8; MAX_GPUS]; MAX_GPUS]> =
        const { std::cell::RefCell::new([[-1i8; MAX_GPUS]; MAX_GPUS]) };
}

/// can logical device `src` DIRECTLY peer-access `dst`? memoized. false when they fold onto the
/// same physical card (a device cannot peer with itself) or there is no p2p link -- in which
/// case `PeerCopy` stages over managed memory instead. this is the single switch that makes one
/// transport correct on every machine.
#[cfg(feature = "gpu")]
fn peer_ok(src: i32, dst: i32) -> bool {
    PEER_CAP.with(|c| {
        let mut cap = c.borrow_mut();
        let cached = cap[src as usize][dst as usize];
        if cached >= 0 {
            return cached == 1;
        }
        let ok = can_access_peer(src, dst).unwrap_or(false);
        cap[src as usize][dst as usize] = ok as i8;
        ok
    })
}

/// enable bidirectional peer access for every directly-peerable pair among `devices`, once at
/// setup. pairs that cannot peer (same card / no link) are skipped -- `PeerCopy` stages those.
/// idempotent and best-effort: a failure to enable just means that pair stages instead of peers.
#[cfg(feature = "gpu")]
pub fn enable_peer_mesh(devices: &[i32]) {
    let mut uniq: Vec<i32> = Vec::new();
    for &d in devices {
        if !uniq.contains(&d) {
            uniq.push(d);
        }
    }
    for &a in &uniq {
        for &b in &uniq {
            if a != b && peer_ok(a, b) {
                let _ = symbi_xpu::enable_peer_access(a, b);
            }
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub fn enable_peer_mesh(_devices: &[i32]) {}

/// the cross-device halo transport (docs/design/37 M3): gather the strip into the source
/// device's contiguous buffer, `cuMemcpyPeer` it to the destination device's buffer over the
/// link (nvlink intra-node), then scatter it into the destination strip. the gather and
/// scatter ARE `StagedCopy`'s proven halves; only the peer move in the middle is new. when
/// `src_dev == dst_dev` there is nothing to move across, so it defers to the proven
/// single-device `StagedCopy`. host fallback for a cpu backend. NOT exercisable on one gpu (a
/// device cannot peer with itself); the equivalence oracle runs it on a real multi-gpu node.
#[cfg(feature = "gpu")]
pub struct PeerCopy;

#[cfg(feature = "gpu")]
impl HaloTransport for PeerCopy {
    fn copy_region<const D: usize, M: MemorySpace>(
        &self,
        src: &Field<f64, D, M>,
        src_region: &Domain<D>,
        dst: &Field<f64, D, M>,
        dst_region: &Domain<D>,
        src_dev: i32,
        dst_dev: i32,
    ) {
        if !M::IS_DEVICE_ACCESSIBLE {
            LocalCopy.copy_region(src, src_region, dst, dst_region, src_dev, dst_dev);
            return;
        }
        // fall back to the staged copy (contiguous gather/scatter over managed memory, correct
        // for ANY device pair) unless the two logical devices can DIRECTLY peer. this is the
        // universal-transport invariant: the SAME `PeerCopy` works on one card (logical devices
        // fold onto the same physical gpu -> no peer -> staged), on a node with nvlink (real peer
        // -> cuMemcpyPeer fast path), and on a node without p2p (staged) -- for any gpu count, no
        // machine-specific code. only a genuine cross-device link takes the peer path below.
        if src_dev == dst_dev || !peer_ok(src_dev, dst_dev) {
            StagedCopy.copy_region(src, src_region, dst, dst_region, src_dev, dst_dev);
            return;
        }

        let n = src_region.volume();
        debug_assert_eq!(n, dst_region.volume(), "src/dst strip shapes differ");
        if n == 0 {
            return;
        }
        let sdom = src.domain();
        let ddom = dst.domain();
        let src_ptr = src.as_ptr() as u64;
        let dst_ptr = dst.as_mut_ptr() as u64;
        let n_u32 = n as u32;
        let config = LaunchConfig::for_1d(n_u32, 64);

        PEER_BUFS.with(|cell| {
            // stage 0: ensure both devices have buffers, write the per-cell flat offsets into
            // each device's index buffer (host writes to managed memory), then take the raw
            // device pointers so the borrow ends before any launch.
            let (src_idx_ptr, src_buf_ptr, dst_idx_ptr, dst_buf_ptr) = {
                let mut pool = cell.borrow_mut();
                ensure_peer_bufs(&mut pool, src_dev, n);
                ensure_peer_bufs(&mut pool, dst_dev, n);

                let sp = pool[src_dev as usize].as_mut().unwrap().idx.as_mut_ptr::<u32>();
                for (i, sc) in src_region.iter().enumerate() {
                    unsafe { *sp.add(i) = sdom.flat_index(sc) as u32 };
                }
                let dp = pool[dst_dev as usize].as_mut().unwrap().idx.as_mut_ptr::<u32>();
                for (i, dc) in dst_region.iter().enumerate() {
                    unsafe { *dp.add(i) = ddom.flat_index(dc) as u32 };
                }

                let s = pool[src_dev as usize].as_mut().unwrap();
                let src_idx_ptr = s.idx.as_ptr::<u32>() as u64;
                let src_buf_ptr = s.buf.as_mut_ptr::<f64>() as u64;
                let d = pool[dst_dev as usize].as_mut().unwrap();
                let dst_idx_ptr = d.idx.as_ptr::<u32>() as u64;
                let dst_buf_ptr = d.buf.as_mut_ptr::<f64>() as u64;
                (src_idx_ptr, src_buf_ptr, dst_idx_ptr, dst_buf_ptr)
            };

            // stage 1: gather on the source device -- src_buf[i] = src[sidx[i]].
            with_device(src_dev, || {
                let gather = current_dispatcher().jit_kernel_keyed(
                    HALO_GATHER_KERNEL,
                    "decomp/halo_gather",
                    "halo_gather",
                );
                let mut g = KernelArgs::new();
                g.push(&src_ptr);
                g.push(&src_buf_ptr);
                g.push(&src_idx_ptr);
                g.push(&n_u32);
                unsafe {
                    current_dispatcher()
                        .runtime()
                        .launch(&gather, config, g.as_mut_slice())
                        .expect("peer gather launch failed");
                }
                ctx_sync();
            });

            // stage 2: move the contiguous buffer across the link (synchronous).
            memcpy_peer(
                dst_buf_ptr,
                dst_dev,
                src_buf_ptr,
                src_dev,
                n * std::mem::size_of::<f64>(),
            )
            .expect("cuMemcpyPeer failed");

            // stage 3: scatter on the destination device -- dst[didx[i]] = dst_buf[i].
            with_device(dst_dev, || {
                let scatter = current_dispatcher().jit_kernel_keyed(
                    HALO_SCATTER_KERNEL,
                    "decomp/halo_scatter",
                    "halo_scatter",
                );
                let mut s = KernelArgs::new();
                s.push(&dst_ptr);
                s.push(&dst_buf_ptr);
                s.push(&dst_idx_ptr);
                s.push(&n_u32);
                unsafe {
                    current_dispatcher()
                        .runtime()
                        .launch(&scatter, config, s.as_mut_slice())
                        .expect("peer scatter launch failed");
                }
                ctx_sync();
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::decompose_grid;

    #[test]
    fn decompose_one_part_is_monolithic() {
        assert_eq!(decompose_grid([64usize], 1).unwrap(), [1]);
        assert_eq!(decompose_grid([64usize, 32], 1).unwrap(), [1, 1]);
    }

    #[test]
    fn decompose_cuts_longest_axis_first() {
        assert_eq!(decompose_grid([64usize], 2).unwrap(), [2]);
        assert_eq!(decompose_grid([64usize, 64], 2).unwrap(), [2, 1]);
        assert_eq!(decompose_grid([64usize, 64], 4).unwrap(), [2, 2]);
        assert_eq!(decompose_grid([64usize, 64, 64], 8).unwrap(), [2, 2, 2]);
        // products always equal the part count.
        for parts in [2usize, 4, 8, 16] {
            let c = decompose_grid([64usize, 64, 64], parts).unwrap();
            assert_eq!(c.iter().product::<usize>(), parts);
        }
    }

    #[test]
    fn decompose_respects_divisibility() {
        // 48 = 16*3, evenly splittable into 16 along one axis.
        assert_eq!(decompose_grid([48usize], 16).unwrap(), [16]);
        // every per-axis count must divide its axis: each tile gets whole cells.
        let c = decompose_grid([96usize, 48], 12).unwrap();
        assert_eq!(c.iter().product::<usize>(), 12);
        assert_eq!(96 % c[0], 0);
        assert_eq!(48 % c[1], 0);
    }

    #[test]
    fn decompose_errors_when_indivisible() {
        // 3 divides neither 100-cell axis.
        assert!(decompose_grid([100usize, 100], 3).is_err());
        // 16 tiles cannot fit on an 8-cell axis.
        assert!(decompose_grid([8usize], 16).is_err());
    }
}
