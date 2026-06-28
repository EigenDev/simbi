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

use crate::state::{FieldStore, PartitionGeometry};
use symbi_algebra::{Domain, Side};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;
#[cfg(feature = "cuda")]
use symbi_xpu::cuda::{ctx_sync, memcpy_peer, UnifiedMemory, MAX_GPUS};
#[cfg(feature = "cuda")]
use symbi_xpu::runtime::{cuda_runtime::current_dispatcher, GpuRuntime};
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
struct HaloIdxBufs {
    sidx: MemoryBlock<UnifiedMemory>,
    didx: MemoryBlock<UnifiedMemory>,
    cap: usize,
}

#[cfg(feature = "cuda")]
thread_local! {
    static HALO_IDX_BUFS: std::cell::RefCell<Option<HaloIdxBufs>> =
        const { std::cell::RefCell::new(None) };
}

/// device-side transport: the strip copy runs as a gpu kernel, with no host roundtrip of
/// the field data. the per-cell flat-offset arrays are precomputed on the host into pooled
/// unified buffers; the data itself never leaves the device. on a host backend it falls
/// back to the proven cell-for-cell copy.
#[cfg(feature = "cuda")]
pub struct DeviceCopy;

#[cfg(feature = "cuda")]
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
                    sidx: MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("sidx alloc"),
                    didx: MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("didx alloc"),
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

/// the prim components the flux stage reconstructs from: rho, each velocity, pressure.
fn prim_fields<const D: usize, const DOF: usize, M: MemorySpace>(
    store: &FieldStore<D, DOF, M>,
) -> Vec<&Field<f64, D, M>> {
    std::iter::once(&store.fields.prim.rho)
        .chain(store.fields.prim.vel.iter())
        .chain(store.fields.prim.pre.as_ref())
        .collect()
}

/// exchange same-level halos across the shared face between `lo` (its hi face on `axis`)
/// and `hi` (its lo face). `processed` marks the axes already exchanged this pass (for
/// the two-pass transverse rule); `counts` is the per-axis tile count. fills only the
/// prim ghost cells -- cons ghosts are never read by the flux stage.
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

// the gather/scatter kernels for `StagedCopy`: pack a strided strip into a contiguous
// buffer, and scatter a contiguous buffer back into a strided strip. the contiguous buffer
// is the interchange a peer-copy (nvlink) or mpi transfer moves across the link.
#[cfg(feature = "cuda")]
const HALO_GATHER_KERNEL: &str = r#"
extern "C" __global__ void halo_gather(
    const double* src, double* buf, const unsigned int* idx, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = src[idx[i]];
}
"#;

#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
struct HaloStageBufs {
    sidx: MemoryBlock<UnifiedMemory>,
    didx: MemoryBlock<UnifiedMemory>,
    buf: MemoryBlock<UnifiedMemory>,
    cap: usize,
}

#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
pub struct StagedCopy;

#[cfg(feature = "cuda")]
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
                    sidx: MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("sidx alloc"),
                    didx: MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("didx alloc"),
                    buf: MemoryBlock::<UnifiedMemory>::for_elements::<f64>(n).expect("stage alloc"),
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
#[cfg(feature = "cuda")]
struct PeerBufs {
    idx: MemoryBlock<UnifiedMemory>,
    buf: MemoryBlock<UnifiedMemory>,
    cap: usize,
}

#[cfg(feature = "cuda")]
thread_local! {
    static PEER_BUFS: std::cell::RefCell<[Option<PeerBufs>; MAX_GPUS]> =
        const { std::cell::RefCell::new([const { None }; MAX_GPUS]) };
}

// ensure device `dev` has staging buffers of at least `n` elements, allocated in its context.
#[cfg(feature = "cuda")]
fn ensure_peer_bufs(pool: &mut [Option<PeerBufs>; MAX_GPUS], dev: i32, n: usize) {
    let slot = &mut pool[dev as usize];
    if slot.as_ref().map_or(true, |b| b.cap < n) {
        let (idx, buf) = with_device(dev, || {
            (
                MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("peer idx alloc"),
                MemoryBlock::<UnifiedMemory>::for_elements::<f64>(n).expect("peer buf alloc"),
            )
        });
        *slot = Some(PeerBufs { idx, buf, cap: n });
    }
}

/// the cross-device halo transport (docs/design/37 M3): gather the strip into the source
/// device's contiguous buffer, `cuMemcpyPeer` it to the destination device's buffer over the
/// link (nvlink intra-node), then scatter it into the destination strip. the gather and
/// scatter ARE `StagedCopy`'s proven halves; only the peer move in the middle is new. when
/// `src_dev == dst_dev` there is nothing to move across, so it defers to the proven
/// single-device `StagedCopy`. host fallback for a cpu backend. NOT exercisable on one gpu (a
/// device cannot peer with itself); the equivalence oracle runs it on a real multi-gpu node.
#[cfg(feature = "cuda")]
pub struct PeerCopy;

#[cfg(feature = "cuda")]
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
        // same device: no link to cross -- the proven staged gather/scatter handles it.
        if src_dev == dst_dev {
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
