// =============================================================================
// decomp.rs
//
// same-level domain decomposition: halo exchange between neighboring subdomains on a
// grid, behind a transport interface.
//
// the exchange LOGIC (which cells move where) is fixed: a decomposed run reproduces the
// monolithic one. the TRANSPORT (how the bytes move) varies
// behind `HaloTransport`: a local memory copy, a gpu peer copy, or an mpi
// pack/send/unpack. swapping the transport never touches the decomposition.
//
// the exchange is a two-pass scheme: the caller processes axes in order, and a cut
// face's transverse extent is the interior for cut axes not yet exchanged, the full
// allocated extent otherwise. that carries corner ghosts to the diagonal neighbor
// without any explicit diagonal traffic.
//
// everything is built on `Domain` (slab / boundary / iter): the strips are domains and
// the copy is a cell-for-cell walk of two equal-shape domains, so the index arithmetic
// lives in one tested place.
// =============================================================================

use crate::driver::{advance_state_clock, book_horizon_receipt, horizon_request, prof};
use crate::state::{FieldStore, PartitionGeometry, Timestepping};
use crate::substrate_seam::KernelSet;
use std::ops::ControlFlow;
use symbi_algebra::{Domain, Side};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;
#[cfg(feature = "gpu")]
use symbi_xpu::runtime::{GpuRuntime, current_dispatcher};
#[cfg(feature = "gpu")]
use symbi_xpu::{DeviceMemory, MAX_GPUS, can_access_peer, ctx_sync, memcpy_peer};
#[cfg(feature = "gpu")]
use symbi_xpu::{KernelArgs, LaunchConfig, MemoryBlock, with_device};

/// move `src` over `src_region` into `dst` over `dst_region`, cell-for-cell. the two
/// regions have identical shape (same per-axis extents); the transport pairs them by
/// iteration order. `src_dev`/`dst_dev` are the LOGICAL devices the two fields live on --
/// the one piece of device identity a cross-device exchange genuinely needs (the rest of the
/// code uses the ambient current-device model). single-device transports
/// (`LocalCopy`, `DeviceCopy`, `StagedCopy`) ignore them; `PeerCopy` uses them to drive
/// a device-to-device peer copy between the two devices. the transport is BACKEND-GENERIC —
/// `symbi_xpu::memcpy_peer` resolves to `cuMemcpyPeer` on nvidia and `hipMemcpyPeer` on amd — and it
/// is correct whether or not peer access could be enabled for the pair: without it the driver stages
/// through host memory, which costs bandwidth but never correctness.
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
// strip geometry is fixed across steps, so this reuses the same device allocations and
// avoids a `cuMemAllocManaged` per `copy_region` (the dominant per-exchange cost). thread-local
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

            let kernel = current_dispatcher().jit_kernel_keyed(
                HALO_COPY_KERNEL,
                "decomp/halo_copy",
                "halo_copy",
            );
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
/// plus the primitives (rho, velocity, pressure), and -- for MHD -- the cell-centered B `bcell`
/// (the checkpoint writer reads it for both the `mag` conserved slot and the `bcell` primitive).
/// used by `gather_interiors` to reassemble a decomposed run into one global store for output --
/// the gathered global then writes through the existing single-grid checkpoint path unchanged.
/// the STAGGERED `bface` lives on a face domain, so it is gathered separately
/// by `gather_faces`. the cell-centered set is DERIVED from the store's populated slots
/// (`ConsFields::exchange_fields` + `PrimFields::exchange_fields`), so an optional field like the
/// passive scalar rides along automatically and no hand-listed set can drop it. hydro/iso stores
/// have no `mhd`, so the bcell tail is empty there; global and tiles share a regime and dye opt-in,
/// so both lists align component-for-component.
fn data_fields<const D: usize, const DOF: usize, M: MemorySpace>(
    store: &FieldStore<D, DOF, M>,
) -> Vec<&Field<f64, D, M>> {
    let mut fields = store.fields.cons.exchange_fields();
    fields.extend(store.fields.prim.exchange_fields());
    if let Some(mhd) = store.fields.mhd.as_ref() {
        fields.extend(mhd.bcell.b.iter());
    }
    fields
}

/// reassemble a decomposed run into one full-size `global` store: copy each tile's INTERIOR
/// (cons + prim) into the matching sub-box of `global`. the inverse of the
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

/// gather the STAGGERED face-normal B `bface[d]` from every tile into `global` (MHD only). the
/// cell gather (`gather_interiors`) cannot carry `bface`: each axis-`d` face field lives on the
/// interior face domain (cells extended +1 on `d`). for each tile and axis,
/// copy the tile's interior face box into the matching sub-box of the global face domain. the
/// shared internal face (a tile's hi-`d` face == its neighbor's lo-`d` face) is written by both
/// neighbors from CT-consistent (bit-identical) values, so the overwrite is harmless. no-op when
/// the store carries no `mhd` fields (hydro/iso), so the shared run loop calls it unconditionally.
/// reassemble the decomposed tracer population into `global.tracers` for output: union every
/// tile's set, sorted by id, so the checkpoint's tracer order is identical to a single-grid run
/// (ids are the stable identity; tile membership is a decomposition detail). no-op when the run
/// carries no tracers.
pub fn gather_tracers<const D: usize, const DOF: usize, M: MemorySpace>(
    global: &mut FieldStore<D, DOF, M, f64>,
    tiles: &[&FieldStore<D, DOF, M>],
) {
    let Some(g) = global.tracers.as_mut() else {
        return;
    };
    g.x.clear();
    g.id.clear();
    g.cohort.clear();
    g.flags.clear();
    g.owner.clear();
    g.step_owner.clear();
    g.step_flags.clear();
    g.next_id = 0;
    g.injection_remainder = 0.0;
    for t in tiles {
        if let Some(tr) = t.tracers.as_ref() {
            g.x.extend_from_slice(&tr.x);
            g.id.extend_from_slice(&tr.id);
            g.cohort.extend_from_slice(&tr.cohort);
            g.flags.extend_from_slice(&tr.flags);
            g.owner.extend_from_slice(&tr.owner);
            g.step_owner.extend_from_slice(&tr.step_owner);
            g.step_flags.extend_from_slice(&tr.step_flags);
            g.weight = tr.weight;
            g.run_seed = tr.run_seed;
            g.next_id = g.next_id.max(tr.next_id);
            g.injection_remainder += tr.injection_remainder;
        }
    }
    // stable id order: build the permutation, apply to each SoA column.
    let mut perm: Vec<usize> = (0..g.id.len()).collect();
    perm.sort_by_key(|&i| g.id[i]);
    g.x = perm.iter().map(|&i| g.x[i]).collect();
    g.id = perm.iter().map(|&i| g.id[i]).collect();
    g.cohort = perm.iter().map(|&i| g.cohort[i]).collect();
    g.flags = perm.iter().map(|&i| g.flags[i]).collect();
    g.owner = perm.iter().map(|&i| g.owner[i]).collect();
    g.step_owner = perm.iter().map(|&i| g.step_owner[i]).collect();
    g.step_flags = perm.iter().map(|&i| g.step_flags[i]).collect();
}

pub fn gather_faces<const D: usize, const DOF: usize, M: MemorySpace>(
    global: &FieldStore<D, DOF, M>,
    tiles: &[&FieldStore<D, DOF, M>],
    counts: [usize; D],
) {
    let Some(gmhd) = global.fields.mhd.as_ref() else {
        return;
    };
    let gint = &global.geom.interior;
    let m: [usize; D] = std::array::from_fn(|ax| gint.spaces[ax].size() / counts[ax]);
    let glo: [isize; D] = std::array::from_fn(|ax| gint.spaces[ax].lo);

    for (flat, tile) in tiles.iter().enumerate() {
        let Some(tmhd) = tile.fields.mhd.as_ref() else {
            continue;
        };
        let tc = unflatten(flat, counts);
        let tint = &tile.geom.interior;
        for d in 0..D {
            // the tile's interior face domain on axis d (interior extended +1 on d) -> the
            // matching sub-box of the global interior face domain (same extension).
            let src = tint.extend(d, 0, 1);
            let mut dst = gint.extend(d, 0, 1);
            for ax in 0..D {
                let lo = glo[ax] + (tc[ax] * m[ax]) as isize;
                let hi = lo + m[ax] as isize + if ax == d { 1 } else { 0 };
                dst = dst.slab(ax, (lo, hi));
            }
            LocalCopy.copy_region(&tmhd.bface[d], &src, &gmhd.bface[d], &dst, 0, 0);
        }
    }
}

/// the CELL-CENTERED components the flux stage reconstructs from, DERIVED from the store's
/// populated primitive slots (`PrimFields::exchange_fields`): rho, each velocity, pressure, the
/// passive scalar when present, and -- for MHD -- the cell-centered magnetic field `bcell` (the
/// reconstruction reads it like any other primitive). hydro/iso stores have no `mhd`, so the bcell
/// tail is empty there and the exchange is unchanged. the STAGGERED `bface` is exchanged separately
/// (`bface_strips`), since its faces live on a different domain than the cells.
fn prim_fields<const D: usize, const DOF: usize, M: MemorySpace>(
    store: &FieldStore<D, DOF, M>,
) -> Vec<&Field<f64, D, M>> {
    let mut fields = store.fields.prim.exchange_fields();
    if let Some(mhd) = store.fields.mhd.as_ref() {
        fields.extend(mhd.bcell.b.iter());
    }
    fields
}

/// the ghost strip on a STAGGERED face field's own allocated domain (for MHD `bface`). mirrors
/// `ghost_strip` but takes the field's own `alloc` domain, and
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
            strip = strip.slab(
                b,
                (
                    geom.interior.spaces[b].lo,
                    geom.interior.spaces[b].hi + hi_ext,
                ),
            );
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
    // halos + prim + bcell -- so it stays bit-identical and div(B) is preserved.
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
            let lo_ghost_f =
                face_ghost_strip(&lo_alloc, &lo.geom, axis, Side::Hi, d, processed, counts);
            let hi_src_f = lo_ghost_f.slab(axis, (i_lo_hi, i_lo_hi + ng));
            let hi_ghost_f =
                face_ghost_strip(&hi_alloc, &hi.geom, axis, Side::Lo, d, processed, counts);
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
                ));
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
/// real spmd driver keeps it); the field carries none -- the exchange is the one cross-device step.
pub fn exchange_grid<const D: usize, const DOF: usize, M: MemorySpace, T: HaloTransport>(
    tiles: &[&FieldStore<D, DOF, M>],
    counts: [usize; D],
    devices: &[i32],
    transport: &T,
) {
    let total: usize = counts.iter().product();
    debug_assert_eq!(
        tiles.len(),
        total,
        "tiles slice length does not match counts"
    );
    debug_assert_eq!(
        devices.len(),
        total,
        "devices slice length does not match counts"
    );
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
            exchange_faces(
                lo,
                hi,
                axis,
                &processed,
                &counts,
                devices[lo_flat],
                devices[hi_flat],
                transport,
            );
        }
        processed[axis] = true;
    }
}

fn exchange_ito_coefficients<const D: usize, const DOF: usize, M: MemorySpace, T: HaloTransport>(
    tiles: &[&FieldStore<D, DOF, M>],
    counts: [usize; D],
    devices: &[i32],
    transport: &T,
) {
    let total: usize = counts.iter().product();
    let mut processed = [false; D];
    for axis in 0..D {
        for flat in 0..total {
            let tc = unflatten(flat, counts);
            if tc[axis] + 1 >= counts[axis] {
                continue;
            }
            let mut tc_hi = tc;
            tc_hi[axis] += 1;
            let lo_flat = flatten(tc, counts);
            let hi_flat = flatten(tc_hi, counts);
            let lo = tiles[lo_flat];
            let hi = tiles[hi_flat];
            let lo_fields = lo
                .ito_coefficients
                .as_ref()
                .expect("continuous-tracer tile carries ito coefficients");
            let hi_fields = hi
                .ito_coefficients
                .as_ref()
                .expect("continuous-tracer tile carries ito coefficients");
            let ng = lo.geom.ng as isize;
            let i_hi_lo = lo.geom.interior.spaces[axis].hi;
            let i_lo_hi = hi.geom.interior.spaces[axis].lo;
            let lo_ghost = ghost_strip(&lo.geom, axis, Side::Hi, &processed, &counts);
            let hi_src = lo_ghost.slab(axis, (i_lo_hi, i_lo_hi + ng));
            let hi_ghost = ghost_strip(&hi.geom, axis, Side::Lo, &processed, &counts);
            let lo_src = hi_ghost.slab(axis, (i_hi_lo - ng, i_hi_lo));
            for dd in 0..D {
                for (fl, fr) in [
                    (&lo_fields.drift[dd], &hi_fields.drift[dd]),
                    (&lo_fields.variance[dd], &hi_fields.variance[dd]),
                    (&lo_fields.third[dd], &hi_fields.third[dd]),
                ] {
                    transport.copy_region(
                        fr,
                        &hi_src,
                        fl,
                        &lo_ghost,
                        devices[hi_flat],
                        devices[lo_flat],
                    );
                    transport.copy_region(
                        fl,
                        &lo_src,
                        fr,
                        &hi_ghost,
                        devices[lo_flat],
                        devices[hi_flat],
                    );
                }
            }
        }
        processed[axis] = true;
    }
}

// drain every UNIQUE tile device's context so async writes are visible to a consumer running
// in another context (the cross-device read barrier). no-op on a host backend. mirrors the
// equivalence test's `sync_devices`.
#[cfg(feature = "gpu")]
pub fn drain_devices<M: MemorySpace>(devices: &[i32]) {
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
pub fn drain_devices<M: MemorySpace>(_devices: &[i32]) {}

/// drive a decomposed simulation: `stores.len()` tiles evolved in LOCKSTEP at a shared dt,
/// with a same-level halo exchange after each ssp stage. the multi-gpu python entry and the
/// decomposition-equivalence checks drive this same loop, so there is ONE tested path.
///
/// - `stores[i]` / `kernels[i]`: tile i's field store + kernel set, in flat tile order
///   (matching `flatten(tile_coord, counts)`).
/// - `counts`: the per-axis tile grid; `devices[i]`: tile i's logical device (device identity
///   lives in the decomposition).
/// - `transport`: moves halos (`LocalCopy` host, `DeviceCopy`/`StagedCopy` single-gpu,
///   `PeerCopy` cross-gpu).
/// - `on_checkpoint(iteration, time)`: fires every `interval` steps (and once at the end);
///   returning `Break` stops the run. the device queue is drained before each call so the
///   callback can read coherent tile state for output.
///
/// the global dt is the min over all tiles' cfl -- identical to a monolithic run, since the
/// union of the tile interiors is the monolithic interior. rk2 exchanges BETWEEN stages so the
/// corrector reconstructs from each neighbor's stage-updated interior.
/// the per-step immersed-body bookkeeping for a DECOMPOSED run. each tile's `body_feedback` has
/// already reduced its LOCAL interior force/torque/accreted-mass into its own (interior-mutable)
/// diagnostics; this SUMS those partials across tiles into a global per-body delta (the true
/// disk-on-body reaction, == the monolithic single-grid reduction since the tile interiors
/// partition the global interior), then applies the IDENTICAL global delta to EVERY tile's bodies +
/// advances the prescribed binary orbit identically + resets each accumulator. identical input ->
/// identical body state, so all tiles stay in lockstep and the next step's body_source reads the
/// same body positions everywhere. no-op for tiles without bodies.
fn step_bodies_decomposed<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    dt: f64,
) {
    use symbi_ib::{BodyDelta, apply_body_deltas};
    let mut global: Vec<BodyDelta<f64, D>> = Vec::new();
    for s in stores.iter() {
        let Some(im) = s.immersed.as_ref() else {
            continue;
        };
        for d in im.diagnostics.consolidate() {
            match global.iter_mut().find(|g| g.idx == d.idx) {
                Some(g) => {
                    g.force_delta = g.force_delta + d.force_delta;
                    g.torque_delta = g.torque_delta + d.torque_delta;
                    g.mass_delta += d.mass_delta;
                }
                None => global.push(d),
            }
        }
    }
    for s in stores.iter_mut() {
        let Some(im) = s.immersed.as_mut() else {
            continue;
        };
        apply_body_deltas(&mut im.bodies, &global, dt);
        // bonded fragments: the cross-tile-summed fluid loads drive the DEM subcycle (bonds +
        // contact + mutual gravity + gas drag), replicated identically on every tile. the bodies
        // and the summed external loads are identical on every tile, so the fragment motion is
        // too -- the decomposed analog of the single-grid `evolve_bodies` fragment step. a fragment
        // straddling a cut gets its total fluid load from the reduction above, so the cluster moves
        // as one regardless of which tiles its pieces sit in.
        if let Some(sys) = im.fragment_physics.as_mut() {
            let n_src = im.bodies.source_count();
            let mut external = vec![symbi_ib::ExternalLoad::zero(); im.bodies.len()];
            for d in &global {
                if d.idx >= n_src && d.idx < external.len() {
                    external[d.idx].force = d.force_delta;
                    external[d.idx].torque = d.torque_delta;
                }
            }
            sys.advance(&mut im.bodies, dt, &external);
        }
        im.diagnostics.reset();
    }
}

fn migrate_mass_transport_tracers<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    counts: [usize; D],
) {
    let local_cells: [usize; D] =
        std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
    let global_cells: [usize; D] = std::array::from_fn(|dd| local_cells[dd] * counts[dd]);
    let destination = |owner: crate::mass_transport::ContainerId| {
        let mut linear = owner.0 as usize;
        let global: [usize; D] = std::array::from_fn(|dd| {
            let index = linear % global_cells[dd];
            linear /= global_cells[dd];
            index
        });
        let tile = std::array::from_fn(|dd| global[dd] / local_cells[dd]);
        flatten(tile, counts)
    };

    let mut moved = Vec::new();
    for (source, store) in stores.iter_mut().enumerate() {
        let Some(tracers) = store.tracers.as_mut() else {
            continue;
        };
        let mut ii = 0;
        while ii < tracers.len() {
            let target = if tracers.flags[ii].escaped {
                source
            } else {
                destination(tracers.owner[ii])
            };
            if target == source {
                ii += 1;
                continue;
            }
            moved.push((
                target,
                tracers.x.swap_remove(ii),
                tracers.id.swap_remove(ii),
                tracers.cohort.swap_remove(ii),
                tracers.flags.swap_remove(ii),
                tracers.owner.swap_remove(ii),
                tracers.step_owner.swap_remove(ii),
                tracers.step_flags.swap_remove(ii),
            ));
        }
    }
    for (target, x, id, cohort, flags, owner, step_owner, step_flags) in moved {
        let tracers = stores[target]
            .tracers
            .as_mut()
            .expect("every decomposed tile carries a tracer set");
        tracers.x.push(x);
        tracers.id.push(id);
        tracers.cohort.push(cohort);
        tracers.flags.push(flags);
        tracers.owner.push(owner);
        tracers.step_owner.push(step_owner);
        tracers.step_flags.push(step_flags);
    }
}

fn migrate_continuous_tracers<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    counts: [usize; D],
) -> Result<usize, String> {
    if !M::IS_HOST_ACCESSIBLE {
        return Err("continuous tracer migration requires host-accessible memory".to_string());
    }
    if stores.iter().any(|store| {
        store.geom.coords != symbi_geometry::Geometry::Cartesian || store.geom.maps.is_some()
    }) {
        return Err(
            "continuous tracer migration requires a uniform Cartesian decomposition".to_string(),
        );
    }
    let local_cells: [usize; D] =
        std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
    let dx = stores[0].geom.dx;
    let global_lo: [f64; D] = std::array::from_fn(|dd| {
        stores
            .iter()
            .map(|store| crate::tracers::partition_physical_bounds(&store.geom)[dd].0)
            .fold(f64::INFINITY, f64::min)
    });
    let global_hi: [f64; D] = std::array::from_fn(|dd| {
        stores
            .iter()
            .map(|store| crate::tracers::partition_physical_bounds(&store.geom)[dd].1)
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let metadata = stores
        .iter()
        .filter_map(|store| store.continuous_tracers.as_ref())
        .max_by_key(|set| usize::from(set.len > 0))
        .map(|set| {
            (
                set.order,
                set.weight,
                set.run_seed,
                set.next_id,
                set.injection_remainder,
            )
        })
        .ok_or_else(|| "continuous tracer population is missing".to_string())?;
    let mut moved = Vec::new();
    for (source, store) in stores.iter_mut().enumerate() {
        let Some(tracers) = store.continuous_tracers.as_mut() else {
            continue;
        };
        let mut ii = 0;
        while ii < tracers.len {
            let escaped = unsafe { *tracers.escaped.as_ptr::<u8>().add(ii) != 0 };
            let crossed_sink = unsafe { *tracers.crossed_sink.as_ptr::<u8>().add(ii) != 0 };
            if escaped || crossed_sink {
                ii += 1;
                continue;
            }
            let position: [f64; D] =
                unsafe { std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(ii)) };
            if (0..D).any(|dd| position[dd] < global_lo[dd] || position[dd] >= global_hi[dd]) {
                ii += 1;
                continue;
            }
            let global_cell: [usize; D] = std::array::from_fn(|dd| {
                ((position[dd] - global_lo[dd]) / dx[dd]).floor() as usize
            });
            let tile: [usize; D] = std::array::from_fn(|dd| global_cell[dd] / local_cells[dd]);
            let target = flatten(tile, counts);
            let mut linear = 0usize;
            let mut stride = 1usize;
            for dd in 0..D {
                linear += global_cell[dd] * stride;
                stride *= local_cells[dd] * counts[dd];
            }
            let owner = crate::mass_transport::ContainerId(linear as u64);
            if target == source {
                unsafe {
                    *tracers
                        .owner
                        .as_mut_ptr::<crate::mass_transport::ContainerId>()
                        .add(ii) = owner;
                }
                ii += 1;
            } else {
                let mut record = tracers.swap_remove_host(ii)?;
                record.owner = owner;
                moved.push((target, record));
            }
        }
    }
    let count = moved.len();
    for (target, record) in moved {
        if stores[target].continuous_tracers.is_none() {
            let mut set = crate::tracers::ContinuousTracerSet::allocate(0, metadata.0)?;
            set.weight = metadata.1;
            set.run_seed = metadata.2;
            set.next_id = metadata.3;
            set.injection_remainder = metadata.4;
            stores[target].continuous_tracers = Some(set);
        }
        let target_set = stores[target]
            .continuous_tracers
            .as_mut()
            .expect("continuous tracer target was initialized");
        if target_set.order != metadata.0 {
            return Err("continuous tracer order differs across decomposed tiles".to_string());
        }
        if target_set.len == 0 {
            target_set.weight = metadata.1;
            target_set.run_seed = metadata.2;
            target_set.next_id = metadata.3;
            target_set.injection_remainder = metadata.4;
        }
        target_set.push_host(record)?;
    }
    Ok(count)
}

fn blend_mass_transport_ancestry<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    candidate_weight: f64,
    stage: usize,
) {
    if candidate_weight == 1.0 {
        return;
    }
    let ids: Vec<u64> = stores
        .iter()
        .filter_map(|store| store.tracers.as_ref())
        .flat_map(|tracers| tracers.id.iter().copied())
        .collect();
    let first = stores
        .iter()
        .find_map(|store| store.tracers.as_ref())
        .expect("decomposed tracer population is present");
    let key = crate::mass_transport::SamplingKey {
        run_seed: first.run_seed,
        epoch: stores[0]
            .iteration
            .wrapping_mul(4)
            .wrapping_add(stage as u64),
    };
    let selections: std::collections::BTreeMap<u64, bool> =
        crate::mass_transport::sample_convex_blend(&ids, candidate_weight, key)
            .expect("valid ssp ancestry weight")
            .into_iter()
            .collect();
    for store in stores.iter_mut() {
        let Some(tracers) = store.tracers.as_mut() else {
            continue;
        };
        for ii in 0..tracers.len() {
            if !selections[&tracers.id[ii]] {
                tracers.owner[ii] = tracers.step_owner[ii];
                tracers.flags[ii] = tracers.step_flags[ii];
            }
        }
    }
}

fn spawn_decomposed_injection<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    counts: [usize; D],
    ledgers: Vec<std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>>,
) {
    let mut injections = std::collections::BTreeMap::new();
    for ledger in ledgers {
        for (destination, mass) in ledger {
            *injections.entry(destination).or_insert(0.0) += mass;
        }
    }
    if injections.is_empty() {
        return;
    }
    let local_cells: [usize; D] =
        std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
    let global_cells: [usize; D] = std::array::from_fn(|dd| local_cells[dd] * counts[dd]);
    let global_lo = stores[0].geom.x_lo;
    let dx = stores[0].geom.dx;
    let first = stores
        .iter()
        .find_map(|store| store.tracers.as_ref())
        .expect("decomposed tracer population is present");
    let mut spawned = crate::tracers::TracerSet {
        weight: first.weight,
        run_seed: first.run_seed,
        next_id: stores
            .iter()
            .filter_map(|store| store.tracers.as_ref())
            .map(|tracers| tracers.next_id)
            .max()
            .unwrap_or(0),
        injection_remainder: stores
            .iter()
            .filter_map(|store| store.tracers.as_ref())
            .map(|tracers| tracers.injection_remainder)
            .sum(),
        ..Default::default()
    };
    let key = crate::mass_transport::SamplingKey {
        run_seed: spawned.run_seed,
        epoch: stores[0].iteration | (1 << 62),
    };
    crate::tracers::spawn_injected_tracers(
        &mut spawned,
        injections
            .into_iter()
            .map(|(destination, mass)| crate::mass_transport::MassTransfer { destination, mass }),
        |owner| {
            let mut linear = owner.0 as usize;
            std::array::from_fn(|dd| {
                let index = linear % global_cells[dd];
                linear /= global_cells[dd];
                global_lo[dd] + (index as f64 + 0.5) * dx[dd]
            })
        },
        key,
    )
    .unwrap_or_else(|detail| panic!("decomposed tracer injection: {detail}"));

    for store in stores.iter_mut() {
        let tracers = store
            .tracers
            .as_mut()
            .expect("every decomposed tile carries tracers");
        tracers.next_id = spawned.next_id;
        tracers.injection_remainder = 0.0;
    }
    stores[0]
        .tracers
        .as_mut()
        .expect("tile zero carries tracers")
        .injection_remainder = spawned.injection_remainder;

    for ii in 0..spawned.len() {
        let mut linear = spawned.owner[ii].0 as usize;
        let global: [usize; D] = std::array::from_fn(|dd| {
            let index = linear % global_cells[dd];
            linear /= global_cells[dd];
            index
        });
        let tile = std::array::from_fn(|dd| global[dd] / local_cells[dd]);
        let target = flatten(tile, counts);
        let tracers = stores[target].tracers.as_mut().unwrap();
        tracers.x.push(spawned.x[ii]);
        tracers.id.push(spawned.id[ii]);
        tracers.cohort.push(spawned.cohort[ii]);
        tracers.flags.push(spawned.flags[ii]);
        tracers.owner.push(spawned.owner[ii]);
        tracers.step_owner.push(spawned.step_owner[ii]);
        tracers.step_flags.push(spawned.step_flags[ii]);
    }
}

fn spawn_decomposed_continuous_injection<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    counts: [usize; D],
    ledgers: Vec<std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>>,
) -> Result<usize, String> {
    let mut injections = std::collections::BTreeMap::new();
    for ledger in ledgers {
        for (destination, mass) in ledger {
            *injections.entry(destination).or_insert(0.0) += mass;
        }
    }
    if injections.is_empty() {
        return Ok(0);
    }
    let local_cells: [usize; D] =
        std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
    let global_cells: [usize; D] = std::array::from_fn(|dd| local_cells[dd] * counts[dd]);
    let global_lo: [f64; D] = std::array::from_fn(|dd| {
        stores
            .iter()
            .map(|store| crate::tracers::partition_physical_bounds(&store.geom)[dd].0)
            .fold(f64::INFINITY, f64::min)
    });
    let dx = stores[0].geom.dx;
    let template = stores
        .iter()
        .filter_map(|store| store.continuous_tracers.as_ref())
        .next()
        .ok_or_else(|| "decomposed continuous tracer population is missing".to_string())?;
    let mut spawned = crate::tracers::ContinuousTracerSet::<D, M>::allocate(0, template.order)?;
    spawned.weight = template.weight;
    spawned.run_seed = template.run_seed;
    spawned.next_id = stores
        .iter()
        .filter_map(|store| store.continuous_tracers.as_ref())
        .map(|tracers| tracers.next_id)
        .max()
        .unwrap_or(0);
    spawned.injection_remainder = stores
        .iter()
        .filter_map(|store| store.continuous_tracers.as_ref())
        .map(|tracers| tracers.injection_remainder)
        .sum();
    let key = crate::mass_transport::SamplingKey {
        run_seed: spawned.run_seed,
        epoch: stores[0].iteration | (1 << 62),
    };
    let count = crate::tracers::spawn_continuous_injected_tracers(
        &mut spawned,
        injections
            .into_iter()
            .map(|(destination, mass)| crate::mass_transport::MassTransfer { destination, mass }),
        |owner| {
            let mut linear = owner.0 as usize;
            let index: [usize; D] = std::array::from_fn(|dd| {
                let index = linear % global_cells[dd];
                linear /= global_cells[dd];
                index
            });
            (
                std::array::from_fn(|dd| global_lo[dd] + index[dd] as f64 * dx[dd]),
                dx,
            )
        },
        key,
    )?;
    for store in stores.iter_mut() {
        if let Some(tracers) = store.continuous_tracers.as_mut() {
            tracers.next_id = spawned.next_id;
            tracers.injection_remainder = 0.0;
        }
    }
    if let Some(tracers) = stores[0].continuous_tracers.as_mut() {
        tracers.injection_remainder = spawned.injection_remainder;
    }
    while spawned.len > 0 {
        let record = spawned.swap_remove_host(spawned.len - 1)?;
        let mut linear = record.owner.0 as usize;
        let global: [usize; D] = std::array::from_fn(|dd| {
            let index = linear % global_cells[dd];
            linear /= global_cells[dd];
            index
        });
        let tile = std::array::from_fn(|dd| global[dd] / local_cells[dd]);
        let target = flatten(tile, counts);
        stores[target]
            .continuous_tracers
            .as_mut()
            .expect("every decomposed tile carries continuous tracer storage")
            .push_host(record)?;
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
pub fn evolve_decomposed<const D: usize, const DOF: usize, M, K, T, F>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
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
    F: FnMut(u64, f64, &[&FieldStore<D, DOF, M, f64>]) -> ControlFlow<()>,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
{
    let stages = ts.stages();
    let multistage = crate::driver::needs_step_snapshot(stages);
    let n = stores.len();
    debug_assert_eq!(n, kernels.len(), "stores/kernels length mismatch");
    debug_assert_eq!(n, devices.len(), "stores/devices length mismatch");
    // bodies are replicated identically on every tile; the per-step backward feedback + prescribed
    // advance run once per step when any tile carries them.
    let has_bodies = stores.iter().any(|s| s.immersed.is_some());
    let has_discrete_tracers = stores.iter().any(|s| s.tracers.is_some());
    let has_continuous_tracers = stores.iter().any(|s| s.continuous_tracers.is_some());
    let tracer_bounds: [(f64, f64); D] = std::array::from_fn(|dd| {
        stores
            .iter()
            .map(|store| crate::tracers::partition_physical_bounds(&store.geom)[dd])
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(global_lo, global_hi), (lo, hi)| (global_lo.min(lo), global_hi.max(hi)),
            )
    });
    let tracer_boundaries = crate::state::Boundaries::per_axis(std::array::from_fn(|dd| {
        let lo = stores
            .iter()
            .min_by(|left, right| {
                crate::tracers::partition_physical_bounds(&left.geom)[dd]
                    .0
                    .total_cmp(&crate::tracers::partition_physical_bounds(&right.geom)[dd].0)
            })
            .expect("decomposition has at least one tile")
            .boundaries
            .lo(dd);
        let hi = stores
            .iter()
            .max_by(|left, right| {
                crate::tracers::partition_physical_bounds(&left.geom)[dd]
                    .1
                    .total_cmp(&crate::tracers::partition_physical_bounds(&right.geom)[dd].1)
            })
            .expect("decomposition has at least one tile")
            .boundaries
            .hi(dd);
        [lo, hi]
    }));

    // a fresh SHARED reborrow of the tiles for the field phases (kernels / exchange / checkpoint).
    // rebuilt per phase so the per-step body bookkeeping can take `&mut` between phases -- the
    // bodies live on the same FieldStore the fields do, so the shared slice must be dropped before
    // `step_bodies_decomposed` mutates them.
    macro_rules! shared {
        () => {{
            let sh: Vec<&FieldStore<D, DOF, M, f64>> = stores.iter().map(|s| &**s).collect();
            sh
        }};
    }

    // prime prim + ghosts (the stage entry contract), then seed the cut halos.
    {
        let sh = shared!();
        for i in 0..n {
            symbi_xpu::with_device(devices[i], || {
                kernels[i].c2p(sh[i]);
                kernels[i].ghost_fill(sh[i]);
            });
        }
        drain_devices::<M>(devices);
        exchange_grid(&sh, counts, devices, transport);
        // re-fill PHYSICAL boundary ghosts AFTER the exchange. at a corner where a domain-boundary
        // (outflow/reflect) meets a tile cut, the boundary ghost is derived from cells that include
        // the cut halo -- only valid post-exchange. with ghost_fill BEFORE the exchange, that corner
        // reads a stale (unexchanged) cut cell; for hydro it is harmless (uniform corners), but for
        // mhd the edge-EMF there is spurious and poisons the RK2 corrector. no-op interior cost.
        for i in 0..n {
            symbi_xpu::with_device(devices[i], || kernels[i].ghost_fill(sh[i]));
        }
        // fail loud on unphysical initial conditions, exactly like the uni-grid
        // and hierarchy drivers: without this, a bad decomposed IC marches to a
        // NaN-dt crash mid-run instead of naming the c2p failure at t = 0.
        for i in 0..n {
            let err = crate::hydro_ops::scan_c2p_errors(sh[i]);
            assert!(
                !err.is_err(),
                "decomposed c2p failed on initial conditions (tile {i}): {err}"
            );
        }
    }

    let mut t = start_time;
    let mut iter: u64 = 0;
    let mut last_cb: u64 = 0;
    while t < t_final {
        // global dt = min over tiles' cfl, clamped so the last step lands exactly on t_final.
        let mut dt = {
            let sh = shared!();
            let candidates =
                (0..n).map(|i| symbi_xpu::with_device(devices[i], || kernels[i].cfl(sh[i])));
            let dt = crate::driver::select_timestep(candidates, t_final - t, iter, t)
                .unwrap_or_else(|err| panic!("{}", err.detail));
            dt
        };
        // homologous mesh motion: each stage's dispatches bind geometry / grid-velocity
        // scalars from the tile's motion state, which must hold a(t) at the stage's
        // shu-osher ENTRY time — the same per-stage refresh the single-grid step performs,
        // applied in lockstep on every tile (identical inputs -> identical a). `a` is
        // restored to the step-entry value after the stages; the canonical step advance
        // lives at the step tail. static meshes assign a_n back to itself — no change.
        let motion_n: Vec<_> = stores.iter().map(|s| s.motion).collect();
        let a_n: Vec<f64> = motion_n.iter().map(|motion| motion.a).collect();
        let injection_ledgers = 'attempt: loop {
            let mut injection_ledgers = vec![std::collections::BTreeMap::new(); n];
            {
                let sh = shared!();
                for ii in 0..n {
                    if kernels[ii].fofc_active() {
                        symbi_xpu::with_device(devices[ii], || kernels[ii].snapshot_retry(sh[ii]));
                    }
                    if multistage {
                        symbi_xpu::with_device(devices[ii], || kernels[ii].snapshot(sh[ii]));
                    }
                }
            }
            for s in stores.iter_mut() {
                s.dt = dt;
                if has_discrete_tracers {
                    crate::tracers::snapshot_transport_state(&mut **s);
                }
                if has_continuous_tracers {
                    let geometry = s.geom.block_geometry(symbi_geometry::Cartesian);
                    crate::tracers::begin_ito_transport_store(&mut **s, &geometry)
                        .unwrap_or_else(|err| panic!("ito transport initialization failed: {err}"));
                }
            }
            let mut retry = false;
            for stage in crate::driver::stage_schedule(stages) {
                for (i, s) in stores.iter_mut().enumerate() {
                    let t_entry = s.time + stage.entry * dt;
                    let law_value = s
                        .motion_law
                        .as_ref()
                        .map(|law| (law.a_at(t_entry), law.adot_at(t_entry)));
                    crate::driver::set_stage_motion(
                        &mut s.motion,
                        law_value,
                        dt,
                        a_n[i],
                        stage.entry,
                    );
                }
                let sh = shared!();
                // the stage TAG is minted canonically inside the fold
                // (stage_tag: euler = 0, rk2 = 1 then 2). a per-driver `sidx + 1`
                // instead labels forward-euler as tag 1, which is the rk2-predictor
                // identity — the shared fold makes that divergence unrepresentable.
                for i in 0..n {
                    let outcome = symbi_xpu::with_device(devices[i], || {
                        // the full per-stage pipeline (evolve.rs STAGE_PIPELINE). wave_speeds / efield
                        // / post_godunov are the MHD constrained-transport hooks; they are no-op
                        // defaults for hydro + iso, so this is byte-identical to the prior sequence
                        // there, and drives the CT curl (edge emf -> bface -> bcell) for mhd.
                        // snapshot_stage / source_apply are the ADDITIVE (non-fused) source pass: gated
                        // on `has_additive_source`, so source-free runs of every regime skip them and
                        // stay byte-identical. body_source is the forward immersed-body pass (gravity +
                        // accretion sink), POINTWISE from the body's GLOBAL position so each tile
                        // applies it to its own cells -- no cross-tile coupling. all of these are
                        // pointwise/local; the only cross-tile work is the post-stage halo exchange.
                        // the shared stage table (symbi-sim::stage): identical
                        // phase sequence to every other driver. this loop never
                        // elides the stage-input copy (no cross-tile alias
                        // tracking); the halo exchange + second ghost fill
                        // follow OUTSIDE the fold — the decomposed sequence's
                        // documented delta.
                        crate::stage::fold_stage(
                            sh[i],
                            kernels[i],
                            crate::stage::StageArgs {
                                dt,
                                a0: stage.a0,
                                ac: stage.ac,
                                stage: stage.index,
                                n_stages: stages.len(),
                                allow_elision: false,
                            },
                            &mut |_| {},
                        )
                    });
                    retry |= outcome == crate::stage::StageOutcome::RetryStep;
                }
                if retry {
                    break;
                }
                // refresh the cut halos from each neighbor's stage-updated interior.
                drain_devices::<M>(devices);
                exchange_grid(&sh, counts, devices, transport);
                // re-fill physical boundary ghosts post-exchange (cut-corner consistency, see prime).
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || kernels[i].ghost_fill(sh[i]));
                }
                drop(sh);
                if has_continuous_tracers {
                    for store in stores.iter_mut() {
                        let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                        crate::tracers::accumulate_ito_transport_stage_store(
                            &mut **store,
                            &geometry,
                            stage.ac,
                        )
                        .unwrap_or_else(|err| panic!("ito transport accumulation failed: {err}"));
                    }
                }
                if has_discrete_tracers || has_continuous_tracers {
                    let local_cells: [usize; D] =
                        std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
                    for (flat, store) in stores.iter_mut().enumerate() {
                        if store.geom.coords != symbi_geometry::Geometry::Cartesian {
                            panic!(
                                "decomposed mass-transport tracers require explicit curvilinear geometry"
                            );
                        }
                        let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                        let tile = unflatten(flat, counts);
                        let layout = crate::tracers::TransportLayout {
                            global_cells: std::array::from_fn(|dd| local_cells[dd] * counts[dd]),
                            tile_offset: std::array::from_fn(|dd| tile[dd] * local_cells[dd]),
                            level: 0,
                        };
                        let mut injections = crate::tracers::boundary_injection_transfers_store(
                            &**store, &geometry, layout,
                        );
                        injections.extend(crate::tracers::source_injection_transfers_store(
                            &**store, &geometry, layout, stage.a0, stage.ac,
                        ));
                        crate::tracers::fold_injection_ledger(
                            &mut injection_ledgers[flat],
                            injections,
                            stage.ac,
                        );
                        if has_discrete_tracers {
                            crate::tracers::advance_stage_mass_transport_store(
                                &mut **store,
                                &geometry,
                                layout,
                                0.0,
                                1.0,
                                stage.index,
                            )
                            .unwrap_or_else(|err| panic!("tracer mass transport failed: {err}"));
                        }
                    }
                    if has_discrete_tracers {
                        migrate_mass_transport_tracers(stores, counts);
                        blend_mass_transport_ancestry(stores, stage.ac, stage.index);
                        migrate_mass_transport_tracers(stores, counts);
                    }
                }
            }
            if !retry {
                break 'attempt injection_ledgers;
            }
            // per-tile rollback restores the fields and each tracer's step-entry ancestry, but the
            // stage loop MIGRATES tracer records between tiles as they cross a cut. a migrated
            // record now lives in a different tile's storage and no per-tile restore puts it back.
            assert!(
                !has_discrete_tracers && !has_continuous_tracers,
                "a decomposed step was rejected with tracers attached: cross-tile tracer \
                 migration has no per-tile inverse"
            );
            for (ii, store) in stores.iter_mut().enumerate() {
                if kernels[ii].fofc_active() {
                    symbi_xpu::with_device(devices[ii], || kernels[ii].restore_step(&**store));
                }
                crate::tracers::restore_transport_state(&mut **store);
                store.motion = motion_n[ii];
            }
            drain_devices::<M>(devices);
            {
                let sh = shared!();
                exchange_grid(&sh, counts, devices, transport);
                for ii in 0..n {
                    symbi_xpu::with_device(devices[ii], || kernels[ii].ghost_fill(sh[ii]));
                }
            }
            dt =
                crate::driver::retry_timestep(dt, t).unwrap_or_else(|err| panic!("{}", err.detail));
        };
        if has_discrete_tracers {
            spawn_decomposed_injection(stores, counts, injection_ledgers.clone());
        }
        // the stage refresh mutated a; the step-tail advance below starts from the
        // step-entry value, mirroring the single-grid step's restore.
        for (i, s) in stores.iter_mut().enumerate() {
            s.motion.a = a_n[i];
        }
        if has_continuous_tracers {
            for store in stores.iter_mut() {
                let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                crate::tracers::materialize_ito_coefficients_store(&mut **store, &geometry)
                    .unwrap_or_else(|err| panic!("ito coefficient materialization failed: {err}"));
                crate::tracers::fill_ito_coefficient_boundaries_host(
                    store
                        .ito_coefficients
                        .as_ref()
                        .expect("ito coefficients were materialized"),
                    &store.geom,
                    store.boundaries,
                )
                .unwrap_or_else(|err| panic!("ito coefficient boundary fill failed: {err}"));
            }
            {
                let sh = shared!();
                exchange_ito_coefficients(&sh, counts, devices, transport);
            }
            drain_devices::<M>(devices);
            for store in stores.iter_mut() {
                let Some(mut tracers) = store.continuous_tracers.take() else {
                    continue;
                };
                let coefficients = store
                    .ito_coefficients
                    .as_ref()
                    .expect("ito coefficients were materialized");
                let (scale_start, scale_end, offset_start, offset_end) =
                    crate::tracers::continuous_tracer_mesh_step(&**store, dt);
                crate::tracers::advance_continuous_tracers(
                    &mut tracers,
                    coefficients,
                    &store.geom,
                    scale_start,
                    scale_end,
                    offset_start,
                    offset_end,
                    dt,
                )
                .unwrap_or_else(|err| panic!("ito tracer advancement failed: {err}"));
                let physical_bounds = crate::tracers::map_continuous_tracer_bounds(
                    tracer_bounds,
                    scale_end,
                    offset_end,
                );
                crate::tracers::apply_continuous_boundaries_host(
                    &mut tracers,
                    physical_bounds,
                    tracer_boundaries,
                )
                .unwrap_or_else(|err| panic!("ito tracer boundaries failed: {err}"));
                store.continuous_tracers = Some(tracers);
            }
            migrate_continuous_tracers(stores, counts)
                .unwrap_or_else(|err| panic!("continuous tracer migration failed: {err}"));
            spawn_decomposed_continuous_injection(stores, counts, injection_ledgers)
                .unwrap_or_else(|err| panic!("continuous tracer injection failed: {err}"));
        }
        let mut horizon_receipt = None;
        let mut accretion_density = Vec::new();
        {
            let sh = shared!();
            // the viscous transport, per tile, once per step after
            // the final halo exchange (the +-1 stencil reads the neighbor's
            // exchanged edge). body-independent; inert when inviscid.
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || {
                    prof("viscous", || kernels[i].viscous(sh[i], dt))
                });
            }
            drain_devices::<M>(devices);
            // horizon excision, once per step after the RK combination, mirroring
            // the monolithic loop's phase order. the pass count comes from the
            // store, so every tile runs the same number and the tiled sequence
            // stays bit-identical to the monolithic one; a final exchange
            // publishes the finalized
            // (rebuilt) excised state into the neighbors' halos before the next
            // step's stencils read them. inert (zero passes) when unexcised.
            let passes = (0..n)
                .map(|i| kernels[i].excise_pass_count(sh[i]))
                .max()
                .unwrap_or(0);
            if passes > 0 {
                for _ in 0..passes {
                    for i in 0..n {
                        symbi_xpu::with_device(devices[i], || kernels[i].excise_sweep(sh[i]));
                    }
                    drain_devices::<M>(devices);
                    exchange_grid(&sh, counts, devices, transport);
                }
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || kernels[i].excise_finalize(sh[i]));
                }
                drain_devices::<M>(devices);
                exchange_grid(&sh, counts, devices, transport);
            }
            let horizon = sh.first().and_then(|store| horizon_request(*store));
            if let Some((index, diagnostic_radius)) = horizon {
                let (mdot, edot) = (0..n)
                    .map(|ii| {
                        symbi_xpu::with_device(devices[ii], || {
                            prof("horizon_accretion", || {
                                kernels[ii].horizon_accretion(sh[ii], diagnostic_radius)
                            })
                        })
                    })
                    .fold((0.0, 0.0), |(mass, energy), (local_mass, local_energy)| {
                        (mass + local_mass, energy + local_energy)
                    });
                horizon_receipt = Some((index, mdot, edot));
            }
            // backward immersed-body feedback (per STEP, after all stages): each tile reduces its
            // LOCAL interior force/torque/accreted-mass into its own accumulator. the cross-tile sum
            // + the prescribed-orbit advance happen in `step_bodies_decomposed` below, which needs
            // `&mut` -- so `sh` (a shared reborrow of `stores`) MUST be dropped first.
            if has_bodies {
                // the wall relaxation's Alfven stiffness c_a2 = max|B|^2/rho is a GLOBAL max over
                // the domain; reduce the per-tile maxima and publish the global value to every tile
                // so a magnetized wall straddling a cut relaxes at the monolithic rate (a per-tile
                // local max would diverge from the monolithic single-grid run). inert (0) off MHD.
                let global_c_a2 = (0..n)
                    .map(|i| crate::state::local_c_a2_max(sh[i]))
                    .fold(0.0_f64, f64::max);
                for i in 0..n {
                    if let Some(im) = sh[i].immersed.as_ref() {
                        im.set_c_a2_override(global_c_a2);
                    }
                }
                if has_discrete_tracers {
                    drain_devices::<M>(devices);
                    accretion_density = sh
                        .iter()
                        .map(|store| crate::tracers::snapshot_accretion_density(*store))
                        .collect();
                }
                for i in 0..n {
                    // IBM surface physics ONCE per step, after all stages
                    // (receipt == removal; see evolve.rs), then the feedback —
                    // gated like the other drivers: only bodies whose dynamics
                    // consume the reduction (two-way or accreting) pay for it.
                    symbi_xpu::with_device(devices[i], || {
                        prof("penalize", || kernels[i].penalize(sh[i], dt))
                    });
                    let needs_fb = sh[i]
                        .immersed
                        .as_ref()
                        .is_some_and(|im| im.bodies.needs_feedback());
                    if needs_fb {
                        symbi_xpu::with_device(devices[i], || {
                            prof("body_feedback", || kernels[i].body_feedback(sh[i], dt))
                        });
                    }
                }
                drain_devices::<M>(devices);
            }
        }
        if let Some((index, mdot, edot)) = horizon_receipt {
            for store in stores.iter_mut() {
                book_horizon_receipt(&mut **store, index, mdot, edot, dt);
            }
        }
        if !accretion_density.is_empty() {
            let local_cells: [usize; D] =
                std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
            for (flat, store) in stores.iter_mut().enumerate() {
                let tile = unflatten(flat, counts);
                let layout = crate::tracers::TransportLayout {
                    global_cells: std::array::from_fn(|dd| local_cells[dd] * counts[dd]),
                    tile_offset: std::array::from_fn(|dd| tile[dd] * local_cells[dd]),
                    level: 0,
                };
                let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                let crossing_time = store.time + dt;
                crate::tracers::advance_accretion_transport_store(
                    &mut **store,
                    &geometry,
                    layout,
                    &accretion_density[flat],
                    crossing_time,
                )
                .unwrap_or_else(|detail| panic!("tracer accretion transport: {detail}"));
                crate::tracers::advance_continuous_accretion_transport_store(
                    &mut **store,
                    &geometry,
                    layout,
                    &accretion_density[flat],
                    crossing_time,
                )
                .unwrap_or_else(|detail| panic!("continuous tracer accretion transport: {detail}"));
            }
        }
        if has_bodies {
            step_bodies_decomposed(stores, dt);
        }
        // per-tile clocks + mesh-motion advance, in lockstep (identical law + identical dt on
        // every tile -> identical scale factors; the cuts sit at fixed comoving indices, so the
        // halo exchange is unaffected). the tile time feeds time-dependent driven-boundary
        // prescriptions and the checkpoint metadata. the homologous linear advance matches the
        // single-grid step; a traced motion law is then sampled EXACTLY at the new time
        // (a constant-rate extrapolation would overshoot a decelerating mesh).
        for s in stores.iter_mut() {
            advance_state_clock(&mut **s, dt);
        }
        if has_discrete_tracers {
            let local_cells: [usize; D] =
                std::array::from_fn(|dd| stores[0].geom.interior.spaces[dd].size());
            for (flat, store) in stores.iter_mut().enumerate() {
                let tile = unflatten(flat, counts);
                let layout = crate::tracers::TransportLayout {
                    global_cells: std::array::from_fn(|dd| local_cells[dd] * counts[dd]),
                    tile_offset: std::array::from_fn(|dd| tile[dd] * local_cells[dd]),
                    level: 0,
                };
                let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                crate::tracers::refresh_derived_positions_store(&mut **store, &geometry, layout);
            }
        }
        if std::env::var_os("SYMBI_TRACE_DT").is_some() {
            eprintln!("SYMBI_TRACE_DT iter={iter} t={t:.6e} dt={dt:.6e}");
        }
        (t, iter) = crate::driver::advance_clock(t, iter, dt);
        if iter - last_cb >= interval {
            last_cb = iter;
            drain_devices::<M>(devices);
            let sh = shared!();
            if on_checkpoint(iter, t, &sh).is_break() {
                return;
            }
        }
    }
    drain_devices::<M>(devices);
    let sh = shared!();
    let _ = on_checkpoint(iter, t, &sh);
}

// the gather/scatter kernels for `StagedCopy`: pack a strided strip into a contiguous
// buffer, and scatter a contiguous buffer back into a strided strip. the contiguous buffer
// is the interchange a peer copy moves across the device-to-device link.
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
/// a peer copy over the intra-node device fabric moves between
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
            let gather = current_dispatcher().jit_kernel_keyed(
                HALO_GATHER_KERNEL,
                "decomp/halo_gather",
                "halo_gather",
            );
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
            // this is where the peer copy moves `buf` to the neighbor's
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
/// idempotent and best-effort: a failure to enable just means that pair stages its copies.
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

/// the cross-device halo transport: gather the strip into the source
/// device's contiguous buffer, peer-copy it to the destination device's buffer over the
/// intra-node device fabric, then scatter it into the destination strip. the gather and
/// scatter ARE `StagedCopy`'s proven halves; only the peer move in the middle is new. when
/// `src_dev == dst_dev` there is nothing to move across, so it defers to the proven
/// single-device `StagedCopy`. host fallback for a cpu backend. NOT exercisable on one gpu (a
/// device cannot peer with itself); the equivalence test runs it on a real multi-gpu node.
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
        // fold onto the same physical gpu -> no peer -> staged), on a node whose devices can peer
        // (direct fast path), and on a node without peer access (staged) -- for any gpu count, no
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

                let sp = pool[src_dev as usize]
                    .as_mut()
                    .unwrap()
                    .idx
                    .as_mut_ptr::<u32>();
                for (i, sc) in src_region.iter().enumerate() {
                    unsafe { *sp.add(i) = sdom.flat_index(sc) as u32 };
                }
                let dp = pool[dst_dev as usize]
                    .as_mut()
                    .unwrap()
                    .idx
                    .as_mut_ptr::<u32>();
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
            .expect("device peer copy failed");

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
    use super::{
        LocalCopy, decompose_grid, exchange_ito_coefficients, migrate_continuous_tracers,
        spawn_decomposed_continuous_injection,
    };

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

    #[test]
    fn ito_coefficient_halos_cross_tile_cuts() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use crate::tracers::ItoCoefficientFields;
        use symbi_geometry::Cartesian;
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let make = |x_lo| {
            SimState::<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
                Newtonian,
                IdealGas { gamma: 1.4 },
                Cartesian,
                [2],
                [x_lo],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap()
        };
        let mut lo = make(0.0);
        let mut hi = make(2.0);
        lo.ito_coefficients = Some(ItoCoefficientFields::zeros(&lo.geom.allocated).unwrap());
        hi.ito_coefficients = Some(ItoCoefficientFields::zeros(&hi.geom.allocated).unwrap());
        for coord in lo.geom.interior.iter() {
            lo.ito_coefficients.as_ref().unwrap().drift[0]
                .view_mut()
                .set(coord, 1.0);
        }
        for coord in hi.geom.interior.iter() {
            hi.ito_coefficients.as_ref().unwrap().drift[0]
                .view_mut()
                .set(coord, 2.0);
        }

        exchange_ito_coefficients(&[&lo, &hi], [2], &[0, 0], &LocalCopy);

        let lo_hi_ghost = [lo.geom.interior.spaces[0].hi];
        let hi_lo_ghost = [hi.geom.interior.spaces[0].lo - 1];
        assert_eq!(
            *lo.ito_coefficients.as_ref().unwrap().drift[0]
                .view()
                .at(lo_hi_ghost),
            2.0
        );
        assert_eq!(
            *hi.ito_coefficients.as_ref().unwrap().drift[0]
                .view()
                .at(hi_lo_ghost),
            1.0
        );
    }

    #[test]
    fn continuous_migration_preserves_identity_and_counter_across_a_cut() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use crate::tracers::{ContinuousTracerSet, TracerSet};
        use symbi_geometry::Cartesian;
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let make = |x_lo| {
            SimState::<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
                Newtonian,
                IdealGas { gamma: 1.4 },
                Cartesian,
                [2],
                [x_lo],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap()
        };
        let mut lo = make(0.0);
        let mut hi = make(2.0);
        let seed = TracerSet::<1>::seed_stratified(&[([2.0], [1.0])], &[1], 0.5);
        let mut particles = ContinuousTracerSet::<1, HostMemory>::from_discrete(
            &seed,
            crate::mass_transport::ItoOrder::Three,
        )
        .unwrap();
        unsafe {
            *particles.random_counter.as_mut_ptr::<u64>() = 23;
        }
        lo.continuous_tracers = Some(particles);
        hi.continuous_tracers =
            Some(ContinuousTracerSet::allocate(0, crate::mass_transport::ItoOrder::Three).unwrap());

        let migrated = migrate_continuous_tracers(&mut [&mut lo, &mut hi], [2]).unwrap();

        assert_eq!(migrated, 1);
        assert_eq!(lo.continuous_tracers.as_ref().unwrap().len, 0);
        let target = hi.continuous_tracers.as_mut().unwrap();
        assert_eq!(target.len, 1);
        let record = target.swap_remove_host(0).unwrap();
        assert_eq!(record.id, seed.id[0]);
        assert_eq!(record.x, [2.5]);
        assert_eq!(record.random_counter, 23);
        assert_eq!(record.owner, crate::mass_transport::ContainerId(2));
    }

    #[test]
    fn decomposed_continuous_injection_has_one_global_remainder_and_destination() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use crate::tracers::ContinuousTracerSet;
        use symbi_geometry::Cartesian;
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let make = |x_lo| {
            SimState::<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
                Newtonian,
                IdealGas { gamma: 1.4 },
                Cartesian,
                [2],
                [x_lo],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap()
        };
        let mut lo = make(0.0);
        let mut hi = make(2.0);
        for store in [&mut lo, &mut hi] {
            let mut tracers =
                ContinuousTracerSet::allocate(0, crate::mass_transport::ItoOrder::Two).unwrap();
            tracers.weight = 0.1;
            tracers.run_seed = 5;
            tracers.next_id = 9;
            store.continuous_tracers = Some(tracers);
        }
        let mut ledger = std::collections::BTreeMap::new();
        ledger.insert(crate::mass_transport::ContainerId(2), 0.25);

        let spawned = spawn_decomposed_continuous_injection(
            &mut [&mut lo, &mut hi],
            [2],
            vec![ledger, Default::default()],
        )
        .unwrap();

        assert_eq!(spawned, 2);
        assert_eq!(lo.continuous_tracers.as_ref().unwrap().len, 0);
        assert_eq!(hi.continuous_tracers.as_ref().unwrap().len, 2);
        assert!(
            (lo.continuous_tracers.as_ref().unwrap().injection_remainder - 0.05).abs() < 1.0e-15
        );
        assert_eq!(hi.continuous_tracers.as_ref().unwrap().next_id, 11);
    }
}
