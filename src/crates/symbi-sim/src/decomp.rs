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

/// move `src` over `src_region` into `dst` over `dst_region`, cell-for-cell. the two
/// regions have identical shape (same per-axis extents); the transport pairs them by
/// iteration order. impls: `LocalCopy` (in-process), and later a gpu peer copy and an
/// mpi pack/send/unpack.
pub trait HaloTransport {
    fn copy_region<const D: usize, M: MemorySpace>(
        &self,
        src: &Field<f64, D, M>,
        src_region: &Domain<D>,
        dst: &Field<f64, D, M>,
        dst_region: &Domain<D>,
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

/// device-side transport: the strip copy runs as a gpu kernel, with no host roundtrip of
/// the field data. the per-cell flat-offset arrays are precomputed on the host and uploaded
/// (small metadata); the data itself never leaves the device. on a host backend it falls
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
    ) {
        use symbi_xpu::cuda::{ctx_sync, UnifiedMemory};
        use symbi_xpu::runtime::cuda_runtime::DISPATCHER;
        use symbi_xpu::runtime::GpuRuntime;
        use symbi_xpu::{KernelArgs, LaunchConfig, MemoryBlock};

        // the kernel dereferences device pointers; a host backend has none.
        if !M::IS_DEVICE_ACCESSIBLE {
            LocalCopy.copy_region(src, src_region, dst, dst_region);
            return;
        }

        let n = src_region.volume();
        debug_assert_eq!(n, dst_region.volume(), "src/dst strip shapes differ");
        if n == 0 {
            return;
        }

        // precompute the flat offset of each strip cell into its field buffer. these match
        // the View::at offsets because both index the field's allocated domain.
        // NOTE: allocates the index buffers per call -- correct but not yet pooled.
        let mut sidx = MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("sidx alloc");
        let mut didx = MemoryBlock::<UnifiedMemory>::for_elements::<u32>(n).expect("didx alloc");
        let sp = sidx.as_mut_ptr::<u32>();
        let dp = didx.as_mut_ptr::<u32>();
        let sdom = src.domain();
        let ddom = dst.domain();
        for (i, (sc, dc)) in src_region.iter().zip(dst_region.iter()).enumerate() {
            unsafe {
                *sp.add(i) = sdom.flat_index(sc) as u32;
                *dp.add(i) = ddom.flat_index(dc) as u32;
            }
        }

        let kernel = DISPATCHER.jit_kernel_keyed(HALO_COPY_KERNEL, "decomp/halo_copy", "halo_copy");
        let src_ptr = src.as_ptr() as u64;
        let dst_ptr = dst.as_mut_ptr() as u64;
        let sidx_ptr = sidx.as_ptr::<u32>() as u64;
        let didx_ptr = didx.as_ptr::<u32>() as u64;
        let n_u32 = n as u32;

        let mut args = KernelArgs::new();
        args.push(&src_ptr);
        args.push(&dst_ptr);
        args.push(&sidx_ptr);
        args.push(&didx_ptr);
        args.push(&n_u32);

        let config = LaunchConfig::for_1d(n_u32, 64);
        unsafe {
            DISPATCHER
                .runtime()
                .launch(&kernel, config, args.as_mut_slice())
                .expect("halo_copy launch failed");
        }
        // drain before the index buffers drop: the kernel reads them and launch is async.
        ctx_sync();
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
        transport.copy_region(fr, &hi_src, fl, &lo_ghost); // hi interior -> lo ghost
        transport.copy_region(fl, &lo_src, fr, &hi_ghost); // lo interior -> hi ghost
    }
}
