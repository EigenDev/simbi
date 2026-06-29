// =============================================================================
// substrate_gpu_dispatch.rs
//
// proof of the structured-ABI GPU mapping (docs/design/15 §5, step 3c-2) AND its
// precision-genericity (§4): build the SAME `KernelInvocation` the substrate
// KernelSet hands `run_cpu`, back it with unified memory, and route it through
// `substrate_gpu::dispatch`. with a device-accessible `Mem`, dispatch renders the
// neutral IR at the scalar's precision, NVRTC-compiles it, reorders the buffers
// into kernel binding order, packs the args, and launches — and the device output
// must match the CPU kernel. run at BOTH f64 and f32 in one process, which also
// proves the precision-keyed render + module caches don't collide.
//
// runs on the HOST GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test substrate_gpu_dispatch
// =============================================================================

#![cfg(feature = "gpu")]

use symbi::regimes::substrate_gpu::dispatch;
use symbi_aot::{
    iso_c2p_1d, Buf, BufHandle, CpuField, CpuFieldMut, KernelInvocation, OrderedNumeric, Scalar,
    ISO_C2P_1D_IR,
};
use symbi_xpu::DeviceMemory;
use symbi_xpu::MemoryBlock;

// dispatch iso_c2p over unified `S` buffers and assert the device prim matches the
// CPU kernel to `tol` (relative). generic over the scalar — f64 and f32 both render
// from the one IR blob, at their own precision.
fn iso_c2p_gpu_matches_cpu<S: Scalar + OrderedNumeric>(tol: f64) {
    let n = 6usize;
    let den: Vec<S> = [1.0, 2.0, 0.5, 1.5, 0.8, 1.2].iter().map(|&x| S::from_f64(x)).collect();
    let mom: Vec<S> = [0.3, -0.4, 0.1, 0.6, -0.2, 0.05].iter().map(|&x| S::from_f64(x)).collect();
    // cs2 is now a per-cell FIELD (the prescribed sound-speed-squared). a varying cs2 here
    // exercises the LOCALLY isothermal path: p = cs2(x)*rho, not a global constant.
    let cs2: Vec<S> = [0.49, 0.6, 0.4, 0.55, 0.5, 0.45].iter().map(|&x| S::from_f64(x)).collect();

    // CPU reference (the AOT-compiled kernel via the descriptor ABI), at S
    // precision. 6 buffers (den, mom, cs2 in; rho, vel, pre out), no scalar.
    let z = S::from_f64(0.0);
    let (mut rc, mut vc, mut pc) = (vec![z; n], vec![z; n], vec![z; n]);
    {
        let blo = [0i32];
        let bext = [n as u32];
        let inputs = [
            CpuField::from_layout(den.as_slice(), &blo, &bext),
            CpuField::from_layout(mom.as_slice(), &blo, &bext),
            CpuField::from_layout(cs2.as_slice(), &blo, &bext),
        ];
        let mut outputs = [
            CpuFieldMut::from_layout(rc.as_mut_slice(), &blo, &bext),
            CpuFieldMut::from_layout(vc.as_mut_slice(), &blo, &bext),
            CpuFieldMut::from_layout(pc.as_mut_slice(), &blo, &bext),
        ];
        iso_c2p_1d::<S>(&inputs, &mut outputs, &[n as u32], &[0], &[], &[]);
    }

    // 6 unified buffers (den, mom, cs2 inputs; rho, vel, pre outputs) — host- and
    // device-addressable, so they go straight into the invocation handles.
    let mut blocks: Vec<MemoryBlock<DeviceMemory>> =
        (0..6).map(|_| MemoryBlock::<DeviceMemory>::for_elements::<S>(n).unwrap()).collect();
    let ptrs: Vec<*mut S> = blocks.iter_mut().map(|b| b.as_mut_ptr::<S>()).collect();
    for (i, data) in [&den, &mom, &cs2].iter().enumerate() {
        for (j, &x) in data.iter().enumerate() {
            unsafe { *ptrs[i].add(j) = x; }
        }
    }

    let lo = [0i32];
    let ext = [n as u32];
    let grid = [n as u32];
    let dom_lo = [0i32];
    // SAFETY: `blocks` outlives `inv`; the 6 buffers are disjoint, each n elements.
    let inv = KernelInvocation::<S> {
        buffers: vec![
            Buf { handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(ptrs[0], n) }), lo: &lo, extent: &ext },
            Buf { handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(ptrs[1], n) }), lo: &lo, extent: &ext },
            Buf { handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(ptrs[2], n) }), lo: &lo, extent: &ext },
            Buf { handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(ptrs[3], n) }), lo: &lo, extent: &ext },
            Buf { handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(ptrs[4], n) }), lo: &lo, extent: &ext },
            Buf { handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(ptrs[5], n) }), lo: &lo, extent: &ext },
        ],
        grid: &grid,
        dom_lo: &dom_lo,
        ints: &[],
        scalars: &[],
    };

    // DeviceMemory is device-accessible -> dispatch routes to run_gpu (render at
    // S's precision + NVRTC). the cpu fn is the (unused here) host fallback.
    dispatch::<S, DeviceMemory, _>(inv, ISO_C2P_1D_IR, "iso_c2p_1d", iso_c2p_1d);
    // launches are asynchronous: drain the device queue before the host reads
    // the unified buffers (the B12 host-read barrier).
    symbi::regimes::substrate_gpu::device_sync::<DeviceMemory>();

    let read = |p: *mut S| (0..n).map(|j| unsafe { (*p.add(j)).to_f64() }).collect::<Vec<f64>>();
    let (rg, vg, pg) = (read(ptrs[3]), read(ptrs[4]), read(ptrs[5]));
    let close = |g: f64, c: f64, what: &str, i: usize| {
        let rel = (g - c).abs() / c.abs().max(1.0);
        assert!(rel < tol, "{what}[{i}]: gpu {g} != cpu {c} (rel {rel:e}, tol {tol:e})");
    };
    for i in 0..n {
        close(rg[i], rc[i].to_f64(), "rho", i);
        close(vg[i], vc[i].to_f64(), "vel", i);
        close(pg[i], pc[i].to_f64(), "pre", i);
    }
}

#[test]
fn dispatch_routes_iso_c2p_to_gpu_matching_cpu() {
    iso_c2p_gpu_matches_cpu::<f64>(1e-12);
}

// the precision-generic GPU path (docs/design/15 §4): the SAME IR blob renders at
// f32, the f32 buffers/scalars launch, and match the f32 CPU kernel (looser tol —
// single precision + FMA). running in the same process as the f64 test above proves
// the precision-keyed render/module caches keep the two builds distinct.
#[test]
fn dispatch_routes_iso_c2p_f32_to_gpu_matching_cpu() {
    iso_c2p_gpu_matches_cpu::<f32>(1e-5);
}
