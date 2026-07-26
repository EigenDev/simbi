// =============================================================================
// gpu_rhd_c2p.rs
//
// GPU<->CPU runtime validation of the RHD cons->prim kernel — the final leg of
// the substrate verification gate (CPU numerics + nvcc PTX + on-device run). the
// SAME substrate IR graph is emitted to two backends: the CPU Rust fn
// `symbi_aot::rhd_c2p_1d` and the neutral IR blob `symbi_aot::RHD_C2P_1D_IR`,
// rendered to CUDA source at test time via `render_from_ir`.
// the CUDA is nvcc-compiled to PTX, launched on the GPU; the test asserts the
// device output matches BOTH the CPU kernel AND the analytic primitives the
// conserved states were built from. proves the iterative kernel (20-step masked
// Newton, lowered to nested SELECT) computes correct physics ON THE DEVICE.
//
// run (in the symbi-cuda distrobox; NVCC_CCBIN points nvcc at gcc-11):
//   NVCC_CCBIN=/usr/bin/g++ cargo test -p symbi-xpu --features cuda --test gpu_rhd_c2p
// =============================================================================

#![cfg(feature = "cuda")]

use symbi_aot::{CpuField, CpuFieldMut, RHD_C2P_1D_IR, rhd_c2p_1d__raw as rhd_c2p_1d};
use symbi_ir::emit::{Precision, Target};
use symbi_ir::render_from_ir;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::*;

const GAMMA: f64 = 5.0 / 3.0;

// host-side mirror of the CUDA `__symbi_View` struct (the kernel's by-value buffer
// ABI; see device_view_abi.rs). `#[repr(C)]` so the bytes KernelArgs copies match
// the device struct field-for-field: pointer, then lo / strides / extent.
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceView {
    data: *const std::ffi::c_void,
    lo: [i32; 4],
    strides: [i32; 4],
    extent: [i32; 4],
}

// a 1D view over a contiguous `n`-element buffer at origin 0 (stride 1).
fn view_1d(ptr: *const f64, n: usize) -> DeviceView {
    DeviceView {
        data: ptr as *const std::ffi::c_void,
        lo: [0; 4],
        strides: [1, 0, 0, 0],
        extent: [n as i32, 0, 0, 0],
    }
}

// render a kernel's neutral IR blob to CUDA source at f64.
fn cuda_src(ir: &str) -> String {
    render_from_ir(ir, Target::Cuda, Precision::F64).source
}

// (rho, v, p): static + mild-relativistic states (|v| <= 0.8), converge within
// the 20 baked Newton steps.
const CASES: &[(f64, f64, f64)] = &[
    (1.0, 0.0, 1.0),
    (1.0, 0.3, 0.5),
    (2.0, 0.5, 1.0),
    (0.5, -0.4, 0.2),
    (1.0, 0.6, 2.0),
    (3.0, 0.8, 5.0),
    (1.0, -0.7, 1.0),
];

// analytic forward map (1D ideal gas): primitives -> conserved (D, S, tau).
fn prim_to_cons(rho: f64, v: f64, p: f64, gamma: f64) -> (f64, f64, f64) {
    let w = 1.0 / (1.0 - v * v).sqrt();
    let eps = p / ((gamma - 1.0) * rho);
    let h = 1.0 + eps + p / rho;
    let rhw2 = rho * h * w * w;
    let d = rho * w;
    (d, rhw2 * v, rhw2 - p - d)
}

// nvcc-compile CUDA source to PTX (native arch). honors NVCC_CCBIN from the env.
fn compile_to_ptx(src: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join("symbi_gpu_rhd");
    std::fs::create_dir_all(&dir).unwrap();
    let cu = dir.join("rhd_c2p_1d.cu");
    let ptx = dir.join("rhd_c2p_1d.ptx");
    std::fs::write(&cu, src).unwrap();
    let out = std::process::Command::new("nvcc")
        .args([
            "-ptx",
            "-O3",
            "--gpu-architecture=native",
            "-o",
            ptx.to_str().unwrap(),
            cu.to_str().unwrap(),
        ])
        .output()
        .expect("nvcc not found");
    assert!(
        out.status.success(),
        "nvcc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read(&ptx).unwrap()
}

#[test]
fn rhd_c2p_gpu_matches_cpu_and_analytic() {
    let n = CASES.len();
    let den: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).0)
        .collect();
    let mom: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).1)
        .collect();
    let nrg: Vec<f64> = CASES
        .iter()
        .map(|&(r, v, p)| prim_to_cons(r, v, p, GAMMA).2)
        .collect();

    // ---- CPU backend (the AOT Rust kernel) ----
    // the __raw kernel takes view-wrapped buffers (CpuField/CpuFieldMut) + grid +
    // dom_lo + gamma. scope the field wrappers so their borrows of the output Vecs
    // end before the results are read back.
    let (mut rho_cpu, mut vel_cpu, mut pre_cpu) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    {
        let lo = [0i32];
        let ext = [n as u32];
        let din = CpuField::from_layout(&den, &lo, &ext);
        let min = CpuField::from_layout(&mom, &lo, &ext);
        let nin = CpuField::from_layout(&nrg, &lo, &ext);
        let mut rout = CpuFieldMut::from_layout(&mut rho_cpu, &lo, &ext);
        let mut vout = CpuFieldMut::from_layout(&mut vel_cpu, &lo, &ext);
        let mut pout = CpuFieldMut::from_layout(&mut pre_cpu, &lo, &ext);
        rhd_c2p_1d(
            &din, &min, &nin, &mut rout, &mut vout, &mut pout, n as i32, 0, GAMMA,
        );
    }

    // ---- GPU backend (the CUDA emit of the SAME IR graph) ----
    let exec = Executor::<CudaSpace>::new(0).unwrap();
    let ptx = compile_to_ptx(&cuda_src(RHD_C2P_1D_IR));
    let module = CudaSpace::load_module(&ptx).unwrap();
    let kernel = CudaSpace::get_function(&module, "rhd_c2p_1d").unwrap();

    // six unified buffers: 3 conserved in, 3 primitive out.
    let mk = || MemoryBlock::<UnifiedMemory>::for_elements::<f64>(n).unwrap();
    let (mut b_den, mut b_mom, mut b_nrg) = (mk(), mk(), mk());
    let (mut b_rho, mut b_vel, mut b_pre) = (mk(), mk(), mk());
    for (blk, src) in [(&mut b_den, &den), (&mut b_mom, &mom), (&mut b_nrg, &nrg)] {
        let p = blk.as_mut_ptr::<f64>();
        for i in 0..n {
            unsafe {
                *p.add(i) = src[i];
            }
        }
    }

    // the kernel takes 6 `__symbi_View` structs (3 conserved in, 3 primitive out) by
    // value, then grid_size_0 (u32), dom_lo_0 (i32), gamma (f64).
    let views = [
        view_1d(b_den.as_ptr::<f64>(), n),
        view_1d(b_mom.as_ptr::<f64>(), n),
        view_1d(b_nrg.as_ptr::<f64>(), n),
        view_1d(b_rho.as_mut_ptr::<f64>(), n),
        view_1d(b_vel.as_mut_ptr::<f64>(), n),
        view_1d(b_pre.as_mut_ptr::<f64>(), n),
    ];
    let grid: u32 = n as u32;
    let zero: i32 = 0;
    let mut args = KernelArgs::new();
    for v in &views {
        args.push(v);
    }
    args.push(&grid);
    args.push(&zero); // dom_lo_0
    args.push(&GAMMA);

    unsafe {
        exec.launch(&kernel, LaunchConfig::for_1d(grid, 64), &mut args)
            .unwrap();
    }
    exec.sync().unwrap();

    let (gr, gv, gp) = (
        b_rho.as_ptr::<f64>(),
        b_vel.as_ptr::<f64>(),
        b_pre.as_ptr::<f64>(),
    );
    for (i, &(r0, v0, p0)) in CASES.iter().enumerate() {
        let (rg, vg, pg) = unsafe { (*gr.add(i), *gv.add(i), *gp.add(i)) };
        // GPU vs analytic ground truth (round-trip). nvcc fuses FMA, so allow a
        // loose-but-meaningful relative tolerance (project_fma_discipline).
        let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(1.0);
        assert!(
            rel(rg, r0) < 1e-7,
            "case {i}: GPU rho {rg} != analytic {r0}"
        );
        assert!(
            rel(vg, v0) < 1e-7,
            "case {i}: GPU vel {vg} != analytic {v0}"
        );
        assert!(
            rel(pg, p0) < 1e-7,
            "case {i}: GPU pre {pg} != analytic {p0}"
        );
        // GPU vs CPU (same IR graph, two backends): agree modulo FMA drift.
        assert!(
            rel(rg, rho_cpu[i]) < 1e-9,
            "case {i}: GPU rho {rg} != CPU {}",
            rho_cpu[i]
        );
        assert!(
            rel(vg, vel_cpu[i]) < 1e-9,
            "case {i}: GPU vel {vg} != CPU {}",
            vel_cpu[i]
        );
        assert!(
            rel(pg, pre_cpu[i]) < 1e-9,
            "case {i}: GPU pre {pg} != CPU {}",
            pre_cpu[i]
        );
    }
}
