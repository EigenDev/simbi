// =============================================================================
// gpu_regimes.rs
//
// GPU<->CPU runtime validation for the iso / adiabatic / RHD c2p + flux + snapshot
// + wave-speed + mass + ghost-fill kernels: the SAME substrate IR graph emitted to
// two backends — the CPU Rust fn (`symbi_aot::*_1d__raw`) and the neutral IR blob
// (`symbi_aot::*_IR`), rendered to CUDA source at test time via `render_from_ir`.
// each test nvcc-compiles the CUDA to PTX, launches it on the
// GPU, and asserts the device output matches the CPU kernel (modulo nvcc FMA fusion).
//
// ABI: the CPU `__raw` fns take view-wrapped
// buffers (`CpuField` / `CpuFieldMut`, carrying lo+strides) + `grid_size_0` +
// `dom_lo_0` + the per-kernel scalar tail (NO per-buffer `buf_lo` args — folded into
// the view). the GPU kernels take the matching `__symbi_View` structs by value
// (`DeviceView` Rust mirror; see device_view_abi.rs) + the same scalar tail. the
// flux / wave-speed kernels carry geometry scalars (mesh_adot / x_lo / dx /
// mesh_vtrans); cartesian = all-zero geometry, dx arbitrary (cartesian flux is
// position-independent), passed IDENTICALLY to both backends so the comparison holds.
//
// the godunov euler/rk2 integrators are AOT-emitted as part of the unified
// `*_godunov_stage_*` family; their GPU<->CPU validation lives in
// `crates/symbi/tests/substrate_hydro_gpu.rs` (the production dispatch path runs
// godunov_euler/rk2 on-device). the standalone single-kernel mass integrator
// (`godunov_mass_1d`) is covered here.
//
// run (host-native; NVCC_CCBIN -> g++-15):
//   NVCC_CCBIN=/usr/bin/g++-15 cargo test -p symbi-xpu --features cuda --test gpu_regimes
// =============================================================================

#![cfg(feature = "cuda")]

use symbi_aot::{
    ADIABATIC_C2P_1D_IR, ADIABATIC_FACE_FLUX_1D_0_IR, CpuField, CpuFieldMut, GODUNOV_MASS_1D_IR,
    ISO_C2P_1D_IR, ISO_FACE_FLUX_1D_0_IR, ISO_GHOST_FILL_1D_IR, ISO_SNAPSHOT_1D_IR,
    ISO_WAVE_SPEED_MAP_1D_IR, RHD_FACE_FLUX_1D_0_IR, adiabatic_c2p_1d__raw as adiabatic_c2p_1d,
    adiabatic_face_flux_1d_0__raw as adiabatic_face_flux_1d,
    godunov_mass_1d__raw as godunov_mass_1d, iso_c2p_1d__raw as iso_c2p_1d,
    iso_face_flux_1d_0__raw as iso_face_flux_1d, iso_ghost_fill_1d__raw as iso_ghost_fill_1d,
    iso_snapshot_1d__raw as iso_snapshot_1d, iso_wave_speed_map_1d__raw as iso_wave_speed_map_1d,
    rhd_face_flux_1d_0__raw as rhd_face_flux_1d,
};
use symbi_ir::emit::{Precision, Target};
use symbi_ir::render_from_ir;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::*;

// render a kernel's neutral IR blob to CUDA source at f64.
fn cuda_src(ir: &str) -> String {
    render_from_ir(ir, Target::Cuda, Precision::F64).source
}

// ---- CPU-side view wrappers (1D, origin 0, contiguous) -------------------------
// the __raw kernels take `CpuField` / `CpuFieldMut`; in 1D the stride is always 1, so
// the buffer's own length is a valid extent for `from_layout`.
fn cf(v: &[f64]) -> CpuField<'_, f64> {
    CpuField::from_layout(v, &[0], &[v.len() as u32])
}
fn cfm(v: &mut [f64]) -> CpuFieldMut<'_, f64> {
    let n = v.len() as u32;
    CpuFieldMut::from_layout(v, &[0], &[n])
}

// ---- GPU-side view ABI ---------------------------------------------------------
// host-side mirror of the CUDA `__symbi_View` struct (device_view_abi.rs). `#[repr(C)]`
// so KernelArgs copies the bytes the device struct expects: pointer, lo, strides, extent.
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceView {
    data: *const std::ffi::c_void,
    lo: [i32; 4],
    strides: [i32; 4],
    extent: [i32; 4],
}

fn view_1d(ptr: *const f64, n: usize) -> DeviceView {
    DeviceView {
        data: ptr as *const std::ffi::c_void,
        lo: [0; 4],
        strides: [1, 0, 0, 0],
        extent: [n as i32, 0, 0, 0],
    }
}

// nvcc-compile CUDA source -> PTX (native arch). honors NVCC_CCBIN from the env.
fn compile_to_ptx(src: &str, name: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join("symbi_gpu_regimes");
    std::fs::create_dir_all(&dir).unwrap();
    let cu = dir.join(format!("{name}.cu"));
    let ptx = dir.join(format!("{name}.ptx"));
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
        "nvcc failed for {name}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read(&ptx).unwrap()
}

// launch a standard 1D substrate kernel (clean in-then-out buffer split) and return
// its `n_out` output buffers. ABI: in/out `__symbi_View` structs, grid_size_0 (u32),
// dom_lo_0 (i32), then the scalar params (f64) in declared order. all buffers are
// `buf_len` long.
fn launch_1d(
    cuda: &str,
    name: &str,
    ins: &[&[f64]],
    n_out: usize,
    scalars: &[f64],
    grid: u32,
    dom_lo: i32,
) -> Vec<Vec<f64>> {
    let buf_len = ins[0].len();
    let exec = Executor::<CudaSpace>::new(0).unwrap();
    let ptx = compile_to_ptx(cuda, name);
    let module = CudaSpace::load_module(&ptx).unwrap();
    let kernel = CudaSpace::get_function(&module, name).unwrap();

    let n_in = ins.len();
    let nbuf = n_in + n_out;
    let mut blocks: Vec<MemoryBlock<UnifiedMemory>> = (0..nbuf)
        .map(|_| MemoryBlock::<UnifiedMemory>::for_elements::<f64>(buf_len).unwrap())
        .collect();
    for (i, data) in ins.iter().enumerate() {
        let p = blocks[i].as_mut_ptr::<f64>();
        for (j, &v) in data.iter().enumerate() {
            unsafe {
                *p.add(j) = v;
            }
        }
    }

    let views: Vec<DeviceView> = blocks
        .iter()
        .map(|b| view_1d(b.as_ptr::<f64>(), buf_len))
        .collect();
    let mut args = KernelArgs::new();
    for v in &views {
        args.push(v);
    }
    args.push(&grid);
    args.push(&dom_lo);
    for s in scalars {
        args.push(s);
    }
    unsafe {
        exec.launch(&kernel, LaunchConfig::for_1d(grid, 64), &mut args)
            .unwrap();
    }
    exec.sync().unwrap();

    (n_in..nbuf)
        .map(|i| {
            let p = blocks[i].as_ptr::<f64>();
            (0..buf_len).map(|j| unsafe { *p.add(j) }).collect()
        })
        .collect()
}

// raw launcher for kernels whose buffers are NOT a clean in-then-out split (in-place
// updates: ghost_fill gathers in-place). allocates one unified buffer per `bufs` entry
// (all `buf_len` long), launches with the View prefix then the scalar params as `i32`s
// followed by `f64`s (the order these kernels declare them), and reads ALL buffers back.
fn launch_raw(
    cuda: &str,
    name: &str,
    bufs: &[Vec<f64>],
    ints: &[i32],
    floats: &[f64],
    grid: u32,
    dom_lo: i32,
) -> Vec<Vec<f64>> {
    let buf_len = bufs[0].len();
    let exec = Executor::<CudaSpace>::new(0).unwrap();
    let ptx = compile_to_ptx(cuda, name);
    let module = CudaSpace::load_module(&ptx).unwrap();
    let kernel = CudaSpace::get_function(&module, name).unwrap();

    let blocks: Vec<MemoryBlock<UnifiedMemory>> = bufs
        .iter()
        .map(|data| {
            let mut b = MemoryBlock::<UnifiedMemory>::for_elements::<f64>(buf_len).unwrap();
            let p = b.as_mut_ptr::<f64>();
            for (j, &v) in data.iter().enumerate() {
                unsafe {
                    *p.add(j) = v;
                }
            }
            b
        })
        .collect();
    let views: Vec<DeviceView> = blocks
        .iter()
        .map(|b| view_1d(b.as_ptr::<f64>(), buf_len))
        .collect();
    let mut args = KernelArgs::new();
    for v in &views {
        args.push(v);
    }
    args.push(&grid);
    args.push(&dom_lo);
    for x in ints {
        args.push(x);
    }
    for x in floats {
        args.push(x);
    }
    unsafe {
        exec.launch(&kernel, LaunchConfig::for_1d(grid, 64), &mut args)
            .unwrap();
    }
    exec.sync().unwrap();
    blocks
        .iter()
        .map(|b| {
            let p = b.as_ptr::<f64>();
            (0..buf_len).map(|j| unsafe { *p.add(j) }).collect()
        })
        .collect()
}

// GPU vs CPU agree modulo nvcc FMA fusion (rustc/CPU doesn't fuse) — ULP-bounded
// drift accepted (project_fma_discipline). loose-but-meaningful relative tol.
fn assert_close(gpu: &[f64], cpu: &[f64], lo: usize, hi: usize, what: &str) {
    for i in lo..hi {
        let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1.0);
        assert!(
            rel < 1e-9,
            "{what} cell {i}: GPU {} != CPU {} (rel {rel:e})",
            gpu[i],
            cpu[i]
        );
    }
}

// cartesian geometry scalars shared by the flux / wave-speed kernels: no moving mesh
// (mesh_adot = mesh_vtrans = 0), origin 0. dx is position-independent for cartesian
// flux but must be passed identically to both backends.
const X_LO: f64 = 0.0;
const MESH_ADOT: f64 = 0.0;
const MESH_VTRANS: f64 = 0.0;

#[test]
fn iso_c2p_gpu_matches_cpu() {
    let n = 6usize;
    let den: Vec<f64> = vec![1.0, 2.0, 0.5, 1.5, 0.8, 1.2];
    let mom: Vec<f64> = vec![0.3, -0.4, 0.1, 0.6, -0.2, 0.05];
    // cs2 is the substrate-owned per-cell sound-speed-squared input buffer (locally-
    // isothermal closure). uniform cs2 = cs*cs reduces to globally isothermal.
    let cs = 0.7_f64;
    let cs2: Vec<f64> = vec![cs * cs; n];
    let (mut r, mut v, mut p) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    iso_c2p_1d(
        &cf(&den),
        &cf(&mom),
        &cf(&cs2),
        &mut cfm(&mut r),
        &mut cfm(&mut v),
        &mut cfm(&mut p),
        n as i32,
        0,
    );

    let g = launch_1d(
        &cuda_src(ISO_C2P_1D_IR),
        "iso_c2p_1d",
        &[&den, &mom, &cs2],
        3,
        &[],
        n as u32,
        0,
    );
    assert_close(&g[0], &r, 0, n, "iso rho");
    assert_close(&g[1], &v, 0, n, "iso vel");
    assert_close(&g[2], &p, 0, n, "iso pre");
}

#[test]
fn adiabatic_c2p_gpu_matches_cpu() {
    let n = 6usize;
    let den: Vec<f64> = vec![1.0, 2.0, 0.5, 1.5, 0.8, 1.2];
    let mom: Vec<f64> = vec![0.3, -0.4, 0.1, 0.6, -0.2, 0.05];
    let nrg: Vec<f64> = vec![2.5, 4.0, 1.2, 3.0, 1.8, 2.2];
    let gamma = 5.0 / 3.0;
    let (mut r, mut v, mut p) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    adiabatic_c2p_1d(
        &cf(&den),
        &cf(&mom),
        &cf(&nrg),
        &mut cfm(&mut r),
        &mut cfm(&mut v),
        &mut cfm(&mut p),
        n as i32,
        0,
        gamma,
    );

    let g = launch_1d(
        &cuda_src(ADIABATIC_C2P_1D_IR),
        "adiabatic_c2p_1d",
        &[&den, &mom, &nrg],
        3,
        &[gamma],
        n as u32,
        0,
    );
    assert_close(&g[0], &r, 0, n, "adiabatic rho");
    assert_close(&g[1], &v, 0, n, "adiabatic vel");
    assert_close(&g[2], &p, 0, n, "adiabatic pre");
}

// a non-uniform primitive profile so PLM reconstruction + the HLLE wave structure
// are actually exercised (a uniform field would make the flux trivially F(U)).
fn varying_prims(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let rho: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let v: Vec<f64> = (0..n).map(|i| 0.2 * (i as f64 * 0.5).sin()).collect();
    let p: Vec<f64> = (0..n).map(|i| 0.5 + 0.05 * i as f64).collect();
    (rho, v, p)
}

#[test]
fn iso_face_flux_gpu_matches_cpu() {
    let n = 8usize;
    let (rho, v, p) = varying_prims(n);
    let theta = 1.5_f64; // theta-MC slope limiter (project_theta_mc_limiter default)
    let dx = 1.0_f64;
    // scalar tail: theta, mesh_adot_0, x_lo_0, dx_0, mesh_vtrans_0.
    let scal = [theta, MESH_ADOT, X_LO, dx, MESH_VTRANS];
    // interior only: stencil reads coord-2..coord+1 -> iterate cells 2..6.
    let (mut fd, mut fm) = (vec![0.0; n], vec![0.0; n]);
    iso_face_flux_1d(
        &cf(&rho),
        &cf(&v),
        &cf(&p),
        &mut cfm(&mut fd),
        &mut cfm(&mut fm),
        4,
        2,
        scal[0],
        scal[1],
        scal[2],
        scal[3],
        scal[4],
    );

    let g = launch_1d(
        &cuda_src(ISO_FACE_FLUX_1D_0_IR),
        "iso_face_flux_1d_0",
        &[&rho, &v, &p],
        2,
        &scal,
        4,
        2,
    );
    assert_close(&g[0], &fd, 2, 6, "iso flux_den");
    assert_close(&g[1], &fm, 2, 6, "iso flux_mom");
}

#[test]
fn adiabatic_face_flux_gpu_matches_cpu() {
    let n = 8usize;
    let (rho, v, p) = varying_prims(n);
    let gamma = 5.0 / 3.0;
    let theta = 1.5_f64;
    let dx = 1.0_f64;
    // scalar tail: gamma, theta, mesh_adot_0, x_lo_0, dx_0, mesh_vtrans_0.
    let scal = [gamma, theta, MESH_ADOT, X_LO, dx, MESH_VTRANS];
    let (mut fd, mut fm, mut fn_) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    adiabatic_face_flux_1d(
        &cf(&rho),
        &cf(&v),
        &cf(&p),
        &mut cfm(&mut fd),
        &mut cfm(&mut fm),
        &mut cfm(&mut fn_),
        4,
        2,
        scal[0],
        scal[1],
        scal[2],
        scal[3],
        scal[4],
        scal[5],
    );

    let g = launch_1d(
        &cuda_src(ADIABATIC_FACE_FLUX_1D_0_IR),
        "adiabatic_face_flux_1d_0",
        &[&rho, &v, &p],
        3,
        &scal,
        4,
        2,
    );
    assert_close(&g[0], &fd, 2, 6, "adiabatic flux_den");
    assert_close(&g[1], &fm, 2, 6, "adiabatic flux_mom");
    assert_close(&g[2], &fn_, 2, 6, "adiabatic flux_nrg");
}

#[test]
fn rhd_face_flux_gpu_matches_cpu() {
    let n = 8usize;
    let (rho, v, p) = varying_prims(n); // |v| <= 0.2 here — safely sub-luminal
    let gamma = 5.0 / 3.0;
    let theta = 1.5_f64;
    let dx = 1.0_f64;
    let scal = [gamma, theta, MESH_ADOT, X_LO, dx, MESH_VTRANS];
    let (mut fd, mut fm, mut fn_) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    rhd_face_flux_1d(
        &cf(&rho),
        &cf(&v),
        &cf(&p),
        &mut cfm(&mut fd),
        &mut cfm(&mut fm),
        &mut cfm(&mut fn_),
        4,
        2,
        scal[0],
        scal[1],
        scal[2],
        scal[3],
        scal[4],
        scal[5],
    );

    let g = launch_1d(
        &cuda_src(RHD_FACE_FLUX_1D_0_IR),
        "rhd_face_flux_1d_0",
        &[&rho, &v, &p],
        3,
        &scal,
        4,
        2,
    );
    assert_close(&g[0], &fd, 2, 6, "rhd flux_den");
    assert_close(&g[1], &fm, 2, 6, "rhd flux_mom");
    assert_close(&g[2], &fn_, 2, 6, "rhd flux_nrg");
}

#[test]
fn iso_snapshot_gpu_matches_cpu() {
    // u_n = cons (pointwise copy): 2 in (cons.den, mom_0) -> 2 out (u_n.den, mom_0).
    let n = 6usize;
    let den: Vec<f64> = vec![1.0, 2.0, 0.5, 1.5, 0.8, 1.2];
    let mom: Vec<f64> = vec![0.3, -0.4, 0.1, 0.6, -0.2, 0.05];
    let (mut un_d, mut un_m) = (vec![0.0; n], vec![0.0; n]);
    iso_snapshot_1d(
        &cf(&den),
        &cf(&mom),
        &mut cfm(&mut un_d),
        &mut cfm(&mut un_m),
        n as i32,
        0,
    );

    let g = launch_1d(
        &cuda_src(ISO_SNAPSHOT_1D_IR),
        "iso_snapshot_1d",
        &[&den, &mom],
        2,
        &[],
        n as u32,
        0,
    );
    assert_close(&g[0], &un_d, 0, n, "snapshot u_n.den");
    assert_close(&g[1], &un_m, 0, n, "snapshot u_n.mom");
}

#[test]
fn iso_wave_speed_map_gpu_matches_cpu() {
    // per-cell CFL lambda from the reconstructed primitives + gamma + 1/dx.
    let n = 6usize;
    let rho: Vec<f64> = vec![1.0, 2.0, 0.5, 1.5, 0.8, 1.2];
    let v: Vec<f64> = vec![0.3, -0.4, 0.1, 0.6, -0.2, 0.05];
    let p: Vec<f64> = vec![0.5, 1.0, 0.2, 0.8, 0.4, 0.6];
    let gamma = 5.0 / 3.0;
    let inv_dx = 2.0_f64;
    let dx = 0.5_f64; // 1 / inv_dx (consistent; cartesian lambda only uses inv_dx)
    // scalar tail: gamma, inv_dx_0, x_lo_0, dx_0, mesh_adot_0, mesh_vtrans_0.
    let scal = [gamma, inv_dx, X_LO, dx, MESH_ADOT, MESH_VTRANS];
    let mut lam = vec![0.0; n];
    iso_wave_speed_map_1d(
        &cf(&rho),
        &cf(&v),
        &cf(&p),
        &mut cfm(&mut lam),
        n as i32,
        0,
        scal[0],
        scal[1],
        scal[2],
        scal[3],
        scal[4],
        scal[5],
    );

    let g = launch_1d(
        &cuda_src(ISO_WAVE_SPEED_MAP_1D_IR),
        "iso_wave_speed_map_1d",
        &[&rho, &v, &p],
        1,
        &scal,
        n as u32,
        0,
    );
    assert_close(&g[0], &lam, 0, n, "wave-speed lambda");
}

#[test]
fn godunov_mass_gpu_matches_cpu() {
    // single mass law to a SEPARATE buffer: rho_new = rho - dt*div(mass_flux). the
    // flux is read at ii and ii+1, so pad all buffers to n+1 and iterate 0..n.
    let n = 8usize;
    let den: Vec<f64> = (0..=n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let mflux: Vec<f64> = (0..=n).map(|i| 0.3 - 0.02 * i as f64).collect();
    let (dt, dx) = (0.01_f64, 0.5_f64);
    let mut den_new = vec![0.0; n + 1];
    godunov_mass_1d(
        &cf(&den),
        &cf(&mflux),
        &mut cfm(&mut den_new),
        n as i32,
        0,
        dt,
        dx,
    );

    let g = launch_1d(
        &cuda_src(GODUNOV_MASS_1D_IR),
        "godunov_mass_1d",
        &[&den, &mflux],
        1,
        &[dt, dx],
        n as u32,
        0,
    );
    assert_close(&g[0], &den_new, 0, n, "godunov_mass rho_new");
}

#[test]
fn iso_ghost_fill_gpu_matches_cpu() {
    // lattice-map pullback (in-place gather): prim[ii] = map(prim[src]). exercise the
    // REFLECT map (map_type=2: src = arg - coord) with a vel sign flip. iterate a
    // left-ghost range [0,2) whose source cells [5,6] are disjoint -> no in-place
    // read/write hazard, so parallel (GPU) == sequential (CPU).
    let n = 8usize;
    let rho: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let vel: Vec<f64> = (0..n).map(|i| 0.2 + 0.05 * i as f64).collect();
    let pre: Vec<f64> = (0..n).map(|i| 0.5 + 0.07 * i as f64).collect();
    let (map_type, arg, vel_sign) = (2_i32, 6_i32, -1.0_f64);
    let (grid, dom_lo) = (2u32, 0i32);

    let (mut r_c, mut v_c, mut p_c) = (rho.clone(), vel.clone(), pre.clone());
    iso_ghost_fill_1d(
        &mut cfm(&mut r_c),
        &mut cfm(&mut v_c),
        &mut cfm(&mut p_c),
        grid as i32,
        dom_lo,
        map_type,
        arg,
        vel_sign,
    );

    let g = launch_raw(
        &cuda_src(ISO_GHOST_FILL_1D_IR),
        "iso_ghost_fill_1d",
        &[rho, vel, pre],
        &[map_type, arg],
        &[vel_sign],
        grid,
        dom_lo,
    );
    assert_close(&g[0], &r_c, 0, grid as usize, "ghost_fill rho");
    assert_close(&g[1], &v_c, 0, grid as usize, "ghost_fill vel");
    assert_close(&g[2], &p_c, 0, grid as usize, "ghost_fill pre");
}
