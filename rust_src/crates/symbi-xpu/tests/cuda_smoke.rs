// =============================================================================
// cuda_smoke.rs
//
// smoke test for the new xpu API on CUDA.
// exercises: executor, memory block, kernel launch, tokens.
//
// usage:
//   cargo test -p symbi-xpu --test cuda_smoke --features cuda
// =============================================================================

#![cfg(feature = "cuda")]

use symbi_xpu::*;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};

#[test]
fn executor_lifecycle() {
    let exec = Executor::<CudaSpace>::new(0).unwrap();
    assert!(exec.ready().unwrap());
    exec.sync().unwrap();
}

#[test]
fn memory_block_unified() {
    let mut block = MemoryBlock::<UnifiedMemory>::for_elements::<f64>(100).unwrap();
    assert_eq!(block.bytes(), 800);
    assert!(!block.is_empty());

    let ptr = block.as_mut_ptr::<f64>();
    for ii in 0..100 {
        unsafe { *ptr.add(ii) = ii as f64; }
    }
    for ii in 0..100 {
        assert_eq!(unsafe { *ptr.add(ii) }, ii as f64);
    }
}

#[test]
fn launch_trivial_kernel() {
    let exec = Executor::<CudaSpace>::new(0).unwrap();

    let ptx_src = b"\
extern \"C\" __global__ void double_it(const double* src, double* dst, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = src[tid] * 2.0;
}
";
    let ptx = compile_ptx_for_test(ptx_src);
    let module = CudaSpace::load_module(&ptx).unwrap();
    let kernel = CudaSpace::get_function(&module, "double_it").unwrap();

    let nn = 256usize;
    let mut src = MemoryBlock::<UnifiedMemory>::for_elements::<f64>(nn).unwrap();
    let dst = MemoryBlock::<UnifiedMemory>::for_elements::<f64>(nn).unwrap();

    let sp = src.as_mut_ptr::<f64>();
    for ii in 0..nn {
        unsafe { *sp.add(ii) = ii as f64; }
    }

    let src_ptr = src.as_ptr::<f64>() as u64;
    let dst_ptr = dst.as_ptr::<f64>() as u64;
    let nu = nn as u32;
    let mut args = KernelArgs::new();
    args.push(&src_ptr);
    args.push(&dst_ptr);
    args.push(&nu);

    unsafe {
        exec.launch(&kernel, LaunchConfig::for_1d(nu, 256), &mut args).unwrap();
    }
    exec.sync().unwrap();

    let dp = dst.as_ptr::<f64>();
    for ii in 0..nn {
        let expected = ii as f64 * 2.0;
        let got = unsafe { *dp.add(ii) };
        assert!(
            (got - expected).abs() < 1e-12,
            "mismatch at {}: got {}, expected {}", ii, got, expected
        );
    }
}

#[test]
fn token_cross_stream() {
    let exec1 = Executor::<CudaSpace>::new(0).unwrap();
    let exec2 = Executor::<CudaSpace>::new(0).unwrap();

    let ptx_src = b"\
extern \"C\" __global__ void noop_kernel(unsigned int n) {
}
";
    let ptx = compile_ptx_for_test(ptx_src);
    let module = CudaSpace::load_module(&ptx).unwrap();
    let kernel = CudaSpace::get_function(&module, "noop_kernel").unwrap();

    let nu = 1u32;
    let mut args = KernelArgs::new();
    args.push(&nu);

    let token = unsafe {
        exec1.launch(&kernel, LaunchConfig::for_1d(1, 1), &mut args).unwrap()
    };

    token.wait_on(&exec2).unwrap();

    exec1.sync().unwrap();
    exec2.sync().unwrap();
    assert!(token.ready().unwrap());
}

#[test]
fn error_bad_function_name() {
    let ptx_src = b"\
extern \"C\" __global__ void real_name(unsigned int n) {}
";
    let ptx = compile_ptx_for_test(ptx_src);
    let module = CudaSpace::load_module(&ptx).unwrap();
    let result = CudaSpace::get_function(&module, "wrong_name");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code, 500); // CUDA_ERROR_NOT_FOUND
}

#[test]
fn error_bad_ptx() {
    let result = CudaSpace::load_module(b"this is not ptx");
    assert!(result.is_err());
}

fn compile_ptx_for_test(src: &[u8]) -> Vec<u8> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join("symbi_xpu_test");
    std::fs::create_dir_all(&dir).unwrap();
    let cu_path = dir.join(format!("test_{id}.cu"));
    let ptx_path = dir.join(format!("test_{id}.ptx"));
    std::fs::write(&cu_path, src).unwrap();

    let output = std::process::Command::new("nvcc")
        .args([
            "-ptx", "-o", ptx_path.to_str().unwrap(),
            cu_path.to_str().unwrap(),
            "--gpu-architecture=native",
        ])
        .output()
        .expect("nvcc not found");

    assert!(
        output.status.success(),
        "nvcc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read(&ptx_path).unwrap()
}
