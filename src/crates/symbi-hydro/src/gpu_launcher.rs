// =============================================================================
// gpu_launcher.rs
//
// **the GPU launch wiring**, gated behind `--features cuda`. takes the CUDA
// source emitted by `GpuSourceKernel`, JIT-compiles via NVRTC (no nvcc, no
// host C++ compiler required — `DISPATCHER.jit_kernel_keyed` owns the
// compiler), launches over a 1D thread grid, and reads back per-component
// output buffers.
//
// **the symmetry**:
//   - `SourceEvaluator::eval(field, values)` — CPU, per-cell, f64 vec out.
//   - `launch_source_kernel(...)`            — GPU, per-domain, double* buffers.
//
// the IR is THE SAME `BuiltSource.graph` for both paths (A1). this layer
// is the GPU dispatch; the CPU equivalent's interpreter call is its peer.
//
// **what's tested separately** (`tests/gpu_source_launch.rs`):
//   - the launched kernel produces ULP-equivalent results to the CPU
//     evaluator on a real grid.
//   - argument packing order matches the GPU ABI declared by
//     `GpuSourceKernel.params_for` / `output_count`.
//   - unified memory routes input + output through one allocator.
// =============================================================================

use symbi_xpu::{ctx_sync, DeviceMemory};
use symbi_xpu::runtime::current_dispatcher;
use symbi_xpu::runtime::GpuRuntime;
use symbi_xpu::{KernelArgs, LaunchConfig, MemoryBlock};

use crate::gpu_source_kernel::GpuSourceKernel;

/// thread-block size for the source-kernel launches. matches the existing
/// substrate kernels' conventions (`substrate_hydro_gpu`-style 64-thread
/// blocks); the source kernels are simple enough that 64 is plenty.
const BLOCK_SIZE: u32 = 64;

/// launch one `__global__` kernel for `field` over `n_cells` threads.
///
/// **input contract** (matches `GpuSourceKernel.params_for(field)` order):
///   `input_buffers` — one `Vec<f64>` per declared param, each of length
///   `n_cells` (one cell-value per thread index).
///
/// **output contract** (matches `GpuSourceKernel.output_count(field)`):
///   returns `Vec<Vec<f64>>` — one output buffer per source component,
///   each of length `n_cells`.
///
/// **panics** if `field` is not in the kernel's table, if `input_buffers`
/// don't match the declared param count, or if any input buffer's length
/// doesn't equal `n_cells`. these are programmer errors caught early —
/// the runtime contract is precise; a mismatch is a bug, not a
/// recoverable condition.
///
/// # Safety
///
/// the kernel is `extern "C" __global__` declared in
/// `GpuSourceKernel::cuda_source`; the param/output buffer ABI matches
/// what this function packs (one `const double*` per param, one
/// `double*` per output, `unsigned int n_cells` last). misuse via a
/// hand-edited kernel string is the only way to break ABI; that's why
/// callers obtain the source through `GpuSourceKernel`.
pub fn launch_source_kernel(
    kernel: &GpuSourceKernel,
    field: &str,
    input_buffers: &[Vec<f64>],
    n_cells: usize,
) -> Vec<Vec<f64>> {
    let source = kernel
        .cuda_source(field)
        .unwrap_or_else(|| panic!("launch_source_kernel: no kernel for field '{field}'"));
    let entry_name = kernel.entry_name(field).expect("entry_name available");
    let params = kernel.params_for(field).expect("params declared");
    let n_outputs = kernel.output_count(field).expect("output_count declared");

    assert_eq!(
        params.len(),
        input_buffers.len(),
        "launch_source_kernel: param count mismatch for field '{field}' \
         (kernel expects {} buffers, caller provided {})",
        params.len(),
        input_buffers.len(),
    );
    for (i, buf) in input_buffers.iter().enumerate() {
        assert_eq!(
            buf.len(),
            n_cells,
            "launch_source_kernel: input buffer {i} ('{}') has length {} \
             (expected {n_cells})",
            params[i],
            buf.len(),
        );
    }

    // allocate unified memory for inputs + outputs. unified is host- AND
    // device-addressable (no explicit copies; the GPU dereferences the
    // same pointer the host writes to). matches the existing substrate
    // conventions (`substrate_hydro_gpu`-style).
    let input_blocks: Vec<MemoryBlock<DeviceMemory>> = input_buffers
        .iter()
        .map(|data| {
            let mut block = MemoryBlock::<DeviceMemory>::for_elements::<f64>(n_cells)
                .expect("unified alloc for source-kernel input");
            let ptr = block.as_mut_ptr::<f64>();
            for (j, &v) in data.iter().enumerate() {
                unsafe { *ptr.add(j) = v; }
            }
            block
        })
        .collect();

    let mut output_blocks: Vec<MemoryBlock<DeviceMemory>> = (0..n_outputs)
        .map(|_| {
            MemoryBlock::<DeviceMemory>::for_elements::<f64>(n_cells)
                .expect("unified alloc for source-kernel output")
        })
        .collect();

    // JIT-compile the kernel via NVRTC. the dispatcher dedups by
    // `(cache_key, hash(source))` internally — distinct sources with the
    // same `cache_key` never collide. a plain diagnostic name suffices
    // here; the dispatcher enforces content-addressed correctness.
    let cache_key = format!("hydro/source/{entry_name}");
    let jit_kernel = current_dispatcher().jit_kernel_keyed(source, &cache_key, entry_name);

    // pack arguments in the order declared by the wrapped __global__:
    //   const double* param_0, ..., double* out_0, ..., unsigned int n_cells.
    // `KernelArgs::push(&val)` copies the value; pointers are the addressable
    // device-side ptrs of the unified blocks.
    let input_ptrs: Vec<u64> = input_blocks
        .iter()
        .map(|b| b.as_ptr::<f64>() as u64)
        .collect();
    let output_ptrs: Vec<u64> = output_blocks
        .iter_mut()
        .map(|b| b.as_mut_ptr::<f64>() as u64)
        .collect();

    let mut args = KernelArgs::new();
    for p in &input_ptrs { args.push(p); }
    for p in &output_ptrs { args.push(p); }
    let n_cells_u32 = n_cells as u32;
    args.push(&n_cells_u32);

    let config = LaunchConfig::for_1d(n_cells_u32, BLOCK_SIZE);
    unsafe {
        current_dispatcher()
            .runtime()
            .launch(&jit_kernel, config, args.as_mut_slice())
            .unwrap_or_else(|e| panic!(
                "GPU launch '{entry_name}' failed: {e:?}"
            ));
    }
    ctx_sync();

    // read back outputs into host vecs.
    output_blocks
        .iter()
        .map(|block| {
            let ptr = block.as_ptr::<f64>();
            (0..n_cells).map(|i| unsafe { *ptr.add(i) }).collect::<Vec<f64>>()
        })
        .collect()
}
