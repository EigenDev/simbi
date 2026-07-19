// =============================================================================
// multi_device.rs
//
// validates the device-binding infrastructure WITHOUT a second
// gpu: logical device 1 round-robins onto the one physical card as a SECOND cuda context.
// each context has its own module table, so the per-device dispatcher must jit + launch in
// the right context -- a single shared dispatcher would launch device 0's module in device
// 1's context and hit "invalid resource handle". running a kernel under each `with_device`
// proves the context registry, the per-device dispatcher, and current-device restore.
// =============================================================================

#![cfg(feature = "cuda")]

use symbi_xpu::cuda::{ctx_sync, UnifiedMemory};
use symbi_xpu::runtime::cuda_runtime::current_dispatcher;
use symbi_xpu::runtime::GpuRuntime;
use symbi_xpu::{with_device, KernelArgs, LaunchConfig, MemoryBlock};

const FILL: &str = r#"
extern "C" __global__ void fill(double* out, double val, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = val;
}
"#;

// allocate a unified buffer in the CURRENT context, fill it via a kernel launched through
// the CURRENT device's dispatcher, sync, and read cell 0 back.
fn fill_on_current_device(val: f64) -> f64 {
    let n = 64usize;
    let mut buf = MemoryBlock::<UnifiedMemory>::for_elements::<f64>(n).expect("alloc");
    let ptr = buf.as_mut_ptr::<f64>() as u64;
    let n_u32 = n as u32;

    let kernel = current_dispatcher().jit_kernel_keyed(FILL, "test/fill", "fill");
    let mut args = KernelArgs::new();
    args.push(&ptr);
    args.push(&val);
    args.push(&n_u32);
    let config = LaunchConfig::for_1d(n_u32, 64);
    unsafe {
        current_dispatcher()
            .runtime()
            .launch(&kernel, config, args.as_mut_slice())
            .expect("fill launch");
    }
    ctx_sync();
    unsafe { *buf.as_ptr::<f64>() }
}

#[test]
fn kernels_run_on_distinct_logical_devices() {
    // each runs in its own context (device 1 is a second context on the one card).
    let a = with_device(0, || fill_on_current_device(3.0));
    let b = with_device(1, || fill_on_current_device(7.0));
    assert_eq!(a, 3.0, "device 0 kernel produced wrong value");
    assert_eq!(b, 7.0, "device 1 kernel produced wrong value");

    // back to device 0 after the switch: confirms `with_device` restores the prior context
    // and device 0's module/dispatcher is still valid.
    let c = with_device(0, || fill_on_current_device(5.0));
    assert_eq!(c, 5.0, "device 0 kernel after a device switch produced wrong value");
}
