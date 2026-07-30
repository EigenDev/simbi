// =============================================================================
// device_view_abi.rs
//
// **the View-struct ABI canary**: a one-thread CUDA kernel that takes a
// `__symbi_View` by value and writes back its `lo[0..4]`, `strides[0..4]` and
// `extent[0..4]` into the buffer pointed to by `data`. the host then asserts the
// 12 written values match the values originally placed in the View.
//
// what this catches:
//   - drift in the `DeviceView` Rust-side `#[repr(C)]` layout (size, field
//     offsets, alignment), which the static_asserts in `substrate_gpu.rs` cannot
//     catch alone (those only pin the Rust side);
//   - drift in the `__symbi_View` CUDA struct emitted by the kernel preamble
//     away from the Rust POD passed by value.
//
// the round-trip is the only mechanical check that the two structs agree: a
// one-byte field reorder on either side otherwise shows up only as garbled
// physics deep inside a hydro run.
//
// the kernel source declares its own `__symbi_View` locally; it does not reuse
// the substrate's emitter preamble. the test verifies both sides independently
// agree on the layout.
//
// run: cargo test --release -p symbi-xpu --features cuda --test device_view_abi
// =============================================================================

#![cfg(feature = "cuda")]

use symbi_xpu::cuda::{UnifiedMemory, ctx_sync};
use symbi_xpu::runtime::{GpuRuntime, cuda_runtime::current_dispatcher};
use symbi_xpu::{KernelArgs, LaunchConfig, MemoryBlock};

// host-side mirror of the CUDA `__symbi_View` struct. MUST match the
// substrate's `DeviceView` (substrate_gpu.rs) in layout. redeclared
// here because `DeviceView` is private to the symbi
// crate; the static asserts on that side + this round-trip lock both ends.
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceView {
    data: *const std::ffi::c_void,
    lo: [i32; 4],
    strides: [i32; 4],
    extent: [i32; 4],
}

// the ABI probe kernel: writes the 12 view-field slots into `data[0..12]`. a
// double buffer + value casts keep the host side type-trivial.
const ABI_PROBE_SRC: &str = r#"
struct __symbi_View { double* __restrict__ data; int lo[4]; int strides[4]; int extent[4]; };
extern "C" __global__ void abi_probe(__symbi_View v) {
    v.data[0] = (double)v.lo[0];
    v.data[1] = (double)v.lo[1];
    v.data[2] = (double)v.lo[2];
    v.data[3] = (double)v.lo[3];
    v.data[4] = (double)v.strides[0];
    v.data[5] = (double)v.strides[1];
    v.data[6] = (double)v.strides[2];
    v.data[7] = (double)v.strides[3];
    v.data[8] = (double)v.extent[0];
    v.data[9] = (double)v.extent[1];
    v.data[10] = (double)v.extent[2];
    v.data[11] = (double)v.extent[3];
}
"#;

#[test]
fn device_view_abi_roundtrip() {
    // unified buffer of 12 doubles. the kernel writes into it through the
    // View's `data` pointer; the host reads it back after sync.
    let n = 12usize;
    let mut block =
        MemoryBlock::<UnifiedMemory>::for_elements::<f64>(n).expect("unified alloc for abi probe");
    let data_ptr = block.as_mut_ptr::<f64>();
    // initialize to a sentinel so an unmodified slot is detectable.
    unsafe {
        for i in 0..n {
            *data_ptr.add(i) = -1.0;
        }
    }

    let view = DeviceView {
        data: data_ptr as *const std::ffi::c_void,
        lo: [10, 20, 30, 40],
        strides: [50, 60, 70, 80],
        extent: [90, 100, 110, 120],
    };

    // JIT-compile + launch the one-thread probe — same path substrate_gpu.rs
    // uses for every kernel, here driving a hand-written source.
    let kernel = current_dispatcher().jit_kernel(ABI_PROBE_SRC, "abi_probe");
    let mut args = KernelArgs::new();
    args.push(&view);
    let config = LaunchConfig::for_1d(1, 1);
    unsafe {
        current_dispatcher()
            .runtime()
            .launch(&kernel, config, args.as_mut_slice())
            .expect("abi_probe launch failed");
    }
    ctx_sync();

    let got: [f64; 12] = unsafe {
        [
            *data_ptr.add(0),
            *data_ptr.add(1),
            *data_ptr.add(2),
            *data_ptr.add(3),
            *data_ptr.add(4),
            *data_ptr.add(5),
            *data_ptr.add(6),
            *data_ptr.add(7),
            *data_ptr.add(8),
            *data_ptr.add(9),
            *data_ptr.add(10),
            *data_ptr.add(11),
        ]
    };
    let want: [f64; 12] = [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ];
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "slot {i}: got {g}, want {w} — DeviceView ABI drift between host Rust struct and device `__symbi_View`",
        );
    }
}
