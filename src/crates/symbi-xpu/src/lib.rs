// =============================================================================
// symbi-xpu
//
// execution and memory abstraction for heterogeneous computing.
// answers three questions: where does data live, how does work execute,
// how to wait for work.
//
// all fallible operations return Result<T, XpuError>; production paths
// propagate errors to the caller.
//
// the crate's rules:
//   - backend-agnostic plumbing (compile-time dispatch, no dyn)
//   - manages memory lifetime, leaving layout to callers
//   - stream-ordered execution (executor owns a stream)
//   - loads pre-compiled kernels; code generation lives elsewhere
//   - explicit transfers
//   - stateless except for the executor
//   - leaf crate (no symbi dependencies)
// =============================================================================

mod args;
mod config;
pub mod error;
mod executor;
mod handle;
mod memory;
pub mod runtime;
mod space;
mod token;
// nvcc AOT compilation (CUDA source -> PTX). pure std (shells nvcc); always
// available so symbi/build.rs can share the host-compiler probe independent of cuda.
pub mod compile_cuda;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub mod nvrtc;

// the amd backend. sibling to cuda behind the same traits.
#[cfg(feature = "hip")]
pub mod hip;

#[cfg(feature = "hip")]
pub mod hiprtc;

// exactly one concrete gpu backend at a time. the `gpu` umbrella feature (implied by both)
// gates backend-agnostic device code; `cuda` / `hip` select the concrete backend.
#[cfg(all(feature = "cuda", feature = "hip"))]
compile_error!("features `cuda` and `hip` are mutually exclusive: enable exactly one gpu backend");

// traits
pub use memory::{HostMemory, MemoryBlock, MemorySpace};
pub use space::ExecutionSpace;

// types
pub use args::{KernelArgs, with_pooled_args};
pub use config::{LaunchConfig, block_for, extent_aware_block};
pub use error::{Result, XpuError};
pub use executor::Executor;
pub use handle::SharedHandle;
pub use token::Token;

// the active gpu backend module, aliased neutrally so the crate's neutral surface binds to it
// without naming cuda/hip. cuda wins if both are somehow set (the hip arm is `not(cuda)`).
#[cfg(feature = "cuda")]
use cuda as gpu_backend;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
use hip as gpu_backend;

// neutral device api (gpu = cuda || hip): downstream calls `symbi_xpu::ctx_sync`,
// `symbi_xpu::memcpy_peer`, etc. without naming a backend module.
#[cfg(feature = "gpu")]
pub use gpu_backend::{
    DeviceInfo, MAX_GPUS, can_access_peer, ctx_sync, current_device, device_count, device_info,
    enable_peer_access, memcpy_peer,
};

// neutral device space/memory aliases. the concrete `CudaSpace`/`UnifiedMemory` (and the hip
// names) stay exported for backend-specific tests; downstream code names `DeviceSpace`/
// `DeviceMemory`, which resolve to whichever backend is compiled in.
#[cfg(feature = "cuda")]
pub use cuda::{CudaSpace, UnifiedMemory};
#[cfg(feature = "cuda")]
pub use cuda::{CudaSpace as DeviceSpace, UnifiedMemory as DeviceMemory};
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub use hip::{HipManaged, HipSpace};
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub use hip::{HipManaged as DeviceMemory, HipSpace as DeviceSpace};

/// run `f` with gpu device `ord` bound on this thread, restoring the previous device after. on
/// a host (no-gpu) build this is just `f()` -- device binding is a no-op there, so callers can
/// wrap a tile's work uniformly regardless of backend.
#[cfg(feature = "gpu")]
pub fn with_device<R>(ord: i32, f: impl FnOnce() -> R) -> R {
    gpu_backend::with_device(ord, f)
}
#[cfg(not(feature = "gpu"))]
pub fn with_device<R>(_ord: i32, f: impl FnOnce() -> R) -> R {
    f()
}

// default space selection: any gpu backend -> the device space/memory; else cpu/host.
#[cfg(feature = "gpu")]
pub type DefaultSpace = DeviceSpace;

#[cfg(not(feature = "gpu"))]
pub type DefaultSpace = CpuSpace;

#[cfg(feature = "gpu")]
pub type DefaultMemory = DeviceMemory;

#[cfg(not(feature = "gpu"))]
pub type DefaultMemory = HostMemory;

/// cpu execution space. always available.
pub struct CpuSpace;

impl ExecutionSpace for CpuSpace {
    type Stream = ();
    type Event = ();
    type Module = ();
    type Kernel = ();

    const IS_HOST: bool = true;
    const IS_DEVICE: bool = false;
    const SUPPORTS_ASYNC: bool = false;

    fn create_stream(_device_id: i64) -> error::Result<()> {
        Ok(())
    }
    fn destroy_stream(_stream: &mut ()) {}
    fn sync_stream(_stream: &()) -> error::Result<()> {
        Ok(())
    }
    fn stream_ready(_stream: &()) -> error::Result<bool> {
        Ok(true)
    }

    fn create_event() -> error::Result<()> {
        Ok(())
    }
    fn destroy_event(_event: ()) {}
    fn record_event(_event: &(), _stream: &()) -> error::Result<()> {
        Ok(())
    }
    fn event_ready(_event: &()) -> error::Result<bool> {
        Ok(true)
    }
    fn sync_event(_event: &()) -> error::Result<()> {
        Ok(())
    }
    fn stream_wait_event(_stream: &(), _event: &()) -> error::Result<()> {
        Ok(())
    }

    fn load_module(_bytes: &[u8]) -> error::Result<()> {
        Ok(())
    }
    fn get_function(_module: &(), _name: &str) -> error::Result<()> {
        Ok(())
    }

    unsafe fn launch(
        _stream: &(),
        _kernel: &(),
        _config: LaunchConfig,
        _args: &mut [*mut std::ffi::c_void],
    ) -> error::Result<()> {
        Ok(())
    }
}
