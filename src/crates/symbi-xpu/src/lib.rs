// =============================================================================
// symbi-xpu
//
// execution and memory abstraction for heterogeneous computing.
// answers three questions: where does data live, how does work execute,
// how do you wait for work.
//
// all fallible operations return Result<T, XpuError>. no panics in
// production paths — errors propagate to the caller.
//
// seven rules:
//   1. backend-agnostic plumbing (compile-time dispatch, no dyn)
//   2. manages memory lifetime, not layout
//   3. stream-ordered execution (executor owns a stream)
//   4. does not generate code (loads pre-compiled kernels)
//   5. explicit transfers
//   6. stateless except for the executor
//   7. leaf crate (no symbi dependencies)
// =============================================================================

mod space;
mod memory;
mod handle;
mod executor;
mod token;
mod args;
mod config;
pub mod error;
pub mod runtime;
// nvcc AOT compilation (CUDA source -> PTX). pure std (shells nvcc); always
// available so symbi/build.rs can share the host-compiler probe without cuda.
pub mod compile_cuda;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub mod nvrtc;

// traits
pub use space::ExecutionSpace;
pub use memory::{MemorySpace, MemoryBlock, HostMemory};

// types
pub use handle::SharedHandle;
pub use executor::Executor;
pub use token::Token;
pub use args::{KernelArgs, with_pooled_args};
pub use config::{LaunchConfig, block_dims, block_for, extent_aware_block};
pub use error::{XpuError, Result};

// conditional re-exports
#[cfg(feature = "cuda")]
pub use cuda::{CudaSpace, UnifiedMemory};

/// run `f` with gpu device `ord` bound on this thread (its context current), restoring the
/// previous device after. on a host (non-cuda) build this is just `f()` -- device binding is
/// a no-op there, so callers can wrap a tile's work uniformly regardless of backend
/// (docs/design/37).
#[cfg(feature = "cuda")]
pub fn with_device<R>(ord: i32, f: impl FnOnce() -> R) -> R {
    cuda::with_device(ord, f)
}
#[cfg(not(feature = "cuda"))]
pub fn with_device<R>(_ord: i32, f: impl FnOnce() -> R) -> R {
    f()
}

// default space selection
#[cfg(feature = "cuda")]
pub type DefaultSpace = CudaSpace;

#[cfg(not(feature = "cuda"))]
pub type DefaultSpace = CpuSpace;

#[cfg(feature = "cuda")]
pub type DefaultMemory = UnifiedMemory;

#[cfg(not(feature = "cuda"))]
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

    fn create_stream(_device_id: i64) -> error::Result<()> { Ok(()) }
    fn destroy_stream(_stream: &mut ()) {}
    fn sync_stream(_stream: &()) -> error::Result<()> { Ok(()) }
    fn stream_ready(_stream: &()) -> error::Result<bool> { Ok(true) }

    fn create_event() -> error::Result<()> { Ok(()) }
    fn destroy_event(_event: ()) {}
    fn record_event(_event: &(), _stream: &()) -> error::Result<()> { Ok(()) }
    fn event_ready(_event: &()) -> error::Result<bool> { Ok(true) }
    fn sync_event(_event: &()) -> error::Result<()> { Ok(()) }
    fn stream_wait_event(_stream: &(), _event: &()) -> error::Result<()> { Ok(()) }

    fn load_module(_bytes: &[u8]) -> error::Result<()> { Ok(()) }
    fn get_function(_module: &(), _name: &str) -> error::Result<()> { Ok(()) }

    unsafe fn launch(
        _stream: &(),
        _kernel: &(),
        _config: LaunchConfig,
        _args: &mut [*mut std::ffi::c_void],
    ) -> error::Result<()> {
        Ok(())
    }
}
