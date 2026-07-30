// =============================================================================
// space.rs
//
// the execution space trait. defines HOW work runs on a backend.
// all fallible operations return Result<T, XpuError>.
// =============================================================================

use crate::config::LaunchConfig;
use crate::error::Result;
use std::ffi::c_void;

/// the execution space defines how work runs on a backend.
/// implement this for each backend (CPU, CUDA, Metal, HIP).
pub trait ExecutionSpace: 'static + Send + Sync + Sized {
    type Stream: Send;
    type Event: Send;
    type Module: Send + Sync;
    type Kernel: Send + Sync + Copy;

    const IS_HOST: bool;
    const IS_DEVICE: bool;
    const SUPPORTS_ASYNC: bool;

    // ---- stream lifecycle ----
    fn create_stream(device_id: i64) -> Result<Self::Stream>;
    fn destroy_stream(stream: &mut Self::Stream);
    fn sync_stream(stream: &Self::Stream) -> Result<()>;
    fn stream_ready(stream: &Self::Stream) -> Result<bool>;

    // ---- event lifecycle ----
    fn create_event() -> Result<Self::Event>;
    fn destroy_event(event: Self::Event);
    fn record_event(event: &Self::Event, stream: &Self::Stream) -> Result<()>;
    fn event_ready(event: &Self::Event) -> Result<bool>;
    fn sync_event(event: &Self::Event) -> Result<()>;
    fn stream_wait_event(stream: &Self::Stream, event: &Self::Event) -> Result<()>;

    // ---- module / kernel management ----
    fn load_module(bytes: &[u8]) -> Result<Self::Module>;
    fn get_function(module: &Self::Module, name: &str) -> Result<Self::Kernel>;

    /// # safety
    /// arg pointers must match the kernel signature.
    unsafe fn launch(
        stream: &Self::Stream,
        kernel: &Self::Kernel,
        config: LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<()>;
}
