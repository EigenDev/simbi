// =============================================================================
// executor.rs
//
// the executor owns a stream and launches pre-compiled kernels.
// move-only. RAII — drop destroys the stream.
// all fallible operations return Result<T, XpuError>.
// =============================================================================

use crate::space::ExecutionSpace;
use crate::args::KernelArgs;
use crate::config::LaunchConfig;
use crate::token::Token;
use crate::error::Result;

/// owns a stream, launches kernels, synchronizes. move-only.
pub struct Executor<S: ExecutionSpace> {
    stream: S::Stream,
    device_id: i64,
}

impl<S: ExecutionSpace> Executor<S> {
    /// create a new executor on the given device.
    pub fn new(device_id: i64) -> Result<Self> {
        let stream = S::create_stream(device_id)?;
        Ok(Executor { stream, device_id })
    }

    /// launch a pre-compiled kernel on this executor's stream.
    ///
    /// # safety
    /// the kernel args must match the kernel's signature.
    pub unsafe fn launch(
        &self,
        kernel: &S::Kernel,
        config: LaunchConfig,
        args: &mut KernelArgs,
    ) -> Result<Token<S>> {
        unsafe { S::launch(&self.stream, kernel, config, args.as_mut_slice())?; }
        let mut token = Token::create()?;
        token.record(&self.stream)?;
        Ok(token)
    }

    /// block until all work on this stream completes.
    pub fn sync(&self) -> Result<()> {
        S::sync_stream(&self.stream)
    }

    /// non-blocking query: is all submitted work done?
    pub fn ready(&self) -> Result<bool> {
        S::stream_ready(&self.stream)
    }

    /// raw stream handle.
    pub fn stream(&self) -> &S::Stream {
        &self.stream
    }

    /// which device this executor runs on.
    pub fn device_id(&self) -> i64 {
        self.device_id
    }
}

impl<S: ExecutionSpace> Drop for Executor<S> {
    fn drop(&mut self) {
        // best-effort sync before destroying stream
        let _ = S::sync_stream(&self.stream);
        S::destroy_stream(&mut self.stream);
    }
}
