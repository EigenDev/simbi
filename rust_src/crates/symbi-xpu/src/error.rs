// =============================================================================
// error.rs
//
// error type for xpu operations. wraps vendor error codes with context.
// all fallible xpu operations return Result<T, XpuError>.
//
// usage:
//   let stream = CudaSpace::create_stream(0)?;
//   let block = MemoryBlock::<UnifiedMemory>::try_new(1024)?;
// =============================================================================

use std::fmt;

/// error from an xpu operation.
#[derive(Debug)]
pub struct XpuError {
    /// which operation failed.
    pub operation: &'static str,
    /// vendor-specific error code (e.g. CUresult for CUDA).
    pub code: i32,
    /// human-readable detail.
    pub detail: String,
}

impl fmt::Display for XpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "xpu: {} failed (error code {})", self.operation, self.code)?;
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        Ok(())
    }
}

impl std::error::Error for XpuError {}

impl XpuError {
    pub fn new(operation: &'static str, code: i32) -> Self {
        XpuError {
            operation,
            code,
            detail: cuda_error_name(code),
        }
    }
}

/// convenience alias.
pub type Result<T> = std::result::Result<T, XpuError>;

/// map common CUDA error codes to names.
fn cuda_error_name(code: i32) -> String {
    match code {
        0 => "CUDA_SUCCESS".into(),
        1 => "CUDA_ERROR_INVALID_VALUE".into(),
        2 => "CUDA_ERROR_OUT_OF_MEMORY".into(),
        3 => "CUDA_ERROR_NOT_INITIALIZED".into(),
        100 => "CUDA_ERROR_NO_DEVICE".into(),
        101 => "CUDA_ERROR_INVALID_DEVICE".into(),
        200 => "CUDA_ERROR_INVALID_IMAGE".into(),
        201 => "CUDA_ERROR_INVALID_CONTEXT".into(),
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU".into(),
        214 => "CUDA_ERROR_ECC_UNCORRECTABLE".into(),
        301 => "CUDA_ERROR_FILE_NOT_FOUND".into(),
        302 => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND".into(),
        400 => "CUDA_ERROR_INVALID_HANDLE".into(),
        500 => "CUDA_ERROR_NOT_FOUND".into(),
        700 => "CUDA_ERROR_LAUNCH_FAILED".into(),
        701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES".into(),
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT".into(),
        719 => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING".into(),
        _ => format!("CUDA_ERROR_UNKNOWN({})", code),
    }
}
