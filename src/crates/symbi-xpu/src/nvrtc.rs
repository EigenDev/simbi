// =============================================================================
// nvrtc.rs
//
// NVRTC: NVIDIA's runtime CUDA compiler (libnvrtc.so). compiles a CUDA C++
// source string straight to PTX IN-PROCESS — no `nvcc` binary, no host C++
// compiler, no temp files (docs/design/15 §1: every accelerator compiles at
// runtime via its own runtime compiler). this is what lets `Sim<UnifiedMemory>`
// JIT + run a substrate kernel on the GPU without the nvcc toolchain (the host
// gcc-16 breaks nvcc; NVRTC ships its own front-end, so it does not care).
//
// the driver still JITs the PTX to SASS at module load, so compilation targets the device's
// VIRTUAL arch (`compute_<major><minor>`, queried from the driver) and let the
// driver finish the lowering for whatever GPU is present.
//
// link: -lnvrtc (set in build.rs under the cuda feature).
// =============================================================================

#![allow(non_camel_case_types)]

use crate::error::{self, XpuError};
use std::ffi::{c_char, c_int, c_void, CString};

type nvrtcResult = c_int;
type nvrtcProgram = *mut c_void;

const NVRTC_SUCCESS: nvrtcResult = 0;

unsafe extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> nvrtcResult;
    fn nvrtcCompileProgram(prog: nvrtcProgram, num_options: c_int, options: *const *const c_char) -> nvrtcResult;
    fn nvrtcGetPTXSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    fn nvrtcGetProgramLogSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
}

/// compile a CUDA C++ source string to PTX bytes via NVRTC. targets the present
/// device's `compute_<major><minor>` virtual arch. on a compile error, the NVRTC
/// program log (the actual compiler diagnostics) is returned in the error detail.
/// the returned bytes are NUL-terminated — `CudaSpace::load_module` accepts that.
pub fn compile_ptx(source: &str, kernel_name: &str) -> error::Result<Vec<u8>> {
    let (major, minor) = crate::cuda::device_compute_capability()?;
    let arch = CString::new(format!("--gpu-architecture=compute_{major}{minor}"))
        .expect("arch option has no interior NUL");
    // FMA fusion: nvcc's default `a*b + c -> fma(a,b,c)` stays ON
    // ([[project_fma_discipline]]): trust the compiler, accept ULP-bounded drift
    // vs the CPU (which doesn't auto-fuse). do NOT re-introduce `--fmad=false`.
    // -O3 is NOT a recognized NVRTC option on the current driver; the PTX -> SASS
    // JIT done at module load by the CUDA driver already optimizes at -O3, so
    // there is no perf left on the table here. do NOT add it back without
    // confirming the NVRTC version actually accepts it.

    let src = CString::new(source)
        .map_err(|_| XpuError { operation: "nvrtc source", code: -1, detail: "kernel source has an interior NUL".into() })?;
    let name = CString::new(format!("{kernel_name}.cu"))
        .map_err(|_| XpuError { operation: "nvrtc name", code: -1, detail: "kernel name has an interior NUL".into() })?;

    let mut prog: nvrtcProgram = std::ptr::null_mut();
    check(
        unsafe {
            nvrtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null())
        },
        "nvrtcCreateProgram",
    )?;

    let opts: [*const c_char; 1] = [arch.as_ptr()];
    let res = unsafe { nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr()) };
    if res != NVRTC_SUCCESS {
        let log = program_log(prog);
        unsafe { let _ = nvrtcDestroyProgram(&mut prog); }
        return Err(XpuError {
            operation: "nvrtcCompileProgram",
            code: res,
            detail: format!("NVRTC failed to compile '{kernel_name}':\n{log}"),
        });
    }

    let mut size: usize = 0;
    check(unsafe { nvrtcGetPTXSize(prog, &mut size) }, "nvrtcGetPTXSize")?;
    let mut ptx = vec![0u8; size];
    check(unsafe { nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char) }, "nvrtcGetPTX")?;
    unsafe { let _ = nvrtcDestroyProgram(&mut prog); }
    Ok(ptx)
}

/// read the NVRTC program log (compiler diagnostics) for a failed compile.
fn program_log(prog: nvrtcProgram) -> String {
    let mut size: usize = 0;
    if unsafe { nvrtcGetProgramLogSize(prog, &mut size) } != NVRTC_SUCCESS || size <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    if unsafe { nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char) } != NVRTC_SUCCESS {
        return String::new();
    }
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn check(res: nvrtcResult, op: &'static str) -> error::Result<()> {
    if res == NVRTC_SUCCESS {
        Ok(())
    } else {
        Err(XpuError::new(op, res))
    }
}
