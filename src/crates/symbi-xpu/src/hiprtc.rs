// =============================================================================
// hiprtc.rs
//
// hipRTC: AMD's runtime HIP compiler (libhiprtc.so, often folded into libamdhip64).
// compiles a HIP/cuda-c++ source string straight to a CODE OBJECT in-process -- the
// hip analog of nvrtc. same rule: every
// accelerator compiles at runtime via its OWN runtime compiler, no shelled toolchain.
//
// unlike nvrtc (source -> virtual ptx, re-jitted by the driver at load), hiprtc emits
// a FINAL code object for a specific gpu arch. so the target arch matters: it comes
// from `SYMBI_HIP_ARCH` (e.g. `gfx90a`, `gfx942`) when set, else hiprtc defaults to the
// attached device. set the env var on the cluster only if auto-detect is insufficient
// (cross-arch build, or a driver that does not default).
//
// link: -lhiprtc (set in build.rs under the hip feature).
// =============================================================================

#![allow(non_camel_case_types)]

use crate::error::{self, XpuError};
use std::ffi::{c_char, c_int, c_void, CString};

type hiprtcResult = c_int;
type hiprtcProgram = *mut c_void;

const HIPRTC_SUCCESS: hiprtcResult = 0;

unsafe extern "C" {
    fn hiprtcCreateProgram(
        prog: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> hiprtcResult;
    fn hiprtcCompileProgram(
        prog: hiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> hiprtcResult;
    fn hiprtcGetCodeSize(prog: hiprtcProgram, size: *mut usize) -> hiprtcResult;
    fn hiprtcGetCode(prog: hiprtcProgram, code: *mut c_char) -> hiprtcResult;
    fn hiprtcGetProgramLogSize(prog: hiprtcProgram, size: *mut usize) -> hiprtcResult;
    fn hiprtcGetProgramLog(prog: hiprtcProgram, log: *mut c_char) -> hiprtcResult;
    fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> hiprtcResult;
}

/// compile a HIP/cuda-c++ source string to a code object via hipRTC. on a compile error, the
/// hipRTC program log (the actual compiler diagnostics) is returned in the error detail. the
/// returned bytes are a binary code object -- `HipSpace::load_module` loads them as-is.
pub fn compile_code(source: &str, kernel_name: &str) -> error::Result<Vec<u8>> {
    // optional `--offload-arch=<arch>`; unset -> hiprtc targets the attached device.
    let arch_opt = std::env::var("SYMBI_HIP_ARCH")
        .ok()
        .map(|a| CString::new(format!("--offload-arch={a}")).expect("arch option has no interior NUL"));

    let src = CString::new(source).map_err(|_| XpuError {
        operation: "hiprtc source",
        code: -1,
        detail: "kernel source has an interior NUL".into(),
    })?;
    let name = CString::new(format!("{kernel_name}.hip")).map_err(|_| XpuError {
        operation: "hiprtc name",
        code: -1,
        detail: "kernel name has an interior NUL".into(),
    })?;

    let mut prog: hiprtcProgram = std::ptr::null_mut();
    check(
        unsafe {
            hiprtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null())
        },
        "hiprtcCreateProgram",
    )?;

    let opts: Vec<*const c_char> = arch_opt.iter().map(|c| c.as_ptr()).collect();
    let res = unsafe {
        hiprtcCompileProgram(
            prog,
            opts.len() as c_int,
            if opts.is_empty() { std::ptr::null() } else { opts.as_ptr() },
        )
    };
    if res != HIPRTC_SUCCESS {
        let log = program_log(prog);
        unsafe {
            let _ = hiprtcDestroyProgram(&mut prog);
        }
        return Err(XpuError {
            operation: "hiprtcCompileProgram",
            code: res,
            detail: format!("hipRTC failed to compile '{kernel_name}':\n{log}"),
        });
    }

    let mut size: usize = 0;
    check(unsafe { hiprtcGetCodeSize(prog, &mut size) }, "hiprtcGetCodeSize")?;
    let mut code = vec![0u8; size];
    check(unsafe { hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char) }, "hiprtcGetCode")?;
    unsafe {
        let _ = hiprtcDestroyProgram(&mut prog);
    }
    Ok(code)
}

/// read the hipRTC program log (compiler diagnostics) for a failed compile.
fn program_log(prog: hiprtcProgram) -> String {
    let mut size: usize = 0;
    if unsafe { hiprtcGetProgramLogSize(prog, &mut size) } != HIPRTC_SUCCESS || size <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    if unsafe { hiprtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char) } != HIPRTC_SUCCESS {
        return String::new();
    }
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn check(res: hiprtcResult, op: &'static str) -> error::Result<()> {
    if res == HIPRTC_SUCCESS {
        Ok(())
    } else {
        Err(XpuError::new(op, res))
    }
}
