// =============================================================================
// cuda.rs
//
// CUDA execution space and unified memory space. raw driver API bindings.
// no external crate dependencies — just extern "C" to libcuda.so.
//
// binds only what we need:
//   cuInit, cuDeviceGet, cuCtxCreate, cuCtxGetCurrent,
//   cuStreamCreate, cuStreamDestroy, cuStreamSynchronize, cuStreamQuery,
//   cuEventCreate, cuEventDestroy, cuEventRecord, cuEventQuery,
//   cuEventSynchronize, cuStreamWaitEvent,
//   cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel,
//   cuMemAllocManaged, cuMemFree, cuMemsetD8
//
// link: -lcuda (set in Cargo.toml build script or link attribute)
// =============================================================================

#![allow(non_camel_case_types)]

use crate::config::LaunchConfig;
use crate::error::{self, XpuError};
use crate::memory::MemorySpace;
use crate::space::ExecutionSpace;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

// =============================================================================
// raw CUDA driver API types
// =============================================================================

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
const CU_MEM_ATTACH_GLOBAL: c_uint = 1;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const CU_STREAM_NON_BLOCKING: c_uint = 1;

// =============================================================================
// raw CUDA driver API bindings
// =============================================================================

unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    fn cuStreamQuery(stream: CUstream) -> CUresult;
    fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    fn cuEventQuery(event: CUevent) -> CUresult;
    fn cuEventSynchronize(event: CUevent) -> CUresult;
    fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuMemAllocManaged(dptr: *mut CUdeviceptr, size: usize, flags: c_uint) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
}

// CUdevice_attribute enum values (cuda.h): the SM compute capability of a device.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// the (major, minor) compute capability of device 0, e.g., (7, 5) for an RTX 2070.
/// NVRTC needs it to pick `--gpu-architecture=compute_<major><minor>` so the PTX it
/// emits matches the GPU the driver will JIT it onto. queried from the driver, not
/// hardcoded — device-agnostic by design.
pub fn device_compute_capability() -> error::Result<(i32, i32)> {
    ensure_init()?;
    let mut dev: CUdevice = 0;
    unsafe {
        check(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
    }
    let (mut major, mut minor): (c_int, c_int) = (0, 0);
    unsafe {
        check(
            cuDeviceGetAttribute(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            ),
            "cuDeviceGetAttribute(major)",
        )?;
        check(
            cuDeviceGetAttribute(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            ),
            "cuDeviceGetAttribute(minor)",
        )?;
    }
    Ok((major, minor))
}

/// **B10 — device info** for the live progress table. queries device 0 (the
/// one symbi launches kernels on) for its name, total VRAM in bytes, and
/// compute capability. used by `symbi-display::system_info_rows` to populate
/// the System Info table at the top of every run.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub total_memory_bytes: u64,
    pub compute_capability: (i32, i32),
    pub device_count: i32,
}

pub fn device_info() -> error::Result<DeviceInfo> {
    ensure_init()?;
    let mut count: c_int = 0;
    unsafe {
        check(cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
    }
    let mut dev: CUdevice = 0;
    unsafe {
        check(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
    }
    let mut name_buf = [0_i8; 256];
    unsafe {
        check(
            cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as c_int, dev),
            "cuDeviceGetName",
        )?;
    }
    let name_bytes: Vec<u8> = name_buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    let name = String::from_utf8_lossy(&name_bytes).into_owned();
    let mut total_bytes: usize = 0;
    unsafe {
        check(
            cuDeviceTotalMem_v2(&mut total_bytes, dev),
            "cuDeviceTotalMem_v2",
        )?;
    }
    let (major, minor) = device_compute_capability()?;
    Ok(DeviceInfo {
        name,
        total_memory_bytes: total_bytes as u64,
        compute_capability: (major, minor),
        device_count: count,
    })
}

/// block until all outstanding work on the current CUDA context finishes.
/// panics on driver error — the caller is in dispatch code that cannot recover.
pub fn ctx_sync() {
    let res = unsafe { cuCtxSynchronize() };
    if res != CUDA_SUCCESS {
        panic!("cuCtxSynchronize failed: error {}", res);
    }
}

fn check(res: CUresult, op: &'static str) -> error::Result<()> {
    if res == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(XpuError::new(op, res))
    }
}

// =============================================================================
// CUDA initialization: a per-device context registry (docs/design/37)
// =============================================================================

unsafe extern "C" {
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
}

#[derive(Clone, Copy)]
struct SyncCtx(CUcontext);
unsafe impl Send for SyncCtx {}
unsafe impl Sync for SyncCtx {}

/// max gpus per node we bind. gpu counts are tiny; a fixed array avoids locking on the
/// hot path. raise if a node ever has more.
pub const MAX_GPUS: usize = 16;

// one lazily-created context per device ordinal. cuda modules and allocations are bound to
// the context they were made in, so each device gets its own (and its own dispatcher, see
// runtime.rs).
static CUDA_CTX: [OnceLock<SyncCtx>; MAX_GPUS] = [const { OnceLock::new() }; MAX_GPUS];

thread_local! {
    // the device whose context is current on THIS thread. cuda's current-context is
    // per-thread, so the "current device" is too. defaults to 0 (the single-device path).
    static CURRENT_DEVICE: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// the device ordinal whose context is current on this thread.
pub fn current_device() -> i32 {
    CURRENT_DEVICE.with(|c| c.get())
}

/// number of cuda devices visible to the process.
pub fn device_count() -> error::Result<i32> {
    let mut count: c_int = 0;
    unsafe {
        check(cuInit(0), "cuInit")?;
        check(cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
    }
    Ok(count)
}

/// ensure device `ord`'s context exists and is current on this thread; returns it.
fn ensure_init_device(ord: i32) -> error::Result<CUcontext> {
    assert!((ord as usize) < MAX_GPUS, "device ordinal {ord} exceeds MAX_GPUS");
    let ctx = CUDA_CTX[ord as usize]
        .get_or_init(|| {
            unsafe {
                // these panics are acceptable — init happens once per device at startup.
                check(cuInit(0), "cuInit").expect("cuInit failed");
                let mut ctx: CUcontext = std::ptr::null_mut();
                if ord == 0 {
                    // preserve the embed-friendly reuse of an already-current context.
                    let _ = cuCtxGetCurrent(&mut ctx);
                }
                if ctx.is_null() {
                    // logical ordinals round-robin onto the physical devices: identity when
                    // there are at least as many gpus as logical ids (the production case),
                    // and wrapping otherwise -- so N logical devices run as N distinct contexts
                    // on a single card, which is how the multi-device path is validated without
                    // a second gpu (docs/design/37).
                    let mut count: c_int = 0;
                    check(cuDeviceGetCount(&mut count), "cuDeviceGetCount")
                        .expect("cuDeviceGetCount failed");
                    let physical = if count > 0 { ord % count } else { 0 };
                    let mut dev: CUdevice = 0;
                    check(cuDeviceGet(&mut dev, physical), "cuDeviceGet").expect("cuDeviceGet failed");
                    check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")
                        .expect("cuCtxCreate failed");
                }
                SyncCtx(ctx)
            }
        })
        .0;
    unsafe { check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")? };
    Ok(ctx)
}

/// ensure CUDA is initialized and the CURRENT device's context is current on this thread.
/// the single-device path (current device defaults to 0) is behaviorally unchanged.
fn ensure_init() -> error::Result<()> {
    ensure_init_device(current_device()).map(|_| ())
}

/// run `f` with device `ord`'s context current on this thread, restoring the previous
/// device afterward. this binds a tile's kernels to its gpu (docs/design/37): the launch /
/// alloc / sync code targets "the current context", so we make the right one current rather
/// than threading a device id through every signature.
pub fn with_device<R>(ord: i32, f: impl FnOnce() -> R) -> R {
    let prev = current_device();
    ensure_init_device(ord).expect("with_device: ensure_init_device");
    CURRENT_DEVICE.with(|c| c.set(ord));
    let r = f();
    ensure_init_device(prev).expect("with_device: restore device");
    CURRENT_DEVICE.with(|c| c.set(prev));
    r
}

// =============================================================================
// handle wrappers (Send-safe)
// =============================================================================

/// CUDA stream handle.
pub struct CudaStream(CUstream);
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// the default (null) CUDA stream. synchronous with the context.
    /// used by kernel macro dispatch when no explicit stream is available.
    pub fn null() -> Self {
        CudaStream(std::ptr::null_mut())
    }
}

/// CUDA event handle.
pub struct CudaEvent(CUevent);
unsafe impl Send for CudaEvent {}

/// CUDA module handle.
pub struct CudaModule(CUmodule);
unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

/// CUDA kernel handle. Copy because it's just a function pointer.
#[derive(Clone, Copy, Debug)]
pub struct CudaKernel(CUfunction);
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

// =============================================================================
// CudaSpace: ExecutionSpace implementation
// =============================================================================

pub struct CudaSpace;

impl ExecutionSpace for CudaSpace {
    type Stream = CudaStream;
    type Event = CudaEvent;
    type Module = CudaModule;
    type Kernel = CudaKernel;

    const IS_HOST: bool = false;
    const IS_DEVICE: bool = true;
    const SUPPORTS_ASYNC: bool = true;

    fn create_stream(_device_id: i64) -> error::Result<CudaStream> {
        ensure_init()?;
        let mut stream: CUstream = std::ptr::null_mut();
        unsafe {
            check(
                cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate",
            )?;
        }
        Ok(CudaStream(stream))
    }

    fn destroy_stream(stream: &mut CudaStream) {
        if !stream.0.is_null() {
            unsafe {
                let _ = cuStreamDestroy_v2(stream.0);
            }
            stream.0 = std::ptr::null_mut();
        }
    }

    fn sync_stream(stream: &CudaStream) -> error::Result<()> {
        unsafe { check(cuStreamSynchronize(stream.0), "cuStreamSynchronize") }
    }

    fn stream_ready(stream: &CudaStream) -> error::Result<bool> {
        let res = unsafe { cuStreamQuery(stream.0) };
        if res == CUDA_SUCCESS {
            Ok(true)
        } else if res == 600 {
            Ok(false)
        }
        // CUDA_ERROR_NOT_READY
        else {
            Err(XpuError::new("cuStreamQuery", res))
        }
    }

    fn create_event() -> error::Result<CudaEvent> {
        let mut event: CUevent = std::ptr::null_mut();
        unsafe {
            check(
                cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING),
                "cuEventCreate",
            )?;
        }
        Ok(CudaEvent(event))
    }

    fn destroy_event(event: CudaEvent) {
        if !event.0.is_null() {
            unsafe {
                let _ = cuEventDestroy_v2(event.0);
            }
        }
    }

    fn record_event(event: &CudaEvent, stream: &CudaStream) -> error::Result<()> {
        unsafe { check(cuEventRecord(event.0, stream.0), "cuEventRecord") }
    }

    fn event_ready(event: &CudaEvent) -> error::Result<bool> {
        let res = unsafe { cuEventQuery(event.0) };
        if res == CUDA_SUCCESS {
            Ok(true)
        } else if res == 600 {
            Ok(false)
        }
        // CUDA_ERROR_NOT_READY
        else {
            Err(XpuError::new("cuEventQuery", res))
        }
    }

    fn sync_event(event: &CudaEvent) -> error::Result<()> {
        unsafe { check(cuEventSynchronize(event.0), "cuEventSynchronize") }
    }

    fn stream_wait_event(stream: &CudaStream, event: &CudaEvent) -> error::Result<()> {
        unsafe { check(cuStreamWaitEvent(stream.0, event.0, 0), "cuStreamWaitEvent") }
    }

    fn load_module(bytes: &[u8]) -> error::Result<CudaModule> {
        ensure_init()?;
        let mut handle: CUmodule = std::ptr::null_mut();
        let ptx = if bytes.last() == Some(&0) {
            bytes.to_vec()
        } else {
            let mut v = bytes.to_vec();
            v.push(0);
            v
        };
        unsafe {
            check(
                cuModuleLoadData(&mut handle, ptx.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
        }
        Ok(CudaModule(handle))
    }

    fn get_function(module: &CudaModule, name: &str) -> error::Result<CudaKernel> {
        let name_c = CString::new(name).map_err(|_| XpuError {
            operation: "get_function",
            code: -1,
            detail: "invalid kernel name".into(),
        })?;
        let mut handle: CUfunction = std::ptr::null_mut();
        unsafe {
            check(
                cuModuleGetFunction(&mut handle, module.0, name_c.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(CudaKernel(handle))
    }

    unsafe fn launch(
        stream: &CudaStream,
        kernel: &CudaKernel,
        config: LaunchConfig,
        args: &mut [*mut c_void],
    ) -> error::Result<()> {
        let res = unsafe {
            cuLaunchKernel(
                kernel.0,
                config.grid[0],
                config.grid[1],
                config.grid[2],
                config.block[0],
                config.block[1],
                config.block[2],
                config.shared_mem_bytes,
                stream.0,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        check(res, "cuLaunchKernel")
    }
}

// =============================================================================
// UnifiedMemory: MemorySpace implementation
// =============================================================================

/// CUDA unified (managed) memory. accessible from both host and device.
pub struct UnifiedMemory;

impl MemorySpace for UnifiedMemory {
    const IS_HOST_ACCESSIBLE: bool = true;
    const IS_DEVICE_ACCESSIBLE: bool = true;
    const IS_UNIFIED: bool = true;
    const PREFERRED_ALIGNMENT: usize = 256;

    fn allocate(bytes: usize) -> error::Result<*mut u8> {
        ensure_init()?;
        let mut dptr: CUdeviceptr = 0;
        unsafe {
            check(
                cuMemAllocManaged(&mut dptr, bytes, CU_MEM_ATTACH_GLOBAL),
                "cuMemAllocManaged",
            )?;
            check(cuMemsetD8_v2(dptr, 0, bytes), "cuMemsetD8")?;
        }
        Ok(dptr as *mut u8)
    }

    fn deallocate(ptr: *mut u8, _bytes: usize) {
        unsafe {
            let _ = cuMemFree_v2(ptr as CUdeviceptr);
        }
    }
}
