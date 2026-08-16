// =============================================================================
// hip.rs
//
// amd HIP/ROCm execution space and managed memory space. raw driver API bindings,
// no external crate dependencies -- just extern "C" to libamdhip64.so.
//
// the sibling of cuda.rs. the key difference: HIP has no context
// api -- a device is bound per-thread with `hipSetDevice(ord)` and the primary
// context is implicit, so a per-thread device slot stands in for cuda's context
// registry; `with_device` here is just save / set / restore, and the per-device
// dispatcher registry keys on the ordinal exactly as the cuda path does. peer copy
// takes device ordinals.
//
// link: -lamdhip64 (set in build.rs under the hip feature).
// =============================================================================

#![allow(non_camel_case_types)]

use crate::config::LaunchConfig;
use crate::error::{self, XpuError};
use crate::memory::MemorySpace;
use crate::space::ExecutionSpace;
use std::cell::Cell;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

// =============================================================================
// raw HIP driver API types
// =============================================================================

type hipError_t = c_int;
type hipDevice_t = c_int;
type hipStream_t = *mut c_void;
type hipEvent_t = *mut c_void;
type hipModule_t = *mut c_void;
type hipFunction_t = *mut c_void;
type hipDeviceptr_t = *mut c_void;

const HIP_SUCCESS: hipError_t = 0;
const HIP_ERROR_NOT_READY: hipError_t = 600;
// re-enabling already-enabled peer access counts as success. value mirrors
// the cuda numbering; verify against hip_runtime_api.h on the cluster if peer-enable misreports
// (it is best-effort -- hipMemcpyPeer works regardless, so a wrong code here still leaves
// correctness intact).
const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: hipError_t = 704;
const HIP_MEM_ATTACH_GLOBAL: c_uint = 1;
const HIP_EVENT_DISABLE_TIMING: c_uint = 2;
const HIP_STREAM_NON_BLOCKING: c_uint = 1;

// =============================================================================
// raw HIP driver API bindings
// =============================================================================

unsafe extern "C" {
    fn hipInit(flags: c_uint) -> hipError_t;
    fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    fn hipSetDevice(device: c_int) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    fn hipStreamQuery(stream: hipStream_t) -> hipError_t;
    fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> hipError_t;
    fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    fn hipEventQuery(event: hipEvent_t) -> hipError_t;
    fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: c_uint) -> hipError_t;
    fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    fn hipMallocManaged(dptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    fn hipFree(ptr: *mut c_void) -> hipError_t;
    fn hipMemsetD8(dst: hipDeviceptr_t, value: u8, count: usize) -> hipError_t;
    fn hipDeviceGetName(name: *mut c_char, len: c_int, dev: hipDevice_t) -> hipError_t;
    fn hipDeviceTotalMem(bytes: *mut usize, dev: hipDevice_t) -> hipError_t;
    // peer access + cross-device copy. hipMemcpyPeer takes device
    // ordinals, simpler than the cuda context-based form.
    fn hipMemcpyPeer(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        byte_count: usize,
    ) -> hipError_t;
    fn hipDeviceCanAccessPeer(can: *mut c_int, dev: hipDevice_t, peer: hipDevice_t) -> hipError_t;
    fn hipDeviceEnablePeerAccess(peer_device: c_int, flags: c_uint) -> hipError_t;
}

fn check(res: hipError_t, op: &'static str) -> error::Result<()> {
    if res == HIP_SUCCESS {
        Ok(())
    } else {
        Err(XpuError::new(op, res))
    }
}

// =============================================================================
// device binding: hipSetDevice (no context api)
// =============================================================================

/// max gpus per node bound. mirrors the cuda path; a fixed array avoids hot-path locking.
pub const MAX_GPUS: usize = 16;

static HIP_INIT: OnceLock<()> = OnceLock::new();
static DEVICE_COUNT: OnceLock<i32> = OnceLock::new();

thread_local! {
    // the device bound on the current thread. hip's current device is per-thread, so this is too.
    static CURRENT_DEVICE: Cell<i32> = const { Cell::new(0) };
}

/// the device ordinal bound on this thread.
pub fn current_device() -> i32 {
    CURRENT_DEVICE.with(|c| c.get())
}

fn ensure_hip_init() {
    HIP_INIT.get_or_init(|| unsafe {
        check(hipInit(0), "hipInit").expect("hipInit failed");
    });
}

fn cached_device_count() -> i32 {
    *DEVICE_COUNT.get_or_init(|| {
        ensure_hip_init();
        let mut count: c_int = 0;
        unsafe {
            let _ = hipGetDeviceCount(&mut count);
        }
        count
    })
}

/// logical ordinals round-robin onto the physical devices: identity when there are at least as
/// many gpus as logical ids (the production case), wrapping otherwise. unlike cuda, folding two
/// logical ids onto one card shares the primary context (no per-context module table), so the
/// local "n logical devices on one card" trick leaves hip's context binding untested locally --
/// but hip is cluster-validated anyway, where the mapping is identity.
fn physical(ord: i32) -> i32 {
    let count = cached_device_count();
    if count > 0 { ord % count } else { 0 }
}

/// number of hip devices visible to the process.
pub fn device_count() -> error::Result<i32> {
    let mut count: c_int = 0;
    unsafe {
        check(hipInit(0), "hipInit")?;
        check(hipGetDeviceCount(&mut count), "hipGetDeviceCount")?;
    }
    Ok(count)
}

/// ensure hip is initialized and device `ord` (mapped to a physical device) is current.
fn ensure_init_device(ord: i32) -> error::Result<()> {
    assert!(
        (ord as usize) < MAX_GPUS,
        "device ordinal {ord} exceeds MAX_GPUS"
    );
    ensure_hip_init();
    unsafe { check(hipSetDevice(physical(ord)), "hipSetDevice") }
}

/// ensure hip is initialized and the current thread's device is bound.
fn ensure_init() -> error::Result<()> {
    ensure_init_device(current_device())
}

/// run `f` with device `ord` bound on this thread, restoring the previous device after. binds a
/// tile's kernels to its gpu: launch / alloc / sync target "the current
/// device", so the target device is made current on this thread for the closure.
pub fn with_device<R>(ord: i32, f: impl FnOnce() -> R) -> R {
    let prev = current_device();
    ensure_init_device(ord).expect("with_device: hipSetDevice");
    CURRENT_DEVICE.with(|c| c.set(ord));
    let r = f();
    ensure_init_device(prev).expect("with_device: restore device");
    CURRENT_DEVICE.with(|c| c.set(prev));
    r
}

/// block until all outstanding work on the current device finishes. panics on driver error,
/// the only recovery available to dispatch code at this point.
pub fn ctx_sync() {
    let res = unsafe { hipDeviceSynchronize() };
    if res != HIP_SUCCESS {
        panic!("hipDeviceSynchronize failed: error {res}");
    }
}

// =============================================================================
// peer access: direct device-to-device halo copy. mirrors the cuda
// surface so `decomp::PeerCopy` is backend-agnostic; hip keys on device ordinals.
// =============================================================================

/// can device `ord` directly read memory resident on device `peer`?
pub fn can_access_peer(ord: i32, peer: i32) -> error::Result<bool> {
    ensure_hip_init();
    let mut flag: c_int = 0;
    unsafe {
        check(
            hipDeviceCanAccessPeer(&mut flag, physical(ord), physical(peer)),
            "hipDeviceCanAccessPeer",
        )?;
    }
    Ok(flag != 0)
}

/// enable direct peer access from device `ord` to memory on device `peer`. directional;
/// idempotent ("already enabled" is success). hip enables from the current device, so bind
/// `ord` first.
pub fn enable_peer_access(ord: i32, peer: i32) -> error::Result<()> {
    with_device(ord, || {
        let res = unsafe { hipDeviceEnablePeerAccess(physical(peer), 0) };
        if res == HIP_SUCCESS || res == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            Ok(())
        } else {
            Err(XpuError::new("hipDeviceEnablePeerAccess", res))
        }
    })
}

/// copy `bytes` from `src` (resident on `src_ord`) to `dst` (resident on `dst_ord`),
/// synchronously, over the peer link. the one new primitive in the multi-gpu transport; the
/// gather/scatter halves are shared with `StagedCopy`.
pub fn memcpy_peer(
    dst: u64,
    dst_ord: i32,
    src: u64,
    src_ord: i32,
    bytes: usize,
) -> error::Result<()> {
    ensure_hip_init();
    unsafe {
        check(
            hipMemcpyPeer(
                dst as *mut c_void,
                physical(dst_ord),
                src as *const c_void,
                physical(src_ord),
                bytes,
            ),
            "hipMemcpyPeer",
        )
    }
}

// =============================================================================
// device info (cosmetic: the live system-info table). amd has no meaningful (major, minor)
// compute capability; the arch is `gcnArchName`, surfaced via SYMBI_HIP_ARCH at jit time.
// =============================================================================

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub total_memory_bytes: u64,
    pub compute_capability: (i32, i32),
    pub device_count: i32,
}

pub fn device_info() -> error::Result<DeviceInfo> {
    ensure_init()?;
    let count = device_count()?;
    let mut name_buf = [0_i8; 256];
    unsafe {
        check(
            hipDeviceGetName(
                name_buf.as_mut_ptr(),
                name_buf.len() as c_int,
                physical(current_device()),
            ),
            "hipDeviceGetName",
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
            hipDeviceTotalMem(&mut total_bytes, physical(current_device())),
            "hipDeviceTotalMem",
        )?;
    }
    Ok(DeviceInfo {
        name,
        total_memory_bytes: total_bytes as u64,
        // amd arch is `gcnArchName`, a string, so the numeric field reports (0, 0); rely on
        // SYMBI_HIP_ARCH.
        compute_capability: (0, 0),
        device_count: count,
    })
}

// =============================================================================
// handle wrappers (Send-safe)
// =============================================================================

/// HIP stream handle.
pub struct HipStream(hipStream_t);
unsafe impl Send for HipStream {}

impl HipStream {
    /// the default (null) HIP stream. synchronous with the device.
    pub fn null() -> Self {
        HipStream(std::ptr::null_mut())
    }
}

/// HIP event handle.
pub struct HipEvent(hipEvent_t);
unsafe impl Send for HipEvent {}

/// HIP module handle.
pub struct HipModule(hipModule_t);
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

/// HIP kernel handle. Copy because it's just a function pointer.
#[derive(Clone, Copy, Debug)]
pub struct HipKernel(hipFunction_t);
unsafe impl Send for HipKernel {}
unsafe impl Sync for HipKernel {}

// =============================================================================
// HipSpace: ExecutionSpace implementation
// =============================================================================

pub struct HipSpace;

impl ExecutionSpace for HipSpace {
    type Stream = HipStream;
    type Event = HipEvent;
    type Module = HipModule;
    type Kernel = HipKernel;

    const IS_HOST: bool = false;
    const IS_DEVICE: bool = true;
    const SUPPORTS_ASYNC: bool = true;

    fn create_stream(_device_id: i64) -> error::Result<HipStream> {
        ensure_init()?;
        let mut stream: hipStream_t = std::ptr::null_mut();
        unsafe {
            check(
                hipStreamCreateWithFlags(&mut stream, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags",
            )?;
        }
        Ok(HipStream(stream))
    }

    fn destroy_stream(stream: &mut HipStream) {
        if !stream.0.is_null() {
            unsafe {
                let _ = hipStreamDestroy(stream.0);
            }
            stream.0 = std::ptr::null_mut();
        }
    }

    fn sync_stream(stream: &HipStream) -> error::Result<()> {
        unsafe { check(hipStreamSynchronize(stream.0), "hipStreamSynchronize") }
    }

    fn stream_ready(stream: &HipStream) -> error::Result<bool> {
        let res = unsafe { hipStreamQuery(stream.0) };
        if res == HIP_SUCCESS {
            Ok(true)
        } else if res == HIP_ERROR_NOT_READY {
            Ok(false)
        } else {
            Err(XpuError::new("hipStreamQuery", res))
        }
    }

    fn create_event() -> error::Result<HipEvent> {
        let mut event: hipEvent_t = std::ptr::null_mut();
        unsafe {
            check(
                hipEventCreateWithFlags(&mut event, HIP_EVENT_DISABLE_TIMING),
                "hipEventCreateWithFlags",
            )?;
        }
        Ok(HipEvent(event))
    }

    fn destroy_event(event: HipEvent) {
        if !event.0.is_null() {
            unsafe {
                let _ = hipEventDestroy(event.0);
            }
        }
    }

    fn record_event(event: &HipEvent, stream: &HipStream) -> error::Result<()> {
        unsafe { check(hipEventRecord(event.0, stream.0), "hipEventRecord") }
    }

    fn event_ready(event: &HipEvent) -> error::Result<bool> {
        let res = unsafe { hipEventQuery(event.0) };
        if res == HIP_SUCCESS {
            Ok(true)
        } else if res == HIP_ERROR_NOT_READY {
            Ok(false)
        } else {
            Err(XpuError::new("hipEventQuery", res))
        }
    }

    fn sync_event(event: &HipEvent) -> error::Result<()> {
        unsafe { check(hipEventSynchronize(event.0), "hipEventSynchronize") }
    }

    fn stream_wait_event(stream: &HipStream, event: &HipEvent) -> error::Result<()> {
        unsafe {
            check(
                hipStreamWaitEvent(stream.0, event.0, 0),
                "hipStreamWaitEvent",
            )
        }
    }

    fn load_module(bytes: &[u8]) -> error::Result<HipModule> {
        ensure_init()?;
        let mut handle: hipModule_t = std::ptr::null_mut();
        // the hiprtc output is a raw binary code object; load the bytes as-is -- no nul
        // terminator required.
        unsafe {
            check(
                hipModuleLoadData(&mut handle, bytes.as_ptr() as *const c_void),
                "hipModuleLoadData",
            )?;
        }
        Ok(HipModule(handle))
    }

    fn get_function(module: &HipModule, name: &str) -> error::Result<HipKernel> {
        let name_c = CString::new(name).map_err(|_| XpuError {
            operation: "get_function",
            code: -1,
            detail: "invalid kernel name".into(),
        })?;
        let mut handle: hipFunction_t = std::ptr::null_mut();
        unsafe {
            check(
                hipModuleGetFunction(&mut handle, module.0, name_c.as_ptr()),
                "hipModuleGetFunction",
            )?;
        }
        Ok(HipKernel(handle))
    }

    unsafe fn launch(
        stream: &HipStream,
        kernel: &HipKernel,
        config: LaunchConfig,
        args: &mut [*mut c_void],
    ) -> error::Result<()> {
        let res = unsafe {
            hipModuleLaunchKernel(
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
        check(res, "hipModuleLaunchKernel")
    }
}

// =============================================================================
// HipManaged: MemorySpace implementation
// =============================================================================

/// HIP managed (unified) memory. accessible from both host and device. the amd analog of
/// `UnifiedMemory`; the same managed-thrash caveat applies -- a device-local space is
/// future work.
pub struct HipManaged;

impl MemorySpace for HipManaged {
    const IS_HOST_ACCESSIBLE: bool = true;
    const IS_DEVICE_ACCESSIBLE: bool = true;
    const IS_UNIFIED: bool = true;
    const PREFERRED_ALIGNMENT: usize = 256;

    fn allocate(bytes: usize) -> error::Result<*mut u8> {
        ensure_init()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            check(
                hipMallocManaged(&mut ptr, bytes, HIP_MEM_ATTACH_GLOBAL),
                "hipMallocManaged",
            )?;
            check(hipMemsetD8(ptr, 0, bytes), "hipMemsetD8")?;
        }
        Ok(ptr as *mut u8)
    }

    fn deallocate(ptr: *mut u8, _bytes: usize) {
        unsafe {
            let _ = hipFree(ptr as *mut c_void);
        }
    }
}
