// =============================================================================
// runtime.rs
//
// generic GPU kernel dispatch: compile/load, cache, launch.
//
// the GpuRuntime trait is the single abstraction that backends implement.
// KernelDispatcher<R> wraps a runtime with a thread-safe kernel cache.
// adding a new backend (HIP, Metal, SYCL) = implement GpuRuntime.
//
// usage (internal — hidden from users):
//   let kernel = DISPATCHER.jit_kernel_keyed(source, cache_key, entry_name);
//   DISPATCHER.runtime().launch(&kernel, config, args);
// =============================================================================

use crate::config::LaunchConfig;
use std::collections::HashMap;
use std::sync::Mutex;

// =============================================================================
// the trait
// =============================================================================

/// backend-specific GPU runtime. one impl per target (CUDA, HIP, Metal).
/// all methods are stateless — the runtime itself carries no mutable state.
/// caching lives in KernelDispatcher; the runtime holds none.
pub trait GpuRuntime: 'static + Send + Sync {
    type Module: Send + Sync;
    type Kernel: Send + Sync + Copy;

    /// compile kernel source to a loadable binary (PTX, HSACO) with the backend's
    /// own in-process runtime compiler — NVRTC for CUDA, hiprtc for HIP — compiling
    /// entirely in-process. default: the backend has no in-process compiler, so
    /// source dispatch is unsupported (override to enable JIT).
    fn compile(&self, source: &str, name: &str) -> crate::Result<Vec<u8>> {
        let _ = (source, name);
        Err(crate::XpuError {
            operation: "GpuRuntime::compile",
            code: -1,
            detail: "this backend has no runtime compiler (override GpuRuntime::compile)".into(),
        })
    }

    /// load a pre-compiled binary (PTX, HSACO, metallib) into a module.
    fn load_binary(&self, binary: &[u8]) -> crate::Result<Self::Module>;

    /// extract a named kernel entry point from a loaded module.
    fn get_kernel(&self, module: &Self::Module, name: &str) -> crate::Result<Self::Kernel>;

    /// launch a kernel with a raw arg array.
    /// safety: arg pointers must match the kernel signature exactly.
    unsafe fn launch(
        &self,
        kernel: &Self::Kernel,
        config: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> crate::Result<()>;
}

// =============================================================================
// the cache + dispatcher
// =============================================================================

/// thread-safe kernel cache + dispatch for a single GPU backend.
/// one static instance per backend (e.g., CUDA_DISPATCHER, HIP_DISPATCHER).
pub struct KernelDispatcher<R: GpuRuntime> {
    runtime: R,
    cache: Mutex<HashMap<String, (R::Module, R::Kernel)>>,
}

impl<R: GpuRuntime> KernelDispatcher<R> {
    pub fn new(runtime: R) -> Self {
        KernelDispatcher {
            runtime,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// get the kernel handle for `source`, compiling via the runtime's own JIT
    /// (NVRTC) and caching the module per kernel_name — for callers that build a
    /// non-standard arg array (e.g., interleaved int/float scalar params) and launch
    /// via `runtime()` themselves. no caller-supplied compile closure.
    pub fn jit_kernel(&self, source: &str, kernel_name: &str) -> R::Kernel {
        self.jit_kernel_keyed(source, kernel_name, kernel_name)
    }

    /// like `jit_kernel`, but the module cache is keyed by `cache_key` (which may
    /// encode e.g., the precision) while the entry point is resolved by `entry_name`
    /// (the real symbol in the PTX). lets the same kernel name compile at two
    /// precisions in one process, each precision keeping its own cache slot so the
    /// f32 and f64 modules coexist.
    ///
    /// **content-addressed dedup.** the cache is internally keyed by
    /// `(cache_key, hash(source))` — so two distinct sources with the same
    /// `cache_key` get distinct cache slots. user-supplied `cache_key`s are
    /// diagnostic labels; correctness is enforced by hashing the source. this
    /// discipline is what stops two callers passing the same `cache_key` with
    /// different source strings from silently launching the first caller's
    /// cached PTX — a content-vs-name footgun. the source hash closes it.
    pub fn jit_kernel_keyed(&self, source: &str, cache_key: &str, entry_name: &str) -> R::Kernel {
        let internal_key = compute_internal_cache_key(cache_key, source.as_bytes());

        let mut cache = self.cache.lock().unwrap();
        let entry = cache.entry(internal_key).or_insert_with(|| {
            let binary = self
                .runtime
                .compile(source, entry_name)
                .unwrap_or_else(|e| panic!("JIT compile of '{}' failed: {:?}", entry_name, e));
            let module = self
                .runtime
                .load_binary(&binary)
                .unwrap_or_else(|e| panic!("failed to load JIT module '{}': {:?}", entry_name, e));
            let func = self
                .runtime
                .get_kernel(&module, entry_name)
                .unwrap_or_else(|e| panic!("failed to get kernel '{}': {:?}", entry_name, e));
            (module, func)
        });
        entry.1
    }

    /// expose the runtime for direct launch calls (callers that pack a
    /// non-standard arg array launch via `runtime().launch` themselves).
    pub fn runtime(&self) -> &R {
        &self.runtime
    }
}

/// the content-addressed cache key the dispatcher uses internally for
/// every JIT/binary cache lookup. callers pass a `name` (the entry symbol,
/// or a precision-augmented variant); this fn appends a content hash so
/// two distinct sources / binaries with the same `name` get distinct cache
/// slots.
///
/// **the discipline**: `name` is a diagnostic label; the content hash
/// enforces correctness. callers can pass any `name` they like — the
/// dispatcher's actual cache index includes the source's hash, so name
/// reuse resolves safely to the correct kernel.
///
/// public for testability — `tests/dispatcher_cache.rs` asserts the
/// content-hashing discipline entirely on the host, no GPU needed.
#[doc(hidden)]
pub fn compute_internal_cache_key(name: &str, content: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    let h = hasher.finish();
    format!("{name}#{h:016x}")
}

// =============================================================================
// CUDA runtime
// =============================================================================

#[cfg(feature = "cuda")]
pub mod cuda_runtime {
    use super::*;
    use crate::ExecutionSpace;
    use crate::cuda::{CudaKernel, CudaModule, CudaSpace, CudaStream};

    pub struct CudaRuntime;

    impl GpuRuntime for CudaRuntime {
        type Module = CudaModule;
        type Kernel = CudaKernel;

        fn compile(&self, source: &str, name: &str) -> crate::Result<Vec<u8>> {
            crate::nvrtc::compile_ptx(source, name)
        }

        fn load_binary(&self, binary: &[u8]) -> crate::Result<CudaModule> {
            CudaSpace::load_module(binary)
        }

        fn get_kernel(&self, module: &CudaModule, name: &str) -> crate::Result<CudaKernel> {
            CudaSpace::get_function(module, name)
        }

        unsafe fn launch(
            &self,
            kernel: &CudaKernel,
            config: LaunchConfig,
            args: &mut [*mut std::ffi::c_void],
        ) -> crate::Result<()> {
            let stream = CudaStream::null();
            unsafe { CudaSpace::launch(&stream, kernel, config, args) }
        }
    }

    fn make_dispatcher() -> KernelDispatcher<CudaRuntime> {
        KernelDispatcher::new(CudaRuntime)
    }

    // one dispatcher per device ordinal: cuda modules are bound to the context they were
    // loaded into, so device N's kernels must live in device N's dispatcher.
    // each is initialized on first use, in whatever context is current at that point.
    static DISPATCHERS: [std::sync::LazyLock<KernelDispatcher<CudaRuntime>>;
        crate::cuda::MAX_GPUS] = [const {
        std::sync::LazyLock::new(make_dispatcher as fn() -> KernelDispatcher<CudaRuntime>)
    }; crate::cuda::MAX_GPUS];

    /// the dispatcher for the device whose context is current on this thread. callers JIT and
    /// launch inside `cuda::with_device(ord, ...)`, so the modules land in the right context.
    /// on the single-device path this is always device 0's dispatcher.
    pub fn current_dispatcher() -> &'static KernelDispatcher<CudaRuntime> {
        &DISPATCHERS[crate::cuda::current_device() as usize]
    }
}

// =============================================================================
// HIP runtime. the amd sibling of cuda_runtime: hiprtc -> code object ->
// hipModuleLoadData. same per-device dispatcher registry (modules are per-device), keyed on
// the hip current-device ordinal.
// =============================================================================

#[cfg(feature = "hip")]
pub mod hip_runtime {
    use super::*;
    use crate::ExecutionSpace;
    use crate::hip::{HipKernel, HipModule, HipSpace, HipStream};

    pub struct HipRuntime;

    impl GpuRuntime for HipRuntime {
        type Module = HipModule;
        type Kernel = HipKernel;

        fn compile(&self, source: &str, name: &str) -> crate::Result<Vec<u8>> {
            crate::hiprtc::compile_code(source, name)
        }

        fn load_binary(&self, binary: &[u8]) -> crate::Result<HipModule> {
            HipSpace::load_module(binary)
        }

        fn get_kernel(&self, module: &HipModule, name: &str) -> crate::Result<HipKernel> {
            HipSpace::get_function(module, name)
        }

        unsafe fn launch(
            &self,
            kernel: &HipKernel,
            config: LaunchConfig,
            args: &mut [*mut std::ffi::c_void],
        ) -> crate::Result<()> {
            let stream = HipStream::null();
            unsafe { HipSpace::launch(&stream, kernel, config, args) }
        }
    }

    fn make_dispatcher() -> KernelDispatcher<HipRuntime> {
        KernelDispatcher::new(HipRuntime)
    }

    // one dispatcher per device ordinal: hip modules are bound to the device they were loaded
    // for, so device N's kernels must live in device N's dispatcher.
    static DISPATCHERS: [std::sync::LazyLock<KernelDispatcher<HipRuntime>>; crate::hip::MAX_GPUS] = [const {
        std::sync::LazyLock::new(make_dispatcher as fn() -> KernelDispatcher<HipRuntime>)
    };
        crate::hip::MAX_GPUS];

    /// the dispatcher for the device bound on this thread. callers JIT and launch inside
    /// `hip::with_device(ord, ...)`, so the modules land for the right device.
    pub fn current_dispatcher() -> &'static KernelDispatcher<HipRuntime> {
        &DISPATCHERS[crate::hip::current_device() as usize]
    }
}

// =============================================================================
// neutral backend selection: downstream names `runtime::DeviceRuntime` and
// `runtime::current_dispatcher()`; the active backend feature binds them. cuda wins if both
// are somehow set (the hip arms are `not(cuda)`), so the tree binds exactly once.
// =============================================================================

#[cfg(feature = "cuda")]
pub use cuda_runtime::{CudaRuntime as DeviceRuntime, current_dispatcher};

#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub use hip_runtime::{HipRuntime as DeviceRuntime, current_dispatcher};
