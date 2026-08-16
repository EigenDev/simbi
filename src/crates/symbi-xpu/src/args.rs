// =============================================================================
// args.rs
//
// kernel argument builder. type-safe construction of the void** arg array
// that cuLaunchKernel / hipModuleLaunchKernel expect.
//
// implementation: a single flat byte arena (`storage: Vec<u8>`) holding every pushed
// arg. push appends the value's bytes (properly aligned for `T`) and
// records the byte offset; `as_mut_slice` rebuilds the void* pointer table by
// offsetting the arena's base pointer. on reuse (`clear`) the arena retains
// capacity — after the first launch hits steady-state size, subsequent launches
// pay zero allocations.
//
// safety: cuLaunchKernel reads the pointed-to bytes synchronously (the driver
// copies them out before returning), so the arena's contents only need to stay
// stable through the launch call. a push after `as_mut_slice` invalidates the
// pointers already handed out (it could realloc the arena), so any push
// following `as_mut_slice` requires a fresh call to it before the next launch.
//
// usage:
//   let mut args = KernelArgs::new();
//   args.push(&device_ptr);
//   args.push(&grid_size);
//   args.push(&gamma);
//   unsafe { E::launch(&ctx, &kernel, config, args.as_mut_slice()); }
//   args.clear();   // ready for the next launch — capacity retained
// =============================================================================

/// type-safe kernel argument builder. one byte arena, one offset table.
pub struct KernelArgs {
    /// concatenated arg bytes, each properly aligned for its `T`.
    storage: Vec<u8>,
    /// byte offset (into `storage`) for each pushed arg.
    offsets: Vec<u32>,
    /// pointer table built by `as_mut_slice`. cuLaunchKernel takes `void**`.
    ptrs: Vec<*mut std::ffi::c_void>,
}

impl KernelArgs {
    pub fn new() -> Self {
        KernelArgs {
            storage: Vec::new(),
            offsets: Vec::new(),
            ptrs: Vec::new(),
        }
    }

    /// reserve byte capacity for the arena and slot capacity for the offset /
    /// pointer tables. pre-sizing to the largest plausible arg footprint makes
    /// the entire `push` sequence allocation-free.
    pub fn with_capacity_bytes(bytes: usize, n_args: usize) -> Self {
        KernelArgs {
            storage: Vec::with_capacity(bytes),
            offsets: Vec::with_capacity(n_args),
            ptrs: Vec::with_capacity(n_args),
        }
    }

    /// reset for a fresh launch. drops no buffers; retains all capacity.
    pub fn clear(&mut self) {
        self.storage.clear();
        self.offsets.clear();
        self.ptrs.clear();
    }

    /// push a value of any `Copy` type as a kernel argument. the value's bytes
    /// are appended to the arena at an offset properly aligned for `T`.
    pub fn push<T: Copy>(&mut self, val: &T) {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        // pad the cursor up to the next multiple of `align`. align is always a
        // power of 2 for Rust types; this is the standard mask trick written
        // out longhand for clarity.
        let cur = self.storage.len();
        let pad = (align - cur % align) % align;
        if pad > 0 {
            self.storage.resize(cur + pad, 0);
        }
        let offset = self.storage.len();
        // safety: `T: Copy` so it has no destructor; the byte view is sound
        // for any `Copy` plain-data type. `size_of::<T>` bytes are valid.
        let bytes = unsafe { std::slice::from_raw_parts(val as *const T as *const u8, size) };
        self.storage.extend_from_slice(bytes);
        self.offsets.push(offset as u32);
    }

    /// build the void** array for cuLaunchKernel. valid until the next `push`
    /// or `clear` — i.e., valid through the launch call. a push after this call
    /// invalidates the previously handed-out pointers (it could realloc the
    /// arena); re-call `as_mut_slice` before the next launch if more args follow.
    pub fn as_mut_slice(&mut self) -> &mut [*mut std::ffi::c_void] {
        self.ptrs.clear();
        let base = self.storage.as_mut_ptr();
        for &off in &self.offsets {
            // safety: `off` came from `storage.len()` at push time and the
            // arena has only grown since; the offset is in-bounds.
            let p = unsafe { base.add(off as usize) };
            self.ptrs.push(p as *mut std::ffi::c_void);
        }
        &mut self.ptrs
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

impl Default for KernelArgs {
    fn default() -> Self {
        Self::new()
    }
}

// ---- thread-local pool ------------------------------------------------------
//
// every kernel launch on the same thread reuses the same KernelArgs instance —
// the arena retains its grown capacity across launches, so after warmup pushes
// hit existing arena bytes and allocate nothing. CPU sims are usually
// single-threaded; multi-threaded callers (e.g., rayon-parallel CPU kernels)
// pool one arena per worker, no shared state, no lock.
//
// initial sizing covers the widest substrate kernel comfortably: rmhd face_flux
// is ~30 buffers \times 40-byte DeviceView + ~15 i32/f64 scalars \approx 1.3 KB, rounded
// up to 4 KB so the first launch already sits at steady-state capacity.

thread_local! {
    static POOL: std::cell::RefCell<KernelArgs> =
        std::cell::RefCell::new(KernelArgs::with_capacity_bytes(4096, 64));
}

/// run a closure with the calling thread's pooled `KernelArgs`. the arena is
/// `clear()`-ed before the closure runs (so the closure starts with an empty
/// arg list) and the closure-return is forwarded out. after warmup, the
/// arena has hit steady-state capacity and the closure's `push` calls
/// allocate nothing.
///
/// the closure drives a single kernel launch only; triggering another dispatch
/// via the pool on the same thread panics on the re-entrant `borrow_mut`. CUDA
/// kernel launches return control to the host directly, so the substrate
/// dispatch path is safe.
pub fn with_pooled_args<R>(f: impl FnOnce(&mut KernelArgs) -> R) -> R {
    POOL.with_borrow_mut(|a| {
        a.clear();
        f(a)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_then_read_back_int_double() {
        let mut a = KernelArgs::new();
        let x: i32 = 42;
        let y: f64 = std::f64::consts::PI;
        a.push(&x);
        a.push(&y);
        let s = a.as_mut_slice();
        assert_eq!(s.len(), 2);
        // safety: pointers point into the arena's stable bytes.
        unsafe {
            assert_eq!(*(s[0] as *const i32), 42);
            assert!((*(s[1] as *const f64) - std::f64::consts::PI).abs() < 1e-15);
        }
    }

    #[test]
    fn alignment_is_respected_for_f64_after_i32() {
        let mut a = KernelArgs::new();
        a.push(&1i32);
        a.push(&2.0f64);
        let s = a.as_mut_slice();
        // the f64 pointer must be 8-byte aligned
        assert_eq!(s[1] as usize % std::mem::align_of::<f64>(), 0);
    }

    #[test]
    fn clear_retains_capacity() {
        let mut a = KernelArgs::with_capacity_bytes(256, 32);
        for _ in 0..16 {
            a.push(&0u64);
        }
        let cap_before = a.storage.capacity();
        a.clear();
        assert_eq!(a.len(), 0);
        assert_eq!(a.storage.capacity(), cap_before);
        // a second cycle stays within retained capacity
        for _ in 0..16 {
            a.push(&0u64);
        }
        assert_eq!(a.storage.capacity(), cap_before);
    }
}
