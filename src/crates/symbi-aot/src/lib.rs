// =============================================================================
// symbi-aot
//
// build-time AOT kernel library. build.rs
// runs the substrate to lower every kernel to IR and emit it as both compilable
// Rust source (via emit_kernel_cpu) and a serialized backend-NEUTRAL lowered IR
// blob (the Prepared artifact, via prepare + prepared_to_ir), then writes one
// registry module that `include!`s every CPU kernel and exposes every kernel's IR
// blob as a `<KERNEL>_IR` const. the blob renders to any backend at runtime via
// `symbi_ir::render_from_ir`. this crate `include!`s that single registry —
// so adding a kernel touches only build.rs's kernel list; this file carries
// no hand-maintained registration to update.
//
// each generated `pub fn` is a normal compiled function exported from this crate;
// downstream consumers (the tests here, `symbi`'s SubstrateKernelSet) call them
// directly. the generated kernels are self-contained Rust over slices — this crate
// has no RUNTIME dependency on the substrate (only the build does).
// =============================================================================

// the CPU field descriptor the generated kernels take. carries the buffer
// pointer plus its pre-multiplied row-major strides, so a kernel's per-cell index
// arithmetic reads `lo` / `strides` off a single descriptor and indexes `data`
// directly. the strides are computed ONCE at construction (host-side, per kernel
// launch) from `symbi_algebra::strides_from_extent`, the single definition of the
// stride formula.
//
// `lo` and `strides` are fixed `[i32; 4]` (no slice indirection, no bounds
// checks) — supports rank <= 4 (more than the project's 3D ceiling). the unused
// tail holds `0` and stays unread.
//
// `extent` is intentionally absent: no emitted kernel reads it (the strides
// already encode the only layout fact the index math needs). carrying it
// would waste 16 bytes per buffer x ~20 buffers = ~320 bytes of stack per
// kernel call for a field nobody reads.
//
// the name-keyed host/test invocation built on the registry + manifest — the
// sanctioned way to call an emitted kernel from host code (binds by field name,
// fails loud + named on drift). see named_call.rs.
pub mod named_call;
pub use named_call::NamedKernel;

// generic over scalar `T` (kernel precision); `T = f64` default for the
// f64 call sites + the f64-emitted kernels. `Copy` — a trivial read-only view
// (a shared slice + two small index arrays), so the same field can fill several
// slice slots (e.g., a zero-B buffer bound to all three magnetic components).
#[derive(Clone, Copy)]
pub struct CpuField<'a, T = f64> {
    pub data: &'a [T],
    pub lo: [i32; 4],
    pub strides: [i32; 4],
}

pub struct CpuFieldMut<'a, T = f64> {
    pub data: &'a mut [T],
    pub lo: [i32; 4],
    pub strides: [i32; 4],
}

/// strides from extent under the **physical-x-fastest convention**:
/// `strides[0] = 1`, `strides[d] = prod(extent[0..d])`. axis 0 is the
/// fastest-varying in memory — under the CFD-standard mapping (axis 0 = x),
/// adjacent CUDA `threadIdx.x` lanes hit adjacent bytes (coalesced reads).
///
/// delegates to `symbi_algebra::strides_from_extent` — the single definition of
/// the stride formula, shared with `Domain`, `Layout`, and symbi-grid's `View`.
/// `pub` so the GPU view-construction path reuses the same helper.
#[inline]
pub fn compute_strides(extent: &[u32]) -> [i32; 4] {
    let n = extent.len().min(4);
    let ext_i32: [i32; 4] = std::array::from_fn(|d| if d < n { extent[d] as i32 } else { 0 });
    let mut s = [0i32; 4];
    symbi_algebra::strides_from_extent(&ext_i32[..n], &mut s[..n]);
    // the emitted Rust index omits the `* strides[CONTIGUOUS_AXIS]` factor because it is 1 by
    // construction. if that ever stopped holding, every kernel would mis-index this buffer silently.
    debug_assert!(
        n == 0 || s[symbi_algebra::CONTIGUOUS_AXIS] == 1,
        "compute_strides: CONTIGUOUS_AXIS must be unit-stride, got {s:?} for extent {extent:?}",
    );
    s
}

/// widen a runtime `&[i32]` axis-lo into the fixed `[i32; 4]` view field, zero-
/// padding any unused tail axes. shared with the GPU view path (see
/// `compute_strides`).
#[inline]
pub fn copy_lo(lo: &[i32]) -> [i32; 4] {
    let mut out = [0i32; 4];
    for (d, &v) in lo.iter().enumerate().take(4) {
        out[d] = v;
    }
    out
}

/// widen a runtime `&[u32]` per-axis extent into the fixed `[i32; 4]` view field,
/// zero-padding unused tail axes. the shared-memory cooperative load clamps each
/// global-memory read to `[lo, lo + extent - 1]`, so the device view must carry extent.
#[inline]
pub fn copy_extent(extent: &[u32]) -> [i32; 4] {
    let mut out = [0i32; 4];
    for (d, &v) in extent.iter().enumerate().take(4) {
        out[d] = v as i32;
    }
    out
}

impl<'a, T> CpuField<'a, T> {
    /// construct from runtime slices (the substrate / dispatcher path). computes
    /// strides ONCE; subsequent accesses use the cached values.
    #[inline]
    pub fn from_layout(data: &'a [T], lo: &[i32], extent: &[u32]) -> Self {
        Self {
            data,
            lo: copy_lo(lo),
            strides: compute_strides(extent),
        }
    }
}

impl<'a, T> CpuFieldMut<'a, T> {
    #[inline]
    pub fn from_layout(data: &'a mut [T], lo: &[i32], extent: &[u32]) -> Self {
        Self {
            data,
            lo: copy_lo(lo),
            strides: compute_strides(extent),
        }
    }
}

// ---- the structured binding ABI ----
//
// the backend-NEUTRAL kernel invocation. the call site builds one
// `KernelInvocation`: an ordered buffer list (each a data HANDLE + its layout) +
// the packed params, keeping the CPU-specific `&[CpuField]` / `&mut [CpuFieldMut]`
// host slices (a host-memory-ism) out of the call site. the same invocation maps to a CPU
// call (`run_cpu`, below) or a GPU launch, by interpreting the handle. the
// device-pointer handle variant for the runtime GPU render is future work.

/// where a buffer's data lives. the variant encodes the kernel role: `Host` is a
/// read-only input, `HostMut` an output (incl. in-place — the kernel reads + writes
/// it). a `Device` variant for GPU launches is future work.
pub enum BufHandle<'a, T> {
    Host(&'a [T]),
    HostMut(&'a mut [T]),
}

/// one buffer binding: its data handle + its per-axis layout (`lo`/`extent`) on the
/// buffer's own domain (allocated, or a staggered face/edge domain).
pub struct Buf<'a, T> {
    pub handle: BufHandle<'a, T>,
    pub lo: &'a [i32],
    pub extent: &'a [u32],
}

/// a structured kernel invocation: the ordered buffers (inputs first, then outputs,
/// matching the kernel's binding order) + the packed params (`grid`/`dom_lo` exec
/// window, the `ints` and `scalars` lanes). this is the call-site-facing API; each
/// backend maps it to its own model.
pub struct KernelInvocation<'a, T> {
    pub buffers: Vec<Buf<'a, T>>,
    pub grid: &'a [u32],
    pub dom_lo: &'a [i32],
    pub ints: &'a [i32],
    pub scalars: &'a [T],
}

impl<'a, T> KernelInvocation<'a, T> {
    /// map onto a generated CPU kernel: split the buffers by handle into the
    /// `inputs: &[CpuField]` (Host) + `outputs: &mut [CpuFieldMut]` (HostMut) the
    /// generated `fn k<S>(..)` takes — in binding order — then call it. consumes
    /// self so the `HostMut` borrows move out cleanly into the disjoint `&mut`s.
    pub fn run_cpu<F>(self, kernel: F)
    where
        F: FnOnce(&[CpuField<'_, T>], &mut [CpuFieldMut<'_, T>], &[u32], &[i32], &[i32], &[T]),
    {
        // bounded by the kernel's buffer count (<= ~12), so the input/output
        // binding lists live on the stack — no heap allocation per dispatch.
        // SmallVec derefs to the `&[..]` / `&mut [..]` the kernel fn takes.
        use smallvec::SmallVec;
        let mut inputs: SmallVec<[CpuField<'a, T>; 16]> = SmallVec::new();
        let mut outputs: SmallVec<[CpuFieldMut<'a, T>; 16]> = SmallVec::new();
        for b in self.buffers {
            match b.handle {
                BufHandle::Host(d) => inputs.push(CpuField::from_layout(d, b.lo, b.extent)),
                BufHandle::HostMut(d) => outputs.push(CpuFieldMut::from_layout(d, b.lo, b.extent)),
            }
        }
        kernel(
            &inputs,
            &mut outputs,
            self.grid,
            self.dom_lo,
            self.ints,
            self.scalars,
        );
    }
}

// the kernel scalar type — the precision-generic parameter. the
// generated CPU kernels are `fn k<S: Scalar + OrderedNumeric>(..)`, so `Sim<f64>`
// and `Sim<f32>` pick the precision by the buffer type they pass (S inferred).
// the OrderedNumeric bound is the Tier-1.7 closure: the CPU emitter writes native
// `if cond { x } else { y }` for graph Branch nodes, so the body needs host
// ordering. f64/f32 both impl OrderedNumeric; the carrier-generic Gv path uses
// the GpuRenderer (CUDA C codegen) for its rendering.
pub use symbi_algebra::OrderedNumeric;
pub use symbi_ir::algebra::Scalar;

/// the structured CPU-kernel ABI as a fn pointer: `(inputs, outputs, grid, dom_lo,
/// ints, scalars)`. every generated `pub fn k<S: Scalar + OrderedNumeric>(..)` has
/// this shape, so the generic fn item `k::<S>` coerces to `KernelFn<S>`. `kernel_by_name`
/// (generated in the registry below) returns one of these, so the D-generic
/// SubstrateKernelSet picks a kernel instance by name through a single lookup
/// covering every regime. CpuField / CpuFieldMut / Scalar / OrderedNumeric
/// are in scope above.
pub type KernelFn<S> =
    fn(&[CpuField<'_, S>], &mut [CpuFieldMut<'_, S>], &[u32], &[i32], &[i32], &[S]);

// the build-time-generated kernel registry: one `include!` per CPU kernel (each a
// `pub fn k<S: Scalar>(..)`) + one `pub const <KERNEL>_IR: &str` per kernel (the
// serialized backend-neutral lowered IR / Prepared; render it to a backend at
// runtime via `symbi_ir::render_from_ir`, e.g., for the GPU<->CPU validation
// gate). generated by build.rs from the set of kernels emitted this run — see
// write_registry. CpuField / CpuFieldMut / Scalar must be in scope above this point
// (the CPU kernels use them).
include!(concat!(env!("OUT_DIR"), "/kernels_registry.rs"));
