// =============================================================================
// symbi-aot
//
// build-time AOT kernel library (docs/design/10 §4, docs/design/15 §3). build.rs
// runs the substrate to lower every kernel to IR and emit it as BOTH compilable
// Rust source (via emit_kernel_cpu) and a serialized backend-NEUTRAL lowered IR
// blob (the Prepared artifact, via prepare + prepared_to_ir), then writes ONE
// registry module that `include!`s every CPU kernel and exposes every kernel's IR
// blob as a `<KERNEL>_IR` const. the blob renders to ANY backend at runtime via
// `symbi_ir::render_from_ir`. this crate `include!`s that single registry —
// so adding a kernel touches only build.rs's kernel list, never this file
// (docs/design/15 §3: no hand-maintained registration).
//
// each generated `pub fn` is a normal compiled function exported from this crate;
// downstream consumers (the tests here, `symbi`'s SubstrateKernelSet) call them
// directly. the generated kernels are self-contained Rust over slices — this crate
// has no RUNTIME dependency on the substrate (only the build does).
// =============================================================================

// the CPU field descriptor the generated kernels take. carries the buffer
// pointer plus its pre-multiplied row-major strides, so per-cell index arithmetic is ONE method
// call (`at_Nd`) reading struct fields, not a dozen scattered scalar args. the
// strides are computed ONCE at construction (host-side, per kernel launch).
//
// SINGLE SOURCE OF TRUTH for index arithmetic: `at_1d` / `at_2d` / `at_3d` are
// the only methods that know the formula. every generated kernel calls them.
//
// `lo` and `strides` are fixed `[i32; 4]` (no slice indirection, no bounds
// checks) — supports rank ≤ 4 (more than the project's 3D ceiling). the unused
// tail is `0` and never read by the rank-specific accessors.
//
// `extent` is intentionally absent: no emitted kernel reads it (the strides
// already encode the only layout fact the index math needs). carrying it
// would waste 16 bytes per buffer × ~20 buffers = ~320 bytes of stack per
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
// slice slots (e.g. a zero-B buffer bound to all three magnetic components).
#[derive(Clone, Copy)]
pub struct CpuField<'a, T = f64> {
    pub data:    &'a [T],
    pub lo:      [i32; 4],
    pub strides: [i32; 4],
}

pub struct CpuFieldMut<'a, T = f64> {
    pub data:    &'a mut [T],
    pub lo:      [i32; 4],
    pub strides: [i32; 4],
}

/// strides from extent under the **physical-x-fastest convention**:
/// `strides[0] = 1`, `strides[d] = prod(extent[0..d])`. axis 0 is the
/// fastest-varying in memory — under the CFD-standard mapping (axis 0 = x),
/// adjacent CUDA `threadIdx.x` lanes hit adjacent bytes (coalesced reads).
///
/// delegates to `symbi_algebra::strides_from_extent` — THE single definition of
/// the stride formula, shared with `Domain`, `Layout`, and symbi-grid's `View`.
/// `pub` so the GPU view-construction path reuses the SAME helper.
#[inline]
pub fn compute_strides(extent: &[u32]) -> [i32; 4] {
    let n = extent.len().min(4);
    let ext_i32: [i32; 4] = std::array::from_fn(|d| if d < n { extent[d] as i32 } else { 0 });
    let mut s = [0i32; 4];
    symbi_algebra::strides_from_extent(&ext_i32[..n], &mut s[..n]);
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
/// zero-padding unused tail axes. the Gate-3 smem cooperative load clamps each
/// gmem read to `[lo, lo + extent - 1]`, so the device view must carry extent.
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
        Self { data, lo: copy_lo(lo), strides: compute_strides(extent) }
    }

    /// flat offset of `(i)` in a 1D buffer. inlined; no slice access in the body.
    #[inline(always)]
    pub fn flat_1d(&self, i: i32) -> usize {
        ((i - self.lo[0]) * self.strides[0]) as usize
    }

    /// flat offset of `(i, j)` in a 2D buffer.
    #[inline(always)]
    pub fn flat_2d(&self, i: i32, j: i32) -> usize {
        ((i - self.lo[0]) * self.strides[0]
            + (j - self.lo[1]) * self.strides[1]) as usize
    }

    /// flat offset of `(i, j, k)` in a 3D buffer. THE hot-path index — every
    /// PLM stencil read goes through here.
    #[inline(always)]
    pub fn flat_3d(&self, i: i32, j: i32, k: i32) -> usize {
        ((i - self.lo[0]) * self.strides[0]
            + (j - self.lo[1]) * self.strides[1]
            + (k - self.lo[2]) * self.strides[2]) as usize
    }

    /// rank-1 dereference. cheaper than going through `flat_1d` only because
    /// the index arithmetic inlines either way.
    #[inline(always)]
    pub fn at_1d(&self, i: i32) -> &T { &self.data[self.flat_1d(i)] }
    #[inline(always)]
    pub fn at_2d(&self, i: i32, j: i32) -> &T { &self.data[self.flat_2d(i, j)] }
    #[inline(always)]
    pub fn at_3d(&self, i: i32, j: i32, k: i32) -> &T { &self.data[self.flat_3d(i, j, k)] }
}

impl<'a, T> CpuFieldMut<'a, T> {
    #[inline]
    pub fn from_layout(data: &'a mut [T], lo: &[i32], extent: &[u32]) -> Self {
        Self { data, lo: copy_lo(lo), strides: compute_strides(extent) }
    }

    #[inline(always)]
    pub fn flat_1d(&self, i: i32) -> usize {
        ((i - self.lo[0]) * self.strides[0]) as usize
    }
    #[inline(always)]
    pub fn flat_2d(&self, i: i32, j: i32) -> usize {
        ((i - self.lo[0]) * self.strides[0]
            + (j - self.lo[1]) * self.strides[1]) as usize
    }
    #[inline(always)]
    pub fn flat_3d(&self, i: i32, j: i32, k: i32) -> usize {
        ((i - self.lo[0]) * self.strides[0]
            + (j - self.lo[1]) * self.strides[1]
            + (k - self.lo[2]) * self.strides[2]) as usize
    }

    #[inline(always)]
    pub fn at_1d(&self, i: i32) -> &T { &self.data[self.flat_1d(i)] }
    #[inline(always)]
    pub fn at_2d(&self, i: i32, j: i32) -> &T { &self.data[self.flat_2d(i, j)] }
    #[inline(always)]
    pub fn at_3d(&self, i: i32, j: i32, k: i32) -> &T { &self.data[self.flat_3d(i, j, k)] }

    #[inline(always)]
    pub fn at_1d_mut(&mut self, i: i32) -> &mut T {
        let idx = self.flat_1d(i);
        &mut self.data[idx]
    }
    #[inline(always)]
    pub fn at_2d_mut(&mut self, i: i32, j: i32) -> &mut T {
        let idx = self.flat_2d(i, j);
        &mut self.data[idx]
    }
    #[inline(always)]
    pub fn at_3d_mut(&mut self, i: i32, j: i32, k: i32) -> &mut T {
        let idx = self.flat_3d(i, j, k);
        &mut self.data[idx]
    }
}

// ---- the structured binding ABI (docs/design/15 §5) ----
//
// the backend-NEUTRAL kernel invocation. instead of the call site building the
// CPU-specific `&[CpuField]` / `&mut [CpuFieldMut]` host slices directly (a host-
// memory-ism), it builds ONE `KernelInvocation`: an ordered buffer list (each a
// data HANDLE + its layout) + the packed params. the same invocation maps to a CPU
// call (`run_cpu`, below) or — in 3c — a GPU launch, by interpreting the handle. the
// device-pointer handle variant arrives with the runtime GPU render (step 3c).

/// where a buffer's data lives. the variant encodes the kernel role: `Host` is a
/// read-only input, `HostMut` an output (incl. in-place — the kernel reads + writes
/// it). a `Device` variant joins in 3c for GPU launches.
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
                BufHandle::Host(d)    => inputs.push(CpuField::from_layout(d, b.lo, b.extent)),
                BufHandle::HostMut(d) => outputs.push(CpuFieldMut::from_layout(d, b.lo, b.extent)),
            }
        }
        kernel(&inputs, &mut outputs, self.grid, self.dom_lo, self.ints, self.scalars);
    }
}

// the kernel scalar type — the precision genericity of docs/design/15. the
// generated CPU kernels are `fn k<S: Scalar + OrderedNumeric>(..)`, so `Sim<f64>`
// and `Sim<f32>` pick the precision by the buffer type they pass (S inferred).
// the OrderedNumeric bound is the Tier-1.7 closure: the CPU emitter writes native
// `if cond { x } else { y }` for graph Branch nodes, so the body needs host
// ordering. f64/f32 both impl OrderedNumeric; the carrier-generic Gv path uses
// the GpuRenderer (CUDA C codegen), not this CPU rendering.
pub use symbi_ir::algebra::Scalar;
pub use symbi_algebra::OrderedNumeric;

/// the structured CPU-kernel ABI as a fn pointer: `(inputs, outputs, grid, dom_lo,
/// ints, scalars)`. every generated `pub fn k<S: Scalar + OrderedNumeric>(..)` has
/// this shape, so the generic fn item `k::<S>` coerces to `KernelFn<S>`. `kernel_by_name`
/// (generated in the registry below) returns one of these, so the D-generic
/// SubstrateKernelSet picks a kernel instance by name rather than a hand-maintained
/// per-regime match (docs/design/15 §5). CpuField / CpuFieldMut / Scalar / OrderedNumeric
/// are in scope above.
pub type KernelFn<S> =
    fn(&[CpuField<'_, S>], &mut [CpuFieldMut<'_, S>], &[u32], &[i32], &[i32], &[S]);

// the build-time-generated kernel registry: one `include!` per CPU kernel (each a
// `pub fn k<S: Scalar>(..)`) + one `pub const <KERNEL>_IR: &str` per kernel (the
// serialized backend-neutral lowered IR / Prepared; render it to a backend at
// runtime via `symbi_ir::render_from_ir`, e.g. for the GPU<->CPU validation
// gate). generated by build.rs from the set of kernels emitted this run — see
// write_registry. CpuField / CpuFieldMut / Scalar must be in scope above this point
// (the CPU kernels use them).
include!(concat!(env!("OUT_DIR"), "/kernels_registry.rs"));
