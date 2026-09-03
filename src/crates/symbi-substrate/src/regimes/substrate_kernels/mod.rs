// =============================================================================
// regimes/substrate_kernels/mod.rs
//
// shared structured-binding-ABI plumbing for the D-generic hydro SubstrateKernelSets
// (iso / adiabatic / rhd). each method gathers its cell-centered field refs
// (inputs-then-outputs, in the generated kernel's binding order), names the kernel
// instance by (regime, ndim, dir), and routes one KernelInvocation through the
// CPU/GPU dispatch seam. the kernel instance is resolved by name
// via the generated `symbi_aot::kernel_by_name` registry — no hand-maintained
// `match (D, dir)` per regime.
//
// the RMHD set keeps its own copies of these layout helpers: it also drives
// staggered face/edge buffers (bface/efield on non-allocated domains), which these
// cell-centered hydro helpers do not model.
//
// the responsibilities are split into one submodule each (srp), re-exported flat so
// every item resolves at `crate::regimes::substrate_kernels::<name>` exactly as before:
//   types          — the Solver / RegimeKind classification enums
//   layout         — per-axis layouts + kernel-name suffixes + registry lookups
//   exec           — the cpu/gpu executor seam + parallelism policy
//   binding        — the buffer manifest parse / split / FieldRef resolution
//   params         — the typed scalar vocabulary + geom/motion/body resolvers
//   dispatch       — the per-physics dispatch chokepoints (cfl/flux/godunov/body)
//   runtime_source — runtime user sources (cpu interp / fused jit / gpu nvrtc)
//   boundary       — driven boundaries (the (Coord, Assign) DAG instance)
// =============================================================================

mod binding;
mod boundary;
mod dispatch;
mod exec;
mod layout;
mod params;
mod runtime_source;
mod types;

pub use exec::*;
pub use layout::*;
pub use types::*;
// the manifest-driven field binding: `bind_by_manifest` splits a kernel's baked manifest into
// (inputs, outputs) through a per-site slot resolver; `resolve_path` binds the typed
// `FieldRef` vocabulary and `resolve_snapshot_into` the snapshot family's explicit target.
// a resolver reads a bind by its typed family: `ct_role` for the reserved CT scratch,
// `free_payload` for the spelling of a free scratch slot.
pub(crate) use binding::{
    bind_by_manifest, ct_role, free_payload, resolve_path, resolve_snapshot_into,
};
// the typed (field, is_output) manifest: what a kernel actually reads and writes. public so a
// caller deciding whether to run a producer can ask the consumer instead of re-deriving which
// solver arm consumes what.
pub use binding::kernel_bindings;
pub mod census_compiled;
pub use boundary::*;
pub use dispatch::*;
pub use params::*;
pub use runtime_source::*;
// binding holds only crate-internal helpers (resolve_path / bind_manifest / kernel_bindings /
// parse_*), reached by the sibling submodules via `super::binding::*` — no external `substrate_kernels::`
// consumer, so it is not re-exported at the module root.
