// =============================================================================
// regimes/substrate_kernels/exec.rs
//
// re-export shim: the CPU/GPU executor seam + parallelism policy live in the
// `symbi-exec` crate; this module republishes them so the substrate's
// `super::exec::X` / `substrate_kernels::X` paths resolve.
// =============================================================================

pub use symbi_exec::policy::*;
