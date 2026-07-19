// =============================================================================
// regimes/substrate_kernels/exec.rs
//
// re-export shim: the CPU/GPU executor seam + parallelism policy moved to the
// `symbi-exec` crate. this module is kept so the substrate's
// `super::exec::X` / `substrate_kernels::X` paths resolve unchanged.
// =============================================================================

pub use symbi_exec::policy::*;
