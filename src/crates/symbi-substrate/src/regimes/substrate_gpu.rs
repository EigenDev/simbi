// =============================================================================
// regimes/substrate_gpu.rs
//
// re-export shim: the dispatch engine lives in the `symbi-exec` crate.
// this path is kept so callers `crate::regimes::substrate_gpu::X`
// resolve unchanged.
// =============================================================================

pub use symbi_exec::engine::*;
