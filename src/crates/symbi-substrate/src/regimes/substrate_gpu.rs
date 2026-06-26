// =============================================================================
// regimes/substrate_gpu.rs
//
// re-export shim: the dispatch engine moved to the `symbi-exec` crate
// (docs/design/40). this path is kept so callers `crate::regimes::substrate_gpu::X`
// resolve unchanged.
// =============================================================================

pub use symbi_exec::engine::*;
