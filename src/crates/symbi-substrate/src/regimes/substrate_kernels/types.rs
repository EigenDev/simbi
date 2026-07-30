// =============================================================================
// regimes/substrate_kernels/types.rs
//
// re-export shim: the `Solver` / `RegimeKind` classification enums live in
// `symbi_sim` so the sim core can name them without depending UP on `regimes`.
// this re-exports them so the substrate's
// `substrate_kernels::{Solver, RegimeKind}` paths resolve.
// =============================================================================

pub use symbi_sim::substrate_seam::{RegimeKind, Solver};
