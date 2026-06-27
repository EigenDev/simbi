// =============================================================================
// regimes/substrate_kernels/types.rs
//
// re-export shim: the `Solver` / `RegimeKind` classification enums live at
// the sim<->substrate seam (docs/design/41) so the sim core can name them
// without depending UP on `regimes`. this re-exports them so the substrate's
// `substrate_kernels::{Solver, RegimeKind}` paths resolve.
// =============================================================================

pub use symbi_sim::substrate_seam::{RegimeKind, Solver};
