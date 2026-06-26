// =============================================================================
// regimes/substrate_kernels/types.rs
//
// re-export shim: the `Solver` / `RegimeKind` classification enums moved DOWN to
// the sim<->substrate seam (docs/design/41 step 1, the cycle-break) so the sim
// core can name them without depending UP on `regimes`. kept so the substrate's
// `substrate_kernels::{Solver, RegimeKind}` paths resolve unchanged.
// =============================================================================

pub use symbi_sim::substrate_seam::{RegimeKind, Solver};
