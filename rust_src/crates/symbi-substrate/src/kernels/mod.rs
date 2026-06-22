// =============================================================================
// kernels/mod.rs
//
// shared kernel-set support utilities. the hand-written per-regime kernel
// sets were retired (docs/design/18) once the substrate KernelSets replaced
// and out-validated them; only the regime-agnostic support submodule
// (cfl_from_lambda, GhostFillDriver, FaceDomain, to_bc_array) remains, used
// by the substrate sets in regimes/.
// =============================================================================

pub mod support;
